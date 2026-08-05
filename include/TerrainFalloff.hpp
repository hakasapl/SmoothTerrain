#pragma once

#include "TerrainSubdivision.hpp"

#include "PCH.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace SmoothTerrain {

/**
 * @brief Keeps the smoothed terrain in a square of grids around the player and swaps meshes
 * without stutter as the player moves
 *
 * With iSmoothedGrids > 0 the build hook no longer subdivides at build time; every land quad
 * starts with its vanilla mesh and is registered here instead. This class then maintains the
 * invariant that exactly the cells within the configured square around the player's cell (a
 * Chebyshev radius counting the player's own grid, so 2 covers the 3x3 neighbor square)
 * render the smoothed mesh, and everything further out renders vanilla. The square is capped
 * at the game's actual loaded grid, read from the grid array itself at runtime, so any value
 * whose square would reach past the loaded grid simply smooths everything that is loaded.
 *
 * The stutter-free part rests on three legs:
 *  - Reverting to vanilla is free. The engine's own vanilla buffers are never released while
 *    a quad is tracked, so bringing them back is three field writes (see
 *    TerrainSubdivision::setMesh).
 *  - Smoothing is built off the main thread. Vertex interpolation and GPU buffer creation run
 *    on a worker thread against an immutable snapshot captured at build time (the engine
 *    itself builds land buffers off-thread through the same creators, which is what proves
 *    them thread-safe). The main thread only ever executes the final pointer swap.
 *  - Swaps happen on the main thread via SKSE tasks, the same synchronization point every
 *    scene-graph-mutating SKSE plugin uses, so a draw in flight can never observe a half
 *    swapped shape.
 *
 * Region updates are driven by three triggers that together cover every way the region can
 * change: the player's own cell-change broadcast (BGSActorCellEvent, the authoritative
 * "player crossed a cell border" signal - cell attach/detach traffic alone is not enough,
 * since re-entering buffered cells reattaches their existing land without rebuilding it and
 * a sparse wilderness cell may dispatch no reference events at all), the cell attach/detach
 * stream (grid changes while the player stands still), and every quad registration (async
 * land builds finishing after the events went quiet). Each trigger only sets a flag; one
 * queued main-thread pass (recompute) then walks the loaded grid, decides the target state
 * per cell, reverts instantly where smoothing must go away and queues worker builds where it
 * is missing - one pass handles entering and leaving cells together. A generation counter
 * per quad discards results that a faster moving player already made stale.
 *
 * Tracked quads are resolved through the scene graph (the children of the per-quadrant
 * multibound nodes in LoadedLandData::mesh), not through LoadedLandData::geom: the engine's
 * geometry init rebuilds every geom slot once per quadrant and attaches only the generation
 * current at that quadrant's turn, so geom ends up describing the rendered mesh for the last
 * quadrant alone. Registration happens per builder call, so the short-lived intermediate
 * generations also pass through the registry; they are never found in the scene graph and
 * fall out through the prune pass once the engine drops them.
 *
 * Stitching: a smoothed cell whose neighbor stays coarse gets that edge flagged, and the mesh
 * builder pins the vertices on that border line to the straight vanilla edge (see
 * TerrainSubdivision::buildSmoothedMesh). Because the region is a square in cell coordinates,
 * whether an edge is flagged is pure coordinate math and both cells of any shared edge always
 * agree on its treatment, so no cracks can open anywhere: smoothed-smoothed borders share the
 * same spline, smoothed-coarse borders share the straight vanilla line, and every corner vert
 * is an original LAND vert that is bit-exact in all variants.
 *
 * While a region shift is still in flight the two sides of an edge can briefly disagree
 * (quads apply as their builds finish, not as one atomic batch). The mismatch is bounded by
 * the spline's deviation from the straight edge, sits a full cell or more from the player,
 * and settles within the few frames the worker needs to drain the batch - a hairline that
 * buys keeping every swap a plain field write, which is what makes crossings stutter-free.
 */
class TerrainFalloff {
public:
    /**
     * @brief Decides whether the falloff is active and logs the effective mode
     *
     * No hooks of its own: the build hook (TerrainSubdivision) feeds quads in through
     * registerQuad, and the event sink is registered later by onDataLoaded.
     */
    static void install();

    /**
     * @brief Whether quads should be registered for distance-based smoothing instead of being
     * subdivided at build time
     */
    [[nodiscard]] static auto isEnabled() -> bool;

    /**
     * @brief Registers the cell attach/detach sink; call once on SKSE's kDataLoaded
     *
     * Event sources do not exist yet at plugin load, hence the deferred registration.
     */
    static void onDataLoaded();

    /**
     * @brief Starts tracking a freshly built vanilla land quad
     *
     * Called by the build hook on whichever thread the engine builds land on. Validates and
     * snapshots the quad (see TerrainSubdivision::snapshotQuad; anything that is not the
     * vanilla layout is left alone forever, same as the eager path's fallback), stores the
     * vanilla buffer set for later instant reverts, takes a strong reference to the shape and
     * queues a region pass so a quad born inside the smoothed square gets its mesh promptly.
     *
     * @param shape The vanilla-built quad shape
     * @param data Loaded land data of the cell (the height tables)
     * @param quad Quadrant index 0-3 (0 = SW, 1 = SE, 2 = NW, 3 = NE)
     */
    static void registerQuad(RE::BSTriShape* shape,
                             const RE::TESObjectLAND::LoadedLandData& data,
                             std::uint32_t quad);

    TerrainFalloff() = delete;

private:
    constexpr static int K_DEFAULT_LOADED_RADIUS = 2; /**< Vanilla uGridsToLoad=5 fallback when the grid array
                                                         is not available to read the real value from */
    constexpr static std::chrono::seconds K_PRUNE_MIN_AGE {5}; /**< A registered quad is never pruned younger
                                                                  than this: between the build hook and the engine
                                                                  filing the shape into LoadedLandData::geom the
                                                                  only reference may briefly be ours (the builder
                                                                  holds a raw pointer on its stack), and pruning
                                                                  then would destroy a shape about to be attached */

    /**
     * @brief Everything tracked about one registered land quad
     */
    struct Entry {
        RE::NiPointer<RE::BSTriShape> shape; /**< Strong ref: keeps the shape (and its address, which is our
                                                  registry key) alive while tracked */
        std::shared_ptr<const TerrainSubdivision::QuadSnapshot> snapshot; /**< Immutable build input, shared
                                                                              with worker jobs */
        std::uint32_t quad {}; /**< Quadrant index 0-3 */
        TerrainSubdivision::MeshBuffers vanilla; /**< The engine's own buffers, kept alive for instant reverts */
        std::optional<TerrainSubdivision::MeshBuffers> smoothed; /**< The smoothed set currently installed */
        std::uint8_t appliedEdges {}; /**< linearEdges mask the installed smoothed set was built with */
        bool wantSmoothed {}; /**< Target state as of the last region pass (may still be in flight) */
        std::uint8_t wantEdges {}; /**< Target linearEdges mask */
        std::uint64_t generation {}; /**< Globally unique id a worker result must match to be applied */
        std::chrono::steady_clock::time_point registeredAt; /**< For the prune age guard */
        bool seen {}; /**< Scratch flag of the current region pass: found in the loaded grid */
    };

    /**
     * @brief One smoothed-mesh build handed to the worker thread
     *
     * Self-contained: the snapshot is immutable and keepAlive pins the shape so its address
     * cannot be recycled (and matched against the wrong registry entry) while the job or its
     * result is in flight.
     */
    struct Job {
        RE::BSTriShape* key {}; /**< Registry key; only ever compared, never dereferenced off-main */
        RE::NiPointer<RE::BSTriShape> keepAlive;
        std::shared_ptr<const TerrainSubdivision::QuadSnapshot> snapshot;
        int level {};
        std::uint8_t linearEdges {};
        std::uint64_t generation {};
    };

    /**
     * @brief Sink that turns cell attach/detach traffic into queued region passes
     *
     * Fires once per reference, so it is deliberately nothing but a flag set; the real work
     * runs once on the main thread no matter how many events a grid shift produces.
     */
    struct CellSink final : RE::BSTEventSink<RE::TESCellAttachDetachEvent> {
        auto ProcessEvent(const RE::TESCellAttachDetachEvent* event,
                          RE::BSTEventSource<RE::TESCellAttachDetachEvent>* source)
            -> RE::BSEventNotifyControl override;
    };

    /**
     * @brief Sink on the player's own cell-change broadcast, the primary region trigger
     *
     * The player crossing a cell border is exactly the moment the smoothed square must move,
     * and this event fires on every crossing regardless of whether the entering cells were
     * buffered (no land rebuild, so no registration trigger) or how much reference traffic
     * the crossing produced (which is all the attach/detach sink ever sees).
     */
    struct PlayerCellSink final : RE::BSTEventSink<RE::BGSActorCellEvent> {
        auto ProcessEvent(const RE::BGSActorCellEvent* event,
                          RE::BSTEventSource<RE::BGSActorCellEvent>* source)
            -> RE::BSEventNotifyControl override;
    };

    /**
     * @brief Queues one main-thread region pass unless one is already queued; any thread
     */
    static void scheduleRecompute();

    /**
     * @brief The region pass: walks the loaded grid and reconciles every tracked quad
     *
     * Main thread only. Decides the smoothed square from the player's current cell, reverts
     * quads that left it (instant), and queues worker builds for quads that entered it or
     * whose edge mask changed. Anything the walk fails to resolve reverts too - a smoothed
     * mesh only ever stays installed while a pass can positively place its cell inside the
     * square - and is pruned once its shape has no other owners (sole-ownership test with an
     * age guard, see K_PRUNE_MIN_AGE). Without an exterior cell under the player everything
     * reverts to vanilla.
     */
    static void recompute();

    /**
     * @brief Puts the vanilla buffers back and drops the smoothed set, if any
     *
     * Main thread only (writes live shape fields); requires s_entryMutex held.
     */
    static void revertEntry(Entry& entry);

    /**
     * @brief Installs a worker-built mesh, unless the world moved on while it was building
     *
     * Main thread only (runs as an SKSE task). A generation mismatch or a vanished entry
     * releases the buffers instead; a failed build (nullopt) resets the entry's target so a
     * later region change retries instead of wedging the quad.
     */
    static void applyBuiltMesh(const Job& job,
                               const std::optional<TerrainSubdivision::MeshBuffers>& mesh);

    /**
     * @brief Hands a job to the worker thread, starting it on first use
     */
    static void enqueueJob(Job&& job);

    /**
     * @brief The worker: builds smoothed meshes and posts them back as main-thread tasks
     *
     * Runs detached for the lifetime of the process. Skips jobs whose generation is already
     * stale before doing any work, so a sprinting player does not pile up dead builds.
     */
    static void workerLoop();

    static inline bool s_enabled = false; /**< Set once by install(), read by the build hook */

    static inline std::mutex s_entryMutex; /**< Guards s_entries (registrations come from the land build thread) */
    static inline std::unordered_map<RE::BSTriShape*, Entry> s_entries;
    static inline std::atomic<bool> s_recomputeQueued {false};
    static inline std::atomic<std::uint64_t> s_nextGeneration {1};
    static inline CellSink s_cellSink;
    static inline PlayerCellSink s_playerCellSink;

    static inline std::mutex s_jobMutex; /**< Guards s_jobs and s_workerStarted */
    static inline std::condition_variable s_jobSignal;
    static inline std::deque<Job> s_jobs;
    static inline bool s_workerStarted = false;
};

} // namespace SmoothTerrain
