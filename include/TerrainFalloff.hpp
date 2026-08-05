#pragma once

#include "TerrainSubdivision.hpp"

#include "PCH.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace SmoothTerrain {

/**
 * @brief Keeps the smoothed terrain in a square of grids around the player and swaps meshes
 * without stutter as the player moves
 *
 * The build hook never subdivides at build time; every land quad starts with its vanilla
 * mesh and is registered here instead. This class then maintains a level map over the
 * landscape quads around the quad under the player (Chebyshev distance in quad units - a
 * quad is a quarter cell, 2048 world units): the full iSubdivisions level within the
 * iSmoothedQuads square (its radius counts the player's own quad, so 2 covers the 3x3 quad
 * square), and beyond it one level less every iGradientStep quads of distance until the
 * terrain is vanilla - or straight to vanilla when the gradient is off (iGradientStep = 0).
 * Quad units mean every boundary moves at half-cell resolution and keeps a consistent
 * distance from the player wherever they stand in a cell; two quads of one cell can differ,
 * with their shared interior line stitched exactly like a cell border. Everything is capped
 * at the game's actual loaded grid, read from the grid array itself at runtime;
 * iSmoothedQuads = 0 lifts the distance limit, which is simply the full-level square
 * covering the whole loaded grid - the builds still run through the same background
 * machinery, never inside a cell load.
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
 * Region updates are driven by four triggers that together cover every way the region can
 * change: the worker's position poll (quad crossings inside a cell have no engine event, see
 * pollPlayerQuad), the player's cell-change broadcast (BGSActorCellEvent - cell attach/detach
 * traffic alone is not enough, since re-entering buffered cells reattaches their existing
 * land without rebuilding it and a sparse wilderness cell may dispatch no reference events at
 * all), the cell attach/detach stream (grid changes while the player stands still), and
 * every quad registration (async land builds finishing after the events went quiet). Each
 * trigger only sets a flag; one queued main-thread pass (recompute) then walks the loaded
 * grid, decides the target state per quad, reverts instantly where smoothing must go away
 * and queues worker builds where it is missing - one pass handles entering and leaving quads
 * together. A generation counter per quad discards results that a faster moving player
 * already made stale.
 *
 * Tracked quads are resolved through the scene graph (the children of the per-quadrant
 * multibound nodes in LoadedLandData::mesh), not through LoadedLandData::geom: the engine's
 * geometry init rebuilds every geom slot once per quadrant and attaches only the generation
 * current at that quadrant's turn, so geom ends up describing the rendered mesh for the last
 * quadrant alone. Registration happens per builder call, so the short-lived intermediate
 * generations also pass through the registry; they are never found in the scene graph and
 * fall out through the prune pass once the engine drops them.
 *
 * Stitching: every shared border line renders at the minimum of the two adjacent quads'
 * levels - the finer side pins its extra border vertices onto the coarser side's edge
 * polyline, down to the straight vanilla segments against unsmoothed terrain (see
 * TerrainSubdivision::buildSmoothedMesh). Because a quad's level is pure coordinate math,
 * both quads of any shared line always agree on that minimum, so no cracks can open
 * anywhere: equal-level borders share the same spline, unequal ones share the coarser
 * polyline, and every corner vert is an original LAND vert that is bit-exact in all
 * variants. The gradient never puts adjacent quads more than one level apart (distance
 * changes by at most one quad between neighbors), except against unloaded cells, where the
 * pin drops straight to the vanilla line the distant LOD was authored against.
 *
 * While a region shift is still in flight the two sides of an edge can briefly disagree
 * (quads apply as their builds finish, not as one atomic batch). The mismatch is bounded by
 * the spline's deviation from the straight edge, sits a full cell or more from the player,
 * and settles within a moment - a hairline that buys keeping every swap a plain field
 * write, which is what makes crossings stutter-free.
 *
 * Cell loads are treated as sacred: while the engine streams cells in, every core and the
 * driver are already saturated, and VR has no frame headroom to absorb extra work. So
 * registration copies nothing (snapshots are captured lazily when a build is first wanted),
 * the worker runs at below-normal priority, holds its batch until land builds have been
 * quiet for a beat and then paces one build every few milliseconds, and shapes retired by
 * the prune pass drop their last reference on the worker so their GPU buffers are never
 * released in a main-thread burst.
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
     * @brief Whether the manager is running (supported runtime, iSubdivisions > 0) and the
     * build hook should register quads
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
     * Called by the build hook on whichever thread the engine builds land on - during cell
     * attach, the one moment the frame has no headroom - so it does as close to nothing as
     * possible: validate the layout (see TerrainSubdivision::validateQuadLayout; anything
     * that is not the vanilla layout is left alone forever), record the vanilla buffer set
     * for later instant reverts, take a strong reference, and queue a region pass. No copies:
     * the build snapshot is captured lazily by the region pass when a build is first wanted.
     *
     * @param shape The vanilla-built quad shape
     * @param quad Quadrant index 0-3 (0 = SW, 1 = SE, 2 = NW, 3 = NE)
     */
    static void registerQuad(RE::BSTriShape* shape,
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
    constexpr static std::chrono::milliseconds K_BUILD_QUIET_PERIOD {250}; /**< Builds start only after land
                                                                  builds have been quiet this long: registrations
                                                                  only happen while the engine loads cells, which
                                                                  is exactly when no core has cycles to spare
                                                                  (see workerLoop) */
    constexpr static std::chrono::milliseconds K_BUILD_SPACING {8}; /**< Pause between consecutive builds so a
                                                                  region shift's batch is a trickle, not a burst
                                                                  of CPU work and driver uploads (see workerLoop) */
    constexpr static std::chrono::milliseconds K_PLAYER_POLL_INTERVAL {200}; /**< How often the worker checks
                                                                  whether the player crossed a quad line; the
                                                                  engine only announces cell changes, and quads
                                                                  are half that size (see pollPlayerQuad) */

    /**
     * @brief Everything tracked about one registered land quad
     */
    struct Entry {
        RE::NiPointer<RE::BSTriShape> shape; /**< Strong ref: keeps the shape (and its address, which is our
                                                  registry key) alive while tracked */
        std::shared_ptr<const TerrainSubdivision::QuadSnapshot> snapshot; /**< Immutable build input, shared
                                                                              with worker jobs. Captured lazily
                                                                              by the region pass when a build is
                                                                              first wanted - registration must
                                                                              stay copy-free (see registerQuad) */
        std::uint64_t vertexDesc {}; /**< The validated vertex descriptor, kept for the lazy capture */
        std::uint32_t quad {}; /**< Quadrant index 0-3 */
        TerrainSubdivision::MeshBuffers vanilla; /**< The engine's own buffers, kept alive for instant reverts */
        std::optional<TerrainSubdivision::MeshBuffers> smoothed; /**< The smoothed set currently installed */
        std::uint8_t appliedLevel {}; /**< Subdivision level of the installed set (0 while vanilla renders) */
        TerrainSubdivision::EdgeLevels appliedEdgeLevels {}; /**< Edge levels the installed set was built with */
        std::uint8_t wantLevel {}; /**< Target level as of the last region pass (may still be in flight) */
        TerrainSubdivision::EdgeLevels wantEdgeLevels {}; /**< Target edge levels */
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
        TerrainSubdivision::EdgeLevels edgeLevels {};
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
     * @brief Hands a region pass's builds to the worker thread, starting it on first use
     *
     * The caller orders the batch nearest-quad-first; the worker preserves that order, so
     * after a load the terrain around the player smooths before the distance does.
     */
    static void enqueueJobs(std::vector<Job>&& jobs);

    /**
     * @brief Starts the worker thread if it is not running yet; requires s_jobMutex held
     */
    static void ensureWorkerLocked();

    /**
     * @brief Schedules a region pass when the player has crossed into a different quad
     *
     * Runs on the worker thread every K_PLAYER_POLL_INTERVAL: the region's granularity is a
     * landscape quad, but the engine only broadcasts cell changes, so quad crossings inside a
     * cell would otherwise go unnoticed. Reading the player's position off-main is a benign
     * race - a torn value at worst schedules one extra pass, which then recomputes everything
     * from clean main-thread state.
     */
    static void pollPlayerQuad();

    /**
     * @brief Hands retired shapes to the worker thread so their last reference drops there
     *
     * Each destruction releases GPU buffers through the engine, and one region pass can
     * retire dozens of dead builder generations at once - done on the main thread that is a
     * visible spike on VR. Requires s_entryMutex held (the established s_entryMutex ->
     * s_jobMutex order).
     */
    static void buryShapes(std::vector<RE::NiPointer<RE::BSTriShape>>&& shapes);

    /**
     * @brief The worker: builds smoothed meshes and posts them back as main-thread tasks
     *
     * Runs detached for the lifetime of the process at below-normal priority - its work is
     * never latency-critical (the smoothing boundary sits a full cell away), while the game's
     * own threads are. Before starting a build it waits out K_BUILD_QUIET_PERIOD since the
     * last land-build registration, so batches never compete with a cell load in flight, and
     * it sleeps K_BUILD_SPACING between builds so a batch trickles instead of bursting. Skips
     * jobs whose generation is already stale before doing any work, so a sprinting player
     * does not pile up dead builds. Also drains the graveyard (see buryShapes).
     */
    static void workerLoop();

    static inline bool s_enabled = false; /**< Set once by install(), read by the build hook */

    static inline std::mutex s_entryMutex; /**< Guards s_entries (registrations come from the land build thread) */
    static inline std::unordered_map<RE::BSTriShape*, Entry> s_entries;
    static inline std::atomic<bool> s_recomputeQueued {false};
    static inline std::atomic<std::uint64_t> s_nextGeneration {1};
    static inline CellSink s_cellSink;
    static inline PlayerCellSink s_playerCellSink;

    static inline std::mutex s_jobMutex; /**< Guards s_jobs, s_graveyard and s_workerStarted */
    static inline std::condition_variable s_jobSignal;
    static inline std::deque<Job> s_jobs;
    static inline std::vector<RE::NiPointer<RE::BSTriShape>> s_graveyard; /**< Shapes awaiting off-main release */
    static inline bool s_workerStarted = false;
    static inline std::atomic<std::chrono::steady_clock::duration::rep> s_lastLandBuild {}; /**< Timestamp of the
                                                                  newest registration; the worker's quiet-period
                                                                  clock (a land build means a cell load is in
                                                                  flight) */
    static inline std::atomic<std::int64_t> s_lastPlayerQuad {
        std::numeric_limits<std::int64_t>::min()}; /**< The player's quad as of the last region pass or poll,
                                                      packed (x << 32 | y); the poll's change detector */
};

} // namespace SmoothTerrain
