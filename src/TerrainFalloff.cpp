#include "TerrainFalloff.hpp"

#include "ConfigLoader.hpp"
#include "Offsets.hpp"
#include "TerrainSubdivision.hpp"

#include "PCH.h"

#include <Windows.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace SmoothTerrain;

void TerrainFalloff::install()
{
    // The falloff piggybacks on the subdivision hook, so it is only meaningful when that hook
    // is installed (supported runtime, iSubdivisions > 0)
    if (!Offsets::isRuntimeSupported() || ConfigLoader::getSubdivisions() <= 0) {
        return;
    }

    // Every mode runs through this manager, including iSmoothedQuads = 0: subdividing during
    // the engine's own cell loading (the old eager path) is a guaranteed load-time hitch, so
    // "no distance limit" simply means the region covers the whole loaded grid while the
    // builds still happen in the background
    s_enabled = true;
    if (ConfigLoader::getSmoothedQuads() <= 0) {
        spdlog::info("No smoothing distance limit (iSmoothedQuads = 0): every loaded landscape quad renders the "
                     "smoothed mesh, built in the background shortly after its cell loads");
    } else {
        const int side = (2 * ConfigLoader::getSmoothedQuads()) - 1;
        if (ConfigLoader::getGradientStep() > 0) {
            spdlog::info("Distance falloff active: a {0}x{0} landscape-quad square around the player renders "
                         "subdivision level {1} (a quad is half a grid, 2048 units), and the level drops by one "
                         "every {2} quad(s) beyond it until terrain is vanilla",
                         side,
                         ConfigLoader::getSubdivisions(),
                         ConfigLoader::getGradientStep());
        } else {
            spdlog::info("Distance falloff active: a {0}x{0} landscape-quad square around the player renders the "
                         "smoothed mesh (a quad is half a grid, 2048 units); everything further renders the "
                         "vanilla mesh (iGradientStep = 0, no gradient)",
                         side);
        }
    }
}

auto TerrainFalloff::isEnabled() -> bool { return s_enabled; }

void TerrainFalloff::onDataLoaded()
{
    if (!s_enabled) {
        return;
    }

    // The player's own cell-change broadcast is the primary trigger: it fires on every cell
    // border crossing, including back into buffered cells, where the land reattaches without
    // rebuilding (so no registration trigger) and reference attach/detach traffic is not
    // guaranteed. Without it the region only moves when land happens to build somewhere.
    auto* player = RE::PlayerCharacter::GetSingleton();
    auto* playerCellSource = player != nullptr ? player->AsBGSActorCellEventSource() : nullptr;
    if (playerCellSource != nullptr) {
        playerCellSource->AddEventSink(&s_playerCellSink);
        spdlog::info("Terrain falloff player cell sink registered");
    } else {
        spdlog::warn("Player cell event source unavailable; the smoothed square only follows the player "
                     "when land builds or cell attach events arrive");
    }

    // Attach/detach traffic covers grid changes that happen while the player stays put
    auto* holder = RE::ScriptEventSourceHolder::GetSingleton();
    if (holder == nullptr) {
        spdlog::warn("Script event source holder unavailable; terrain falloff runs without the cell "
                     "attach/detach trigger");
        return;
    }
    holder->AddEventSink<RE::TESCellAttachDetachEvent>(&s_cellSink);
    spdlog::info("Terrain falloff cell sink registered");

    // The worker doubles as the quad-crossing detector (see pollPlayerQuad), so it must be
    // alive before the first build is ever wanted
    {
        const std::lock_guard<std::mutex> lock(s_jobMutex);
        ensureWorkerLocked();
    }
}

auto TerrainFalloff::CellSink::ProcessEvent(const RE::TESCellAttachDetachEvent* /*event*/,
                                            RE::BSTEventSource<RE::TESCellAttachDetachEvent>* /*source*/)
    -> RE::BSEventNotifyControl
{
    scheduleRecompute();
    return RE::BSEventNotifyControl::kContinue;
}

auto TerrainFalloff::PlayerCellSink::ProcessEvent(const RE::BGSActorCellEvent* event,
                                                  RE::BSTEventSource<RE::BGSActorCellEvent>* /*source*/)
    -> RE::BSEventNotifyControl
{
    // Only the enter side matters; the matching leave arrives in the same crossing and would
    // just queue the same pass twice
    if (event != nullptr && event->flags == RE::BGSActorCellEvent::CellFlag::kEnter) {
        scheduleRecompute();
    }
    return RE::BSEventNotifyControl::kContinue;
}

void TerrainFalloff::registerQuad(RE::BSTriShape* shape,
                                  std::uint32_t quad)
{
    // A land build means a cell load is in flight; the worker holds its batch until this
    // clock has been quiet for a beat (see workerLoop)
    s_lastLandBuild.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_relaxed);

    const auto vertexDesc = TerrainSubdivision::validateQuadLayout(*shape);
    if (!vertexDesc.has_value()) {
        // Foreign layout or no CPU vertex copy: another mod owns this quad and it stays
        // vanilla for good
        spdlog::debug("Quad {} kept its vanilla land mesh", quad);
        return;
    }

    // Reading the freshly built shape off-main is safe here: the builder has not attached it
    // to anything yet, so no other thread can see it. Deliberately no copies (the engine
    // calls the builder 16 times per cell attach); the snapshot is captured lazily by the
    // region pass when a build is first wanted.
    Entry entry;
    entry.shape = RE::NiPointer<RE::BSTriShape> {shape};
    entry.vertexDesc = *vertexDesc;
    entry.quad = quad;
    entry.vanilla = TerrainSubdivision::currentMesh(*shape);
    entry.registeredAt = std::chrono::steady_clock::now();

    {
        const std::lock_guard<std::mutex> lock(s_entryMutex);
        // A colliding key cannot happen while we hold a reference (the allocator cannot reuse
        // the address), but replacing defensively costs nothing
        const auto found = s_entries.find(shape);
        if (found != s_entries.end()) {
            revertEntry(found->second);
            s_entries.erase(found);
        }
        s_entries.emplace(shape, std::move(entry));
    }

    // The quad may be born inside the smoothed square (game load, fast travel, grid shift);
    // a region pass picks it up once the engine files the shape into the cell grid
    scheduleRecompute();
}

void TerrainFalloff::scheduleRecompute()
{
    if (s_recomputeQueued.exchange(true, std::memory_order_acq_rel)) {
        return; // a queued pass will already see the state this trigger is announcing
    }
    const auto* taskInterface = SKSE::GetTaskInterface();
    if (taskInterface == nullptr) {
        s_recomputeQueued.store(false, std::memory_order_release);
        return;
    }
    taskInterface->AddTask([]() { recompute(); });
}

void TerrainFalloff::recompute()
{
    // Cleared before the walk: triggers landing while this pass runs queue a fresh pass that
    // will see whatever changed under us
    s_recomputeQueued.store(false, std::memory_order_release);

    const std::lock_guard<std::mutex> lock(s_entryMutex);
    if (s_entries.empty()) {
        return;
    }
    for (auto& [key, entry] : s_entries) {
        entry.seen = false;
    }

    // Region center: the exterior cell under the player. Without one (interiors, main menu)
    // nothing is near the player and every quad reverts to vanilla below.
    const RE::EXTERIOR_DATA* center = nullptr;
    auto* player = RE::PlayerCharacter::GetSingleton();
    auto* playerCell = player != nullptr ? player->GetParentCell() : nullptr;
    if (playerCell != nullptr && playerCell->IsExteriorCell()) {
        center = playerCell->GetCoordinates();
    }

    auto* tes = RE::TES::GetSingleton();
    auto* grid = tes != nullptr ? tes->gridCells : nullptr;

    if (center != nullptr && grid != nullptr && grid->cells != nullptr) {
        // The loaded grid bounds everything: a quad whose cell is not loaded has no mesh to
        // smooth. The grid array's length is the live uGridsToLoad, whatever set it.
        const int loadedRadius = grid->length > 0 ? (static_cast<int>(grid->length) - 1) / 2 : K_DEFAULT_LOADED_RADIUS;

        // The region lives in landscape-quad coordinates (a cell is 2x2 quads), centered on
        // the quad under the player's feet, so the boundary sits at half-cell resolution and
        // its distance from the player does not depend on where in a cell they stand. 0 means
        // no distance limit: every quad of a loaded cell qualifies. Membership stays pure
        // coordinate math, so both quads of any shared border line always agree on whether
        // that line is pinned - within one cell and across cells alike.
        const auto playerPosition = player->GetPosition();
        const auto playerQuadX = static_cast<int>(std::floor(playerPosition.x / TerrainSubdivision::K_QUAD_WORLD_SIZE));
        const auto playerQuadY = static_cast<int>(std::floor(playerPosition.y / TerrainSubdivision::K_QUAD_WORLD_SIZE));
        s_lastPlayerQuad.store((static_cast<std::int64_t>(playerQuadX) << 32U)
                                   | static_cast<std::uint32_t>(playerQuadY),
                               std::memory_order_relaxed);

        const int configuredQuads = ConfigLoader::getSmoothedQuads();
        const int quadRadius = configuredQuads <= 0 ? std::numeric_limits<int>::max() : configuredQuads - 1;
        const int gradientStep = ConfigLoader::getGradientStep();
        const int maxLevel = ConfigLoader::getSubdivisions();
        const auto levelAt = [&](int quadX, int quadY) -> int {
            // A quad whose cell is outside the loaded grid has no mesh and counts as vanilla
            // (a cell is 2x2 quads; >> 1 is floor division, defined for negatives since C++20)
            const int cellDeltaX = (quadX >> 1) - center->cellX;
            const int cellDeltaY = (quadY >> 1) - center->cellY;
            if (std::max(std::abs(cellDeltaX), std::abs(cellDeltaY)) > loadedRadius) {
                return 0;
            }
            // Full level inside the configured square around the player's quad, then one
            // level less every gradientStep quads of distance - or straight to vanilla when
            // the gradient is off
            const int distance = std::max(std::abs(quadX - playerQuadX), std::abs(quadY - playerQuadY));
            if (distance <= quadRadius) {
                return maxLevel;
            }
            if (gradientStep <= 0) {
                return 0;
            }
            const int stepsLost = ((distance - quadRadius) + gradientStep - 1) / gradientStep;
            return std::max(0, maxLevel - stepsLost);
        };
        std::vector<std::pair<int, Job>> pendingBuilds;

        // Sweep 1: resolve every rendered quad through the scene graph and record which
        // lattice coordinates actually participate in the smoothing. Resolution goes through
        // the scene graph, not LoadedLandData::geom: the engine's land geometry init rebuilds
        // all four geom slots once per quadrant (16 builder calls per cell) but attaches only
        // the generation current at each quadrant's turn - NiNode::SetAt(0, geom[quad]) on
        // that quadrant's multibound node. geom therefore ends up holding the rendered shape
        // for the last quadrant only; the attached child is the authority on what is visible
        // (verified on 1.6.1170 at 0x2A8D51).
        struct ResolvedQuad {
            Entry* entry;
            RE::TESObjectLAND::LoadedLandData* landData;
            int quadX;
            int quadY;
        };
        std::vector<ResolvedQuad> resolvedQuads;
        std::unordered_set<std::int64_t> participating;
        const auto packQuad = [](int quadX, int quadY) -> std::int64_t {
            return (static_cast<std::int64_t>(quadX) << 32U) | static_cast<std::uint32_t>(quadY);
        };
        for (std::uint32_t gridX = 0; gridX < grid->length; ++gridX) {
            for (std::uint32_t gridY = 0; gridY < grid->length; ++gridY) {
                auto* cell = grid->GetCell(gridX, gridY);
                if (cell == nullptr) {
                    continue;
                }
                auto* land = cell->GetRuntimeData().cellLand;
                if (land == nullptr || land->loadedData == nullptr) {
                    continue;
                }
                auto* coords = cell->GetCoordinates();
                if (coords == nullptr) {
                    continue;
                }
                for (std::uint32_t quad = 0; quad < 4; ++quad) {
                    auto* node = land->loadedData->mesh[quad];
                    if (node == nullptr) {
                        continue;
                    }
                    for (const auto& childPointer : node->GetChildren()) {
                        auto* child = childPointer.get();
                        if (child == nullptr) {
                            continue;
                        }
                        // Pointer-keyed lookup: only shapes we registered can match, so the
                        // cast is never acted on for foreign children
                        const auto found = s_entries.find(static_cast<RE::BSTriShape*>(child));
                        if (found == s_entries.end()) {
                            continue; // not ours (validation refused it: another mod's mesh)
                        }
                        Entry& entry = found->second;
                        entry.seen = true;
                        const int quadX = (coords->cellX * 2) + static_cast<int>(entry.quad & 1U);
                        const int quadY = (coords->cellY * 2) + static_cast<int>(entry.quad >> 1U);
                        resolvedQuads.push_back(ResolvedQuad {
                            .entry = &entry, .landData = land->loadedData, .quadX = quadX, .quadY = quadY});
                        participating.insert(packQuad(quadX, quadY));
                    }
                }
            }
        }

        // A neighbor that does not participate - foreign layout another mod rebuilt, missing
        // land, or simply not (yet) attached - renders a mesh this plugin does not control,
        // so the shared line pins to the straight vanilla segments, the one edge every
        // LAND-derived mesh agrees on. A neighbor that registers later upgrades the edge on
        // the pass its registration schedules.
        const auto participatingLevelAt = [&](int quadX, int quadY) -> int {
            return participating.contains(packQuad(quadX, quadY)) ? levelAt(quadX, quadY) : 0;
        };

        // Sweep 2: reconcile every resolved quad with its target level and edge levels - each
        // border line at the minimum with the quad on the other side (the other half of this
        // cell or a neighboring cell's quad alike)
        for (const auto& resolved : resolvedQuads) {
            Entry& entry = *resolved.entry;
            const int quadLevel = levelAt(resolved.quadX, resolved.quadY);
            TerrainSubdivision::EdgeLevels edgeLevels {};
            if (quadLevel > 0) {
                const auto sharedLevel = [&](int neighborX, int neighborY) -> std::uint8_t {
                    return static_cast<std::uint8_t>(std::min(quadLevel, participatingLevelAt(neighborX, neighborY)));
                };
                edgeLevels = TerrainSubdivision::EdgeLevels {.west = sharedLevel(resolved.quadX - 1, resolved.quadY),
                                                             .east = sharedLevel(resolved.quadX + 1, resolved.quadY),
                                                             .south = sharedLevel(resolved.quadX, resolved.quadY - 1),
                                                             .north = sharedLevel(resolved.quadX, resolved.quadY + 1)};
            }
            if (entry.wantLevel == quadLevel && (quadLevel == 0 || edgeLevels == entry.wantEdgeLevels)) {
                continue; // already there, or already on its way there
            }

            if (quadLevel > 0 && entry.snapshot == nullptr) {
                // The capture deferred from registration, done here because the cell is
                // loaded right now: its height tables and the engine's CPU vertex copy
                // (alive as long as the entry is) are both at hand
                auto* const vanillaData = entry.vanilla.rendererData;
                if (vanillaData == nullptr || vanillaData->rawVertexData == nullptr) {
                    continue; // validated at registration; defensive
                }
                entry.snapshot = TerrainSubdivision::snapshotQuad(
                    *resolved.landData, vanillaData->rawVertexData, entry.quad, entry.vertexDesc);
            }

            entry.generation = s_nextGeneration.fetch_add(1, std::memory_order_relaxed);
            entry.wantLevel = static_cast<std::uint8_t>(quadLevel);
            entry.wantEdgeLevels = edgeLevels;
            if (quadLevel == 0) {
                revertEntry(entry); // instant swap; the buffers go to the worker graveyard
            } else {
                pendingBuilds.emplace_back(
                    std::max(std::abs(resolved.quadX - playerQuadX), std::abs(resolved.quadY - playerQuadY)),
                    Job {.key = entry.shape.get(),
                         .keepAlive = entry.shape,
                         .snapshot = entry.snapshot,
                         .level = quadLevel,
                         .edgeLevels = edgeLevels,
                         .generation = entry.generation});
            }
        }

        // Nearest terrain first: after a load or a teleport the whole region builds at once,
        // and the ground the player is actually looking at should stop being vanilla first
        if (!pendingBuilds.empty()) {
            std::sort(pendingBuilds.begin(), pendingBuilds.end(), [](const auto& jobA, const auto& jobB) -> bool {
                return jobA.first < jobB.first;
            });
            std::vector<Job> jobs;
            jobs.reserve(pendingBuilds.size());
            for (auto& [distance, job] : pendingBuilds) {
                jobs.push_back(std::move(job));
            }
            enqueueJobs(std::move(jobs));
        }
    } else {
        // No exterior grid around the player: everything still smoothed goes back to vanilla
        for (auto& [key, entry] : s_entries) {
            if (entry.smoothed.has_value() || entry.wantLevel != 0) {
                entry.generation = s_nextGeneration.fetch_add(1, std::memory_order_relaxed);
                revertEntry(entry);
            }
        }
    }

    // Everything the walk did not resolve is unverified: revert it, and prune it once nothing
    // else references it anymore.
    std::vector<RE::NiPointer<RE::BSTriShape>> retired;
    const auto now = std::chrono::steady_clock::now();
    for (auto iter = s_entries.begin(); iter != s_entries.end();) {
        Entry& entry = iter->second;
        if (entry.seen) {
            ++iter;
            continue;
        }

        // A smoothed mesh on a quad the loaded grid no longer accounts for is stale by
        // definition - out-of-range and detached cells alike end up here. Reverting is free,
        // and a quad that is merely mid-build loses nothing (it is never smoothed before its
        // first resolved pass). This also restores the vanilla buffers ahead of the erase
        // below, so a dying shape's destructor releases them exactly as if this plugin had
        // never touched the quad.
        if (entry.smoothed.has_value() || entry.wantLevel != 0) {
            entry.generation = s_nextGeneration.fetch_add(1, std::memory_order_relaxed);
            revertEntry(entry);
        }

        // Sole ownership means the engine released its references (the cell detached); the age
        // guard covers the window where a quad fresh out of the builder is only referenced by
        // us and a raw pointer on the build thread's stack (see K_PRUNE_MIN_AGE)
        if (entry.shape->GetRefCount() > 1 || (now - entry.registeredAt) < K_PRUNE_MIN_AGE) {
            // A young, still-referenced, unseen entry is likely mid-attach: the engine files
            // the shape into the scene graph after the build hook, with no event to announce
            // it. Have the worker fire another pass shortly (see s_retryPending) instead of
            // leaving the quad vanilla until the next crossing.
            if (entry.shape->GetRefCount() > 1 && (now - entry.registeredAt) < K_PRUNE_MIN_AGE) {
                s_retryPending.store(true, std::memory_order_relaxed);
            }
            ++iter;
            continue;
        }
        // The last reference must not drop here: one crossing retires dozens of dead builder
        // generations, and each destruction releases GPU buffers through the engine - a
        // main-thread burst VR frames cannot absorb. The worker drops them instead.
        retired.push_back(std::move(entry.shape));
        iter = s_entries.erase(iter);
    }
    if (!retired.empty()) {
        buryShapes(std::move(retired));
    }
}

void TerrainFalloff::revertEntry(Entry& entry)
{
    if (entry.smoothed.has_value()) {
        TerrainSubdivision::setMesh(*entry.shape, entry.vanilla);
        buryMesh(*entry.smoothed); // released on the worker; a pass can revert dozens at once
        entry.smoothed.reset();
    }
    entry.appliedLevel = 0;
    entry.appliedEdgeLevels = {};
    entry.wantLevel = 0;
    entry.wantEdgeLevels = {};
}

void TerrainFalloff::applyBuiltMesh(const Job& job,
                                    const std::optional<TerrainSubdivision::MeshBuffers>& mesh)
{
    const std::lock_guard<std::mutex> lock(s_entryMutex);
    const auto found = s_entries.find(job.key);
    if (found == s_entries.end() || found->second.generation != job.generation) {
        // The world moved on while this mesh was building
        if (mesh.has_value()) {
            buryMesh(*mesh);
        }
        return;
    }

    Entry& entry = found->second;
    if (!mesh.has_value()) {
        // The renderer refused the buffers (likely GPU memory); reset the target to the
        // current state and let the worker's next wakeup schedule a retry pass - waiting for
        // the next region change instead could leave a mismatched edge in place indefinitely
        // while the player stands still. The poll cadence is the retry backoff.
        spdlog::warn("Smoothed land mesh build failed; quad {} keeps its current mesh for now", entry.quad);
        entry.wantLevel = entry.appliedLevel;
        entry.wantEdgeLevels = entry.appliedEdgeLevels;
        s_retryPending.store(true, std::memory_order_relaxed);
        return;
    }

    // The swap itself: a few field writes on the main thread, no allocation, no upload, no
    // release (the replaced set goes to the worker)
    TerrainSubdivision::setMesh(*entry.shape, *mesh);
    if (entry.smoothed.has_value()) {
        buryMesh(*entry.smoothed); // an older variant or level
    }
    entry.smoothed = *mesh;
    entry.appliedLevel = static_cast<std::uint8_t>(job.level);
    entry.appliedEdgeLevels = job.edgeLevels;
}

void TerrainFalloff::ensureWorkerLocked()
{
    if (!s_workerStarted) {
        // Detached on purpose: joining at process exit would deadlock under the loader lock,
        // and the worker owns nothing that outlives the process
        std::thread(&TerrainFalloff::workerLoop).detach();
        s_workerStarted = true;
    }
}

void TerrainFalloff::enqueueJobs(std::vector<Job>&& jobs)
{
    const std::lock_guard<std::mutex> lock(s_jobMutex);
    ensureWorkerLocked();
    for (auto& job : jobs) {
        s_jobs.push_back(std::move(job));
    }
    s_jobSignal.notify_one();
}

void TerrainFalloff::buryShapes(std::vector<RE::NiPointer<RE::BSTriShape>>&& shapes)
{
    const std::lock_guard<std::mutex> lock(s_jobMutex);
    ensureWorkerLocked();
    for (auto& shape : shapes) {
        s_graveyard.push_back(std::move(shape));
    }
    s_jobSignal.notify_one();
}

void TerrainFalloff::buryMesh(const TerrainSubdivision::MeshBuffers& mesh)
{
    const std::lock_guard<std::mutex> lock(s_jobMutex);
    ensureWorkerLocked();
    s_meshGraveyard.push_back(mesh);
    s_jobSignal.notify_one();
}

void TerrainFalloff::pollPlayerQuad()
{
    auto* player = RE::PlayerCharacter::GetSingleton();
    if (player == nullptr) {
        return;
    }
    const auto position = player->GetPosition();
    const auto quadX = static_cast<std::int32_t>(std::floor(position.x / TerrainSubdivision::K_QUAD_WORLD_SIZE));
    const auto quadY = static_cast<std::int32_t>(std::floor(position.y / TerrainSubdivision::K_QUAD_WORLD_SIZE));
    const auto packed = (static_cast<std::int64_t>(quadX) << 32U) | static_cast<std::uint32_t>(quadY);
    if (s_lastPlayerQuad.exchange(packed, std::memory_order_relaxed) != packed) {
        scheduleRecompute();
    }
}

void TerrainFalloff::workerLoop()
{
    // Never latency-critical: the smoothing boundary sits at least a quad away, while the
    // game's own threads fight for every core during cell loads - especially in VR
    ::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);

    for (;;) {
        Job job;
        bool haveJob = false;
        std::vector<RE::NiPointer<RE::BSTriShape>> retired;
        std::vector<TerrainSubdivision::MeshBuffers> retiredMeshes;
        {
            std::unique_lock<std::mutex> lock(s_jobMutex);
            // Timed wait: the wakeups double as the player-position poll below
            s_jobSignal.wait_for(lock, K_PLAYER_POLL_INTERVAL, []() -> bool {
                return !s_jobs.empty() || !s_graveyard.empty() || !s_meshGraveyard.empty();
            });
            retired.swap(s_graveyard);
            retiredMeshes.swap(s_meshGraveyard);
            if (!s_jobs.empty()) {
                job = std::move(s_jobs.front());
                s_jobs.pop_front();
                haveJob = true;
            }
        }

        // Releasing here is the point: each shape destruction and each retired buffer set
        // goes through the engine's buffer manager, off the main thread (the engine destroys
        // shapes on its own loader threads the same way)
        retired.clear();
        for (const auto& mesh : retiredMeshes) {
            TerrainSubdivision::releaseMesh(mesh);
        }

        // Quad crossings inside a cell have no engine event; this poll is what moves the
        // region between cell borders
        pollPlayerQuad();

        // A pass that could not settle everything (a quad mid-attach with no event coming, a
        // build the renderer refused) leaves a note; fire the follow-up pass at poll cadence
        if (s_retryPending.exchange(false, std::memory_order_relaxed)) {
            scheduleRecompute();
        }

        if (!haveJob) {
            continue;
        }

        // Hold the batch until the engine has not built land for a beat: a fresh registration
        // means a cell load is in flight and every core is spoken for. Reverts are not gated
        // by this - they are instant main-thread field writes.
        for (;;) {
            const auto lastBuild = std::chrono::steady_clock::duration(s_lastLandBuild.load(std::memory_order_relaxed));
            const auto sinceLastBuild = std::chrono::steady_clock::now().time_since_epoch() - lastBuild;
            if (sinceLastBuild >= K_BUILD_QUIET_PERIOD) {
                break;
            }
            std::this_thread::sleep_for(K_BUILD_QUIET_PERIOD - sinceLastBuild);
        }

        // Skip builds the player already outran; cheaper than building and discarding
        {
            const std::lock_guard<std::mutex> lock(s_entryMutex);
            const auto found = s_entries.find(job.key);
            if (found == s_entries.end() || found->second.generation != job.generation) {
                continue;
            }
        }

        const auto mesh = TerrainSubdivision::buildSmoothedMesh(*job.snapshot, job.level, job.edgeLevels);

        // Trickle, not burst: the batch spreads over a few hundred milliseconds instead of
        // slamming the driver with back-to-back uploads
        std::this_thread::sleep_for(K_BUILD_SPACING);

        const auto* taskInterface = SKSE::GetTaskInterface();
        if (taskInterface == nullptr) {
            if (mesh.has_value()) {
                TerrainSubdivision::releaseMesh(*mesh);
            }
            // Reset the target like applyBuiltMesh's failure path would have, so a later
            // pass retries instead of believing this level is on its way
            const std::lock_guard<std::mutex> lock(s_entryMutex);
            const auto found = s_entries.find(job.key);
            if (found != s_entries.end() && found->second.generation == job.generation) {
                found->second.wantLevel = found->second.appliedLevel;
                found->second.wantEdgeLevels = found->second.appliedEdgeLevels;
                s_retryPending.store(true, std::memory_order_relaxed);
            }
            continue;
        }
        taskInterface->AddTask([job, mesh]() { applyBuiltMesh(job, mesh); });
    }
}
