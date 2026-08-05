#include "TerrainFalloff.hpp"

#include "ConfigLoader.hpp"
#include "Offsets.hpp"
#include "TerrainSubdivision.hpp"

#include "PCH.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

using namespace SmoothTerrain;

void TerrainFalloff::install()
{
    // The falloff piggybacks on the subdivision hook, so it is only meaningful when that hook
    // is installed (supported runtime, iSubdivisions > 0)
    if (!Offsets::isRuntimeSupported() || ConfigLoader::getSubdivisions() <= 0) {
        return;
    }

    if (ConfigLoader::getSmoothedGrids() <= 0) {
        spdlog::info("Distance falloff disabled (iSmoothedGrids = 0); every loaded grid is smoothed at build time");
        return;
    }

    s_enabled = true;
    const int side = (2 * ConfigLoader::getSmoothedGrids()) - 1;
    spdlog::info("Distance falloff active: a {0}x{0} grid square around the player renders the smoothed mesh "
                 "(radius {1} counting the player's grid, capped at the game's loaded-grid size); everything "
                 "further renders the vanilla mesh",
                 side,
                 ConfigLoader::getSmoothedGrids());
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
                                  const RE::TESObjectLAND::LoadedLandData& data,
                                  std::uint32_t quad)
{
    auto snapshot = TerrainSubdivision::snapshotQuad(*shape, data, quad);
    if (snapshot == nullptr) {
        // Same causes as the eager path's fallback (foreign layout, no CPU vertex copy):
        // another mod owns this quad and it stays vanilla for good
        spdlog::debug("Quad {} kept its vanilla land mesh", quad);
        return;
    }

    // Reading the freshly built shape off-main is safe here: the builder has not attached it
    // to anything yet, so no other thread can see it
    Entry entry;
    entry.shape = RE::NiPointer<RE::BSTriShape> {shape};
    entry.snapshot = std::move(snapshot);
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
        // The smoothed square, never wider than the loaded grid itself - the grid array's
        // length is the live uGridsToLoad, whatever set it. iSmoothedGrids counts the
        // player's own grid, so 2 covers the 3x3 square of the player's cell plus its full
        // neighbor ring. Membership being pure coordinate math is what makes the stitching
        // sound: both cells of any shared edge always agree on whether that edge is pinned.
        const int loadedRadius
            = grid->length > 0 ? (static_cast<int>(grid->length) - 1) / 2 : K_DEFAULT_LOADED_RADIUS;
        const int radius = std::min(ConfigLoader::getSmoothedGrids() - 1, loadedRadius);
        const auto inRegion = [&](int cellDeltaX, int cellDeltaY) -> bool {
            return std::max(std::abs(cellDeltaX), std::abs(cellDeltaY)) <= radius;
        };
        const int level = ConfigLoader::getSubdivisions();

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

                const int deltaX = coords->cellX - center->cellX;
                const int deltaY = coords->cellY - center->cellY;
                const bool wantSmooth = inRegion(deltaX, deltaY);

                // Cell edges whose neighbor stays coarse get pinned to the vanilla edge line;
                // a neighbor outside the loaded grid falls out of the same membership test
                std::uint8_t linearEdges = 0;
                if (wantSmooth) {
                    if (!inRegion(deltaX - 1, deltaY)) {
                        linearEdges |= TerrainSubdivision::K_EDGE_WEST;
                    }
                    if (!inRegion(deltaX + 1, deltaY)) {
                        linearEdges |= TerrainSubdivision::K_EDGE_EAST;
                    }
                    if (!inRegion(deltaX, deltaY - 1)) {
                        linearEdges |= TerrainSubdivision::K_EDGE_SOUTH;
                    }
                    if (!inRegion(deltaX, deltaY + 1)) {
                        linearEdges |= TerrainSubdivision::K_EDGE_NORTH;
                    }
                }

                // Resolve the shapes the cell actually renders through the scene graph, not
                // through LoadedLandData::geom. The engine's land geometry init rebuilds all
                // four geom slots once per quadrant (16 builder calls per cell) but attaches
                // only the generation current at each quadrant's turn - NiNode::SetAt(0,
                // geom[quad]) on that quadrant's multibound node. geom therefore ends up
                // holding the rendered shape for the last quadrant only; the attached child
                // is the authority on what is visible (verified on 1.6.1170 at 0x2A8D51).
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
                            continue; // not ours (or snapshot validation refused it at build time)
                        }
                        Entry& entry = found->second;
                        entry.seen = true;

                        const auto edges = static_cast<std::uint8_t>(
                            linearEdges & TerrainSubdivision::quadEdges(entry.quad));
                        if (wantSmooth == entry.wantSmoothed && (!wantSmooth || edges == entry.wantEdges)) {
                            continue; // already there, or already on its way there
                        }

                        entry.generation = s_nextGeneration.fetch_add(1, std::memory_order_relaxed);
                        entry.wantSmoothed = wantSmooth;
                        entry.wantEdges = edges;
                        if (!wantSmooth) {
                            revertEntry(entry); // instant: the vanilla buffers never went away
                        } else {
                            enqueueJob(Job {.key = found->first,
                                            .keepAlive = entry.shape,
                                            .snapshot = entry.snapshot,
                                            .level = level,
                                            .linearEdges = edges,
                                            .generation = entry.generation});
                        }
                    }
                }
            }
        }
    } else {
        // No exterior grid around the player: everything still smoothed goes back to vanilla
        for (auto& [key, entry] : s_entries) {
            if (entry.smoothed.has_value() || entry.wantSmoothed) {
                entry.generation = s_nextGeneration.fetch_add(1, std::memory_order_relaxed);
                revertEntry(entry);
            }
        }
    }

    // Everything the walk did not resolve is unverified: revert it, and prune it once nothing
    // else references it anymore.
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
        if (entry.smoothed.has_value() || entry.wantSmoothed) {
            entry.generation = s_nextGeneration.fetch_add(1, std::memory_order_relaxed);
            revertEntry(entry);
        }

        // Sole ownership means the engine released its references (the cell detached); the age
        // guard covers the window where a quad fresh out of the builder is only referenced by
        // us and a raw pointer on the build thread's stack (see K_PRUNE_MIN_AGE)
        if (entry.shape->GetRefCount() > 1 || (now - entry.registeredAt) < K_PRUNE_MIN_AGE) {
            ++iter;
            continue;
        }
        iter = s_entries.erase(iter);
    }
}

void TerrainFalloff::revertEntry(Entry& entry)
{
    if (entry.smoothed.has_value()) {
        TerrainSubdivision::setMesh(*entry.shape, entry.vanilla);
        TerrainSubdivision::releaseMesh(*entry.smoothed);
        entry.smoothed.reset();
    }
    entry.appliedEdges = 0;
    entry.wantSmoothed = false;
    entry.wantEdges = 0;
}

void TerrainFalloff::applyBuiltMesh(const Job& job,
                                    const std::optional<TerrainSubdivision::MeshBuffers>& mesh)
{
    const std::lock_guard<std::mutex> lock(s_entryMutex);
    const auto found = s_entries.find(job.key);
    if (found == s_entries.end() || found->second.generation != job.generation) {
        // The world moved on while this mesh was building
        if (mesh.has_value()) {
            TerrainSubdivision::releaseMesh(*mesh);
        }
        return;
    }

    Entry& entry = found->second;
    if (!mesh.has_value()) {
        // The renderer refused the buffers (likely GPU memory); resetting the target to the
        // current state lets the next region change retry instead of wedging the quad
        spdlog::warn("Smoothed land mesh build failed; quad {} keeps its current mesh for now", entry.quad);
        entry.wantSmoothed = entry.smoothed.has_value();
        entry.wantEdges = entry.appliedEdges;
        return;
    }

    // The swap itself: a few field writes on the main thread, no allocation, no upload
    TerrainSubdivision::setMesh(*entry.shape, *mesh);
    if (entry.smoothed.has_value()) {
        TerrainSubdivision::releaseMesh(*entry.smoothed); // an older smoothed variant
    }
    entry.smoothed = *mesh;
    entry.appliedEdges = job.linearEdges;
}

void TerrainFalloff::enqueueJob(Job&& job)
{
    const std::lock_guard<std::mutex> lock(s_jobMutex);
    if (!s_workerStarted) {
        // Detached on purpose: joining at process exit would deadlock under the loader lock,
        // and the worker owns nothing that outlives the process
        std::thread(&TerrainFalloff::workerLoop).detach();
        s_workerStarted = true;
    }
    s_jobs.push_back(std::move(job));
    s_jobSignal.notify_one();
}

void TerrainFalloff::workerLoop()
{
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(s_jobMutex);
            s_jobSignal.wait(lock, []() { return !s_jobs.empty(); });
            job = std::move(s_jobs.front());
            s_jobs.pop_front();
        }

        // Skip builds the player already outran; cheaper than building and discarding
        {
            const std::lock_guard<std::mutex> lock(s_entryMutex);
            const auto found = s_entries.find(job.key);
            if (found == s_entries.end() || found->second.generation != job.generation) {
                continue;
            }
        }

        const auto mesh = TerrainSubdivision::buildSmoothedMesh(*job.snapshot, job.level, job.linearEdges);

        const auto* taskInterface = SKSE::GetTaskInterface();
        if (taskInterface == nullptr) {
            if (mesh.has_value()) {
                TerrainSubdivision::releaseMesh(*mesh);
            }
            continue;
        }
        taskInterface->AddTask([job, mesh]() { applyBuiltMesh(job, mesh); });
    }
}
