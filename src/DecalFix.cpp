#include "DecalFix.hpp"

#include "ConfigLoader.hpp"
#include "HookUtil.hpp"
#include "Offsets.hpp"
#include "TerrainSubdivision.hpp"

#include "PCH.h"

#include <spdlog/spdlog.h>

#include <array>
#include <cstddef>
#include <cstdint>

using namespace SmoothTerrain;

void DecalFix::install()
{
    // Nothing is subdivided in these cases, so every land quad still matches the engine's
    // shared index list (TerrainSubdivision::install already logged why)
    if (!Offsets::isRuntimeSupported() || ConfigLoader::getSubdivisions() <= 0) {
        return;
    }

    const std::uintptr_t collector = Offsets::K_DECAL_COLLECT_TRIANGLES.address();
    CollectHook::s_func = collector;

    // Two entry points reach the collector: the decal's own Initialize, and a helper that
    // resolves the target's geometry first and tail jumps into it. Scan windows generously
    // cover each of them on all three flavors (0xDC9 / 0x2D on 1.6.1170, 0x9E6 / 0x2D on
    // 1.5.97, 0xA05 / 0x2D on VR).
    struct CallerSpec {
        REL::RelocationID id;
        std::size_t window {};
        const char* name {};
    };
    const std::array<CallerSpec, 2> callers {
        CallerSpec {.id = Offsets::K_DECAL_INITIALIZE, .window = 0x1000, .name = "decal initialize"},
        CallerSpec {.id = Offsets::K_DECAL_COLLECT_FROM_OBJECT, .window = 0x40, .name = "decal collect helper"},
    };

    std::size_t patched = 0;
    for (const auto& caller : callers) {
        const auto sites
            = HookUtil::redirectBranches(caller.id.address(), caller.window, collector, CollectHook::thunk);
        for (const auto& site : sites) {
            spdlog::debug("Hooked the decal triangle collector call in {} at {:#x}", caller.name, site.address);
        }
        patched += sites.size();
    }
    if (patched == 0) {
        spdlog::error("No decal triangle collector call site found; decals on subdivided terrain "
                      "would render incorrectly and can crash the game - leaving the decal path vanilla");
        return;
    }

    // Second half of the fix: the getter the collector uses for landscape triangles
    const std::uintptr_t getter = Offsets::K_GET_LAND_INDEX_LIST.address();
    IndexListHook::s_func = getter;
    // Overshooting the collector's body is safe here: the getter has exactly one caller in the
    // whole executable on both flavors, so no branch outside it can match.
    constexpr std::size_t COLLECTOR_WINDOW = 0x800; /**< 0x436 on 1.6.1170, 0x69A on 1.5.97 */
    const auto getterSites = HookUtil::redirectBranches(collector, COLLECTOR_WINDOW, getter, IndexListHook::thunk);
    if (getterSites.empty()) {
        spdlog::error("No land index list call site found in the decal triangle collector ({:#x}); "
                      "decals on subdivided terrain stay broken",
                      collector);
        return;
    }

    spdlog::info("Decal fix installed: {} collector call sites, {} index list call sites", patched, getterSites.size());
}

auto DecalFix::CollectHook::thunk(RE::BSTempEffectSimpleDecal* decal,
                                  RE::BSTriShape* shape) -> std::uintptr_t
{
    // Publish the list for the nested IndexListHook::thunk call and put back whatever was
    // there before, so a collection pass never leaks its list into an unrelated one
    const auto* const previous = s_indices;
    s_indices = landIndicesFor(shape);
    const auto result = s_func(decal, shape);
    s_indices = previous;
    return result;
}

auto DecalFix::IndexListHook::thunk() -> const std::uint16_t*
{
    // Outside a collection pass on one of our quads this is the engine's own getter
    const auto* const indices = CollectHook::s_indices;
    return indices != nullptr ? indices : s_func();
}

auto DecalFix::landIndicesFor(RE::BSTriShape* shape) -> const std::uint16_t*
{
    // The collector ignores everything but plain tri shapes, so anything else can keep the
    // engine's list - it is never read
    if (shape == nullptr || shape->GetType() != RE::BSGeometry::Type::kTriShape) {
        return nullptr;
    }

    // A shape is recognized by its subdivided vertex / triangle counts rather than by
    // remembering the pointers we handed out: shapes are destroyed and their memory reused
    // all the time, and the counts identify the triangulation exactly.
    const auto& triShapeData = shape->GetTrishapeRuntimeData();
    const auto indices = TerrainSubdivision::findIndexData(triShapeData.vertexCount, triShapeData.triangleCount);
    return indices.empty() ? nullptr : indices.data();
}
