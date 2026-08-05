#include "TerrainSubdivision.hpp"

#include "ConfigLoader.hpp"
#include "HookUtil.hpp"
#include "Offsets.hpp"
#include "TerrainFalloff.hpp"

#include "PCH.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

using namespace SmoothTerrain;

void TerrainSubdivision::install()
{
    // One DLL loads on every runtime, but only SE's, AE's, and VR's landscape offsets are
    // verified; hooking an unverified runtime would call into an unrelated function (see Offsets.hpp)
    if (!Offsets::isRuntimeSupported()) {
        spdlog::warn("Terrain subdivision is only supported on Skyrim SE (1.5.x), AE (1.6.x), and VR "
                     "(1.4.15); the landscape offsets for this runtime have not been reverse engineered. "
                     "The plugin stays loaded but installs no hooks, and terrain renders as vanilla.");
        return;
    }

    if (ConfigLoader::getSubdivisions() <= 0) {
        spdlog::info("Terrain subdivision disabled (iSubdivisions = 0); no hooks installed");
        return;
    }

    // Every land quad in the game is built through the call sites below (verified via xrefs on
    // both 1.6.1170 and 1.5.97). Patching the call sites instead of the builder's entry keeps
    // the builder detourable by other mods (Community Shaders, ENB); our call into s_func below
    // then runs through whatever they install.
    const std::uintptr_t builder = Offsets::K_BUILD_QUAD_TRISHAPE.address();
    BuildQuadHook::s_func = builder;

    struct CallerSpec {
        REL::RelocationID id;
        std::size_t window {};
        const char* name {};
        bool callsBuilderDirectly {}; /**< False where this runtime's caller delegates to another entry below */
    };
    const std::array<CallerSpec, 2> callers {
        // AE's geometry init calls the builder itself; SE's and VR's call the quad build helper
        // below instead, so on those flavors that helper's single call site already covers the
        // init path too.
        CallerSpec {.id = Offsets::K_BUILD_LAND_GEOMETRY,
                    .window = 0x800, // 0x5CC on 1.6.1170, 0x4A5 on 1.5.97 and VR
                    .name = "land geometry init",
                    .callsBuilderDirectly = REL::Module::IsAE()},
        // Scan windows generously cover each caller's body without reaching the next function
        // that also calls the builder.
        CallerSpec {.id = Offsets::K_BUILD_LAND_QUADS,
                    .window = 0x200, // 0x126 on 1.6.1170, 0x117 on 1.5.97 and VR
                    .name = "land quad build",
                    .callsBuilderDirectly = true},
    };

    std::size_t patched = 0;
    for (const auto& caller : callers) {
        if (!caller.callsBuilderDirectly) {
            spdlog::debug("Skipping {}: this runtime reaches the builder through another call site", caller.name);
            continue;
        }
        const auto sites
            = HookUtil::redirectBranches(caller.id.address(), caller.window, builder, BuildQuadHook::thunk);
        if (sites.empty()) {
            spdlog::error("No BuildQuadTriShape call site found in {} ({:#x}); that path stays vanilla",
                          caller.name,
                          caller.id.address());
            continue;
        }
        for (const auto& site : sites) {
            ++patched;
            spdlog::info("Hooked BuildQuadTriShape call in {} at {:#x}", caller.name, site.address);
        }
    }

    spdlog::info("Terrain subdivision installed: {} call sites, level {} ({}x{} verts per quad)",
                 patched,
                 ConfigLoader::getSubdivisions(),
                 ((K_COARSE_DIM - 1) << ConfigLoader::getSubdivisions()) + 1,
                 ((K_COARSE_DIM - 1) << ConfigLoader::getSubdivisions()) + 1);
}

auto TerrainSubdivision::findIndexData(std::uint32_t vertexCount,
                                       std::uint32_t triangleCount) -> std::span<const std::uint16_t>
{
    for (int level = 1; level <= ConfigLoader::MAX_SUBDIVISIONS; ++level) {
        const std::uint32_t dim = ((K_COARSE_DIM - 1) << static_cast<unsigned>(level)) + 1;
        if (vertexCount != dim * dim || triangleCount != (dim - 1) * (dim - 1) * 2) {
            continue;
        }
        const std::lock_guard<std::mutex> lock(s_indexBufferMutex);
        const auto& indices = s_indexData.at(static_cast<std::size_t>(level));
        return {indices.data(), indices.size()}; // empty while that level has not been built yet
    }
    return {};
}

auto TerrainSubdivision::BuildQuadHook::thunk(RE::TESObjectLAND::LoadedLandData* data,
                                              std::uint32_t quad) -> RE::BSTriShape*
{
    // Vanilla builds the 17x17 quad first (through any detours other mods put on it)
    auto* shape = s_func(data, quad);

    if (shape == nullptr || data == nullptr || quad >= 4 || ConfigLoader::getSubdivisions() <= 0) {
        return shape;
    }

    // Every quad starts out vanilla, always: subdividing here would put full mesh generation
    // and a GPU upload inside the engine's cell loading, 16 times per cell - the load-time
    // hitch this plugin is structured to avoid. TerrainFalloff installs the smoothed mesh
    // from the main thread later, built in the background.
    if (TerrainFalloff::isEnabled()) {
        TerrainFalloff::registerQuad(shape, quad);
    }
    return shape;
}

auto TerrainSubdivision::validateQuadLayout(RE::BSTriShape& shape) -> std::optional<std::uint64_t>
{
    // The vanilla CPU-side vertex copy is the interpolation source for every attribute.
    // Geometry members live behind CommonLibSSE-NG's runtime accessors so their per-runtime
    // offsets resolve correctly.
    auto& geometryData = shape.GetGeometryRuntimeData();
    auto& triShapeData = shape.GetTrishapeRuntimeData();
    auto* const rendererData = geometryData.rendererData;
    if ((rendererData == nullptr) || (rendererData->rawVertexData == nullptr)) {
        return std::nullopt;
    }

    // Only touch exactly the mesh the vanilla builder makes; any other layout means another
    // mod got here first (or the pipeline changed) and vanilla is the safe outcome
    if (triShapeData.vertexCount != K_COARSE_VERTS || triShapeData.triangleCount != K_COARSE_TRIS) {
        return std::nullopt;
    }
    const auto desc = std::bit_cast<std::uint64_t>(geometryData.vertexDesc);
    constexpr std::uint64_t DESC_STRIDE_MASK = 0xF; // the desc's low nibble holds the stride in dwords
    if ((desc & DESC_STRIDE_MASK) * sizeof(std::uint32_t) != sizeof(LandVertex)) {
        return std::nullopt;
    }
    return desc;
}

auto TerrainSubdivision::snapshotQuad(const RE::TESObjectLAND::LoadedLandData& data,
                                      const void* rawVertexData,
                                      std::uint32_t quad,
                                      std::uint64_t vertexDesc) -> std::shared_ptr<const QuadSnapshot>
{
    // Copy everything a build needs out of the engine structures; the snapshot must stay
    // usable after the cell unloads or the shape changes hands (see the header)
    auto snapshot = std::make_shared<QuadSnapshot>();
    std::memcpy(snapshot->coarse.data(), rawVertexData, sizeof(snapshot->coarse));
    buildCellHeightGrid(data, snapshot->heights);
    snapshot->quad = quad;
    snapshot->vertexDesc = vertexDesc;
    return snapshot;
}

auto TerrainSubdivision::buildSmoothedMesh(const QuadSnapshot& snapshot,
                                           int level,
                                           std::uint8_t linearEdges) -> std::optional<MeshBuffers>
{
    const auto sub = static_cast<std::uint32_t>(1U << static_cast<unsigned>(level)); // segments per coarse quad edge
    const std::uint32_t fineDim = ((K_COARSE_DIM - 1) * sub) + 1;
    const std::uint32_t fineVerts = fineDim * fineDim;
    const std::uint32_t fineTris = (fineDim - 1) * (fineDim - 1) * 2;
    constexpr std::uint64_t MAX_FINE_DIM
        = ((K_COARSE_DIM - 1) << static_cast<unsigned>(ConfigLoader::MAX_SUBDIVISIONS)) + 1;
    static_assert(MAX_FINE_DIM * MAX_FINE_DIM <= std::numeric_limits<std::uint16_t>::max(),
                  "the maximum subdivision level must stay within 16-bit vertex indices");

    const auto& coarse = snapshot.coarse;
    const auto& grid = snapshot.heights;

    // Quadrant placement in mesh-local space and within the cell grid (0 = SW, 1 = SE,
    // 2 = NW, 3 = NE; matches the vanilla builder's offset tables)
    const std::uint32_t quadX = snapshot.quad & 1U;
    const std::uint32_t quadY = snapshot.quad >> 1U;
    const float baseX = (static_cast<float>(quadX) * K_QUAD_SIZE) - K_QUAD_SIZE;
    const float baseY = (static_cast<float>(quadY) * K_QUAD_SIZE) - K_QUAD_SIZE;
    const auto gridBaseX = static_cast<int>(quadX * (K_COARSE_DIM - 1));
    const auto gridBaseY = static_cast<int>(quadY * (K_COARSE_DIM - 1));
    const float step = K_COARSE_STEP / static_cast<float>(sub);

    const float smoothness = ConfigLoader::getSmoothness();
    const float maxRise = ConfigLoader::getMaxRise();

    std::vector<LandVertex> fine(fineVerts);
    for (std::uint32_t j = 0; j < fineDim; ++j) {
        const std::uint32_t coarseY = j / sub;
        const float fracY = static_cast<float>(j % sub) / static_cast<float>(sub);
        for (std::uint32_t i = 0; i < fineDim; ++i) {
            const std::uint32_t coarseX = i / sub;
            const float fracX = static_cast<float>(i % sub) / static_cast<float>(sub);
            LandVertex& out = fine.at((static_cast<std::size_t>(j) * fineDim) + i);

            // Original LAND verts are preserved bit-exact, positions included
            if (fracX == 0.0F && fracY == 0.0F) {
                out = coarse.at((coarseY * K_COARSE_DIM) + coarseX);
                continue;
            }

            // Every attribute except the height is bilinear between the four surrounding
            // vanilla records; on grid lines this collapses to the shared line's records,
            // which keeps quad and cell seams consistent
            const std::uint32_t nextX = std::min(coarseX + 1, K_COARSE_DIM - 1);
            const std::uint32_t nextY = std::min(coarseY + 1, K_COARSE_DIM - 1);
            out = lerpVertex(coarse.at((coarseY * K_COARSE_DIM) + coarseX),
                             coarse.at((coarseY * K_COARSE_DIM) + nextX),
                             coarse.at((nextY * K_COARSE_DIM) + coarseX),
                             coarse.at((nextY * K_COARSE_DIM) + nextX),
                             fracX,
                             fracY);

            out.posX = baseX + (static_cast<float>(i) * step);
            out.posY = baseY + (static_cast<float>(j) * step);

            // A vertex on a pinned border line takes the straight vanilla segments instead of
            // the spline: an unsmoothed neighbor quad renders exactly this line, whether it is
            // a cell border or the cell's interior cross line, so the two meshes meet without
            // a crack. Only the height changes; the bilinear attributes above already collapse
            // to the shared edge records and match the neighbor's shading. A vertex cannot
            // satisfy both tests at once - that would need fracX == fracY == 0, the
            // original-vert case handled above.
            const int gx = gridBaseX + static_cast<int>(coarseX);
            const int gy = gridBaseY + static_cast<int>(coarseY);
            const bool pinX = fracX == 0.0F
                && ((((linearEdges & K_EDGE_WEST) != 0) && gx == gridBaseX)
                    || (((linearEdges & K_EDGE_EAST) != 0)
                        && gx == gridBaseX + static_cast<int>(K_COARSE_DIM) - 1));
            const bool pinY = fracY == 0.0F
                && ((((linearEdges & K_EDGE_SOUTH) != 0) && gy == gridBaseY)
                    || (((linearEdges & K_EDGE_NORTH) != 0)
                        && gy == gridBaseY + static_cast<int>(K_COARSE_DIM) - 1));
            if (pinX) {
                out.posZ = lerp(gridHeight(grid, gx, gy), gridHeight(grid, gx, gy + 1), fracY);
            } else if (pinY) {
                out.posZ = lerp(gridHeight(grid, gx, gy), gridHeight(grid, gx + 1, gy), fracX);
            } else {
                out.posZ = sampleHeight(grid, gx, gy, fracX, fracY, smoothness, maxRise);
            }
        }
    }

    // Bounding sphere over the new verts (same construction as the engine: box center,
    // radius to the farthest vert), so culling accounts for the spline overshoot between
    // the original verts.
    RE::NiPoint3 minPos {
        std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    RE::NiPoint3 maxPos {std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest()};
    for (const auto& vert : fine) {
        minPos.x = std::min(minPos.x, vert.posX);
        minPos.y = std::min(minPos.y, vert.posY);
        minPos.z = std::min(minPos.z, vert.posZ);
        maxPos.x = std::max(maxPos.x, vert.posX);
        maxPos.y = std::max(maxPos.y, vert.posY);
        maxPos.z = std::max(maxPos.z, vert.posZ);
    }
    const RE::NiPoint3 center = (minPos + maxPos) * K_HALF;
    float radiusSq = 0.0F;
    for (const auto& vert : fine) {
        const RE::NiPoint3 delta {vert.posX - center.x, vert.posY - center.y, vert.posZ - center.z};
        radiusSq = std::max(radiusSq, delta.SqrLength());
    }

    auto* const indexBuffer = getIndexBuffer(level);
    if (indexBuffer == nullptr) {
        return std::nullopt;
    }
    auto* const renderer = RE::BSGraphics::Renderer::GetSingleton();
    if (renderer == nullptr) {
        return std::nullopt;
    }
    static const REL::Relocation<Offsets::CreateTriShapeData_t> createTriShapeData {Offsets::K_CREATE_TRISHAPE_DATA};
    auto* const newData = createTriShapeData(
        renderer, fine.data(), fineVerts * sizeof(LandVertex), snapshot.vertexDesc, indexBuffer);
    if (newData == nullptr) {
        return std::nullopt;
    }

    return MeshBuffers {.rendererData = newData,
                        .vertexCount = fineVerts,
                        .triangleCount = fineTris,
                        .boundCenter = center,
                        .boundRadius = std::sqrt(radiusSq)};
}

auto TerrainSubdivision::currentMesh(RE::BSTriShape& shape) -> MeshBuffers
{
    auto& geometryData = shape.GetGeometryRuntimeData();
    auto& triShapeData = shape.GetTrishapeRuntimeData();
    const auto& modelBound = shape.GetModelData().modelBound;
    return MeshBuffers {.rendererData = geometryData.rendererData,
                        .vertexCount = triShapeData.vertexCount,
                        .triangleCount = triShapeData.triangleCount,
                        .boundCenter = modelBound.center,
                        .boundRadius = modelBound.radius};
}

void TerrainSubdivision::setMesh(RE::BSTriShape& shape,
                                 const MeshBuffers& mesh)
{
    auto& geometryData = shape.GetGeometryRuntimeData();
    auto& triShapeData = shape.GetTrishapeRuntimeData();
    geometryData.rendererData = mesh.rendererData;
    triShapeData.vertexCount = static_cast<std::uint16_t>(mesh.vertexCount);
    triShapeData.triangleCount = static_cast<std::uint16_t>(mesh.triangleCount);
    auto& modelBound = shape.GetModelData().modelBound;
    modelBound.center = mesh.boundCenter;
    modelBound.radius = mesh.boundRadius;
}

void TerrainSubdivision::releaseMesh(const MeshBuffers& mesh) { releaseRendererData(mesh.rendererData); }

void TerrainSubdivision::buildCellHeightGrid(const RE::TESObjectLAND::LoadedLandData& data,
                                             std::array<std::array<float,
                                                                   K_CELL_DIM>,
                                                        K_CELL_DIM>& grid)
{
    // Bounds-checked copy of the engine's float[4][289] height table
    std::array<std::array<float, K_COARSE_VERTS>, K_QUAD_COUNT> quadHeights {};
    static_assert(sizeof(quadHeights) == sizeof(RE::TESObjectLAND::LoadedLandData::heights),
                  "the height table layout must match the engine's");
    std::memcpy(quadHeights.data(), &data.heights, sizeof(quadHeights));

    // Quadrants overlap on their shared rows/columns with identical values (they were split
    // from the one 33x33 LAND grid at load), so overwrite order does not matter
    for (std::uint32_t quad = 0; quad < K_QUAD_COUNT; ++quad) {
        const std::uint32_t offsetX = (quad & 1U) * (K_COARSE_DIM - 1);
        const std::uint32_t offsetY = (quad >> 1U) * (K_COARSE_DIM - 1);
        for (std::uint32_t row = 0; row < K_COARSE_DIM; ++row) {
            for (std::uint32_t col = 0; col < K_COARSE_DIM; ++col) {
                grid.at(offsetY + row).at(offsetX + col) = quadHeights.at(quad).at((row * K_COARSE_DIM) + col);
            }
        }
    }
}

auto TerrainSubdivision::gridHeight(const std::array<std::array<float,
                                                                K_CELL_DIM>,
                                                     K_CELL_DIM>& grid,
                                    int x,
                                    int y) -> float
{
    constexpr int LAST = static_cast<int>(K_CELL_DIM) - 1;

    constexpr float MIRROR_WEIGHT = 2.0F; /**< mirror extrapolation: h(-1) = 2 * h(0) - h(1) */

    // Mirror-extrapolate one axis at a time; recursion depth is at most two (corners)
    if (x < 0) {
        return (MIRROR_WEIGHT * gridHeight(grid, 0, y)) - gridHeight(grid, -x, y);
    }
    if (x > LAST) {
        return (MIRROR_WEIGHT * gridHeight(grid, LAST, y)) - gridHeight(grid, (2 * LAST) - x, y);
    }
    if (y < 0) {
        return (MIRROR_WEIGHT * gridHeight(grid, x, 0)) - gridHeight(grid, x, -y);
    }
    if (y > LAST) {
        return (MIRROR_WEIGHT * gridHeight(grid, x, LAST)) - gridHeight(grid, x, (2 * LAST) - y);
    }
    return grid.at(static_cast<std::size_t>(y)).at(static_cast<std::size_t>(x));
}

auto TerrainSubdivision::limitedTangent(float prev,
                                        float knot,
                                        float next,
                                        float rise) -> float
{
    constexpr float TANGENT_SCALE = 0.5F; /**< central-difference tangent weight */

    // Catmull-Rom's own tangent: the central difference of the knot's two neighbors
    const float raw = (next - prev) * TANGENT_SCALE;

    // Only an upward excursion can push terrain into a static mesh the vanilla surface passed
    // under, and each sign of the tangent lifts the curve on one side of the knot only: in the
    // Hermite basis the tangent's weight is t * (t - 1)^2 on the segment ahead of the knot,
    // which is never negative, and t^3 - t^2 on the segment behind it, which is never positive.
    // So a positive tangent can only lift the curve ahead of the knot and a negative one only
    // behind it, which lets each direction be bounded on its own against the secant it would be
    // lifting over. Fritsch-Carlson's factor of three is exactly the steepest a tangent can be
    // without carrying the curve past the far knot of that segment.
    //
    // Both bounds collapse to zero at a crest (heights rise into the knot and fall back out of
    // it), which is the case behind the reported artifact: a span of hidden garbage heights
    // under a mesh covering the landscape reads as a flat run with a drop at each end, and the
    // unlimited curve bulges that whole span up through the mesh. In a dip they go the other
    // way and leave the raw tangent alone, so valley floors and the gentle side of a step stay
    // as round as unlimited Catmull-Rom draws them - dropping below the vanilla surface cannot
    // poke through anything resting above it.
    //
    // Both bounds are then loosened by the caller's rise allowance, which is what lets a crest
    // round over at all. Its conversion into a tangent is exact: the two Hermite tangent bases
    // peak at 4/27, and both tangents of a segment can be pushing it up at once, so letting each
    // exceed its bound by s raises the curve by at most (8 / 27) * s. K_RISE_TO_TANGENT inverts
    // that, so a slack of K_RISE_TO_TANGENT * rise buys exactly `rise` world units of height and
    // no more. Being an absolute allowance rather than a fraction of the local relief is what
    // makes it useful: a gentle crest asks for only a few units of tangent and is left alone
    // entirely, while a cliff edge is still cut hard, which is the one that would have poked.
    constexpr float MONOTONE_LIMIT = 3.0F;
    const float slack = K_RISE_TO_TANGENT * rise;
    const float upper = (MONOTONE_LIMIT * std::max(0.0F, next - knot)) + slack;
    const float lower = (MONOTONE_LIMIT * std::min(0.0F, knot - prev)) - slack;

    return std::clamp(raw, lower, upper);
}

auto TerrainSubdivision::catmullRom(float p0,
                                    float p1,
                                    float p2,
                                    float p3,
                                    float t,
                                    float rise) -> float
{
    // Catmull-Rom in Hermite form: the cubic through p1 (t=0) and p2 (t=1) whose end tangents
    // come from the neighboring points. Both segments meeting at a knot derive the same tangent
    // there, which is what removes the vanilla mesh's creases at the original verts.
    constexpr float BASIS_TWO = 2.0F; /**< Hermite basis coefficient */
    constexpr float BASIS_THREE = 3.0F; /**< Hermite basis coefficient */

    const float tangent1 = limitedTangent(p0, p1, p2, rise);
    const float tangent2 = limitedTangent(p1, p2, p3, rise);
    const float tSq = t * t;
    const float tCu = tSq * t;

    const float basisP1 = ((BASIS_TWO * tCu) - (BASIS_THREE * tSq)) + 1.0F;
    const float basisT1 = (tCu - (BASIS_TWO * tSq)) + t;
    const float basisP2 = (BASIS_THREE * tSq) - (BASIS_TWO * tCu);
    const float basisT2 = tCu - tSq;
    return (basisP1 * p1) + (basisT1 * tangent1) + (basisP2 * p2) + (basisT2 * tangent2);
}

auto TerrainSubdivision::lerp(float valA,
                              float valB,
                              float t) -> float
{
    return valA + ((valB - valA) * t);
}

auto TerrainSubdivision::bilerp(float v00,
                                float v10,
                                float v01,
                                float v11,
                                float fracX,
                                float fracY) -> float
{
    return lerp(lerp(v00, v10, fracX), lerp(v01, v11, fracX), fracY);
}

auto TerrainSubdivision::sampleHeight(const std::array<std::array<float,
                                                                  K_CELL_DIM>,
                                                       K_CELL_DIM>& grid,
                                      int cellX,
                                      int cellY,
                                      float fracX,
                                      float fracY,
                                      float smoothness,
                                      float maxRise) -> float
{
    // The two passes below each get half the allowance, so a vert that both of them lift ends
    // up at most maxRise above the four original verts around it - the figure the INI names
    const float risePerPass = maxRise * K_HALF;

    // Exact grid point: return the stored height untouched
    if (fracX == 0.0F && fracY == 0.0F) {
        return gridHeight(grid, cellX, cellY);
    }

    // Separable Catmull-Rom: rows first, then across. When a fraction is exactly zero the
    // interpolation collapses to that line's samples only, which is what keeps borders
    // seam-free (both sides of a shared line use the very same samples).
    std::array<float, 4> rows {};
    for (int offset = -1; offset <= 2; ++offset) {
        const int sampleY = cellY + offset;
        if (fracX == 0.0F) {
            rows.at(static_cast<std::size_t>(offset + 1)) = gridHeight(grid, cellX, sampleY);
        } else {
            rows.at(static_cast<std::size_t>(offset + 1)) = catmullRom(gridHeight(grid, cellX - 1, sampleY),
                                                                       gridHeight(grid, cellX, sampleY),
                                                                       gridHeight(grid, cellX + 1, sampleY),
                                                                       gridHeight(grid, cellX + 2, sampleY),
                                                                       fracX,
                                                                       risePerPass);
        }
    }
    // Each row already sits at most risePerPass above its own two bracketing grid samples, so
    // bounding this pass the same way puts the result within maxRise of the four verts around it
    const float smooth
        = fracY == 0.0F ? rows.at(1) : catmullRom(rows.at(0), rows.at(1), rows.at(2), rows.at(3), fracY, risePerPass);

    if (smoothness >= 1.0F) {
        return smooth;
    }

    // Blend toward the flat (bilinear) surface; identical to vanilla shading at 0
    const float flat = bilerp(gridHeight(grid, cellX, cellY),
                              gridHeight(grid, cellX + 1, cellY),
                              gridHeight(grid, cellX, cellY + 1),
                              gridHeight(grid, cellX + 1, cellY + 1),
                              fracX,
                              fracY);
    return lerp(flat, smooth, smoothness);
}

auto TerrainSubdivision::lerpVertex(const LandVertex& c00,
                                    const LandVertex& c10,
                                    const LandVertex& c01,
                                    const LandVertex& c11,
                                    float fracX,
                                    float fracY) -> LandVertex
{
    constexpr float UNIT_FLOAT_SCALE = 2.0F; /**< decode of the (v + 1) * 0.5 float encoding */

    LandVertex out {};

    // Position is overwritten by the caller; carry the bilinear as a harmless default
    out.posX = bilerp(c00.posX, c10.posX, c01.posX, c11.posX, fracX, fracY);
    out.posY = bilerp(c00.posY, c10.posY, c01.posY, c11.posY, fracX, fracY);
    out.posZ = bilerp(c00.posZ, c10.posZ, c01.posZ, c11.posZ, fracX, fracY);

    // UVs are linear over the grid, so bilinear reproduces the vanilla mapping exactly
    out.u = floatToHalf(
        bilerp(halfToFloat(c00.u), halfToFloat(c10.u), halfToFloat(c01.u), halfToFloat(c11.u), fracX, fracY));
    out.v = floatToHalf(
        bilerp(halfToFloat(c00.v), halfToFloat(c10.v), halfToFloat(c01.v), halfToFloat(c11.v), fracX, fracY));

    // Unit vectors: decode, bilerp, renormalize, re-encode. Falls back to the nearest
    // corner's bytes if the lerp degenerates (opposed vectors), which cannot happen on
    // sanely authored land normals but costs nothing to guard.
    const auto lerpUnit = [&](auto&& decode, auto&& encode) -> void {
        const auto vec00 = decode(c00);
        const auto vec10 = decode(c10);
        const auto vec01 = decode(c01);
        const auto vec11 = decode(c11);
        RE::NiPoint3 blended {bilerp(vec00.x, vec10.x, vec01.x, vec11.x, fracX, fracY),
                              bilerp(vec00.y, vec10.y, vec01.y, vec11.y, fracX, fracY),
                              bilerp(vec00.z, vec10.z, vec01.z, vec11.z, fracX, fracY)};
        const float length = blended.Length();
        constexpr float MIN_LENGTH = 0.000001F;
        if (length <= MIN_LENGTH) {
            // Degenerate blend: fall back to the nearest corner's value
            if (fracX < K_HALF) {
                blended = fracY < K_HALF ? vec00 : vec01;
            } else {
                blended = fracY < K_HALF ? vec10 : vec11;
            }
        } else {
            blended = blended * (1.0F / length);
        }
        encode(blended);
    };

    lerpUnit(
        [](const LandVertex& corner) -> RE::NiPoint3 {
            return {decodeUnitByte(corner.normal.at(0)),
                    decodeUnitByte(corner.normal.at(1)),
                    decodeUnitByte(corner.normal.at(2))};
        },
        [&out](const RE::NiPoint3& vec) -> void {
            out.normal = {encodeUnitByte(vec.x), encodeUnitByte(vec.y), encodeUnitByte(vec.z)};
        });

    lerpUnit(
        [](const LandVertex& corner) -> RE::NiPoint3 {
            return {(corner.tangentXEnc * UNIT_FLOAT_SCALE) - 1.0F,
                    decodeUnitByte(corner.tangentYEnc),
                    decodeUnitByte(corner.tangentZEnc)};
        },
        [&out](const RE::NiPoint3& vec) -> void {
            out.tangentXEnc = (vec.x + 1.0F) * K_HALF;
            out.tangentYEnc = encodeUnitByte(vec.y);
            out.tangentZEnc = encodeUnitByte(vec.z);
        });

    lerpUnit(
        [](const LandVertex& corner) -> RE::NiPoint3 {
            return {decodeUnitByte(corner.bitangent.at(0)),
                    decodeUnitByte(corner.bitangent.at(1)),
                    decodeUnitByte(corner.bitangent.at(2))};
        },
        [&out](const RE::NiPoint3& vec) -> void {
            out.bitangent = {encodeUnitByte(vec.x), encodeUnitByte(vec.y), encodeUnitByte(vec.z)};
        });

    // Colors and texture blend weights interpolate directly in byte space
    for (std::size_t idx = 0; idx < out.color.size(); ++idx) {
        out.color.at(idx) = roundByte(bilerp(static_cast<float>(c00.color.at(idx)),
                                             static_cast<float>(c10.color.at(idx)),
                                             static_cast<float>(c01.color.at(idx)),
                                             static_cast<float>(c11.color.at(idx)),
                                             fracX,
                                             fracY));
    }
    for (std::size_t idx = 0; idx < out.blend.size(); ++idx) {
        out.blend.at(idx) = roundByte(bilerp(static_cast<float>(c00.blend.at(idx)),
                                             static_cast<float>(c10.blend.at(idx)),
                                             static_cast<float>(c01.blend.at(idx)),
                                             static_cast<float>(c11.blend.at(idx)),
                                             fracX,
                                             fracY));
    }

    out.pad = {0, 0};
    return out;
}

auto TerrainSubdivision::getIndexBuffer(int level) -> IndexBufferData*
{
    const std::lock_guard<std::mutex> lock(s_indexBufferMutex);
    auto& slot = s_indexBuffers.at(static_cast<std::size_t>(level));
    if (slot != nullptr) {
        return slot;
    }

    auto* const renderer = RE::BSGraphics::Renderer::GetSingleton();
    if (renderer == nullptr) {
        return nullptr;
    }

    // The CPU copy outlives the upload: the engine's decal builder reads land triangles from
    // CPU memory, and land shapes carry no index data of their own (see DecalFix)
    auto& indices = s_indexData.at(static_cast<std::size_t>(level));
    if (indices.empty()) {
        indices = buildIndices(level);
    }

    static const REL::Relocation<Offsets::CreateIndexBuffer_t> createIndexBuffer {Offsets::K_CREATE_INDEX_BUFFER};
    slot = static_cast<IndexBufferData*>(
        createIndexBuffer(renderer, static_cast<std::uint32_t>(indices.size()), indices.data()));
    if (slot == nullptr) {
        spdlog::error("Failed to create the level {} land index buffer ({} indices)", level, indices.size());
    } else {
        spdlog::info("Created shared level {} land index buffer ({} indices)", level, indices.size());
    }
    return slot;
}

auto TerrainSubdivision::buildIndices(int level) -> std::vector<std::uint16_t>
{
    // Same triangulation as the vanilla 17x17 builder, generalized: a checkerboard of
    // flipped diagonals ((row ^ col) & 1) with identical winding
    const auto sub = static_cast<std::uint32_t>(1U << static_cast<unsigned>(level));
    const std::uint32_t dim = ((K_COARSE_DIM - 1) * sub) + 1;
    std::vector<std::uint16_t> indices;
    constexpr std::size_t INDICES_PER_GRID_CELL = 6; /**< two triangles per grid cell */
    indices.reserve(static_cast<std::size_t>(dim - 1) * (dim - 1) * INDICES_PER_GRID_CELL);
    for (std::uint32_t row = 0; row < dim - 1; ++row) {
        for (std::uint32_t col = 0; col < dim - 1; ++col) {
            const auto lowerLeft = static_cast<std::uint16_t>((row * dim) + col);
            const auto lowerRight = static_cast<std::uint16_t>(lowerLeft + 1);
            const auto upperLeft = static_cast<std::uint16_t>(lowerLeft + dim);
            const auto upperRight = static_cast<std::uint16_t>(upperLeft + 1);
            if (((row ^ col) & 1U) == 0) {
                indices.insert(indices.end(), {upperRight, upperLeft, lowerLeft, lowerLeft, lowerRight, upperRight});
            } else {
                indices.insert(indices.end(), {upperLeft, lowerLeft, lowerRight, lowerRight, upperRight, upperLeft});
            }
        }
    }
    return indices;
}

void TerrainSubdivision::releaseRendererData(RE::BSGraphics::TriShape* rendererData)
{
    if (rendererData == nullptr) {
        return;
    }

    // BSTriShape::~BSTriShape releases renderer data through vfunc 5 of the geometry buffer
    // manager; using the same path keeps us symmetric with the engine's allocator
    static const REL::Relocation<void**> managerAddress {Offsets::K_GEOMETRY_BUFFER_MANAGER};
    void* const manager = *managerAddress; // NOLINT
    if (manager == nullptr) {
        return; // leaking one buffer set beats calling into a dead manager
    }
    const auto* const vtbl = *reinterpret_cast<Offsets::ReleaseRendererData_t* const*>(manager);
    constexpr std::size_t RELEASE_VFUNC = 5;
    vtbl[RELEASE_VFUNC](manager, rendererData); // NOLINT
}

auto TerrainSubdivision::halfToFloat(std::uint16_t half) -> float
{
    const std::uint32_t sign = static_cast<std::uint32_t>(half & K_HALF_SIGN_MASK) << K_SIGN_SHIFT;
    auto exponent = static_cast<std::int32_t>((half >> K_HALF_MANTISSA_BITS) & K_HALF_EXPONENT_MASK);
    std::uint32_t mantissa = half & K_HALF_MANTISSA_MASK;

    // An all-ones exponent field encodes infinity / NaN
    if (std::cmp_equal(exponent, K_HALF_EXPONENT_MASK)) {
        return std::bit_cast<float>(sign | K_FLOAT_EXPONENT_ALL_ONES | (mantissa << K_MANTISSA_SHIFT));
    }
    if (exponent == 0) {
        if (mantissa == 0) { // signed zero
            return std::bit_cast<float>(sign);
        }
        // Subnormal: shift the implicit leading 1 into place, adjusting the exponent
        while ((mantissa & K_HALF_IMPLICIT_ONE) == 0) {
            mantissa <<= 1U;
            --exponent;
        }
        ++exponent;
        mantissa &= K_HALF_MANTISSA_MASK;
    }
    // Rebias the raw exponent from half to float: value = raw - K_HALF_EXPONENT_BIAS,
    // so the float's raw field is value + K_FLOAT_EXPONENT_BIAS = raw + K_BIAS_DELTA
    return std::bit_cast<float>(sign | (static_cast<std::uint32_t>(exponent + K_BIAS_DELTA) << K_FLOAT_MANTISSA_BITS)
                                | (mantissa << K_MANTISSA_SHIFT));
}

auto TerrainSubdivision::floatToHalf(float value) -> std::uint16_t
{
    // Mirrors the engine's own conversion (seen inline in the vanilla land builder):
    // round to nearest even, overflow to infinity, NaN preserved as an all-ones payload
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> K_SIGN_SHIFT) & K_HALF_SIGN_MASK);
    const std::uint32_t magnitude = bits & K_FLOAT_ABS_MASK;

    // Largest float magnitude that still rounds into a finite half (the engine's own threshold)
    constexpr std::uint32_t MAX_FINITE_HALF_AS_FLOAT = 0x477FE000;
    if (magnitude > MAX_FINITE_HALF_AS_FLOAT) {
        const bool isNan = (magnitude & K_FLOAT_EXPONENT_ALL_ONES) == K_FLOAT_EXPONENT_ALL_ONES
            && (magnitude & K_FLOAT_MANTISSA_MASK) != 0;
        return static_cast<std::uint16_t>(sign | (isNan ? K_HALF_VALUE_MASK : K_HALF_INFINITY));
    }
    if (magnitude == 0) {
        return sign;
    }

    // Smallest float magnitude whose half encoding is still normal (raw exponent K_BIAS_DELTA + 1)
    constexpr std::uint32_t MIN_NORMAL_EXPONENT = static_cast<std::uint32_t>(K_BIAS_DELTA) + 1U;
    constexpr std::uint32_t MIN_NORMAL_HALF_AS_FLOAT = MIN_NORMAL_EXPONENT << K_FLOAT_MANTISSA_BITS;
    constexpr std::uint32_t EXPONENT_REBIAS = static_cast<std::uint32_t>(K_BIAS_DELTA) << K_FLOAT_MANTISSA_BITS;
    std::uint32_t shifted = 0;
    if (magnitude >= MIN_NORMAL_HALF_AS_FLOAT) {
        shifted = magnitude - EXPONENT_REBIAS; // rebias the exponent from float to half
    } else {
        // Subnormal half: shift the implicit one into the mantissa
        const std::uint32_t shift = MIN_NORMAL_EXPONENT - (magnitude >> K_FLOAT_MANTISSA_BITS);
        shifted = ((magnitude & K_FLOAT_MANTISSA_MASK) | K_FLOAT_IMPLICIT_ONE) >> shift;
    }
    constexpr std::uint32_t ROUND_BIAS = (1U << (K_MANTISSA_SHIFT - 1U)) - 1U;
    const std::uint32_t rounded = (shifted + ((shifted >> K_MANTISSA_SHIFT) & 1U) + ROUND_BIAS) >> K_MANTISSA_SHIFT;
    return static_cast<std::uint16_t>(sign | (rounded & K_HALF_VALUE_MASK));
}

auto TerrainSubdivision::encodeUnitByte(float value) -> std::uint8_t
{
    const float encoded = std::round((value + 1.0F) * K_UNIT_BYTE_SCALE);
    return static_cast<std::uint8_t>(std::clamp(encoded, 0.0F, K_BYTE_MAX));
}

auto TerrainSubdivision::decodeUnitByte(std::uint8_t value) -> float
{
    return (static_cast<float>(value) / K_UNIT_BYTE_SCALE) - 1.0F;
}

auto TerrainSubdivision::roundByte(float value) -> std::uint8_t
{
    return static_cast<std::uint8_t>(std::clamp(std::round(value), 0.0F, K_BYTE_MAX));
}
