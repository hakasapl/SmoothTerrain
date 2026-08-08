#include "TerrainCollision.hpp"

#include "ConfigLoader.hpp"
#include "HookUtil.hpp"
#include "Offsets.hpp"
#include "TerrainSubdivision.hpp"

#include "PCH.h"

#include <spdlog/spdlog.h>

#include <cstddef>
#include <cstdint>
#include <iterator>

using namespace SmoothTerrain;

void TerrainCollision::install()
{
    // Same runtime gate as the mesh path: the landscape offsets are only verified on SE, AE
    // and VR (TerrainSubdivision::install already logged why an unsupported runtime stays
    // vanilla)
    if (!Offsets::isRuntimeSupported()) {
        return;
    }

    const int level = ConfigLoader::getSubdivisions();
    if (level <= 0) {
        return; // nothing is smoothed at all (iSubdivisions = 0), so there is nothing to match
    }

    // Inner hook first: with only that one installed nothing ever publishes a refinement and
    // every build passes straight through, whereas the reverse order would leave the outer hook
    // sampling grids nobody consumes. Either way a half-installed pair is harmless.
    const std::uintptr_t initShape = Offsets::K_INIT_HEIGHT_FIELD_SHAPE.address();
    InitHook::s_func = initShape;
    const std::uintptr_t buildCollision = Offsets::K_BUILD_LAND_COLLISION.address();
    BuildHook::s_func = buildCollision;

    // Both targets have exactly one call site in the whole executable on all three flavors, so
    // a generous scan window cannot pick up an unrelated branch.
    constexpr std::size_t COLLISION_WINDOW = 0x600; /**< 0x285 on 1.6.1170, 0x299 on 1.5.97 and VR */
    const auto initSites = HookUtil::redirectBranches(buildCollision, COLLISION_WINDOW, initShape, InitHook::thunk);
    if (initSites.empty()) {
        spdlog::error("No height field shape init call site found in the land collision builder ({:#x}); "
                      "landscape collision stays vanilla",
                      buildCollision);
        return;
    }

    constexpr std::size_t GEOMETRY_WINDOW = 0x800; /**< 0x486 on 1.6.1170, 0x402 on 1.5.97 and VR */
    const auto buildSites = HookUtil::redirectBranches(
        Offsets::K_BUILD_LAND_GEOMETRY.address(), GEOMETRY_WINDOW, buildCollision, BuildHook::thunk);
    if (buildSites.empty()) {
        spdlog::error("No land collision builder call site found in the land geometry init ({:#x}); "
                      "landscape collision stays vanilla",
                      Offsets::K_BUILD_LAND_GEOMETRY.address());
        return;
    }

    const std::uint32_t dim = TerrainSubdivision::fineQuadDim(level);
    spdlog::info("Landscape collision refinement installed: level {} ({}x{} height field per quad, {} world units "
                 "between samples instead of 128)",
                 level,
                 dim,
                 dim,
                 TerrainSubdivision::K_QUAD_WORLD_SIZE / static_cast<float>(dim - 1));
}

auto TerrainCollision::Refinement::fieldFor(const HeightFieldCInfo* cinfo) const
    -> const TerrainSubdivision::QuadHeightField*
{
    if (cinfo == nullptr || cinfo->heights == nullptr) {
        return nullptr;
    }

    // Only the vanilla quadrant grid gets replaced; a different resolution means this is not the
    // build we sampled for and leaving it alone is the safe outcome
    constexpr auto VANILLA_DIM = static_cast<std::int32_t>(TerrainSubdivision::K_VANILLA_QUAD_DIM);
    if (cinfo->xRes != VANILLA_DIM || cinfo->zRes != VANILLA_DIM) {
        return nullptr;
    }

    // The quadrant is wherever this grid starts inside the cell's height table
    constexpr auto QUAD_STRIDE = static_cast<std::ptrdiff_t>(VANILLA_DIM) * VANILLA_DIM;
    const std::ptrdiff_t offset = std::distance(base, cinfo->heights);
    if (offset < 0 || (offset % QUAD_STRIDE) != 0) {
        return nullptr;
    }
    const auto quad = static_cast<std::size_t>(offset / QUAD_STRIDE);
    if (quad >= quads.size()) {
        return nullptr;
    }
    return &quads.at(quad);
}

auto TerrainCollision::buildRefinement(const LandCollisionDesc* desc,
                                       Refinement& refinement) -> bool
{
    const int level = ConfigLoader::getSubdivisions();
    if (level <= 0 || desc == nullptr || desc->heights == nullptr) {
        return false;
    }

    // Anything that is not the vanilla landscape descriptor keeps the collision the engine would
    // have built: the quadrant keying below assumes the four 289 float grids of one cell
    if (desc->quadCount != TerrainSubdivision::K_QUADS_PER_CELL
        || desc->gridDim != TerrainSubdivision::K_VANILLA_QUAD_DIM) {
        return false;
    }

    refinement.base = desc->heights;
    refinement.steps = 1U << static_cast<unsigned>(level);

    // One whole-cell grid for all four quadrants: that is what makes the sampled surface
    // continuous across the cell's interior quad borders
    TerrainSubdivision::CellHeightGrid grid {};
    TerrainSubdivision::buildCellHeightGrid(desc->heights, grid);
    for (std::uint32_t quad = 0; quad < TerrainSubdivision::K_QUADS_PER_CELL; ++quad) {
        refinement.quads.at(quad) = TerrainSubdivision::buildQuadHeightField(grid, quad, level);
    }
    return true;
}

auto TerrainCollision::BuildHook::thunk(void* cellMopp,
                                        const LandCollisionDesc* desc) -> bool
{
    Refinement refinement;
    const Refinement* const previous = s_refinement;
    if (buildRefinement(desc, refinement)) {
        s_refinement = &refinement;
    }

    // The engine's builder runs unchanged; the four nested shape builds inside it pick the
    // refined grids up through InitHook
    const bool result = s_func(cellMopp, desc);

    s_refinement = previous;
    return result;
}

auto TerrainCollision::InitHook::thunk(void* shape,
                                       HeightFieldCInfo* cinfo) -> std::uintptr_t
{
    const Refinement* const refinement = s_refinement;
    const auto* const field = refinement != nullptr ? refinement->fieldFor(cinfo) : nullptr;
    if (field == nullptr) {
        return s_func(shape, cinfo); // not a land quadrant we sampled: vanilla collision
    }

    // Swap the vanilla grid for the subdivided one, for this call only. The heights are in the
    // same stored space (relative to the same per-land base, so heightBias still converts them),
    // the sample spacing shrinks by exactly the subdivision factor, and the height range is the
    // one the engine would have computed itself had it been given this grid. It reads all of it
    // here and copies every sample into its own quantized shape, so restoring the block below
    // leaves nothing of ours behind.
    const auto* const vanillaHeights = cinfo->heights;
    const std::int32_t vanillaXRes = cinfo->xRes;
    const std::int32_t vanillaZRes = cinfo->zRes;
    const float vanillaScaleX = cinfo->scaleX;
    const float vanillaScaleZ = cinfo->scaleZ;
    const float vanillaMin = cinfo->minHeight;
    const float vanillaMax = cinfo->maxHeight;

    const auto steps = static_cast<float>(refinement->steps);
    cinfo->heights = field->heights.data();
    cinfo->xRes = static_cast<std::int32_t>(field->dim);
    cinfo->zRes = cinfo->xRes;
    cinfo->scaleX = vanillaScaleX / steps;
    cinfo->scaleZ = vanillaScaleZ / steps;
    cinfo->minHeight = cinfo->heightBias + field->minHeight;
    cinfo->maxHeight = cinfo->heightBias + field->maxHeight;

    const auto result = s_func(shape, cinfo);

    cinfo->heights = vanillaHeights;
    cinfo->xRes = vanillaXRes;
    cinfo->zRes = vanillaZRes;
    cinfo->scaleX = vanillaScaleX;
    cinfo->scaleZ = vanillaScaleZ;
    cinfo->minHeight = vanillaMin;
    cinfo->maxHeight = vanillaMax;
    return result;
}
