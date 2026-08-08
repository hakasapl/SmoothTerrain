#pragma once

#include "TerrainSubdivision.hpp"

#include "PCH.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace SmoothTerrain {

/**
 * @brief Gives the landscape collision the same smoothed surface the subdivided mesh renders
 *
 * The engine collides against land through a sampled height field, not through the render mesh:
 * for each cell quadrant it copies LoadedLandData::heights[quad] - a 17x17 grid at 128 unit
 * spacing - into a quantized hkpCompressedSampledHeightFieldShape and gives that to a static
 * rigid body. Subdividing the mesh therefore changes nothing about what the player walks on:
 * every original LAND vertex keeps its exact height, so the two surfaces still meet at the grid
 * points, but in between the smoothed mesh curves while collision stays on the vanilla flat
 * triangles. That is the gap this class closes.
 *
 * It closes it at the source rather than by rewriting shapes afterwards. The height field's
 * resolution, spacing and heights all come from one construction info block that the collision
 * builder fills on its stack, so a call site hook on the shape's init reads that block, swaps in
 * a finer grid sampled from the same Catmull-Rom surface TerrainSubdivision draws, and restores
 * it when the call returns. Everything downstream is the engine's own code path working on
 * honest inputs: it quantizes our samples into its compressed shape (a copy - no pointer of ours
 * survives the call), derives the shape's bounds from the range we hand it, and registers the
 * body with the havok world exactly as it always did. Nothing havok-side is created, mutated or
 * released by this plugin, and no state has to be tracked or unwound afterwards.
 *
 * The level is always iSubdivisions, so the surface underfoot is the one on screen. Two
 * differences from the render path are deliberate:
 *  - No distance falloff. Collision is refined for every land quad that loads, at the full
 *    level, because anything in the loaded grid can stand on it while the mesh a mile away is
 *    only ever looked at. It also makes the collision surface continuous everywhere by
 *    construction: neighboring quads always sample the same spline at the same level, so no
 *    edge stitching is needed (see TerrainSubdivision::buildQuadHeightField).
 *  - Built during the cell load, unlike every mesh this plugin makes. Collision has to exist the
 *    moment the cell attaches, and the work is a fraction of what the surrounding engine code
 *    does anyway (sampling one float grid, against building four meshes and uploading them).
 *
 * A quad rendering below full level (the falloff's gradient, or vanilla past it) therefore
 * collides against the finer surface regardless. The two differ by at most the spline's
 * deviation from the coarser mesh's chords, which is bounded by fMaxRise upward, and only ever
 * out where the falloff has already decided the player is not - the full-level square is
 * centered on the player's own quad, so what they stand on always matches what they see.
 */
class TerrainCollision {
public:
    /**
     * @brief Installs the call site hooks; requires SKSE::AllocTrampoline beforehand
     */
    static void install();

    TerrainCollision() = delete;

private:
    /**
     * @brief The engine's landscape collision descriptor
     *
     * Built on the stack by the land geometry init and handed to K_BUILD_LAND_COLLISION, which
     * walks it into one height field per quadrant. Only heights, quadCount and gridDim are read
     * here, the rest is documented to pin the layout down (verified identical on 1.5.97,
     * 1.6.1170 and VR 1.4.15).
     */
    struct LandCollisionDesc {
        const float* heights; /**< 00: &LoadedLandData::heights[0][0]; quadrant q starts 289 floats in */
        float xSpacing; /**< 08: world units between grid columns (128) */
        float heightScale; /**< 0C: world units per stored height unit (1) */
        float ySpacing; /**< 10: world units between grid rows (128) */
        std::uint32_t pad14; /**< 14 */
        void** quadShapes; /**< 18: the quadrant render shapes the bodies are linked to */
        std::uint32_t quadCount; /**< 20: 4 */
        float baseHeight; /**< 24: the per-land height base the stored heights are relative to */
        std::uint32_t gridDim; /**< 28: 17 */
        std::uint32_t material; /**< 2C */
        std::uint32_t filterInfo; /**< 30 */
        std::uint32_t pad34; /**< 34 */
    };
    static_assert(sizeof(LandCollisionDesc) == 0x38);

    /**
     * @brief Construction info of one quadrant's sampled height field
     *
     * K_BUILD_LAND_COLLISION fills one of these per quadrant and passes it to
     * K_INIT_HEIGHT_FIELD_SHAPE, which reads it and nothing else. Scales are in havok units
     * (world units times bhkWorld::GetWorldScale()); heights are in the engine's stored height
     * space and heightBias converts them to world Z.
     */
    struct HeightFieldCInfo {
        std::uint64_t unk00; /**< 00 */
        std::uint64_t unk08; /**< 08 */
        float scaleX; /**< 10: havok units between grid columns */
        float scaleY; /**< 14: havok units per stored height unit */
        float scaleZ; /**< 18: havok units between grid rows */
        float scaleW; /**< 1C */
        std::int32_t xRes; /**< 20: samples along X (17) */
        std::int32_t zRes; /**< 24: samples along Z (17) */
        float minHeight; /**< 28: heightBias plus the lowest sample */
        float maxHeight; /**< 2C: heightBias plus the highest sample */
        std::uint8_t unk30; /**< 30 */
        std::array<std::uint8_t, 15> pad31; /**< 31 */
        const float* heights; /**< 40: the quadrant's grid, row major */
        float heightBias; /**< 48: world Z of a stored height of zero */
        std::uint32_t pad4C; /**< 4C */
        bool unk50; /**< 50 */
        std::array<std::uint8_t, 7> pad51; /**< 51 */
    };
    static_assert(offsetof(HeightFieldCInfo, scaleX) == 0x10);
    static_assert(offsetof(HeightFieldCInfo, xRes) == 0x20);
    static_assert(offsetof(HeightFieldCInfo, minHeight) == 0x28);
    static_assert(offsetof(HeightFieldCInfo, heights) == 0x40);
    static_assert(offsetof(HeightFieldCInfo, heightBias) == 0x48);
    static_assert(sizeof(HeightFieldCInfo) == 0x58);

    /**
     * @brief The subdivided height grids of one cell, alive for the duration of one collision build
     *
     * Lives on the outer hook's stack and is published to the inner hook through s_refinement.
     * The engine copies every sample it is given before the inner call returns, so nothing here
     * has to outlive the build.
     */
    struct Refinement {
        const float* base {}; /**< The descriptor's height table, the origin quadrants are keyed off */
        std::uint32_t steps {}; /**< Fine grid steps per vanilla grid step (1 << level) */
        std::array<TerrainSubdivision::QuadHeightField, TerrainSubdivision::K_QUADS_PER_CELL> quads;

        /**
         * @brief The refined grid standing in for the one a cinfo points at, if there is one
         *
         * Identifies the quadrant by where its height array sits in the cell's table, which is
         * exact integer arithmetic rather than a guess at call order, and refuses anything that
         * is not one of the four vanilla quadrant grids of this very cell.
         *
         * @param cinfo The construction info the engine is about to build a shape from
         * @return const TerrainSubdivision::QuadHeightField* The replacement, or nullptr to leave
         *         the build alone
         */
        [[nodiscard]] auto fieldFor(const HeightFieldCInfo* cinfo) const
            -> const TerrainSubdivision::QuadHeightField*;
    };

    /**
     * @brief Call site hook around the engine's land collision builder
     *
     * Samples the cell's four refined grids up front - one whole-cell height grid serves all
     * four quadrants - and publishes them for the nested shape builds below.
     */
    struct BuildHook {
        static auto thunk(void* cellMopp,
                          const LandCollisionDesc* desc) -> bool;
        static inline REL::Relocation<decltype(thunk)> s_func; /**< The unpatched builder */
    };

    /**
     * @brief Call site hook around the engine's height field shape init
     *
     * Runs inside BuildHook, on the same thread, once per quadrant.
     */
    struct InitHook {
        static auto thunk(void* shape,
                          HeightFieldCInfo* cinfo) -> std::uintptr_t;
        static inline REL::Relocation<decltype(thunk)> s_func; /**< The unpatched init */
    };

    /**
     * @brief Samples a cell's four refined height grids from its descriptor
     *
     * @param desc The descriptor the engine is about to build collision from
     * @param refinement Filled in on success
     * @return bool False when this is not the vanilla landscape descriptor (another plugin's
     *         layout, or a shape count / grid dimension this code was not written against), in
     *         which case the build stays vanilla
     */
    [[nodiscard]] static auto buildRefinement(const LandCollisionDesc* desc,
                                              Refinement& refinement) -> bool;

    /**
     * @brief The cell whose collision is being built on this thread, or nullptr
     *
     * Thread local because the engine loads cells on several threads at once; the inner hook
     * always runs inside its own thread's outer call, and the saved-and-restored handoff keeps
     * the pairing correct even if the engine ever nests two builds.
     */
    static thread_local inline const Refinement* s_refinement = nullptr;
};

} // namespace SmoothTerrain
