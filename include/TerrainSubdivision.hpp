#pragma once

#include "ConfigLoader.hpp"

#include "PCH.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace SmoothTerrain {

/**
 * @brief Subdivides the engine-built landscape meshes for smoother terrain
 *
 * The engine builds one BSTriShape per cell quadrant from the LAND record's 17x17 height grid
 * (BuildQuadTriShape, see Offsets.hpp). This class hooks every call site of that builder and
 * hands each freshly built vanilla quad to TerrainFalloff, which decides per cell - based on
 * the player's position - when the quad renders a subdivided version and when the vanilla
 * one. Nothing is ever built during the engine's own cell loading: subdivision happens later,
 * off the main thread, and swapping is a handful of field writes (a cell streaming in shows
 * its vanilla terrain for a beat before the smoothed mesh lands, which is the price of never
 * adding work to a load spike).
 *
 * The subdivided mesh itself: every original vertex is kept bit-exact in place and new
 * vertices are interpolated in between. Heights follow a Catmull-Rom spline over the whole
 * cell - the original verts act as fixed knots and both sides of every knot share one
 * tangent, so the faceted creases of the vanilla mesh disappear. The spline's tangents are
 * limited against upward overshoot (see limitedTangent), which holds every new vertex within
 * a fixed number of world units of the highest original vert around it and so keeps the
 * subdivided mesh out of static meshes the vanilla surface passed under; dips are left
 * unbounded. Every other attribute is bilinear. Materials, multibounds and the scene graph are
 * untouched, so the change is invisible to other plugins - only the mesh density changes. What
 * the player walks on is a separate engine structure (a sampled height field, not this mesh);
 * TerrainCollision rebuilds it from the same spline, through buildQuadHeightField below.
 *
 * This class supplies the mesh machinery: snapshotQuad captures everything a later
 * (off-thread) build needs, buildSmoothedMesh turns a snapshot into ready GPU buffers, and
 * setMesh / currentMesh / releaseMesh swap buffer sets on a shape without rebuilding
 * anything. Neighboring quads may render at different levels (the falloff can step the level
 * down with distance); the finer quad then emits its border-line vertices on the coarser
 * neighbor's edge polyline - down to the straight vanilla segments against unsmoothed
 * terrain - so the two meshes share their edge exactly and no cracks open between them (see
 * buildSmoothedMesh's edgeLevels).
 *
 * Call sites are patched with the SKSE trampoline (write_call) instead of a function-entry
 * detour, so other mods (Community Shaders, ENB, ...) can still detour the builder or anything
 * around it; our call into the "original" goes through whatever they install.
 */
class TerrainSubdivision {
public:
    /**
     * @brief Installs the call-site hooks; requires SKSE::AllocTrampoline beforehand
     */
    static void install();

    /**
     * @brief Returns the index list of the subdivided quad with these counts, if there is one
     *
     * Everything about a quad's triangulation follows from its grid dimension, so the vertex
     * and triangle counts identify the shared index list of a subdivided quad exactly. Used by
     * DecalFix, which has to hand the engine's decal builder the triangles of the shape it is
     * projecting onto.
     *
     * @param vertexCount Vertex count of the shape in question
     * @param triangleCount Triangle count of the shape in question
     * @return std::span<const std::uint16_t> The list (3 indices per triangle), or an empty
     *         span when no subdivision level built at runtime matches those counts
     */
    [[nodiscard]] static auto findIndexData(std::uint32_t vertexCount,
                                            std::uint32_t triangleCount) -> std::span<const std::uint16_t>;

private:
    //
    // Vanilla land mesh constants (verified against SkyrimSE.exe 1.6.1170 and 1.5.97, which
    // share the landscape vertex layout and LoadedLandData field offsets byte for byte)
    //
    constexpr static std::uint32_t K_QUAD_COUNT = 4; /**< Quadrants per cell (SW, SE, NW, NE) */
    constexpr static std::uint32_t K_COARSE_DIM = 17; /**< Verts per quadrant side in the vanilla mesh */
    constexpr static std::uint32_t K_COARSE_VERTS = K_COARSE_DIM * K_COARSE_DIM; /**< 289 verts per vanilla quad */
    constexpr static std::uint32_t K_COARSE_TRIS
        = (K_COARSE_DIM - 1) * (K_COARSE_DIM - 1) * 2; /**< 512 triangles per vanilla quad */
    constexpr static std::uint32_t K_CELL_DIM
        = (2 * (K_COARSE_DIM - 1)) + 1; /**< 33 verts per cell side (4 quadrants share edges) */
    constexpr static float K_COARSE_STEP = 128.0F; /**< World units between two original land verts */
    constexpr static float K_QUAD_SIZE
        = static_cast<float>(K_COARSE_DIM - 1) * K_COARSE_STEP; /**< 2048 world units per quadrant side */
    constexpr static std::size_t K_VERTEX_STRIDE = 40; /**< Byte stride of the engine's landscape vertex layout */
    constexpr static std::size_t K_BLEND_LAYERS = 6; /**< Texture blend weights stored per land vertex */
    constexpr static std::size_t K_INDEX_DATA_SIZE = 16; /**< Size of the engine's index buffer wrapper */

    //
    // Interpolation and attribute encoding constants
    //
    constexpr static float K_HALF = 0.5F; /**< Midpoints, nearest-corner splits and the (v + 1) / 2 encoding */
    constexpr static float K_BYTE_MAX = 255.0F; /**< Upper clamp when rounding an interpolated byte attribute */
    constexpr static float K_UNIT_BYTE_SCALE = 127.5F; /**< The engine's (v + 1) * 127.5 byte encoding of [-1, 1] */
    constexpr static float K_RISE_TO_TANGENT
        = 27.0F / 8.0F; /**< World units of allowed rise -> tangent slack (see limitedTangent) */

    //
    // IEEE 754 binary16 (half) and binary32 (float) field layouts, shared by halfToFloat and
    // floatToHalf; every mask and shift below is derived from these
    //
    constexpr static std::uint32_t K_HALF_MANTISSA_BITS = 10;
    constexpr static std::uint32_t K_HALF_EXPONENT_BITS = 5;
    constexpr static std::int32_t K_HALF_EXPONENT_BIAS = 15;
    constexpr static std::uint32_t K_FLOAT_MANTISSA_BITS = 23;
    constexpr static std::uint32_t K_FLOAT_EXPONENT_BITS = 8;
    constexpr static std::int32_t K_FLOAT_EXPONENT_BIAS = 127;

    constexpr static std::uint32_t K_HALF_SIGN_MASK = 1U << (K_HALF_EXPONENT_BITS + K_HALF_MANTISSA_BITS);
    constexpr static std::uint32_t K_HALF_EXPONENT_MASK = (1U << K_HALF_EXPONENT_BITS) - 1U;
    constexpr static std::uint32_t K_HALF_MANTISSA_MASK = (1U << K_HALF_MANTISSA_BITS) - 1U;
    constexpr static std::uint32_t K_HALF_IMPLICIT_ONE = 1U << K_HALF_MANTISSA_BITS;
    constexpr static std::uint32_t K_HALF_INFINITY = K_HALF_EXPONENT_MASK << K_HALF_MANTISSA_BITS;
    constexpr static std::uint32_t K_HALF_VALUE_MASK = K_HALF_SIGN_MASK - 1U; /**< everything but the sign bit */
    constexpr static std::uint32_t K_SIGN_SHIFT
        = (K_FLOAT_EXPONENT_BITS + K_FLOAT_MANTISSA_BITS) - (K_HALF_EXPONENT_BITS + K_HALF_MANTISSA_BITS);
    constexpr static std::uint32_t K_MANTISSA_SHIFT = K_FLOAT_MANTISSA_BITS - K_HALF_MANTISSA_BITS;
    constexpr static std::uint32_t K_FLOAT_EXPONENT_ALL_ONES = ((1U << K_FLOAT_EXPONENT_BITS) - 1U)
        << K_FLOAT_MANTISSA_BITS;
    constexpr static std::uint32_t K_FLOAT_MANTISSA_MASK = (1U << K_FLOAT_MANTISSA_BITS) - 1U;
    constexpr static std::uint32_t K_FLOAT_IMPLICIT_ONE = 1U << K_FLOAT_MANTISSA_BITS;
    constexpr static std::uint32_t K_FLOAT_ABS_MASK = K_FLOAT_EXPONENT_ALL_ONES | K_FLOAT_MANTISSA_MASK;
    constexpr static std::int32_t K_BIAS_DELTA = K_FLOAT_EXPONENT_BIAS - K_HALF_EXPONENT_BIAS;

    /**
     * @brief One vertex record of the engine's landscape vertex layout (stride 40,
     * vertex desc 0x000BB0080765040A)
     *
     * Encodings follow the vanilla builder: normals are biased bytes (signed char + 0x80),
     * tangent/bitangent components are (v + 1) * 127.5 bytes except tangent X which lives at
     * offset 12 as a (v + 1) * 0.5 float, and UVs are half floats.
     */
#pragma pack(push, 1)
    struct LandVertex {
        float posX {}; /**< 00: model-space X (quad-local, [-2048..0] or [0..2048]) */
        float posY {}; /**< 04: model-space Y */
        float posZ {}; /**< 08: model-space Z (the land height) */
        float tangentXEnc {}; /**< 12: (tangent.x + 1) * 0.5 */
        std::uint16_t u {}; /**< 16: half-float U */
        std::uint16_t v {}; /**< 18: half-float V */
        std::array<std::uint8_t, 3> normal {}; /**< 20: biased normal bytes */
        std::uint8_t tangentYEnc {}; /**< 23: (tangent.y + 1) * 127.5 */
        std::array<std::uint8_t, 3> bitangent {}; /**< 24: (bitangent + 1) * 127.5 bytes */
        std::uint8_t tangentZEnc {}; /**< 27: (tangent.z + 1) * 127.5 */
        std::array<std::uint8_t, 4> color {}; /**< 28: RGBA vertex color (A = 255) */
        std::array<std::uint8_t, K_BLEND_LAYERS> blend {}; /**< 32: texture blend weights, layer 0 = base */
        std::array<std::uint8_t, 2> pad {}; /**< 38: unused */
    };
#pragma pack(pop)
    static_assert(sizeof(LandVertex) == K_VERTEX_STRIDE);

    /**
     * @brief The engine's 16-byte shared index buffer wrapper (created by CreateIndexBuffer)
     */
    struct IndexBufferData {
        void* buffer; /**< 00: ID3D11Buffer* (R16_UINT) */
        volatile std::uint32_t refCount; /**< 08 */
        std::uint32_t pad0C; /**< 0C */
    };
    static_assert(sizeof(IndexBufferData) == K_INDEX_DATA_SIZE);

public:
    constexpr static float K_QUAD_WORLD_SIZE = 2048.0F; /**< World units per landscape quad side; the falloff's
                                                           distance unit (a cell is 2x2 quads) */
    static_assert(K_QUAD_WORLD_SIZE == K_QUAD_SIZE,
                  "the public quad size must match the mesh constants");

    constexpr static std::uint32_t K_QUADS_PER_CELL = 4; /**< Landscape quadrants per cell (0 = SW, 1 = SE,
                                                            2 = NW, 3 = NE) */
    static_assert(K_QUADS_PER_CELL == K_QUAD_COUNT,
                  "the public quadrant count must match the mesh constants");

    constexpr static std::uint32_t K_VANILLA_QUAD_DIM = 17; /**< Verts per side of a vanilla quadrant grid, and so
                                                               the dimension of LoadedLandData::heights[quad] */
    static_assert(K_VANILLA_QUAD_DIM == K_COARSE_DIM,
                  "the public vanilla grid dimension must match the mesh constants");

    /**
     * @brief The whole cell's land heights, indexed [y][x] with x/y growing east/north
     *
     * Assembled from the four overlapping quadrant grids so that everything derived from it -
     * mesh heights and collision heights alike - is continuous across the cell's interior quad
     * borders by construction (see buildCellHeightGrid).
     */
    using CellHeightGrid = std::array<std::array<float, K_CELL_DIM>, K_CELL_DIM>;

    /**
     * @brief One quadrant's smoothed heights on a subdivided grid
     *
     * Laid out exactly like the engine's LoadedLandData::heights[quad] it stands in for: row
     * major, row growing north and column growing east, heights relative to the same per-land
     * base. Only the dimension changes, so anything that reads the vanilla table with its own
     * dimension reads this one unchanged (see TerrainCollision).
     */
    struct QuadHeightField {
        std::vector<float> heights; /**< dim * dim samples, row major */
        std::uint32_t dim {}; /**< Verts per side */
        float minHeight {}; /**< Lowest sample, for the shape's bounds */
        float maxHeight {}; /**< Highest sample */
    };

    /**
     * @brief Verts per side of a quadrant grid at a subdivision level
     *
     * @param level Subdivision level (0 = the vanilla 17)
     * @return std::uint32_t Grid dimension
     */
    [[nodiscard]] static auto fineQuadDim(int level) -> std::uint32_t;

    /**
     * @brief Assembles the whole-cell height grid from the engine's raw quadrant height table
     *
     * @param quadHeights The engine's float[4][289] table (i.e. &LoadedLandData::heights[0][0])
     * @param grid Output grid
     */
    static void buildCellHeightGrid(const float* quadHeights,
                                    CellHeightGrid& grid);

    /**
     * @brief Samples one quadrant's smoothed surface onto a subdivided height grid
     *
     * The same Catmull-Rom surface buildSmoothedMesh draws, evaluated on a plain grid instead of
     * into vertex records - the collision counterpart of the render mesh. No edge pinning: a
     * height field is built at one level for every quad of every loaded cell, so both sides of
     * any shared border line sample the very same spline over the very same shared heights (a
     * cell border collapses to that border column, an interior quad border is inside the one
     * cell grid) and the surfaces meet exactly.
     *
     * @param grid The cell's height grid
     * @param quad Quadrant index 0-3
     * @param level Subdivision level (0 reproduces the vanilla heights bit-exactly)
     * @return QuadHeightField The sampled grid and its height range
     */
    [[nodiscard]] static auto buildQuadHeightField(const CellHeightGrid& grid,
                                                   std::uint32_t quad,
                                                   int level) -> QuadHeightField;

    /**
     * @brief Target mesh level of a quad's four border lines, named from the quad's point of
     * view (west = toward the quad at x - 1, which may be the other half of the same cell or
     * a quad of the neighboring cell)
     *
     * A border line renders at the minimum of the two adjacent quads' subdivision levels, so
     * the finer side always adapts to the coarser one (see buildSmoothedMesh). An entry equal
     * to the build level means the full spline (nothing to adapt to), 0 means the straight
     * vanilla segments.
     */
    struct EdgeLevels {
        std::uint8_t west {};
        std::uint8_t east {};
        std::uint8_t south {};
        std::uint8_t north {};

        [[nodiscard]] auto operator==(const EdgeLevels&) const -> bool = default;
    };

    /**
     * @brief One complete, installable set of land mesh buffers plus the shape fields that
     * describe it
     *
     * Everything setMesh writes into a BSTriShape and everything currentMesh reads back out
     * of one. Holding a MeshBuffers keeps nothing alive by itself - the renderer data must be
     * released through releaseMesh (or by the shape's own destructor while installed).
     */
    struct MeshBuffers {
        RE::BSGraphics::TriShape* rendererData {}; /**< The engine-allocated GPU buffer set */
        std::uint32_t vertexCount {};
        std::uint32_t triangleCount {};
        RE::NiPoint3 boundCenter; /**< Model bound over these buffers' vertices */
        float boundRadius {};
    };

    /**
     * @brief Everything a smoothed-mesh build needs, captured from a freshly built vanilla quad
     *
     * Immutable once created, so it can be handed to a worker thread while the engine goes on
     * to unload the cell or another mod touches the shape: buildSmoothedMesh reads only this.
     * Contents are private - outside of TerrainSubdivision a snapshot is an opaque token.
     */
    class QuadSnapshot {
    private:
        friend class TerrainSubdivision;

        std::array<LandVertex, K_COARSE_VERTS> coarse {}; /**< The vanilla vertex records */
        std::array<std::array<float, K_CELL_DIM>, K_CELL_DIM> heights {}; /**< Whole-cell height grid */
        std::uint32_t quad {}; /**< Quadrant index 0-3 */
        std::uint64_t vertexDesc {}; /**< The shape's vertex descriptor, revalidated at capture */
    };

    /**
     * @brief Checks that a shape is exactly the mesh the vanilla land builder makes
     *
     * Vanilla vertex/triangle counts, land vertex stride, CPU-side vertex copy present;
     * anything else means another mod got there first and leaving the shape alone is the safe
     * outcome. Validation is separate from snapshotQuad so the build hook can vet a quad
     * without copying anything - the engine calls the builder 16 times per cell attach, and
     * cell attach is exactly the moment the frame has no headroom to spare.
     *
     * Thread-safe; called from whichever thread the engine builds land on.
     *
     * @param shape The freshly built quad shape (only read)
     * @return std::optional<std::uint64_t> The shape's vertex descriptor (needed later by
     *         snapshotQuad), or nullopt when the layout is not the vanilla one
     */
    [[nodiscard]] static auto validateQuadLayout(RE::BSTriShape& shape) -> std::optional<std::uint64_t>;

    /**
     * @brief Captures a validated quad's data for a smoothed-mesh build
     *
     * Pure capture, no validation (see validateQuadLayout): copies the vanilla vertex records
     * and assembles the whole-cell height grid. Deferred until a build is actually wanted so
     * that quads which never smooth - most of the loaded grid, plus the 15 dead builder
     * generations per cell attach - never pay for the copies.
     *
     * @param data Loaded land data of the cell (the height tables; caller guarantees liveness)
     * @param rawVertexData The vanilla CPU vertex copy validateQuadLayout confirmed present
     * @param quad Quadrant index 0-3 (0 = SW, 1 = SE, 2 = NW, 3 = NE)
     * @param vertexDesc The descriptor validateQuadLayout returned for this shape
     * @return std::shared_ptr<const QuadSnapshot> The captured data
     */
    [[nodiscard]] static auto snapshotQuad(const RE::TESObjectLAND::LoadedLandData& data,
                                           const void* rawVertexData,
                                           std::uint32_t quad,
                                           std::uint64_t vertexDesc) -> std::shared_ptr<const QuadSnapshot>;

    /**
     * @brief Builds ready-to-install smoothed buffers from a snapshot
     *
     * Callable from any thread: interpolates the fine vertex grid, then creates the GPU
     * buffers through the engine's own creators (the same ones the vanilla land builder uses
     * off-thread, so off-main creation is proven safe).
     * Nothing is installed anywhere - the caller owns the result and must either setMesh it
     * or releaseMesh it.
     *
     * Vertices on one of this quad's four border lines whose EdgeLevels entry is below the
     * build level are pinned onto the coarser neighbor's edge polyline: the neighbor's own
     * vertices along the shared line are spline samples at its coarser step (or the original
     * LAND verts at level 0), and this mesh's extra vertices in between land exactly on the
     * straight segments the neighbor renders between them. This works the same whether the
     * line is a cell border or the cell's interior cross line between two quads of one cell.
     * Vertices at positions both sides share evaluate the very same interpolation over the
     * very same shared height samples (the whole-cell grid inside a cell, the identical
     * border column across cells), so they match bit for bit and the meshes meet without
     * cracks; only the segments between them flatten, and the rest of the quad keeps its
     * full smoothing. Edges between equal-level quads need no pinning.
     *
     * @param snapshot A quad captured by snapshotQuad
     * @param level Subdivision level 1-3
     * @param edgeLevels Per-edge render level of this quad's four border lines, each 0..level
     * @return std::optional<MeshBuffers> The new buffers, or nullopt when the renderer refused
     *         them (out of GPU memory) or is unavailable
     */
    [[nodiscard]] static auto buildSmoothedMesh(const QuadSnapshot& snapshot,
                                                int level,
                                                const EdgeLevels& edgeLevels) -> std::optional<MeshBuffers>;

    /**
     * @brief Reads the buffer set a shape currently renders with
     *
     * Main thread only (reads the live shape fields).
     */
    [[nodiscard]] static auto currentMesh(RE::BSTriShape& shape) -> MeshBuffers;

    /**
     * @brief Points a shape at a different buffer set
     *
     * Writes the renderer data pointer, the vertex/triangle counts and the model bound -
     * nothing else, so materials, transforms and the scene graph stay untouched. Ownership of
     * the previous buffers stays with the caller (read them out with currentMesh first).
     *
     * Main thread only: the render loop submits draws on the main thread, so swapping there
     * cannot race a draw that is half way through reading the shape.
     */
    static void setMesh(RE::BSTriShape& shape,
                        const MeshBuffers& mesh);

    /**
     * @brief Releases a buffer set that is not installed on any shape
     *
     * Same engine path BSTriShape::~BSTriShape uses; never call this for buffers a live shape
     * still points at (the shape's destructor releases those).
     */
    static void releaseMesh(const MeshBuffers& mesh);

private:
    /**
     * @brief Call-site hook around the engine's BuildQuadTriShape
     */
    struct BuildQuadHook {
        static auto thunk(RE::TESObjectLAND::LoadedLandData* data,
                          std::uint32_t quad) -> RE::BSTriShape*;
        static inline REL::Relocation<decltype(thunk)> s_func; /**< The unpatched builder entry */
    };

    static inline std::mutex s_indexBufferMutex; /**< Guards the index caches (land builds can run off-thread) */
    static inline std::array<IndexBufferData*, ConfigLoader::MAX_SUBDIVISIONS + 1>
        s_indexBuffers {}; /**< One engine-shared index buffer per subdivision level (slot 0 unused) */
    static inline std::array<std::vector<std::uint16_t>, ConfigLoader::MAX_SUBDIVISIONS + 1>
        s_indexData {}; /**< CPU copy of each level's index list; kept for the engine's decal builder,
                             which reads triangles from CPU memory (see DecalFix). Filled once per
                             level and never resized afterwards, so the data pointer stays valid. */

public:
    TerrainSubdivision() = delete;

private:
    /**
     * @brief Assembles the full 33x33 cell height grid from the four 17x17 quadrant grids
     *
     * Working on the whole cell (instead of one quadrant) makes the Catmull-Rom result
     * identical for the shared rows/columns of neighboring quadrants, so quad seams line up
     * by construction.
     *
     * @param data Loaded land data holding heights[4][289]
     * @param grid Output grid, indexed [y][x] with x/y growing east/north
     */
    static void buildCellHeightGrid(const RE::TESObjectLAND::LoadedLandData& data,
                                    std::array<std::array<float,
                                                          K_CELL_DIM>,
                                               K_CELL_DIM>& grid);

    /**
     * @brief Samples the cell height grid with mirror extrapolation outside the cell
     *
     * Verts on a shared cell border only ever sample their own border line (see
     * sampleHeight), so the extrapolated samples never cause cross-cell seams.
     *
     * @param grid The 33x33 cell height grid
     * @param x Grid X, may be outside [0, 32]
     * @param y Grid Y, may be outside [0, 32]
     * @return float The (possibly extrapolated) height
     */
    static auto gridHeight(const std::array<std::array<float,
                                                       K_CELL_DIM>,
                                            K_CELL_DIM>& grid,
                           int x,
                           int y) -> float;

    /**
     * @brief Height at a fractional cell-grid position along the Catmull-Rom spline
     *
     * Separable Catmull-Rom collapses to the exact grid value at integer positions and to
     * line-only samples on grid lines, which keeps original verts fixed and makes both quad
     * and cell borders seam-free (both sides of a border interpolate from the same shared
     * line of samples).
     *
     * @param grid The 33x33 cell height grid
     * @param cellX Integer grid X of the containing coarse quad (0-32)
     * @param cellY Integer grid Y of the containing coarse quad (0-32)
     * @param fracX Fractional position within the coarse quad [0, 1)
     * @param fracY Fractional position within the coarse quad [0, 1)
     * @param maxRise World units the height may rise above the highest of the four surrounding
     *        verts; split evenly between the two interpolation passes (dips are never bounded)
     * @return float The interpolated height
     */
    static auto sampleHeight(const std::array<std::array<float,
                                                         K_CELL_DIM>,
                                              K_CELL_DIM>& grid,
                             int cellX,
                             int cellY,
                             float fracX,
                             float fracY,
                             float maxRise) -> float;

    /**
     * @brief 1D Catmull-Rom through p1 (t=0) and p2 (t=1)
     *
     * @param rise Passed to limitedTangent; the segment cannot exceed max(p1, p2) by more
     */
    static auto catmullRom(float p0,
                           float p1,
                           float p2,
                           float p3,
                           float t,
                           float rise) -> float;

    /**
     * @brief The spline tangent at one knot, limited against upward overshoot
     *
     * Raw Catmull-Rom uses the central difference of the knot's neighbors, which lets the curve
     * swing past the knots wherever the sampled heights turn or step sharply. Swinging upward is
     * the only way a subdivided vertex ends up higher than every original vertex around it, and
     * so the only way this plugin can push terrain through a static mesh the vanilla surface
     * cleared - hidden garbage heights under a mesh that covers the landscape make a flat span
     * bounded by steep drops, which is the worst case for it.
     *
     * The limit is therefore one sided: it is Fritsch-Carlson's bound applied separately to each
     * sign of the tangent, since a positive tangent can only lift the curve ahead of the knot
     * and a negative one only behind it. Downward swing is left unbounded, so dips keep their
     * full curvature - only rising into occupied space is a problem. Both bounds depend on this
     * knot alone, so the two segments meeting here still share one tangent and the curve stays
     * C1 at the original verts, and on terrain that varies smoothly they are slack enough that
     * the tangent comes back unchanged.
     *
     * The allowance is in world units rather than a fraction of the local relief, which is what
     * keeps it useful across the whole landscape: the height a static clears the terrain by does
     * not scale with how dramatic that terrain is, so neither should the budget. A gentle crest
     * asks for only a few units of tangent and passes through untouched, while a cliff edge -
     * the shape that actually pokes - is still cut back hard.
     *
     * @param prev Height at the previous knot
     * @param knot Height at this knot
     * @param next Height at the next knot
     * @param rise World units the curve may rise past the knots of either adjoining segment;
     *        0 pins it to them exactly
     * @return float The tangent to use at this knot
     */
    static auto limitedTangent(float prev,
                               float knot,
                               float next,
                               float rise) -> float;

    /**
     * @brief Linear interpolation
     */
    static auto lerp(float valA,
                     float valB,
                     float t) -> float;

    /**
     * @brief Bilinear interpolation of four corner values
     */
    static auto bilerp(float v00,
                       float v10,
                       float v01,
                       float v11,
                       float fracX,
                       float fracY) -> float;

    /**
     * @brief Interpolates a full vertex record bilinearly between four coarse records
     *
     * Unit-length attributes (normal, tangent, bitangent) are decoded, lerped, renormalized
     * and re-encoded; everything else is lerped in its storage encoding. Position and height
     * are overwritten by the caller afterwards.
     *
     * @param c00 Record at (x0, y0)
     * @param c10 Record at (x1, y0)
     * @param c01 Record at (x0, y1)
     * @param c11 Record at (x1, y1)
     * @param fracX Weight toward c10/c11
     * @param fracY Weight toward c01/c11
     * @return LandVertex The interpolated record
     */
    static auto lerpVertex(const LandVertex& c00,
                           const LandVertex& c10,
                           const LandVertex& c01,
                           const LandVertex& c11,
                           float fracX,
                           float fracY) -> LandVertex;

    /**
     * @brief Returns (creating on first use) the engine index buffer for a subdivision level
     *
     * Mirrors the vanilla builder, which shares one static index buffer between every land
     * quad in the game: the triangulation only depends on the grid dimension, so one buffer
     * per level serves all quads. Never freed; each shape AddRefs the underlying D3D buffer.
     * The CPU copy the buffer was built from is kept in s_indexData.
     *
     * @param level Subdivision level 1-3
     * @return IndexBufferData* The shared wrapper, or nullptr when creation failed
     */
    static auto getIndexBuffer(int level) -> IndexBufferData*;

    /**
     * @brief Builds the index list of one subdivision level
     *
     * Same triangulation as the vanilla 17x17 builder, generalized to the finer grid: a
     * checkerboard of flipped diagonals with identical winding.
     *
     * @param level Subdivision level 1-3
     * @return std::vector<std::uint16_t> Three indices per triangle, in row major order
     */
    [[nodiscard]] static auto buildIndices(int level) -> std::vector<std::uint16_t>;

    /**
     * @brief Releases a replaced renderer data through the engine's geometry buffer manager
     *
     * Same call BSTriShape::~BSTriShape makes, keeping allocation and release symmetric
     * with the engine.
     */
    static void releaseRendererData(RE::BSGraphics::TriShape* rendererData);

    /**
     * @brief IEEE 754 binary16 -> binary32
     */
    static auto halfToFloat(std::uint16_t half) -> float;

    /**
     * @brief IEEE 754 binary32 -> binary16 (round to nearest even, matching the engine)
     */
    static auto floatToHalf(float value) -> std::uint16_t;

    /**
     * @brief Encodes a [-1, 1] component into the engine's unsigned byte encoding
     */
    static auto encodeUnitByte(float value) -> std::uint8_t;

    /**
     * @brief Decodes the engine's unsigned byte encoding back to [-1, 1]
     */
    static auto decodeUnitByte(std::uint8_t value) -> float;

    /**
     * @brief Rounds an interpolated byte attribute (colors, blend weights) back to storage
     */
    static auto roundByte(float value) -> std::uint8_t;
};

} // namespace SmoothTerrain
