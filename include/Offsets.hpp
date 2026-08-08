#pragma once

#include "PCH.h"

#include <cstdint>

namespace SmoothTerrain::Offsets {

/**
 * @brief Whether the running game's landscape offsets are known to this plugin
 *
 * @return bool True on SE (1.5.x), AE (1.6.x), and VR (1.4.15), the flavors whose IDs are verified
 */
[[nodiscard]] inline auto isRuntimeSupported() -> bool
{
    return REL::Module::IsSE() || REL::Module::IsAE() || REL::Module::IsVR();
}

/**
 * @brief BSTriShape* BuildQuadTriShape(TESObjectLAND::LoadedLandData* data, uint32_t quad)
 *
 * Builds one cell-quadrant landscape mesh: 289 vertices (17x17 grid, 128 unit spacing),
 * 512 triangles, named "Block (x, y)". Reads heights / normals / colors / texture blend
 * percents out of LoadedLandData and produces the final GPU vertex + index buffers.
 * 1.5.97: 0x25CBA0. 1.6.1170: 0x2AF3A0. VR 1.4.15: 0x26E050.
 */
constexpr REL::RelocationID K_BUILD_QUAD_TRISHAPE(18380,
                                                  18806);

/**
 * @brief Land geometry init: gets all four quads built and swapped into LoadedLandData::geom
 * on cell attach, then sets up multibounds / materials / collision.
 *
 * AE calls BuildQuadTriShape from here directly (its second call site); SE and VR have no
 * direct call and reach the builder through K_BUILD_LAND_QUADS instead, so there is nothing
 * to patch here on those flavors. 1.5.97: 0x2567F0. 1.6.1170: 0x2A8B00. VR 1.4.15: 0x267BA0.
 */
constexpr REL::RelocationID K_BUILD_LAND_GEOMETRY(18334,
                                                  18750);

/**
 * @brief Builds all four land quads and swaps them into LoadedLandData::geom (refcounting the
 * shapes, no collision). Holds a call site to BuildQuadTriShape on all three flavors - the
 * only one on SE and VR, where K_BUILD_LAND_GEOMETRY calls this instead of the builder.
 * 1.5.97: 0x25B710. 1.6.1170: 0x2ADBC0. VR 1.4.15: 0x26CB50.
 */
constexpr REL::RelocationID K_BUILD_LAND_QUADS(18369,
                                               18792);

/**
 * @brief bool CellMopp::BuildLandCollision(CellMopp* mopp, const LandCollisionDesc* desc)
 *
 * Builds a cell's landscape collision: one bhkTriSampledHeightFieldBvTreeShape per quadrant over
 * the LAND height grid, each wrapped in a bhkRigidBody queued into the cell's havok world. Called
 * once per land load, from the tail of K_BUILD_LAND_GEOMETRY - its only call site in the whole
 * executable on all three flavors, so the scan for it can overshoot freely.
 * 1.5.97: 0x39B080. 1.6.1170: 0x3F3570. VR 1.4.15: 0x3AA900.
 */
constexpr REL::RelocationID K_BUILD_LAND_COLLISION(25262,
                                                   25784);

/**
 * @brief bhkTriSampledHeightFieldBvTreeShape::InitFromCInfo(bhkShape* shape, HeightFieldCInfo* cinfo)
 *
 * Turns one quadrant's height grid into the havok shape the rigid body collides against: it fills
 * a stack hkBSHeightFieldShape from the cinfo, samples that into a heap
 * hkpCompressedSampledHeightFieldShape (quantized copy - the engine keeps no pointer into the
 * cinfo's height array), wraps it in a hkpTriSampledHeightFieldCollection plus its BV tree and
 * hands the result to the bhk wrapper. Called only by K_BUILD_LAND_COLLISION, once per quadrant,
 * and that is the function's only call site in the whole executable (it is also vfunc 46 of the
 * shape, which nothing reaches).
 * 1.5.97: 0xDEC000. 1.6.1170: 0xECDAB0. VR 1.4.15: 0xE40FD0.
 */
constexpr REL::RelocationID K_INIT_HEIGHT_FIELD_SHAPE(77186,
                                                      79074);

/**
 * @brief BSGraphics::TriShape* CreateTriShapeRendererData(Renderer*, const void* vertexData,
 *          uint32_t sizeBytes, uint64_t vertexDesc, IndexBufferData* sharedIndexBuffer)
 *
 * Engine-allocates the 48-byte renderer data: creates the D3D11 vertex buffer, retains a
 * CPU copy of the vertex data (rawVertexData), stores the vertex descriptor, and AddRefs
 * the passed index buffer's D3D11 buffer into slot +8. The input vertex data is copied,
 * so the caller keeps ownership of its buffer. 1.5.97: 0xD6BD10. 1.6.1170: 0xE46170.
 * VR 1.4.15: 0xDBDA30.
 */
constexpr REL::RelocationID K_CREATE_TRISHAPE_DATA(75475,
                                                   77261);

/**
 * @brief Signature of the function K_CREATE_TRISHAPE_DATA points at
 */
using CreateTriShapeData_t = RE::BSGraphics::TriShape*(RE::BSGraphics::Renderer* renderer,
                                                       const void* vertexData,
                                                       std::uint32_t sizeBytes,
                                                       std::uint64_t vertexDesc,
                                                       void* indexBuffer);

/**
 * @brief IndexBufferData* CreateIndexBuffer(Renderer*, uint32_t indexCount, const uint16_t* indices)
 *
 * Engine-allocates a 16-byte {ID3D11Buffer*, refCount=1} wrapper holding an R16_UINT index
 * buffer of indexCount indices (data copied at creation). The vanilla 17x17 land grid keeps
 * one of these in a global and shares it between every land quad in the game; this plugin
 * does the same per subdivision level. 1.5.97: 0xD6DA10. 1.6.1170: 0xE47FE0.
 * VR 1.4.15: 0xDBF800.
 */
constexpr REL::RelocationID K_CREATE_INDEX_BUFFER(75503,
                                                  77294);

/**
 * @brief Signature of the function K_CREATE_INDEX_BUFFER points at
 */
using CreateIndexBuffer_t = void*(RE::BSGraphics::Renderer* renderer,
                                  std::uint32_t indexCount,
                                  const std::uint16_t* indices);

/**
 * @brief const uint16_t* GetLandIndexList()
 *
 * Returns the engine's one static landscape index list: 1536 indices (512 triangles) laid out
 * for the vanilla 17x17 grid, shared by every land quad in the game. Land shapes keep no CPU
 * side index data of their own, so the decal builder (K_DECAL_COLLECT_TRIANGLES) calls this
 * whenever it has to walk the triangles of a landscape material. All three flavors are the same
 * two instructions (load the global, return) with exactly one caller in the whole executable.
 * 1.5.97: 0x25A7C0. 1.6.1170: 0x2ACAC0. VR 1.4.15: 0x26BB70.
 */
constexpr REL::RelocationID K_GET_LAND_INDEX_LIST(18359,
                                                  18780);

/**
 * @brief Signature of the function K_GET_LAND_INDEX_LIST points at
 */
using GetLandIndexList_t = const std::uint16_t*();

/**
 * @brief BSTempEffectSimpleDecal triangle collection: clips the triangles of the geometry a
 * decal was placed on against the decal's box and accumulates the surviving pieces.
 *
 * Reads the vertex positions from the shape's rendererData->rawVertexData and the triangles
 * from either the shape's own CPU index data or, for landscape materials, from
 * K_GET_LAND_INDEX_LIST - always looping over the shape's own triangleCount.
 * 1.5.97: 0x44F6F0. 1.6.1170: 0x4ABD70. VR 1.4.15: 0x45F2D0.
 */
constexpr REL::RelocationID K_DECAL_COLLECT_TRIANGLES(29259,
                                                      30114);

/**
 * @brief Signature of the function K_DECAL_COLLECT_TRIANGLES points at
 */
using DecalCollectTriangles_t = std::uintptr_t(RE::BSTempEffectSimpleDecal* decal,
                                               RE::BSTriShape* shape);

/**
 * @brief BSTempEffectSimpleDecal::Initialize's implementation, which builds the decal's
 * planes and then calls K_DECAL_COLLECT_TRIANGLES for the geometry the decal sits on.
 * 1.5.97: 0x44DBF0. 1.6.1170: 0x4A9C70. VR 1.4.15: 0x45D7B0.
 */
constexpr REL::RelocationID K_DECAL_INITIALIZE(29251,
                                               30106);

/**
 * @brief Small helper that resolves an NiAVObject's geometry and tail jumps into
 * K_DECAL_COLLECT_TRIANGLES; the collector's only other entry point.
 * 1.5.97: 0x44F6B0. 1.6.1170: 0x4ABD30. VR 1.4.15: 0x45F290.
 */
constexpr REL::RelocationID K_DECAL_COLLECT_FROM_OBJECT(29258,
                                                        30113);

/**
 * @brief Global pointer to the geometry buffer manager
 *
 * BSTriShape::~BSTriShape releases its rendererData through this manager's vfunc 5 (vtable
 * offset 0x28): manager->ReleaseRendererData(data). Calling the same path releases a
 * renderer data we replace, keeping alloc/free symmetric with the engine.
 * 1.5.97: 0x30136C0. 1.6.1170: 0x3274040. VR 1.4.15: 0x316BEC0.
 */
constexpr REL::RelocationID K_GEOMETRY_BUFFER_MANAGER(523950,
                                                      410530);

/**
 * @brief Signature of vfunc 5 (vtable offset 0x28) of the manager K_GEOMETRY_BUFFER_MANAGER points at
 */
using ReleaseRendererData_t = void (*)(void* manager,
                                       RE::BSGraphics::TriShape* rendererData);

} // namespace SmoothTerrain::Offsets
