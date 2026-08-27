#include "replay_draw_validation.hpp"

#include "../gx/pipeline.hpp"
#include "../internal.hpp"

#include <cstring>

namespace aurora::gfx::replay_draw_validation {
static Module Log("aurora::gfx::replay_draw_validation");

namespace {
constexpr uint32_t RmlGeometry = 0;
constexpr uint32_t RmlFullscreen = 1;
constexpr uint32_t RmlDynamicGroup1 = 1u << 1u;
constexpr uint32_t RmlDynamicGroup2 = 1u << 2u;
constexpr uint32_t RmlDynamicGroups = RmlDynamicGroup1 | RmlDynamicGroup2;

void validate_window(const char* name, const BufferWindow& window) {
  CHECK(window.replayPrefix <= window.highWater,
        "Replay {} prefix {} exceeds the encode operation's high-water mark {}", name, window.replayPrefix,
        window.highWater);
  CHECK(window.highWater <= window.capacity, "Replay {} high-water mark {} exceeds buffer capacity {}", name,
        window.highWater, window.capacity);
}

void validate_bounds(const FrameBounds& bounds) {
  validate_window("vertex", bounds.vertices);
  validate_window("uniform", bounds.uniforms);
  validate_window("index", bounds.indices);
  validate_window("storage", bounds.storage);
  CHECK(bounds.uniformOffsetAlignment != 0,
        "Replay uniform dynamic-offset alignment is zero; device limits were not captured");
  CHECK(bounds.persistentStorageEnd >= StorageBufferSize &&
            bounds.persistentStorageEnd <= StorageBufferSize + PersistentStorageSize,
        "Replay persistent-storage end {} is outside arena [{}, {}]", bounds.persistentStorageEnd, StorageBufferSize,
        StorageBufferSize + PersistentStorageSize);
}

void validate_range(const char* drawKind, const char* rangeName, const Range& range, const BufferWindow& window,
                    uint32_t alignment) {
  CHECK(alignment != 0, "Replay {} {} alignment is zero", drawKind, rangeName);
  CHECK(range.offset % alignment == 0, "Replay {} {} offset {} is not aligned to {} bytes", drawKind, rangeName,
        range.offset, alignment);
  CHECK(range.offset <= window.highWater && range.size <= window.highWater - range.offset,
        "Replay {} {} range [{}..{}) exceeds encode-operation high-water mark {} (replay prefix {})", drawKind,
        rangeName, range.offset, static_cast<uint64_t>(range.offset) + range.size, window.highWater,
        window.replayPrefix);
  CHECK(range.offset <= window.capacity && range.size <= window.capacity - range.offset,
        "Replay {} {} range [{}..{}) exceeds buffer capacity {}", drawKind, rangeName, range.offset,
        static_cast<uint64_t>(range.offset) + range.size, window.capacity);
}

void validate_relative_span(const char* name, uint32_t offset, uint64_t size, uint32_t uniformSize) {
  CHECK(offset <= uniformSize && size <= static_cast<uint64_t>(uniformSize - offset),
        "Replay GX {} span [{}..{}) exceeds draw uniform size {}", name, offset, static_cast<uint64_t>(offset) + size,
        uniformSize);
}

void validate_indexed_array_range(uint32_t attribute, uint32_t shaderOffset, bool used, const Range& retainedRange,
                                  const FrameBounds& bounds) {
  if (!used) {
    CHECK(shaderOffset == 0, "Replay GX unused indexed attribute {} has shader-visible storage offset {}", attribute,
          shaderOffset);
    CHECK(retainedRange.offset == 0 && retainedRange.size == 0,
          "Replay GX unused indexed attribute {} retains storage range [{}..{})", attribute, retainedRange.offset,
          static_cast<uint64_t>(retainedRange.offset) + retainedRange.size);
    return;
  }

  CHECK(retainedRange.size != 0, "Replay GX indexed attribute {} has an empty retained storage range", attribute);
  CHECK(shaderOffset == retainedRange.offset,
        "Replay GX indexed attribute {} shader-visible storage offset {} disagrees with retained range start {}",
        attribute, shaderOffset, retainedRange.offset);

  if (retainedRange.offset < StorageBufferSize) {
    CHECK(retainedRange.offset <= bounds.storage.highWater &&
              retainedRange.size <= bounds.storage.highWater - retainedRange.offset,
          "Replay GX indexed attribute {} per-frame storage range [{}..{}) exceeds encode-operation high-water mark {}",
          attribute, retainedRange.offset, static_cast<uint64_t>(retainedRange.offset) + retainedRange.size,
          bounds.storage.highWater);
    CHECK(retainedRange.offset <= bounds.storage.capacity &&
              retainedRange.size <= bounds.storage.capacity - retainedRange.offset,
          "Replay GX indexed attribute {} per-frame storage range [{}..{}) exceeds buffer capacity {}", attribute,
          retainedRange.offset, static_cast<uint64_t>(retainedRange.offset) + retainedRange.size,
          bounds.storage.capacity);
    return;
  }

  CHECK(retainedRange.offset < bounds.persistentStorageEnd &&
            retainedRange.size <= bounds.persistentStorageEnd - retainedRange.offset,
        "Replay GX indexed attribute {} persistent storage range [{}..{}) exceeds arena end {}", attribute,
        retainedRange.offset, static_cast<uint64_t>(retainedRange.offset) + retainedRange.size,
        bounds.persistentStorageEnd);
}
} // namespace

void validate_gx(const gx::DrawData& draw, std::span<const uint8_t> uniformBytes, const FrameBounds& bounds) {
  validate_bounds(bounds);
  validate_range("GX", "vertex", draw.vertRange, bounds.vertices, 4);
  validate_range("GX", "index", draw.idxRange, bounds.indices, 4);
  validate_range("GX", "uniform", draw.uniformRange, bounds.uniforms, bounds.uniformOffsetAlignment);

  const uint64_t expectedVertexBytes = static_cast<uint64_t>(draw.vtxCount) * draw.vtxStride;
  CHECK(expectedVertexBytes == draw.vertRange.size,
        "Replay GX vertex count/stride require {} bytes, but DrawData names {}", expectedVertexBytes,
        draw.vertRange.size);
  const uint64_t expectedIndexBytes = static_cast<uint64_t>(draw.indexCount) * sizeof(uint16_t);
  CHECK(expectedIndexBytes == draw.idxRange.size, "Replay GX index count requires {} bytes, but DrawData names {}",
        expectedIndexBytes, draw.idxRange.size);
  CHECK(draw.uniformRange.size > 0 && draw.uniformRange.size <= gx::MaxUniformSize,
        "Replay GX uniform size {} is outside (0, {}]", draw.uniformRange.size, gx::MaxUniformSize);
  CHECK(draw.uniformRange.offset <= UniformBufferSize &&
            gx::MaxUniformSize <= UniformBufferSize - draw.uniformRange.offset,
        "Replay GX dynamic uniform binding [{}..{}) exceeds uniform-buffer capacity {}", draw.uniformRange.offset,
        static_cast<uint64_t>(draw.uniformRange.offset) + gx::MaxUniformSize, UniformBufferSize);
  CHECK(uniformBytes.size() >= bounds.uniforms.highWater,
        "Replay GX CPU uniform snapshot has {} bytes, short of encode-operation high-water mark {}",
        uniformBytes.size(), bounds.uniforms.highWater);

  constexpr uint32_t MatrixBytes = sizeof(Mat3x4<float>);
  constexpr uint32_t PositionMatrixBytes = gx::MaxPnMtx * MatrixBytes;
  constexpr uint32_t MatrixBlockBeforeNormals = (gx::MaxPnMtx + gx::MaxTexMtx) * MatrixBytes;
  constexpr uint32_t ArrayOffsetBytes = gx::MaxIndexAttr * sizeof(uint32_t);
  validate_relative_span("indexed-array offsets", draw.posArrayUniformOffset, ArrayOffsetBytes, draw.uniformRange.size);
  validate_relative_span("position matrices", draw.mtxPosOffset, PositionMatrixBytes, draw.uniformRange.size);
  validate_relative_span("normal matrices", draw.mtxNrmOffset, PositionMatrixBytes, draw.uniformRange.size);
  CHECK(draw.mtxNrmOffset == draw.mtxPosOffset + MatrixBlockBeforeNormals,
        "Replay GX normal-matrix offset {} does not follow position/texture matrix block ending at {}",
        draw.mtxNrmOffset, draw.mtxPosOffset + MatrixBlockBeforeNormals);

  constexpr uint32_t TextureMatrixBits = ((1u << (gx::MaxPnMtx + gx::MaxTexMtx)) - 1u) & ~((1u << gx::MaxPnMtx) - 1u);
  CHECK((draw.texMtxCamMask & ~TextureMatrixBits) == 0,
        "Replay GX camera texture-matrix mask {:#x} references slots outside [{}, {})", draw.texMtxCamMask,
        gx::MaxPnMtx, gx::MaxPnMtx + gx::MaxTexMtx);
  CHECK(draw.texMtxCamMask == 0 || draw.pnMtxSlot < gx::MaxPnMtx,
        "Replay GX camera texture matrices reference position-matrix slot {} outside [0, {})", draw.pnMtxSlot,
        gx::MaxPnMtx);

  const uint8_t* arrayOffsets = uniformBytes.data() + draw.uniformRange.offset + draw.posArrayUniformOffset;
  constexpr uint32_t IndexedArrayBits = (1u << gx::MaxIndexAttr) - 1u;
  CHECK((draw.indexedArrayUsedMask & ~IndexedArrayBits) == 0,
        "Replay GX indexed-array used mask {:#x} has bits outside [0, {})", draw.indexedArrayUsedMask,
        gx::MaxIndexAttr);
  for (uint32_t attribute = 0; attribute < gx::MaxIndexAttr; ++attribute) {
    uint32_t storageOffset = 0;
    std::memcpy(&storageOffset, arrayOffsets + attribute * sizeof(storageOffset), sizeof(storageOffset));
    validate_indexed_array_range(attribute, storageOffset, (draw.indexedArrayUsedMask & (1u << attribute)) != 0,
                                 draw.indexedArrayRanges[attribute], bounds);
  }
}

void validate_rml(const RmlDrawReferences& draw, const FrameBounds& bounds) {
  validate_bounds(bounds);
  CHECK((draw.dynamicBindGroupMask & ~RmlDynamicGroups) == 0,
        "Replay Rml DrawData has unknown dynamic bind-group mask bits {:#x}",
        draw.dynamicBindGroupMask & ~RmlDynamicGroups);

  const auto validate_dynamic_offset = [&](const char* groupName, uint32_t groupBit, uint64_t bindGroup,
                                           uint32_t offset, uint32_t extent) {
    if ((draw.dynamicBindGroupMask & groupBit) == 0) {
      CHECK(extent == 0, "Replay Rml {} has dynamic extent {} without a dynamic-group bit", groupName, extent);
      return;
    }
    CHECK(bindGroup != 0, "Replay Rml {} dynamic-group bit has no bind group", groupName);
    CHECK(extent != 0, "Replay Rml {} dynamic binding has an empty byte extent", groupName);
    validate_range("Rml", groupName, {offset, extent}, bounds.uniforms, bounds.uniformOffsetAlignment);
  };

  validate_dynamic_offset("dynamic group 1", RmlDynamicGroup1, draw.bindGroup1, draw.bindGroup1DynamicOffset,
                          draw.bindGroup1DynamicExtent);
  validate_dynamic_offset("dynamic group 2", RmlDynamicGroup2, draw.bindGroup2, draw.bindGroup2DynamicOffset,
                          draw.bindGroup2DynamicExtent);

  switch (draw.drawKind) {
  case RmlGeometry: {
    validate_range("Rml geometry", "vertex", draw.vertexRange, bounds.vertices, 4);
    validate_range("Rml geometry", "index", draw.indexRange, bounds.indices, 4);
    validate_range("Rml geometry", "uniform", draw.uniformRange, bounds.uniforms, bounds.uniformOffsetAlignment);
    CHECK(draw.vertexStride != 0 && draw.vertexRange.size % draw.vertexStride == 0,
          "Replay Rml vertex range size {} is not a multiple of stride {}", draw.vertexRange.size, draw.vertexStride);
    const uint64_t expectedVertexBytes = static_cast<uint64_t>(draw.vertexCount) * draw.vertexStride;
    CHECK(expectedVertexBytes == draw.vertexRange.size,
          "Replay Rml vertex count requires {} bytes, but DrawData names {}", expectedVertexBytes,
          draw.vertexRange.size);
    const uint64_t expectedIndexBytes = static_cast<uint64_t>(draw.indexCount) * sizeof(uint32_t);
    CHECK(expectedIndexBytes == draw.indexRange.size, "Replay Rml index count requires {} bytes, but DrawData names {}",
          expectedIndexBytes, draw.indexRange.size);
    CHECK(draw.requiredUniformSize != 0 && draw.uniformRange.size == draw.requiredUniformSize,
          "Replay Rml geometry uniform size {} does not match required size {}", draw.uniformRange.size,
          draw.requiredUniformSize);
    break;
  }
  case RmlFullscreen:
    CHECK(draw.vertexRange.size == 0 && draw.indexRange.size == 0 && draw.indexCount == 0 && draw.vertexCount > 0,
          "Replay Rml fullscreen draw has vertex bytes {}, index bytes {}, index count {}, vertex count {}",
          draw.vertexRange.size, draw.indexRange.size, draw.indexCount, draw.vertexCount);
    break;
  default:
    FATAL("Replay Rml DrawData has unknown draw kind {}", draw.drawKind);
  }
}

} // namespace aurora::gfx::replay_draw_validation
