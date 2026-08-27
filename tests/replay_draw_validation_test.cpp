#include "../lib/gfx/replay_draw_validation.hpp"
#include "../lib/gx/pipeline.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace {
namespace validation = aurora::gfx::replay_draw_validation;

struct ValidGxFixture {
  std::vector<uint8_t> uniforms = std::vector<uint8_t>(1856);
  validation::FrameBounds bounds{
      .vertices = {20, 20, static_cast<uint32_t>(aurora::gfx::VertexBufferSize)},
      .uniforms = {1856, 0, static_cast<uint32_t>(aurora::gfx::UniformBufferSize)},
      .indices = {10, 10, static_cast<uint32_t>(aurora::gfx::IndexBufferSize)},
      .storage = {0, 0, static_cast<uint32_t>(aurora::gfx::StorageBufferSize)},
      .persistentStorageEnd = static_cast<uint32_t>(aurora::gfx::StorageBufferSize) + 16,
      .uniformOffsetAlignment = 256,
  };
  aurora::gx::DrawData draw{
      .vertRange = {4, 16},
      .idxRange = {4, 6},
      .uniformRange = {256, 1600},
      .vtxCount = 1,
      .indexCount = 3,
      .instanceCount = 1,
      .posArrayUniformOffset = 32,
      .mtxPosOffset = 128,
      .mtxNrmOffset = 1088,
      .vtxStride = 16,
      .indexedArrayUsedMask = 1,
  };

  ValidGxFixture() {
    const uint32_t persistentOffset = static_cast<uint32_t>(aurora::gfx::StorageBufferSize);
    std::memcpy(uniforms.data() + draw.uniformRange.offset + draw.posArrayUniformOffset, &persistentOffset,
                sizeof(persistentOffset));
    draw.indexedArrayRanges[0] = {persistentOffset, 16};
  }
};

TEST(ReplayDrawValidation, AcceptsBoundaryRangesBackedByReplayPrefix) {
  ValidGxFixture fixture;
  validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds);
}

TEST(ReplayDrawValidation, AcceptsIndexedRangeAtStartOfPerFrameStorage) {
  ValidGxFixture fixture;
  const uint32_t perFrameOffset = 0;
  std::memcpy(fixture.uniforms.data() + fixture.draw.uniformRange.offset + fixture.draw.posArrayUniformOffset,
              &perFrameOffset, sizeof(perFrameOffset));
  fixture.draw.indexedArrayRanges[0] = {0, 16};
  fixture.bounds.storage.highWater = 16;
  validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds);
}

TEST(ReplayDrawValidation, RejectsVertexRangePastOperationHighWater) {
  ValidGxFixture fixture;
  fixture.draw.vertRange.size += 1;
  EXPECT_DEATH(validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds),
               "Replay GX vertex range.*exceeds encode-operation high-water mark");
}

TEST(ReplayDrawValidation, RejectsIndexRangePastOperationHighWater) {
  ValidGxFixture fixture;
  fixture.draw.idxRange.size += 2;
  EXPECT_DEATH(validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds),
               "Replay GX index range.*exceeds encode-operation high-water mark");
}

TEST(ReplayDrawValidation, RejectsUniformRangePastOperationHighWater) {
  ValidGxFixture fixture;
  fixture.draw.uniformRange.size += 1;
  EXPECT_DEATH(validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds),
               "Replay GX uniform range.*exceeds encode-operation high-water mark");
}

TEST(ReplayDrawValidation, RejectsCountToByteMismatch) {
  ValidGxFixture fixture;
  fixture.draw.indexCount += 1;
  EXPECT_DEATH(validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds),
               "Replay GX index count requires 8 bytes, but DrawData names 6");
}

TEST(ReplayDrawValidation, RejectsMisalignedRange) {
  ValidGxFixture fixture;
  fixture.draw.vertRange.offset = 2;
  EXPECT_DEATH(validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds),
               "Replay GX vertex offset 2 is not aligned to 4 bytes");
}

TEST(ReplayDrawValidation, RejectsInterpolationSpanOutsideUniform) {
  ValidGxFixture fixture;
  fixture.draw.posArrayUniformOffset = fixture.draw.uniformRange.size - 4;
  EXPECT_DEATH(validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds),
               "Replay GX indexed-array offsets span.*exceeds draw uniform size");
}

TEST(ReplayDrawValidation, RejectsStorageOffsetWithoutCapturedOwner) {
  ValidGxFixture fixture;
  const uint32_t invalidOffset = 64;
  std::memcpy(fixture.uniforms.data() + fixture.draw.uniformRange.offset + fixture.draw.posArrayUniformOffset,
              &invalidOffset, sizeof(invalidOffset));
  EXPECT_DEATH(validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds),
               "indexed attribute 0 shader-visible storage offset 64 disagrees with retained range start");
}

TEST(ReplayDrawValidation, RejectsPersistentIndexedRangePastArenaEnd) {
  ValidGxFixture fixture;
  fixture.draw.indexedArrayRanges[0].size += 1;
  EXPECT_DEATH(validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds),
               "indexed attribute 0 persistent storage range.*exceeds arena end");
}

TEST(ReplayDrawValidation, RejectsPerFrameIndexedRangePastOperationHighWater) {
  ValidGxFixture fixture;
  const uint32_t perFrameOffset = 0;
  std::memcpy(fixture.uniforms.data() + fixture.draw.uniformRange.offset + fixture.draw.posArrayUniformOffset,
              &perFrameOffset, sizeof(perFrameOffset));
  fixture.draw.indexedArrayRanges[0] = {0, 17};
  fixture.bounds.storage.highWater = 16;
  EXPECT_DEATH(validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds),
               "indexed attribute 0 per-frame storage range.*exceeds encode-operation high-water mark");
}

TEST(ReplayDrawValidation, RejectsUsedIndexedArrayWithoutExtent) {
  ValidGxFixture fixture;
  fixture.draw.indexedArrayRanges[0].size = 0;
  EXPECT_DEATH(validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds),
               "indexed attribute 0 has an empty retained storage range");
}

TEST(ReplayDrawValidation, RmlAcceptsBoundedDynamicBindingExtent) {
  const validation::FrameBounds bounds{
      .vertices = {40, 40, static_cast<uint32_t>(aurora::gfx::VertexBufferSize)},
      .uniforms = {256, 0, static_cast<uint32_t>(aurora::gfx::UniformBufferSize)},
      .indices = {12, 12, static_cast<uint32_t>(aurora::gfx::IndexBufferSize)},
      .storage = {0, 0, static_cast<uint32_t>(aurora::gfx::StorageBufferSize)},
      .persistentStorageEnd = static_cast<uint32_t>(aurora::gfx::StorageBufferSize),
      .uniformOffsetAlignment = 256,
  };
  const validation::RmlDrawReferences draw{
      .vertexRange = {0, 40},
      .indexRange = {0, 12},
      .uniformRange = {0, 80},
      .bindGroup1 = 1,
      .bindGroup1DynamicOffset = 0,
      .dynamicBindGroupMask = 1u << 1u,
      .drawKind = 0,
      .vertexCount = 2,
      .indexCount = 3,
      .vertexStride = 20,
      .requiredUniformSize = 80,
      .bindGroup1DynamicExtent = 80,
  };
  validation::validate_rml(draw, bounds);
}

TEST(ReplayDrawValidation, RejectsRmlDynamicBindingPastUniformHighWater) {
  const validation::FrameBounds bounds{
      .vertices = {0, 0, static_cast<uint32_t>(aurora::gfx::VertexBufferSize)},
      .uniforms = {256, 0, static_cast<uint32_t>(aurora::gfx::UniformBufferSize)},
      .indices = {0, 0, static_cast<uint32_t>(aurora::gfx::IndexBufferSize)},
      .storage = {0, 0, static_cast<uint32_t>(aurora::gfx::StorageBufferSize)},
      .persistentStorageEnd = static_cast<uint32_t>(aurora::gfx::StorageBufferSize),
      .uniformOffsetAlignment = 256,
  };
  const validation::RmlDrawReferences draw{
      .bindGroup2 = 1,
      .bindGroup2DynamicOffset = 256,
      .dynamicBindGroupMask = 1u << 2u,
      .drawKind = 1,
      .vertexCount = 3,
      .bindGroup2DynamicExtent = 1,
  };
  EXPECT_DEATH(validation::validate_rml(draw, bounds),
               "Replay Rml dynamic group 2 range.*exceeds encode-operation high-water mark");
}

TEST(ReplayDrawValidation, RejectsRmlVertexCountToByteMismatch) {
  const validation::FrameBounds bounds{
      .vertices = {40, 40, static_cast<uint32_t>(aurora::gfx::VertexBufferSize)},
      .uniforms = {80, 0, static_cast<uint32_t>(aurora::gfx::UniformBufferSize)},
      .indices = {12, 12, static_cast<uint32_t>(aurora::gfx::IndexBufferSize)},
      .storage = {0, 0, static_cast<uint32_t>(aurora::gfx::StorageBufferSize)},
      .persistentStorageEnd = static_cast<uint32_t>(aurora::gfx::StorageBufferSize),
      .uniformOffsetAlignment = 256,
  };
  const validation::RmlDrawReferences draw{
      .vertexRange = {0, 40},
      .indexRange = {0, 12},
      .uniformRange = {0, 80},
      .drawKind = 0,
      .vertexCount = 3,
      .indexCount = 3,
      .vertexStride = 20,
      .requiredUniformSize = 80,
  };
  EXPECT_DEATH(validation::validate_rml(draw, bounds),
               "Replay Rml vertex count requires 60 bytes, but DrawData names 40");
}

} // namespace
