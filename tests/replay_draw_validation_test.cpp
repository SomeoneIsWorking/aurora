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
  };

  ValidGxFixture() {
    const uint32_t persistentOffset = static_cast<uint32_t>(aurora::gfx::StorageBufferSize);
    std::memcpy(uniforms.data() + draw.uniformRange.offset + draw.posArrayUniformOffset, &persistentOffset,
                sizeof(persistentOffset));
  }
};

TEST(ReplayDrawValidation, AcceptsBoundaryRangesBackedByReplayPrefix) {
  ValidGxFixture fixture;
  const auto result = validation::validate_gx(fixture.draw, fixture.uniforms, fixture.bounds);
  EXPECT_EQ(result.unchecked,
            validation::UncheckedGxStorageRangeExtents | validation::UncheckedGxPersistentRangeExtents);
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
               "indexed attribute 0 names storage offset 64 outside per-frame high-water 0");
}

TEST(ReplayDrawValidation, RmlReturnsUncheckedDynamicBindingExtent) {
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
  };
  const auto result = validation::validate_rml(draw, bounds);
  EXPECT_EQ(result.unchecked, validation::UncheckedRmlDynamicGroup1BindingExtent);
}

TEST(ReplayDrawValidation, RecordsUncheckedCoverageForShippingReplay) {
  const uint32_t coverage = validation::UncheckedGxStorageRangeExtents | validation::UncheckedGxPersistentRangeExtents |
                            validation::UncheckedRmlDynamicGroup1BindingExtent;
  validation::record_unchecked({.unchecked = coverage});
  EXPECT_EQ(validation::observed_unchecked() & coverage, coverage);
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
