#include "../lib/gfx/gpu_submit_probe.hpp"

#include <gtest/gtest.h>

namespace aurora::gfx::gpu_submit_probe {
namespace {

template <typename Draw, typename Mutator>
void expect_hash_change(const Draw& baseline, Mutator mutate, const char* field) {
  Draw changed = baseline;
  mutate(changed);
  EXPECT_NE(hash_draw(baseline), hash_draw(changed)) << field;
}

#define EXPECT_HASH_CHANGE(draw, field, value)                                                                         \
  expect_hash_change(draw, [](auto& changed) { changed.field = value; }, #field)

TEST(GpuSubmitProbe, ClearHashCoversEverySemanticField) {
  const ClearDrawInput draw{};
  EXPECT_HASH_CHANGE(draw, pipeline, 1);
  for (size_t component = 0; component < draw.color.size(); ++component) {
    expect_hash_change(draw, [component](auto& changed) { changed.color[component] = 1.0; }, "color");
  }
  EXPECT_HASH_CHANGE(draw, depth, 1.0f);
  EXPECT_HASH_CHANGE(draw, rectEnabled, 1);
  EXPECT_HASH_CHANGE(draw, rectX, 1);
  EXPECT_HASH_CHANGE(draw, rectY, 1);
  EXPECT_HASH_CHANGE(draw, rectWidth, 1);
  EXPECT_HASH_CHANGE(draw, rectHeight, 1);
}

TEST(GpuSubmitProbe, GxHashCoversEverySemanticField) {
  const GxDrawInput draw{};
  EXPECT_HASH_CHANGE(draw, pipeline, 1);
  EXPECT_HASH_CHANGE(draw, vertexRange.offset, 1);
  EXPECT_HASH_CHANGE(draw, vertexRange.size, 1);
  EXPECT_HASH_CHANGE(draw, indexRange.offset, 1);
  EXPECT_HASH_CHANGE(draw, indexRange.size, 1);
  EXPECT_HASH_CHANGE(draw, uniformRange.offset, 1);
  EXPECT_HASH_CHANGE(draw, uniformRange.size, 1);
  EXPECT_HASH_CHANGE(draw, vertexCount, 1);
  EXPECT_HASH_CHANGE(draw, indexCount, 1);
  EXPECT_HASH_CHANGE(draw, instanceCount, 1);
  EXPECT_HASH_CHANGE(draw, textureBindGroup, 1);
  EXPECT_HASH_CHANGE(draw, destinationAlpha, 1);
  EXPECT_HASH_CHANGE(draw, tag, 1);
  EXPECT_HASH_CHANGE(draw, population, 1);
  EXPECT_HASH_CHANGE(draw, exact, 1);
  EXPECT_HASH_CHANGE(draw, indexedPositionSample, 1);
  EXPECT_HASH_CHANGE(draw, positionArrayUniformOffset, 1);
  EXPECT_HASH_CHANGE(draw, matrixPositionOffset, 1);
  EXPECT_HASH_CHANGE(draw, matrixNormalOffset, 1);
  EXPECT_HASH_CHANGE(draw, orthographic, 1);
  EXPECT_HASH_CHANGE(draw, vertexStride, 1);
  EXPECT_HASH_CHANGE(draw, positionOffset, 1);
  EXPECT_HASH_CHANGE(draw, positionF32Xyz, 1);
  EXPECT_HASH_CHANGE(draw, positionS16Xyz, 1);
  EXPECT_HASH_CHANGE(draw, positionFraction, 1);
  EXPECT_HASH_CHANGE(draw, deformF32OffsetMask, 1);
  EXPECT_HASH_CHANGE(draw, cameraTextureMatrixMask, 1);
  EXPECT_HASH_CHANGE(draw, positionMatrixSlot, 1);
}

TEST(GpuSubmitProbe, RmlHashCoversEverySemanticField) {
  const RmlDrawInput draw{};
  EXPECT_HASH_CHANGE(draw, pipeline, 1);
  EXPECT_HASH_CHANGE(draw, vertexRange.offset, 1);
  EXPECT_HASH_CHANGE(draw, vertexRange.size, 1);
  EXPECT_HASH_CHANGE(draw, indexRange.offset, 1);
  EXPECT_HASH_CHANGE(draw, indexRange.size, 1);
  EXPECT_HASH_CHANGE(draw, uniformRange.offset, 1);
  EXPECT_HASH_CHANGE(draw, uniformRange.size, 1);
  EXPECT_HASH_CHANGE(draw, bindGroup1, 1);
  EXPECT_HASH_CHANGE(draw, bindGroup2, 1);
  EXPECT_HASH_CHANGE(draw, bindGroup1DynamicOffset, 1);
  EXPECT_HASH_CHANGE(draw, bindGroup2DynamicOffset, 1);
  EXPECT_HASH_CHANGE(draw, dynamicBindGroupMask, 1);
  EXPECT_HASH_CHANGE(draw, drawKind, 1);
  EXPECT_HASH_CHANGE(draw, vertexCount, 1);
  EXPECT_HASH_CHANGE(draw, indexCount, 1);
  EXPECT_HASH_CHANGE(draw, stencilReference, 1);
  for (size_t component = 0; component < draw.blendConstant.size(); ++component) {
    expect_hash_change(draw, [component](auto& changed) { changed.blendConstant[component] = 1.0f; }, "blendConstant");
  }
  EXPECT_HASH_CHANGE(draw, hasBlendConstant, 1);
}

TEST(GpuSubmitProbe, DrawTailIsBoundedAndChronological) {
  Builder builder{FrameInput{.passCount = 1}};
  builder.begin_pass(PassInput{.label = "tail-control", .commandCount = 12});
  for (uint32_t index = 0; index < 12; ++index) {
    builder.add_draw(GxDrawInput{.pipeline = 0x100 + index, .tag = 0x200 + index});
  }
  builder.end_pass();
  const AuroraGpuSubmitInfo info = builder.finish();

  ASSERT_EQ(info.drawCount, 12u);
  ASSERT_EQ(info.recordedDrawCount, AURORA_GPU_PROBE_MAX_DRAWS);
  ASSERT_EQ(info.firstRecordedDraw, 3u);
  for (uint32_t index = 0; index < info.recordedDrawCount; ++index) {
    EXPECT_EQ(info.draws[index].drawIndex, index + 3);
    EXPECT_EQ(info.draws[index].tag, 0x203u + index);
  }
}

TEST(GpuSubmitProbe, DestinationAlphaFlagUsesTheDisabledSentinel) {
  const auto finish = [](uint32_t destinationAlpha) {
    Builder builder{FrameInput{.passCount = 1}};
    builder.begin_pass(PassInput{.label = "destination-alpha", .commandCount = 1});
    builder.add_draw(GxDrawInput{.destinationAlpha = destinationAlpha});
    builder.end_pass();
    return builder.finish();
  };

  const AuroraGpuSubmitInfo disabled = finish(UINT32_MAX);
  const AuroraGpuSubmitInfo enabledZero = finish(0);
  ASSERT_EQ(disabled.recordedDrawCount, 1u);
  ASSERT_EQ(enabledZero.recordedDrawCount, 1u);
  EXPECT_EQ(disabled.draws[0].flags & AURORA_GPU_DRAW_FLAG_DEST_ALPHA, 0u);
  EXPECT_NE(enabledZero.draws[0].flags & AURORA_GPU_DRAW_FLAG_DEST_ALPHA, 0u);
}

#undef EXPECT_HASH_CHANGE

} // namespace
} // namespace aurora::gfx::gpu_submit_probe
