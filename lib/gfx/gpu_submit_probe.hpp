#pragma once

#include "common.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace aurora::gfx::gpu_submit_probe {

struct RangeInput {
  uint32_t offset = 0;
  uint32_t size = 0;
};

struct ClearDrawInput {
  PipelineRef pipeline = 0;
  std::array<double, 4> color{};
  float depth = 0.0f;
  uint8_t rectEnabled = 0;
  int32_t rectX = 0;
  int32_t rectY = 0;
  int32_t rectWidth = 0;
  int32_t rectHeight = 0;
};

struct GxDrawInput {
  PipelineRef pipeline = 0;
  RangeInput vertexRange;
  RangeInput indexRange;
  RangeInput uniformRange;
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
  uint32_t instanceCount = 0;
  BindGroupRef textureBindGroup = 0;
  uint32_t destinationAlpha = UINT32_MAX;
  uint64_t tag = 0;
  uint8_t population = 0;
  uint8_t exact = 0;
  uint32_t indexedPositionSample = 0;
  uint32_t positionArrayUniformOffset = 0;
  uint32_t matrixPositionOffset = 0;
  uint32_t matrixNormalOffset = 0;
  uint8_t orthographic = 0;
  uint16_t vertexStride = 0;
  uint16_t positionOffset = 0;
  uint8_t positionF32Xyz = 0;
  uint8_t positionS16Xyz = 0;
  uint8_t positionFraction = 0;
  uint64_t deformF32OffsetMask = 0;
  uint32_t cameraTextureMatrixMask = 0;
  uint8_t positionMatrixSlot = 0;
  uint32_t indexedArrayUsedMask = 0;
  std::array<RangeInput, 12> indexedArrayRanges{};
};

struct RmlDrawInput {
  PipelineRef pipeline = 0;
  RangeInput vertexRange;
  RangeInput indexRange;
  RangeInput uniformRange;
  BindGroupRef bindGroup1 = 0;
  BindGroupRef bindGroup2 = 0;
  uint32_t bindGroup1DynamicOffset = 0;
  uint32_t bindGroup2DynamicOffset = 0;
  uint32_t dynamicBindGroupMask = 0;
  uint32_t drawKind = 0;
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
  uint32_t stencilReference = 0;
  std::array<float, 4> blendConstant{};
  uint32_t hasBlendConstant = 0;
  uint32_t bindGroup1DynamicExtent = 0;
  uint32_t bindGroup2DynamicExtent = 0;
};

struct FrameInput {
  uint32_t replayEmission = 0;
  uint64_t frameId = 0;
  uint32_t frameIndex = 0;
  uint32_t passCount = 0;
  uint32_t operationCount = 0;
  uint32_t textureUploadCount = 0;
  uint32_t textureCopyCount = 0;
  uint32_t vertexBytes = 0;
  uint32_t uniformBytes = 0;
  uint32_t indexBytes = 0;
  uint32_t storageBytes = 0;
  uint32_t textureUploadBytes = 0;
  uint32_t cachedTextureObjects = 0;
  uint32_t cachedTlutObjects = 0;
  uint32_t cachedCopyTextures = 0;
  uint32_t cachedBindGroups = 0;
  uint32_t persistentStorageEntries = 0;
  uint32_t persistentStorageBytes = 0;
};

struct PassInput {
  std::string_view label;
  uint32_t commandCount = 0;
  uint32_t targetWidth = 0;
  uint32_t targetHeight = 0;
  uint32_t flags = 0;
};

struct ViewportInput {
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float minDepth = 0.0f;
  float maxDepth = 0.0f;
};

struct ScissorInput {
  int32_t x = 0;
  int32_t y = 0;
  int32_t width = 0;
  int32_t height = 0;
};

uint64_t hash_draw(const ClearDrawInput& draw);
uint64_t hash_draw(const GxDrawInput& draw);
uint64_t hash_draw(const RmlDrawInput& draw);

class Builder {
public:
  explicit Builder(const FrameInput& frame);

  void begin_pass(const PassInput& pass);
  void add_viewport(const ViewportInput& viewport);
  void add_scissor(const ScissorInput& scissor);
  void add_draw(const ClearDrawInput& draw);
  void add_draw(const GxDrawInput& draw);
  void add_draw(const RmlDrawInput& draw);
  void add_debug_marker(uint64_t markerIndex);
  void end_pass();
  AuroraGpuSubmitInfo finish();

private:
  template <typename Draw>
  void add_draw_impl(const Draw& draw, uint8_t shaderType, uint64_t pipeline, uint64_t tag, RangeInput vertexRange,
                     RangeInput indexRange, RangeInput uniformRange, uint8_t population, uint8_t flags);

  AuroraGpuSubmitInfo m_info{};
  std::array<AuroraGpuDrawProbe, AURORA_GPU_PROBE_MAX_DRAWS> m_drawTail{};
  Hasher m_frameCommands;
  Hasher m_framePipelines;
  Hasher m_passCommands;
  Hasher m_passPipelines;
  uint32_t m_passIndex = 0;
  uint32_t m_commandIndex = 0;
  uint32_t m_passDrawCount = 0;
  bool m_passOpen = false;
};

} // namespace aurora::gfx::gpu_submit_probe
