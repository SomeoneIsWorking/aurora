#include "gpu_submit_probe.hpp"

#include <algorithm>

namespace aurora::gfx::gpu_submit_probe {
namespace {

Module Log("aurora::gfx::gpu_submit_probe");

enum class CommandKind : uint8_t {
  PaletteConversion,
  Viewport,
  Scissor,
  Draw,
  DebugMarker,
  Resolve,
};

template <typename T>
void hash_value(Hasher& hasher, const T& value) {
  hasher.update(&value, sizeof(value));
}

void hash_range(Hasher& hasher, const RangeInput& range) {
  hash_value(hasher, range.offset);
  hash_value(hasher, range.size);
}

void hash_resource(Hasher& hasher, const TextureResourceInput& resource) {
  hash_value(hasher, resource.generation);
  hash_value(hasher, resource.width);
  hash_value(hasher, resource.height);
  hash_value(hasher, resource.depthOrArrayLayers);
  hash_value(hasher, resource.format);
  hash_value(hasher, resource.mipCount);
  hash_value(hasher, resource.gxFormat);
}

void hash_draw_fields(Hasher& hasher, const ClearDrawInput& draw) {
  hash_value(hasher, static_cast<uint8_t>(AURORA_GPU_DRAW_CLEAR));
  hash_value(hasher, draw.pipeline);
  for (const double component : draw.color) {
    hash_value(hasher, component);
  }
  hash_value(hasher, draw.depth);
  hash_value(hasher, draw.rectEnabled);
  hash_value(hasher, draw.rectX);
  hash_value(hasher, draw.rectY);
  hash_value(hasher, draw.rectWidth);
  hash_value(hasher, draw.rectHeight);
}

void hash_draw_fields(Hasher& hasher, const GxDrawInput& draw) {
  hash_value(hasher, static_cast<uint8_t>(AURORA_GPU_DRAW_GX));
  hash_value(hasher, draw.pipeline);
  hash_range(hasher, draw.vertexRange);
  hash_range(hasher, draw.indexRange);
  hash_range(hasher, draw.uniformRange);
  hash_value(hasher, draw.vertexCount);
  hash_value(hasher, draw.indexCount);
  hash_value(hasher, draw.instanceCount);
  hash_value(hasher, draw.textureBindGroup);
  hash_value(hasher, draw.destinationAlpha);
  hash_value(hasher, draw.tag);
  hash_value(hasher, draw.population);
  hash_value(hasher, draw.exact);
  hash_value(hasher, draw.indexedPositionSample);
  hash_value(hasher, draw.positionArrayUniformOffset);
  hash_value(hasher, draw.matrixPositionOffset);
  hash_value(hasher, draw.matrixNormalOffset);
  hash_value(hasher, draw.orthographic);
  hash_value(hasher, draw.vertexStride);
  hash_value(hasher, draw.positionOffset);
  hash_value(hasher, draw.positionF32Xyz);
  hash_value(hasher, draw.positionS16Xyz);
  hash_value(hasher, draw.positionFraction);
  hash_value(hasher, draw.deformF32OffsetMask);
  hash_value(hasher, draw.cameraTextureMatrixMask);
  hash_value(hasher, draw.positionMatrixSlot);
  hash_value(hasher, draw.indexedArrayUsedMask);
  for (const RangeInput& range : draw.indexedArrayRanges) {
    hash_range(hasher, range);
  }
}

void hash_draw_fields(Hasher& hasher, const RmlDrawInput& draw) {
  hash_value(hasher, static_cast<uint8_t>(AURORA_GPU_DRAW_RML));
  hash_value(hasher, draw.pipeline);
  hash_range(hasher, draw.vertexRange);
  hash_range(hasher, draw.indexRange);
  hash_range(hasher, draw.uniformRange);
  hash_value(hasher, draw.bindGroup1);
  hash_value(hasher, draw.bindGroup2);
  hash_value(hasher, draw.bindGroup1DynamicOffset);
  hash_value(hasher, draw.bindGroup2DynamicOffset);
  hash_value(hasher, draw.dynamicBindGroupMask);
  hash_value(hasher, draw.drawKind);
  hash_value(hasher, draw.vertexCount);
  hash_value(hasher, draw.indexCount);
  hash_value(hasher, draw.stencilReference);
  for (const float component : draw.blendConstant) {
    hash_value(hasher, component);
  }
  hash_value(hasher, draw.hasBlendConstant);
  hash_value(hasher, draw.bindGroup1DynamicExtent);
  hash_value(hasher, draw.bindGroup2DynamicExtent);
}

template <typename Draw>
uint64_t hash_draw_impl(const Draw& draw) {
  Hasher hasher;
  hash_draw_fields(hasher, draw);
  return hasher.digest();
}

} // namespace

uint64_t hash_draw(const ClearDrawInput& draw) { return hash_draw_impl(draw); }

uint64_t hash_draw(const GxDrawInput& draw) { return hash_draw_impl(draw); }

uint64_t hash_draw(const RmlDrawInput& draw) { return hash_draw_impl(draw); }

Builder::Builder(const FrameInput& frame) {
  m_info.structSize = sizeof(m_info);
  m_info.version = AURORA_GPU_PROBE_VERSION;
  m_info.kind = AURORA_GPU_SUBMIT_FRAME;
  m_info.replayEmission = frame.replayEmission;
  m_info.frameId = frame.frameId;
  m_info.frameIndex = frame.frameIndex;
  m_info.passCount = frame.passCount;
  m_info.recordedPassCount = std::min<uint32_t>(frame.passCount, AURORA_GPU_PROBE_MAX_PASSES);
  m_info.operationCount = frame.operationCount;
  m_info.textureUploadCount = frame.textureUploadCount;
  m_info.textureCopyCount = frame.textureCopyCount;
  m_info.vertexBytes = frame.vertexBytes;
  m_info.uniformBytes = frame.uniformBytes;
  m_info.indexBytes = frame.indexBytes;
  m_info.storageBytes = frame.storageBytes;
  m_info.textureUploadBytes = frame.textureUploadBytes;
  m_info.cachedTextureObjects = frame.cachedTextureObjects;
  m_info.cachedTlutObjects = frame.cachedTlutObjects;
  m_info.cachedCopyTextures = frame.cachedCopyTextures;
  m_info.cachedBindGroups = frame.cachedBindGroups;
  m_info.persistentStorageEntries = frame.persistentStorageEntries;
  m_info.persistentStorageBytes = frame.persistentStorageBytes;
  m_info.replaySourceFrameId = frame.replaySourceFrameId;
  m_info.replaySourceCommandHash = frame.replaySourceCommandHash;
  m_info.replaySourceUniformHash = frame.replaySourceUniformHash;
}

void Builder::begin_pass(const PassInput& pass) {
  CHECK(!m_passOpen, "GPU submit probe began pass {} before ending pass {}", m_passIndex, m_passIndex - 1);
  CHECK(m_passIndex < m_info.passCount, "GPU submit probe received more passes than declared ({})", m_info.passCount);
  m_passOpen = true;
  m_commandIndex = 0;
  m_passDrawCount = 0;
  m_passCommands = Hasher{};
  m_passPipelines = Hasher{};

  const uint64_t labelSize = pass.label.size();
  hash_value(m_passCommands, labelSize);
  m_passCommands.update(pass.label.data(), pass.label.size());
  hash_value(m_passCommands, pass.commandCount);
  hash_value(m_passCommands, pass.targetWidth);
  hash_value(m_passCommands, pass.targetHeight);
  hash_value(m_passCommands, pass.flags);

  if (m_passIndex < m_info.recordedPassCount) {
    auto& out = m_info.passes[m_passIndex];
    out.labelHash = xxh3_hash_s(pass.label.data(), pass.label.size());
    out.commandCount = pass.commandCount;
    out.targetWidth = pass.targetWidth;
    out.targetHeight = pass.targetHeight;
    out.flags = pass.flags;
  }
}

void Builder::add_viewport(const ViewportInput& viewport) {
  CHECK(m_passOpen, "GPU submit probe received viewport outside a pass");
  hash_value(m_passCommands, CommandKind::Viewport);
  hash_value(m_passCommands, viewport.x);
  hash_value(m_passCommands, viewport.y);
  hash_value(m_passCommands, viewport.width);
  hash_value(m_passCommands, viewport.height);
  hash_value(m_passCommands, viewport.minDepth);
  hash_value(m_passCommands, viewport.maxDepth);
  ++m_commandIndex;
}

void Builder::add_scissor(const ScissorInput& scissor) {
  CHECK(m_passOpen, "GPU submit probe received scissor outside a pass");
  hash_value(m_passCommands, CommandKind::Scissor);
  hash_value(m_passCommands, scissor.x);
  hash_value(m_passCommands, scissor.y);
  hash_value(m_passCommands, scissor.width);
  hash_value(m_passCommands, scissor.height);
  ++m_commandIndex;
}

template <typename Draw>
void Builder::add_draw_impl(const Draw& draw, uint8_t shaderType, uint64_t pipeline, uint64_t tag,
                            RangeInput vertexRange, RangeInput indexRange, RangeInput uniformRange, uint8_t population,
                            uint8_t flags) {
  CHECK(m_passOpen, "GPU submit probe received draw outside a pass");
  hash_value(m_passCommands, CommandKind::Draw);
  hash_draw_fields(m_passCommands, draw);
  hash_value(m_passPipelines, pipeline);

  AuroraGpuDrawProbe& probe = m_drawTail[m_info.drawCount % AURORA_GPU_PROBE_MAX_DRAWS];
  probe = {
      .drawHash = hash_draw(draw),
      .pipelineId = pipeline,
      .tag = tag,
      .passIndex = m_passIndex,
      .commandIndex = m_commandIndex,
      .drawIndex = m_info.drawCount,
      .vertexOffset = vertexRange.offset,
      .vertexBytes = vertexRange.size,
      .indexOffset = indexRange.offset,
      .indexBytes = indexRange.size,
      .uniformOffset = uniformRange.offset,
      .uniformBytes = uniformRange.size,
      .shaderType = shaderType,
      .population = population,
      .flags = flags,
  };
  ++m_info.drawCount;
  ++m_passDrawCount;
  ++m_commandIndex;
}

void Builder::add_draw(const ClearDrawInput& draw) {
  add_draw_impl(draw, AURORA_GPU_DRAW_CLEAR, draw.pipeline, 0, {}, {}, {}, 0, 0);
}

void Builder::add_draw(const GxDrawInput& draw) {
  const uint8_t flags = (draw.exact ? AURORA_GPU_DRAW_FLAG_EXACT : 0u) |
                        (draw.orthographic ? AURORA_GPU_DRAW_FLAG_ORTHOGRAPHIC : 0u) |
                        (draw.indexedPositionSample != 0 ? AURORA_GPU_DRAW_FLAG_INDEXED_POSITION : 0u) |
                        (draw.deformF32OffsetMask != 0 ? AURORA_GPU_DRAW_FLAG_DEFORMING : 0u) |
                        (draw.cameraTextureMatrixMask != 0 ? AURORA_GPU_DRAW_FLAG_CAMERA_TEX_MATRIX : 0u) |
                        (draw.destinationAlpha != UINT32_MAX ? AURORA_GPU_DRAW_FLAG_DEST_ALPHA : 0u);
  add_draw_impl(draw, AURORA_GPU_DRAW_GX, draw.pipeline, draw.tag, draw.vertexRange, draw.indexRange, draw.uniformRange,
                draw.population, flags);
}

void Builder::add_draw(const RmlDrawInput& draw) {
  add_draw_impl(draw, AURORA_GPU_DRAW_RML, draw.pipeline, 0, draw.vertexRange, draw.indexRange, draw.uniformRange, 0,
                0);
}

void Builder::add_debug_marker(std::string_view label) {
  CHECK(m_passOpen, "GPU submit probe received debug marker outside a pass");
  hash_value(m_passCommands, CommandKind::DebugMarker);
  const uint64_t labelSize = label.size();
  hash_value(m_passCommands, labelSize);
  m_passCommands.update(label.data(), label.size());
  ++m_commandIndex;
}

void Builder::add_palette_conversion(const PaletteConversionInput& conversion) {
  CHECK(m_passOpen, "GPU submit probe received palette conversion outside a pass");
  hash_value(m_passCommands, CommandKind::PaletteConversion);
  hash_value(m_passCommands, conversion.variant);
  hash_resource(m_passCommands, conversion.source);
  hash_resource(m_passCommands, conversion.destination);
  hash_resource(m_passCommands, conversion.palette);
}

void Builder::add_resolve(const ResolveInput& resolve) {
  CHECK(m_passOpen, "GPU submit probe received resolve outside a pass");
  hash_value(m_passCommands, CommandKind::Resolve);
  hash_value(m_passCommands, resolve.format);
  hash_value(m_passCommands, resolve.rectX);
  hash_value(m_passCommands, resolve.rectY);
  hash_value(m_passCommands, resolve.rectWidth);
  hash_value(m_passCommands, resolve.rectHeight);
  hash_range(m_passCommands, resolve.uniformRange);
  hash_resource(m_passCommands, resolve.source);
  hash_resource(m_passCommands, resolve.destination);
  hash_value(m_passCommands, resolve.sourceSamples);
  hash_value(m_passCommands, resolve.path);
  hash_value(m_passCommands, resolve.sourceIsDepth);
}

void Builder::end_pass() {
  CHECK(m_passOpen, "GPU submit probe ended a pass when none was open");
  const uint64_t commandHash = m_passCommands.digest();
  const uint64_t pipelineHash = m_passPipelines.digest();
  if (m_passIndex < m_info.recordedPassCount) {
    auto& out = m_info.passes[m_passIndex];
    out.drawCount = m_passDrawCount;
    out.commandHash = commandHash;
    out.pipelineHash = pipelineHash;
  }
  hash_value(m_frameCommands, commandHash);
  hash_value(m_framePipelines, pipelineHash);
  ++m_passIndex;
  m_passOpen = false;
}

AuroraGpuSubmitInfo Builder::finish() {
  CHECK(!m_passOpen, "GPU submit probe finished with pass {} still open", m_passIndex);
  CHECK(m_passIndex == m_info.passCount, "GPU submit probe recorded {} of {} declared passes", m_passIndex,
        m_info.passCount);
  m_info.commandHash = m_frameCommands.digest();
  m_info.pipelineHash = m_framePipelines.digest();
  m_info.recordedDrawCount = std::min<uint32_t>(m_info.drawCount, AURORA_GPU_PROBE_MAX_DRAWS);
  m_info.firstRecordedDraw = m_info.drawCount - m_info.recordedDrawCount;
  for (uint32_t index = 0; index < m_info.recordedDrawCount; ++index) {
    m_info.draws[index] = m_drawTail[(m_info.firstRecordedDraw + index) % AURORA_GPU_PROBE_MAX_DRAWS];
  }
  return m_info;
}

} // namespace aurora::gfx::gpu_submit_probe
