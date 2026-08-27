#pragma once

#include "common.hpp"

#include <cstdint>
#include <span>

namespace aurora::gx {
struct DrawData;
}

namespace aurora::gfx::replay_draw_validation {

struct BufferWindow {
  uint32_t highWater = 0;
  uint32_t replayPrefix = 0;
  uint32_t capacity = 0;
};

struct FrameBounds {
  BufferWindow vertices;
  BufferWindow uniforms;
  BufferWindow indices;
  BufferWindow storage;
  uint32_t persistentStorageEnd = 0;
  uint32_t uniformOffsetAlignment = 1;
};

struct RmlDrawReferences {
  Range vertexRange;
  Range indexRange;
  Range uniformRange;
  uint64_t bindGroup1 = 0;
  uint64_t bindGroup2 = 0;
  uint32_t bindGroup1DynamicOffset = 0;
  uint32_t bindGroup2DynamicOffset = 0;
  uint32_t dynamicBindGroupMask = 0;
  uint32_t drawKind = 0;
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
  uint32_t vertexStride = 0;
  uint32_t requiredUniformSize = 0;
  uint32_t bindGroup1DynamicExtent = 0;
  uint32_t bindGroup2DynamicExtent = 0;
};

void validate_gx(const gx::DrawData& draw, std::span<const uint8_t> uniformBytes, const FrameBounds& bounds);
void validate_rml(const RmlDrawReferences& draw, const FrameBounds& bounds);

} // namespace aurora::gfx::replay_draw_validation
