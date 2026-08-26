#pragma once

#include "common.hpp"

#include <cstdint>
#include <span>

namespace aurora::gx {
struct DrawData;
}

namespace aurora::gfx::replay_draw_validation {

enum UncheckedReference : uint32_t {
  UncheckedNone = 0,
  // gx::DrawData retains indexed-array start offsets, but no used-slot mask or per-array extents.
  UncheckedGxStorageRangeExtents = 1u << 0u,
  UncheckedGxPersistentRangeExtents = 1u << 1u,
  // The Rml bind-group cache retains a Dawn bind group, but not each buffer binding's byte extent.
  UncheckedRmlDynamicGroup1BindingExtent = 1u << 2u,
  UncheckedRmlDynamicGroup2BindingExtent = 1u << 3u,
};

struct Result {
  uint32_t unchecked = UncheckedNone;
};

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
};

Result validate_gx(const gx::DrawData& draw, std::span<const uint8_t> uniformBytes, const FrameBounds& bounds);
Result validate_rml(const RmlDrawReferences& draw, const FrameBounds& bounds);

// The shipping replay seam records every returned coverage gap here. This monotonic bitmask is
// intentionally queryable by diagnostics without logging from the per-draw encode path.
void record_unchecked(Result result);
uint32_t observed_unchecked();

} // namespace aurora::gfx::replay_draw_validation
