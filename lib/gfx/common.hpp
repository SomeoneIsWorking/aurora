#pragma once

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <type_traits>
#include <utility>

#include <aurora/gfx.h>
#include <aurora/math.hpp>
#include <dolphin/gx/GXEnum.h>
#include <webgpu/webgpu_cpp.h>
#define XXH_STATIC_LINKING_ONLY
#include <xxhash.h>

namespace aurora {
#if INTPTR_MAX == INT32_MAX
using HashType = XXH32_hash_t;
#else
using HashType = XXH64_hash_t;
#endif
static inline HashType xxh3_hash_s(const void* input, size_t len, HashType seed = 0) {
  return static_cast<HashType>(XXH3_64bits_withSeed(input, len, seed));
}
template <typename T>
static inline HashType xxh3_hash(const T& input, HashType seed = 0) {
  // Validate that the type has no padding bytes, which can easily cause
  // hash mismatches. This also disallows floats, but that's okay for us.
  static_assert(std::has_unique_object_representations_v<T>);
  return xxh3_hash_s(&input, sizeof(T), seed);
}

class Hasher {
public:
  explicit Hasher(const XXH64_hash_t seed = 0) {
    XXH3_INITSTATE(&state);
    XXH3_64bits_reset_withSeed(&state, seed);
  }

  void update(const void* data, const size_t size) { XXH3_64bits_update(&state, data, size); }

  template <typename T>
  void update(const T& data) {
    static_assert(std::has_unique_object_representations_v<T>);
    update(&data, sizeof(T));
  }

  [[nodiscard]] XXH64_hash_t digest() const { return XXH3_64bits_digest(&state); }

private:
  XXH3_state_t state;
};

class ByteBuffer {
public:
  ByteBuffer() noexcept = default;
  explicit ByteBuffer(size_t size) noexcept
  : m_data(static_cast<uint8_t*>(calloc(1, size))), m_length(size), m_capacity(size) {}
  explicit ByteBuffer(uint8_t* data, size_t size) noexcept : m_data(data), m_capacity(size), m_owned(false) {}
  ~ByteBuffer() noexcept {
    if (m_data != nullptr && m_owned) {
      free(m_data);
    }
  }
  ByteBuffer(ByteBuffer&& rhs) noexcept
  : m_data(rhs.m_data), m_length(rhs.m_length), m_capacity(rhs.m_capacity), m_owned(rhs.m_owned) {
    rhs.m_data = nullptr;
    rhs.m_length = 0;
    rhs.m_capacity = 0;
    rhs.m_owned = true;
  }
  ByteBuffer& operator=(ByteBuffer&& rhs) noexcept {
    if (m_data != nullptr && m_owned) {
      free(m_data);
    }
    m_data = rhs.m_data;
    m_length = rhs.m_length;
    m_capacity = rhs.m_capacity;
    m_owned = rhs.m_owned;
    rhs.m_data = nullptr;
    rhs.m_length = 0;
    rhs.m_capacity = 0;
    rhs.m_owned = true;
    return *this;
  }
  ByteBuffer(ByteBuffer const&) = delete;
  ByteBuffer& operator=(ByteBuffer const&) = delete;
  operator ArrayRef<uint8_t>() const noexcept { return {m_data, m_length}; }

  [[nodiscard]] uint8_t* data() noexcept { return m_data; }
  [[nodiscard]] const uint8_t* data() const noexcept { return m_data; }
  [[nodiscard]] size_t size() const noexcept { return m_length; }
  [[nodiscard]] bool empty() const noexcept { return m_length == 0; }

  void append(const void* data, size_t size) {
    resize(m_length + size, false);
    memcpy(m_data + m_length, data, size);
    m_length += size;
  }

  template <typename T>
  void append(const T& obj) {
    append(&obj, sizeof(T));
  }

  void append_zeroes(size_t size) {
    resize(m_length + size, true);
    m_length += size;
  }

  void release() {
    if (m_data != nullptr && m_owned) {
      free(m_data);
    }
    m_data = nullptr;
    m_length = 0;
    m_capacity = 0;
    m_owned = true;
  }

  void clear() {
    m_length = 0;
  }

  void reserve_extra(size_t size) { resize(m_length + size, true); }

  ByteBuffer clone() const {
    ByteBuffer clone{m_length};
    std::memcpy(clone.data(), m_data, m_length);
    return clone;
  }

private:
  uint8_t* m_data = nullptr;
  size_t m_length = 0;
  size_t m_capacity = 0;
  bool m_owned = true;

  void resize(size_t size, bool zeroed) {
    if (size == 0) {
      clear();
    } else if (m_data == nullptr) {
      if (zeroed) {
        m_data = static_cast<uint8_t*>(calloc(1, size));
      } else {
        m_data = static_cast<uint8_t*>(malloc(size));
      }
      m_owned = true;
    } else if (size > m_capacity) {
      if (!m_owned) {
        // A mapped GPU staging region (verts/uniforms/indices/storage) can't be
        // realloc'd — overflowing it is a real capacity error, not something to
        // silently abort on. Report which by how much (see the *BufferSize
        // constants) so it can be sized to the workload / this points at a
        // runaway upload.
        extern const char* aurora_gfx_last_draw_desc(); // command_processor.cpp
        std::fprintf(stderr,
            "[aurora FATAL gfx] mapped ByteBuffer overflow: have %zu bytes "
            "(capacity %zu), need %zu more -> %zu total. A per-frame staging "
            "region is too small for this scene (or a runaway upload). "
            "last draw: %s\n",
            m_length, m_capacity, size - m_length, size, aurora_gfx_last_draw_desc());
        abort();
      }
      // Exponential expansion to avoid O(n^2) time complexity.
      if (size < m_capacity * 2) {
        size = m_capacity * 2;
      }
      m_data = static_cast<uint8_t*>(realloc(m_data, size));
      if (zeroed) {
        memset(m_data + m_capacity, 0, size - m_capacity);
      }
    } else {
      return;
    }
    m_capacity = size;
  }
};
} // namespace aurora

namespace aurora::gfx {
inline constexpr bool UseTextureBuffer = true;
inline constexpr uint64_t UniformBufferSize = 25165824;  // 24mb
inline constexpr uint64_t VertexBufferSize = 3145728;    // 3mb
inline constexpr uint64_t IndexBufferSize = 1048576;     // 1mb
inline constexpr uint64_t StorageBufferSize = 50331648;  // 48mb (was 8mb, a
// title-era size). Measured: a single Delfino Plaza pass (SB_SKIP_GHOST=1, ghost
// pass OFF) overflows 8MB — gameplay map geometry genuinely needs more indexed-
// array storage per frame than the title ever did. 32MB fits the single pass
// (verified: no overflow with the ghost pass skipped). NOTE: with the redundant
// phase-1 ghost pass ON (default) storage ~doubles to ~33MB and still overflows
// this — that doubling is a SEPARATE wart to remove (see
// sunbright debug_journal/2026-07-15_delfino_storage_overflow_ghost_pass.md), not
// a reason to keep growing this cap. 2026-07-16: grown 32->48mb anyway — the
// title/file-select frame already sat at 33.44mb (99.7% of 32mb, ghost-pass
// doubling included) and the faithful TMBindShadowManager volume passes tipped
// it over. The ghost-pass wart REMAINS the real fix; this headroom just stops
// capacity from gating unrelated ports. The overflow fatal now prints the last
// 16 draw identities, so a future runaway is self-diagnosing.
inline constexpr uint64_t TextureUploadSize = 25165824;  // 24mb

extern AuroraStats g_stats;
extern uint32_t g_drawCallCount;
extern uint32_t g_mergedDrawCallCount;
extern wgpu::Buffer g_vertexBuffer;
extern wgpu::Buffer g_uniformBuffer;
extern wgpu::Buffer g_indexBuffer;
extern wgpu::Buffer g_storageBuffer;
extern wgpu::BindGroupLayout g_staticBindGroupLayout;
extern wgpu::BindGroup g_staticBindGroup;
extern wgpu::BindGroupLayout g_uniformBindGroupLayout;
extern wgpu::BindGroup g_uniformBindGroup;

using BindGroupRef = HashType;
using PipelineRef = HashType;
using SamplerRef = HashType;
using ShaderRef = HashType;
struct Range {
  uint32_t offset = 0;
  uint32_t size = 0;

  bool operator==(const Range& rhs) const { return memcmp(this, &rhs, sizeof(*this)) == 0; }
  bool operator!=(const Range& rhs) const { return !(*this == rhs); }
};

struct ClipRect {
  int32_t x;
  int32_t y;
  int32_t width;
  int32_t height;

  bool operator==(const ClipRect& rhs) const { return memcmp(this, &rhs, sizeof(*this)) == 0; }
  bool operator!=(const ClipRect& rhs) const { return !(*this == rhs); }
};

using webgpu::Viewport;

struct TextureRef;
using TextureHandle = std::shared_ptr<TextureRef>;
using EndFrameCallback = std::function<void(wgpu::CommandEncoder&)>;

enum class ShaderType : uint8_t {
  Clear = 0,
  GX = 1,
  Rml = 2,
};

void initialize();
void shutdown();

bool begin_frame();
void finish();
void end_frame(EndFrameCallback callback);

// Replay present (AURORA_REPLAY_PRESENT=1) — re-present the frame just recorded, unchanged. Step
// 2 of the interpolated-60fps ladder and, at this step, a control experiment: the two presents of
// a tick are byte-identical, so any difference between them is the EFB's own history. The full
// contract, and what is deliberately NOT copied, is documented at capture_replay_snapshot in
// common.cpp. Call order per tick:
//   finish() -> capture_replay_snapshot() -> end_frame() -> begin_frame() ->
//   install_replay_snapshot() -> end_frame()
bool replay_present_enabled() noexcept;
bool has_replay_snapshot() noexcept;
// After finish(), before end_frame(): deep-copy the recording packet's passes and shadow its whole
// uniform region into RAM. False (and logged) if no frame is recording.
bool capture_replay_snapshot();
// After begin_frame() on the SECOND packet of the tick: replace that packet's passes with the
// snapshot's, re-push the uniform block at offset 0, and enqueue every pass over this packet's own
// deque and slot. Consumes the snapshot. False (and logged) if there is none.
bool install_replay_snapshot();
// AURORA_INTERP_ALPHA: how far the FIRST of the tick's two presents is displaced back in time.
// 0.5 is the natural value — halfway between the previous tick's pose and this one — and gives an
// even 60 Hz cadence. Negative (the default, meaning unset) disables interpolation entirely while
// leaving the doubled present intact, which is the control the EFB-idempotence check needs.
float interp_alpha() noexcept;
// Between capture_replay_snapshot() and end_frame(): rewrite the recorded frame's matrices toward
// the previous tick's, reading the true values from the snapshot and writing only to staging.
bool interpolate_recorded_frame(float alpha);
uint32_t current_frame() noexcept;
void render_pass(const wgpu::RenderPassEncoder& pass, uint32_t idx);
void after_submit() noexcept;
void gpu_synchronize();
void after_present() noexcept;
float calculate_fps() noexcept;
void resolve_pass(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                  Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat = GX_TF_RGBA8);

struct ColorPassDescriptor {
  const char* label = nullptr;
  wgpu::TextureView colorView;
  wgpu::TextureView resolveView;
  wgpu::TextureView depthStencilView;
  wgpu::Extent3D targetSize;
  uint32_t sampleCount = 1;
  wgpu::LoadOp colorLoadOp = wgpu::LoadOp::Clear;
  wgpu::StoreOp colorStoreOp = wgpu::StoreOp::Store;
  wgpu::Color clearColor{0.f, 0.f, 0.f, 0.f};
  bool hasDepth = false;
  wgpu::LoadOp depthLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp depthStoreOp = wgpu::StoreOp::Undefined;
  float depthClearValue = 0.f;
  bool hasStencil = false;
  wgpu::LoadOp stencilLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp stencilStoreOp = wgpu::StoreOp::Undefined;
  uint32_t stencilClearValue = 0;
  bool observable = true;
};

void begin_color_pass(const ColorPassDescriptor& desc);
void end_color_pass();
void queue_texture_copy(wgpu::TexelCopyTextureInfo src, wgpu::TexelCopyTextureInfo dst, wgpu::Extent3D size);

void begin_offscreen(uint32_t width, uint32_t height);
void end_offscreen();
bool is_offscreen() noexcept;
uint32_t get_sample_count() noexcept;
void clear_caches() noexcept;

namespace tex_palette_conv {
struct ConvRequest;
} // namespace tex_palette_conv
void queue_palette_conv(tex_palette_conv::ConvRequest req);

Range push_verts(const uint8_t* data, size_t length, size_t alignment);
template <typename T>
static Range push_verts(ArrayRef<T> data, size_t alignment) {
  return push_verts(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T), alignment);
}
Range push_indices(const uint8_t* data, size_t length, size_t alignment);
template <typename T>
static Range push_indices(ArrayRef<T> data, size_t alignment) {
  return push_indices(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T), alignment);
}
Range push_uniform(const uint8_t* data, size_t length);
template <typename T>
static Range push_uniform(const T& data) {
  return push_uniform(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
}
Range push_storage(const uint8_t* data, size_t length);
template <typename T>
static Range push_storage(ArrayRef<T> data) {
  return push_storage(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T));
}
template <typename T>
static Range push_storage(const T& data) {
  return push_storage(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
}
Range push_texture_data(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage);

template <typename State>
const State& get_state();
template <typename DrawData>
void push_draw_command(DrawData data);
template <typename DrawData>
DrawData* get_last_draw_command();

template <typename PipelineConfig>
PipelineRef pipeline_ref(const PipelineConfig& config);
bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass);

BindGroupRef bind_group_ref(const WGPUBindGroupDescriptor& descriptor);
wgpu::BindGroup find_bind_group(BindGroupRef id);

wgpu::Sampler sampler_ref(const wgpu::SamplerDescriptor& descriptor);

uint32_t align_uniform(uint32_t value);

Vec2<uint32_t> get_render_target_size() noexcept;
void set_viewport(const Viewport& viewport) noexcept;
void set_scissor(const ClipRect& scissor) noexcept;

void push_debug_group(std::string label);
void insert_debug_marker(std::string label);
} // namespace aurora::gfx
