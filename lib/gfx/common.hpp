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

  // Reserve `size` bytes and hand back a pointer to write them directly, advancing the length
  // immediately — so the caller MUST write all `size` bytes.
  //
  // This exists for index generation, which appends two bytes at a time: a quad was six separate
  // append() calls, each repeating the capacity check, to produce twelve bytes. Taking the tail
  // pointer once turns that into one check plus a straight-line write.
  [[nodiscard]] uint8_t* append_uninitialized(size_t size) {
    resize(m_length + size, false);
    uint8_t* const p = m_data + m_length;
    m_length += size;
    return p;
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

  void clear() { m_length = 0; }

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
        // NAME THE REGION. Without it the message gives only a capacity, and identifying which of
        // verts/uniforms/indices/storage/textureUpload that number belongs to means grepping the
        // constants below — which is how the last one was diagnosed. The mapping is by capacity
        // because that is the only thing a ByteBuffer knows about itself.
        extern const char* aurora_gfx_staging_region_name(size_t capacity); // common.cpp
        std::fprintf(stderr,
                     "[aurora FATAL gfx] mapped ByteBuffer overflow: the %s staging region has %zu bytes "
                     "of its %zu-byte capacity used and needs %zu more -> %zu total. Too small for this "
                     "scene, or a runaway upload — the two look identical here, so check whether the size "
                     "is stable across frames (too small) or climbing (runaway). "
                     "last draw: %s\n",
                     aurora_gfx_staging_region_name(m_capacity), m_length, m_capacity, size - m_length, size,
                     aurora_gfx_last_draw_desc());
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
inline constexpr uint64_t UniformBufferSize = 25165824; // 24mb
inline constexpr uint64_t VertexBufferSize = 3145728;   // 3mb
// INDEX capacity is derived from VERTEX capacity, not picked independently.
//
// It was 1 MB against a 3 MB vertex buffer, and that ratio cannot be right for GX geometry: the
// indices are Uint16 (gx/pipeline.cpp), a GX vertex with indexed attributes costs about 7 bytes,
// and fans/strips expand to roughly 2.3 indices per vertex. Filling the vertex buffer therefore
// needs ~3 MB of indices, three times what was reserved. Pinna Park (stage 13) crashed on exactly
// that: 1,101,044 bytes of indices against a 1,048,576-byte cap — 5% over, stable frame to frame,
// no runaway, just a scene slightly larger than a limit that was never tied to anything.
//
// 4 MB covers a completely full vertex buffer with headroom and stays proportionate if either is
// resized later.
inline constexpr uint64_t IndexBufferSize = 4194304; // 4mb (see above; ~1.4x VertexBufferSize)
// Persistent geometry arena, appended AFTER the staging-mirrored storage region.
//
// The per-frame storage path re-uploads every indexed array every frame. Measured on Delfino:
// 20.44 MB/frame of which 100% is byte-identical to the previous frame, and every array lands at
// the same offset every frame. The obvious shortcut -- skip the staging write and rely on the
// bytes still being there -- is UNSOUND: the staging buffer is Unmap()'d and re-MapAsync'd each
// frame, so its contents are undefined across frames per the WebGPU mapping contract.
//
// So stable arrays live here instead, written once with queue.WriteBuffer and re-written only when
// their content hash changes. This region sits past StorageBufferSize so the staging->GPU copy,
// which maps staging offsets 1:1 onto storage offsets, cannot collide with it. The static bind
// group binds the whole buffer with no dynamic offset and shaders index it by byte offset from a
// uniform, so nothing about the binding changes.
inline constexpr uint64_t PersistentStorageSize = 33554432; // 32mb
inline constexpr uint64_t StorageBufferSize = 50331648;     // 48mb (was 8mb, a
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
inline constexpr uint64_t TextureUploadSize = 25165824; // 24mb

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
using EndFrameCallback = std::function<void(wgpu::CommandEncoder&, const AuroraGpuSubmitInfo&)>;

enum class ShaderType : uint8_t {
  Clear = 0,
  GX = 1,
  Rml = 2,
};

void initialize();
void shutdown();

bool begin_frame();
// Exact ID of the packet currently being recorded, or zero outside a begin/end frame pair.
uint64_t recording_frame_id() noexcept;
void finish();
void end_frame(EndFrameCallback callback);

// Replay present (AURORA_REPLAY_PRESENT=1) — re-present the frame just recorded. The diagnostic
// path emits two byte-identical frames; the host path may retain it for several interpolation
// samples at the display rate. The full
// contract, and what is deliberately NOT copied, is documented at capture_replay_snapshot in
// common.cpp. Call order per tick:
//   finish() -> capture_replay_snapshot() -> end_frame() -> begin_frame() ->
//   install_replay_snapshot() -> end_frame()
bool replay_present_enabled() noexcept;
unsigned replay_presentation_count() noexcept;
bool has_replay_snapshot() noexcept;
// After finish(), before end_frame(): deep-copy the recording packet's passes and shadow its whole
// uniform region into RAM. False (and logged) if no frame is recording.
bool capture_replay_snapshot();
// After begin_frame() on a replay packet: replace that packet's passes with the snapshot's, re-push
// the uniform block at offset 0, and enqueue every pass over this packet's own deque and slot.
// `consume=false` retains the immutable source for another alpha; the final replay consumes it.
bool install_replay_snapshot(bool consume = true);
// AURORA_INTERP_ALPHA: how far the FIRST of the tick's two presents is displaced back in time.
// 0.5 is the natural value — halfway between the previous tick's pose and this one — and gives an
// even 60 Hz cadence. Negative (the default, meaning unset) disables interpolation entirely while
// leaving the doubled present intact, which is the control the EFB-idempotence check needs.
float interp_alpha() noexcept;
// Between capture_replay_snapshot() and end_frame(): rewrite the recorded frame's matrices toward
// the previous tick's, reading the true values from the snapshot and writing only to staging.
bool interpolate_recorded_frame(float alpha, bool resampling = false);
// Tell the next interpolate_recorded_frame that the GAME has declared this tick's camera
// discontinuous, so it must present the tick exactly rather than a halfway pose the game never
// simulated. The host supplies this because aurora cannot tell a cut from fast motion by magnitude
// — the measured camera-step distribution has no gap to put a threshold in.
void snap_next_interpolation();
long snapped_tick_count() noexcept;
// Turn interpolated 60fps on from the host with a single call: doubled present AND the given alpha.
// This is the user-facing path — a player should not have to set two environment variables that
// agree with each other, because a configuration that can be half-set is a configuration that will
// eventually be half-set. AURORA_REPLAY_PRESENT / AURORA_INTERP_ALPHA still work and still win, so
// the diagnostic runs that need the two halves driven independently are unaffected.
void force_interpolation(float alpha);
void set_replay_presentation_count(unsigned count);

// ---- CROSS-FRAME vs INTRA-FRAME EFB COPIES, decided from the frame itself -----------------------
// An EFB copy's DESTINATION says nothing about whether its consumer is this frame or the next, and
// that distinction is the whole question for interpolated 60fps: a copy read by the NEXT frame is
// temporal feedback and must advance once per tick, while a copy read LATER IN THIS FRAME must run
// on both emissions or that read gets nothing. Identifying it by which effect samples it is not
// enough — SMS uses one screen texture for both purposes in different scenes.
//
// The property that does decide it is ORDER: if every sample of a copy's result happens at or
// before the pass that writes it, the samples were reading the PREVIOUS frame's contents.
void note_copy_texture_sampled(const void* dest); // a draw sampled the copy result for `dest`
void note_copy_resolve_dest(const void* dest);    // the pass being resolved copies into `dest`
uint32_t current_frame() noexcept;
void render_pass(const wgpu::RenderPassEncoder& pass, uint32_t idx);
void after_submit() noexcept;
void gpu_synchronize();
void after_present() noexcept;
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
// Identity of an indexed vertex array upload: the EXACT (pointer, size) pair, compared field by
// field. An earlier version folded them into one u64 as `(ptr << 8) ^ size`, which is lossy —
// size's high bits land on pointer bits, so two distinct arrays can collide and the cache then
// serves one array's bytes for another. That is silent geometry corruption, and it bought nothing:
// the map can compare the real fields just as cheaply.
struct ArrayUploadKey {
  const void* data;
  uint32_t size;
  bool operator==(const ArrayUploadKey& rhs) const { return data == rhs.data && size == rhs.size; }
};
struct ArrayUploadKeyHash {
  size_t operator()(const ArrayUploadKey& k) const {
    return std::hash<const void*>{}(k.data) ^ (static_cast<size_t>(k.size) * 0x9E3779B97F4A7C15ull);
  }
};
Range push_storage(const uint8_t* data, size_t length);
// Persistent geometry arena (see PersistentStorageSize above). Uploads only when contentHash
// differs from what the arena already holds for `key`. A returned range of size 0 means the arena
// is full — the caller MUST fall back to push_storage, never treat it as a valid binding.
Range push_storage_persistent(const uint8_t* data, size_t length, ArrayUploadKey key, uint64_t contentHash,
                              bool* outUploaded);
uint64_t persistent_storage_used();
size_t persistent_storage_entries();
void persistent_storage_reset();
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
