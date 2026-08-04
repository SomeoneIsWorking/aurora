#include "common.hpp"
#include "interp.hpp"

#include "clear.hpp"
#include "depth_peek.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"
#include "../gx/pipeline.hpp"
#ifdef AURORA_ENABLE_RMLUI
#include "../rmlui/pipeline.hpp"
#endif
#include "pipeline_cache.hpp"
#include "render_worker.hpp"
#include "tex_copy_conv.hpp"
#include "tex_palette_conv.hpp"
#include "texture_replacement.hpp"
#include "texture.hpp"
#include "../window.hpp"

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <thread>
#include <type_traits>

#include <absl/container/flat_hash_map.h>
#include <magic_enum.hpp>

#include "tracy/Tracy.hpp"

extern "C" void sb_timeline_frame();
extern "C" void sb_timeline_log(const char* fmt, ...);

namespace aurora::gfx {
static Module Log("aurora::gfx");

using webgpu::g_device;
using webgpu::g_instance;
using webgpu::g_queue;

#ifdef AURORA_GFX_DEBUG_GROUPS
std::vector<std::string> g_debugGroupStack;
std::vector<std::string> g_debugMarkers;
#endif

static std::string pass_label(std::string_view kind) {
#ifdef AURORA_GFX_DEBUG_GROUPS
  if (!g_debugGroupStack.empty()) {
    return fmt::format("{} ({})", kind, g_debugGroupStack.back());
  }
#endif
  return std::string{kind};
}

constexpr uint64_t StagingBufferSize = UniformBufferSize + VertexBufferSize + IndexBufferSize + StorageBufferSize +
                                       (UseTextureBuffer ? TextureUploadSize : 0);
// THREE, not two, because of the replay present (AURORA_REPLAY_PRESENT, see
// capture_replay_snapshot below): that path emits TWO packets per game tick, and with only two
// slots the next tick's begin_frame blocks in acquire_frame_slot until the worker has retired the
// oldest frame — destroying the CPU/GPU overlap that the pool exists to provide. The symptom is
// not a hang but a silent one: "60fps mode is SLOWER than 30fps". Each slot costs one staging
// buffer (StagingBufferSize, ~101 MB with the sizes in common.hpp) plus its packet.
constexpr size_t FrameSlotCount = 3;
constexpr size_t StagingBufferCount = FrameSlotCount + 3;

struct StagingHighWater {
  uint32_t verts = 0;
  uint32_t uniforms = 0;
  uint32_t indices = 0;
  uint32_t storage = 0;
  uint32_t textureUpload = 0;
  size_t textureUploadCount = 0;
};

struct ShaderDrawCommand {
  ShaderType type;
  union {
    clear::DrawData clear;
    gx::DrawData gx;
#ifdef AURORA_ENABLE_RMLUI
    rmlui::DrawData rml;
#endif
  };
};
enum class CommandType {
  SetViewport,
  SetScissor,
  Draw,
  DebugMarker,
};
struct Command {
  CommandType type;
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> debugGroupStack;
#endif
  union Data {
    Viewport setViewport;
    ClipRect setScissor;
    ShaderDrawCommand draw;
    size_t debugMarkerIndex;
  } data;
};
} // namespace aurora::gfx

namespace aurora {
// For types that we can't ensure are safe to hash with has_unique_object_representations,
// we create specialized methods to handle them. Note that these are highly dependent on
// the structure definition, which could easily change with Dawn updates.
template <>
inline HashType xxh3_hash(const WGPUBindGroupDescriptor& input, HashType seed) {
  constexpr auto offset = offsetof(WGPUBindGroupDescriptor, layout); // skip nextInChain, label
  const auto hash = xxh3_hash_s(reinterpret_cast<const u8*>(&input) + offset,
                                sizeof(WGPUBindGroupDescriptor) - offset - sizeof(void*) /* skip entries */, seed);
  return xxh3_hash_s(input.entries, sizeof(WGPUBindGroupEntry) * input.entryCount, hash);
}
template <>
inline HashType xxh3_hash(const wgpu::SamplerDescriptor& input, HashType seed) {
  constexpr auto offset = offsetof(wgpu::SamplerDescriptor, addressModeU); // skip nextInChain, label
  return xxh3_hash_s(reinterpret_cast<const u8*>(&input) + offset,
                     sizeof(wgpu::SamplerDescriptor) - offset - 2 /* skip padding */, seed);
}
} // namespace aurora

namespace aurora::gfx {
namespace {
struct CachedBindGroup {
  wgpu::BindGroup bindGroup;
  uint32_t lastUsedFrame = 0;
};

constexpr uint32_t BindGroupCacheRetainFrames = 32;
constexpr uint32_t BindGroupCacheSweepPeriod = 16;
} // namespace

static absl::flat_hash_map<BindGroupRef, CachedBindGroup> g_cachedBindGroups;
static absl::flat_hash_map<SamplerRef, wgpu::Sampler> g_cachedSamplers;
static std::mutex g_bindGroupCacheMutex;
static std::mutex g_samplerCacheMutex;

wgpu::Buffer g_vertexBuffer;
wgpu::Buffer g_uniformBuffer;
wgpu::Buffer g_indexBuffer;
wgpu::Buffer g_storageBuffer;
enum class BufferMapState {
  Unmapped,
  Mapping,
  Mapped,
};
static std::array<wgpu::Buffer, StagingBufferCount> g_stagingBuffers;
static std::array<std::atomic<BufferMapState>, StagingBufferCount> s_mappingStates;
static wgpu::Limits g_cachedLimits;
static uint32_t g_frameIndex = UINT32_MAX;
static PipelineRef g_currentPipeline;
wgpu::BindGroupLayout g_staticBindGroupLayout;
wgpu::BindGroup g_staticBindGroup;
wgpu::BindGroupLayout g_uniformBindGroupLayout;
wgpu::BindGroup g_uniformBindGroup;

// for imgui debug
AuroraStats g_stats{};
uint32_t g_drawCallCount = 0;
uint32_t g_mergedDrawCallCount = 0;

using CommandList = std::vector<Command>;
// SB_PASS_DBG=1: log each render-pass boundary with the finished pass's
// draw count and the new pass's clear flags — makes the per-frame pass
// structure readable (who clears, where the scene draws land).
static void sb_log_pass_boundary(const char* kind);

struct RenderPass {
  std::string label;
  wgpu::TextureView colorView;
  wgpu::TextureView resolveView; // MSAA resolve target; null if msaaSamples == 1
  wgpu::TextureView depthStencilView;
  wgpu::Texture copySourceTexture;
  wgpu::TextureView copySourceView;
  wgpu::TextureView copySourceDepthView;
  wgpu::Extent3D targetSize;
  uint32_t msaaSamples = 1;

  TextureHandle resolveTarget;
  // True when this pass's EFB copy feeds a LATER FRAME rather than a later pass of this one — a
  // temporal feedback texture (SMS's TAfterEffect trail samples the previous frame's copy). Such a
  // copy must happen exactly ONCE per game tick; see interpolate_recorded_frame.
  bool resolveIsFeedback = false;
  GXTexFmt resolveFormat = GX_TF_RGBA8;
  ClipRect resolveRect;
  Range resolveUniformRange;
  Vec4<float> clearColorValue{0.f, 0.f, 0.f, 0.f};
  float clearDepthValue = gx::UseReversedZ ? 0.f : 1.f;
  wgpu::LoadOp colorLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp colorStoreOp = wgpu::StoreOp::Store;
  wgpu::LoadOp depthLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp depthStoreOp = wgpu::StoreOp::Store;
  wgpu::LoadOp stencilLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp stencilStoreOp = wgpu::StoreOp::Undefined;
  uint32_t stencilClearValue = 0;
  CommandList commands;
  bool clearColor = true;
  bool clearDepth = true;
  bool hasDepth = true;
  bool hasStencil = false;
  bool offscreen = false;
  bool observable = false;
  bool captureDepthSnapshot = false;
  bool sealed = false;
  std::vector<tex_palette_conv::ConvRequest> paletteConvs;
};

struct TextureCopy {
  wgpu::TexelCopyTextureInfo src;
  wgpu::TexelCopyTextureInfo dst;
  wgpu::Extent3D size;
};

enum class FrameOpType : uint8_t {
  RenderPass,
  TextureCopy,
};

struct FrameOp {
  FrameOpType type = FrameOpType::RenderPass;
  uint32_t index = 0;
  RenderPass* renderPass = nullptr;
  TextureCopy* textureCopy = nullptr;
  StagingHighWater highWater;
  std::vector<const TextureUpload*> textureUploads;
};

using RenderPassList = std::deque<RenderPass>;
struct FramePacket {
  RenderPassList renderPasses;
  std::deque<TextureCopy> textureCopies;
  std::deque<FrameOp> ops;
  std::deque<TextureUpload> textureUploads;
  ByteBuffer verts;
  ByteBuffer uniforms;
  ByteBuffer indices;
  ByteBuffer storage;
  ByteBuffer textureUpload;
  wgpu::CommandEncoder encoder;
  uint64_t frameId = 0;
  uint32_t frameIndex = 0;
  size_t stagingBuffer = 0;
  StagingHighWater copied;
  AuroraStats stats{};
  // This packet is a replay emission (a re-present of the previous packet's recorded content)
  // rather than a frame the game drew. It must not publish stats, and it must not have recorded
  // any geometry of its own — see install_replay_snapshot / end_frame.
  bool replayEmission = false;
};

static std::array<FramePacket, FrameSlotCount> g_framePackets;
static FramePacket* g_recordingFrame = nullptr;
static size_t g_recordingFrameSlot = 0;
static uint64_t g_nextFrameId = 1;
static render_worker::FrameSlotPool g_frameSlots{FrameSlotCount};
static render_worker::FrameSlotPool g_stagingSlots{StagingBufferCount};
static u32 g_currentRenderPass = UINT32_MAX;
static bool g_inOffscreen = false;
static std::optional<RenderPass> g_suspendedEfbPass;
static Viewport g_suspendedEfbViewport;
static ClipRect g_suspendedEfbScissor;
static webgpu::TextureWithSampler g_offscreenColor;
static webgpu::TextureWithSampler g_offscreenDepth;
static Viewport g_cachedViewport;
static ClipRect g_cachedScissor;

using PresentClock = std::chrono::steady_clock;
static constexpr auto PresentFpsWindow = std::chrono::seconds{1};
static std::mutex g_presentStatsMutex;
static std::deque<PresentClock::time_point> g_presentTimes;
static std::atomic_bool g_processEventsQueued = false;
static std::atomic_int64_t g_lastPresentNs = 0;
static std::atomic_int64_t g_presentPeriodNs = 0;
static std::atomic_int64_t g_cpuFrameTimeNs = 0;
static PresentClock::time_point g_cpuFrameStart;
static constexpr auto FrameStartSafetyMargin = std::chrono::milliseconds{2};
static constexpr auto MaxPacingSample = std::chrono::milliseconds{250};
static constexpr uint32_t PacingEmaWeight = 8;

static int64_t timestamp_ns(PresentClock::time_point time) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

static int64_t duration_ns(PresentClock::duration duration) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

static void update_ema(std::atomic_int64_t& value, int64_t sample) {
  if (sample <= 0 || sample > duration_ns(MaxPacingSample)) {
    return;
  }

  int64_t current = value.load(std::memory_order_acquire);
  while (true) {
    const int64_t next = current == 0 ? sample : current + (sample - current) / static_cast<int64_t>(PacingEmaWeight);
    if (value.compare_exchange_weak(current, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return;
    }
  }
}

static void prune_present_times(PresentClock::time_point now) {
  while (!g_presentTimes.empty() && g_presentTimes.front() + PresentFpsWindow < now) {
    g_presentTimes.pop_front();
  }
}

static void process_events() {
  ZoneScopedN("ProcessEvents");
  if (g_instance) {
    g_instance.ProcessEvents();
  }
}

static void enqueue_process_events() {
  if (render_worker::is_worker_thread()) {
    process_events();
    return;
  }

  bool expected = false;
  if (!g_processEventsQueued.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
    return;
  }

  render_worker::enqueue_work([] {
    process_events();
    g_processEventsQueued.store(false, std::memory_order_release);
  });
}

static void wait_for_gpu_progress(std::chrono::nanoseconds sleepDuration) {
  if (render_worker::is_idle()) {
    enqueue_process_events();
  }
  std::this_thread::sleep_for(sleepDuration);
}

static void pace_frame_start() {
  ZoneScopedN("Frame start pacing");
  if (g_frameSlots.free_count() == FrameSlotCount) {
    return;
  }

  const int64_t lastPresentNs = g_lastPresentNs.load(std::memory_order_acquire);
  const int64_t presentPeriodNs = g_presentPeriodNs.load(std::memory_order_acquire);
  const int64_t cpuFrameTimeNs = g_cpuFrameTimeNs.load(std::memory_order_acquire);
  if (lastPresentNs == 0 || presentPeriodNs == 0 || cpuFrameTimeNs == 0) {
    return;
  }

  const int64_t safetyMarginNs = duration_ns(FrameStartSafetyMargin);
  const int64_t targetStartNs = lastPresentNs + presentPeriodNs - cpuFrameTimeNs - safetyMarginNs;
  int64_t nowNs = timestamp_ns(PresentClock::now());
  if (targetStartNs <= nowNs) {
    return;
  }

  const double initialWaitMs = static_cast<double>(targetStartNs - nowNs) / 1'000'000.0;
  TracyPlot("aurora: frameStartPaceWaitMs", initialWaitMs);
  while (nowNs < targetStartNs) {
    const int64_t remainingNs = targetStartNs - nowNs;
    const auto sleepDuration = remainingNs > 1'000'000 ? std::chrono::milliseconds{1}
                                                       : std::chrono::nanoseconds{remainingNs};
    wait_for_gpu_progress(sleepDuration);
    nowNs = timestamp_ns(PresentClock::now());
  }
}

static void map_staging_buffer(size_t slot, bool releaseSlotOnCompletion = false) {
  auto expected = BufferMapState::Unmapped;
  if (!s_mappingStates[slot].compare_exchange_strong(expected, BufferMapState::Mapping, std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
    return;
  }

  g_stagingBuffers[slot].MapAsync(
      wgpu::MapMode::Write, 0, StagingBufferSize, wgpu::CallbackMode::AllowSpontaneous,
      [slot, releaseSlotOnCompletion](wgpu::MapAsyncStatus status, wgpu::StringView message) {
        if (status == wgpu::MapAsyncStatus::CallbackCancelled || status == wgpu::MapAsyncStatus::Aborted) {
          Log.warn("Buffer mapping {}: {}", magic_enum::enum_name(status), message);
          s_mappingStates[slot].store(BufferMapState::Unmapped, std::memory_order_release);
          if (releaseSlotOnCompletion) {
            g_stagingSlots.release(slot);
          }
          return;
        }
        ASSERT(status == wgpu::MapAsyncStatus::Success, "Buffer mapping failed: {} {}", magic_enum::enum_name(status),
               message);
        s_mappingStates[slot].store(BufferMapState::Mapped, std::memory_order_release);
        if (releaseSlotOnCompletion) {
          g_stagingSlots.release(slot);
        }
      });
}

static void set_efb_targets(RenderPass& pass) {
  pass.colorView = webgpu::g_frameBuffer.view;
  pass.resolveView = webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_frameBufferResolved.view : nullptr;
  pass.depthStencilView = webgpu::g_depthBuffer.view;
  pass.copySourceTexture =
      webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_frameBufferResolved.texture : webgpu::g_frameBuffer.texture;
  pass.copySourceView =
      webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_frameBufferResolved.view : webgpu::g_frameBuffer.view;
  pass.copySourceDepthView = webgpu::g_depthBuffer.view;
  pass.targetSize = webgpu::g_frameBuffer.size;
  pass.msaaSamples = webgpu::g_graphicsConfig.msaaSamples;
  pass.hasDepth = true;
  pass.hasStencil = false;
}

struct OffscreenCacheKey {
  uint32_t width;
  uint32_t height;

  bool operator==(const OffscreenCacheKey& rhs) const { return width == rhs.width && height == rhs.height; }
  template <typename H>
  friend H AbslHashValue(H h, const OffscreenCacheKey& key) {
    return H::combine(std::move(h), key.width, key.height);
  }
};
struct OffscreenCacheEntry {
  webgpu::TextureWithSampler color;
  webgpu::TextureWithSampler depth;
};
static absl::flat_hash_map<OffscreenCacheKey, OffscreenCacheEntry> g_offscreenCache;

static FramePacket& current_frame_packet() {
  CHECK(g_recordingFrame != nullptr, "No active frame packet");
  return *g_recordingFrame;
}

static RenderPassList& current_render_passes() { return current_frame_packet().renderPasses; }

static void sb_log_pass_boundary(const char* kind) {
  static int s_dbg = -1;
  if (s_dbg < 0) {
    const char* e = std::getenv("SB_PASS_DBG");
    s_dbg = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
  }
  if (s_dbg == 0) {
    return;
  }
  static long n = 0;
  ++n;
  if (g_currentRenderPass != UINT32_MAX && g_currentRenderPass < current_render_passes().size()) {
    const auto& p = current_render_passes()[g_currentRenderPass];
    std::fprintf(stderr, "[pass] n=%ld end '%s' cmds=%zu clearC=%d clearD=%d clr=(%.2f,%.2f,%.2f) -> new %s\n", n,
                 p.label.c_str(), p.commands.size(), p.clearColor ? 1 : 0, p.clearDepth ? 1 : 0,
                 p.clearColorValue.x(), p.clearColorValue.y(), p.clearColorValue.z(), kind);
  } else {
    std::fprintf(stderr, "[pass] n=%ld begin frame -> new %s\n", n, kind);
  }
}

static StagingHighWater current_high_water(const FramePacket& frame) noexcept {
  return {
      .verts = static_cast<uint32_t>(frame.verts.size()),
      .uniforms = static_cast<uint32_t>(frame.uniforms.size()),
      .indices = static_cast<uint32_t>(frame.indices.size()),
      .storage = static_cast<uint32_t>(frame.storage.size()),
      .textureUpload = static_cast<uint32_t>(frame.textureUpload.size()),
      .textureUploadCount = frame.textureUploads.size(),
  };
}

static FrameOp capture_frame_op(FramePacket& frame, FrameOpType type, uint32_t index) {
  FrameOp op{
      .type = type,
      .index = index,
      .renderPass =
          type == FrameOpType::RenderPass && index < frame.renderPasses.size() ? &frame.renderPasses[index] : nullptr,
      .textureCopy = type == FrameOpType::TextureCopy && index < frame.textureCopies.size()
                         ? &frame.textureCopies[index]
                         : nullptr,
      .highWater = current_high_water(frame),
  };
  op.textureUploads.reserve(op.highWater.textureUploadCount);
  for (size_t i = 0; i < op.highWater.textureUploadCount; ++i) {
    op.textureUploads.push_back(&frame.textureUploads[i]);
  }
  return op;
}

static void seal_pass(FramePacket& frame, uint32_t passIndex) {
  if (passIndex >= frame.renderPasses.size()) {
    return;
  }
  auto& pass = frame.renderPasses[passIndex];
  if (pass.sealed) {
    return;
  }
  pass.sealed = true;
}

static void encode_op(wgpu::CommandEncoder& cmd, FramePacket& frame, const FrameOp& op);
static void render(wgpu::CommandEncoder& cmd, FramePacket& frame, RenderPass& passInfo, uint32_t passIndex);
static void render_pass(const wgpu::RenderPassEncoder& pass, FramePacket& frame, const RenderPass& passInfo);
static void expire_cached_bind_groups();
static void push_command(CommandType type, const Command::Data& data);

static void enqueue_op(FramePacket& frame, size_t frameSlot, uint32_t opIndex) {
  if (opIndex >= frame.ops.size()) {
    return;
  }
  auto op = frame.ops[opIndex];
  render_worker::enqueue_encode_pass(frame.frameId, opIndex, [frameSlot, op = std::move(op)] {
    if (op.renderPass == nullptr && op.textureCopy == nullptr) {
      return;
    }
    auto& packet = g_framePackets[frameSlot];
    encode_op(packet.encoder, packet, op);
  });
}

static void enqueue_pass(FramePacket& frame, size_t frameSlot, uint32_t passIndex) {
  seal_pass(frame, passIndex);
  const auto opIndex = static_cast<uint32_t>(frame.ops.size());
  frame.ops.emplace_back(capture_frame_op(frame, FrameOpType::RenderPass, passIndex));
  enqueue_op(frame, frameSlot, opIndex);
}

// ---------------------------------------------------------------------------
// Replay present (AURORA_REPLAY_PRESENT=1): present one recorded frame TWICE.
//
// Step 2 of the interpolated-60fps ladder, and deliberately a CONTROL rather than a feature: both
// presents of a tick carry byte-identical content, so anything that differs between them is the
// EFB's own history and not the replay. Pass 0 is LoadOp::Load with no eager clear (see the
// LAZY/NO EAGER CLEAR note in begin_frame), so a replayed frame starts from the EFB as the first
// emission LEFT it rather than as it FOUND it. Whether that is observable cannot be settled by
// reading the code; this path is how it gets measured.
//
// What is copied, and what deliberately is NOT:
//
//   * renderPasses, with their CommandLists, are deep-copied. RenderPass's members are a string,
//     refcounted wgpu views/textures, a shared_ptr TextureHandle and a vector of palette convs —
//     all copyable, all refcount-SHARING, so the copy aliases exactly the same GPU objects.
//
//   * the WHOLE uniform region [0, uniforms.size()) is shadowed into cacheable RAM and re-pushed
//     at offset 0 of the replay packet, so every uniformRange.offset recorded in the copied
//     commands still resolves to the same bytes. It has to be the whole region and not per-draw
//     blocks: resolve_pass pushes its UV transform (resolveUniformRange) into the same buffer,
//     and tex_copy_conv reads it back at encode time.
//
//   * verts / indices / storage are NOT re-pushed. DrawData.vertRange is never read at encode
//     time — vertex and storage data are reached through g_staticBindGroup, built once at init
//     over the whole global buffers — and the global buffers still hold the first emission's
//     bytes because nothing writes them in between. Pushing nothing also skips their staging
//     copies automatically: needs_staging_copy gates on highWater > copied, and with nothing
//     pushed both are 0. Do NOT "help" that along by pre-seeding frame.copied — a later push
//     would then satisfy highWater <= copied, emit no copy at all, and silently draw the
//     PREVIOUS frame's bytes. The zero-length assert in end_frame is the guard for the day this
//     path stops being a pure copy.
//
//   * FrameOps are NOT copied. A FrameOp holds raw pointers into its own packet's renderPasses
//     deque and is resolved through its frame SLOT; the original's pointers die the moment its
//     end_frame does `packet = {}`. The replay re-runs capture_frame_op/enqueue_pass over its own
//     deque and its own slot instead.
// ---------------------------------------------------------------------------
struct ReplaySnapshot {
  RenderPassList renderPasses;
  std::vector<uint8_t> uniforms;
  bool valid = false;
};
static ReplaySnapshot g_replaySnapshot;

namespace {
// Host-set overrides for the two interpolation knobs. They exist so the host can turn the WHOLE
// feature on with one switch: this is a user-facing mode, and requiring a player to set three
// environment variables in agreement is a way to ship a broken configuration, not a knob. The env
// vars still work and still win, because the diagnostic paths (the EFB-idempotence control, the A/B
// runs) need to drive the two halves independently.
bool g_replayForced = false;
float g_alphaForced = -1.0f;
} // namespace

// This tick's uniform bytes in ordinary cached RAM, mirrored at the same offsets as the GPU staging
// buffer they were written to. See push_uniform for why this exists rather than reading the staging
// back. Grows to the high-water mark of a tick and then stops reallocating.
std::vector<uint8_t> g_uniformShadow;
size_t g_uniformShadowSize = 0;

void force_interpolation(float alpha) {
  g_replayForced = true;
  g_alphaForced = alpha;
}

bool replay_present_enabled() noexcept {
  static const bool s_env = [] {
    const char* e = std::getenv("AURORA_REPLAY_PRESENT");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return s_env || g_replayForced;
}

bool has_replay_snapshot() noexcept { return g_replaySnapshot.valid; }

float interp_alpha() noexcept {
  // Read once, same convention as replay_present_enabled above. Negative means unset: the doubled
  // present still happens, but both emissions show the tick exactly. Keeping that as the DEFAULT
  // rather than folding it into the replay switch is what makes the EFB-idempotence control
  // reachable at all — with interpolation on, two presents differ for two reasons at once and the
  // control can no longer distinguish them.
  static const float s_alpha = [] {
    const char* e = std::getenv("AURORA_INTERP_ALPHA");
    if (e == nullptr || e[0] == '\0') {
      return -1.0f;
    }
    const float v = std::strtof(e, nullptr);
    if (v < 0.0f || v > 1.0f) {
      Log.error("AURORA_INTERP_ALPHA={} is outside [0,1]; interpolation stays OFF. An alpha outside "
                "that range extrapolates rather than interpolates, which is a different feature "
                "with different failure modes, not a tuning value.",
                e);
      return -1.0f;
    }
    return v;
  }();
  // Env wins, so every existing diagnostic run keeps behaving exactly as it did.
  return s_alpha >= 0.0f ? s_alpha : g_alphaForced;
}

namespace {
// Set by the host when the GAME declares this tick's camera discontinuous (see the recomp's
// camera_cut.cpp). Consumed by the next interpolate_recorded_frame, which then forces alpha 1 —
// both emissions show the tick exactly, which is a snap.
bool g_snapNextTick = false;
long g_snappedTicks = 0;
} // namespace

void snap_next_interpolation() { g_snapNextTick = true; }

namespace {
// The destination of the copy that feeds a temporal-feedback effect, supplied by the host because
// aurora cannot tell it apart from any other EFB copy: structurally they are identical, and the
// difference is only in WHO reads the result and WHEN.
const void* g_feedbackCopyDest = nullptr;
bool g_nextResolveIsFeedback = false;
} // namespace

void set_feedback_copy_dest(const void* dest) { g_feedbackCopyDest = dest; }
bool is_feedback_copy_dest(const void* dest) {
  return dest != nullptr && dest == g_feedbackCopyDest;
}
void mark_next_resolve_feedback(bool feedback) { g_nextResolveIsFeedback = feedback; }

long snapped_tick_count() noexcept { return g_snappedTicks; }

bool capture_replay_snapshot() {
  ZoneScoped;
  // Command::Data is a UNION: a member with a non-trivial copy ctor or destructor (rmlui::DrawData
  // arrives under AURORA_ENABLE_RMLUI) makes the CommandList copy below silently wrong rather than
  // merely slow, because the union copies as bytes and nothing runs the member's own copy. Break
  // the build instead of the frame.
  static_assert(std::is_trivially_copyable_v<Command::Data>,
                "Command::Data is a union; a non-trivially-copyable member makes the replay's "
                "CommandList copy wrong. Give Command a real copy constructor before enabling it.");
#if !defined(AURORA_GFX_DEBUG_GROUPS)
  // Command itself is trivially copyable only without the debug-group string vector; that vector
  // copies correctly (it is not in the union), it is just not free.
  static_assert(std::is_trivially_copyable_v<Command>, "Command is expected to be a POD command record");
#endif
  if (g_recordingFrame == nullptr) {
    Log.error("capture_replay_snapshot: no frame is recording; nothing to snapshot");
    return false;
  }
  auto& frame = *g_recordingFrame;
  // Sub-timers, because "the snapshot costs 13 ms" does not say WHICH half: the pass-list copy and
  // the uniform copy have nothing in common and would be fixed differently.
  static const bool s_timeIt = [] {
    const char* e = std::getenv("AURORA_REPLAY_PROFILE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  const auto tStart = s_timeIt ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  g_replaySnapshot.renderPasses = frame.renderPasses;
  const auto tPasses = s_timeIt ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  // Read from the RAM shadow, never from frame.uniforms — that is a view over write-mapped GPU
  // staging, which is write-combined, and reading ~2.85 MB of it back per tick is what made this
  // feature run the game in slow motion. push_uniform mirrors every write here at the same offset,
  // so this is a plain cached copy.
  //
  // The sizes must agree. If they do not, the shadow missed writes and the replay would present a
  // frame built from stale uniforms — geometry from this tick with some other tick's transforms,
  // which looks like a render bug rather than a bookkeeping one. Refuse loudly instead.
  if (g_uniformShadowSize != frame.uniforms.size()) {
    Log.error("uniform shadow is {} bytes but the frame recorded {} — some uniform write bypassed "
              "push_uniform. Refusing to snapshot: replaying from a short shadow would present this "
              "tick's geometry with stale transforms and read as a render defect.",
              g_uniformShadowSize, frame.uniforms.size());
    return false;
  }
  g_replaySnapshot.uniforms.resize(g_uniformShadowSize);
  if (g_uniformShadowSize != 0) {
    memcpy(g_replaySnapshot.uniforms.data(), g_uniformShadow.data(), g_uniformShadowSize);
  }
  g_replaySnapshot.valid = true;
  if (s_timeIt) {
    const auto tEnd = std::chrono::steady_clock::now();
    auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
    static double accPasses = 0, accUniforms = 0;
    static long n = 0;
    static size_t cmds = 0;
    accPasses += us(tStart, tPasses);
    accUniforms += us(tPasses, tEnd);
    size_t c = 0;
    for (const auto& p : g_replaySnapshot.renderPasses) {
      c += p.commands.size();
    }
    cmds += c;
    if (++n % 200 == 0) {
      Log.info("replay snapshot avg over {} ticks: pass-list copy {:.0f} us ({} commands), uniform "
               "copy {:.0f} us ({} B)",
               n, accPasses / (double)n, cmds / n, accUniforms / (double)n,
               g_replaySnapshot.uniforms.size());
    }
  }
  return true;
}

// Rewrite the RECORDED frame's matrices to an in-between pose, leaving the snapshot — and therefore
// the replay emission — carrying the tick's true state. The pair then presents (t-1+alpha) followed
// by (t).
//
// Order matters and is not interchangeable: this must run AFTER capture_replay_snapshot, because
// the snapshot is where the true matrices are read from, and BEFORE end_frame, because that is when
// the staging is unmapped. Called with the snapshot as the source and the live staging as the
// destination, so no read ever touches write-combined memory.
bool interpolate_recorded_frame(float alpha) {
  ZoneScoped;
  if (g_recordingFrame == nullptr) {
    Log.error("interpolate_recorded_frame: no frame is recording");
    return false;
  }
  if (!g_replaySnapshot.valid) {
    Log.error("interpolate_recorded_frame: no snapshot, so there is nothing to read the true "
              "matrices from. Interpolating from the staging itself would read back what this "
              "function is about to overwrite.");
    return false;
  }
  // The attribution numbers in interp::report() are only worth reading if the discriminator behind
  // them actually discriminates. Prove that once, on synthetic input, before it is ever pointed at
  // the game — a self-test that nobody runs is the same bug one level up.
  static const bool s_selftestOk = interp::selftest();
  (void)s_selftestOk;
  // A tick the game declared discontinuous has no meaningful in-between: the halfway pose is a
  // viewpoint it never simulated. Force alpha 1 rather than skipping the pass, so the pairing table
  // is still filled from this tick and the tick AFTER the cut can interpolate normally — skipping
  // would leave the table holding the pre-cut pose and move the artefact one frame later.
  const bool snapping = g_snapNextTick;
  if (snapping) {
    g_snapNextTick = false;
    ++g_snappedTicks;
    alpha = 1.0f;
  }
  auto& frame = *g_recordingFrame;
  const auto& snap = g_replaySnapshot.uniforms;

  // A TEMPORAL FEEDBACK copy must happen exactly ONCE per game tick, so drop it from THIS emission
  // and leave it to the replay emission, which carries the tick's true image.
  //
  // Both emissions replay the same recorded passes, so without this the feedback texture is written
  // twice per tick from two different images. The visible consequence in SMS: the dash afterimage
  // (TAfterEffect, MarioUtil/ScreenUtil.cpp) composites the previous frame's EFB copy back over the
  // viewport, slightly scaled, to make a motion trail. Written twice, the replay emission's trail
  // samples the INTERPOLATED image this same tick wrote a moment earlier rather than the previous
  // tick's — so the trail is full length on one present and half on the next, which reads as jitter.
  //
  // This runs AFTER capture_replay_snapshot, so the snapshot (and therefore the replay emission)
  // still holds the resolve. Clearing the handle is how a pass is told not to resolve; every other
  // EFB copy is untouched, because an INTRA-frame copy — the sea reflection, an indirect pass —
  // must still run on both emissions, since a later pass of each emission samples what that
  // emission wrote.
  {
    // Counted and reported, because a silent zero here is the failure mode: if the host never
    // identifies the feedback destination, or the pointer it supplies does not match the one the
    // copy carries, nothing is suppressed and the trail keeps jittering with no indication why.
    static long s_suppressed = 0, s_ticks = 0;
    long thisTick = 0;
    for (auto& pass : frame.renderPasses) {
      if (pass.resolveIsFeedback && pass.resolveTarget) {
        pass.resolveTarget = {};
        ++thisTick;
      }
    }
    s_suppressed += thisTick;
    if ((++s_ticks % 300) == 0) {
      Log.info("feedback copies suppressed on the interpolated emission: {} over {} ticks ({} this "
               "tick). ZERO means the host never named a feedback destination, or named one that no "
               "copy matches — not that the scene has no feedback effect.",
               s_suppressed, s_ticks, thisTick);
    }
  }

  interp::begin_tick();
  // The camera delta is computed ONCE for the tick and applied to every draw that could not be
  // paired. Without it, unpaired draws render from the current viewpoint while paired ones sit at
  // the in-between one, and the frame is drawn from two viewpoints at once — measured as worse than
  // not interpolating at all.
  interp::begin_camera_delta(alpha);
  if (snapping) {
    // Printed with the tick index so the game's declared cut can be lined up against the per-tick
    // camera measurements in interp::report(). If the snapped ticks do not coincide with the ticks
    // that measured a large camera step, then either the signal is firing on non-cuts or it is
    // missing real ones — and a snap count alone could not tell you which.
    Log.info("tick {}: SNAPPED (alpha forced to 1) — the game declared the camera discontinuous, so "
             "this tick has no in-between to show",
             interp::tick_index());
  }
  for (const auto& pass : frame.renderPasses) {
    for (const auto& cmd : pass.commands) {
      if (cmd.type != CommandType::Draw || cmd.data.draw.type != ShaderType::GX) {
        continue;
      }
      const gx::DrawData& d = cmd.data.draw.gx;
      if (d.uniformRange.offset + d.uniformRange.size > snap.size()) {
        continue;   // outside the snapshot: cannot have been recorded by this frame
      }
      uint8_t* dst = frame.uniforms.data() + d.uniformRange.offset;
      // Every draw ends up on the interpolated viewpoint, one way or the other. A draw that was
      // genuinely paired already carries it, because the camera is baked into the matrices being
      // lerped. Everything else — untagged, or tagged but unpaired this tick — takes the camera
      // delta alone. Leaving ANY draw on the current viewpoint is what tears the frame.
      if (!interp::patch_draw(d.tag, d.vtxCount, snap.data() + d.uniformRange.offset, dst,
                              d.uniformRange.size, d.mtxPosOffset, d.mtxNrmOffset, alpha) &&
          d.ortho == 0) {
        // Perspective only. An orthographic draw's matrix is not model x view, so a camera delta
        // does not belong in it — it would slide the HUD bodily every other frame.
        interp::patch_camera_only(snap.data() + d.uniformRange.offset, dst, d.uniformRange.size,
                                  d.mtxPosOffset, d.mtxNrmOffset);
      }
    }
  }
  interp::end_tick();
  // Pairing coverage on a slow cadence. Without it, "interpolation is on" and "interpolation is on
  // and pairing nothing, so every object snaps" produce the same smooth-looking log and the same
  // doubled present count.
  static long s_ticks = 0;
  if ((++s_ticks % 300) == 0) {
    interp::report();
  }
  return true;
}

bool install_replay_snapshot() {
  ZoneScoped;
  if (!g_replaySnapshot.valid) {
    Log.error("install_replay_snapshot: no snapshot has been captured; the replay frame would present "
              "whatever the EFB happens to hold");
    return false;
  }
  if (g_recordingFrame == nullptr) {
    Log.error("install_replay_snapshot: no frame is recording; call begin_frame first");
    return false;
  }
  auto& frame = *g_recordingFrame;
  ASSERT(frame.uniforms.size() == 0,
         "Replay packet already holds {} uniform bytes; the snapshot must land at offset 0 or every "
         "copied uniformRange.offset is wrong",
         frame.uniforms.size());
  // Discard the passes begin_frame() created (a fresh EFB pass carrying only its viewport/scissor
  // commands) — the replay's content is the snapshot, entirely.
  frame.renderPasses = std::move(g_replaySnapshot.renderPasses);
  const size_t uniformSize = g_replaySnapshot.uniforms.size();
  frame.uniforms.append(g_replaySnapshot.uniforms.data(), uniformSize);
  ASSERT(frame.uniforms.size() == uniformSize, "Replay uniform block landed at {} bytes, expected {}",
         frame.uniforms.size(), uniformSize);
  // CONSUME the snapshot. A tick whose capture failed must fall back to presenting once, never to
  // re-presenting a stale frame — a stale replay is invisible on a static scene and looks like a
  // one-frame stutter on a moving one.
  g_replaySnapshot = {};
  frame.replayEmission = true;
  // Nothing further may be recorded into this packet: finish() must find no open pass, because
  // enqueueing a pass a second time would encode it twice.
  g_currentRenderPass = UINT32_MAX;
  for (uint32_t i = 0; i < frame.renderPasses.size(); ++i) {
    // Suppress the depth snapshot on the replay emission. finish() set captureDepthSnapshot on the
    // last pass of the ORIGINAL frame and depth_peek rate-limits the request, so whichever emission
    // reached it first would consume it — meaning the game could read back depth belonging to a
    // frame state it never simulated.
    frame.renderPasses[i].captureDepthSnapshot = false;
    enqueue_pass(frame, g_recordingFrameSlot, i);
  }
  return true;
}

void queue_texture_upload(TextureUpload upload) {
  if (g_currentRenderPass != UINT32_MAX) {
    ASSERT(!current_render_passes()[g_currentRenderPass].sealed,
           "Attempted to append texture upload to sealed render pass {}", g_currentRenderPass);
  }
  current_frame_packet().textureUploads.emplace_back(std::move(upload));
}

void queue_texture_upload_data(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage,
                               wgpu::TexelCopyTextureInfo tex, wgpu::Extent3D size) {
  const auto copyBytesPerRow = AURORA_ALIGN(bytesPerRow, 256);
  auto& frame = current_frame_packet();
  if (frame.textureUpload.size() + copyBytesPerRow * rowsPerImage <= TextureUploadSize) {
    const auto range = push_texture_data(data, bytesPerRow, rowsPerImage);
    const wgpu::TexelCopyBufferLayout layout{
        .offset = range.offset,
        .bytesPerRow = bytesPerRow,
        .rowsPerImage = rowsPerImage,
    };
    queue_texture_upload(TextureUpload{layout, std::move(tex), size});
    return;
  }

  const uint64_t uploadSize = copyBytesPerRow * rowsPerImage;
  const wgpu::BufferDescriptor descriptor{
      .label = "Overflow Texture Upload Buffer",
      .usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc,
      .size = uploadSize,
      .mappedAtCreation = true,
  };
  auto buffer = g_device.CreateBuffer(&descriptor);
  auto* dst = static_cast<uint8_t*>(buffer.GetMappedRange(0, uploadSize));
  for (uint32_t row = 0; row < rowsPerImage; ++row) {
    memcpy(dst, data, bytesPerRow);
    data += bytesPerRow;
    dst += copyBytesPerRow;
  }
  buffer.Unmap();

  const wgpu::TexelCopyBufferLayout layout{
      .offset = 0,
      .bytesPerRow = bytesPerRow,
      .rowsPerImage = rowsPerImage,
  };
  queue_texture_upload(TextureUpload{layout, std::move(tex), size, std::move(buffer)});
}

void queue_texture_copy(wgpu::TexelCopyTextureInfo src, wgpu::TexelCopyTextureInfo dst, wgpu::Extent3D size) {
  ZoneScoped;
  auto& frame = current_frame_packet();
  if (g_currentRenderPass != UINT32_MAX) {
    enqueue_pass(frame, g_recordingFrameSlot, g_currentRenderPass);
    g_currentRenderPass = UINT32_MAX;
  }

  const auto copyIndex = static_cast<uint32_t>(frame.textureCopies.size());
  frame.textureCopies.emplace_back(TextureCopy{
      .src = std::move(src),
      .dst = std::move(dst),
      .size = size,
  });
  const auto opIndex = static_cast<uint32_t>(frame.ops.size());
  frame.ops.emplace_back(capture_frame_op(frame, FrameOpType::TextureCopy, copyIndex));
  enqueue_op(frame, g_recordingFrameSlot, opIndex);
}

void begin_color_pass(const ColorPassDescriptor& desc) {
  ZoneScoped;
  auto& frame = current_frame_packet();
  if (g_currentRenderPass != UINT32_MAX) {
    enqueue_pass(frame, g_recordingFrameSlot, g_currentRenderPass);
  }
  sb_log_pass_boundary("color");

  RenderPass pass{
      .label = desc.label != nullptr ? desc.label : "",
      .colorView = desc.colorView,
      .resolveView = desc.resolveView,
      .depthStencilView = desc.depthStencilView,
      .targetSize = desc.targetSize,
      .msaaSamples = desc.sampleCount,
      .clearColorValue =
          {
              static_cast<float>(desc.clearColor.r),
              static_cast<float>(desc.clearColor.g),
              static_cast<float>(desc.clearColor.b),
              static_cast<float>(desc.clearColor.a),
          },
      .clearDepthValue = desc.depthClearValue,
      .colorLoadOp = desc.colorLoadOp,
      .colorStoreOp = desc.colorStoreOp,
      .depthLoadOp = desc.depthLoadOp,
      .depthStoreOp = desc.depthStoreOp,
      .stencilLoadOp = desc.stencilLoadOp,
      .stencilStoreOp = desc.stencilStoreOp,
      .stencilClearValue = desc.stencilClearValue,
      .clearColor = desc.colorLoadOp == wgpu::LoadOp::Clear,
      .clearDepth = desc.depthLoadOp == wgpu::LoadOp::Clear,
      .hasDepth = desc.hasDepth,
      .hasStencil = desc.hasStencil,
      .observable = desc.observable,
  };
  pass.commands.reserve(128);
  frame.renderPasses.emplace_back(std::move(pass));
  g_currentRenderPass = static_cast<uint32_t>(frame.renderPasses.size() - 1);

  g_cachedViewport = {0.f, 0.f, static_cast<float>(desc.targetSize.width), static_cast<float>(desc.targetSize.height),
                      0.f, 1.f};
  g_cachedScissor = {0, 0, static_cast<int32_t>(desc.targetSize.width), static_cast<int32_t>(desc.targetSize.height)};
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
}

void end_color_pass() {
  ZoneScoped;
  if (g_currentRenderPass == UINT32_MAX) {
    return;
  }
  enqueue_pass(current_frame_packet(), g_recordingFrameSlot, g_currentRenderPass);
  g_currentRenderPass = UINT32_MAX;
}

static inline void push_command(CommandType type, const Command::Data& data) {
  if (g_currentRenderPass == UINT32_MAX)
    UNLIKELY {
      Log.warn("Dropping command {}", magic_enum::enum_name(type));
      return;
    }
  auto& renderPass = current_render_passes()[g_currentRenderPass];
  ASSERT(!renderPass.sealed, "Attempted to append command {} to sealed render pass {}", magic_enum::enum_name(type),
         g_currentRenderPass);
  renderPass.commands.push_back({
      .type = type,
#ifdef AURORA_GFX_DEBUG_GROUPS
      .debugGroupStack = g_debugGroupStack,
#endif
      .data = data,
  });
}

template <>
gx::DrawData* get_last_draw_command() {
  if (g_currentRenderPass >= current_render_passes().size()) {
    return nullptr;
  }
  auto& last = current_render_passes()[g_currentRenderPass].commands.back();
  if (last.type != CommandType::Draw || last.data.draw.type != ShaderType::GX) {
    return nullptr;
  }
  return &last.data.draw.gx;
}

static void push_draw_command(ShaderDrawCommand data) {
  push_command(CommandType::Draw, Command::Data{.draw = data});
  ++g_drawCallCount;
}

Vec2<uint32_t> get_render_target_size() noexcept {
  if (g_currentRenderPass < current_render_passes().size()) {
    const auto& size = current_render_passes()[g_currentRenderPass].targetSize;
    return {size.width, size.height};
  }
  const auto windowSize = window::get_window_size();
  return {windowSize.fb_width, windowSize.fb_height};
}

void set_viewport(const Viewport& cmd) noexcept {
  if (cmd != g_cachedViewport) {
    push_command(CommandType::SetViewport, Command::Data{.setViewport = cmd});
    g_cachedViewport = cmd;
  }
}

void set_scissor(const ClipRect& cmd) noexcept {
  if (cmd != g_cachedScissor) {
    push_command(CommandType::SetScissor, Command::Data{.setScissor = cmd});
    g_cachedScissor = cmd;
  }
}

template <>
void push_draw_command(clear::DrawData data) {
  push_draw_command(ShaderDrawCommand{.type = ShaderType::Clear, .clear = data});
}

template <>
PipelineRef pipeline_ref(const clear::PipelineConfig& config) {
  return find_pipeline(ShaderType::Clear, config, [=] { return create_pipeline(config); });
}

void resolve_pass(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                  Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat) {
  sb_log_pass_boundary("resolve");
  // Resolve current render pass
  auto& prevPass = current_render_passes()[g_currentRenderPass];
  prevPass.resolveTarget = std::move(texture);
  prevPass.resolveIsFeedback = g_nextResolveIsFeedback;
  g_nextResolveIsFeedback = false;
  prevPass.resolveRect = rect;
  prevPass.resolveFormat = resolveFormat;
  // Push UV transform uniform for tex_copy_conv (crop region in UV space)
  const auto srcW = static_cast<float>(prevPass.targetSize.width);
  const auto srcH = static_cast<float>(prevPass.targetSize.height);
  const std::array uvTransform{
      static_cast<float>(rect.x) / srcW,
      static_cast<float>(rect.y) / srcH,
      static_cast<float>(rect.width) / srcW,
      static_cast<float>(rect.height) / srcH,
  };
  prevPass.resolveUniformRange = push_uniform(uvTransform);
  enqueue_pass(current_frame_packet(), g_recordingFrameSlot, g_currentRenderPass);

  // Populate new render pass from previous
  const auto msaaSamples = prevPass.msaaSamples;
  RenderPass newPass{
      .label = pass_label("EFB"),
      .colorView = prevPass.colorView,
      .resolveView = prevPass.resolveView,
      .depthStencilView = prevPass.depthStencilView,
      .copySourceTexture = prevPass.copySourceTexture,
      .copySourceView = prevPass.copySourceView,
      .copySourceDepthView = prevPass.copySourceDepthView,
      .targetSize = prevPass.targetSize,
      .msaaSamples = msaaSamples,
      .clearColorValue = clearColorValue,
      .clearDepthValue = clearDepthValue,
      .clearColor = clearColor && clearAlpha,
      .clearDepth = clearDepth,
      .hasDepth = prevPass.hasDepth,
      .hasStencil = prevPass.hasStencil,
  };
  newPass.commands.reserve(2048);
  current_render_passes().emplace_back(std::move(newPass));
  ++g_currentRenderPass;

  if (!newPass.clearColor && (clearColor || clearAlpha)) {
    // If we're only clearing color _or_ alpha, perform a clear draw
    push_draw_command(clear::DrawData{
        .pipeline = pipeline_ref(clear::PipelineConfig{
            .msaaSamples = msaaSamples,
            .clearColor = clearColor,
            .clearAlpha = clearAlpha,
            .clearDepth = false, // Depth cleared via render attachment
        }),
        .color =
            wgpu::Color{
                .r = clearColorValue.x(),
                .g = clearColorValue.y(),
                .b = clearColorValue.z(),
                .a = clearColorValue.w(),
            },
    });
  }
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
}

void queue_palette_conv(tex_palette_conv::ConvRequest req) {
  auto& renderPass = current_render_passes()[g_currentRenderPass];
  ASSERT(!renderPass.sealed, "Attempted to append palette conversion to sealed render pass {}", g_currentRenderPass);
  renderPass.paletteConvs.push_back(std::move(req));
}

bool is_offscreen() noexcept { return g_inOffscreen; }

uint32_t get_sample_count() noexcept {
  CHECK(g_currentRenderPass != UINT32_MAX, "get_sample_count called outside of a frame");
  return current_render_passes()[g_currentRenderPass].msaaSamples;
}

void clear_caches() noexcept {
  g_offscreenCache.clear();
  std::lock_guard lock{g_bindGroupCacheMutex};
  g_cachedBindGroups.clear();
}

static OffscreenCacheEntry get_offscreen_textures(uint32_t width, uint32_t height) {
  OffscreenCacheKey key{width, height};
  if (const auto it = g_offscreenCache.find(key); it != g_offscreenCache.end()) {
    return it->second;
  }
  const auto colorFormat = webgpu::g_graphicsConfig.surfaceConfiguration.format;
  const wgpu::Extent3D size{width, height, 1};
  const wgpu::TextureDescriptor colorDesc{
      .label = "Offscreen Color",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc |
               wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = colorFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto colorTexture = g_device.CreateTexture(&colorDesc);
  auto colorView = colorTexture.CreateView();
  webgpu::TextureWithSampler color{
      .texture = std::move(colorTexture),
      .view = std::move(colorView),
      .size = size,
      .format = colorFormat,
  };
  const auto depthFormat = webgpu::g_graphicsConfig.depthFormat;
  const wgpu::TextureDescriptor depthDesc{
      .label = "Offscreen Depth",
      .usage = wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = depthFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto depthTexture = g_device.CreateTexture(&depthDesc);
  auto depthView = depthTexture.CreateView();
  webgpu::TextureWithSampler depth{
      .texture = std::move(depthTexture),
      .view = std::move(depthView),
      .size = size,
      .format = depthFormat,
  };
  OffscreenCacheEntry entry{
      .color = std::move(color),
      .depth = std::move(depth),
  };
  auto [insertIt, _] = g_offscreenCache.emplace(key, std::move(entry));
  return insertIt->second;
}

void begin_offscreen(uint32_t width, uint32_t height) {
  ZoneScoped;
  CHECK(g_currentRenderPass != UINT32_MAX, "begin_offscreen called outside of a frame");

  // If the current EFB pass has no resolve target, its output is unobservable.
  // Suspend it so that we can resume it after the offscreen pass.
  if (!g_inOffscreen) {
    auto& currentPass = current_render_passes()[g_currentRenderPass];
    if (!currentPass.resolveTarget) {
      g_suspendedEfbPass = std::move(currentPass);
      current_render_passes().pop_back();
      --g_currentRenderPass;
    } else {
      enqueue_pass(current_frame_packet(), g_recordingFrameSlot, g_currentRenderPass);
    }
    g_suspendedEfbViewport = g_cachedViewport;
    g_suspendedEfbScissor = g_cachedScissor;
  }

  // Create offscreen textures
  auto offscreenEntry = get_offscreen_textures(width, height);
  g_offscreenColor = std::move(offscreenEntry.color);
  g_offscreenDepth = std::move(offscreenEntry.depth);

  // Start a new pass with offscreen targets
  RenderPass newPass{
      .label = pass_label("Offscreen"),
      .colorView = g_offscreenColor.view,
      .depthStencilView = g_offscreenDepth.view,
      .copySourceTexture = g_offscreenColor.texture,
      .copySourceView = g_offscreenColor.view,
      .copySourceDepthView = g_offscreenDepth.view,
      .targetSize = {width, height, 1},
      .msaaSamples = 1,
      .clearColorValue = {0.f, 0.f, 0.f, 0.f},
      .clearDepthValue = gx::UseReversedZ ? 0.f : 1.f,
      .clearColor = true,
      .clearDepth = true,
      .hasDepth = true,
      .hasStencil = false,
      .offscreen = true,
  };
  current_render_passes().emplace_back(std::move(newPass));
  ++g_currentRenderPass;

  g_inOffscreen = true;

  g_cachedViewport = {0.f, 0.f, static_cast<float>(width), static_cast<float>(height), 0.f, 1.f};
  g_cachedScissor = {0, 0, static_cast<int32_t>(width), static_cast<int32_t>(height)};
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
}

void end_offscreen() {
  ZoneScoped;
  CHECK(g_inOffscreen, "end_offscreen called without begin_offscreen");

  enqueue_pass(current_frame_packet(), g_recordingFrameSlot, g_currentRenderPass);

  g_inOffscreen = false;
  g_offscreenColor = {};
  g_offscreenDepth = {};

  // Resume suspended EFB pass, or start a new one (load existing content)
  if (g_suspendedEfbPass) {
    current_render_passes().emplace_back(std::move(*g_suspendedEfbPass));
    g_suspendedEfbPass.reset();
  } else {
    auto& pass = current_render_passes().emplace_back();
    pass.label = pass_label("EFB");
    pass.clearColor = false;
    pass.clearDepth = false;
  }
  ++g_currentRenderPass;
  set_efb_targets(current_render_passes()[g_currentRenderPass]);

  g_cachedViewport = g_suspendedEfbViewport;
  g_cachedScissor = g_suspendedEfbScissor;
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
}

template <>
void push_draw_command(gx::DrawData data) {
  push_draw_command(ShaderDrawCommand{.type = ShaderType::GX, .gx = data});
}

#ifdef AURORA_ENABLE_RMLUI
template <>
void push_draw_command(rmlui::DrawData data) {
  push_draw_command(ShaderDrawCommand{.type = ShaderType::Rml, .rml = data});
}
#endif

template <>
PipelineRef pipeline_ref(const gx::PipelineConfig& config) {
  return find_pipeline(ShaderType::GX, config, [=] { return create_pipeline(config); });
}

#ifdef AURORA_ENABLE_RMLUI
template <>
PipelineRef pipeline_ref(const rmlui::PipelineConfig& config) {
  return find_pipeline(ShaderType::Rml, config, [=] { return rmlui::create_pipeline(config); });
}
#endif

void initialize() {
  g_frameIndex = 0;
  g_processEventsQueued.store(false, std::memory_order_release);
  g_lastPresentNs.store(0, std::memory_order_release);
  g_presentPeriodNs.store(0, std::memory_order_release);
  g_cpuFrameTimeNs.store(0, std::memory_order_release);
  g_cpuFrameStart = {};
  {
    std::lock_guard lock{g_presentStatsMutex};
    g_presentTimes.clear();
  }
  render_worker::initialize();
  // This appears to take a while and blocks the render thread for periods of time
  // render_worker::set_event_pump([] {
  //   if (g_instance) {
  //     g_instance.ProcessEvents();
  //   }
  // });
  depth_peek::initialize();
  tex_copy_conv::initialize();
  tex_palette_conv::initialize();
  texture_replacement::initialize();

  // For uniform & storage buffer offset alignments
  g_device.GetLimits(&g_cachedLimits);

  const auto createBuffer = [](wgpu::Buffer& out, wgpu::BufferUsage usage, uint64_t size, const char* label) {
    if (size <= 0) {
      return;
    }
    const wgpu::BufferDescriptor descriptor{
        .label = label,
        .usage = usage,
        .size = size,
    };
    out = g_device.CreateBuffer(&descriptor);
  };
  createBuffer(g_uniformBuffer, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, UniformBufferSize,
               "Shared Uniform Buffer");
  createBuffer(g_vertexBuffer, wgpu::BufferUsage::Storage | wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst,
               VertexBufferSize, "Shared Vertex Buffer");
  createBuffer(g_indexBuffer, wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst, IndexBufferSize,
               "Shared Index Buffer");
  createBuffer(g_storageBuffer, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, StorageBufferSize,
               "Shared Storage Buffer");
  for (size_t i = 0; i < g_stagingBuffers.size(); ++i) {
    const auto label = fmt::format("Staging Buffer {}", i);
    createBuffer(g_stagingBuffers[i], wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc, StagingBufferSize,
                 label.c_str());
  }
  for (auto& state : s_mappingStates) {
    state.store(BufferMapState::Unmapped, std::memory_order_release);
  }
  for (size_t slot = 0; slot < g_stagingBuffers.size(); ++slot) {
    map_staging_buffer(slot);
  }

  {
    constexpr std::array layoutEntries{
        // Vertex data buffer
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Vertex,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::ReadOnlyStorage,
                },
        },
        // Storage data buffer
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Vertex,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::ReadOnlyStorage,
                },
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDesc{
        .label = "Static bind group layout",
        .entryCount = layoutEntries.size(),
        .entries = layoutEntries.data(),
    };
    g_staticBindGroupLayout = g_device.CreateBindGroupLayout(&layoutDesc);
    const std::array entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .buffer = g_vertexBuffer,
        },
        wgpu::BindGroupEntry{
            .binding = 1,
            .buffer = g_storageBuffer,
        },
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .label = "Static bind group",
        .layout = g_staticBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_staticBindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  }

  {
    constexpr std::array layoutEntries{
        // Uniform buffer (dynamic offset)
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::Uniform,
                    .hasDynamicOffset = true,
                },
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDesc{
        .label = "Uniform bind group layout",
        .entryCount = layoutEntries.size(),
        .entries = layoutEntries.data(),
    };
    g_uniformBindGroupLayout = g_device.CreateBindGroupLayout(&layoutDesc);
    const std::array entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .buffer = g_uniformBuffer,
            .size = gx::MaxUniformSize,
        },
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .label = "Uniform bind group",
        .layout = g_uniformBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_uniformBindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  }

  gx::initialize();
#ifdef AURORA_ENABLE_RMLUI
  rmlui::initialize_pipeline();
#endif
  initialize_pipeline_cache();
}

void shutdown() {
  render_worker::synchronize();
  render_worker::shutdown();
  g_processEventsQueued.store(false, std::memory_order_release);
  g_lastPresentNs.store(0, std::memory_order_release);
  g_presentPeriodNs.store(0, std::memory_order_release);
  g_cpuFrameTimeNs.store(0, std::memory_order_release);
  g_cpuFrameStart = {};
  {
    std::lock_guard lock{g_presentStatsMutex};
    g_presentTimes.clear();
  }
  shutdown_pipeline_cache();
  depth_peek::shutdown();
  tex_copy_conv::shutdown();
  tex_palette_conv::shutdown();
  texture_replacement::shutdown();
  gx::shutdown();
#ifdef AURORA_ENABLE_RMLUI
  rmlui::shutdown_pipeline();
#endif

  {
    std::lock_guard lock{g_bindGroupCacheMutex};
    g_cachedBindGroups.clear();
  }
  {
    std::lock_guard lock{g_samplerCacheMutex};
    g_cachedSamplers.clear();
  }
  g_vertexBuffer = {};
  g_uniformBuffer = {};
  g_indexBuffer = {};
  g_storageBuffer = {};
  g_stagingBuffers.fill({});
  for (auto& packet : g_framePackets) {
    packet = {};
  }
  g_recordingFrame = nullptr;
  g_currentRenderPass = UINT32_MAX;
  g_offscreenCache.clear();
  g_offscreenColor = {};
  g_offscreenDepth = {};
  g_staticBindGroup = {};
  g_staticBindGroupLayout = {};
  g_uniformBindGroup = {};
  g_uniformBindGroupLayout = {};
  g_inOffscreen = false;
  g_frameIndex = UINT32_MAX;
  g_frameSlots.reset();
  g_stagingSlots.reset();
  for (auto& state : s_mappingStates) {
    state.store(BufferMapState::Unmapped, std::memory_order_release);
  }
}

static bool wait_for_staging_buffer(size_t slot) {
  ZoneScopedN("Wait for buffer map");
  map_staging_buffer(slot);
  while (true) {
    const auto mappingState = s_mappingStates[slot].load(std::memory_order_acquire);
    if (mappingState == BufferMapState::Mapped) {
      return true;
    }
    if (mappingState == BufferMapState::Unmapped) {
      return false;
    }
    wait_for_gpu_progress(std::chrono::milliseconds{1});
  }
}

static size_t acquire_frame_slot() {
  ZoneScopedN("Acquire frame slot");
  const auto waitStart = PresentClock::now();
  while (true) {
    if (const auto slot = g_frameSlots.try_acquire()) {
      const auto waitDuration = PresentClock::now() - waitStart;
      const double waitMs = std::chrono::duration<double, std::milli>{waitDuration}.count();
      TracyPlot("aurora: frameSlotWaitMs", waitMs);
      return *slot;
    }
    wait_for_gpu_progress(std::chrono::microseconds{100});
  }
}

static std::optional<size_t> acquire_mapped_staging_buffer() {
  ZoneScopedN("Acquire mapped staging buffer");
  while (true) {
    if (auto slot = g_stagingSlots.try_acquire()) {
      if (wait_for_staging_buffer(*slot)) {
        return *slot;
      }
      g_stagingSlots.release(*slot);
      return std::nullopt;
    }
    wait_for_gpu_progress(std::chrono::microseconds{100});
  }
}

bool begin_frame() {
  ZoneScoped;
  // pace_frame_start();
  const size_t frameSlot = acquire_frame_slot();
  const auto stagingSlot = acquire_mapped_staging_buffer();
  if (!stagingSlot) {
    g_frameSlots.release(frameSlot);
    return false;
  }

  auto& frame = g_framePackets[frameSlot];
  frame = {};
  // The RAM shadow tracks THIS packet's uniform region, so it resets exactly when the packet's
  // does. Only the size resets — the buffer keeps its capacity, so a steady scene stops allocating
  // after the first few ticks.
  g_uniformShadowSize = 0;
  frame.frameId = g_nextFrameId++;
  frame.frameIndex = g_frameIndex;
  frame.stagingBuffer = *stagingSlot;
  g_recordingFrame = &frame;
  g_recordingFrameSlot = frameSlot;

  size_t bufferOffset = 0;
  const auto& stagingBuf = g_stagingBuffers[*stagingSlot];
  const auto mapBuffer = [&](ByteBuffer& buf, uint64_t size) {
    if (size <= 0) {
      return;
    }
    buf = ByteBuffer{static_cast<u8*>(stagingBuf.GetMappedRange(bufferOffset, size)), static_cast<size_t>(size)};
    bufferOffset += size;
  };
  mapBuffer(frame.verts, VertexBufferSize);
  mapBuffer(frame.uniforms, UniformBufferSize);
  mapBuffer(frame.indices, IndexBufferSize);
  mapBuffer(frame.storage, StorageBufferSize);
  if constexpr (UseTextureBuffer) {
    mapBuffer(frame.textureUpload, TextureUploadSize);
  }

  g_drawCallCount = 0;
  g_mergedDrawCallCount = 0;
  g_suspendedEfbPass.reset();

  current_render_passes().emplace_back();
  auto& pass = current_render_passes()[0];
  pass.label = pass_label("EFB");
  set_efb_targets(pass);
  pass.clearColorValue = gx::g_gxState.clearColor;
  pass.clearDepthValue = gx::clear_depth_value();
  // LAZY/NO EAGER CLEAR (2026-07-08, fixes GXCopyDisp black-frame bug):
  // on real GC HW the EFB is a persistent buffer — it is erased ONLY by an
  // explicit copy-clear (GXCopyTex/GXCopyDisp with clear=GX_TRUE), which
  // resolve_pass()/copy_tex() below already implement correctly (it creates
  // the NEXT pass with clearColor/clearDepth taken from the game's own
  // clear flag). reference/sms calls GXCopyDisp(clear=true) as the LAST GX
  // op of every frame, via TDisplay::endRendering, which runs AFTER
  // TVideo::waitForRetrace() returns — i.e. AFTER this begin_frame() has
  // already run for the new frame (sb_frame_present's seam order is
  // end_frame -> begin_frame -> return to game -> GXCopyDisp). So this pass
  // 0 is exactly the render pass GXCopyDisp's resolve_pass() will read from
  // and then correctly replace with a freshly-cleared pass. If pass 0
  // defaults to LoadOp::Clear here (as it did previously — RenderPass's
  // struct default is clearColor=clearDepth=true, and this function never
  // overrode it), that eager clear stomps the still-live, fully-drawn
  // previous frame before GXCopyDisp ever gets to resolve it, and every
  // GXCopyDisp resolve reads a blank pass with zero draws in it (uniform
  // clear color, not the finished scene). Default to Load instead: pass 0
  // starts as a no-op continuation of whatever physically sits in
  // webgpu::g_frameBuffer (the previous frame's finished image) until
  // something ACTUALLY clears it — either the game's own GXCopyDisp/
  // GXCopyTex(clear=true), whose resolve_pass() creates a properly-cleared
  // successor pass, or (defensively, should a frame have zero copies) an
  // explicit clear draw. Do not restore an unconditional clear here; that
  // is the bug, not a missing feature.
  pass.clearColor = false;
  pass.clearDepth = false;
  sb_timeline_frame();
  sb_timeline_log("begin_frame LOAD efb (no eager clear) color=(%.2f,%.2f,%.2f)", pass.clearColorValue.x(),
                  pass.clearColorValue.y(), pass.clearColorValue.z());
  // SB_CLEAR_OVERRIDE=RRGGBB (hex, diagnostic): force an EAGER frame-start
  // clear to a known color — separates "what value is cleared" from "how
  // the clear reaches the screen" when chasing channel-order/stale-color
  // defects. Diagnostic only: turning this on reintroduces the eager-clear
  // bug above on purpose, for A/B comparison.
  {
    static int s_ovr = -2;
    static Vec4<float> s_ovrColor{};
    if (s_ovr == -2) {
      const char* e = std::getenv("SB_CLEAR_OVERRIDE");
      if (e != nullptr && e[0] != '\0') {
        const uint32_t v = static_cast<uint32_t>(std::strtoul(e, nullptr, 16));
        s_ovrColor = {static_cast<float>((v >> 16) & 0xff) / 255.f, static_cast<float>((v >> 8) & 0xff) / 255.f,
                      static_cast<float>(v & 0xff) / 255.f, 1.f};
        s_ovr = 1;
      } else {
        s_ovr = 0;
      }
    }
    if (s_ovr == 1) {
      pass.clearColorValue = s_ovrColor;
      pass.clearColor = true;
      pass.clearDepth = true;
    }
  }
  g_currentRenderPass = 0;
  // Refresh render viewport/scissor from logical in case FB size changed
  g_cachedViewport = gx::map_logical_viewport(gx::g_gxState.logicalViewport);
  g_cachedScissor = gx::map_logical_scissor(gx::g_gxState.logicalScissor);
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
  begin_pipeline_frame();
  render_worker::enqueue_begin_frame(frame.frameId, [frameSlot] {
    static constexpr wgpu::CommandEncoderDescriptor EncoderDescriptor{.label = "Redraw encoder"};
    g_framePackets[frameSlot].encoder = g_device.CreateCommandEncoder(&EncoderDescriptor);
    webgpu::gpu_prof::frame_begin(g_framePackets[frameSlot].encoder);
  });
  g_cpuFrameStart = PresentClock::now();
  return true;
}

void finish() {
  ZoneScoped;
  if (g_recordingFrame == nullptr) {
    return;
  }
  ASSERT(!g_inOffscreen, "finish called while offscreen rendering is active");
  if (g_currentRenderPass != UINT32_MAX) {
    auto& frame = current_frame_packet();
    frame.uniforms.append_zeroes(gx::MaxUniformSize);
    // Mirror the padding into the RAM shadow. This is not a uniform block anyone binds, but the
    // shadow has to stay byte-for-byte the same LENGTH as the staging region or the snapshot's
    // size check cannot distinguish padding from a genuinely missed write — and that check is the
    // only thing standing between a short shadow and a replay presented with stale transforms.
    if (replay_present_enabled() && !frame.replayEmission) {
      const size_t end = frame.uniforms.size();
      if (g_uniformShadow.size() < end) {
        g_uniformShadow.resize(end);
      }
      memset(g_uniformShadow.data() + g_uniformShadowSize, 0, end - g_uniformShadowSize);
      g_uniformShadowSize = end;
    }
    auto& pass = frame.renderPasses[g_currentRenderPass];
    pass.observable = true;
    pass.captureDepthSnapshot = true;
    enqueue_pass(frame, g_recordingFrameSlot, g_currentRenderPass);
    g_currentRenderPass = UINT32_MAX;
  }
}

void end_frame(EndFrameCallback callback) {
  ZoneScoped;
  ASSERT(!g_inOffscreen, "end_frame called while offscreen rendering is active");
  ASSERT(g_currentRenderPass == UINT32_MAX, "end_frame called before finish finalized the current render pass");
  if (g_cpuFrameStart.time_since_epoch().count() != 0) {
    const auto cpuFrameTime = PresentClock::now() - g_cpuFrameStart;
    update_ema(g_cpuFrameTimeNs, duration_ns(cpuFrameTime));
    const double cpuFrameTimeMs = std::chrono::duration<double, std::milli>{cpuFrameTime}.count();
    TracyPlot("aurora: cpuFrameTimeMs", cpuFrameTimeMs);
  }
  auto& frame = current_frame_packet();
  if (frame.replayEmission) {
    // The replay path is a PURE COPY of the previous packet, and its copied draws point at
    // vertex/index/storage ranges that only still mean anything because nothing wrote those global
    // buffers in between. The moment this packet records geometry of its own, those two facts stop
    // holding together and the frame renders one tick's vertices with another's indices — plausible
    // garbage rather than an error. Fail at the first byte, not at the artifact.
    ASSERT(frame.verts.size() == 0 && frame.indices.size() == 0 && frame.storage.size() == 0,
           "Replay emission recorded geometry of its own: verts={} indices={} storage={} bytes", frame.verts.size(),
           frame.indices.size(), frame.storage.size());
  }
  frame.stats.drawCallCount = g_drawCallCount;
  frame.stats.mergedDrawCallCount = g_mergedDrawCallCount;
  frame.stats.lastVertSize = frame.verts.size();
  frame.stats.lastUniformSize = frame.uniforms.size();
  frame.stats.lastIndexSize = frame.indices.size();
  frame.stats.lastStorageSize = frame.storage.size();
  frame.stats.lastTextureUploadSize = frame.textureUpload.size();
  // AURORA_REPLAY_LOG_EVERY=N (0 = off, the default): report the recorded frame's staging byte
  // counts every N game frames. The replay path shadows the whole uniform region into normal RAM
  // once per tick, so its per-tick memory cost is exactly this uniforms= number — worth measuring
  // rather than guessing. Replay emissions are skipped: their sizes are the snapshot's by
  // construction and would only halve the effective cadence.
  static const int s_frameSizeLogEvery = [] {
    const char* e = std::getenv("AURORA_REPLAY_LOG_EVERY");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  if (s_frameSizeLogEvery > 0 && !frame.replayEmission) {
    static int s_frameSizeLogCount = 0;
    if (++s_frameSizeLogCount % s_frameSizeLogEvery == 0) {
      Log.info("frame sizes: uniforms={} B verts={} B indices={} B storage={} B textureUpload={} B passes={} draws={}",
               frame.uniforms.size(), frame.verts.size(), frame.indices.size(), frame.storage.size(),
               frame.textureUpload.size(), frame.renderPasses.size(), g_drawCallCount);
    }
  }

  const size_t frameSlot = g_recordingFrameSlot;
  const uint64_t frameId = frame.frameId;
  g_currentRenderPass = UINT32_MAX;
  for (auto& array : gx::g_gxState.arrays) {
    array.cachedRange = {};
  }
  // The draw tag must not survive a frame. If the emitter stops tagging, a leaked tag would keep
  // stamping the previous object's identity onto every later draw, and interpolation would then
  // pair those draws with the wrong object's matrices — wrong, plausible, and silent.
  gx::fifo::g_pendingDrawTag = 0;
  end_pipeline_frame();
  ++g_frameIndex;
  g_recordingFrame = nullptr;

#if defined(AURORA_GFX_DEBUG_GROUPS)
  if (!g_debugGroupStack.empty()) {
    for (auto& it : std::ranges::reverse_view(g_debugGroupStack)) {
      Log.warn("Debug group was not popped at end of frame: {}", it);
    }
    g_debugGroupStack.clear();
  }

  if (g_debugMarkers.size() > 0) {
    g_debugMarkers.clear();
  }
#endif

  const size_t stagingSlot = frame.stagingBuffer;
  // A replay emission pushes no verts/indices/storage and records no draws, so publishing its
  // stats would make every second sample read zero — Tracy plots and the imgui overlay would
  // alternate real/zero and read exactly like a frame-dropping defect. The numbers the user cares
  // about belong to the frame the game actually drew, so leave the last real publish standing.
  const bool publishStats = !frame.replayEmission;
  render_worker::enqueue_end_frame(frameId, [frameSlot, stagingSlot, publishStats,
                                             callback = std::move(callback)]() mutable {
    auto& packet = g_framePackets[frameSlot];
    g_stagingBuffers[stagingSlot].Unmap();
    s_mappingStates[stagingSlot].store(BufferMapState::Unmapped, std::memory_order_release);
    auto encoder = std::move(packet.encoder);
    const auto stats = packet.stats;
    packet = {};
    if (publishStats) {
      g_stats.drawCallCount = stats.drawCallCount;
      g_stats.mergedDrawCallCount = stats.mergedDrawCallCount;
      g_stats.lastVertSize = stats.lastVertSize;
      g_stats.lastUniformSize = stats.lastUniformSize;
      g_stats.lastIndexSize = stats.lastIndexSize;
      g_stats.lastStorageSize = stats.lastStorageSize;
      g_stats.lastTextureUploadSize = stats.lastTextureUploadSize;
    }
    if (callback) {
      callback(encoder);
    }
    g_frameSlots.release(frameSlot);
    expire_cached_bind_groups();
    map_staging_buffer(stagingSlot, true);
    process_events();
  });
}

uint32_t current_frame() noexcept { return g_frameIndex; }

static void expire_cached_bind_groups() {
  std::lock_guard lock{g_bindGroupCacheMutex};
  if (g_cachedBindGroups.empty() || g_frameIndex == UINT32_MAX || g_frameIndex % BindGroupCacheSweepPeriod != 0) {
    return;
  }

  ZoneScoped;
  for (auto it = g_cachedBindGroups.begin(); it != g_cachedBindGroups.end();) {
    if (g_frameIndex - it->second.lastUsedFrame > BindGroupCacheRetainFrames) {
      g_cachedBindGroups.erase(it++);
    } else {
      ++it;
    }
  }
}

static constexpr uint64_t VertexStagingOffset = 0;
static constexpr uint64_t UniformStagingOffset = VertexStagingOffset + VertexBufferSize;
static constexpr uint64_t IndexStagingOffset = UniformStagingOffset + UniformBufferSize;
static constexpr uint64_t StorageStagingOffset = IndexStagingOffset + IndexBufferSize;
static constexpr uint64_t TextureUploadStagingOffset = StorageStagingOffset + StorageBufferSize;

static constexpr uint32_t align_down_copy_offset(uint32_t value) noexcept { return value & ~3u; }

static void copy_staging_buffer_range(wgpu::CommandEncoder& cmd, const FramePacket& frame, uint32_t& copied,
                                      uint32_t highWater, uint64_t stagingOffset, const wgpu::Buffer& dst) {
  if (highWater <= copied) {
    return;
  }
  const uint32_t copyStart = align_down_copy_offset(copied);
  const uint32_t copyEnd = AURORA_ALIGN(highWater, 4);
  cmd.CopyBufferToBuffer(g_stagingBuffers[frame.stagingBuffer], stagingOffset + copyStart, dst, copyStart,
                         copyEnd - copyStart);
  copied = highWater;
}

static bool needs_staging_copy(const FramePacket& frame, const FrameOp& op) {
  const auto& highWater = op.highWater;
  if (highWater.verts > frame.copied.verts || highWater.uniforms > frame.copied.uniforms ||
      highWater.indices > frame.copied.indices || highWater.storage > frame.copied.storage) {
    return true;
  }
  if constexpr (UseTextureBuffer) {
    return op.textureUploads.size() > frame.copied.textureUploadCount;
  }
  return false;
}

static void copy_staging_to_high_water(wgpu::CommandEncoder& cmd, FramePacket& frame, const FrameOp& op) {
  if (!needs_staging_copy(frame, op)) {
    return;
  }
  const webgpu::gpu_prof::Zone zone{cmd, "Staging copies"};
  const auto& highWater = op.highWater;
  copy_staging_buffer_range(cmd, frame, frame.copied.verts, highWater.verts, VertexStagingOffset, g_vertexBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.uniforms, highWater.uniforms, UniformStagingOffset,
                            g_uniformBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.indices, highWater.indices, IndexStagingOffset, g_indexBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.storage, highWater.storage, StorageStagingOffset, g_storageBuffer);

  if constexpr (UseTextureBuffer) {
    for (size_t i = frame.copied.textureUploadCount; i < op.textureUploads.size(); ++i) {
      const auto& item = *op.textureUploads[i];
      const wgpu::TexelCopyBufferInfo buf{
          .layout =
              wgpu::TexelCopyBufferLayout{
                  .offset = item.buffer ? item.layout.offset : item.layout.offset + TextureUploadStagingOffset,
                  .bytesPerRow = AURORA_ALIGN(item.layout.bytesPerRow, 256),
                  .rowsPerImage = item.layout.rowsPerImage,
              },
          .buffer = item.buffer ? item.buffer : g_stagingBuffers[frame.stagingBuffer],
      };
      cmd.CopyBufferToTexture(&buf, &item.tex, &item.size);
    }
    frame.copied.textureUpload = highWater.textureUpload;
    frame.copied.textureUploadCount = op.textureUploads.size();
  }
}

static void encode_op(wgpu::CommandEncoder& cmd, FramePacket& frame, const FrameOp& op) {
  copy_staging_to_high_water(cmd, frame, op);
  switch (op.type) {
  case FrameOpType::RenderPass:
    if (op.renderPass != nullptr) {
      render(cmd, frame, *op.renderPass, op.index);
    }
    break;
  case FrameOpType::TextureCopy:
    if (op.textureCopy != nullptr) {
      const webgpu::gpu_prof::Zone zone{cmd, "Texture copy"};
      cmd.CopyTextureToTexture(&op.textureCopy->src, &op.textureCopy->dst, &op.textureCopy->size);
    }
    break;
  }
}

static void render(wgpu::CommandEncoder& cmd, FramePacket& frame, RenderPass& passInfo, uint32_t passIndex) {
  ZoneScoped;
  if (!passInfo.sealed) {
    return;
  }

  for (const auto& conv : passInfo.paletteConvs) {
    tex_palette_conv::run(cmd, conv);
  }
  if (!passInfo.observable && !passInfo.resolveTarget && !passInfo.offscreen) {
    // Skip intermediate EFB render passes without observable output.
    return;
  }

  const std::array attachments{
      wgpu::RenderPassColorAttachment{
          .view = passInfo.colorView,
          .resolveTarget = passInfo.resolveView,
          .loadOp = passInfo.colorLoadOp != wgpu::LoadOp::Undefined
                        ? passInfo.colorLoadOp
                        : (passInfo.clearColor ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load),
          .storeOp = passInfo.colorStoreOp,
          .clearValue =
              {
                  .r = passInfo.clearColorValue.x(),
                  .g = passInfo.clearColorValue.y(),
                  .b = passInfo.clearColorValue.z(),
                  .a = passInfo.clearColorValue.w(),
              },
      },
  };
  wgpu::RenderPassDepthStencilAttachment depthStencilAttachment{};
  const wgpu::RenderPassDepthStencilAttachment* depthStencilAttachmentPtr = nullptr;
  if (passInfo.depthStencilView) {
    depthStencilAttachment = {
        .view = passInfo.depthStencilView,
        .depthLoadOp = passInfo.hasDepth ? (passInfo.depthLoadOp != wgpu::LoadOp::Undefined
                                                ? passInfo.depthLoadOp
                                                : (passInfo.clearDepth ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load))
                                         : wgpu::LoadOp::Undefined,
        .depthStoreOp = passInfo.hasDepth ? passInfo.depthStoreOp : wgpu::StoreOp::Undefined,
        .depthClearValue = passInfo.clearDepthValue,
        .stencilLoadOp = passInfo.hasStencil ? passInfo.stencilLoadOp : wgpu::LoadOp::Undefined,
        .stencilStoreOp = passInfo.hasStencil ? passInfo.stencilStoreOp : wgpu::StoreOp::Undefined,
        .stencilClearValue = passInfo.stencilClearValue,
    };
    depthStencilAttachmentPtr = &depthStencilAttachment;
  }
  const auto label = passInfo.label.empty() ? fmt::format("Render pass {}", passIndex)
                                            : fmt::format("{} {}", passInfo.label, passIndex);
  const wgpu::RenderPassDescriptor renderPassDescriptor{
      .label = label.c_str(),
      .colorAttachmentCount = attachments.size(),
      .colorAttachments = attachments.data(),
      .depthStencilAttachment = depthStencilAttachmentPtr,
      .timestampWrites = webgpu::gpu_prof::pass_writes(label),
  };

  auto pass = cmd.BeginRenderPass(&renderPassDescriptor);
  render_pass(pass, frame, passInfo);
  pass.End();

  if (passInfo.captureDepthSnapshot) {
    depth_peek::encode_frame_snapshot(cmd, passInfo.copySourceDepthView, passInfo.targetSize, passInfo.msaaSamples);
  }

  if (passInfo.resolveTarget) {
    const auto& dstSize = passInfo.resolveTarget->size;
    const bool needsConversion = tex_copy_conv::needs_conversion(passInfo.resolveFormat);
    const bool needsScaling = dstSize.width != static_cast<uint32_t>(passInfo.resolveRect.width) ||
                              dstSize.height != static_cast<uint32_t>(passInfo.resolveRect.height);
    const bool isDepth = gx::is_depth_format(passInfo.resolveFormat);
    if (isDepth && passInfo.msaaSamples > 1) {
      Log.fatal("Depth tex copies from multisampled EFB targets are not supported");
    }
    const tex_copy_conv::ConvRequest convReq{
        .fmt = passInfo.resolveFormat,
        .srcView = isDepth ? passInfo.copySourceDepthView : passInfo.copySourceView,
        .uniformRange = passInfo.resolveUniformRange,
        .dst = passInfo.resolveTarget,
        .sampleFilter = needsScaling ? tex_copy_conv::SampleFilter::Linear : tex_copy_conv::SampleFilter::Nearest,
    };
    if (needsConversion) {
      tex_copy_conv::run(cmd, convReq);
    } else if (needsScaling) {
      tex_copy_conv::blit(cmd, convReq);
    } else {
      const webgpu::gpu_prof::Zone zone{cmd, "EFB copy"};
      const wgpu::TexelCopyTextureInfo src{
          .texture = passInfo.copySourceTexture,
          .origin =
              wgpu::Origin3D{
                  .x = static_cast<uint32_t>(passInfo.resolveRect.x),
                  .y = static_cast<uint32_t>(passInfo.resolveRect.y),
              },
      };
      const wgpu::TexelCopyTextureInfo dst{
          .texture = passInfo.resolveTarget->texture,
      };
      const wgpu::Extent3D size{
          .width = static_cast<uint32_t>(passInfo.resolveRect.width),
          .height = static_cast<uint32_t>(passInfo.resolveRect.height),
          .depthOrArrayLayers = 1,
      };
      cmd.CopyTextureToTexture(&src, &dst, &size);
    }
  }
}

void after_submit() noexcept { depth_peek::after_submit(); }

void gpu_synchronize() { render_worker::synchronize(); }

void after_present() noexcept {
  const auto now = PresentClock::now();
  const int64_t nowNs = timestamp_ns(now);
  const int64_t previousPresentNs = g_lastPresentNs.exchange(nowNs, std::memory_order_acq_rel);
  if (previousPresentNs != 0) {
    update_ema(g_presentPeriodNs, nowNs - previousPresentNs);
    const double presentPeriodMs =
        static_cast<double>(g_presentPeriodNs.load(std::memory_order_acquire)) / 1'000'000.0;
    TracyPlot("aurora: presentPeriodMs", presentPeriodMs);
  }
  std::lock_guard lock{g_presentStatsMutex};
  g_presentTimes.push_back(now);
  prune_present_times(now);
}

float calculate_fps() noexcept {
  const auto now = PresentClock::now();
  std::lock_guard lock{g_presentStatsMutex};
  prune_present_times(now);
  if (g_presentTimes.size() < 2) {
    return 0.f;
  }

  const auto elapsed = std::chrono::duration<float>(g_presentTimes.back() - g_presentTimes.front()).count();
  if (elapsed <= 0.f) {
    return 0.f;
  }
  return static_cast<float>(g_presentTimes.size() - 1) / elapsed;
}

static void render_pass(const wgpu::RenderPassEncoder& pass, FramePacket& frame, const RenderPass& passInfo) {
  ZoneScoped;
  g_currentPipeline = UINTPTR_MAX;
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> lastDebugGroupStack;
#endif

  // Bind static bind group for the whole pass
  pass.SetBindGroup(0, g_staticBindGroup);
  pass.SetBindGroup(2, gx::g_emptyTextureBindGroup);

  for (const auto& cmd : passInfo.commands) {
#ifdef AURORA_GFX_DEBUG_GROUPS
    {
      size_t firstDiff = lastDebugGroupStack.size();
      for (size_t i = 0; i < lastDebugGroupStack.size(); ++i) {
        if (i >= cmd.debugGroupStack.size() || cmd.debugGroupStack[i] != lastDebugGroupStack[i]) {
          firstDiff = i;
          break;
        }
      }
      for (size_t i = firstDiff; i < lastDebugGroupStack.size(); ++i) {
        pass.PopDebugGroup();
      }
      for (size_t i = firstDiff; i < cmd.debugGroupStack.size(); ++i) {
        pass.PushDebugGroup(cmd.debugGroupStack[i].c_str());
      }
      lastDebugGroupStack = cmd.debugGroupStack;
    }
#endif
    switch (cmd.type) {
    case CommandType::SetViewport: {
      const auto& vp = cmd.data.setViewport;
      const float minDepth = gx::UseReversedZ ? 1.f - vp.zfar : vp.znear;
      const float maxDepth = gx::UseReversedZ ? 1.f - vp.znear : vp.zfar;
      pass.SetViewport(vp.left, vp.top, vp.width, vp.height, minDepth, maxDepth);
    } break;
    case CommandType::SetScissor: {
      const auto& sc = cmd.data.setScissor;
      const auto& size = passInfo.targetSize;
      const auto x = std::clamp(static_cast<uint32_t>(sc.x), 0u, size.width);
      const auto y = std::clamp(static_cast<uint32_t>(sc.y), 0u, size.height);
      const auto w = std::clamp(static_cast<uint32_t>(sc.width), 0u, size.width - x);
      const auto h = std::clamp(static_cast<uint32_t>(sc.height), 0u, size.height - y);
      pass.SetScissorRect(x, y, w, h);
    } break;
    case CommandType::Draw: {
      const auto& draw = cmd.data.draw;
      switch (draw.type) {
      case ShaderType::Clear:
        clear::render(draw.clear, pass, passInfo.targetSize);
        break;
      case ShaderType::GX:
        gx::render(draw.gx, pass);
        break;
#ifdef AURORA_ENABLE_RMLUI
      case ShaderType::Rml:
        rmlui::render(draw.rml, pass);
        break;
#endif
      }
    } break;
    case CommandType::DebugMarker: {
#if defined(AURORA_GFX_DEBUG_GROUPS)
      pass.InsertDebugMarker(wgpu::StringView(g_debugMarkers[cmd.data.debugMarkerIndex]));
#endif
    } break;
    }
  }

#ifdef AURORA_GFX_DEBUG_GROUPS
  for (size_t i = 0; i < lastDebugGroupStack.size(); ++i) {
    pass.PopDebugGroup();
  }
#endif
}

void render_pass(const wgpu::RenderPassEncoder& pass, u32 idx) {
  auto& frame = current_frame_packet();
  render_pass(pass, frame, frame.renderPasses[idx]);
}

bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass) {
  if (ref == g_currentPipeline) {
    return true;
  }
  wgpu::RenderPipeline pipeline;
  // A recorded draw referencing a pipeline that never finished compiling was
  // previously a SILENT DRAW SKIP (renderers returned early) — banned
  // (2026-07-14 user directive). Pipelines compile synchronously by default;
  // reaching this with a missing pipeline is a real bug (or a misuse of the
  // SB_ASYNC_PIPELINES opt-in) and must crash at the root cause, not render
  // a frame with holes.
  ASSERT(get_pipeline(ref, pipeline),
         "draw references pipeline {:x} which is not compiled -- silent draw-skip is banned "
         "(async compile is opt-in via SB_ASYNC_PIPELINES and still must never lose a draw)",
         ref);
  pass.SetPipeline(pipeline);
  g_currentPipeline = ref;
  return true;
}

static Range push(ByteBuffer& target, const uint8_t* data, size_t length, size_t alignment) {
  if (alignment != 0) {
    const size_t begin = target.size();
    const size_t alignedBegin = AURORA_ALIGN(begin, alignment);
    if (alignedBegin > begin) {
      target.append_zeroes(alignedBegin - begin);
    }
  }
  const auto begin = target.size();
  if (length > 0) {
    target.append(data, length);
  }
  return {static_cast<uint32_t>(begin), static_cast<uint32_t>(length)};
}

static Range map(ByteBuffer& target, size_t length, size_t alignment) {
  if (alignment != 0) {
    const size_t begin = target.size();
    const size_t alignedBegin = AURORA_ALIGN(begin, alignment);
    if (alignedBegin > begin) {
      target.append_zeroes(alignedBegin - begin);
    }
  }
  auto begin = target.size();
  if (length > 0) {
    target.append_zeroes(length);
  }
  return {static_cast<uint32_t>(begin), static_cast<uint32_t>(length)};
}

Range push_verts(const uint8_t* data, size_t length, size_t alignment) {
  ZoneScoped;
  return push(current_frame_packet().verts, data, length, alignment);
}

Range push_indices(const uint8_t* data, size_t length, size_t alignment) {
  ZoneScoped;
  return push(current_frame_packet().indices, data, length, alignment);
}

Range push_uniform(const uint8_t* data, size_t length) {
  ZoneScoped;
  const Range range =
      push(current_frame_packet().uniforms, data, length, g_cachedLimits.minUniformBufferOffsetAlignment);
  // SHADOW the uniform bytes into ordinary cached RAM as they are written.
  //
  // The replay emission needs this tick's true uniform bytes after the fact, and the only other way
  // to get them is to read the staging buffer back — which is WRITE-COMBINED. Uncached reads run at
  // a small fraction of RAM bandwidth, and a Delfino tick writes ~2.85 MB of uniforms, so that
  // read-back alone cost more per tick than the entire rest of the frame: measured 40 ticks/s ->
  // ~10 with interpolation on, i.e. the feature made the game run in slow motion.
  //
  // Writing a second copy as we go is nearly free by comparison: the bytes are already in cache
  // here, and cached-to-cached is orders of magnitude faster than the uncached read it replaces.
  // Mirroring at the same OFFSET (not appending) is what keeps every recorded uniformRange.offset
  // valid against the shadow without any translation.
  if (replay_present_enabled() && !current_frame_packet().replayEmission) {
    auto& shadow = g_uniformShadow;
    const size_t end = range.offset + range.size;
    if (shadow.size() < end) {
      shadow.resize(end);
    }
    memcpy(shadow.data() + range.offset, data, length);
    g_uniformShadowSize = end;
  }
  return range;
}

Range push_storage(const uint8_t* data, size_t length) {
  ZoneScoped;
  return push(current_frame_packet().storage, data, length, g_cachedLimits.minStorageBufferOffsetAlignment);
}

Range push_texture_data(const uint8_t* data, u32 bytesPerRow, u32 rowsPerImage) {
  // For CopyBufferToTexture, we need an alignment of 256 per row (see Dawn kTextureBytesPerRowAlignment)
  const auto copyBytesPerRow = AURORA_ALIGN(bytesPerRow, 256);
  const auto range = map(current_frame_packet().textureUpload, copyBytesPerRow * rowsPerImage, 0);
  u8* dst = current_frame_packet().textureUpload.data() + range.offset;
  for (u32 i = 0; i < rowsPerImage; ++i) {
    memcpy(dst, data, bytesPerRow);
    data += bytesPerRow;
    dst += copyBytesPerRow;
  }
  return range;
}

BindGroupRef bind_group_ref(const WGPUBindGroupDescriptor& descriptor) {
  const auto id = xxh3_hash(descriptor);
  std::lock_guard lock{g_bindGroupCacheMutex};
  const auto it = g_cachedBindGroups.find(id);
  if (it == g_cachedBindGroups.end()) {
    auto bg = wgpu::BindGroup::Acquire(wgpuDeviceCreateBindGroup(g_device.Get(), &descriptor));
    g_cachedBindGroups.emplace(id, CachedBindGroup{
                                       .bindGroup = std::move(bg),
                                       .lastUsedFrame = g_frameIndex,
                                   });
  } else {
    it->second.lastUsedFrame = g_frameIndex;
  }
  return id;
}

wgpu::BindGroup find_bind_group(BindGroupRef id) {
  std::lock_guard lock{g_bindGroupCacheMutex};
  const auto it = g_cachedBindGroups.find(id);
  CHECK(it != g_cachedBindGroups.end(), "get_bind_group: failed to locate {:x}", id);
  return it->second.bindGroup;
}

wgpu::Sampler sampler_ref(const wgpu::SamplerDescriptor& descriptor) {
  const auto id = xxh3_hash(descriptor);
  std::lock_guard lock{g_samplerCacheMutex};
  auto it = g_cachedSamplers.find(id);
  if (it == g_cachedSamplers.end()) {
    it = g_cachedSamplers.try_emplace(id, g_device.CreateSampler(&descriptor)).first;
  }
  return it->second;
}

uint32_t align_uniform(uint32_t value) { return AURORA_ALIGN(value, g_cachedLimits.minUniformBufferOffsetAlignment); }

void insert_debug_marker(std::string label) {
#if defined(AURORA_GFX_DEBUG_GROUPS)
  auto idx = g_debugMarkers.size();
  g_debugMarkers.emplace_back(std::move(label));
  push_command(CommandType::DebugMarker, {.debugMarkerIndex = idx});
#endif
}

} // namespace aurora::gfx

void aurora::gfx::push_debug_group(std::string label) {
#if defined(AURORA_GFX_DEBUG_GROUPS)
  g_debugGroupStack.push_back(std::move(label));
#endif
}
void push_debug_group(const char* label) {
#ifdef AURORA_GFX_DEBUG_GROUPS
  aurora::gfx::g_debugGroupStack.emplace_back(label);
#endif
}
void pop_debug_group() {
#ifdef AURORA_GFX_DEBUG_GROUPS
  if (aurora::gfx::g_debugGroupStack.empty()) {
    aurora::gfx::Log.error("Debug group stack underflowed!");
    return;
  }

  aurora::gfx::g_debugGroupStack.pop_back();
#endif
}

const AuroraStats* aurora_get_stats() { return &aurora::gfx::g_stats; }
float aurora_get_fps() { return aurora::gfx::calculate_fps(); }
