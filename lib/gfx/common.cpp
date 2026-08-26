#include "common.hpp"

#include <unordered_map>
#include "interp.hpp"
#include "indexed_interp.hpp"

#include "clear.hpp"
#include "depth_peek.hpp"
#include "gpu_submit_probe.hpp"
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
// Three slots preserve overlap for capped 60. Match-refresh may emit more packets from a tick; it
// deliberately blocks on this bounded pool instead of multiplying ~101 MB staging buffers merely
// to queue an entire high-refresh tick at once.
constexpr size_t FrameSlotCount = 3;
constexpr size_t StagingBufferCount = FrameSlotCount + 3;
constexpr auto GpuWaitTimeout =
    std::chrono::duration_cast<std::chrono::seconds>(render_worker::DefaultWorkerWaitTimeout);

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
static std::atomic_uint32_t g_frameIndex = UINT32_MAX;
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
  const void* resolveDest = nullptr;
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
  // rather than a frame the game drew. It must not publish stats, and everything it records of its
  // own (an overlay) must land above replayPrefix — see install_replay_snapshot / end_frame.
  bool replayEmission = false;
  // Bytes of verts/indices/storage reserved for the emission this packet replays. Zero on a packet
  // the game drew. Below these offsets the packet's own staging holds nothing meaningful: the data
  // the copied draws read lives in the GLOBAL buffers, written by the first emission.
  StagingHighWater replayPrefix;
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
    const auto sleepDuration =
        remainingNs > 1'000'000 ? std::chrono::milliseconds{1} : std::chrono::nanoseconds{remainingNs};
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
                 p.label.c_str(), p.commands.size(), p.clearColor ? 1 : 0, p.clearDepth ? 1 : 0, p.clearColorValue.x(),
                 p.clearColorValue.y(), p.clearColorValue.z(), kind);
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

static_assert(static_cast<uint8_t>(ShaderType::Clear) == AURORA_GPU_DRAW_CLEAR);
static_assert(static_cast<uint8_t>(ShaderType::GX) == AURORA_GPU_DRAW_GX);
static_assert(static_cast<uint8_t>(ShaderType::Rml) == AURORA_GPU_DRAW_RML);

static AuroraGpuSubmitInfo build_submit_probe(const FramePacket& frame) {
  const auto textureCaches = gx::texture_cache_counts();
  gpu_submit_probe::FrameInput input{
      .replayEmission = frame.replayEmission ? 1u : 0u,
      .frameId = frame.frameId,
      .frameIndex = frame.frameIndex,
      .passCount = static_cast<uint32_t>(frame.renderPasses.size()),
      .operationCount = static_cast<uint32_t>(frame.ops.size()),
      .textureUploadCount = static_cast<uint32_t>(frame.textureUploads.size()),
      .textureCopyCount = static_cast<uint32_t>(frame.textureCopies.size()),
      .vertexBytes = static_cast<uint32_t>(frame.verts.size()),
      .uniformBytes = static_cast<uint32_t>(frame.uniforms.size()),
      .indexBytes = static_cast<uint32_t>(frame.indices.size()),
      .storageBytes = static_cast<uint32_t>(frame.storage.size()),
      .textureUploadBytes = static_cast<uint32_t>(frame.textureUpload.size()),
      .cachedTextureObjects = textureCaches.textureObjects,
      .cachedTlutObjects = textureCaches.tlutObjects,
      .cachedCopyTextures = textureCaches.copyTextures,
      .persistentStorageEntries = static_cast<uint32_t>(persistent_storage_entries()),
      .persistentStorageBytes = static_cast<uint32_t>(persistent_storage_used()),
  };
  {
    std::lock_guard lock{g_bindGroupCacheMutex};
    input.cachedBindGroups = static_cast<uint32_t>(g_cachedBindGroups.size());
  }
  gpu_submit_probe::Builder builder{input};
  for (const auto& pass : frame.renderPasses) {
    builder.begin_pass({
        .label = pass.label,
        .commandCount = static_cast<uint32_t>(pass.commands.size()),
        .targetWidth = pass.targetSize.width,
        .targetHeight = pass.targetSize.height,
        .flags = (pass.observable ? 1u : 0u) | (pass.offscreen ? 1u << 1 : 0u) | (pass.resolveTarget ? 1u << 2 : 0u) |
                 (pass.hasDepth ? 1u << 3 : 0u) | (pass.clearColor ? 1u << 4 : 0u) | (pass.clearDepth ? 1u << 5 : 0u) |
                 ((pass.msaaSamples & 0xffu) << 8),
    });
    for (const auto& command : pass.commands) {
      switch (command.type) {
      case CommandType::SetViewport:
        builder.add_viewport({
            .x = command.data.setViewport.left,
            .y = command.data.setViewport.top,
            .width = command.data.setViewport.width,
            .height = command.data.setViewport.height,
            .minDepth = command.data.setViewport.znear,
            .maxDepth = command.data.setViewport.zfar,
        });
        break;
      case CommandType::SetScissor:
        builder.add_scissor({
            .x = command.data.setScissor.x,
            .y = command.data.setScissor.y,
            .width = command.data.setScissor.width,
            .height = command.data.setScissor.height,
        });
        break;
      case CommandType::Draw:
        switch (command.data.draw.type) {
        case ShaderType::Clear:
          builder.add_draw({
              .pipeline = command.data.draw.clear.pipeline,
              .color = {command.data.draw.clear.color.r, command.data.draw.clear.color.g,
                        command.data.draw.clear.color.b, command.data.draw.clear.color.a},
              .depth = command.data.draw.clear.depth,
              .rectEnabled = static_cast<uint8_t>(command.data.draw.clear.rectEnabled ? 1u : 0u),
              .rectX = command.data.draw.clear.rect.x,
              .rectY = command.data.draw.clear.rect.y,
              .rectWidth = command.data.draw.clear.rect.width,
              .rectHeight = command.data.draw.clear.rect.height,
          });
          break;
        case ShaderType::GX:
          builder.add_draw({
              .pipeline = command.data.draw.gx.pipeline,
              .vertexRange = {command.data.draw.gx.vertRange.offset, command.data.draw.gx.vertRange.size},
              .indexRange = {command.data.draw.gx.idxRange.offset, command.data.draw.gx.idxRange.size},
              .uniformRange = {command.data.draw.gx.uniformRange.offset, command.data.draw.gx.uniformRange.size},
              .vertexCount = command.data.draw.gx.vtxCount,
              .indexCount = command.data.draw.gx.indexCount,
              .instanceCount = command.data.draw.gx.instanceCount,
              .textureBindGroup = command.data.draw.gx.bindGroups.textureBindGroup,
              .destinationAlpha = command.data.draw.gx.dstAlpha,
              .tag = command.data.draw.gx.tag,
              .population = command.data.draw.gx.pop,
              .exact = command.data.draw.gx.exact,
              .indexedPositionSample = command.data.draw.gx.indexedPosSample,
              .positionArrayUniformOffset = command.data.draw.gx.posArrayUniformOffset,
              .matrixPositionOffset = command.data.draw.gx.mtxPosOffset,
              .matrixNormalOffset = command.data.draw.gx.mtxNrmOffset,
              .orthographic = command.data.draw.gx.ortho,
              .vertexStride = command.data.draw.gx.vtxStride,
              .positionOffset = command.data.draw.gx.posOffset,
              .positionF32Xyz = command.data.draw.gx.posF32XYZ,
              .positionS16Xyz = command.data.draw.gx.posS16XYZ,
              .positionFraction = command.data.draw.gx.posFrac,
              .deformF32OffsetMask = command.data.draw.gx.deformF32OffsetMask,
              .cameraTextureMatrixMask = command.data.draw.gx.texMtxCamMask,
              .positionMatrixSlot = command.data.draw.gx.pnMtxSlot,
          });
          break;
#ifdef AURORA_ENABLE_RMLUI
        case ShaderType::Rml:
          builder.add_draw({
              .pipeline = command.data.draw.rml.pipeline,
              .vertexRange = {command.data.draw.rml.vertexRange.offset, command.data.draw.rml.vertexRange.size},
              .indexRange = {command.data.draw.rml.indexRange.offset, command.data.draw.rml.indexRange.size},
              .uniformRange = {command.data.draw.rml.uniformRange.offset, command.data.draw.rml.uniformRange.size},
              .bindGroup1 = command.data.draw.rml.bindGroup1,
              .bindGroup2 = command.data.draw.rml.bindGroup2,
              .bindGroup1DynamicOffset = command.data.draw.rml.bindGroup1DynamicOffset,
              .bindGroup2DynamicOffset = command.data.draw.rml.bindGroup2DynamicOffset,
              .dynamicBindGroupMask = command.data.draw.rml.dynamicBindGroupMask,
              .drawKind = command.data.draw.rml.drawKind,
              .vertexCount = command.data.draw.rml.vertexCount,
              .indexCount = command.data.draw.rml.indexCount,
              .stencilReference = command.data.draw.rml.stencilRef,
              .blendConstant = command.data.draw.rml.blendConstant,
              .hasBlendConstant = command.data.draw.rml.hasBlendConstant,
          });
          break;
#endif
        }
        break;
      case CommandType::DebugMarker:
        builder.add_debug_marker(command.data.debugMarkerIndex);
        break;
      }
    }
    builder.end_pass();
  }
  return builder.finish();
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
// Replay presentation: present one recorded frame more than once.
//
// AURORA_REPLAY_PRESENT=1 remains the two-presentation diagnostic control: both presentations of a
// tick carry byte-identical content, so anything that differs between them is the EFB's own history
// and not the replay. Host-driven interpolation may instead retain this snapshot for several
// display-rate samples. Pass 0 is LoadOp::Load with no eager clear (see the
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
//     bytes because nothing writes them in between.
//
//     A replay emission is nevertheless allowed to RECORD GEOMETRY OF ITS OWN, because an overlay
//     that must appear on every presented frame (the RmlUi settings menu draws through
//     gfx::push_verts/push_indices) is drawn per EMISSION, not per tick — skipping it on the
//     in-betweens would run the menu at half the present rate. What it may not do is write over the
//     first emission's bytes, which is exactly what an unreserved packet would do: its staging
//     starts at offset 0 and the staging copy lands at the same offsets in the global buffers.
//     install_replay_snapshot therefore RESERVES the first emission's high-water mark in this
//     packet's verts/indices/storage and seeds frame.copied to the same value, so a push lands
//     above the game's bytes and the reserved prefix (which belongs to a different staging buffer
//     and was never written here) is never copied down over them. The two seeds must move
//     together: seeding `copied` alone would make a later push satisfy highWater <= copied, emit no
//     copy at all and silently draw the PREVIOUS frame's bytes; reserving the buffer alone would
//     copy this packet's garbage prefix over the real data.
//
//   * FrameOps are NOT copied. A FrameOp holds raw pointers into its own packet's renderPasses
//     deque and is resolved through its frame SLOT; the original's pointers die the moment its
//     end_frame does `packet = {}`. The replay re-runs capture_frame_op/enqueue_pass over its own
//     deque and its own slot instead.
// ---------------------------------------------------------------------------
struct ReplaySnapshot {
  RenderPassList renderPasses;
  std::vector<uint8_t> uniforms;
  // High-water marks of the emission this snapshot was taken from, rounded up to the staging copy's
  // 4-byte granularity. A replay emission reserves these so anything it records lands ABOVE the
  // bytes its copied draws still point at. Rounded UP because copy_staging_buffer_range aligns its
  // start DOWN to 4: an unaligned seed would copy up to three of this packet's stale bytes over the
  // tail of the first emission's data.
  uint32_t verts = 0;
  uint32_t indices = 0;
  uint32_t storage = 0;
  bool valid = false;
};
static ReplaySnapshot g_replaySnapshot;

namespace {
// Host-set overrides for interpolation policy. They exist so the host can turn the WHOLE
// feature on with one switch: this is a user-facing mode, and requiring a player to set three
// environment variables in agreement is a way to ship a broken configuration, not a knob. The env
// vars still work and still win, because the diagnostic paths (the EFB-idempotence control, the A/B
// runs) need to drive the diagnostic pair independently.
bool g_replayForced = false;
float g_alphaForced = -1.0f;
unsigned g_replayPresentationCount = 2;

bool replay_env_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("AURORA_REPLAY_PRESENT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}
} // namespace

// This tick's uniform bytes in ordinary cached RAM, mirrored at the same offsets as the GPU staging
// buffer they were written to. See push_uniform for why this exists rather than reading the staging
// back. Grows to the high-water mark of a tick and then stops reallocating.
std::vector<uint8_t> g_uniformShadow;
size_t g_uniformShadowSize = 0;

void force_interpolation(float alpha) {
  // Negative disables the host override. The setting is user-facing and can
  // change from an in-game menu, so a one-way latch would make the UI say Off
  // while replay presentation remained on for the rest of the process.
  g_replayForced = alpha >= 0.0f;
  g_alphaForced = g_replayForced ? alpha : -1.0f;
}

void set_replay_presentation_count(unsigned count) { g_replayPresentationCount = std::clamp(count, 1u, 64u); }

bool replay_present_enabled() noexcept { return replay_env_enabled() || g_replayForced; }

unsigned replay_presentation_count() noexcept {
  return replay_env_enabled() ? 2u : (g_replayForced ? g_replayPresentationCount : 1u);
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
      Log.error(
          "AURORA_INTERP_ALPHA={} is outside [0,1]; interpolation stays OFF. An alpha outside "
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
bool g_snapCurrentTick = false;
long g_snappedTicks = 0;
} // namespace

void snap_next_interpolation() { g_snapNextTick = true; }

namespace {
// The destination of the copy that feeds a temporal-feedback effect, supplied by the host because
// aurora cannot tell it apart from any other EFB copy: structurally they are identical, and the
// difference is only in WHO reads the result and WHEN.

// Per-frame ordering record. Keyed by copy destination: the highest pass index at which a draw
// sampled that copy's result, and the pass index of the copy itself. Both reset every frame.
std::unordered_map<const void*, uint32_t> g_copySampledAtPass;
std::unordered_map<const void*, uint32_t> g_copyResolvedAtPass;
const void* g_pendingResolveDest = nullptr;
} // namespace

long g_noteSampledCalls = 0;
long g_noteSampledRefused = 0;

void note_copy_texture_sampled(const void* dest) {
  if (dest == nullptr || g_recordingFrame == nullptr || g_recordingFrame->replayEmission) {
    ++g_noteSampledRefused;
    return;
  }
  ++g_noteSampledCalls;
  // The pass currently recording. A sample inside the SAME pass that later resolves still counts as
  // "before the copy", because the resolve happens at the end of the pass.
  const uint32_t pass = g_currentRenderPass == UINT32_MAX ? 0 : g_currentRenderPass;
  auto it = g_copySampledAtPass.find(dest);
  if (it == g_copySampledAtPass.end() || it->second < pass) {
    g_copySampledAtPass[dest] = pass;
  }
}

void note_copy_resolve_dest(const void* dest) { g_pendingResolveDest = dest; }

// THE DECISION, on its own so it can be tested. `copyPass` is the pass whose resolve writes `dest`.
//
// Cross-frame feedback iff the frame sampled this copy's result and EVERY such sample was in a
// STRICTLY earlier pass — those samples can only have read what the previous frame left there.
// A sample in a later pass, or in the same pass (this record has pass granularity, so a same-pass
// sample cannot be shown to precede the resolve), means a consumer in THIS frame depends on the
// copy and it must run on both emissions.
bool is_cross_frame_feedback(const void* dest, uint32_t copyPass) {
  if (dest == nullptr) {
    return false;
  }
  const auto it = g_copySampledAtPass.find(dest);
  if (it == g_copySampledAtPass.end()) {
    return false; // never sampled this frame: nothing here proves it is feedback
  }
  // <=, NOT <. The resolve does not sit at some unknown point inside the pass — resolve_pass()
  // attaches it to the CURRENT pass and ends it, so every draw recorded in that pass happened
  // BEFORE the copy by construction. A same-pass sample therefore read what the PREVIOUS frame left
  // in the texture, which is exactly what feedback means.
  //
  // The strict form made this classifier structurally unable to see feedback in this game: SMS
  // composites the dash trail and copies the result inside ONE pass, so every copy scored
  // intra-frame and the count read 0 suppressed of 55,198 over 13,800 ticks. That zero looked like
  // "this scene has no feedback effects" and was really "this classifier cannot represent the case".
  // note_copy_texture_sampled had documented the correct rule all along ("A sample inside the SAME
  // pass that later resolves still counts as before the copy"); the two functions disagreed and the
  // selftest encoded the disagreement rather than catching it.
  return it->second <= copyPass;
}

// Run the classifier against BOTH classes it must separate, before it is trusted on a real frame.
// This project has shipped a discriminator that scored backwards on both classes, and the previous
// attempt at THIS decision (by texture identity) blanked the title background because its premise
// was never tested against the intra-frame case.
bool copy_classifier_selftest() {
  const auto savedSamples = g_copySampledAtPass;
  bool ok = true;
  const void* kFeedback = reinterpret_cast<const void*>(0x1000);
  const void* kIntra = reinterpret_cast<const void*>(0x2000);
  const void* kSame = reinterpret_cast<const void*>(0x3000);
  const void* kUnsampled = reinterpret_cast<const void*>(0x4000);
  g_copySampledAtPass.clear();
  g_copySampledAtPass[kFeedback] = 1; // sampled in pass 1, written in pass 3 -> feedback
  g_copySampledAtPass[kIntra] = 4;    // sampled in pass 4, written in pass 2 -> intra-frame
  g_copySampledAtPass[kSame] = 2;     // sampled in the same pass that writes it -> intra-frame
  struct Case {
    const char* name;
    const void* dest;
    uint32_t pass;
    bool want;
  };
  const Case cases[] = {
      {"sampled BEFORE the copy (previous frame's contents)", kFeedback, 3, true},
      {"sampled AFTER the copy (this frame's consumer)", kIntra, 2, false},
      // Same-pass IS feedback: resolve_pass ends the pass, so a draw recorded in it precedes the
      // copy. This case previously expected `false` with the rationale "order unknown", which was
      // the classifier's own limitation written down as a property of the world.
      {"sampled in the SAME pass as the copy (the resolve ENDS the pass, so the sample precedes it)", kSame, 2, true},
      {"never sampled at all", kUnsampled, 2, false},
  };
  for (const Case& c : cases) {
    const bool got = is_cross_frame_feedback(c.dest, c.pass);
    if (got != c.want) {
      ok = false;
      Log.error(
          "SELFTEST FAILED [{}]: classified cross-frame={} but expected {}. The copy "
          "classifier does not separate the two cases, so suppressing on its verdict would "
          "either starve an intra-frame consumer or leave feedback running twice per tick.",
          c.name, got, c.want);
    }
  }
  g_copySampledAtPass = savedSamples;
  if (ok) {
    Log.info(
        "copy classifier selftest PASSED: separates sampled-before (feedback) from "
        "sampled-after and same-pass (intra-frame), and refuses an unsampled copy — all four "
        "run, not just the one it is expected to find.");
  }
  return ok;
}

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
  g_replaySnapshot.renderPasses = frame.renderPasses;
  // Read from the RAM shadow, never from frame.uniforms — that is a view over write-mapped GPU
  // staging, which is write-combined, and reading ~2.85 MB of it back per tick is what made this
  // feature run the game in slow motion. push_uniform mirrors every write here at the same offset,
  // so this is a plain cached copy.
  //
  // The sizes must agree. If they do not, the shadow missed writes and the replay would present a
  // frame built from stale uniforms — geometry from this tick with some other tick's transforms,
  // which looks like a render bug rather than a bookkeeping one. Refuse loudly instead.
  if (g_uniformShadowSize != frame.uniforms.size()) {
    Log.error(
        "uniform shadow is {} bytes but the frame recorded {} — some uniform write bypassed "
        "push_uniform. Refusing to snapshot: replaying from a short shadow would present this "
        "tick's geometry with stale transforms and read as a render defect.",
        g_uniformShadowSize, frame.uniforms.size());
    return false;
  }
  g_replaySnapshot.uniforms.resize(g_uniformShadowSize);
  if (g_uniformShadowSize != 0) {
    memcpy(g_replaySnapshot.uniforms.data(), g_uniformShadow.data(), g_uniformShadowSize);
  }
  g_replaySnapshot.verts = static_cast<uint32_t>(AURORA_ALIGN(frame.verts.size(), 4));
  g_replaySnapshot.indices = static_cast<uint32_t>(AURORA_ALIGN(frame.indices.size(), 4));
  g_replaySnapshot.storage = static_cast<uint32_t>(AURORA_ALIGN(frame.storage.size(), 4));
  g_replaySnapshot.valid = true;
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
bool interpolate_recorded_frame(float alpha, bool resampling) {
  ZoneScoped;
  if (g_recordingFrame == nullptr) {
    Log.error("interpolate_recorded_frame: no frame is recording");
    return false;
  }
  if (!g_replaySnapshot.valid) {
    Log.error(
        "interpolate_recorded_frame: no snapshot, so there is nothing to read the true "
        "matrices from. Interpolating from the staging itself would read back what this "
        "function is about to overwrite.");
    return false;
  }
  // The attribution numbers in interp::report() are only worth reading if the discriminator behind
  // them actually discriminates. Prove that once, on synthetic input, before it is ever pointed at
  // the game — a self-test that nobody runs is the same bug one level up.
  static const bool s_selftestOk = interp::selftest();
  static const bool s_indexedSelftestOk = indexed_interp::selftest();
  static const bool s_copyClassifierOk = copy_classifier_selftest();
  (void)s_selftestOk;
  (void)s_indexedSelftestOk;
  (void)s_copyClassifierOk;
  // A tick the game declared discontinuous has no meaningful in-between: the halfway pose is a
  // viewpoint it never simulated. Force alpha 1 rather than skipping the pass, so the pairing table
  // is still filled from this tick and the tick AFTER the cut can interpolate normally — skipping
  // would leave the table holding the pre-cut pose and move the artefact one frame later.
  if (!resampling) {
    g_snapCurrentTick = g_snapNextTick;
    g_snapNextTick = false;
    if (g_snapCurrentTick)
      ++g_snappedTicks;
  }
  const bool snapping = g_snapCurrentTick;
  if (snapping) {
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
    // WHICH copies are temporal feedback, decided from THIS frame's own ordering rather than from
    // which effect samples them. A copy is feedback iff its result was sampled during the frame and
    // every one of those samples happened at or before the pass that writes it — meaning they read
    // the PREVIOUS frame's contents. A copy sampled by a LATER pass is intra-frame and must run on
    // both emissions, or that later pass reads nothing.
    //
    // Identity cannot decide this and the attempt is recorded in
    // debug_journal/2026-08-04_afterimage_effect_and_frame_budget.md: SMS uses ONE screen texture
    // for both purposes — cross-frame for the dash trail, intra-frame for the title's sky composite
    // — so suppressing "the texture TAfterEffect samples" blanked the title background on every
    // interpolated present.
    // ── WAIT FOR THE WORKER BEFORE TOUCHING RECORDED PASSES ───────────────────────────────────
    //
    // The loop below WRITES into frame.renderPasses, and those RenderPass objects were handed to
    // the render worker as each pass was sealed — long before this runs. Two things were wrong with
    // that, and only one of them was visible:
    //
    //  * the crash. `pass.resolveTarget = {}` nulled a handle while the worker was encoding the
    //    same pass, and tex_copy_conv::execute faulted dereferencing it. Reading the field twice in
    //    the caller proved it: 0x12825140 when tested, 0x0 a few instructions later, in the same
    //    function with no store between. Stage 24 died about one run in two with interpolation on
    //    and never once with it off, which is exactly the shape of a race this path alone creates.
    //  * the quiet one. Whether a copy was suppressed at all depended on whether the worker had
    //    already encoded that pass — so the classifier's decision was applied to some passes and
    //    silently lost on others, differently every run. A frame that renders correctly by winning
    //    a race is not a frame that renders correctly.
    //
    // Synchronizing costs a pipeline stall once per interpolated tick. That is real and it is
    // stated rather than hidden; it is also the only ordering under which the mutation means
    // anything, because the alternative is a decision applied to a random subset of the frame.
    render_worker::synchronize();
    static long s_suppressed = 0, s_kept = 0, s_ticks = 0;
    long thisTick = 0;
    for (uint32_t i = 0; i < frame.renderPasses.size(); ++i) {
      auto& pass = frame.renderPasses[i];
      if (!pass.resolveTarget || pass.resolveDest == nullptr) {
        continue;
      }
      // AURORA_EFB_FEEDBACK=present: let the interpolated emission perform the feedback copy too,
      // so the effect advances once per PRESENT rather than once per game tick.
      //
      // Why this is worth a switch rather than a decision. A temporal-feedback effect is a filter
      // over the frame sequence, and the dash afterimage composites the previous copy back over the
      // viewport each time it runs. Suppressing it here keeps the filter running at the SIM rate,
      // which preserves the retail trail length exactly — and leaves the ghost stepping at 30 Hz
      // behind a Mario drawn at 60, which is what the user reports. Running it per present makes the
      // ghost advance with him, at the cost of halving the filter's time constant: the same
      // per-step decay applied twice as often is a shorter trail in wall-clock terms.
      //
      // The note this replaces claimed running it twice makes the trail "full length on one present
      // and half on the next". That is a claim about alternation and it is measurable; the switch
      // exists so it can be measured rather than argued.
      //   (unset)  classify, and suppress only what the classifier calls feedback  [default]
      //   present   never suppress — the effect advances once per PRESENT
      //   tick      suppress EVERY copy on the interpolated emission, classifier or not
      //
      // The third mode exists because the classifier reports 0 feedback copies in this game (they
      // are all sampled in a LATER pass than the one that writes them, i.e. genuinely intra-frame),
      // so `present` and the default are byte-identical and neither can tell you whether the copy
      // RATE is what makes a screen effect judder. `tick` forces the other end of the axis so the
      // question is answerable by looking rather than by argument.
      static const int s_feedbackMode = [] {
        const char* e = std::getenv("AURORA_EFB_FEEDBACK");
        if (e == nullptr)
          return 0;
        if (std::strcmp(e, "present") == 0)
          return 1;
        if (std::strcmp(e, "tick") == 0)
          return 2;
        return 0;
      }();
      const bool suppress =
          s_feedbackMode == 2 || (s_feedbackMode == 0 && is_cross_frame_feedback(pass.resolveDest, i));
      if (suppress) {
        pass.resolveTarget = {}; // feedback: leave it to the replay emission, once per tick
        ++thisTick;
      } else {
        ++s_kept;
      }
    }
    s_suppressed += thisTick;
    if ((++s_ticks % 300) == 0) {
      // The classifier's INPUT, printed beside its output. "0 suppressed" has two completely
      // different causes — no copy is feedback, or nothing ever told the classifier that a copy's
      // result was sampled — and without this line they are the same number.
      Log.info(
          "  classifier input: note_copy_texture_sampled accepted {} call(s), refused {} "
          "(null dest, no recording frame, or the replay emission); {} distinct copy dest(s) "
          "have a recorded sample this frame.{}",
          g_noteSampledCalls, g_noteSampledRefused, g_copySampledAtPass.size(),
          g_noteSampledCalls == 0 ? "  <-- NEVER ACCEPTED A SAMPLE. The verdict below is not about the scene; the "
                                    "classifier was never given anything to classify."
                                  : "");
      Log.info(
          "EFB copies over {} ticks: {} suppressed on the interpolated emission (cross-frame "
          "feedback, sampled only BEFORE the pass that writes them) and {} kept (intra-frame, "
          "a later pass reads them). Both numbers matter: all-suppressed means an intra-frame "
          "copy is being starved, all-kept means no feedback copy was recognised.",
          s_ticks, s_suppressed, s_kept);
    }
  }

  // Can this packet's own staging be read at the range a recorded command names? On a frame the
  // game drew, yes, up to its high-water mark. On a REPLAY emission the recorded ranges sit below
  // replayPrefix, where this packet's staging holds another frame's leftovers rather than the
  // vertices those commands draw (those live in the global buffer, written by the first emission) —
  // reading them would silently interpolate garbage.
  const auto verts_readable = [&frame](const Range& r) {
    return r.size > 0 && r.offset >= frame.replayPrefix.verts && r.offset + r.size <= frame.verts.size();
  };

  interp::begin_tick(resampling);
  // The camera delta is computed ONCE for the tick and applied to every draw that could not be
  // paired. Without it, unpaired draws render from the current viewpoint while paired ones sit at
  // the in-between one, and the frame is drawn from two viewpoints at once — measured as worse than
  // not interpolating at all.
  interp::begin_camera_delta(alpha);
  if (snapping && !resampling) {
    // Printed with the tick index so the game's declared cut can be lined up against the per-tick
    // camera measurements in interp::report(). If the snapped ticks do not coincide with the ticks
    // that measured a large camera step, then either the signal is firing on non-cuts or it is
    // missing real ones — and a snap count alone could not tell you which.
    Log.info(
        "tick {}: SNAPPED (alpha forced to 1) — the game declared the camera discontinuous, so "
        "this tick has no in-between to show",
        interp::tick_index());
  }
  for (auto& pass : frame.renderPasses) {
    for (auto& cmd : pass.commands) {
      if (cmd.type != CommandType::Draw || cmd.data.draw.type != ShaderType::GX) {
        continue;
      }
      gx::DrawData& d = cmd.data.draw.gx;

      if (d.uniformRange.offset + d.uniformRange.size > snap.size()) {
        continue; // outside the snapshot: cannot have been recorded by this frame
      }
      uint8_t* dst = frame.uniforms.data() + d.uniformRange.offset;
      // Every draw ends up on the interpolated viewpoint, one way or the other. A draw that was
      // genuinely paired already carries it, because the camera is baked into the matrices being
      // lerped. Everything else — untagged, or tagged but unpaired this tick — takes the camera
      // delta alone. Leaving ANY draw on the current viewpoint is what tears the frame.
      // THE AUDIT. Every draw lands in exactly one of these five, so the columns sum to the draw
      // count and a population cannot quietly go missing between them.
      if (d.ortho != 0) {
        // Measure whether this screen-space draw is actually STILL. See interp.hpp — `snap:2D` is
        // reported as correct on the assumption that a 2D element has no in-between, and that
        // assumption had never been checked against an animating one.
        // Prefer the draw's OWN vertex data. A 2D element's geometry is in its vertices; the
        // orthographic position matrix may be one matrix shared by every 2D draw in the frame, and
        // hashing that attributes one global change to every population separately — which is what
        // six unrelated sites all reading "387 of 388" turned out to be.
        const bool haveVerts = d.posF32XYZ != 0 && d.vtxCount > 0 && verts_readable(d.vertRange);
        interp::note_ortho_geometry(d.pop, snap.data() + d.uniformRange.offset, d.uniformRange.size, d.mtxPosOffset,
                                    haveVerts ? frame.verts.data() + d.vertRange.offset : nullptr,
                                    haveVerts ? d.vertRange.size : 0);
      }
      interp::note_disposition(d.pop, d.ortho != 0   ? interp::Disposition::SnappedOrtho
                                      : d.exact != 0 ? interp::Disposition::SnappedExact
                                                     : interp::Disposition::Pending);
      // EXACT: the emitter has declared this draw screen-space under a perspective projection (an
      // identity position matrix with eye-space vertices — SMS_FillScreenAlpha's dst-alpha mask).
      // The ortho test cannot see it and the camera delta must not touch it: sliding a full-screen
      // mask by a fraction of the camera's motion is not smoothing, it is moving something that is
      // nailed to the display.
      if (d.exact != 0) {
        continue;
      }
      // BILLBOARDS FIRST. A JPA particle's position lives in its VERTEX data, so patch_draw would
      // pair it and lerp an identity matrix against an identity matrix — a no-op that also
      // SUPPRESSES the camera delta, leaving the particle worse off than untagged. patch_billboard
      // recognises the tag by having a recorded world position for it and applies the object's own
      // displacement as a translation on top of the camera delta.
      if (interp::patch_billboard(d.tag, snap.data() + d.uniformRange.offset, dst, d.uniformRange.size, d.mtxPosOffset,
                                  d.mtxNrmOffset, alpha)) {
        interp::note_disposition(d.pop, interp::Disposition::Billboard);
        continue;
      }

      // TDL quad batches rebuild a separate indexed array under an identity matrix. Repoint only
      // this interpolated emission at a temporary array; the replay snapshot keeps the exact
      // current-tick offset in its untouched uniform copy.
      if (d.indexedPosSample != 0) {
        std::vector<uint8_t> indexedPositions;
        if (indexed_interp::patch(d.indexedPosSample, alpha, resampling, indexedPositions)) {
          const Range range = push_storage(indexedPositions.data(), indexedPositions.size());
          ASSERT(d.posArrayUniformOffset + sizeof(uint32_t) <= d.uniformRange.size,
                 "indexed position array uniform offset {} is outside {}-byte block", d.posArrayUniformOffset,
                 d.uniformRange.size);
          std::memcpy(dst + d.posArrayUniformOffset, &range.offset, sizeof(range.offset));
          interp::note_disposition(d.pop, interp::Disposition::Paired);
          continue;
        }
        // Do not fall through to matrix pairing: these batches deliberately carry identity
        // matrices, so identity-to-identity would report a pair while leaving the array snapped.
        // A first sighting or incompatible array receives the same camera-only fallback as other
        // unpaired perspective geometry.
        if (d.ortho == 0) {
          interp::patch_camera_only(snap.data() + d.uniformRange.offset, dst, d.uniformRange.size, d.mtxPosOffset,
                                    d.mtxNrmOffset, d.texMtxCamMask);
          const interp::Disposition disposition =
              indexed_interp::birth_only(d.indexedPosSample)          ? interp::Disposition::CameraOnlyBirth
              : indexed_interp::reappearance_only(d.indexedPosSample) ? interp::Disposition::CameraOnlyReappearance
                                                                      : interp::Disposition::CameraOnly;
          interp::note_disposition(d.pop, disposition);
        }
        continue;
      }

      // ── DEFORMING GEOMETRY: interpolate the VERTICES ──────────────────────────────────────────
      //
      // LAST, deliberately. A flag or the sea ripple grid rebuilds its mesh every tick, so its
      // motion is in the vertex data and no matrix reaches it — but "has direct f32 positions and a
      // tag" also describes a JPA billboard, whose positions are baked in EYE SPACE. Lerping those
      // across two ticks mixes two different view transforms; running this before patch_billboard
      // did exactly that and moved 516,562 particle draws off the correct path onto this one.
      // Ordering it after the specific paths makes it the fallback it should be.
      //
      // The lerp goes into a SEPARATE buffer and only THIS emission's command is repointed at it.
      // Both emissions replay the same recorded passes and therefore the same vertRange, so
      // patching in place would corrupt the tick's own frame — uniforms escape that because the
      // snapshot re-pushes them, vertices have no such path. The snapshot's copy of this command
      // keeps the original range, so the replay emission still draws the tick exactly.
      if ((d.posF32XYZ != 0 || d.posS16XYZ != 0) && d.tag != 0 && d.vtxCount > 0 && verts_readable(d.vertRange)) {
        std::vector<uint8_t> tmp(d.vertRange.size);
        memcpy(tmp.data(), frame.verts.data() + d.vertRange.offset, d.vertRange.size);
        if (interp::patch_vertices(d.tag, d.vtxCount, d.vtxStride, d.posOffset, d.posS16XYZ != 0, d.posFrac,
                                   d.deformF32OffsetMask, frame.verts.data() + d.vertRange.offset, tmp.data(), alpha,
                                   d.pop)) {
          d.vertRange = push_verts(tmp.data(), tmp.size(), 4);
          interp::note_disposition(d.pop, interp::Disposition::Paired);
          continue;
        }
      }
      bool firstEverSighting = false;
      if (interp::patch_draw(d.tag, d.vtxCount, snap.data() + d.uniformRange.offset, dst, d.uniformRange.size,
                             d.mtxPosOffset, d.mtxNrmOffset, alpha, d.texMtxCamMask, d.pnMtxSlot, d.pop,
                             &firstEverSighting)) {
        interp::note_disposition(d.pop, interp::Disposition::Paired);
      } else if (d.ortho == 0) {
        // Perspective only. An orthographic draw's matrix is not model x view, so a camera delta
        // does not belong in it — it would slide the HUD bodily every other frame.
        interp::patch_camera_only(snap.data() + d.uniformRange.offset, dst, d.uniformRange.size, d.mtxPosOffset,
                                  d.mtxNrmOffset, d.texMtxCamMask);
        // The treatment is the same either way — the camera delta is what an unpaired draw needs.
        // The AUDIT distinguishes them, because a birth is unpairable by construction and a miss on
        // an object that drew before is a defect, and one number cannot say which happened.
        interp::note_disposition(d.pop, firstEverSighting ? interp::Disposition::CameraOnlyBirth
                                                          : interp::Disposition::CameraOnly);
      }
    }
  }
  interp::end_tick();
  // Pairing coverage on a slow cadence. Without it, "interpolation is on" and "interpolation is on
  // and pairing nothing, so every object snaps" produce the same smooth-looking log and the same
  // doubled present count.
  static long s_ticks = 0;
  if (!resampling && (++s_ticks % 300) == 0) {
    interp::report();
    indexed_interp::report();
  }
  return true;
}

bool install_replay_snapshot(bool consume) {
  ZoneScoped;
  if (!g_replaySnapshot.valid) {
    Log.error(
        "install_replay_snapshot: no snapshot has been captured; the replay frame would present "
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
  frame.renderPasses = consume ? std::move(g_replaySnapshot.renderPasses) : g_replaySnapshot.renderPasses;
  const size_t uniformSize = g_replaySnapshot.uniforms.size();
  frame.uniforms.append(g_replaySnapshot.uniforms.data(), uniformSize);
  ASSERT(frame.uniforms.size() == uniformSize, "Replay uniform block landed at {} bytes, expected {}",
         frame.uniforms.size(), uniformSize);
  // RESERVE the first emission's staging, so an overlay recorded into this packet appends above the
  // game's bytes instead of over them, and seed `copied` to the same mark so the reserved prefix —
  // this packet's own staging buffer, holding some other frame's leftovers — is never copied down
  // into the global buffers the copied draws read from. See the block comment above ReplaySnapshot.
  const auto reserve = [](ByteBuffer& buf, uint32_t& copied, uint32_t& prefix, uint32_t size) {
    if (size != 0) {
      (void)buf.append_uninitialized(size);
    }
    copied = size;
    prefix = size;
  };
  reserve(frame.verts, frame.copied.verts, frame.replayPrefix.verts, g_replaySnapshot.verts);
  reserve(frame.indices, frame.copied.indices, frame.replayPrefix.indices, g_replaySnapshot.indices);
  reserve(frame.storage, frame.copied.storage, frame.replayPrefix.storage, g_replaySnapshot.storage);
  // CONSUME the snapshot. A tick whose capture failed must fall back to presenting once, never to
  // re-presenting a stale frame — a stale replay is invisible on a static scene and looks like a
  // one-frame stutter on a moving one.
  if (consume) {
    g_replaySnapshot = {};
  }
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
  prevPass.resolveDest = g_pendingResolveDest;
  if (g_pendingResolveDest != nullptr) {
    g_copyResolvedAtPass[g_pendingResolveDest] = g_currentRenderPass;
    g_pendingResolveDest = nullptr;
  }
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
  const bool fullRect = rect.x == 0 && rect.y == 0 && rect.width == prevPass.targetSize.width &&
                        rect.height == prevPass.targetSize.height;
  // A render-pass loadOp clears the whole attachment. That is faithful only for a full-EFB copy;
  // GC clears the COPY RECTANGLE after a partial GXCopyTex. Hx_Test5 depends on this literally:
  // it captures and clears eighty adjacent 64x64 tiles, so a full clear after tile zero makes the
  // remaining seventy-nine captures black.
  const bool attachmentColorClear = fullRect && clearColor && clearAlpha;
  const bool attachmentDepthClear = fullRect && clearDepth;
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
      .clearColor = attachmentColorClear,
      .clearDepth = attachmentDepthClear,
      .hasDepth = prevPass.hasDepth,
      .hasStencil = prevPass.hasStencil,
  };
  newPass.commands.reserve(2048);
  current_render_passes().emplace_back(std::move(newPass));
  ++g_currentRenderPass;

  const bool drawColor = clearColor && !attachmentColorClear;
  const bool drawAlpha = clearAlpha && !attachmentColorClear;
  const bool drawDepth = clearDepth && !attachmentDepthClear;
  if (drawColor || drawAlpha || drawDepth) {
    // A partial clear, or a full clear of only color/alpha, must be a draw. LoadOp cannot express
    // either contract. The clear shader writes depth too, so partial depth clears stay rectangular.
    push_draw_command(clear::DrawData{
        .pipeline = pipeline_ref(clear::PipelineConfig{
            .msaaSamples = msaaSamples,
            .clearColor = drawColor,
            .clearAlpha = drawAlpha,
            .clearDepth = drawDepth,
        }),
        .color =
            wgpu::Color{
                .r = clearColorValue.x(),
                .g = clearColorValue.y(),
                .b = clearColorValue.z(),
                .a = clearColorValue.w(),
            },
        .depth = clearDepthValue,
        .rectEnabled = !fullRect,
        .rect = rect,
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
  // Sized StorageBufferSize + PersistentStorageSize: the low region mirrors staging 1:1, the high
  // region is the persistent geometry arena (see PersistentStorageSize in common.hpp).
  createBuffer(g_storageBuffer, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
               StorageBufferSize + PersistentStorageSize, "Shared Storage Buffer");
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

static void settle_staging_maps_for_shutdown() {
  const auto deadline = PresentClock::now() + GpuWaitTimeout;
  for (;;) {
    bool mapping = false;
    for (const auto& state : s_mappingStates) {
      mapping |= state.load(std::memory_order_acquire) == BufferMapState::Mapping;
    }
    if (!mapping) {
      break;
    }
    if (PresentClock::now() >= deadline) {
      FATAL("Timed out after {}s waiting for staging-buffer map callbacks during shutdown", GpuWaitTimeout.count());
    }
    render_worker::enqueue_work(process_events);
    render_worker::synchronize();
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }

  // Mapped-at-rest is the steady-state pool policy. Explicitly unmap before dropping the handles;
  // destroying a buffer with MapAsync still pending produced an Aborted callback and a synthetic
  // device-loss warning on every otherwise-clean recomp shutdown.
  for (size_t slot = 0; slot < g_stagingBuffers.size(); ++slot) {
    if (s_mappingStates[slot].load(std::memory_order_acquire) == BufferMapState::Mapped) {
      g_stagingBuffers[slot].Unmap();
      s_mappingStates[slot].store(BufferMapState::Unmapped, std::memory_order_release);
    }
  }
}

void shutdown() {
  render_worker::synchronize();
  settle_staging_maps_for_shutdown();
  render_worker::shutdown();
  g_processEventsQueued.store(false, std::memory_order_release);
  g_lastPresentNs.store(0, std::memory_order_release);
  g_presentPeriodNs.store(0, std::memory_order_release);
  g_cpuFrameTimeNs.store(0, std::memory_order_release);
  g_cpuFrameStart = {};
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
  // The persistent arena's offsets are only meaningful for the g_storageBuffer being torn down
  // here. Keeping them would hand out ranges into a destroyed buffer on the next device bring-up.
  persistent_storage_reset();
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
  const auto deadline = PresentClock::now() + GpuWaitTimeout;
  map_staging_buffer(slot);
  while (true) {
    const auto mappingState = s_mappingStates[slot].load(std::memory_order_acquire);
    if (mappingState == BufferMapState::Mapped) {
      return true;
    }
    if (mappingState == BufferMapState::Unmapped) {
      return false;
    }
    if (PresentClock::now() >= deadline) {
      FATAL("Timed out after {}s waiting for staging-buffer slot {} to map", GpuWaitTimeout.count(), slot);
    }
    wait_for_gpu_progress(std::chrono::milliseconds{1});
  }
}

static size_t acquire_frame_slot() {
  ZoneScopedN("Acquire frame slot");
  const auto waitStart = PresentClock::now();
  const auto deadline = waitStart + GpuWaitTimeout;
  while (true) {
    if (const auto slot = g_frameSlots.try_acquire()) {
      const auto waitDuration = PresentClock::now() - waitStart;
      const double waitMs = std::chrono::duration<double, std::milli>{waitDuration}.count();
      TracyPlot("aurora: frameSlotWaitMs", waitMs);
      return *slot;
    }
    if (PresentClock::now() >= deadline) {
      FATAL("Timed out after {}s waiting to acquire a GPU frame slot", GpuWaitTimeout.count());
    }
    wait_for_gpu_progress(std::chrono::microseconds{100});
  }
}

static std::optional<size_t> acquire_mapped_staging_buffer() {
  ZoneScopedN("Acquire mapped staging buffer");
  const auto deadline = PresentClock::now() + GpuWaitTimeout;
  while (true) {
    if (auto slot = g_stagingSlots.try_acquire()) {
      if (wait_for_staging_buffer(*slot)) {
        return *slot;
      }
      g_stagingSlots.release(*slot);
      return std::nullopt;
    }
    if (PresentClock::now() >= deadline) {
      FATAL("Timed out after {}s waiting to acquire a staging-buffer slot", GpuWaitTimeout.count());
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
  // The copy-ordering record describes ONE frame; carrying it over would let last frame's sample
  // order decide this frame's copies.
  g_copySampledAtPass.clear();
  g_copyResolvedAtPass.clear();
  g_pendingResolveDest = nullptr;
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
    // The copied draws point at vertex/index/storage ranges that only still mean anything because
    // nothing wrote those global buffers in between. An overlay recorded into this packet is fine
    // — install_replay_snapshot reserved the first emission's high-water mark for exactly that —
    // but a byte written BELOW that mark would land on top of the tick's own geometry and render
    // one tick's vertices with another's indices: plausible garbage rather than an error. The
    // staging cursors only ever move forward, so this catches a packet that reached end_frame
    // without the reservation at all.
    ASSERT(frame.verts.size() >= frame.replayPrefix.verts && frame.indices.size() >= frame.replayPrefix.indices &&
               frame.storage.size() >= frame.replayPrefix.storage,
           "Replay emission wrote below its reserved prefix: verts {}/{} indices {}/{} storage {}/{} bytes",
           frame.verts.size(), frame.replayPrefix.verts, frame.indices.size(), frame.replayPrefix.indices,
           frame.storage.size(), frame.replayPrefix.storage);
    // Say it ONCE when an overlay actually rides on a replay emission. Without this the fix for
    // "the RmlUi menu aborted the run" and a run where the menu simply never drew produce the same
    // silence, and the reservation above would read as verified by a run that never exercised it.
    if (frame.verts.size() > frame.replayPrefix.verts || frame.indices.size() > frame.replayPrefix.indices ||
        frame.storage.size() > frame.replayPrefix.storage) {
      static bool s_reported = false;
      if (!s_reported) {
        s_reported = true;
        Log.info(
            "replay emission carried its own geometry above the reserved prefix: verts +{} indices +{} storage "
            "+{} bytes (an overlay drawing on an in-between frame)",
            frame.verts.size() - frame.replayPrefix.verts, frame.indices.size() - frame.replayPrefix.indices,
            frame.storage.size() - frame.replayPrefix.storage);
      }
    }
    ASSERT(frame.copied.verts >= frame.replayPrefix.verts && frame.copied.indices >= frame.replayPrefix.indices &&
               frame.copied.storage >= frame.replayPrefix.storage,
           "Replay emission is set to copy staging below its reserved prefix: verts {}/{} indices {}/{} storage "
           "{}/{} bytes — that copy would overwrite the geometry this frame replays",
           frame.copied.verts, frame.replayPrefix.verts, frame.copied.indices, frame.replayPrefix.indices,
           frame.copied.storage, frame.replayPrefix.storage);
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
      // Per-pass draw counts. The long-open question is whether a redundant full-scene "ghost pass"
      // doubles the frame's work; two passes with comparable draw counts is what that would look
      // like, and no amount of totals can show it.
      std::string perPass;
      for (const auto& p : frame.renderPasses) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%zu ", p.commands.size());
        perPass += buf;
      }
      Log.info("per-pass command counts: {}", perPass);
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
  // The data-keyed upload cache holds ranges into the frame packet's storage buffer, which is
  // reset with the frame — so it has exactly the lifetime of the per-slot cachedRange above and
  // must be cleared in the same place. Outliving the frame would hand out offsets into a buffer
  // that has since been rewound.
  gx::array_upload_cache_clear();
  // The draw tag must not survive a frame. If the emitter stops tagging, a leaked tag would keep
  // stamping the previous object's identity onto every later draw, and interpolation would then
  // pair those draws with the wrong object's matrices — wrong, plausible, and silent.
  CHECK(gx::fifo::g_pendingDrawExact == 0,
        "frame ended with an unconsumed GX_AURORA_DRAW_EXACT marker and no following draw");
  CHECK(gx::fifo::g_pendingDrawIndexedKeys.empty(),
        "frame ended with {} unconsumed GX_AURORA_DRAW_INDEXED_KEYS and no following draw",
        gx::fifo::g_pendingDrawIndexedKeys.size());
  CHECK(gx::fifo::g_pendingDrawIndexedDeform == 0,
        "frame ended with an unconsumed indexed-deform marker and no following draw");
  gx::fifo::g_pendingDrawTag = 0;
  gx::fifo::g_pendingDrawIndexedDeform = 0;
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
  const AuroraGpuSubmitInfo submitProbe = build_submit_probe(frame);
  // A replay emission pushes no verts/indices/storage and records no draws, so publishing its
  // stats would make every second sample read zero — Tracy plots and the imgui overlay would
  // alternate real/zero and read exactly like a frame-dropping defect. The numbers the user cares
  // about belong to the frame the game actually drew, so leave the last real publish standing.
  const bool publishStats = !frame.replayEmission;
  render_worker::enqueue_end_frame(
      frameId, [frameSlot, stagingSlot, publishStats, submitProbe, callback = std::move(callback)]() mutable {
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
          callback(encoder, submitProbe);
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
    // THE SAME FIELD, READ TWICE, MUST GIVE THE SAME ANSWER. A crash in tex_copy_conv::execute
    // faulting at 0x10 (a null TextureRef + offsetof attachmentTextureView) reached here past the
    // `if` above, which means the handle was non-null when tested and null by the time the request
    // was built. Either that is true — and this is a lifetime race on the RenderPass, not a missing
    // resolve target — or the two loads are of different things and the whole diagnosis is wrong.
    // Reading it into a local and comparing is what tells those apart; a null-check that merely
    // skipped the copy would hide which one it is.
    const TextureRef* const dstAtCheck = passInfo.resolveTarget.get();
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
    if (convReq.dst.get() != dstAtCheck) {
      Log.fatal(
          "EFB copy target changed underneath this pass: it was {} when tested and {} when "
          "the request was built, in the SAME function with no intervening store. The "
          "RenderPass this reads from has been destroyed or reused by another thread while "
          "the render worker was encoding it. pass fmt {}, frame {}, replay {}",
          static_cast<const void*>(dstAtCheck), static_cast<const void*>(convReq.dst.get()),
          static_cast<int>(passInfo.resolveFormat), frame.frameId, frame.replayEmission);
    }
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
    const double presentPeriodMs = static_cast<double>(g_presentPeriodNs.load(std::memory_order_acquire)) / 1'000'000.0;
    TracyPlot("aurora: presentPeriodMs", presentPeriodMs);
  }
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
      case ShaderType::GX: {
        // SB_VIZ_TAG=untagged|tagged (diagnostic): draw only one class, so "which geometry is not
        // reaching the tag seam" can be ANSWERED BY LOOKING rather than inferred from draw counts.
        // Counts have been the wrong denominator twice in this arc — what matters is screen area,
        // and the eye reads that off a frame dump immediately.
        static const int s_vizTag = [] {
          const char* e = std::getenv("SB_VIZ_TAG");
          if (e == nullptr || e[0] == '\0')
            return 0;
          if (std::strcmp(e, "untagged") == 0)
            return 1;
          if (std::strcmp(e, "tagged") == 0)
            return 2;
          Log.error("SB_VIZ_TAG={} is not 'untagged' or 'tagged'; showing everything", e);
          return 0;
        }();
        if ((s_vizTag == 1 && draw.gx.tag != 0) || (s_vizTag == 2 && draw.gx.tag == 0)) {
          break;
        }
        gx::render(draw.gx, pass);
        break;
      }
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

// --- Persistent geometry arena -------------------------------------------------------------
// See PersistentStorageSize in common.hpp for why this exists and why the cheaper trick (relying
// on staging retaining its bytes) is unsound.
namespace {
struct PersistentArrayEntry {
  uint32_t offset;
  uint32_t size;
  uint64_t hash;
};
std::unordered_map<ArrayUploadKey, PersistentArrayEntry, ArrayUploadKeyHash> sPersistentArrays;
uint64_t sPersistentTop = StorageBufferSize;
} // namespace

uint64_t persistent_storage_used() { return sPersistentTop - StorageBufferSize; }
size_t persistent_storage_entries() { return sPersistentArrays.size(); }

static void write_storage_region(uint64_t offset, const uint8_t* data, size_t length) {
  // WriteBuffer requires a 4-byte multiple size and offset. Array extents are not guaranteed to
  // be one, and rounding the size UP would read past the end of the game's array — so the tail is
  // copied through a small padded scratch instead of over-reading guest memory.
  const size_t whole = length & ~size_t{3};
  if (whole > 0) {
    g_device.GetQueue().WriteBuffer(g_storageBuffer, offset, data, whole);
  }
  const size_t tail = length - whole;
  if (tail > 0) {
    uint8_t pad[4] = {};
    std::memcpy(pad, data + whole, tail);
    g_device.GetQueue().WriteBuffer(g_storageBuffer, offset + whole, pad, 4);
  }
}

// Returns a range in the persistent arena, uploading only when the content hash differs from what
// is already there. `outUploaded` reports whether a GPU write actually happened. A returned range
// with size 0 means the arena is full and the caller must fall back to the per-frame path — never
// a silent partial result.
Range push_storage_persistent(const uint8_t* data, size_t length, ArrayUploadKey key, uint64_t contentHash,
                              bool* outUploaded) {
  if (outUploaded != nullptr) {
    *outUploaded = false;
  }
  if (length == 0 || data == nullptr) {
    return {0, 0};
  }
  const auto it = sPersistentArrays.find(key);
  if (it != sPersistentArrays.end() && it->second.size == length) {
    if (it->second.hash != contentHash) {
      // Same array, rewritten in place by the game: refresh the SAME region so every draw that
      // already resolved this offset stays correct.
      write_storage_region(it->second.offset, data, length);
      it->second.hash = contentHash;
      if (outUploaded != nullptr) {
        *outUploaded = true;
      }
    }
    return {it->second.offset, static_cast<uint32_t>(length)};
  }
  const uint64_t alignment = g_cachedLimits.minStorageBufferOffsetAlignment;
  const uint64_t off = AURORA_ALIGN(sPersistentTop, alignment);
  // Reserve a 4-byte MULTIPLE. write_storage_region finishes a non-multiple length with a 4-byte
  // padded write, which would otherwise spill up to 3 bytes into whatever is allocated next.
  const uint64_t reserved = AURORA_ALIGN(length, 4);
  if (off + reserved > StorageBufferSize + PersistentStorageSize) {
    return {0, 0}; // full — caller falls back
  }
  sPersistentTop = off + reserved;
  write_storage_region(off, data, length);
  sPersistentArrays[key] = {static_cast<uint32_t>(off), static_cast<uint32_t>(length), contentHash};
  if (outUploaded != nullptr) {
    *outUploaded = true;
  }
  return {static_cast<uint32_t>(off), static_cast<uint32_t>(length)};
}

void persistent_storage_reset() {
  sPersistentArrays.clear();
  sPersistentTop = StorageBufferSize;
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

namespace aurora {
// Which mapped staging region a capacity belongs to. Used only by the overflow message in
// common.hpp, which otherwise reports a bare number and leaves the reader to grep the constants.
// Sizes are distinct today; if two are ever made equal this reports the ambiguity rather than
// silently picking one, because a confident wrong name is worse than no name.
const char* aurora_gfx_staging_region_name(size_t capacity) {
  const char* found = nullptr;
  const struct {
    size_t size;
    const char* name;
  } kRegions[] = {
      {gfx::UniformBufferSize, "uniform"}, {gfx::VertexBufferSize, "vertex"},         {gfx::IndexBufferSize, "index"},
      {gfx::StorageBufferSize, "storage"}, {gfx::TextureUploadSize, "textureUpload"},
  };
  for (const auto& r : kRegions) {
    if (r.size != capacity)
      continue;
    if (found != nullptr)
      return "AMBIGUOUS (two staging regions share this capacity)";
    found = r.name;
  }
  return found != nullptr ? found : "UNKNOWN (capacity matches no staging region constant)";
}
} // namespace aurora
