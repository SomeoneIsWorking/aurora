#include <aurora/aurora.h>
#include <dlfcn.h>
#include "renderdoc_app.h"

#ifdef AURORA_ENABLE_GX
#include "gfx/common.hpp"
#include "gfx/render_worker.hpp"
#include "gx/command_processor.hpp"
#include "gx/fifo.hpp"
#include "imgui.hpp"
#include "webgpu/gpu.hpp"
#include "webgpu/gpu_prof.hpp"
#include <webgpu/webgpu_cpp.h>
#endif

#ifdef AURORA_ENABLE_RMLUI
#include "rmlui.hpp"
#endif

#include "input.hpp"
#include "internal.hpp"
#include "window.hpp"

#include <SDL3/SDL_filesystem.h>
#include <magic_enum.hpp>

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <memory>
#include <vector>

#include "system_info.hpp"
#include "tracy/Tracy.hpp"

// Legacy two-presentation pacing hook, retained for standalone diagnostic users.
extern "C" __attribute__((weak)) void aurora_replay_midpoint();
// General host hook fired before each retained-snapshot sample. The host uses it to advance only
// presentation-side state; game simulation remains untouched.
extern "C" __attribute__((weak)) void aurora_replay_sample(float alpha, unsigned index, unsigned count);

namespace aurora {
static AuroraFrameSink g_frameSink = nullptr;
static void* g_frameSinkUser = nullptr;
static int g_frameSinkEvery = 0;
// aurora_set_dump_tag: a short role label the caller stamps onto the next dump's filename, so a
// dump series is self-describing rather than needing its files identified by inference.
static char g_dumpTag[32] = {0};

AuroraConfig g_config;
uint32_t g_sdlCustomEventsStart;
char g_gameName[4];

namespace {
Module Log("aurora");

// On-screen picture aspect (see aurora_set_present_aspect). 4:3 = the GameCube TV picture.
static uint32_t g_presentAspectW = 640;
static uint32_t g_presentAspectH = 480;
static std::atomic_bool g_presentationEnabled = true;

void set_present_aspect(uint32_t width, uint32_t height) {
  if (width == 0 || height == 0) {
    Log.error("aurora_set_present_aspect({}, {}): neither may be zero", width, height);
    return;
  }
  g_presentAspectW = width;
  g_presentAspectH = height;
}

bool presentation_enabled_impl() noexcept { return g_presentationEnabled.load(std::memory_order_acquire); }

void set_presentation_enabled_impl(bool enabled) noexcept {
  if (g_presentationEnabled.exchange(enabled, std::memory_order_acq_rel) == enabled) {
    return;
  }
#ifdef AURORA_ENABLE_GX
  gfx::render_worker::synchronize();
  if (!enabled) {
    webgpu::release_surface();
  }
#endif
  Log.info("operating-system presentation {}", enabled ? "enabled" : "disabled; rendering offscreen as oracle");
}

// ---- SB_RDOC: RenderDoc in-application capture trigger ----------------------
// SB_RDOC=<present#> arms a single-frame RenderDoc capture: present 0 = the
// first frame ever rendered (TriggerCapture fires at init), N>0 = the frame
// AFTER the Nth present. librenderdoc.so must be dlopen'd BEFORE any Vulkan
// init for its layer to inject — hence the call at the top of initialize().
// SB_RDOC_PATH sets the capture file template (default scratch/rdoc/aurora).
// Open the resulting .rdc in qrenderdoc.
RENDERDOC_API_1_6_0* s_rdocApi = nullptr;
long s_rdocTarget = -1;
long s_rdocPresentCount = 0;

void sb_rdoc_init() {
  const char* e = std::getenv("SB_RDOC");
  if (e == nullptr || e[0] == '\0') {
    return;
  }
  s_rdocTarget = std::atol(e);
  void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_GLOBAL);
  if (mod == nullptr) {
    mod = dlopen("/usr/lib64/renderdoc/librenderdoc.so", RTLD_NOW | RTLD_GLOBAL);
  }
  if (mod == nullptr) {
    Log.error("[rdoc] dlopen librenderdoc.so failed: {}", dlerror());
    s_rdocTarget = -1;
    return;
  }
  auto getApi = reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"));
  if (getApi == nullptr || getApi(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void**>(&s_rdocApi)) != 1) {
    Log.error("[rdoc] RENDERDOC_GetAPI failed");
    s_rdocTarget = -1;
    return;
  }
  const char* p = std::getenv("SB_RDOC_PATH");
  s_rdocApi->SetCaptureFilePathTemplate(p != nullptr && p[0] != '\0' ? p : "scratch/rdoc/aurora");
  if (s_rdocTarget == 0) {
    // Explicit capture (works headless, where no WSI present delimits frames):
    // start NOW; ended by the first sb_rdoc_on_present().
    s_rdocApi->StartFrameCapture(nullptr, nullptr);
    Log.info("[rdoc] StartFrameCapture (frame 0)");
  }
  Log.info("[rdoc] armed: capture frame after present {}", s_rdocTarget);
}

void sb_rdoc_on_present() {
  if (s_rdocApi == nullptr || s_rdocTarget < 0) {
    return;
  }
  if (s_rdocPresentCount == s_rdocTarget) {
    const unsigned ok = s_rdocApi->EndFrameCapture(nullptr, nullptr);
    Log.info("[rdoc] EndFrameCapture at present {} -> {}", s_rdocPresentCount, ok);
    s_rdocTarget = -1; // one-shot
  }
  ++s_rdocPresentCount;
  if (s_rdocPresentCount == s_rdocTarget) {
    s_rdocApi->StartFrameCapture(nullptr, nullptr);
    Log.info("[rdoc] StartFrameCapture at present {}", s_rdocPresentCount);
  }
}
// -----------------------------------------------------------------------------

#ifdef AURORA_ENABLE_GX
// GPU
using webgpu::g_device;
using webgpu::g_queue;
using webgpu::g_surface;

uint32_t clamp_scissor_coord(double value, uint32_t maximum) noexcept {
  if (!std::isfinite(value)) {
    return 0;
  }
  return static_cast<uint32_t>(std::clamp(value, 0.0, static_cast<double>(maximum)));
}

void set_present_viewport(const wgpu::RenderPassEncoder& pass, const gfx::Viewport& viewport, uint32_t surfaceWidth,
                          uint32_t surfaceHeight) noexcept {
  pass.SetViewport(viewport.left, viewport.top, viewport.width, viewport.height, viewport.znear, viewport.zfar);
  const auto scissorX = clamp_scissor_coord(std::floor(viewport.left), surfaceWidth);
  const auto scissorY = clamp_scissor_coord(std::floor(viewport.top), surfaceHeight);
  const auto scissorRight = clamp_scissor_coord(std::ceil(viewport.left + viewport.width), surfaceWidth);
  const auto scissorBottom = clamp_scissor_coord(std::ceil(viewport.top + viewport.height), surfaceHeight);
  pass.SetScissorRect(scissorX, scissorY, scissorRight - scissorX, scissorBottom - scissorY);
}
#endif

#ifdef AURORA_ENABLE_GX
constexpr std::array PreferredBackendOrder{
#ifdef ENABLE_BACKEND_WEBGPU
    BACKEND_WEBGPU,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D12
    BACKEND_D3D12,
#endif
#ifdef DAWN_ENABLE_BACKEND_METAL
    BACKEND_METAL,
#endif
#ifdef DAWN_ENABLE_BACKEND_VULKAN
    BACKEND_VULKAN,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D11
    BACKEND_D3D11,
#endif
// #ifdef DAWN_ENABLE_BACKEND_DESKTOP_GL
//     BACKEND_OPENGL,
// #endif
#ifdef DAWN_ENABLE_BACKEND_OPENGLES
    BACKEND_OPENGLES,
#endif
#ifdef DAWN_ENABLE_BACKEND_NULL
    BACKEND_NULL,
#endif
};
#else
constexpr std::array<AuroraBackend, 0> PreferredBackendOrder{};
#endif

bool g_initialFrame = false;

AuroraInfo initialize(int argc, char* argv[], const AuroraConfig& config) noexcept {
  g_presentationEnabled.store(true, std::memory_order_release);
  g_config = config;
  Log.info("Aurora initializing");
  log_system_information();
  // SB_RDOC=<present#>: load the RenderDoc in-application API BEFORE any
  // graphics init (dlopen'ing librenderdoc.so this early injects its Vulkan
  // layer) and arm a single-frame capture at the given present index (see
  // end_frame). Capture files land under SB_RDOC_PATH (default scratch-less
  // "aurora_rdoc" template in cwd).
  sb_rdoc_init();
  if (g_config.appName == nullptr) {
    g_config.appName = "Aurora";
  } else {
    g_config.appName = strdup(g_config.appName);
  }
  if (g_config.userPath == nullptr) {
    g_config.userPath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.userPath = strdup(g_config.userPath);
  }
  if (g_config.cachePath == nullptr) {
    g_config.cachePath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.cachePath = strdup(g_config.cachePath);
  }
  if (g_config.resourcesPath == nullptr) {
    g_config.resourcesPath = SDL_GetBasePath();
  } else {
    g_config.resourcesPath = strdup(g_config.resourcesPath);
  }
  if (g_config.msaa == 0) {
    g_config.msaa = 1;
  }
  if (g_config.maxTextureAnisotropy == 0) {
    g_config.maxTextureAnisotropy = 16;
  }
  ASSERT(window::initialize(), "Error initializing window");

  g_sdlCustomEventsStart = SDL_RegisterEvents(2);
  ASSERT(g_sdlCustomEventsStart, "Failed to allocate user events: {}", SDL_GetError());
  ASSERT(window::initialize_event_watch(), "Error initializing SDL event watch");

#ifdef AURORA_ENABLE_GX
  /* Attempt to create a window using the calling application's desired backend */
  AuroraBackend selectedBackend = config.desiredBackend;
  bool windowCreated = false;
  if (selectedBackend != BACKEND_AUTO && window::create_window(selectedBackend)) {
    if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
      windowCreated = true;
    } else {
      window::destroy_window();
    }
  }

  if (!windowCreated) {
    for (const auto backendType : PreferredBackendOrder) {
      selectedBackend = backendType;
      if (!window::create_window(selectedBackend)) {
        continue;
      }
      if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
        windowCreated = true;
        break;
      } else {
        window::destroy_window();
      }
    }
  }

  ASSERT(windowCreated, "Error creating window: {}", SDL_GetError());

  // Initialize SDL_Renderer for ImGui when we can't use a Dawn backend
  if (webgpu::g_backendType == wgpu::BackendType::Null) {
    ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
  }
#else
  AuroraBackend selectedBackend = BACKEND_NULL;
  ASSERT(window::create_window(BACKEND_NULL), "Error creating window: {}", SDL_GetError());
  ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
#endif

  // SB_HEADLESS=1: keep the window hidden — diagnostic/CI runs render and
  // dump frames without ever mapping an X11 window. See window::is_headless().
  if (!window::is_headless()) {
    window::show_window();
  }

#ifdef AURORA_ENABLE_GX
  gfx::initialize();

  imgui::create_context();
#endif
  const auto size = window::get_window_size();
  Log.info("Using framebuffer size {}x{} scale {}", size.fb_width, size.fb_height, size.scale);
#ifdef AURORA_ENABLE_GX
  if (g_config.imGuiInitCallback != nullptr) {
    g_config.imGuiInitCallback(&size);
  }
  imgui::initialize();
#endif

#ifdef AURORA_ENABLE_RMLUI
  rmlui::initialize(size);
#endif

  g_initialFrame = true;
  g_config.desiredBackend = selectedBackend;
  return {
      .backend = selectedBackend,
      .userPath = g_config.userPath,
      .cachePath = g_config.cachePath,
      .window = window::get_sdl_window(),
      .windowSize = size,
  };
}

void shutdown() noexcept {
#ifdef AURORA_ENABLE_GX
  gfx::render_worker::synchronize();
#ifdef AURORA_ENABLE_RMLUI
  rmlui::shutdown();
#endif
  imgui::shutdown();
  gfx::shutdown();
  webgpu::shutdown();
#endif
  input::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  ZoneScoped;
  if (g_initialFrame) {
    g_initialFrame = false;
    input::initialize();
  }
  return window::poll_events();
}

bool begin_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  if (window::is_headless() || !presentation_enabled_impl()) {
    // Headless: the surface/swapchain is never touched (see is_headless()) --
    // just gate on pause and fall through to render into the offscreen
    // target. is_presentable()/refresh_surface()/release_surface() all deal
    // with the WSI surface and must not run here. Presentation-disabled hosts keep a real window
    // for their own swapchain while Aurora remains an offscreen oracle.
    if (window::is_paused()) {
      return false;
    }
  } else {
    if (!window::is_presentable()) {
      webgpu::release_surface();
      return false;
    }
    if (window::is_paused()) {
      return false;
    }
    if (!webgpu::surface_available()) {
      webgpu::refresh_surface(true);
      if (!webgpu::surface_available()) {
        return false;
      }
    }
  }

  imgui::new_frame(window::get_window_size());
  if (!gfx::begin_frame()) {
    return false;
  }
#endif
  return true;
}

// `replayEmission` marks an additional emission of a tick: a packet whose
// content was installed by gfx::install_replay_snapshot rather than recorded from the game. It
// takes the whole present path below (that is the point — presenting is what it exists to do) but
// must skip the two steps that would consume the NEXT tick's work:
//   * gx::fifo::drain() — the game has not run since the first emission, so anything sitting in
//     the fifo belongs to the tick that has not started yet. Draining it here would both steal
//     those commands and record geometry into a packet that must stay a pure copy.
//   * gfx::finish() — install_replay_snapshot already enqueued every pass and closed the packet;
//     finish() would find no open pass and do nothing, but calling it would still append its
//     MaxUniformSize padding to a uniform block that is supposed to be exactly the snapshot.
void end_frame_impl(bool replayEmission) noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  if (!replayEmission) {
    gx::fifo::drain();
  }
  if (!replayEmission) {
    gfx::finish();
    if (gfx::replay_present_enabled()) {
      // Between finish() and end_frame(): finish() has sealed and enqueued the last pass, so every
      // pass in the packet is complete, and the staging buffer is still mapped.
      gfx::capture_replay_snapshot();
      // Then move THIS emission to the first in-between pose. The snapshot already holds the tick's
      // true matrices, so retained replay samples can independently reconstruct later poses and the
      // final replay still shows the tick exactly. Off by default (AURORA_INTERP_ALPHA
      // unset) so the doubled present can be exercised — and its EFB idempotence checked — without
      // interpolation confusing the picture.
      if (gfx::interp_alpha() >= 0.0f) {
        const unsigned count = gfx::replay_presentation_count();
        const float alpha = count == 2 ? gfx::interp_alpha() : 1.0f / static_cast<float>(count);
        gfx::interpolate_recorded_frame(alpha);
      }
    }
  }
  auto imguiDrawData = imgui::freeze();

  // Snapshot the complete refcounted source on the game thread. GXCopyDisp replaces the global
  // display-copy texture on this thread while the render worker encodes the previous packet. The
  // old code took an unused reference here and reread the mutable global inside the worker, racing
  // WebGPU handle assignment/destruction. A value copy keeps this frame's texture/view/sampler
  // alive and makes the worker consume exactly the source selected at this frame boundary.
  const auto presentSource = webgpu::present_source();
  // Aspect from the TV's fixed 4:3 picture, NOT presentSource's own texel
  // size: VI scan-out stretches whatever XFB the display copy produced
  // across the physical picture, erasing the XFB's own aspect by design.
  // The render mode's raw viWidth/viHeight must not drive this either —
  // SMS programs viWidth=660/viHeight=448 (overscan-domain values), yet a
  // real TV and the Dolphin oracle both show the full 4:3 picture.
  // presentSource.size stays authoritative for sampling inside the resample
  // pass; only the on-screen rectangle's aspect is the TV's.
  // g_presentAspect defaults to the TV's 4:3; a runtime rendering anamorphically
  // (widescreen) sets it via aurora_set_present_aspect so the wide picture is not
  // presented squeezed back into a 4:3 rectangle.
  const auto viewport = webgpu::calculate_present_viewport(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                           webgpu::g_graphicsConfig.surfaceConfiguration.height,
                                                           g_presentAspectW, g_presentAspectH);

  wgpu::BindGroup rmlBindGroup;
  bool rmlOverlay = false;
#if AURORA_ENABLE_RMLUI
  if (rmlui::is_initialized()) {
    auto rmlFrame = rmlui::record_frame(viewport);
    rmlBindGroup = std::move(rmlFrame.bindGroup);
    rmlOverlay = rmlFrame.overlay;
  }
#endif

  gfx::end_frame([presentSource, rmlBindGroup = std::move(rmlBindGroup), rmlOverlay, viewport,
                  imguiDrawData = std::move(imguiDrawData)](wgpu::CommandEncoder& encoder,
                                                            const AuroraGpuSubmitInfo& submitProbe) mutable {
    // SB_DUMP_FRAME=/path/to.raw : one-shot raw dump of the framebuffer, taken
    // SB_DUMP_FRAME_AFTER frames in (default 60). The OUTPUT is always normalized
    // to true RGBA8 (R,G,B,A byte order) regardless of the host surface format, so
    // the file matches its label and any standard RGBA reader is correct. (The raw
    // surface is BGRA8 on most platforms; writing it unconverted caused a recurring
    // false "red/blue swapped / wrong colors" diagnosis — the render was correct,
    // the dump conversion wasn't. See debug_journal/2026-07-11_dump_bgra_mislabel.md.)
    // Per-job dump state: each queued dump owns its buffer/dims/path, so
    // periodic dumps (SB_DUMP_FRAME_EVERY) can overlap safely. The old design
    // reused ONE static buffer/path across dumps: re-arming while the previous
    // MapAsync was still pending destroyed its buffer ("map failed status=4")
    // and let a later dump's dims/path corrupt an earlier callback's write —
    // torn/mislabeled frames in short FIFO-replay runs.
    struct SbDumpJob {
      wgpu::Buffer buffer;
      uint32_t width = 0, height = 0;
      bool swapRB = false; // surface is BGRA8 -> swap R/B to emit RGBA8
      std::string path;    // empty => deliver to the frame sink instead of a file
      AuroraFrameSink sink = nullptr;
      void* sinkUser = nullptr;
    };
    static int s_dumpFramesLeft = -2;
    static const char* s_dumpPath = nullptr;
    // Jobs whose texture->buffer copy was encoded last present; their MapAsync
    // is requested on the NEXT present (mapping a buffer with an unsubmitted
    // pending copy is a usage error).
    static std::vector<std::shared_ptr<SbDumpJob>> s_dumpAwaitingMap;
    // SB_DUMP_FRAME_EVERY=N: instead of a one-shot, dump every N presents to
    // <path>.<seq>.rgba — turbo timing varies run-to-run, so one fixed frame
    // index samples a different game moment each run; a periodic series shows
    // the whole boot/title progression from a single run.
    static int s_dumpEvery = -1;
    static int s_dumpSeq = 0;
    // SB_DUMP_FRAME_COUNT=N: stop the periodic series after N dumps.
    //
    // WHY IT MATTERS AND NOT JUST FOR TIDINESS. SB_DUMP_FRAME_AFTER counts PRESENTS, so two runs
    // that present at different rates reach a different game moment at the same index -- and any
    // feature that adds a present (an interpolated sub-frame) silently makes the one-shot dumps of
    // two configurations incomparable while both still look like "frame 1500". Bounding the series
    // lets a run capture a few CONSECUTIVE presents instead, which is a comparison inside one run
    // at one moment and cannot drift that way. 0 or unset = unbounded, as before.
    static int s_dumpCount = -1;
    // Frame sink (aurora_set_frame_sink): an independent capture cadence, so an in-process
    // parity comparison can run alongside — or without — the file dumps.
    static int s_sinkCountdown = 0;
    if (s_dumpFramesLeft == -2) {
      s_dumpPath = std::getenv("SB_DUMP_FRAME");
      if (s_dumpPath && s_dumpPath[0]) {
        const char* wait = std::getenv("SB_DUMP_FRAME_AFTER");
        s_dumpFramesLeft = wait ? std::atoi(wait) : 60;
        Log.info("SB_DUMP_FRAME armed: dumping after {} more frames to {}", s_dumpFramesLeft, s_dumpPath);
      } else {
        s_dumpFramesLeft = -1;
      }
    }
    if (s_dumpEvery < 0) {
      const char* e = std::getenv("SB_DUMP_FRAME_EVERY");
      s_dumpEvery = (e != nullptr && e[0] != '\0') ? std::atoi(e) : 0;
    }
    if (s_dumpCount < 0) {
      const char* e = std::getenv("SB_DUMP_FRAME_COUNT");
      s_dumpCount = (e != nullptr && e[0] != '\0') ? std::atoi(e) : 0;
    }
    // Request the map for jobs whose copies were submitted last present. The
    // callback owns its job (shared_ptr capture), so any number of dumps can
    // be in flight without clobbering each other.
    for (auto& jobRef : s_dumpAwaitingMap) {
      auto job = jobRef;
      const uint32_t bpr = ((job->width * 4 + 255) / 256) * 256;
      const uint64_t totalBytes = static_cast<uint64_t>(bpr) * job->height;
      job->buffer.MapAsync(wgpu::MapMode::Read, 0, totalBytes, wgpu::CallbackMode::AllowSpontaneous,
                           [job](wgpu::MapAsyncStatus status, wgpu::StringView) {
                             if (status != wgpu::MapAsyncStatus::Success) {
                               Log.error("SB_DUMP_FRAME map failed status={} ({})", static_cast<int>(status),
                                         job->path);
                               return;
                             }
                             const uint32_t bpr = ((job->width * 4 + 255) / 256) * 256;
                             const auto* mapped = static_cast<const uint8_t*>(
                                 job->buffer.GetConstMappedRange(0, static_cast<uint64_t>(bpr) * job->height));
                             if (!mapped) {
                               Log.error("SB_DUMP_FRAME: GetConstMappedRange returned null ({})", job->path);
                               return;
                             }
                             if (job->sink != nullptr) {
                               // Repack into tightly-packed RGBA8: the mapped rows are padded to a
                               // 256-byte stride, which a consumer expecting width*4 would misread.
                               const uint32_t rowBytes = static_cast<uint32_t>(job->width) * 4;
                               std::vector<uint8_t> frame(static_cast<size_t>(rowBytes) * job->height);
                               for (uint32_t y = 0; y < job->height; ++y) {
                                 const uint8_t* src = mapped + static_cast<size_t>(y) * bpr;
                                 uint8_t* dst = frame.data() + static_cast<size_t>(y) * rowBytes;
                                 if (job->swapRB) {
                                   for (uint32_t p = 0; p < rowBytes; p += 4) {
                                     dst[p + 0] = src[p + 2];
                                     dst[p + 1] = src[p + 1];
                                     dst[p + 2] = src[p + 0];
                                     dst[p + 3] = src[p + 3];
                                   }
                                 } else {
                                   std::memcpy(dst, src, rowBytes);
                                 }
                               }
                               job->sink(frame.data(), job->width, job->height, job->sinkUser);
                               job->buffer.Unmap();
                               return;
                             }
                             FILE* f = std::fopen(job->path.c_str(), "wb");
                             if (!f) {
                               Log.error("SB_DUMP_FRAME: fopen failed {}", job->path);
                             } else {
                               const uint32_t rowBytes = static_cast<size_t>(job->width) * 4;
                               std::vector<uint8_t> row; // scratch for optional R/B swap
                               if (job->swapRB)
                                 row.resize(rowBytes);
                               for (uint32_t y = 0; y < job->height; ++y) {
                                 const uint8_t* src = mapped + static_cast<size_t>(y) * bpr;
                                 if (job->swapRB) {
                                   // BGRA8 -> RGBA8 (swap bytes 0<->2 of each pixel)
                                   for (uint32_t p = 0; p < rowBytes; p += 4) {
                                     row[p + 0] = src[p + 2]; // R <- B
                                     row[p + 1] = src[p + 1]; // G
                                     row[p + 2] = src[p + 0]; // B <- R
                                     row[p + 3] = src[p + 3]; // A
                                   }
                                   std::fwrite(row.data(), 1, rowBytes, f);
                                 } else {
                                   std::fwrite(src, 1, rowBytes, f);
                                 }
                               }
                               std::fclose(f);
                               Log.info("SB_DUMP_FRAME: wrote {}x{} RGBA8{} to {} ({} bytes)", job->width, job->height,
                                        job->swapRB ? " (swapped from BGRA8)" : "", job->path,
                                        static_cast<size_t>(job->width) * job->height * 4);
                             }
                             job->buffer.Unmap();
                           });
    }
    s_dumpAwaitingMap.clear();
    if (s_dumpFramesLeft > 0) {
      --s_dumpFramesLeft;
    } else if (s_dumpFramesLeft == 0) {
      const auto& src = presentSource;
      auto job = std::make_shared<SbDumpJob>();
      job->width = src.size.width;
      job->height = src.size.height;
      // The present-source texture is in the host surface format; if that's
      // BGRA8, the mapped bytes are B,G,R,A and must be swapped to R,G,B,A to
      // honor the "RGBA" output contract.
      job->swapRB = (webgpu::g_graphicsConfig.surfaceConfiguration.format == wgpu::TextureFormat::BGRA8Unorm);
      job->path = s_dumpPath;
      if (s_dumpEvery > 0) {
        job->path += "." + std::to_string(s_dumpSeq++);
      }
      // The caller's label for THIS present, appended to the filename.
      //
      // A dump series carries no record of what each present WAS. When a runtime issues more than
      // one kind of present per game tick (a main frame and an interpolated sub-frame), the role of
      // each file has to be inferred from the pattern of the diffs — and inferring it wrongly is
      // silent, because every file looks equally valid. Putting the role in the NAME makes the
      // artifact self-describing, and removes the join between "dump index" and "what the runtime
      // was doing", which is exactly the kind of cross-instrument ordinal join that has produced
      // false findings in this project before.
      if (g_dumpTag[0] != '\0') {
        job->path += ".";
        job->path += g_dumpTag;
      }
      const uint32_t bytesPerRow = ((job->width * 4 + 255) / 256) * 256;
      const wgpu::BufferDescriptor bd{
          .label = "framebuffer dump",
          .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
          .size = static_cast<uint64_t>(bytesPerRow) * job->height,
      };
      job->buffer = webgpu::g_device.CreateBuffer(&bd);
      const wgpu::TexelCopyTextureInfo srcInfo{
          .texture = src.texture,
          .mipLevel = 0,
          .origin = {0, 0, 0},
          .aspect = wgpu::TextureAspect::All,
      };
      const wgpu::TexelCopyBufferInfo dstInfo{
          .layout = {.offset = 0, .bytesPerRow = bytesPerRow, .rowsPerImage = job->height},
          .buffer = job->buffer,
      };
      const wgpu::Extent3D copySize{job->width, job->height, 1};
      encoder.CopyTextureToBuffer(&srcInfo, &dstInfo, &copySize);
      Log.info("SB_DUMP_FRAME: queued dump ({}x{}, bytesPerRow={}) -> {}", job->width, job->height, bytesPerRow,
               job->path);
      s_dumpAwaitingMap.push_back(std::move(job));
      // Re-arm for the next periodic capture (EVERY=1 -> a dump at every
      // present: the countdown consumes one present per unit), or disarm
      // for the one-shot.
      s_dumpFramesLeft = s_dumpEvery > 0 ? s_dumpEvery - 1 : -1;
      if (s_dumpCount > 0 && s_dumpSeq >= s_dumpCount) {
        s_dumpFramesLeft = -1; // series complete; disarm
        Log.info("SB_DUMP_FRAME: series complete after {} dumps (SB_DUMP_FRAME_COUNT)", s_dumpSeq);
      }
    }
    if (g_frameSink != nullptr && g_frameSinkEvery > 0 && --s_sinkCountdown <= 0) {
      s_sinkCountdown = g_frameSinkEvery;
      const auto& src = presentSource;
      auto job = std::make_shared<SbDumpJob>();
      job->width = src.size.width;
      job->height = src.size.height;
      job->swapRB = (webgpu::g_graphicsConfig.surfaceConfiguration.format == wgpu::TextureFormat::BGRA8Unorm);
      job->sink = g_frameSink;
      job->sinkUser = g_frameSinkUser; // path stays empty: routed to the sink
      const uint32_t bytesPerRow = ((job->width * 4 + 255) / 256) * 256;
      const wgpu::BufferDescriptor bd{
          .label = "frame sink readback",
          .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
          .size = static_cast<uint64_t>(bytesPerRow) * job->height,
      };
      job->buffer = webgpu::g_device.CreateBuffer(&bd);
      const wgpu::TexelCopyTextureInfo srcInfo{
          .texture = src.texture,
          .mipLevel = 0,
          .origin = {0, 0, 0},
          .aspect = wgpu::TextureAspect::All,
      };
      const wgpu::TexelCopyBufferInfo dstInfo{
          .layout = {.offset = 0, .bytesPerRow = bytesPerRow, .rowsPerImage = job->height},
          .buffer = job->buffer,
      };
      const wgpu::Extent3D copySize{job->width, job->height, 1};
      encoder.CopyTextureToBuffer(&srcInfo, &dstInfo, &copySize);
      s_dumpAwaitingMap.push_back(std::move(job));
    }
    const bool headless = window::is_headless();
    const bool presentationEnabled = presentation_enabled_impl();
    wgpu::Texture currentTexture;
    wgpu::TextureView currentView;
    auto surfaceStatus = wgpu::SurfaceGetCurrentTextureStatus::Error;
    if (!headless && presentationEnabled) {
      // Headless never acquires a swapchain texture at all -- GetCurrentTexture
      // on a hidden window's surface is what deadlocks in the Vulkan WSI's
      // explicit-sync release wait when the compositor never displays it.
      window::SurfaceLock surfaceLock;
      if (window::is_presentable() && g_surface) {
        ZoneScopedN("Acquire texture");
        wgpu::SurfaceTexture surfaceTexture;
        g_surface.GetCurrentTexture(&surfaceTexture);
        surfaceStatus = surfaceTexture.status;
        if (surfaceStatus == wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal) {
          currentTexture = std::move(surfaceTexture.texture);
          currentView = currentTexture.CreateView();
        }
      }
    }

    const bool canPresent = !headless && presentationEnabled && currentTexture && currentView;
    if (canPresent) {
      wgpu::BindGroup presentBindGroup;
      if (rmlBindGroup && !rmlOverlay) {
        presentBindGroup = rmlBindGroup;
      } else {
        const auto& resampledSource = webgpu::resample_present_source(encoder, viewport, presentSource);
        presentBindGroup = webgpu::create_copy_bind_group(resampledSource);
      }
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = currentView,
                .loadOp = wgpu::LoadOp::Clear,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "EFB copy render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
            .timestampWrites = webgpu::gpu_prof::pass_writes("Present blit"),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        // Copy EFB -> XFB (swapchain)
        pass.SetPipeline(webgpu::g_CopyPipeline);
        pass.SetBindGroup(0, presentBindGroup, 0, nullptr);
        set_present_viewport(pass, viewport, webgpu::g_graphicsConfig.surfaceConfiguration.width,
                             webgpu::g_graphicsConfig.surfaceConfiguration.height);

        pass.Draw(3);
        if (rmlBindGroup && rmlOverlay) {
          pass.SetPipeline(webgpu::g_CopyPremultipliedAlphaPipeline);
          pass.SetBindGroup(0, rmlBindGroup, 0, nullptr);
          pass.Draw(3);
        }
        pass.End();
      }
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = currentView,
                .loadOp = wgpu::LoadOp::Load,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "ImGui render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
            .timestampWrites = webgpu::gpu_prof::pass_writes("ImGui"),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        pass.SetViewport(0.f, 0.f, static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.width),
                         static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.height), 0.f, 1.f);
        imgui::render(pass, imguiDrawData);
        pass.End();
      }
    } else if (!headless && presentationEnabled) {
      Log.info("Skipping present; window not presentable");
    }
    webgpu::gpu_prof::frame_end(encoder);
    const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Redraw command buffer"};
    const auto buffer = encoder.Finish(&cmdBufDescriptor);
    auto probe = submitProbe;
    probe.presentEnabled = presentationEnabled ? 1u : 0u;
    probe.headless = headless ? 1u : 0u;
    {
      ZoneScopedN("Queue Submit");
      webgpu::submit_command_buffer(buffer, probe);
    }
    webgpu::gpu_prof::after_submit();
    if (headless) {
      // No WSI present in headless mode -- GPU progress is driven by the
      // submitted queue work above (fifo drain + gfx::finish), not by a
      // present, so pacing/wait_for_gpu_progress must not depend on this.
    } else if (canPresent && g_surface) {
      ZoneScopedN("Present");
      wgpu::ConvertibleStatus status = wgpu::Status::Error;
      {
        window::SurfaceLock surfaceLock;
        // Acquisition owns a surface texture that must be released through Present before the
        // surface can be unconfigured. A minimize event can race this callback after acquisition;
        // rechecking presentability here used to skip Present and unconfigure while the live
        // texture/view were still referenced by submitted work.
        status = g_surface.Present();
      }
      if (status) {
        gfx::after_present();
      } else {
        Log.warn("Surface present failed; deferring surface recreation to the next frame boundary");
        webgpu::invalidate_surface();
      }
    } else if (presentationEnabled && webgpu::surface_available()) {
      switch (surfaceStatus) {
      case wgpu::SurfaceGetCurrentTextureStatus::Timeout:
        Log.warn("Surface texture acquisition timed out");
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
      case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
        Log.info("Surface texture is {}, reconfiguring swapchain", magic_enum::enum_name(surfaceStatus));
        window::push_custom_event(window::CustomEvent::RefreshSurface);
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::Lost:
        Log.warn("Surface texture is {}; deferring recreation", magic_enum::enum_name(surfaceStatus));
        webgpu::invalidate_surface();
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::Error:
        Log.warn("Surface texture is {}; deferring recreation", magic_enum::enum_name(surfaceStatus));
        webgpu::invalidate_surface();
        break;
      default:
        if (!window::is_presentable()) {
          webgpu::invalidate_surface();
        } else {
          Log.error("Failed to get surface texture: {}", magic_enum::enum_name(surfaceStatus));
        }
        break;
      }
    }
    gfx::after_submit();

    TracyPlotConfig("aurora: lastVertSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastUniformSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastIndexSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastStorageSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastTextureUploadSize", tracy::PlotFormatType::Memory, false, true, 0);

    TracyPlot("aurora: queuedPipelines", static_cast<int64_t>(gfx::g_stats.queuedPipelines));
    TracyPlot("aurora: createdPipelines", static_cast<int64_t>(gfx::g_stats.createdPipelines));
    TracyPlot("aurora: drawCallCount", static_cast<int64_t>(gfx::g_stats.drawCallCount));
    TracyPlot("aurora: mergedDrawCallCount", static_cast<int64_t>(gfx::g_stats.mergedDrawCallCount));
    TracyPlot("aurora: lastVertSize", static_cast<int64_t>(gfx::g_stats.lastVertSize));
    TracyPlot("aurora: lastUniformSize", static_cast<int64_t>(gfx::g_stats.lastUniformSize));
    TracyPlot("aurora: lastIndexSize", static_cast<int64_t>(gfx::g_stats.lastIndexSize));
    TracyPlot("aurora: lastStorageSize", static_cast<int64_t>(gfx::g_stats.lastStorageSize));
    TracyPlot("aurora: lastTextureUploadSize", static_cast<int64_t>(gfx::g_stats.lastTextureUploadSize));
  });

#endif
  // SB_RDOC frame delimiter — unconditional (headless included): every
  // end_frame counts as one "present" for the capture window.
  sb_rdoc_on_present();
}

void end_frame() noexcept {
  end_frame_impl(false);
#ifdef AURORA_ENABLE_GX
  // Emit retained samples from the same recorded frame. The whole cycle lives
  // here rather than in the host because presenting is not something gfx::end_frame does on its
  // own — the present blit, the swapchain acquire and the frame sink all live in the callback
  // that this function builds, so each sample needs the full cycle. begin_frame() is reused as-is
  // (swapchain acquire, EFB pass setup, imgui::new_frame);
  // install_replay_snapshot then throws away the pass it created and puts the snapshot in.
  if (!gfx::replay_present_enabled() || !gfx::has_replay_snapshot()) {
    return;
  }
  const unsigned count = gfx::replay_presentation_count();
  for (unsigned index = 1; index < count; ++index) {
    const float alpha = static_cast<float>(index + 1) / static_cast<float>(count);
    if (aurora_replay_sample != nullptr) {
      aurora_replay_sample(alpha, index, count);
    } else if (index == 1 && count == 2 && aurora_replay_midpoint != nullptr) {
      aurora_replay_midpoint();
    }
    if (!begin_frame()) {
      return;
    }
    const bool finalEmission = index + 1 == count;
    if (!gfx::install_replay_snapshot(finalEmission)) {
      Log.fatal("Replay snapshot vanished before presentation sample {} of {}", index + 1, count);
    }
    if (!finalEmission && gfx::interp_alpha() >= 0.0f) {
      gfx::interpolate_recorded_frame(alpha, true);
    }
    end_frame_impl(true);
  }
#endif
}
} // namespace

bool presentation_enabled() noexcept { return presentation_enabled_impl(); }

void set_presentation_enabled(bool enabled) noexcept { set_presentation_enabled_impl(enabled); }
} // namespace aurora

// C API bindings
AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config) {
  return aurora::initialize(argc, argv, *config);
}
void aurora_shutdown() { aurora::shutdown(); }
double aurora_display_refresh_rate() { return aurora::window::display_refresh_rate(); }
const AuroraEvent* aurora_update() { return aurora::update(); }
bool aurora_begin_frame() { return aurora::begin_frame(); }
void aurora_end_frame() { aurora::end_frame(); }
void aurora_set_presentation_enabled(bool enabled) { aurora::set_presentation_enabled(enabled); }
void aurora_set_present_aspect(uint32_t width, uint32_t height) { aurora::set_present_aspect(width, height); }

void aurora_discard_frame() {
#ifdef AURORA_ENABLE_GX
  aurora::gx::fifo::clear_buffer();
#endif
}
void aurora_fifo_replay(const uint8_t* data, uint32_t size, int bigEndian) {
#ifdef AURORA_ENABLE_GX
  // Synchronous replay of a raw command stream through the command processor.
  // Does NOT touch the live FIFO buffer (the game's own commands); the parity
  // harness calls this between begin_frame/end_frame with no game running.
  aurora::gx::fifo::process(data, size, bigEndian != 0);
#endif
}
AuroraBackend aurora_get_backend() { return aurora::g_config.desiredBackend; }
const AuroraBackend* aurora_get_available_backends(size_t* count) {
  if (count != nullptr) {
    *count = aurora::PreferredBackendOrder.size();
  }
  return aurora::PreferredBackendOrder.data();
}
void aurora_set_frame_sink(AuroraFrameSink fn, void* user, int everyNFrames) {
  aurora::g_frameSink = fn;
  aurora::g_frameSinkUser = user;
  aurora::g_frameSinkEvery = (fn != nullptr) ? everyNFrames : 0;
}
void aurora_set_dump_tag(const char* tag) {
  if (tag == nullptr) {
    aurora::g_dumpTag[0] = '\0';
    return;
  }
  std::snprintf(aurora::g_dumpTag, sizeof(aurora::g_dumpTag), "%s", tag);
}
void aurora_set_log_level(AuroraLogLevel level) { aurora::g_config.logLevel = level; }
void aurora_set_pause_on_focus_lost(bool value) { aurora::g_config.pauseOnFocusLost = value; }
void aurora_set_background_input(bool value) {
  aurora::g_config.allowJoystickBackgroundEvents = value;
  aurora::window::set_background_input(value);
}
void aurora_set_resampler(AuroraSampler sampler) {
#ifdef AURORA_ENABLE_GX
  aurora::webgpu::set_resampler(sampler);
#else
  (void)sampler;
#endif
}
