#include <dolphin/vi.h>

#include "../../internal.hpp"
#include "../../window.hpp"
#include "aurora/math.hpp"
#include "vi_internal.hpp"

#include <optional>

// Declared in webgpu/gpu.hpp; not included here because the aurora_vi target
// doesn't carry the Dawn include paths that header requires.
namespace aurora::webgpu {
extern uint32_t g_sbViWidth;
extern uint32_t g_sbViHeight;
} // namespace aurora::webgpu

namespace aurora::vi {
static Module Log("aurora::vi");
std::optional<GXRenderModeObj> g_renderMode;

Vec2<uint32_t> render_mode_size() noexcept {
  if (!g_renderMode) {
    return {640, 480};
  }
  return {g_renderMode->fbWidth, g_renderMode->efbHeight};
}

void configure(const GXRenderModeObj* rm) noexcept {
  const auto oldSize = render_mode_size();
  if (rm == nullptr) {
    g_renderMode.reset();
    webgpu::g_sbViWidth = 640;
    webgpu::g_sbViHeight = 480;
  } else {
    g_renderMode = *rm;
    // Latch the VI-space picture size the present viewport must preserve
    // (see gpu.hpp g_sbViWidth). A zero field would make the viewport math
    // degenerate silently — crash at the bad render mode instead.
    if (rm->viWidth == 0 || rm->viHeight == 0) {
      Log.fatal("VIConfigure: render mode has degenerate viWidth={} viHeight={}", rm->viWidth, rm->viHeight);
    }
    webgpu::g_sbViWidth = rm->viWidth;
    webgpu::g_sbViHeight = rm->viHeight;
  }
  if (render_mode_size() != oldSize) {
    window::request_frame_buffer_resize();
  }
}

Vec2<uint32_t> configured_fb_size() noexcept {
  return render_mode_size();
}
} // namespace aurora::vi

extern "C" {
void VIInit() {}
void VIConfigure(const GXRenderModeObj* rm) { aurora::vi::configure(rm); }
void VIConfigurePan(u16 xOrg, u16 yOrg, u16 width, u16 height) {
  (void)xOrg;
  (void)yOrg;
  (void)width;
  (void)height;
}
u32 VIGetTvFormat() { return 0; }
void VIFlush() {}

void VISetWindowTitle(const char* title) { aurora::window::set_title(title); }
void VISetWindowFullscreen(bool fullscreen) { aurora::window::set_fullscreen(fullscreen); }
bool VIGetWindowFullscreen() { return aurora::window::get_fullscreen(); }
void VISetWindowSize(uint32_t width, uint32_t height) { aurora::window::set_window_size(width, height); }
void VISetWindowPosition(uint32_t x, uint32_t y) { aurora::window::set_window_position(x, y); }
void VICenterWindow() { aurora::window::center_window(); }
void VISetFrameBufferScale(float scale) { aurora::window::set_frame_buffer_scale(scale); }
}
