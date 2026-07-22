#include <dolphin/vi.h>

#include "../../window.hpp"
#include "aurora/math.hpp"
#include "vi_internal.hpp"

#include <optional>

namespace aurora::vi {
std::optional<GXRenderModeObj> g_renderMode;

// Set by aurora_vi_set_fb_size for runtimes that own the VI hardware themselves and so never
// call VIConfigure. Without it such a runtime silently gets the 640x480 default below, and
// every viewport and EFB-copy rectangle is scaled by the wrong target/logical ratio.
std::optional<Vec2<uint32_t>> g_fbSizeOverride;

Vec2<uint32_t> render_mode_size() noexcept {
  if (g_fbSizeOverride) {
    return *g_fbSizeOverride;
  }
  if (!g_renderMode) {
    return {640, 480};
  }
  return {g_renderMode->fbWidth, g_renderMode->efbHeight};
}

void configure(const GXRenderModeObj* rm) noexcept {
  const auto oldSize = render_mode_size();
  if (rm == nullptr) {
    g_renderMode.reset();
  } else {
    g_renderMode = *rm;
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
// For runtimes that drive the VI registers themselves (a recompiled game programming VI MMIO
// directly) and therefore never reach VIConfigure. Aurora cannot infer the framebuffer size in
// that case, and its default would silently mis-scale every viewport and copy rectangle.
void aurora_vi_set_fb_size(uint32_t width, uint32_t height) {
  const auto oldSize = aurora::vi::render_mode_size();
  aurora::vi::g_fbSizeOverride = aurora::Vec2<uint32_t>{width, height};
  if (aurora::vi::render_mode_size() != oldSize) {
    aurora::window::request_frame_buffer_resize();
  }
}
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
