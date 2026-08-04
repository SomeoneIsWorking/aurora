#include "gx.hpp"
#include "__gx.h"

#include "../../gfx/tex_copy_conv.hpp"
#include "../../gfx/texture.hpp"
#include "../../window.hpp"
#include "../../gfx/clear.hpp"
#include "../../webgpu/gpu.hpp"
#include "../vi/vi_internal.hpp"

#include <algorithm>
#include <cmath>

// Frame ordinal for SB_COPY_DBG_AFTER. Provided by the runtime (sms-boot's frame seam, or the
// recomp's present counter); weak so aurora still links standalone, where it reads 0 and the
// window simply opens immediately.
extern "C" unsigned VIGetRetraceCount(void) __attribute__((weak));
// Global draw ordinal, incremented per pushed draw in command_processor.cpp. Logging it with
// each copy puts copies and draws on ONE timeline, which is the only way to see whether a
// render-to-texture grab happens before or after the draw that samples it - the difference
// between a feedback loop with no gain and one that compounds.
namespace aurora::gx::fifo { extern long g_sbPushedDrawCount; }
static unsigned sb_gx_vi_retrace_count() { return (&VIGetRetraceCount) ? VIGetRetraceCount() : 0; }

namespace {
// GXCopyDisp is the GC-HW operation that actually defines the displayed TV
// image (real HW: the same "PE copy" op as GXCopyTex, selected by a
// copy-to-XFB bit on the BP 0x52 command). GXCopyDisp below routes through
// the SAME working copy_tex() cache/resolve mechanism GXCopyTex uses, keyed
// on the reserved sentinel pointer aurora::gx::kDisplayCopyDest so
// copy_tex() can recognize "this resolve is the display copy" and latch the
// result as the present source (see gpu.cpp g_sbDisplayPresent).
// command_processor.cpp's BP-0x52 dispatch routes in-stream copy_to_xfb
// triggers (e.g. FIFO replay) to the same key, so both entry paths converge.
struct DisplayCopyDestTag {};
const DisplayCopyDestTag kDisplayCopyDestTagInstance{};

// GXSetDispCopyYScale/GXGetNumXfbLines RE'd from reference/sms's decomp'd
// dolphin/gx/GXFrameBuf.c (real Nintendo SDK source in the decomp tree, not
// game logic): GXSetDispCopyDst's `ht` argument is a red herring — real HW
// ignores it (only `wd` feeds the copy stride register). The actual XFB
// copy height is GXSetDispCopyYScale's RETURN VALUE, computed from the EFB
// height last configured via GXSetDispCopySrc plus the Y-scale ratio. Our
// GXSetDispCopyYScale was a stub returning 0, which is why every per-frame
// GXCopyDisp resolved a degenerate ~0-line-tall texture once the boot-time
// call was superseded by JDREfbSetting::IssueGXCopyDisp's per-frame call.
u32 s_dispCopyEfbHeight = 480;

u32 gx_get_num_xfb_lines(u32 height, u32 scale) noexcept {
  const u32 numLines = (height - 1) * 0x100;
  u32 actualHeight = (numLines / scale) + 1;
  u32 newScale = scale;
  if (newScale > 0x80 && newScale < 0x100) {
    while (newScale % 2 == 0) {
      newScale /= 2;
    }
    if (height % newScale == 0) {
      ++actualHeight;
    }
  }
  if (actualHeight > 0x400) {
    actualHeight = 0x400;
  }
  return actualHeight;
}

aurora::Vec2<uint32_t> scale_copy_dst(u32 logicalWidth, u32 logicalHeight) {
  if (g_gxState.viewportPolicy == AURORA_VIEWPORT_NATIVE) {
    return {logicalWidth, logicalHeight};
  }

  const auto [logicalFbWidth, logicalFbHeight] = aurora::gx::logical_fb_size();
  const auto [targetWidth, targetHeight] = aurora::gfx::get_render_target_size();
  if (logicalFbWidth == 0 || logicalFbHeight == 0 || targetWidth == 0 || targetHeight == 0) {
    return {logicalWidth, logicalHeight};
  }

  const float scaleX = static_cast<float>(targetWidth) / static_cast<float>(logicalFbWidth);
  const float scaleY = static_cast<float>(targetHeight) / static_cast<float>(logicalFbHeight);
  const auto scaledWidth = std::max<u32>(static_cast<u32>(std::lround(static_cast<float>(logicalWidth) * scaleX)), 1);
  const auto scaledHeight = std::max<u32>(static_cast<u32>(std::lround(static_cast<float>(logicalHeight) * scaleY)), 1);
  return {scaledWidth, scaledHeight};
}
} // namespace

namespace aurora::gx {
const void* const kDisplayCopyDest = &kDisplayCopyDestTagInstance; // declared in gx.hpp
extern "C" const char* sb_gx_last_marker();
extern "C" int sb_timeline_enabled();
extern "C" void sb_timeline_log(const char* fmt, ...);

void copy_tex(const void* dest, GXBool clear) noexcept {
  const auto rect = map_logical_scissor(g_gxState.texCopySrc);
  const auto [dstWidth, dstHeight] = scale_copy_dst(g_gxState.texCopyDstWidth, g_gxState.texCopyDstHeight);
  const auto texCopyFmt = g_gxState.texCopyFmt;

  const GXState::CopyTextureKey key{
      .dest = dest,
      .width = dstWidth,
      .height = dstHeight,
      .format = texCopyFmt,
  };
  auto it = g_gxState.copyTextureCache.find(key);
  if (it == g_gxState.copyTextureCache.end()) {
    gfx::TextureHandle handle;
    if (gfx::tex_copy_conv::needs_conversion(texCopyFmt)) {
      handle = gfx::new_conv_texture(dstWidth, dstHeight, texCopyFmt, "Copy Conv Texture");
    } else {
      // Configure the texture swizzle to use alpha 1.0 if targeting RGB565 or
      // EFB doesn't have alpha. The display copy is also alpha-less: the real
      // XFB is YUYV scan-out with NO alpha channel, so EFB alpha (e.g. SMS's
      // mid-frame dst-alpha stamps) must never leak into the presented image
      // (it washes out any alpha-aware consumer of the present source).
      const auto fmt = texCopyFmt == GX_TF_RGB565 || g_gxState.pixelFmt == GX_PF_RGB8_Z24 ||
                               g_gxState.pixelFmt == GX_PF_RGB565_Z16 || dest == kDisplayCopyDest
                           ? GX_TF_RGB565
                           : GX_TF_RGBA8;
      handle = gfx::new_render_texture(dstWidth, dstHeight, fmt, "Resolved Texture");
    }
    it = g_gxState.copyTextureCache.emplace(key, GXState::CopyTextureRef{.handle = handle, .revision = 0}).first;
  }
  auto& handle = it->second;

  if (g_gxState.alphaUpdate && g_gxState.dstAlpha != UINT32_MAX) {
    if (!clear) {
      // TODO: figure out the right behavior here.
      // should the copy have a specific alpha value but the EFB remains untouched?
    }
    // Overwrite alpha before resolving
    gfx::push_draw_command(gfx::clear::DrawData{
        .pipeline = gfx::pipeline_ref(gfx::clear::PipelineConfig{
            .clearColor = false,
            .clearAlpha = true,
            .clearDepth = false,
        }),
        .color = wgpu::Color{0.f, 0.f, 0.f, g_gxState.dstAlpha / 255.f},
    });
  }
  // GC-FAITHFUL IMMEDIATE COPY-CLEAR (default). A GXCopyTex/GXCopyDisp with
  // clear=true resolves the EFB to the destination texture AND clears the EFB,
  // so a subsequent render pass starts fresh. The SMS title (and any scene
  // using TEfbCtrlTex reflection/capture passes) renders: Pass 1 (capture
  // source) -> GXCopyTex(clear=true) -> Pass 2 (main scene, samples the
  // capture) -> display copy. Honoring the clear keeps Pass 1 out of the
  // displayed EFB. The OLD deferred-clear model (SB_DEFER_COPY_CLEAR=1)
  // deferred this to the next frame start, which — once textures actually
  // rendered — let Pass 1's fullscreen capture-source render accumulate under
  // Pass 2 and saturate the frame white. Deferred is kept only as an A/B
  // escape hatch; do not rely on it (it converges to wrong).
  static int s_deferClear = -1;
  if (s_deferClear < 0) {
    const char* e = std::getenv("SB_DEFER_COPY_CLEAR");
    s_deferClear = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
  }
  const bool effClear = clear && s_deferClear == 0;
  const auto clearColor = effClear && g_gxState.colorUpdate;
  const auto clearAlpha = effClear && g_gxState.alphaUpdate;
  const auto clearDepth = effClear && g_gxState.depthUpdate;
  if (sb_timeline_enabled()) {
    sb_timeline_log("COPY %ux%u fmt=%u clear=%d (deferred=%d) after '%s'", dstWidth, dstHeight,
                    static_cast<unsigned>(texCopyFmt), static_cast<int>(clear), effClear ? 0 : (clear ? 1 : 0),
                    sb_gx_last_marker());
  }
  // SB_CLEAR_RGB=r,g,b (0-255, diagnostic): force the EFB clear colour. If a region of the
  // frame is the wrong colour because nothing ever DRAWS there, it is showing the clear, and
  // forcing an unmistakable colour makes that visible immediately — the one hypothesis a
  // draw-skipping test can never reach, since there is no draw to skip.
  static bool s_useForced = false;
  static Vec4<float> s_forced{};
  {
    static int s_init = 0;
    static bool s_on = false;
    static Vec4<float> s_col{};
    if (!s_init) {
      s_init = 1;
      if (const char* e = std::getenv("SB_CLEAR_RGB"); e != nullptr && e[0] != '\0') {
        int r = 255, g = 0, b = 255;
        std::sscanf(e, "%d,%d,%d", &r, &g, &b);
        s_col = Vec4<float>{r / 255.f, g / 255.f, b / 255.f, 1.f};
        s_on = true;
        std::fprintf(stderr, "[clear-rgb] forcing EFB clear to %d,%d,%d\n", r, g, b);
      }
    }
    s_useForced = s_on;
    if (s_on) s_forced = s_col;
  }
  // Name this copy's destination for the pass about to take the resolve. Interpolated 60fps decides
  // from the frame's own sample ORDER whether it is cross-frame feedback; the display copy is never
  // a candidate.
  gfx::note_copy_resolve_dest(dest != kDisplayCopyDest ? dest : nullptr);
  gfx::resolve_pass(handle.handle, rect, clearColor, clearAlpha, clearDepth,
                    s_useForced ? s_forced : g_gxState.clearColor, clear_depth_value(), texCopyFmt);
  // SB_PRESENT_PASS1: stash the FIRST clear=true copy's resolved texture as a
  // candidate present source (the EFB captured just before it was cleared =
  // the pre-clear pass's full content). Lets end_frame present that pass
  // instead of the accumulated EFB, to test where the main scene lives.
  // SB_PRESENT_COPY=<W>x<H>: present the resolved texture of the copy with those DESTINATION
  // dimensions instead of the finished frame. SB_PRESENT_PASS1 only reaches the first
  // clear=true copy, so a render-to-texture result taken with clear=0 — a water reflection,
  // say — could not be inspected at all, and its content had to be inferred. With this, a
  // frame dump shows the copy itself.
  bool presentThisCopy = false;
  if (const char* want = std::getenv("SB_PRESENT_COPY"); want != nullptr && want[0] != '\0') {
    char dims[32];
    std::snprintf(dims, sizeof(dims), "%ux%u", dstWidth, dstHeight);
    presentThisCopy = std::strcmp(dims, want) == 0;
  }
  if ((presentThisCopy || (clear && std::getenv("SB_PRESENT_PASS1") != nullptr))) {
    if (handle.handle && handle.handle->sampleTextureView) {
      webgpu::g_sbPass1Present = webgpu::TextureWithSampler{
          .texture = handle.handle->texture,
          .view = handle.handle->sampleTextureView,
          .size = handle.handle->size,
          .format = handle.handle->format,
          .sampler = webgpu::g_frameBuffer.sampler,
      };
    }
  }
  // GXCopyDisp (below) routes through this exact function on the reserved
  // kDisplayCopyDest key; latch its resolve unconditionally as the present
  // source — this IS the game's display copy, not a diagnostic guess.
  if (dest == kDisplayCopyDest && handle.handle && handle.handle->sampleTextureView) {
    webgpu::g_sbDisplayPresent = webgpu::TextureWithSampler{
        .texture = handle.handle->texture,
        .view = handle.handle->sampleTextureView,
        .size = handle.handle->size,
        .format = handle.handle->format,
        .sampler = webgpu::g_frameBuffer.sampler,
    };
  }
  ++handle.revision;
  g_gxState.copyTextures[dest] = handle;
  if (std::getenv("SB_COPY_DBG") != nullptr) {
    static long n = 0;
    // SB_COPY_DBG_AFTER=<retrace>: don't start logging until the run reaches that frame.
    // The 40-line budget is otherwise spent entirely on boot's display copies, long before
    // any scene of interest — the copies worth inspecting (render-to-texture content such as
    // the file-select sea reflection) happen thousands of frames later and were unobservable.
    static long s_after = -2;
    if (s_after == -2) {
      const char* a = std::getenv("SB_COPY_DBG_AFTER");
      s_after = (a != nullptr && a[0] != '\0') ? std::atol(a) : -1;
    }
    const bool windowOpen = s_after < 0 || static_cast<long>(sb_gx_vi_retrace_count()) >= s_after;
    // Copies before the window must not consume the budget — counting them is what made the
    // first version of this gate print nothing at all: boot exhausted n past 40 long before
    // the window opened.
    if (windowOpen && n < 40) {
      const auto [lw, lh] = aurora::gx::logical_fb_size();
      const auto [tw, th] = gfx::get_render_target_size();
      std::fprintf(stderr,
                   "[copy-tex] n=%ld dest=%p %ux%u fmt=%u clear=%d rect=(%d,%d %ux%u) src=(%d,%d %ux%u) "
                   "logical=%ux%u target=%ux%u policy=%d offscreen=%d dstAlpha=%d aU=%d cU=%d pixFmt=%d "
                   "afterDraw=%ld "
                   "mark='%s'\n",
                   ++n, dest, dstWidth, dstHeight, static_cast<unsigned>(texCopyFmt), static_cast<int>(clear), rect.x,
                   rect.y, rect.width, rect.height, g_gxState.texCopySrc.x, g_gxState.texCopySrc.y,
                   g_gxState.texCopySrc.width, g_gxState.texCopySrc.height, lw, lh, tw, th,
                   static_cast<int>(g_gxState.viewportPolicy), static_cast<int>(gfx::is_offscreen()),
                   // The copy's alpha is what a sampling draw's TEXA reads, and with a
                   // src-alpha blend that value sets the gain of any feedback loop through
                   // this copy. It is invisible in every other diagnostic.
                   static_cast<int>(g_gxState.dstAlpha), g_gxState.alphaUpdate ? 1 : 0,
                   g_gxState.colorUpdate ? 1 : 0, static_cast<int>(g_gxState.pixelFmt),
                   aurora::gx::fifo::g_sbPushedDrawCount,
                   sb_gx_last_marker());
    } else if (windowOpen) {
      ++n;
    }
  }
}
} // namespace aurora::gx

extern "C" {
GXRenderModeObj GXNtsc480IntDf = {
    VI_TVMODE_NTSC_INT,
    640,
    480,
    480,
    40,
    0,
    640,
    480,
    VI_XFBMODE_DF,
    0,
    0,
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
    {8, 8, 10, 12, 10, 8, 8},
};
GXRenderModeObj GXNtsc480Int = {
    VI_TVMODE_NTSC_INT,
    640,
    480,
    480,
    40,
    0,
    640,
    480,
    VI_XFBMODE_DF,
    0,
    0,
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
    {0, 0, 21, 22, 21, 0, 0},
};
GXRenderModeObj GXPal528IntDf = {
    VI_TVMODE_PAL_INT,
    704,
    528,
    480,
    40,
    0,
    640,
    480,
    VI_XFBMODE_DF,
    0,
    0,
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
    {8, 8, 10, 12, 10, 8, 8},
};
GXRenderModeObj GXMpal480IntDf = {
    VI_TVMODE_PAL_INT,
    640,
    480,
    480,
    40,
    0,
    640,
    480,
    VI_XFBMODE_DF,
    0,
    0,
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
    {8, 8, 10, 12, 10, 8, 8},
};

void GXAdjustForOverscan(GXRenderModeObj* rmin, GXRenderModeObj* rmout, u16 hor, u16 ver) {
  *rmout = *rmin;
  const auto size = aurora::window::get_window_size();
  rmout->fbWidth = size.fb_width;
  rmout->efbHeight = size.fb_height;
  rmout->xfbHeight = size.fb_height;
}

void GXSetDispCopySrc(u16 left, u16 top, u16 wd, u16 ht) {
  if (std::getenv("SB_COPY_DBG") != nullptr) {
    std::fprintf(stderr, "[disp-copy-src] left=%u top=%u wd=%u ht=%u\n", left, top, wd, ht);
  }
  // Real HW: GXSetDispCopySrc and GXSetTexCopySrc program the SAME BP
  // compare-window registers (0x49-0x4C) — there is only one copy-source
  // rect at the hardware level, shared between the disp and tex copy paths
  // and consumed immediately by whichever copy command follows. Mirror
  // GXSetTexCopySrc's queuing exactly.
  s_dispCopyEfbHeight = ht != 0 ? ht : s_dispCopyEfbHeight;
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_SRC);
  GX_WRITE_U32(left);
  GX_WRITE_U32(top);
  GX_WRITE_U32(wd);
  GX_WRITE_U32(ht);
}

void GXSetTexCopySrc(u16 left, u16 top, u16 wd, u16 ht) {
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_SRC);
  GX_WRITE_U32(left);
  GX_WRITE_U32(top);
  GX_WRITE_U32(wd);
  GX_WRITE_U32(ht);
}

void GXSetDispCopyDst(u16 wd, u16 ht) {
  if (std::getenv("SB_COPY_DBG") != nullptr) {
    std::fprintf(stderr, "[disp-copy-dst] wd=%u ht=%u\n", wd, ht);
  }
  // Same shared BP dst-size registers as GXSetTexCopyDst (see GXSetDispCopySrc
  // above). The XFB's hardware pixel format isn't a GXTexFmt at all (YUV/RGB
  // scan-out), but aurora presents through a plain wgpu texture, so route it
  // through the existing RGBA8 render-texture path with no mipmaps.
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_DST);
  GX_WRITE_U32(wd);
  GX_WRITE_U32(ht);
  GX_WRITE_U32(static_cast<u32>(GX_TF_RGBA8));
  GX_WRITE_U8(GX_FALSE);
}

void GXSetTexCopyDst(u16 wd, u16 ht, GXTexFmt fmt, GXBool mipmap) {
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_DST);
  GX_WRITE_U32(wd);
  GX_WRITE_U32(ht);
  GX_WRITE_U32(fmt);
  GX_WRITE_U8(mipmap != GX_FALSE);
}

// TODO GXSetDispCopyFrame2Field
// TODO GXSetCopyClamp

u32 GXSetDispCopyYScale(f32 vscale) {
  // See gx_get_num_xfb_lines comment above: this return value (NOT
  // GXSetDispCopyDst's ht arg) is what actually determines the XFB copy
  // height on real HW.
  const u32 scale = static_cast<u32>(256.0f / vscale) & 0x1FF;
  return gx_get_num_xfb_lines(s_dispCopyEfbHeight, scale);
}

void GXSetCopyClear(GXColor color, u32 depth) {
  // SB_COPY_DBG=1: log every copy-clear color change — the deferred-clear
  // model clears next-frame-start with the LAST color set, which is wrong if
  // the game sets different colors for different copies within a frame.
  if (std::getenv("SB_COPY_DBG") != nullptr) {
    static long n = 0;
    std::fprintf(stderr, "[copy-clear-set] n=%ld rgba=(%u,%u,%u,%u) z=%06x\n", ++n, color.r, color.g, color.b,
                 color.a, depth);
  }
  // BP 0x4F: clear color R + A
  u32 reg0 = 0;
  SET_REG_FIELD(0, reg0, 8, 0, color.r);
  SET_REG_FIELD(0, reg0, 8, 8, color.a);
  SET_REG_FIELD(0, reg0, 8, 24, 0x4F);
  GX_WRITE_RAS_REG(reg0);

  // BP 0x50: clear color B + G
  u32 reg1 = 0;
  SET_REG_FIELD(0, reg1, 8, 0, color.b);
  SET_REG_FIELD(0, reg1, 8, 8, color.g);
  SET_REG_FIELD(0, reg1, 8, 24, 0x50);
  GX_WRITE_RAS_REG(reg1);

  // BP 0x51: clear Z (24-bit)
  u32 reg2 = 0;
  SET_REG_FIELD(0, reg2, 24, 0, depth);
  SET_REG_FIELD(0, reg2, 8, 24, 0x51);
  GX_WRITE_RAS_REG(reg2);
  __gx->bpSent = 1;
}

void GXSetCopyFilter(GXBool aa, const u8 sample_pattern[12][2], GXBool vf, const u8 vfilter[7]) {}

void GXSetDispCopyGamma(GXGamma gamma) {}

void GXCopyDisp(void* dest, GXBool clear) {
  // GXCopyDisp is the LAST GX call in the game's per-frame draw list (after
  // the main scene + Chr + lens flare + all 2D UI — TEfbCtrlDisp /
  // mPerformListGXPost in reference/sms) and is what actually defines the
  // displayed TV image on real hardware. Queue the same BP-0x52 "PE copy"
  // command GXCopyTex uses (see kDisplayCopyDest comment above for why this
  // uses a dest-key discriminator instead of the hardware copy-to-XFB bit),
  // so it resolves at the correct point in the frame relative to every other
  // queued draw/copy, then gets latched as the present source in copy_tex().
  (void)dest; // real XFB address; aurora keys the resolve on kDisplayCopyDest instead (see above)
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_DEST);
  GX_WRITE_U64(reinterpret_cast<u64>(aurora::gx::kDisplayCopyDest));

  SET_REG_FIELD(0, __gx->cpTex, 1, 11, clear != GX_FALSE);
  SET_REG_FIELD(0, __gx->cpTex, 1, 14, 0);
  SET_REG_FIELD(0, __gx->cpTex, 8, 24, 0x52);
  GX_WRITE_RAS_REG(__gx->cpTex);
}

void GXCopyTex(void* dest, GXBool clear) {
  // SB_COPY_DBG: log every EFB->texture copy (the mirror capture + the mid-scene
  // "snapshot" retail composites the title backdrop from). Distinct from the disp copy.
  if (std::getenv("SB_COPY_DBG") != nullptr) {
    static long n = 0;
    std::fprintf(stderr, "[tex-copy] n=%ld dest=%p clear=%d\n", ++n, dest, clear != GX_FALSE);
  }
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_DEST);
  GX_WRITE_U64(reinterpret_cast<u64>(dest));

  SET_REG_FIELD(0, __gx->cpTex, 1, 11, clear != GX_FALSE);
  SET_REG_FIELD(0, __gx->cpTex, 1, 14, 0);
  SET_REG_FIELD(0, __gx->cpTex, 8, 24, 0x52);
  GX_WRITE_RAS_REG(__gx->cpTex);
}

// GXGetNumXfbLines/GXGetYScaleFactor: faithful ports of the SDK bodies in
// reference/sms/src/dolphin/gx/GXFrameBuf.c. The game's render-mode setup
// (JDrama::CalcRenderModeXFBHeight -> rmo->xfbHeight/viHeight) flows through
// these; they were silent 0-returning stubs in sms-boot/runtime/sdk_stubs.cpp
// until 2026-07-14, which zeroed every computed xfbHeight/viHeight.
u16 GXGetNumXfbLines(u16 efbHeight, f32 yScale) {
  const u32 scale = static_cast<u32>(256.0f / yScale) & 0x1FF;
  return static_cast<u16>(gx_get_num_xfb_lines(efbHeight, scale));
}

f32 GXGetYScaleFactor(u16 efbHeight, u16 xfbHeight) {
  u32 height1 = xfbHeight;
  f32 scale1 = static_cast<f32>(xfbHeight) / static_cast<f32>(efbHeight);
  u32 scale = static_cast<u32>(256.0f / scale1) & 0x1FF;
  u32 height2 = gx_get_num_xfb_lines(efbHeight, scale);

  while (height2 > xfbHeight) {
    height1--;
    scale1 = static_cast<f32>(height1) / static_cast<f32>(efbHeight);
    scale = static_cast<u32>(256.0f / scale1) & 0x1FF;
    height2 = gx_get_num_xfb_lines(efbHeight, scale);
  }

  f32 scale2 = scale1;
  while (height2 < xfbHeight) {
    scale2 = scale1;
    height1++;
    scale1 = static_cast<f32>(height1) / static_cast<f32>(efbHeight);
    scale = static_cast<u32>(256.0f / scale1) & 0x1FF;
    height2 = gx_get_num_xfb_lines(efbHeight, scale);
  }

  return scale2;
}
// TODO GXClearBoundingBox
// TODO GXReadBoundingBox
}
