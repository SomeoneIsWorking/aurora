#ifndef AURORA_AURORA_H
#define AURORA_AURORA_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>

extern "C" {
#else
#include "stdbool.h"
#include "stddef.h"
#include "stdint.h"
#endif

typedef enum {
  SAMPLER_BILINEAR,
  SAMPLER_AREA,
} AuroraSampler;

typedef enum {
  BACKEND_AUTO,
  BACKEND_D3D11,
  BACKEND_D3D12,
  BACKEND_METAL,
  BACKEND_VULKAN,
  BACKEND_OPENGL,
  BACKEND_OPENGLES,
  BACKEND_WEBGPU,
  BACKEND_NULL,
} AuroraBackend;

typedef enum {
  LOG_DEBUG,
  LOG_INFO,
  LOG_WARNING,
  LOG_ERROR,
  LOG_FATAL,
} AuroraLogLevel;

typedef struct {
  int32_t x;
  int32_t y;
} AuroraWindowPos;

typedef struct {
  uint32_t width;
  uint32_t height;

  /**
   * Width of the main GX framebuffer.
   */
  uint32_t fb_width;

  /**
   * Height of the main GX framebuffer.
   */
  uint32_t fb_height;

  /**
   * The size of the framebuffer used to present to the operating system.
   * May differ from fb_width if Aurora is instructed to force an aspect ratio or resolution configuration.
   */
  uint32_t native_fb_width;

  /**
   * The size of the framebuffer used to present to the operating system.
   * May differ from fb_height if Aurora is instructed to force an aspect ratio or resolution configuration.
   */
  uint32_t native_fb_height;
  float scale;
} AuroraWindowSize;

typedef struct SDL_Window SDL_Window;
typedef struct AuroraEvent AuroraEvent;

typedef void (*AuroraLogCallback)(AuroraLogLevel level, const char* module, const char* message, unsigned int len);
typedef void (*AuroraImGuiInitCallback)(const AuroraWindowSize* size);

#define MEM1_DEFAULT_SIZE (24 * 1024 * 1024)
#define ARAM_DEFAULT_SIZE (16 * 1024 * 1024)

typedef struct {
  const char* appName;
  const char* userPath;
  const char* cachePath;
  const char* resourcesPath;
  AuroraBackend desiredBackend;
  uint32_t msaa;
  uint16_t maxTextureAnisotropy;
  bool vsync;
  bool startFullscreen;
  bool allowJoystickBackgroundEvents;
  bool pauseOnFocusLost;
  bool allowTextureDumps;
  bool allowCpuAdapter;
  int32_t windowPosX;
  int32_t windowPosY;
  uint32_t windowWidth;
  uint32_t windowHeight;
  void* iconRGBA8;
  uint32_t iconWidth;
  uint32_t iconHeight;
  AuroraLogCallback logCallback;
  AuroraLogLevel logLevel;
  AuroraImGuiInitCallback imGuiInitCallback;

  /*
   * The size of the GameCube's main memory, or MEM1 on the Wii.
   * Note that it will not be allocated at the exact 0x80000000 address, as that cannot be guaranteed.
   * This can be set to 0 to disable allocating this region.
   */
  uint32_t mem1Size;

  /*
   * The size of the GameCube's ARAM, or MEM2 on the Wii.
   * This can be set to 0 to disable allocating this region.
   */
  uint32_t mem2Size;
} AuroraConfig;

typedef struct {
  AuroraBackend backend;
  const char* userPath;
  const char* cachePath;
  SDL_Window* window;
  AuroraWindowSize windowSize;
} AuroraInfo;

AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config);
void aurora_shutdown();
double aurora_display_refresh_rate();
const AuroraEvent* aurora_update();
bool aurora_begin_frame();
void aurora_end_frame();

/* Select whether Aurora owns operating-system presentation. Disabling presentation releases its
 * WSI surface while preserving FIFO consumption, offscreen rendering, and frame-sink readback so
 * Aurora can serve as an oracle beside a host-owned presenter. */
void aurora_set_presentation_enabled(bool enabled);
/* Drop GX commands queued since the last frame without rendering them. For
 * frames produced while the surface was unpresentable (minimized window):
 * begin_frame returned false, end_frame must not run, but the fifo still has
 * to be emptied or it grows without bound. */
void aurora_discard_frame();

/* Aspect ratio of the on-screen picture, as a width:height pair.
 *
 * The default is the TV's 4:3, which is right for a console-faithful present:
 * VI scan-out stretches whatever XFB the display copy produced across the
 * physical picture, so the XFB's own texel size never decides the aspect.
 *
 * A runtime that renders the scene ANAMORPHICALLY — squeezing the projection
 * horizontally so a wider field of view lands in the same EFB — must say so
 * here, otherwise the wide picture is presented into a 4:3 rectangle and comes
 * out horizontally compressed. Pass 16, 9 for widescreen. */
void aurora_set_present_aspect(uint32_t width, uint32_t height);

/* Replay a raw GX FIFO command byte stream through the renderer's command
 * processor (the same path aurora_end_frame's internal drain() takes, but
 * synchronous and without draining the live FIFO buffer). Diagnostic ONLY --
 * drives the SB_FIFO_REPLAY parity harness in sms-boot, which feeds a captured
 * Dolphin .dff's per-frame command bytes here between aurora_begin_frame and
 * aurora_end_frame so the render can be diffed against Dolphin's. See
 * docs/model_interpolation.md (2026-07-11 amendment). bigEndian=true treats the
 * bytes as GC-native big-endian (the .dff wire format). */
void aurora_fifo_replay(const uint8_t* data, uint32_t size, int bigEndian);

/* Frame sink: receive the presented frame as tightly-packed RGBA8 (top-left
 * origin), the same pixels SB_DUMP_FRAME writes, delivered to a callback
 * instead of a file. Intended for IN-PROCESS parity comparison, where a second
 * renderer's output can be scored against aurora's on the SAME frame with no
 * file round-trip and no hand-aligned frame indices.
 *
 * The capture rides the existing asynchronous readback: the copy is encoded at
 * one present and mapped at the next, so the callback arrives a frame or two
 * later and the GPU is never stalled. `everyNFrames` <= 0 disables. Passing a
 * null fn also disables. The bytes are only valid for the duration of the call.
 */
typedef void (*AuroraFrameSink)(const uint8_t* rgba, uint32_t width, uint32_t height, void* user);
void aurora_set_frame_sink(AuroraFrameSink fn, void* user, int everyNFrames);

/* Stamp a short role label onto subsequent SB_DUMP_FRAME filenames (e.g. "main", "sub").
 * A runtime that issues more than one kind of present per tick otherwise produces a dump
 * series whose files must be identified by inference; the label makes each artifact say
 * what it is. Pass NULL to clear. */
void aurora_set_dump_tag(const char* tag);

void aurora_set_log_level(AuroraLogLevel level);
void aurora_set_pause_on_focus_lost(bool value);
void aurora_set_background_input(bool value);
void aurora_set_resampler(AuroraSampler sampler);

AuroraBackend aurora_get_backend();
const AuroraBackend* aurora_get_available_backends(size_t* count);

#ifdef __cplusplus
}
#endif

#endif
