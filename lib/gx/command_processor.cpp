#include <chrono>
#include "command_processor.hpp"

#include <lucent/log.h>
#include <unordered_set>
#include <unordered_map>
#include <cstdarg>

#include "fifo.hpp"
#include "prim_index.hpp"

#include "../gfx/common.hpp"
#include "../gfx/depth_peek.hpp"
#include "dolphin/gx/GXAurora.h"
#include "gx.hpp"
#include "../gfx/interp.hpp"
#include "gx_fmt.hpp"
#include "pipeline.hpp"
#include "shader_info.hpp"
#include "../internal.hpp"

#include <absl/container/flat_hash_map.h>
#include <tracy/Tracy.hpp>

#include <cmath>
#include <array>
#include <vector>
#include <cstdint>
#include <cstring>
#include <optional>

namespace {
// SB_COPY_DBG was an uncached std::getenv per EFB-copy / clear-register write. Now a lucent
// channel, looked up once. Enable with LUCENT_DEBUG=copydbg.
inline bool sb_copy_dbg_chan() {
  static const lucent::Channel ch{"copydbg"};
  return static_cast<bool>(ch);
}
} // namespace


namespace aurora::gx::fifo {
static Module Log("aurora::gx::fifo");

// Last debug marker seen in the stream — names the draw-buffer/pass each
// subsequent draw belongs to (printed by SB_DRAW_DUMP).
static thread_local std::string g_sbLastMarker;
static thread_local int g_sbMarkerDrawIdx = 0; // Nth draw since the current marker began (SB_SKIP_SKY_IDX)
// Exposed for cross-TU diagnostics (e.g. copy_tex logging which J3D buffer/2D
// element a GXCopyTex follows).
extern "C" const char* sb_gx_last_marker() { return g_sbLastMarker.c_str(); }

// SB_PROFILE_GFX per-draw build-phase accumulators (0=shaderinfo+config,
// 1=bind_groups, 2=pipeline_ref, 3=build_uniform, 4=push_draw_command).
// Printed and reset by the frame profiler in aurora.cpp.
extern "C" { double g_sbGxProf[7] = {0, 0, 0, 0, 0, 0, 0};
void sb_gx_prof_add(int slot, double us) { if (slot >= 0 && slot < 7) g_sbGxProf[slot] += us; } }

// VIGetRetraceCount is defined game-side (sms-boot/runtime/sdk_stubs.cpp) and
// advanced once per sb_frame_present (sms-boot/runtime/frame_seam.cpp) — same
// counter SB_DUMP_FRAME_AFTER's frame index tracks. Declared weak so aurora's
// standalone unit tests (which don't link the game) still build.
extern "C" unsigned VIGetRetraceCount(void) __attribute__((weak));
static unsigned sb_gx_vi_retrace_count() { return (&VIGetRetraceCount) ? VIGetRetraceCount() : 0; }
// sms-boot's SB_LOG channel registry (sb_log.h) — weak so aurora still links
// standalone; in-tree the sms-boot runtime always provides it.
extern "C" int sb_log_enabled(const char* chan) __attribute__((weak));
extern "C" void sb_logf(const char* chan, const char* fmt, ...)
    __attribute__((weak, format(printf, 2, 3)));
// A weak symbol that resolves to null answers "channel off" — which is INDISTINGUISHABLE from
// "channel on but the condition never occurred". That silence cost a whole investigation: an
// SB_LOG=pnzero run reported zero zero-rotation matrix uploads while the provider simply was not
// linked, and the zero was read as a measurement. (It is also easy to hit by accident: a weak
// UNDEFINED reference does not pull a member out of a static archive, so shipping the provider in
// a library is not enough — it must be linked into the executable.)
//
// If SB_LOG is set, the user has explicitly asked for diagnostics, and quietly delivering none is
// never the right answer. Fail fast at the seam instead of returning a plausible-looking false.
static bool sb_gx_log_on(const char* chan) {
  if (&sb_log_enabled == nullptr) {
    static int s_checked = 0;
    if (s_checked == 0) {
      s_checked = 1;
      if (const char* e = std::getenv("SB_LOG"); e != nullptr && e[0] != '\0') {
        std::fprintf(stderr,
                     "[aurora] FATAL: SB_LOG=%s was requested but this runtime provides no "
                     "sb_log_enabled — every aurora diagnostic channel would silently report "
                     "nothing, which reads exactly like 'the condition never occurred'. Link a "
                     "channel registry INTO THE EXECUTABLE (a weak undefined reference does not "
                     "pull it out of a static archive).\n",
                     e);
        std::abort();
      }
    }
    return false;
  }
  return sb_log_enabled(chan) != 0;
}

// SB_TIMELINE: ordered per-frame event log shared across TUs (marker changes,
// copies, clears, present) to reconstruct the GC multi-pass frame sequence.
static long g_sbTimelineStart = -1;
extern "C" int sb_timeline_enabled() {
  static int s = -1;
  if (s < 0) {
    const char* e = std::getenv("SB_TIMELINE");
    s = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    // SB_TIMELINE=<startFrame> captures a 3-frame window from that frame.
    g_sbTimelineStart = (s == 1 && e != nullptr) ? std::atol(e) : 0;
    if (g_sbTimelineStart < 1) g_sbTimelineStart = 0;
  }
  return s;
}
static thread_local long g_sbTimelineFrame = 0;
static thread_local long g_sbTimelineSeq = 0;
extern "C" void sb_timeline_log(const char* fmt, ...) {
  if (!sb_timeline_enabled()) return;
  if (g_sbTimelineFrame < g_sbTimelineStart || g_sbTimelineFrame > g_sbTimelineStart + 2) return;
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  std::fprintf(stderr, "[timeline] f%ld #%ld %s\n", g_sbTimelineFrame, g_sbTimelineSeq++, buf);
}
extern "C" void sb_timeline_frame() {
  if (sb_timeline_enabled() && g_sbTimelineFrame >= g_sbTimelineStart && g_sbTimelineFrame <= g_sbTimelineStart + 2)
    std::fprintf(stderr, "[timeline] ===== FRAME %ld =====\n", g_sbTimelineFrame);
  ++g_sbTimelineFrame;
}

static u16 prepare_idx_buffer(ByteBuffer& buf, GXPrimitive prim, u16 vtxStart, u16 vtxCount) {
  return prepare_idx_buffer_impl(
      buf, static_cast<u32>(prim), vtxStart, vtxCount, GX_QUADS, GX_TRIANGLES, GX_TRIANGLEFAN,
      GX_TRIANGLESTRIP, GX_LINES, GX_LINESTRIP, GX_POINTS,
      [](u32 p) { UNLIKELY FATAL("unsupported primitive type {}", p); });
}

// GX FIFO opcodes - use CP_ prefix to avoid clashing with GXCommandList.h macros
static constexpr u8 CP_CMD_NOP = GX_NOP;
static constexpr u8 CP_CMD_LOAD_CP_REG = GX_LOAD_CP_REG;
static constexpr u8 CP_CMD_LOAD_XF_REG = GX_LOAD_XF_REG;
static constexpr u8 CP_CMD_LOAD_INDX_A = GX_LOAD_INDX_A;
static constexpr u8 CP_CMD_LOAD_INDX_B = GX_LOAD_INDX_B;
static constexpr u8 CP_CMD_LOAD_INDX_C = GX_LOAD_INDX_C;
static constexpr u8 CP_CMD_LOAD_INDX_D = GX_LOAD_INDX_D;
static constexpr u8 CP_CMD_CALL_DL = GX_CMD_CALL_DL;
static constexpr u8 CP_CMD_INVAL_VTX = GX_CMD_INVL_VC;
static constexpr u8 CP_CMD_LOAD_BP_REG = GX_LOAD_BP_REG & GX_OPCODE_MASK;

// Primitive type mask
static constexpr u8 CP_OPCODE_MASK = GX_OPCODE_MASK;
static constexpr u8 CP_VAT_MASK = GX_VAT_MASK;

// Read helpers for big/little endian
#if _MSC_VER
template <typename T>
__forceinline // Yes, this was necessary.
    inline T unaligned_load(const T* ptr) {
  return *static_cast<const __unaligned T*>(ptr);
}
#else
template <typename T>
inline T unaligned_load(const T* ptr) {
  T copy;
  memcpy(&copy, ptr, sizeof(T));
  return copy;
}
#endif

static inline u16 read_u16(const u8* ptr, bool bigEndian) {
  const u16 val = unaligned_load(reinterpret_cast<const u16*>(ptr));
  if (bigEndian) {
    return bswap(val);
  }
  return val;
}

static inline u32 read_u32(const u8* ptr, bool bigEndian) {
  const u32 val = unaligned_load(reinterpret_cast<const u32*>(ptr));
  if (bigEndian) {
    return bswap(val);
  }
  return val;
}

static u32 bp_get(u32 reg, u32 size, u32 shift);

static GXPixelFmt decode_pixel_fmt(u32 peCtrl, u32 cmode1) {
  switch (bp_get(peCtrl, 3, 0)) {
  case 0:
    return GX_PF_RGB8_Z24;
  case 1:
    return GX_PF_RGBA6_Z24;
  case 2:
    return GX_PF_RGB565_Z16;
  case 3:
    return GX_PF_Z24;
  case 4:
    switch (bp_get(cmode1, 2, 9)) {
    case 0:
      return GX_PF_Y8;
    case 1:
      return GX_PF_U8;
    case 2:
      return GX_PF_V8;
    default:
      Log.warn("command_processor: unsupported cmode1 pixel subtype {}", bp_get(cmode1, 2, 9));
      return GX_PF_Y8;
    }
  case 5:
    return GX_PF_YUV420;
  default:
    Log.warn("command_processor: unsupported PE pixel format {}", bp_get(peCtrl, 3, 0));
    return GX_PF_RGB8_Z24;
  }
}

static inline u64 read_u64(const u8* ptr, bool bigEndian) {
  u64 loaded;
  // Unaligned-safe load
  memcpy(&loaded, ptr, sizeof(u64));

  if (bigEndian) {
    return bswap(loaded);
  }

  return loaded;
}

struct TexBpRegMapping {
  u8 texMapId;
  enum class Kind : uint8_t { Mode0, Mode1, Image0, Image1, Image2, Image3, Tlut } kind;
};

static std::optional<TexBpRegMapping> decode_tex_bp_reg(u32 regId) {
  constexpr std::array mode0Ids{0x80u, 0x81u, 0x82u, 0x83u, 0xA0u, 0xA1u, 0xA2u, 0xA3u};
  constexpr std::array mode1Ids{0x84u, 0x85u, 0x86u, 0x87u, 0xA4u, 0xA5u, 0xA6u, 0xA7u};
  constexpr std::array image0Ids{0x88u, 0x89u, 0x8Au, 0x8Bu, 0xA8u, 0xA9u, 0xAAu, 0xABu};
  constexpr std::array image1Ids{0x8Cu, 0x8Du, 0x8Eu, 0x8Fu, 0xACu, 0xADu, 0xAEu, 0xAFu};
  constexpr std::array image2Ids{0x90u, 0x91u, 0x92u, 0x93u, 0xB0u, 0xB1u, 0xB2u, 0xB3u};
  constexpr std::array image3Ids{0x94u, 0x95u, 0x96u, 0x97u, 0xB4u, 0xB5u, 0xB6u, 0xB7u};
  constexpr std::array tlutIds{0x98u, 0x99u, 0x9Au, 0x9Bu, 0xB8u, 0xB9u, 0xBAu, 0xBBu};

  for (u8 i = 0; i < MaxTextures; ++i) {
    if (regId == mode0Ids[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Mode0};
    }
    if (regId == mode1Ids[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Mode1};
    }
    if (regId == image0Ids[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Image0};
    }
    if (regId == image1Ids[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Image1};
    }
    if (regId == image2Ids[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Image2};
    }
    if (regId == image3Ids[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Image3};
    }
    if (regId == tlutIds[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Tlut};
    }
  }
  return std::nullopt;
}

// Helper to convert packed RGBA8 to Vec4<float>
static Vec4<float> unpack_color(u32 packed) {
  return {
      static_cast<float>(packed >> 24 & 0xFF) / 255.f,
      static_cast<float>(packed >> 16 & 0xFF) / 255.f,
      static_cast<float>(packed >> 8 & 0xFF) / 255.f,
      static_cast<float>(packed & 0xFF) / 255.f,
  };
}

static inline f32 read_f32(const u8* ptr, bool bigEndian) {
  u32 bits = read_u32(ptr, bigEndian);
  f32 val;
  std::memcpy(&val, &bits, sizeof(val));
  return val;
}

static bool copy_xf_data(u32 addr, const u8* data, u32 len, bool bigEndian) {
  if (addr < 0x78) {
    // Position matrices (0x0000 - 0x0077)
    u32 mtxIdx = addr / 12;
    u32 startOffset = addr % 12;
    // We only support full writes to matrices
    CHECK(mtxIdx < MaxPnMtx, "XF: PosMtx copy oob? Should never happen; mtxIdx={}", mtxIdx);
    CHECK(startOffset == 0 && len == 12, "XF: PosMtx sub-copy unsupported: offs={}, len={}", startOffset, len);
    auto& mtx = g_gxState.pnMtx[mtxIdx].pos;
    f32* flat = reinterpret_cast<f32*>(&mtx);
    for (u32 i = 0; i < len; i++) {
      flat[i] = read_f32(data + i * 4, bigEndian);
    }
    g_gxState.stateDirty = true;
    // Was missing `return true` -- the copy succeeded but reported failure,
    // so every position-matrix indexed load looked "unimplemented" (silent
    // while the failure path was an NDEBUG-gated debug log; exposed the
    // moment it became fatal).
    return true;
  } else if (addr < 0x0F0) {
    // Texture matrices (0x078-0x0EF)
    u32 texBase = addr - 0x078;
    u32 mtxIdx = texBase / 12;
    u32 startOffset = texBase % 12;
    CHECK(mtxIdx < MaxTexMtx, "XF TexMtx copy oob? Should never happen; mtxIdx={}", mtxIdx);
    CHECK(startOffset == 0 && (len == 8 || len == 12), "XF TexMtx sub-copy unsupported: offs={}, len={}", startOffset,
          len);

    // Determine if 2x4 or 3x4 from count
    auto& mtx = g_gxState.texMtxs[mtxIdx];
    f32* flat = reinterpret_cast<f32*>(&mtx);
    for (u32 i = 0; i < len; i++) {
      flat[i] = read_f32(data + i * 4, bigEndian);
    }
    g_gxState.stateDirty = true;
    return true;
  } else if (addr >= 0x400 && addr < 0x45A) {
    // Normal matrices (0x400-0x459)
    u32 nrmBase = addr - 0x400;
    u32 mtxIdx = nrmBase / 9;
    u32 startOffset = nrmBase % 9;
    // We only support full writes to matrices
    CHECK(mtxIdx < MaxPnMtx, "XF: NrmMtx copy oob? Should never happen; mtxIdx={}", mtxIdx);
    CHECK(startOffset == 0 && len == 9, "XF: NrmMtx sub-copy unsupported: offs={}, len={}", startOffset, len);
    auto& mtx = g_gxState.pnMtx[mtxIdx].nrm;
    f32* flat = reinterpret_cast<f32*>(&mtx);
    for (u32 i = 0; i < len; i++) {
      u32 xfIdx = i;
      u32 row = xfIdx / 3;
      u32 col = xfIdx % 3;
      if (row < 3) {
        flat[row * 4 + col] = read_f32(data + i * 4, bigEndian);
      }
    }
    g_gxState.stateDirty = true;
    return true;
  } else if (addr >= 0x500 && addr < 0x5F0) {
    // Post-transform texture matrices (0x500-0x5EF)
    u32 ptBase = addr - 0x500;
    u32 mtxIdx = ptBase / 12;
    u32 startOffset = ptBase % 12;
    CHECK(mtxIdx < MaxPTTexMtx, "XF: PTTexMtx copy oob? Should never happen; mtxIdx={}", mtxIdx);
    CHECK(startOffset == 0 && len == 12, "XF: PTTexMtx sub-copy unsupported: offs={}, len={}", startOffset, len);
    auto& mtx = g_gxState.ptTexMtxs[mtxIdx];
    f32* flat = reinterpret_cast<f32*>(&mtx);
    for (u32 i = 0; i < len; i++) {
      flat[startOffset + i] = read_f32(data + i * 4, bigEndian);
    }
    g_gxState.stateDirty = true;
    return true;
  } else if (addr >= 0x600 && addr < 0x680) {
    // Lights (0x600-0x67F) - 8 lights, 16 values each
    u32 lightBase = addr - 0x600;
    u32 lightIdx = lightBase / 0x10;
    u32 startOffset = lightBase % 0x10;
    CHECK(lightIdx < 8, "XF: Light copy oob? Should never happen; lightIdx={}", lightIdx);
    CHECK(startOffset + len <= 0x10, "XF: Light copy that crosses across light boundaries unsupported: offs={}, len={}",
          startOffset, len);
    auto& light = g_gxState.lights[lightIdx];
    for (u32 i = 0; i < len; i++) {
      u32 field = startOffset + i;
      f32 val = read_f32(data + i * 4, bigEndian);
      u32 ival = read_u32(data + i * 4, bigEndian);
      switch (field) {
      case 3: // Color (packed u32)
        light.color = unpack_color(ival);
        break;
      case 4:
        light.cosAtt[0] = val;
        break; // a0
      case 5:
        light.cosAtt[1] = val;
        break; // a1
      case 6:
        light.cosAtt[2] = val;
        break; // a2
      case 7:
        light.distAtt[0] = val;
        break; // k0
      case 8:
        light.distAtt[1] = val;
        break; // k1
      case 9:
        light.distAtt[2] = val;
        break; // k2
      case 10:
        light.pos[0] = val;
        break; // px
      case 11:
        light.pos[1] = val;
        break; // py
      case 12:
        light.pos[2] = val;
        break; // pz
      case 13:
        light.dir[0] = val;
        break; // nx
      case 14:
        light.dir[1] = val;
        break; // ny
      case 15:
        light.dir[2] = val;
        break; // nz
      default:
        break; // padding (0-2)
      }
    }
    g_gxState.stateDirty = true;
    return true;
  }
  return false;
}

// Forward declarations for register handlers
static void handle_bp(u32 value, bool bigEndian);
static void handle_cp(u8 addr, u32 value, bool bigEndian);
static void handle_xf(const u8* data, u32& pos, u32 size, bool bigEndian);
static void handle_draw(u8 cmd, const u8* data, u32& pos, u32 size, bool bigEndian);

// Per-draw state oracle hooks, defined by the recomp runtime. Weak so aurora links without it, and
// a plain C ABI over arrays so no recomp header has to reach into this layer.
extern "C" {
__attribute__((weak)) bool sbr_state_diff_enabled();
__attribute__((weak)) void sbr_state_oracle_aurora_frame_end();
__attribute__((weak)) void sbr_state_oracle_aurora_raw(unsigned pos, unsigned numStages,
                                                      unsigned numTexGens,
                                                       const unsigned char* texmap,
                                                       const unsigned char* texcoord,
                                                       const unsigned char* texEnable,
                                                       const unsigned* unitId,
                                                       unsigned numChans,
                                                       const unsigned short* chanCtrl,
                                                       const unsigned* ambColor,
                                                       const unsigned* matColor,
                                                       const unsigned char* rasChannel,
                                                       const unsigned* cWord, const unsigned* aWord,
                                                       const unsigned short* kSel,
                                                       const unsigned* konst,
                                                       const unsigned long long* tevReg,
                                                       unsigned raster, unsigned blend,
                                                       const int* scissor, unsigned cull);
}
static void handle_aurora(const u8* data, u32& pos, u32 size, bool bigEndian);

// Ring buffer of recent draws — dumped alongside the opcode ring buffer at
// unknown-opcode FATAL so a fifo desync's originating draw is visible without
// enabling AURORA_DRAW_TRACE (which spams stderr fast enough to skew timing).
struct RecentDraw { u32 pos; u8 cmd; u16 vtxCount; u32 vtxSize; };
static constexpr size_t kRecentDrawN = 16;
static thread_local RecentDraw s_recentDraws[kRecentDrawN];
struct RecentCmd { u32 pos; u8 cmd; };
// 128 (was 32): a fifo desync into vertex/pointer data emits a long run of
// misread short commands (e.g. 31 zero-bytes = 31 NOPs) that floods a small
// ring and hides the pre-desync commands where the mis-advance actually
// happened. 128 reaches back past such a flood to the culprit command.
static constexpr size_t kRecentN = 128;
static thread_local RecentCmd s_recent[kRecentN];
static thread_local size_t s_recentHead = 0;
// Dump the recent-command ring to stderr (diagnostics outside the drain fn).
static void sb_dump_recent_cmds(const char* why) {
  std::string trail;
  for (size_t i = 0; i < kRecentN; ++i) {
    const auto& r = s_recent[(s_recentHead + i) % kRecentN];
    trail += fmt::format(" {}:{:02x}", r.pos, r.cmd);
  }
  std::fprintf(stderr, "[recent-cmds] (%s, oldest first pos:cmd)%s\n", why, trail.c_str());
}

static thread_local size_t s_recentDrawHead = 0;

void process(const u8* data, u32 size, bool bigEndian) {
  ZoneScoped;
  u32 pos = 0;

  // Small ring buffer of recent (opcode, pos) — dumped on unknown-opcode fatal
  // so the caller can see what commands preceded the garbage. AURORA_FIFO_TRACE=1
  // dumps every opcode as it's processed (very noisy; use only when the
  // ring-buffer window is too narrow).
  static thread_local int s_traceEnabled = -1;
  if (s_traceEnabled < 0) {
    const char* env = std::getenv("AURORA_FIFO_TRACE");
    s_traceEnabled = env && env[0] && env[0] != '0' ? 1 : 0;
  }

  // SB_FIFO_TRACE_MARK=<substr>: per-opcode trace of the drain, but only while
  // the last debug marker (draw identity) matches — small enough to read for a
  // single J3D buffer's draw window, unlike the global AURORA_FIFO_TRACE.
  static thread_local const char* s_traceMark = nullptr;
  static thread_local int s_traceMarkInit = 0;
  if (!s_traceMarkInit) {
    s_traceMarkInit = 1;
    s_traceMark = std::getenv("SB_FIFO_TRACE_MARK");
  }
  static thread_local long s_traceMarkBudget = 4000;

  while (pos < size) {
    u32 cmdPos = pos;
    u8 cmd = data[pos++];
    u8 opcode = cmd & CP_OPCODE_MASK;
    s_recent[s_recentHead].pos = cmdPos;
    s_recent[s_recentHead].cmd = cmd;
    s_recentHead = (s_recentHead + 1) % kRecentN;
    if (s_traceEnabled) {
      Log.warn("[fifo trace] pos={} cmd=0x{:02X} opcode=0x{:02X}", cmdPos, cmd, opcode);
    }
    if (s_traceMark != nullptr && s_traceMarkBudget > 0 &&
        g_sbLastMarker.find(s_traceMark) != std::string::npos) {
      --s_traceMarkBudget;
      if (opcode >= 0x80 && opcode < 0xC0) {
        const u16 vc = pos + 2 <= size ? read_u16(data + pos, bigEndian) : 0;
        std::fprintf(stderr, "[fifo-mark] pos=%u cmd=%02x DRAW verts=%u vtxSize=%u fmt=%u\n", cmdPos, cmd, vc,
                     g_gxState.lastVtxFmt == (cmd & CP_VAT_MASK) ? g_gxState.lastVtxSize : 0u,
                     cmd & CP_VAT_MASK);
      } else {
        std::fprintf(stderr, "[fifo-mark] pos=%u cmd=%02x next=[%02x %02x %02x %02x %02x %02x %02x %02x]\n", cmdPos,
                     cmd, pos + 0 < size ? data[pos + 0] : 0, pos + 1 < size ? data[pos + 1] : 0,
                     pos + 2 < size ? data[pos + 2] : 0, pos + 3 < size ? data[pos + 3] : 0,
                     pos + 4 < size ? data[pos + 4] : 0, pos + 5 < size ? data[pos + 5] : 0,
                     pos + 6 < size ? data[pos + 6] : 0, pos + 7 < size ? data[pos + 7] : 0);
      }
    }

    // SB_OPCODE_CENSUS=N: every N presents, tally which FIFO opcodes this runtime actually
    // emits. "Feature X never happens" is otherwise indistinguishable from "the diagnostic for
    // feature X is dead" — this counts the raw stream, so a zero is a measured zero. Compare
    // between runtimes: the decomp runtime is the known-positive for indexed matrix loads
    // (0x20/0x28/0x30/0x38), which is how a zero here is read as a real absence.
    {
      static int s_init = 0;
      static long s_period = 0;
      static long s_counts[32] = {};
      static long s_frames = -1;
      if (!s_init) {
        s_init = 1;
        if (const char* e = std::getenv("SB_OPCODE_CENSUS"); e != nullptr && e[0] != '\0')
          s_period = std::atol(e);
      }
      if (s_period > 0) {
        ++s_counts[(opcode >> 3) & 31];
        const long f = (long)sb_gx_vi_retrace_count();
        if (f != s_frames && (f % s_period) == 0) {
          s_frames = f;
          std::fprintf(stderr, "[opcode-census] frame %ld:", f);
          for (int i = 0; i < 32; ++i)
            if (s_counts[i] != 0) std::fprintf(stderr, " %02x=%ld", i << 3, s_counts[i]);
          std::fprintf(stderr, "\n");
        }
      }
    }

    switch (opcode) {
    case CP_CMD_NOP:
      continue;

    case CP_CMD_LOAD_BP_REG: {
      CHECK(pos + 4 <= size, "BP reg read overrun");
      u32 value = read_u32(data + pos, bigEndian);
      pos += 4;
      handle_bp(value, bigEndian);
      break;
    }

    case CP_CMD_LOAD_CP_REG: {
      CHECK(pos + 5 <= size, "CP reg read overrun");
      u8 addr = data[pos++];
      u32 value = read_u32(data + pos, bigEndian);
      pos += 4;
      handle_cp(addr, value, bigEndian);
      break;
    }

    case CP_CMD_LOAD_XF_REG: {
      handle_xf(data, pos, size, bigEndian);
      break;
    }

    case CP_CMD_LOAD_INDX_A:
    case CP_CMD_LOAD_INDX_B:
    case CP_CMD_LOAD_INDX_C:
    case CP_CMD_LOAD_INDX_D: {
      ZoneScopedN("LOAD_INDX");
      // Indexed XF load, GC encoding: one u32 = index<<16 | (len-1)<<12 | xfAddr.
      // INDX_A..D select the POS_MTX / NRM_MTX / TEX_MTX / LIGHT arrays.
      CHECK(pos + 4 <= size, "indexed XF read overrun");
      const u32 word = read_u32(data + pos, bigEndian);
      pos += 4;
      const u32 arrayType = GX_POS_MTX_ARRAY + ((opcode - CP_CMD_LOAD_INDX_A) >> 3);
      const u16 srcArrayIdx = static_cast<u16>(word >> 16);
      const u16 len = static_cast<u16>(((word >> 12) & 0xF) + 1);
      const u16 dstAddr = word & 0x0FFF;
      auto const& array = g_gxState.arrays[arrayType];
      ASSERT(array.data != nullptr, "indexed XF load (opcode 0x{:02X}) with no array base for attr {}", opcode,
             arrayType);
      const u8* srcData = ((const u8*)array.data) + srcArrayIdx * array.stride;
      // The source endianness is the ARRAY's, not the command stream's:
      // runtime-computed pools (J3D draw/normal matrices) are host-endian
      // even when referenced from a big-endian display list.
      // Silent fails banned (2026-07-14): an indexed XF load into an
      // unimplemented XF region means the matrix/light data the next draws
      // depend on was NOT loaded — crash at the root cause instead of
      // rendering with stale state. (Was an NDEBUG-gated Log.debug.)
      ASSERT(copy_xf_data(dstAddr, srcData, len, !array.le),
             "unimplemented indexed XF load (opcode 0x{:02X}, dstAddr={:04x}, len={})", opcode, dstAddr, len);
      // SB_LOG=pnzero: an indexed pos/nrm matrix load whose source rotation
      // row is all-zero — the upload that poisons the XF palette slot (black
      // patches on later packets that "keep" the slot via 0xFFFF). Logs the
      // ARRAY BASE so the culprit J3DModel can be matched to the game-side
      // [nrmmtx] zeroDrawMtx report (same buffer pointer).
      if (arrayType <= GX_NRM_MTX_ARRAY && sb_gx_log_on("pnzero")) {
        float r0[3];
        for (int i = 0; i < 3; ++i) {
          u32 u;
          std::memcpy(&u, srcData + i * 4, 4);
          if (!array.le) u = __builtin_bswap32(u);
          std::memcpy(&r0[i], &u, 4);
        }
        if (r0[0] == 0.f && r0[1] == 0.f && r0[2] == 0.f)
          sb_logf("pnzero", "%s idx=%u dst=%03x base=%p mark='%s'",
                  arrayType == GX_POS_MTX_ARRAY ? "pos" : "nrm", srcArrayIdx, dstAddr,
                  array.data, g_sbLastMarker.c_str());
      }
      break;
    }

    case CP_CMD_CALL_DL: {
      // Call display list: 8 bytes (address + size)
      CHECK(pos + 8 <= size, "call DL read overrun");
      Log.warn("Ignoring nested GX_CMD_CALL_DL");
      pos += 8;
      break;
    }

    case CP_CMD_INVAL_VTX: {
      // Invalidate vertex cache
      break;
    }

    case GX_AURORA: {
      handle_aurora(data, pos, size, bigEndian);
      break;
    }

    // Draw commands: 0x80-0xBF
    case GX_DRAW_QUADS:
    case GX_DRAW_TRIANGLES:
    case GX_DRAW_TRIANGLE_STRIP:
    case GX_DRAW_TRIANGLE_FAN:
    case GX_DRAW_LINES:
    case GX_DRAW_LINE_STRIP:
    case GX_DRAW_POINTS: {
      handle_draw(cmd, data, pos, size, bigEndian);
      break;
    }

    default:
      // Draw commands live in [0x80..0xBF]: opcode byte is (prim | vat), prim
      // in {0x80,0x90,0x98,0xA0,0xA8,0xB0,0xB8}, vat in [0..7]. Anything at or
      // above 0xC0 is garbage (probably a fifo desync — the previous handler
      // over- or under-consumed its payload); do NOT coerce it into a draw,
      // that just turns a real desync into "unsupported primitive 0xF8" from
      // prepare_idx_buffer several layers deeper. Fall through to the unknown-
      // opcode diagnostic path so the ring buffer + hex dump above surfaces
      // the actual cause.
      if (cmd >= 0x80 && cmd < 0xC0) {
        handle_draw(cmd, data, pos, size, bigEndian);
      } else {
        Log.error("  last draw-identity marker: '{}'", g_sbLastMarker);
        // Hex dump surrounding bytes for debugging
        {
          u32 dumpStart = (pos > 321) ? pos - 321 : 0;
          u32 dumpEnd = (pos + 32 < size) ? pos + 32 : size;
          std::string hex;
          for (u32 i = dumpStart; i < dumpEnd; i++) {
            if (i == pos - 1)
              hex += fmt::format("[{:02x}]", data[i]);
            else
              hex += fmt::format(" {:02x}", data[i]);
          }
          Log.error("  hex dump (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
        }
        // Recent-command ring buffer — what was processed before the garbage.
        // Print each command's byte SPAN (delta to the next recorded command,
        // or to the desync `pos` for the last one): a fifo desync is a single
        // command whose parser advanced by the wrong number of bytes, so the
        // culprit is the entry whose span doesn't match its opcode's real
        // encoding (e.g. a load-CP that should span 6 but the next opcode sits
        // at +5, or an AURORA sub-op that mis-sized a pointer payload).
        {
          std::string trail;
          for (size_t i = 0; i < kRecentN; ++i) {
            const auto& r = s_recent[(s_recentHead + i) % kRecentN];
            if (r.pos == 0 && r.cmd == 0 && i == 0) continue;
            const auto& next = s_recent[(s_recentHead + i + 1) % kRecentN];
            // The chronologically-next command's start (the last entry's "next"
            // is the desync pos itself).
            const u32 endPos = (i + 1 < kRecentN && !(next.pos == 0 && next.cmd == 0)) ? next.pos : pos;
            const long span = static_cast<long>(endPos) - static_cast<long>(r.pos);
            trail += fmt::format(" [pos={} cmd=0x{:02X} span={}]", r.pos, r.cmd, span);
          }
          Log.error("  recent opcodes (oldest first):{}", trail);
        }
        // Recent draws with their vtxCount/vtxSize — where fifo desyncs really
        // originate (a bad vtxSize means the vertex payload was under- or
        // over-consumed, leaving the next byte on a payload boundary).
        {
          std::string trail;
          for (size_t i = 0; i < kRecentDrawN; ++i) {
            const auto& r = s_recentDraws[(s_recentDrawHead + i) % kRecentDrawN];
            if (r.pos == 0 && r.cmd == 0 && i == 0) continue;
            trail += fmt::format(
                " [pos={} cmd=0x{:02X} vtxCount={} vtxSize={} end={}]",
                r.pos, r.cmd, r.vtxCount, r.vtxSize,
                r.pos + 3 + r.vtxCount * r.vtxSize);
          }
          Log.error("  recent draws (oldest first):{}", trail);
        }
        {
          // Raw hexdump around the desync point: the bytes BEFORE pos show what
          // payload the decoder just mis-consumed.
          std::string hex;
          const size_t lo = (pos >= 129) ? pos - 129 : 0;
          const size_t hi = (pos + 31 < size) ? pos + 31 : size;
          for (size_t i = lo; i < hi; ++i) {
            if ((i - lo) % 16 == 0) hex += fmt::format("\n    {:07}:", i);
            hex += fmt::format(" {:02x}", data[i]);
          }
          Log.error("  bytes around pos {}:{}", pos - 1, hex);
        }
        FATAL("command_processor: unknown opcode 0x{:02X} at pos {} (total fifo size {})",
              cmd, pos - 1, size);
      }
      break;
    }
  }
}

// Helper to extract bit fields from a 32-bit register
inline static u32 bp_get(u32 reg, u32 size, u32 shift) { return reg >> shift & (1u << size) - 1; }

// BP register handler - decodes BP (RAS/pixel engine) register writes and updates g_gxState
static constexpr size_t kRecentBpN = 12;
static thread_local u32 s_recentBp[kRecentBpN];
static thread_local size_t s_recentBpHead = 0;

static void handle_bp(u32 value, bool bigEndian) {
  u32 regId = (value >> 24) & 0xFF;
  s_recentBp[s_recentBpHead] = value;
  s_recentBpHead = (s_recentBpHead + 1) % kRecentBpN;
  // Mask off the register ID from the value for field extraction
  // (the regId is stored in bits 24-31, data is in bits 0-23)

  if (regId == 0xFE) {
    g_gxState.bpRegCache[regId] = value & 0x00FFFFFF;
    return;
  } else {
    u32 ssMask = g_gxState.bpRegCache[0xFE];
    g_gxState.bpRegCache[0xFE] = 0x00FFFFFF;
    const u32 merged = (g_gxState.bpRegCache[regId] & ~ssMask) | (value & ssMask);
    value = (regId << 24) | (merged & 0x00FFFFFF);
    if (g_gxState.bpRegCache[regId] == value && regId != 0x52)
      return;
    g_gxState.bpRegCache[regId] = value;
  }

  // TEV color combiner stages (0xC0, 0xC2, 0xC4, ... 0xDE)
  if (regId >= 0xC0 && regId <= 0xDE && (regId & 1) == 0) {
    u32 stage = (regId - 0xC0) / 2;
    if (stage < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage];
      s.colorPass.d = static_cast<GXTevColorArg>(bp_get(value, 4, 0));
      s.colorPass.c = static_cast<GXTevColorArg>(bp_get(value, 4, 4));
      s.colorPass.b = static_cast<GXTevColorArg>(bp_get(value, 4, 8));
      s.colorPass.a = static_cast<GXTevColorArg>(bp_get(value, 4, 12));
      s.colorOp.clamp = bp_get(value, 1, 19) != 0;
      s.colorOp.outReg = static_cast<GXTevRegID>(bp_get(value, 2, 22));
      if (bp_get(value, 2, 16) == 3) {
        // Bias==3 means compare mode: reconstruct GXTevOp enum (8 + 3-bit hw value)
        u32 hwOp = bp_get(value, 1, 18) | (bp_get(value, 2, 20) << 1);
        s.colorOp.op = static_cast<GXTevOp>(hwOp + 8);
        s.colorOp.bias = GX_TB_ZERO;
        s.colorOp.scale = GX_CS_SCALE_1;
      } else {
        // Normal mode: bit18 is op (0=ADD, 1=SUB), bits16-17 is bias, bits20-21 is scale
        s.colorOp.op = static_cast<GXTevOp>(bp_get(value, 1, 18));
        s.colorOp.bias = static_cast<GXTevBias>(bp_get(value, 2, 16));
        s.colorOp.scale = static_cast<GXTevScale>(bp_get(value, 2, 20));
      }
      g_gxState.stateDirty = true;
    }
    return;
  }

  // TEV alpha combiner stages (0xC1, 0xC3, 0xC5, ... 0xDF)
  if (regId >= 0xC1 && regId <= 0xDF && (regId & 1) == 1) {
    u32 stage = (regId - 0xC1) / 2;
    if (stage < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage];
      s.tevSwapRas = static_cast<GXTevSwapSel>(bp_get(value, 2, 0));
      s.tevSwapTex = static_cast<GXTevSwapSel>(bp_get(value, 2, 2));
      s.alphaPass.d = static_cast<GXTevAlphaArg>(bp_get(value, 3, 4));
      s.alphaPass.c = static_cast<GXTevAlphaArg>(bp_get(value, 3, 7));
      s.alphaPass.b = static_cast<GXTevAlphaArg>(bp_get(value, 3, 10));
      s.alphaPass.a = static_cast<GXTevAlphaArg>(bp_get(value, 3, 13));
      s.alphaOp.clamp = bp_get(value, 1, 19) != 0;
      s.alphaOp.outReg = static_cast<GXTevRegID>(bp_get(value, 2, 22));
      if (bp_get(value, 2, 16) == 3) {
        u32 hwOp = bp_get(value, 1, 18) | (bp_get(value, 2, 20) << 1);
        s.alphaOp.op = static_cast<GXTevOp>(hwOp + 8);
        s.alphaOp.bias = GX_TB_ZERO;
        s.alphaOp.scale = GX_CS_SCALE_1;
      } else {
        s.alphaOp.op = static_cast<GXTevOp>(bp_get(value, 1, 18));
        s.alphaOp.bias = static_cast<GXTevBias>(bp_get(value, 2, 16));
        s.alphaOp.scale = static_cast<GXTevScale>(bp_get(value, 2, 20));
      }
      g_gxState.stateDirty = true;
    }
    return;
  }

  switch (regId) {
  // genMode (0x00)
  case 0x00: {
    g_gxState.numTexGens = bp_get(value, 4, 0);
    g_gxState.numChans = bp_get(value, 3, 4);
    g_gxState.numTevStages = bp_get(value, 4, 10) + 1;
    u32 hwCull = bp_get(value, 2, 14);
    // Swap front/back to match GX convention
    switch (hwCull) {
    case GX_CULL_FRONT:
      g_gxState.cullMode = GX_CULL_BACK;
      break;
    case GX_CULL_BACK:
      g_gxState.cullMode = GX_CULL_FRONT;
      break;
    default:
      g_gxState.cullMode = static_cast<GXCullMode>(hwCull);
      break;
    }
    g_gxState.numIndStages = bp_get(value, 3, 16);
    g_gxState.stateDirty = true;
    break;
  }

  // BP mask (0x0F) - internal, applies to next BP write
  case 0x0F:
#ifndef NDEBUG
    Log.debug("BP mask set to {:06x}, but selective updates are not implemented", value & 0xFFFFFF);
#endif
    break;

  // TEV indirect stages (0x10-0x1F)
  case 0x10:
  case 0x11:
  case 0x12:
  case 0x13:
  case 0x14:
  case 0x15:
  case 0x16:
  case 0x17:
  case 0x18:
  case 0x19:
  case 0x1A:
  case 0x1B:
  case 0x1C:
  case 0x1D:
  case 0x1E:
  case 0x1F: {
    u32 stage = regId - 0x10;
    if (stage < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage];
      s.indTexStage = static_cast<GXIndTexStageID>(bp_get(value, 2, 0));
      s.indTexFormat = static_cast<GXIndTexFormat>(bp_get(value, 2, 2));
      s.indTexBiasSel = static_cast<GXIndTexBiasSel>(bp_get(value, 3, 4));
      s.indTexAlphaSel = static_cast<GXIndTexAlphaSel>(bp_get(value, 2, 7));
      s.indTexMtxId = static_cast<GXIndTexMtxID>(bp_get(value, 4, 9));
      s.indTexWrapS = static_cast<GXIndTexWrap>(bp_get(value, 3, 13));
      s.indTexWrapT = static_cast<GXIndTexWrap>(bp_get(value, 3, 16));
      s.indTexUseOrigLOD = bp_get(value, 1, 19) != 0;
      s.indTexAddPrev = bp_get(value, 1, 20) != 0;
      g_gxState.stateDirty = true;
    }
    break;
  }

  // Scissor registers (0x20, 0x21)
  case 0x20:
  case 0x21: {
    const u32 scis0 = g_gxState.bpRegCache[0x20];
    const u32 scis1 = g_gxState.bpRegCache[0x21];
    const int32_t tp = static_cast<int32_t>(bp_get(scis0, 11, 0)) - 342;
    const int32_t lf = static_cast<int32_t>(bp_get(scis0, 11, 12)) - 342;
    const int32_t bm = static_cast<int32_t>(bp_get(scis1, 11, 0)) - 342;
    const int32_t rt = static_cast<int32_t>(bp_get(scis1, 11, 12)) - 342;
    const int32_t wd = std::max(rt - lf + 1, 0);
    const int32_t ht = std::max(bm - tp + 1, 0);
    set_logical_scissor({lf, tp, wd, ht});
    break;
  }

  // Line/point size (0x22)
  case 0x22: {
    g_gxState.lineWidth = static_cast<u8>(bp_get(value, 8, 0));
    g_gxState.pointSize = static_cast<u8>(bp_get(value, 8, 8));
    g_gxState.lineTexOffset = static_cast<GXTexOffset>(bp_get(value, 3, 16));
    g_gxState.pointTexOffset = static_cast<GXTexOffset>(bp_get(value, 3, 19));
    g_gxState.lineHalfAspect = bp_get(value, 1, 22) != 0;
    g_gxState.stateDirty = true;
    break;
  }

  // Indirect texture scale (0x25, 0x26)
  case 0x25: {
    if (MaxIndStages > 0) {
      g_gxState.indStages[0].scaleS = static_cast<GXIndTexScale>(bp_get(value, 4, 0));
      g_gxState.indStages[0].scaleT = static_cast<GXIndTexScale>(bp_get(value, 4, 4));
    }
    if (MaxIndStages > 1) {
      g_gxState.indStages[1].scaleS = static_cast<GXIndTexScale>(bp_get(value, 4, 8));
      g_gxState.indStages[1].scaleT = static_cast<GXIndTexScale>(bp_get(value, 4, 12));
    }
    g_gxState.stateDirty = true;
    break;
  }
  case 0x26: {
    if (MaxIndStages > 2) {
      g_gxState.indStages[2].scaleS = static_cast<GXIndTexScale>(bp_get(value, 4, 0));
      g_gxState.indStages[2].scaleT = static_cast<GXIndTexScale>(bp_get(value, 4, 4));
    }
    if (MaxIndStages > 3) {
      g_gxState.indStages[3].scaleS = static_cast<GXIndTexScale>(bp_get(value, 4, 8));
      g_gxState.indStages[3].scaleT = static_cast<GXIndTexScale>(bp_get(value, 4, 12));
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Indirect texture reference (0x27)
  case 0x27: {
    for (u32 i = 0; i < MaxIndStages; i++) {
      g_gxState.indStages[i].texMapId = static_cast<GXTexMapID>(bp_get(value, 3, i * 6));
      g_gxState.indStages[i].texCoordId = static_cast<GXTexCoordID>(bp_get(value, 3, i * 6 + 3));
    }
    g_gxState.stateDirty = true;
    break;
  }

  // TEV order / tref (0x28-0x2F) - 2 stages per register
  case 0x28:
  case 0x29:
  case 0x2A:
  case 0x2B:
  case 0x2C:
  case 0x2D:
  case 0x2E:
  case 0x2F: {
    u32 idx = regId - 0x28;
    u32 stage0 = idx * 2;
    u32 stage1 = idx * 2 + 1;

    // Channel ID reverse mapping from hardware to GX
    static const GXChannelID r2c[] = {GX_COLOR0A0, GX_COLOR1A1,   GX_COLOR0A0,    GX_COLOR1A1,
                                      GX_COLOR0A0, GX_ALPHA_BUMP, GX_ALPHA_BUMPN, GX_COLOR_ZERO};

    if (stage0 < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage0];
      s.texMapId = static_cast<GXTexMapID>(bp_get(value, 3, 0));
      s.texCoordId = static_cast<GXTexCoordID>(bp_get(value, 3, 3));
      // bit 6 = tex enable
      if (!bp_get(value, 1, 6)) {
        s.texMapId = GX_TEXMAP_NULL;
      }
      u32 chanHw = bp_get(value, 3, 7);
      s.channelId = (chanHw < 8) ? r2c[chanHw] : GX_COLOR_NULL;
    }
    if (stage1 < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage1];
      s.texMapId = static_cast<GXTexMapID>(bp_get(value, 3, 12));
      s.texCoordId = static_cast<GXTexCoordID>(bp_get(value, 3, 15));
      if (!bp_get(value, 1, 18)) {
        s.texMapId = GX_TEXMAP_NULL;
      }
      u32 chanHw = bp_get(value, 3, 19);
      s.channelId = (chanHw < 8) ? r2c[chanHw] : GX_COLOR_NULL;
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Z mode (0x40)
  case 0x40: {
    g_gxState.depthCompare = bp_get(value, 1, 0) != 0;
    g_gxState.depthFunc = static_cast<GXCompare>(bp_get(value, 3, 1));
    g_gxState.depthUpdate = bp_get(value, 1, 4) != 0;
    g_gxState.stateDirty = true;
    break;
  }

  // Blend mode / cmode0 (0x41)
  case 0x41: {
    bool blendEn = bp_get(value, 1, 0) != 0;
    bool logicEn = bp_get(value, 1, 1) != 0;
    bool dither = bp_get(value, 1, 2) != 0;
    g_gxState.colorUpdate = bp_get(value, 1, 3) != 0;
    g_gxState.alphaUpdate = bp_get(value, 1, 4) != 0;
    g_gxState.blendFacDst = static_cast<GXBlendFactor>(bp_get(value, 3, 5));
    g_gxState.blendFacSrc = static_cast<GXBlendFactor>(bp_get(value, 3, 8));
    bool subtract = bp_get(value, 1, 11) != 0;
    g_gxState.blendOp = static_cast<GXLogicOp>(bp_get(value, 4, 12));

    if (subtract) {
      g_gxState.blendMode = GX_BM_SUBTRACT;
    } else if (blendEn) {
      g_gxState.blendMode = GX_BM_BLEND;
    } else if (logicEn) {
      g_gxState.blendMode = GX_BM_LOGIC;
    } else {
      g_gxState.blendMode = GX_BM_NONE;
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Dst alpha / cmode1 (0x42)
  case 0x42: {
    u8 alpha = bp_get(value, 8, 0);
    bool enabled = bp_get(value, 1, 8) != 0;
    g_gxState.dstAlpha = enabled ? alpha : UINT32_MAX;
    g_gxState.pixelFmt = decode_pixel_fmt(g_gxState.bpRegCache[0x43], value);
    g_gxState.stateDirty = true;
    break;
  }

  // PE control (0x43) - pixel format, z format, zcomp location
  case 0x43: {
    g_gxState.pixelFmt = decode_pixel_fmt(value, g_gxState.bpRegCache[0x42]);
    g_gxState.zFmt = static_cast<GXZFmt16>(bp_get(value, 3, 3));
    g_gxState.zCompLocBeforeTex = bp_get(value, 1, 6) != 0;
    g_gxState.stateDirty = true;
    break;
  }

  // EFB copy trigger (texture copy or, with the copy_to_xfb bit, the display
  // copy that defines the presented TV image).
  case 0x52: {
    const bool clear = bp_get(value, 1, 11) != 0;
    if (bp_get(value, 1, 14) != 0) {
      // copy_to_xfb: resolve through the same copy_tex path keyed on
      // kDisplayCopyDest so it latches as the present source — identical to
      // what GXCopyDisp does via its dest-key discriminator. Reached by
      // in-stream triggers (FIFO replay); the copy src/dst state must have
      // been loaded via GX_AURORA_LOAD_COPY_{SRC,DST} beforehand.
      copy_tex(kDisplayCopyDest, clear);
    } else {
      copy_tex(g_gxState.texCopyDest, clear);
    }
    break;
  }

  // TLUT load address / execute (0x64, 0x65)
  case 0x64:
    break;
  case 0x65: {
    const auto idx = bp_get(value, 10, 0);
    if (idx < MaxTluts) {
      auto& slot = g_gxState.loadedTluts[idx];
      slot.loadTlut0 = g_gxState.bpRegCache[0x64];
      slot.numEntries = static_cast<u16>(bp_get(value, 10, 10) + 1);
    }
    break;
  }

  // Alpha compare (0xF3)
  case 0xF3: {
    g_gxState.alphaCompare.ref0 = bp_get(value, 8, 0);
    g_gxState.alphaCompare.ref1 = bp_get(value, 8, 8);
    g_gxState.alphaCompare.comp0 = static_cast<GXCompare>(bp_get(value, 3, 16));
    g_gxState.alphaCompare.comp1 = static_cast<GXCompare>(bp_get(value, 3, 19));
    g_gxState.alphaCompare.op = static_cast<GXAlphaOp>(bp_get(value, 2, 22));
    g_gxState.stateDirty = true;
    break;
  }

  // TEV K color/alpha select (0xF6-0xFD)
  case 0xF6:
  case 0xF7:
  case 0xF8:
  case 0xF9:
  case 0xFA:
  case 0xFB:
  case 0xFC:
  case 0xFD: {
    u32 kselIdx = regId - 0xF6;
    // Swap table entries (packed into pairs of ksel registers)
    if (kselIdx < MaxTevSwap * 2) {
      u32 swapIdx = kselIdx / 2;
      if (kselIdx & 1) {
        g_gxState.tevSwapTable[swapIdx].blue = static_cast<GXTevColorChan>(bp_get(value, 2, 0));
        g_gxState.tevSwapTable[swapIdx].alpha = static_cast<GXTevColorChan>(bp_get(value, 2, 2));
      } else {
        g_gxState.tevSwapTable[swapIdx].red = static_cast<GXTevColorChan>(bp_get(value, 2, 0));
        g_gxState.tevSwapTable[swapIdx].green = static_cast<GXTevColorChan>(bp_get(value, 2, 2));
      }
    }
    // K color/alpha selection for 2 stages per register
    u32 stage0 = kselIdx * 2;
    u32 stage1 = kselIdx * 2 + 1;
    if (stage0 < MaxTevStages) {
      g_gxState.tevStages[stage0].kcSel = static_cast<GXTevKColorSel>(bp_get(value, 5, 4));
      g_gxState.tevStages[stage0].kaSel = static_cast<GXTevKAlphaSel>(bp_get(value, 5, 9));
    }
    if (stage1 < MaxTevStages) {
      g_gxState.tevStages[stage1].kcSel = static_cast<GXTevKColorSel>(bp_get(value, 5, 14));
      g_gxState.tevStages[stage1].kaSel = static_cast<GXTevKAlphaSel>(bp_get(value, 5, 19));
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Fog A/B parameters (0xEE-0xF0)
  // FOG0 (0xEE): A parameter - sign(1)|exp(8)|mantissa(11) partial IEEE 754 float
  case 0xEE: {
    g_gxState.fog.fog0Raw = value;
    // Reconstruct A = a_encoded * 2^b_s
    u32 a_mant = bp_get(value, 11, 0);
    u32 a_exp = bp_get(value, 8, 11);
    u32 a_sign = bp_get(value, 1, 19);
    u32 a_bits = (a_sign << 31) | (a_exp << 23) | (a_mant << 12);
    float a_encoded;
    std::memcpy(&a_encoded, &a_bits, sizeof(a_encoded));
    u32 b_s = g_gxState.fog.fog2Raw & 0x1F;
    g_gxState.fog.a = std::ldexp(a_encoded, static_cast<int>(b_s));
    g_gxState.stateDirty = true;
    break;
  }
  // FOG1 (0xEF): B mantissa (24-bit)
  case 0xEF: {
    g_gxState.fog.fog1Raw = value;
    u32 b_m = bp_get(value, 24, 0);
    u32 b_s = g_gxState.fog.fog2Raw & 0x1F;
    float B_mant = static_cast<float>(b_m) / 8388638.0f;
    g_gxState.fog.b = std::ldexp(B_mant, static_cast<int>(b_s) - 1);
    g_gxState.stateDirty = true;
    break;
  }
  // FOG2 (0xF0): B shift/exponent (5-bit)
  case 0xF0: {
    g_gxState.fog.fog2Raw = value;
    u32 b_s = bp_get(value, 5, 0);
    // Recompute A with updated b_s
    u32 a_mant = bp_get(g_gxState.fog.fog0Raw, 11, 0);
    u32 a_exp = bp_get(g_gxState.fog.fog0Raw, 8, 11);
    u32 a_sign = bp_get(g_gxState.fog.fog0Raw, 1, 19);
    u32 a_bits = (a_sign << 31) | (a_exp << 23) | (a_mant << 12);
    float a_encoded;
    std::memcpy(&a_encoded, &a_bits, sizeof(a_encoded));
    g_gxState.fog.a = std::ldexp(a_encoded, static_cast<int>(b_s));
    // Recompute B with updated b_s
    u32 b_m = bp_get(g_gxState.fog.fog1Raw, 24, 0);
    float B_mant = static_cast<float>(b_m) / 8388638.0f;
    g_gxState.fog.b = std::ldexp(B_mant, static_cast<int>(b_s) - 1);
    g_gxState.stateDirty = true;
    break;
  }

  // Fog type + C parameter from FOG3 (0xF1)
  case 0xF1: {
    GXFogType fogType = static_cast<GXFogType>(bp_get(value, 3, 21));
    g_gxState.fog.type = fogType;
    // Decode C parameter (same partial float encoding as A)
    u32 c_mant = bp_get(value, 11, 0);
    u32 c_exp = bp_get(value, 8, 11);
    u32 c_sign = bp_get(value, 1, 19);
    u32 c_bits = (c_sign << 31) | (c_exp << 23) | (c_mant << 12);
    std::memcpy(&g_gxState.fog.c, &c_bits, sizeof(g_gxState.fog.c));
    g_gxState.stateDirty = true;
    break;
  }

  // Fog color from FOGCLR (0xF2)
  case 0xF2: {
    u8 b = bp_get(value, 8, 0);
    u8 g = bp_get(value, 8, 8);
    u8 r = bp_get(value, 8, 16);
    g_gxState.fog.color = {
        static_cast<float>(r) / 255.f,
        static_cast<float>(g) / 255.f,
        static_cast<float>(b) / 255.f,
        1.f,
    };
    g_gxState.stateDirty = true;
    break;
  }

  // TEV color registers / K color registers (0xE0-0xE7)
  // RA registers: 0xE0, 0xE2, 0xE4, 0xE6 (even)
  // BG registers: 0xE1, 0xE3, 0xE5, 0xE7 (odd)
  // Bit 23 distinguishes: 0 = TEV color register, 1 = K color register
  case 0xE0:
  case 0xE1:
  case 0xE2:
  case 0xE3:
  case 0xE4:
  case 0xE5:
  case 0xE6:
  case 0xE7: {
    u32 idx = (regId - 0xE0) / 2;
    bool isRA = (regId & 1) == 0;
    bool isKColor = bp_get(value, 1, 23) != 0;

    if (isKColor) {
      // K color register (8-bit components)
      if (idx < GX_MAX_KCOLOR) {
        auto& kc = g_gxState.kcolors[idx];
        if (isRA) {
          kc[0] = static_cast<float>(bp_get(value, 8, 0)) / 255.f;  // R
          kc[3] = static_cast<float>(bp_get(value, 8, 12)) / 255.f; // A
        } else {
          kc[2] = static_cast<float>(bp_get(value, 8, 0)) / 255.f;  // B
          kc[1] = static_cast<float>(bp_get(value, 8, 12)) / 255.f; // G
        }
        g_gxState.stateDirty = true;
      }
    } else {
      // TEV color register (11-bit signed components)
      if (idx < MaxTevRegs) {
        auto& cr = g_gxState.colorRegs[idx];
        if (isRA) {
          // 11-bit signed: sign-extend from 11 bits
          s32 r = bp_get(value, 11, 0);
          if (r & 0x400)
            r |= ~0x7FF; // sign extend
          s32 a = bp_get(value, 11, 12);
          if (a & 0x400)
            a |= ~0x7FF;
          cr[0] = static_cast<float>(r) / 255.f;
          cr[3] = static_cast<float>(a) / 255.f;
        } else {
          s32 b = bp_get(value, 11, 0);
          if (b & 0x400)
            b |= ~0x7FF;
          s32 g = bp_get(value, 11, 12);
          if (g & 0x400)
            g |= ~0x7FF;
          cr[2] = static_cast<float>(b) / 255.f;
          cr[1] = static_cast<float>(g) / 255.f;
        }
        g_gxState.stateDirty = true;
      }
    }
    break;
  }

  // Indirect texture matrices (0x06-0x0E)
  // Each matrix uses 3 consecutive registers (one per row of the 3x2 matrix).
  // Matrix 0: 0x06-0x08, Matrix 1: 0x09-0x0B, Matrix 2: 0x0C-0x0E
  case 0x06:
  case 0x07:
  case 0x08:
  case 0x09:
  case 0x0A:
  case 0x0B:
  case 0x0C:
  case 0x0D:
  case 0x0E: {
    u32 idx = (regId - 0x06) / 3;    // matrix index (0-2)
    u32 column = (regId - 0x06) % 3; // column index (0-2)
    auto& info = g_gxState.indTexMtxs[idx];

    // Decode one packed matrix column: [m[0][column], m[1][column]].
    s32 col0 = bp_get(value, 11, 0);
    if (col0 & 0x400)
      col0 |= ~0x7FF; // sign-extend from 11 bits
    s32 col1 = bp_get(value, 11, 11);
    if (col1 & 0x400)
      col1 |= ~0x7FF;

    auto& packedColumn = column == 0 ? info.mtx.m0 : (column == 1 ? info.mtx.m1 : info.mtx.m2);
    packedColumn.x = static_cast<float>(col0) / 1024.0f;
    packedColumn.y = static_cast<float>(col1) / 1024.0f;

    // Accumulate the indirect matrix scale exponent. The SDK writes two bits per column, but
    // the hardware appears to ignore the top bit from the third column, leaving an effective
    // 5-bit value for adjScale = scaleExp + 17.
    u32 scaleBits = bp_get(value, 2, 22);
    u32 shift = column * 2;
    if (column == 2) {
      info.adjScaleRaw = (info.adjScaleRaw & ~(1u << shift)) | ((scaleBits & 1u) << shift);
    } else {
      info.adjScaleRaw = (info.adjScaleRaw & ~(3u << shift)) | (scaleBits << shift);
    }
    info.scaleExp = static_cast<s8>(info.adjScaleRaw) - 17;

    g_gxState.stateDirty = true;
    break;
  }

  // SU texture coordinate scale registers (0x30-0x3F)
  // Even registers (suTs0): S-axis scale, bias, cyl wrap, line/point offset
  // Odd registers (suTs1): T-axis scale, bias, cyl wrap
  case 0x30:
  case 0x31:
  case 0x32:
  case 0x33:
  case 0x34:
  case 0x35:
  case 0x36:
  case 0x37:
  case 0x38:
  case 0x39:
  case 0x3A:
  case 0x3B:
  case 0x3C:
  case 0x3D:
  case 0x3E:
  case 0x3F: {
    u32 coordIdx = (regId - 0x30) / 2;
    bool isT = (regId & 1) != 0;
    auto& tcs = g_gxState.texCoordScales[coordIdx];
    if (isT) {
      tcs.scaleT = static_cast<u16>(bp_get(value, 16, 0));
      tcs.biasT = bp_get(value, 1, 16) != 0;
      tcs.cylWrapT = bp_get(value, 1, 17) != 0;
    } else {
      tcs.scaleS = static_cast<u16>(bp_get(value, 16, 0));
      tcs.biasS = bp_get(value, 1, 16) != 0;
      tcs.cylWrapS = bp_get(value, 1, 17) != 0;
      tcs.lineOffset = bp_get(value, 1, 18) != 0;
      tcs.pointOffset = bp_get(value, 1, 19) != 0;
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Copy clear color (0x4F-0x50) and depth (0x51)
  case 0x4F: {
    u8 r = bp_get(value, 8, 0);
    u8 a = bp_get(value, 8, 8);
    g_gxState.clearColor[0] = static_cast<float>(r) / 255.f;
    g_gxState.clearColor[3] = static_cast<float>(a) / 255.f;
    if (sb_copy_dbg_chan()) {
      static long n = 0;
      std::fprintf(stderr, "[bp-clear] n=%ld reg=4F r=%u a=%u mark='%s'\n", ++n, r, a, g_sbLastMarker.c_str());
    }
    g_gxState.stateDirty = true;
    break;
  }
  case 0x50: {
    u8 b = bp_get(value, 8, 0);
    u8 g = bp_get(value, 8, 8);
    g_gxState.clearColor[2] = static_cast<float>(b) / 255.f;
    g_gxState.clearColor[1] = static_cast<float>(g) / 255.f;
    if (sb_copy_dbg_chan()) {
      static long n = 0;
      std::fprintf(stderr, "[bp-clear] n=%ld reg=50 b=%u g=%u val=%08x mark='%s'\n", ++n, b, g, value,
                   g_sbLastMarker.c_str());
      if (n <= 6) {
        sb_dump_recent_cmds("bp-0x50");
        std::string bps;
        for (size_t i = 0; i < kRecentBpN; ++i)
          bps += fmt::format(" {:08x}", s_recentBp[(s_recentBpHead + i) % kRecentBpN]);
        std::fprintf(stderr, "[recent-bp] (oldest first)%s\n", bps.c_str());
      }
    }
    g_gxState.stateDirty = true;
    break;
  }
  case 0x51: {
    g_gxState.clearDepth = bp_get(value, 24, 0);
    g_gxState.stateDirty = true;
    break;
  }

  default:
    if (const auto mapping = decode_tex_bp_reg(regId); mapping.has_value()) {
      auto& slot = g_gxState.loadedTextures[mapping->texMapId];
      switch (mapping->kind) {
      case TexBpRegMapping::Kind::Mode0:
        slot.mode0 = value;
        break;
      case TexBpRegMapping::Kind::Mode1:
        slot.mode1 = value;
        break;
      case TexBpRegMapping::Kind::Image0:
        slot.image0 = value;
        slot.mWidth = 0;
        slot.mHeight = 0;
        slot.mFormat = gfx::InvalidTextureFormat;
        break;
      case TexBpRegMapping::Kind::Image3:
        slot.image3 = value;
        break;
      case TexBpRegMapping::Kind::Tlut:
        // TLUT region's TMEM offset
        break;
      case TexBpRegMapping::Kind::Image1:
      case TexBpRegMapping::Kind::Image2:
        // GXTexRegion regs
        break;
      }
      g_gxState.stateDirty = true;
    } else {
#ifndef NDEBUG
      Log.debug("Unhandled BP register 0x{:02X} (value 0x{:06X})", regId, value & 0xFFFFFF);
#endif
    }
    break;
  }
}

// CP register handler - decodes CP register writes and updates g_gxState
static void handle_cp(u8 addr, u32 value, bool bigEndian) {
  switch (addr) {
  // VCD low (0x50)
  case 0x50: {
    auto& vd = g_gxState.vtxDesc;
    vd[GX_VA_PNMTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 0));
    vd[GX_VA_TEX0MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 1));
    vd[GX_VA_TEX1MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 2));
    vd[GX_VA_TEX2MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 3));
    vd[GX_VA_TEX3MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 4));
    vd[GX_VA_TEX4MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 5));
    vd[GX_VA_TEX5MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 6));
    vd[GX_VA_TEX6MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 7));
    vd[GX_VA_TEX7MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 8));
    vd[GX_VA_POS] = static_cast<GXAttrType>(bp_get(value, 2, 9));
    vd[GX_VA_NRM] = static_cast<GXAttrType>(bp_get(value, 2, 11));
    vd[GX_VA_CLR0] = static_cast<GXAttrType>(bp_get(value, 2, 13));
    vd[GX_VA_CLR1] = static_cast<GXAttrType>(bp_get(value, 2, 15));
    g_gxState.stateDirty = true;
    g_gxState.clearVtxSizeCache();
    break;
  }

  // VCD high (0x60)
  case 0x60: {
    auto& vd = g_gxState.vtxDesc;
    vd[GX_VA_TEX0] = static_cast<GXAttrType>(bp_get(value, 2, 0));
    vd[GX_VA_TEX1] = static_cast<GXAttrType>(bp_get(value, 2, 2));
    vd[GX_VA_TEX2] = static_cast<GXAttrType>(bp_get(value, 2, 4));
    vd[GX_VA_TEX3] = static_cast<GXAttrType>(bp_get(value, 2, 6));
    vd[GX_VA_TEX4] = static_cast<GXAttrType>(bp_get(value, 2, 8));
    vd[GX_VA_TEX5] = static_cast<GXAttrType>(bp_get(value, 2, 10));
    vd[GX_VA_TEX6] = static_cast<GXAttrType>(bp_get(value, 2, 12));
    vd[GX_VA_TEX7] = static_cast<GXAttrType>(bp_get(value, 2, 14));
    g_gxState.stateDirty = true;
    g_gxState.clearVtxSizeCache();
    break;
  }

  // Matrix index A (0x30)
  case 0x30: {
    g_gxState.currentPnMtx = bp_get(value, 6, 0) / 3;
    g_gxState.stateDirty = true;
    break;
  }

  // Matrix index B (0x40)
  case 0x40:
    // Texture matrix indices - used for multi-matrix texgen
    break;

  default:
    // VAT A registers (0x70-0x77)
    if (addr >= 0x70 && addr <= 0x77) {
      u32 fmt = addr - 0x70;
      auto& vf = g_gxState.vtxFmts[fmt];
      vf.attrs[GX_VA_POS].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 0));
      vf.attrs[GX_VA_POS].type = static_cast<GXCompType>(bp_get(value, 3, 1));
      vf.attrs[GX_VA_POS].frac = static_cast<u8>(bp_get(value, 5, 4));
      const auto nrm_cnt = bp_get(value, 1, 9);
      const auto nrm_nbt3 = bp_get(value, 1, 31);
      vf.attrs[GX_VA_NRM].cnt = static_cast<GXCompCnt>(nrm_nbt3 ? GX_NRM_NBT3 : (nrm_cnt ? GX_NRM_NBT : GX_NRM_XYZ));
      vf.attrs[GX_VA_NRM].type = static_cast<GXCompType>(bp_get(value, 3, 10));
      if (vf.attrs[GX_VA_NRM].type == GX_U8 || vf.attrs[GX_VA_NRM].type == GX_S8) {
        vf.attrs[GX_VA_NRM].frac = 6;
      } else if (vf.attrs[GX_VA_NRM].type == GX_U16 || vf.attrs[GX_VA_NRM].type == GX_S16) {
        vf.attrs[GX_VA_NRM].frac = 14;
      } else {
        vf.attrs[GX_VA_NRM].frac = 0;
      }
      vf.attrs[GX_VA_CLR0].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 13));
      vf.attrs[GX_VA_CLR0].type = static_cast<GXCompType>(bp_get(value, 3, 14));
      vf.attrs[GX_VA_CLR0].frac = 0;
      vf.attrs[GX_VA_CLR1].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 17));
      vf.attrs[GX_VA_CLR1].type = static_cast<GXCompType>(bp_get(value, 3, 18));
      vf.attrs[GX_VA_CLR1].frac = 0;
      vf.attrs[GX_VA_TEX0].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 21));
      vf.attrs[GX_VA_TEX0].type = static_cast<GXCompType>(bp_get(value, 3, 22));
      vf.attrs[GX_VA_TEX0].frac = static_cast<u8>(bp_get(value, 5, 25));
      g_gxState.stateDirty = true;
      g_gxState.clearVtxSizeCache();
    }
    // VAT B registers (0x80-0x87)
    else if (addr >= 0x80 && addr <= 0x87) {
      u32 fmt = addr - 0x80;
      auto& vf = g_gxState.vtxFmts[fmt];
      vf.attrs[GX_VA_TEX1].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 0));
      vf.attrs[GX_VA_TEX1].type = static_cast<GXCompType>(bp_get(value, 3, 1));
      vf.attrs[GX_VA_TEX1].frac = static_cast<u8>(bp_get(value, 5, 4));
      vf.attrs[GX_VA_TEX2].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 9));
      vf.attrs[GX_VA_TEX2].type = static_cast<GXCompType>(bp_get(value, 3, 10));
      vf.attrs[GX_VA_TEX2].frac = static_cast<u8>(bp_get(value, 5, 13));
      vf.attrs[GX_VA_TEX3].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 18));
      vf.attrs[GX_VA_TEX3].type = static_cast<GXCompType>(bp_get(value, 3, 19));
      vf.attrs[GX_VA_TEX3].frac = static_cast<u8>(bp_get(value, 5, 22));
      vf.attrs[GX_VA_TEX4].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 27));
      vf.attrs[GX_VA_TEX4].type = static_cast<GXCompType>(bp_get(value, 3, 28));
      // TEX4 frac is in VAT C
      g_gxState.stateDirty = true;
      g_gxState.clearVtxSizeCache();
    }
    // VAT C registers (0x90-0x97)
    else if (addr >= 0x90 && addr <= 0x97) {
      u32 fmt = addr - 0x90;
      auto& vf = g_gxState.vtxFmts[fmt];
      vf.attrs[GX_VA_TEX4].frac = static_cast<u8>(bp_get(value, 5, 0));
      vf.attrs[GX_VA_TEX5].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 5));
      vf.attrs[GX_VA_TEX5].type = static_cast<GXCompType>(bp_get(value, 3, 6));
      vf.attrs[GX_VA_TEX5].frac = static_cast<u8>(bp_get(value, 5, 9));
      vf.attrs[GX_VA_TEX6].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 14));
      vf.attrs[GX_VA_TEX6].type = static_cast<GXCompType>(bp_get(value, 3, 15));
      vf.attrs[GX_VA_TEX6].frac = static_cast<u8>(bp_get(value, 5, 18));
      vf.attrs[GX_VA_TEX7].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 23));
      vf.attrs[GX_VA_TEX7].type = static_cast<GXCompType>(bp_get(value, 3, 24));
      vf.attrs[GX_VA_TEX7].frac = static_cast<u8>(bp_get(value, 5, 27));
      g_gxState.stateDirty = true;
      g_gxState.clearVtxSizeCache();
    }
    // Array base addresses (0xA0-0xAF): the raw CP write can only carry a
    // 32-bit truncated host pointer, so we cannot use `value` as a real base
    // address on a 64-bit host. However display-list replay routinely contains
    // these writes (baked by GDSetArray/J3D) and the CORRECT 64-bit pointer
    // for the same attr will be supplied separately via GX_AURORA_LOAD_ARRAYBASE
    // (from GXSetArray or J3DLoadArrayBasePtr under an Aurora build). Silently
    // ignore the CP write so the DL can replay cleanly; log at debug for the
    // curious. Callers who want strict rejection can rely on
    // AURORA_ARRAYBASE_REJECT_RAW=1.
    else if (addr >= 0xA0 && addr <= 0xAF) {
      static int s_reject = -1;
      if (s_reject < 0) {
        const char* env = std::getenv("AURORA_ARRAYBASE_REJECT_RAW");
        s_reject = env && env[0] && env[0] != '0' ? 1 : 0;
      }
      if (s_reject) {
        Log.error("CP_REG_ARRAYBASE_ID (addr=0x{:02X}, attr={}) rejected: raw 32-bit "
                  "pointer 0x{:08X}. Emit GX_AURORA_LOAD_ARRAYBASE for the full 64-bit host pointer.",
                  addr, addr - 0xA0, value);
      } else {
        Log.debug("CP_REG_ARRAYBASE_ID (addr=0x{:02X}, attr={}) ignored (raw=0x{:08X}); "
                  "expect GX_AURORA_LOAD_ARRAYBASE for this attr.",
                  addr, addr - 0xA0, value);
      }
    }
    // Array strides (0xB0-0xBF)
    else if (addr >= 0xB0 && addr <= 0xBF) {
      u32 attrIdx = addr - 0xB0 + GX_VA_POS;
      if (attrIdx < GX_VA_MAX_ATTR) {
        auto& array = g_gxState.arrays[attrIdx];
        const auto newStride = static_cast<u8>(value);
        if (array.stride != newStride) {
          array.stride = newStride;
          g_gxState.stateDirty = true;
        }
      }
    }
    break;
  }
}

// XF register handler - decodes XF (transform unit) register writes and updates g_gxState
static void handle_xf(const u8* data, u32& pos, u32 size, bool bigEndian) {
  CHECK(pos + 4 <= size, "XF header read overrun");
  u32 header = read_u32(data + pos, bigEndian);
  pos += 4;

  u32 count = ((header >> 16) & 0xFFFF) + 1;
  u32 addr = header & 0xFFFF;
  u32 dataBytes = count * 4;
  // Log.warn("  xf: addr {:04x} count {} dataBytes {} pos {} -> {}", addr, count, dataBytes, pos, pos + dataBytes);
  CHECK(pos + dataBytes <= size, "XF data read overrun: need {} bytes at pos {}", dataBytes, pos);

  const u8* xfData = data + pos;

  if (copy_xf_data(addr, xfData, count, bigEndian)) {
    // copy_xf_data handled everything.
  } else if (addr >= 0x1000) {
    // XF registers (0x1000+)
    u32 xfAddr = addr - 0x1000;
    for (u32 i = 0; i < count; i++) {
      u32 reg = xfAddr + i;
      u32 val = read_u32(xfData + i * 4, bigEndian);

      // Skip scalar register writes that haven't changed (viewport/projection handled below)
      if (reg <= 0x19 && val == g_gxState.xfRegCache[reg])
        continue;
      if (reg <= 0x19)
        g_gxState.xfRegCache[reg] = val;

      switch (reg) {
      case 0x08:
        // XF vertex specs (numColors, numNormals, numTexCoords) - informational
        break;
      case 0x09:
        // numChans
        g_gxState.numChans = val;
        g_gxState.stateDirty = true;
        break;
      case 0x0A:
        // Ambient color 0
        {
          static const bool s_ambTrace = std::getenv("SB_AMB_TRACE") != nullptr;
          if (s_ambTrace) {
            static long n = 0;
            std::fprintf(stderr, "[amb-trace] n=%ld reg=0A val=%08x mark='%s'\n",
                         ++n, val, g_sbLastMarker.c_str());
          }
        }
        g_gxState.colorChannelState[GX_COLOR0].ambColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA0].ambColor = unpack_color(val);
        g_gxState.stateDirty = true;
        break;
      case 0x0B:
        // Ambient color 1
        {
          static const bool s_ambTrace = std::getenv("SB_AMB_TRACE") != nullptr;
          if (s_ambTrace) {
            static long n = 0;
            std::fprintf(stderr, "[amb-trace] n=%ld reg=0B val=%08x mark='%s'\n",
                         ++n, val, g_sbLastMarker.c_str());
          }
        }
        g_gxState.colorChannelState[GX_COLOR1].ambColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA1].ambColor = unpack_color(val);
        g_gxState.stateDirty = true;
        break;
      case 0x0C:
        // Material color 0
        g_gxState.colorChannelState[GX_COLOR0].matColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA0].matColor = unpack_color(val);
        g_gxState.stateDirty = true;
        break;
      case 0x0D:
        // Material color 1
        g_gxState.colorChannelState[GX_COLOR1].matColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA1].matColor = unpack_color(val);
        g_gxState.stateDirty = true;
        break;
      case 0x0E:
      case 0x0F:
      case 0x10:
      case 0x11: {
        // Channel control registers
        u32 chanId = reg - 0x0E;
        if (chanId < MaxColorChannels) {
          auto& chan = g_gxState.colorChannelConfig[chanId];
          chan.matSrc = static_cast<GXColorSrc>(bp_get(val, 1, 0));
          chan.lightingEnabled = bp_get(val, 1, 1) != 0;
          u32 lightsLo = bp_get(val, 4, 2);
          chan.ambSrc = static_cast<GXColorSrc>(bp_get(val, 1, 6));
          chan.diffFn = static_cast<GXDiffuseFn>(bp_get(val, 2, 7));
          // Real GC/decomp XF chanctrl attn encoding (reference/sms
          // src/dolphin/gx/GXLight.c GXSetChanCtrl, cross-checked vs Dolphin
          // XFMemory.h attnfunc BitField<9,2> {None,Spec,Dir,Spot}):
          //   bit 9  = (attn_fn != GX_AF_NONE)
          //   bit 10 = (attn_fn != GX_AF_SPEC)
          // The prior decode had bit9/bit10 SWAPPED, so a SPEC channel
          // (bit9=1,bit10=0) was misread as NONE — which forces attn=diff=1 and
          // adds the light's FULL color as a constant instead of an attenuated
          // specular highlight. That over-brightened Mario at file-select (his
          // COLOR1 is a GX_AF_SPEC highlight light; TEV stage 4 pulls COLOR1
          // RASC in, and the phantom full-white add blew him toward white).
          bool bit9 = bp_get(val, 1, 9) != 0;
          bool bit10 = bp_get(val, 1, 10) != 0;
          u32 lightsHi = bp_get(val, 4, 11);
          if (!bit9) {
            chan.attnFn = GX_AF_NONE;  // GX_AF_NONE (or hw "Dir": attn=1, diffuse applies)
          } else if (!bit10) {
            chan.attnFn = GX_AF_SPEC;
          } else {
            chan.attnFn = GX_AF_SPOT;
          }
          u32 lightMask = lightsLo | (lightsHi << 4);
          g_gxState.colorChannelState[chanId].lightMask = GX::LightMask{lightMask};
          g_gxState.stateDirty = true;
        }
        break;
      }
      case 0x18: {
        // Matrix index A: PnMtx + TexCoord0-3 matrix indices
        g_gxState.currentPnMtx = bp_get(val, 6, 0) / 3;
        for (u32 i = 0; i < 4 && i < MaxTexCoord; i++) {
          auto texMtx = static_cast<GXTexMtx>(bp_get(val, 6, 6 + i * 6));
          assert(texMtx >= 0 && texMtx <= GXTexMtx::GX_IDENTITY);
          g_gxState.tcgs[i].mtx = texMtx;
        }
        g_gxState.stateDirty = true;
        break;
      }
      case 0x19: {
        // Matrix index B: TexCoord4-7 matrix indices
        for (u32 i = 0; i < 4 && (i + 4) < MaxTexCoord; i++) {
          g_gxState.tcgs[i + 4].mtx = static_cast<GXTexMtx>(bp_get(val, 6, i * 6));
        }
        g_gxState.stateDirty = true;
        break;
      }
      case 0x1A:
      case 0x1B:
      case 0x1C:
      case 0x1D:
      case 0x1E:
      case 0x1F: {
        // Viewport: sx, sy, sz, ox, oy, oz at XF 0x101A-0x101F
        u32 vpOff = reg - 0x1A;
        if (vpOff == 0 && count >= 6) {
          f32 sx = read_f32(xfData + 0, bigEndian);
          f32 sy = read_f32(xfData + 4, bigEndian);
          f32 sz = read_f32(xfData + 8, bigEndian);
          f32 ox = read_f32(xfData + 12, bigEndian);
          f32 oy = read_f32(xfData + 16, bigEndian);
          f32 oz = read_f32(xfData + 20, bigEndian);
          f32 width = sx * 2.0f;
          f32 height = -sy * 2.0f;
          // The hardware viewport-origin bias is 342, not 340: GXSetViewport encodes
          // ox = xOrig + width/2 + 342, so recovering xOrig must subtract the same 342.
          // With 340 every viewport reconstructed from the FIFO landed 2 pixels down and
          // right of where the game asked for it — measured as vp=(2,2 640x448) via the raw
          // FIFO path against vp=(0,0 640x448) for the identical frame through GXSetViewport,
          // which does not go through this reconstruction and so was never affected.
          constexpr f32 kViewportOriginBias = 342.0f;
          set_logical_viewport({
              .left = ox - kViewportOriginBias - width / 2.0f,
              .top = oy - kViewportOriginBias - height / 2.0f,
              .width = width,
              .height = height,
              .znear = (oz - sz) / 1.6777215e7f,
              .zfar = oz / 1.6777215e7f,
          });
        }
        break;
      }
      case 0x20:
      case 0x21:
      case 0x22:
      case 0x23:
      case 0x24:
      case 0x25:
      case 0x26: {
        // Projection: 6 params + type at XF 0x1020-0x1026
        u32 projOff = reg - 0x20;
        if (projOff == 0 && count >= 7) {
          f32 p0 = read_f32(xfData + 0, bigEndian);
          f32 p1 = read_f32(xfData + 4, bigEndian);
          f32 p2 = read_f32(xfData + 8, bigEndian);
          f32 p3 = read_f32(xfData + 12, bigEndian);
          f32 p4 = read_f32(xfData + 16, bigEndian);
          f32 p5 = read_f32(xfData + 20, bigEndian);
          u32 projType = read_u32(xfData + 24, bigEndian);
          g_gxState.projType = static_cast<GXProjectionType>(projType);
          {
            static const lucent::Channel chProjSet{"projset"};
            lucent::debug(chProjSet, "type={} p=({:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}) mark='{}'",
                          projType == GX_ORTHOGRAPHIC ? 'O' : 'P', p0, p1, p2, p3, p4, p5,
                          g_sbLastMarker);
          }
          // Reconstruct 4x4 projection matrix from 6 params
          auto& proj = g_gxState.proj;
          proj = {};
          proj.m0[0] = p0;
          proj.m1[1] = p2;
          proj.m2[2] = p4;
          proj.m2[3] = p5;
          if (projType == GX_ORTHOGRAPHIC) {
            proj.m0[3] = p1;
            proj.m1[3] = p3;
            proj.m3[3] = 1.0f;
          } else {
            proj.m0[2] = p1;
            proj.m1[2] = p3;
            proj.m3[2] = -1.0f;
          }
          g_gxState.stateDirty = true;
        }
        break;
      }
      case 0x3F:
        // numTexGens
        g_gxState.numTexGens = val;
        g_gxState.stateDirty = true;
        break;
      default:
        // TexGen config (0x40-0x4F) and post-transform (0x50-0x5F)
        if (reg >= 0x40 && reg <= 0x4F) {
          u32 tcIdx = reg - 0x40;
          if (tcIdx < MaxTexCoord) {
            auto& tcg = g_gxState.tcgs[tcIdx];
            ++g_gxState.tcgWrites[tcIdx];
            bool proj = bp_get(val, 1, 1) != 0;
            u32 form = bp_get(val, 1, 2);
            u32 tgType = bp_get(val, 3, 4);
            u32 srcRow = bp_get(val, 5, 7);

            if (tgType == 0) {
              tcg.type = proj ? GX_TG_MTX3x4 : GX_TG_MTX2x4;
            } else if (tgType == 1) {
              // Bump mapping: type encodes emboss light
              tcg.type = static_cast<GXTexGenType>(bp_get(val, 3, 15) + 2);
            } else if (tgType == 2 || tgType == 3) {
              tcg.type = GX_TG_SRTG;
            }
            // Emboss source texcoord (bits 12-14); 0 for non-bump types
            tcg.embossSrc = bp_get(val, 3, 12);

            // Decode source from row
            static const GXTexGenSrc rowToSrc[] = {GX_TG_POS,  GX_TG_NRM,  GX_TG_COLOR0, GX_TG_BINRM, GX_TG_TANGENT,
                                                   GX_TG_TEX0, GX_TG_TEX1, GX_TG_TEX2,   GX_TG_TEX3,  GX_TG_TEX4,
                                                   GX_TG_TEX5, GX_TG_TEX6, GX_TG_TEX7};
            if (srcRow < 13) {
              tcg.src = rowToSrc[srcRow];
            } else {
              // SILENTLY SKIPPED INPUT IS A FAILURE, NOT A FILTER. This branch used to do nothing,
              // which left tcg.src at whatever it was before (GX_MAX_TEXGENSRC on a fresh state) and
              // moved the failure three subsystems away, to shader generation's "unhandled tcg src
              // 21" — a message that names neither the register nor the writer. Say it here, where
              // the offending value is still in hand.
              Log.error("XF texgen {}: source row {} is out of range (val 0x{:08X}); tcg.src left at "
                        "{}. This write came from the guest stream, so either the game configured a "
                        "row this decode does not know or the FIFO parse mis-framed the write.",
                        tcIdx, srcRow, val, underlying(tcg.src));
            }
            g_gxState.stateDirty = true;
          }
        } else if (reg >= 0x50 && reg <= 0x5F) {
          u32 tcIdx = reg - 0x50;
          if (tcIdx < MaxTexCoord) {
            g_gxState.tcgs[tcIdx].postMtx = static_cast<GXPTTexMtx>(bp_get(val, 6, 0) + 64);
            g_gxState.tcgs[tcIdx].normalize = bp_get(val, 1, 8) != 0;
            g_gxState.stateDirty = true;
          }
        } else {
#ifndef NDEBUG
          Log.debug("Unhandled XF register 0x{:04X} (value 0x{:08X})", reg, val);
#endif
        }
        break;
      }
    }
  }

  pos += dataBytes;
}

static void handle_draw_overrun [[noreturn]] (u32 totalVtxBytes, const u8* data, const u32& pos, u32 size) {
  // Hex dump around the draw command for debugging
  u32 cmdPos = pos - 2 - 1; // opcode byte position (before vtxCount and pos++)
  u32 dumpStart = (cmdPos > 16) ? cmdPos - 16 : 0;
  u32 dumpEnd = (cmdPos + 32 < size) ? cmdPos + 32 : size;
  std::string hex;
  for (u32 i = dumpStart; i < dumpEnd; i++) {
    if (i == cmdPos)
      hex += fmt::format("[{:02x}]", data[i]);
    else
      hex += fmt::format(" {:02x}", data[i]);
  }
  Log.error("  hex dump around draw cmd (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
  FATAL("draw vertex data overrun: need {} bytes at pos {}, have {}", totalVtxBytes, pos, size);
}

// Denominators for the position-sourced texgen work. Every one is a DENOMINATOR: "0 texture
// matrices patched" means one thing when no draw was a candidate and the opposite when many were
// and the gate rejected them all, and the numerator alone cannot tell those apart.
long g_texgenPosSourced = 0;    // draws with a position-sourced matrix texgen (the candidate set)
long g_texgenRejectIndexed = 0; // ... rejected here: no single matrix to rewrite, or slot unknown
long g_texgenEyeSpace = 0;      // ... passed to interpolation, which applies its own stability gate

// Which texture matrices of this draw are driven by the VERTEX POSITION, and so have to move when
// the position matrices do. Bit k = the shader's `postex_mtx[k]`, which is why the index expression
// below is copied from shader.cpp rather than re-derived.
//
// WHY THIS EXISTS. A GX texgen sourced from GX_TG_POS reads the RAW vertex attribute, not the
// position after the position matrix. Interpolation moves the POSITION matrices — for a paired draw
// to its own in-between pose, for an unpaired one to the in-between viewpoint — and has never
// touched the TEXTURE matrices. Where the texture matrix is a projection through the same camera,
// the geometry then advances half a tick while its UVs stay on the previous tick's mapping, so the
// projected image slides across the surface it is painted on. That happens ONLY while the camera or
// the object is moving, which is exactly how it was reported.
//
// This function decides only "is the position what feeds this UV, and is there a single matrix to
// rewrite". It deliberately does NOT decide whether the texture matrix contains the camera — that
// cannot be read off one frame's state, and guessing it wrong corrupts object-locked projections
// (a decal's UVs are correct unchanged when the camera moves). Interpolation settles it by
// measuring, across ticks, whether the matrix behaves like a projection composed with this draw's
// model-view; see the stability gate in gfx/interp.cpp.
static uint32_t eye_space_texgen_mask(const gx::ShaderInfo& info) {
  uint32_t mask = 0;
  for (u8 i = 0; i < g_gxState.numTexGens && i < MaxTexCoord; ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue; // the shader never emits this texcoord
    }
    const auto& tcg = g_gxState.tcgs[i];
    if (tcg.src != GX_TG_POS || (tcg.type != GX_TG_MTX2x4 && tcg.type != GX_TG_MTX3x4)) {
      continue;
    }
    ++g_texgenPosSourced;
    // A per-vertex texture-matrix index means there is no single matrix in the uniform block that
    // this draw uses, and an identity texgen matrix means there is no matrix at all. A per-vertex
    // POSITION-matrix index means the model-view differs per vertex, so the correction below has no
    // single matrix to express itself against either.
    if (info.indexAttr.test(GX_VA_TEX0MTXIDX + i) || tcg.mtx == GX_IDENTITY ||
        info.indexAttr.test(GX_VA_PNMTXIDX)) {
      ++g_texgenRejectIndexed;
      continue;
    }
    ++g_texgenEyeSpace;
    // The SAME index expression the shader uses (`ubuf.postex_mtx[tcg.mtx / 3]`, shader.cpp), not a
    // re-derivation of it. GX_TEXMTX0 is 30, so this lands at MaxPnMtx and above — the texture half
    // of the shared postex_mtx array. Written this way so the two cannot drift apart.
    const u32 idx = tcg.mtx / 3;
    ASSERT(idx >= MaxPnMtx && idx < MaxPnMtx + MaxTexMtx,
           "texgen {} matrix {} maps to postex_mtx[{}], outside the texture block [{}, {})", i,
           underlying(tcg.mtx), idx, MaxPnMtx, MaxPnMtx + MaxTexMtx);
    mask |= 1u << idx;
  }
  return mask;
}

// Draw command handler - parses vertices inline and caches results
// Byte offset of GX_VA_POS within a vertex record, and whether it is the one shape the vertex lerp
// can handle: DIRECT, three components, f32. GX's attribute order is fixed (PNMTXIDX, the eight
// TEXnMTXIDX, POS, ...), so the offset is the sum of what precedes POS under the current descriptor.
static void calculate_pos_layout(GXVtxFmt fmt, u16& posOffset, u8& posF32XYZ) {
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  u32 off = 0;
  for (int i = GX_VA_PNMTXIDX; i < GX_VA_POS; ++i) {
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE: break;
    case GX_DIRECT: {
      const auto attr = static_cast<GXAttr>(i);
      off += comp_type_size(attr, vtxFmt.attrs[i].type) * comp_cnt_count(attr, vtxFmt.attrs[i].cnt);
      break;
    }
    case GX_INDEX8: off += 1; break;
    case GX_INDEX16: off += 2; break;
    }
  }
  posOffset = static_cast<u16>(off);
  const auto& pf = vtxFmt.attrs[GX_VA_POS];
  posF32XYZ = (g_gxState.vtxDesc[GX_VA_POS] == GX_DIRECT && pf.type == GX_F32 &&
               pf.cnt == GX_POS_XYZ)
                  ? 1u
                  : 0u;
}

static u32 calculate_last_vtx_size(GXVtxFmt fmt) {
  u32 vtxSize = 0;
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE:
      break;
    case GX_DIRECT: {
      const auto attr = static_cast<GXAttr>(i);
      const auto& attrFmt = vtxFmt.attrs[i];
      vtxSize += comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
      break;
    }
    case GX_INDEX8:
      vtxSize += (i == GX_VA_NRM && vtxFmt.attrs[i].cnt == GX_NRM_NBT3) ? 3 : 1;
      break;
    case GX_INDEX16:
      vtxSize += (i == GX_VA_NRM && vtxFmt.attrs[i].cnt == GX_NRM_NBT3) ? 6 : 2;
      break;
    }
  }

  g_gxState.lastVtxFmt = fmt;
  g_gxState.lastVtxSize = vtxSize;

  return vtxSize;
}

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange);
static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices);

// Draw command handler - parses vertices inline and caches results
static ByteBuffer handle_draw_idx_buf;

// Global post-merge draw counter: incremented once per push_gx_draw call —
// the exact indexing scheme SB_DRAW_DUMP's [draw-dump] #N lines use. Read by
// draw_prim's SB_NDC_DRAW window so a [draw-dump] index can be probed at the
// vertex level (a prim that MERGES into the previous draw executes while the
// counter already points one past its draw, hence the window's +1 slack).
long g_skippedBigQuads = 0;
long g_sbPushedDrawCount = 0; // exported: SB_NO_ZWRITE_DRAWS window check in gx.cpp

// The identity currently in force, set by GX_AURORA_DRAW_TAG and stamped onto every DrawData until
// the next tag. 0 = untagged.
//
// Cleared at the start of each frame so a tag can never leak across a frame boundary: an emitter
// that stops tagging would otherwise keep stamping the last object's identity onto every remaining
// draw, and those draws would then pair with the wrong object's matrices — a silent, plausible
// wrong answer rather than a visible failure.
uint64_t g_pendingDrawTag = 0;
uint8_t g_pendingDrawPop = 0;
uint8_t g_pendingDrawExact = 0;
// Coverage, so "tagging is on" can be distinguished from "tagging is silently doing nothing".
long g_taggedDrawCount = 0;
long g_untaggedDrawCount = 0;
// Untagged draws split by projection. An untagged draw SNAPS instead of interpolating, and whether
// that is correct depends entirely on what it is: for 2D/HUD it is right (there is no meaningful
// in-between for a screen-space element), for world geometry it is a defect that will read as
// stutter in an otherwise smooth frame. A single "39% untagged" number cannot tell those apart, and
// a plausible-looking percentage is exactly the kind of thing that gets accepted without being
// checked. This splits it.
long g_untaggedOrthoDrawCount = 0;
long g_untaggedPerspDrawCount = 0;
// And the untagged PERSPECTIVE draws split again by how their positions are supplied. This is the
// discriminator that says whether the remainder is a defect or not:
//
//   DIRECT positions  = immediate-mode geometry, built fresh by the CPU every tick (particles, the
//                       sea's ripple grid, immediate effects). It has no cross-tick identity to
//                       have, so snapping is the correct and only behaviour.
//   INDEXED positions = geometry drawn from a persistent vertex array through a display list, i.e.
//                       exactly the kind of thing that DOES have a stable identity and should be
//                       interpolating. Every one of these is a tag seam we have not covered.
long g_untaggedPerspDirectCount = 0;
long g_untaggedPerspIndexedCount = 0;

// SB_PROFILE_DRAWPRIM=1 accounting. Reported per drain by the caller.
static bool sb_drawprim_profile() {
  static const bool on = [] {
    const char* e = std::getenv("SB_PROFILE_DRAWPRIM");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}
static int64_t sb_now_ns() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

// A free-running cycle counter, for probing INSIDE draw_prim.
//
// clock_gettime is ~20-25 ns even through the vDSO. draw_prim's whole body is ~340 ns, so the
// two clock_gettime calls the original profiler used were already ~12% of what they measured,
// and the two EXTRA ones around the max-index scan were charged to draw_prim's own total — the
// reported "scan = 14% of draw_prim" was therefore partly a measurement of its own probes.
// Splitting the body into seven phases with that probe would have cost more than the body.
//
// No x86 asm (this port targets x86-64 AND arm64); each arch's counter intrinsic, and a
// clock_gettime fallback elsewhere that the report flags as too coarse to trust.
#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
static inline uint64_t sb_tsc() { return __rdtsc(); }
static constexpr bool kSbTscIsCycleCounter = true;
#elif defined(__aarch64__)
static inline uint64_t sb_tsc() {
  uint64_t v;
  asm volatile("mrs %0, cntvct_el0" : "=r"(v));
  return v;
}
static constexpr bool kSbTscIsCycleCounter = true;
#else
static inline uint64_t sb_tsc() { return static_cast<uint64_t>(sb_now_ns()); }
static constexpr bool kSbTscIsCycleCounter = false;
#endif

// ns per tick, measured against CLOCK_MONOTONIC on first use.
static double sb_tsc_ns_per_tick() {
  static const double v = [] {
    if (!kSbTscIsCycleCounter) {
      return 1.0; // fallback counter already IS nanoseconds
    }
    const int64_t n0 = sb_now_ns();
    const uint64_t t0 = sb_tsc();
    // Busy-wait rather than sleep: a sleeping thread can be migrated, and this is a handful
    // of milliseconds once per process.
    while (sb_now_ns() - n0 < 20000000LL) {
    }
    const uint64_t t1 = sb_tsc();
    const int64_t n1 = sb_now_ns();
    const uint64_t dt = t1 - t0;
    return dt != 0 ? static_cast<double>(n1 - n0) / static_cast<double>(dt) : 1.0;
  }();
  return v;
}

// Per-phase accounting. The phases PARTITION draw_prim's body, so that a region without a probe
// shows up as a non-zero "unattributed" in the report rather than being silently folded into a
// neighbour.
enum SbDpPhase {
  SB_DP_PROLOGUE = 0, // vtx size + overrun check
  SB_DP_DIAG_PRE,     // the gated diagnostic blocks before the indexed-array bookkeeping
  SB_DP_ATTRENUM,     // enumerate indexed attributes (26 iterations, runs on EVERY call)
  SB_DP_IDXSCAN,      // per-vertex max-referenced-index scan (only when nFields > 0)
  SB_DP_DIAG_POST,    // the gated diagnostic blocks after it
  SB_DP_PUSHVERTS,    // gfx::push_verts (vertex bytes -> frame packet)
  SB_DP_MERGEIDX,     // prepare_idx_buffer + push_indices + merge bookkeeping
  SB_DP_UNMERGED,     // handle_draw_unmerged
  SB_DP_NPHASES,
};
// How many calls actually EXECUTED each phase. Dividing a phase by the total call count is what
// made handle_draw_unmerged read as "293 ns/call" when it runs 1,289 times out of 45,912 and
// really costs ~10 us per call — an average over calls that never entered it is not a per-call
// cost, and here the two differ by 35x.
long g_dpPhaseCalls[SB_DP_NPHASES] = {};

// Per-call samples for the unmerged phase. A MEAN of 5.7 us over 1,312 calls has two completely
// different explanations with different fixes: every draw costing ~5.7 us (a hot build path worth
// restructuring), or a few dozen shader/pipeline compiles at ~1 ms each dragging up an otherwise
// cheap path (a warm-up cost that vanishes after the first frames and must NOT be optimised for).
// Percentiles separate them; a mean cannot.
uint32_t g_dpUnmergedSamples[4096] = {};
long g_dpUnmergedSampleCount = 0;
long g_dpUnmergedSampleDropped = 0;
uint64_t g_dpPhase[SB_DP_NPHASES] = {};
// Indexed-array storage uploads, per frame. See the use site: total-vs-distinct is what separates
// "uploads a lot of distinct geometry" from "uploads the same array over and over".
uint64_t g_arrUploadCount = 0, g_arrUploadBytes = 0, g_arrUploadDistinctBytes = 0, g_arrCachedHits = 0;
std::unordered_set<uint64_t> g_arrUploadDistinct;
std::unordered_map<uint64_t, uint64_t> g_arrUploadHash;
uint64_t g_arrContentChanged = 0, g_arrDataCacheHits = 0;
// CROSS-FRAME stability. The frame packet's storage buffer is rewound every frame, so every array
// is re-uploaded every frame even when its bytes never change. If most of the 20.4 MB is identical
// frame to frame, a persistent GPU buffer for static geometry would remove nearly all of it — a
// large change, worth sizing before proposing. This measures the ceiling on that win.
std::unordered_map<uint64_t, uint64_t> g_arrHashPrevFrame;
uint64_t g_arrSameAsPrevBytes = 0, g_arrChangedVsPrevBytes = 0, g_arrNewVsPrevBytes = 0;
// Does a given array land at the SAME storage offset every frame? If it does, the single
// persistent g_storageBuffer already holds its bytes from last frame and the memcpy into staging
// is pure waste — which makes skipping it a small local change rather than a persistent allocator.
// Offsets come from append order, so this is a property of scene-walk determinism: measure it.
std::unordered_map<uint64_t, uint32_t> g_arrOffsetPrevFrame, g_arrOffsetThisFrame;
uint64_t g_arrOffsetStable = 0, g_arrOffsetMoved = 0;
uint64_t g_arrPersistUploads = 0, g_arrPersistHits = 0, g_arrArenaFull = 0;
// Thin accessors so fifo.cpp's report does not need the gfx headers.
uint64_t sb_arena_used() { return gfx::persistent_storage_used(); }
size_t sb_arena_entries() { return gfx::persistent_storage_entries(); }
uint64_t g_arrPersistUploadBytes = 0, g_arrPersistHitBytes = 0;
long g_dpMergedCalls = 0, g_dpUnmergedCalls = 0, g_dpEarlyReturns = 0;
uint64_t g_dpWholeTicks = 0;
// Control: two back-to-back reads with nothing between them. This is what ONE probe costs, and
// it bounds how much of the split is the instrument measuring itself. If it is not small next
// to the phases, the phase split is inadmissible and the report says so.
static uint64_t sb_dp_probe_cost_ticks() {
  static const uint64_t v = [] {
    uint64_t best = ~0ull;
    for (int i = 0; i < 4096; ++i) {
      const uint64_t a = sb_tsc();
      const uint64_t b = sb_tsc();
      if (b - a < best) {
        best = b - a;
      }
    }
    return best;
  }();
  return v;
}
// Accessors for the reporter in fifo.cpp (the calibration and the probe-cost control are lazily
// initialised statics local to this TU).
double sb_dp_ns_per_tick_pub() { return sb_tsc_ns_per_tick(); }
uint64_t sb_dp_probe_cost_ticks_pub() { return sb_dp_probe_cost_ticks(); }
// 8 = the 7 DP_PHASE probes + the one dpWhole takes on exit. Both are charged inside the measured
// body, so both belong in the overhead control.
int sb_dp_probes_per_call_pub() { return SB_DP_NPHASES + 1; }

int64_t g_dpTotalNs = 0;
long g_dpCalls = 0;
// Primitive SIZE distribution, because "46k primitives for 1314 merged draws" has two very
// different explanations and the fix differs: a game emitting genuinely tiny primitives needs a
// cheaper per-primitive path, while large primitives arriving unmerged would mean batching is
// leaving work on the table. Bucketed by vertex count.
long g_dpVerts[8] = {};   // 1-2, 3, 4, 5-6, 7-12, 13-24, 25-48, 49+
long g_dpVertTotal = 0;
long g_dpPrimKind[8] = {};   // by GXPrimitive, indexed (prim >> 4) & 7

static void draw_prim(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* data, u32& pos, u32 size) {
  const bool dpProf = sb_drawprim_profile();
  const int64_t dpT0 = dpProf ? sb_now_ns() : 0;
  if (dpProf) {
    const u32 vc = vtxCount;
    int b = vc <= 2 ? 0 : vc == 3 ? 1 : vc == 4 ? 2 : vc <= 6 ? 3 : vc <= 12 ? 4 : vc <= 24 ? 5 : vc <= 48 ? 6 : 7;
    ++g_dpVerts[b];
    g_dpVertTotal += vc;
    ++g_dpPrimKind[(static_cast<u32>(prim) >> 4) & 7u];
  }
  struct DpScope {
    bool on; int64_t t0;
    ~DpScope() { if (on) { g_dpTotalNs += sb_now_ns() - t0; ++g_dpCalls; } }
  } dpScope{dpProf, dpT0};

  // Phase probes. `dpTick` is the timestamp of the previous probe; DP_PHASE(p) closes the region
  // that ends here and charges it to p. Every exit path is covered by dpWhole's destructor, which
  // also computes what NO phase claimed.
  uint64_t dpTick = dpProf ? sb_tsc() : 0;
  struct DpWhole {
    bool on; uint64_t t0;
    ~DpWhole() { if (on) { g_dpWholeTicks += sb_tsc() - t0; } }
  } dpWhole{dpProf, dpTick};
#define DP_PHASE(p)                                                                                \
  do {                                                                                             \
    if (dpProf) {                                                                                  \
      const uint64_t _t = sb_tsc();                                                                \
      g_dpPhase[(p)] += _t - dpTick;                                                               \
      ++g_dpPhaseCalls[(p)];                                                                       \
      dpTick = _t;                                                                                 \
    }                                                                                              \
  } while (0)

  ZoneScoped;
  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calculate_last_vtx_size(fmt); }

  u32 totalVtxBytes = vtxCount * vtxSize;
  if (pos + totalVtxBytes > size)
    UNLIKELY { handle_draw_overrun(totalVtxBytes, data, pos, size); }
  DP_PHASE(SB_DP_PROLOGUE);

  // SB_POS_PROBE=1: for INDEX16 + F32 position draws, decode the first
  // vertex's position index and fetch the XYZ it references — shows whether
  // the GPU-visible position data is sane or garbage.
  {
    static int s_probe = -1;
    if (s_probe < 0) {
      const char* e = std::getenv("SB_POS_PROBE");
      s_probe = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    if (s_probe == 1) {
      const auto& posFmt = g_gxState.vtxFmts[fmt].attrs[GX_VA_POS];
      if (g_gxState.vtxDesc[GX_VA_POS] == GX_INDEX16 && posFmt.type == GX_F32 && vtxCount > 0) {
        static int n = 0;
        if (n < 40) {
          ++n;
          // Offset of the POS index within a vertex: sum of preceding attr sizes
          // (matrix-index attrs are u8 DIRECT).
          u32 off = 0;
          for (int a = GX_VA_PNMTXIDX; a < GX_VA_POS; ++a)
            if (g_gxState.vtxDesc[a] == GX_DIRECT) off += 1;
          const u16 idx = read_u16(data + pos + off, true);
          const auto& arr = g_gxState.arrays[GX_VA_POS];
          const float* p = arr.data != nullptr
                               ? reinterpret_cast<const float*>(static_cast<const u8*>(arr.data) + idx * arr.stride)
                               : nullptr;
          std::fprintf(stderr,
                       "[pos-probe] n=%d verts=%u idx=%u stride=%u arr=%p le=%d xyz=(%g, %g, %g)\n", n,
                       vtxCount, idx, arr.stride, arr.data, static_cast<int>(arr.le),
                       p ? p[0] : 0.f, p ? p[1] : 0.f, p ? p[2] : 0.f);
        }
      }
    }
  }

  // SB_QUAD_RECT=1 (diagnostic): for ORTHOGRAPHIC 4-vertex draws, decode the quad's corners
  // from the vertex stream and report its screen rectangle. When a 2D overlay is covering the
  // scene, its EXTENT is what identifies it — texture and blend state are shared by dozens of
  // unrelated UI quads, but only one of them spans the washed band.
  {
    // SB_QUAD_RECT=<frame>: start logging only once that frame ordinal is reached. A plain
    // count-capped log fills with boot-logo quads and never reaches the scene under
    // investigation — that has now happened often enough to be worth gating properly.
    static int s_init = 0;
    static long s_after = -1;
    if (!s_init) {
      s_init = 1;
      const char* e = std::getenv("SB_QUAD_RECT");
      s_after = (e != nullptr && e[0] != '\0') ? std::atol(e) : -1;
    }
    // Both projections: an orthographic sweep found nothing larger than a text glyph, so a
    // PERSPECTIVE quad is the remaining shape. Identical state and multiplicity between two
    // runtimes says nothing about where a quad actually lands.
    if (s_after >= 0 && static_cast<long>(sb_gx_vi_retrace_count()) >= s_after &&
        vtxCount == 4) {
      const auto& pf = g_gxState.vtxFmts[fmt].attrs[GX_VA_POS];
      const auto pd = g_gxState.vtxDesc[GX_VA_POS];
      // Decode the quad's corners whatever format the positions arrive in. Handling only
      // DIRECT F32 silently skipped 85 of 123 four-vertex draws here (43 orthographic S16 and
      // 42 indexed perspective ones), which is how a sweep "found no large quad" while looking
      // at a third of them.
      const bool indexed = (pd == GX_INDEX8 || pd == GX_INDEX16);
      const bool decodable = (pd == GX_DIRECT || indexed) &&
                             (pf.type == GX_F32 || pf.type == GX_S16 || pf.type == GX_U16 ||
                              pf.type == GX_S8 || pf.type == GX_U8);
      if (decodable) {
        const unsigned comps = pf.cnt == GX_POS_XYZ ? 3 : 2;
        const float scale = 1.0f / (float)(1u << pf.frac);
        const auto& arr = g_gxState.arrays[GX_VA_POS];

        // Byte offset of POS within a vertex: preceding attributes, sized by their own
        // descriptors (matrix indices are one byte each; indexed attrs are their index width).
        u32 off = 0;
        for (int a = GX_VA_PNMTXIDX; a < GX_VA_POS; ++a) {
          const auto d = g_gxState.vtxDesc[a];
          if (d == GX_NONE) continue;
          off += (d == GX_INDEX16) ? 2 : 1;
        }

        auto component = [&](const u8* base, unsigned i) -> float {
          switch (pf.type) {
          case GX_F32: return read_f32(base + i * 4, true);
          case GX_S16: return (float)(s16)read_u16(base + i * 2, true) * scale;
          case GX_U16: return (float)read_u16(base + i * 2, true) * scale;
          case GX_S8:  return (float)(s8)base[i] * scale;
          default:     return (float)base[i] * scale;
          }
        };

        float xs[4], ys[4];
        bool ok = true;
        for (unsigned v = 0; v < 4 && ok; ++v) {
          const u32 vsz = g_gxState.lastVtxFmt == fmt ? g_gxState.lastVtxSize
                                                      : calculate_last_vtx_size(fmt);
          const u32 base = pos + v * vsz + off;
          const u8* src = nullptr;
          if (indexed) {
            const u32 need = (pd == GX_INDEX16) ? 2u : 1u;
            if (base + need > size || arr.data == nullptr) { ok = false; break; }
            const u32 idx = (pd == GX_INDEX16) ? read_u16(data + base, true) : data[base];
            src = static_cast<const u8*>(arr.data) + (size_t)idx * arr.stride;
          } else {
            const u32 width = (pf.type == GX_F32) ? 4u : (pf.type == GX_S16 || pf.type == GX_U16) ? 2u : 1u;
            if (base + comps * width > size) { ok = false; break; }
            src = data + base;
          }
          xs[v] = component(src, 0);
          ys[v] = component(src, 1);
        }
        if (ok) {
          float x0 = xs[0], x1 = xs[0], y0 = ys[0], y1 = ys[0];
          for (unsigned v = 1; v < 4; ++v) {
            x0 = xs[v] < x0 ? xs[v] : x0; x1 = xs[v] > x1 ? xs[v] : x1;
            y0 = ys[v] < y0 ? ys[v] : y0; y1 = ys[v] > y1 ? ys[v] : y1;
          }
          // SB_SKIP_BIGQUAD=<extent>: drop 4-vertex quads whose decoded extent exceeds this,
          // i.e. the scene-covering ones, without touching the hundreds of small UI quads that
          // share their vertex count. Targeted where SB_SKIP_VERTS=4 is not.
          {
            static int s_bqInit = 0;
            static float s_bq = -1.f;
            if (!s_bqInit) {
              s_bqInit = 1;
              const char* e = std::getenv("SB_SKIP_BIGQUAD");
              s_bq = (e != nullptr && e[0] != '\0') ? (float)std::atof(e) : -1.f;
            }
            if (s_bq > 0.f && (x1 - x0) >= s_bq && (y1 - y0) >= s_bq) {
              g_skippedBigQuads++;
              return;
            }
          }
          const auto& t0 = g_gxState.textures[0].texObj;
          static long n = 0;
          if (++n <= 200)
            std::fprintf(stderr,
                         "[quad-rect] #%ld %c x=[%.0f..%.0f] y=[%.0f..%.0f] w=%.0f h=%.0f "
                         "tex0=%ux%u tev=%u bm=%d bf=%d/%d cU=%d aU=%d\n",
                         n, g_gxState.projType == GX_ORTHOGRAPHIC ? 'O' : 'P',
                         x0, x1, y0, y1, x1 - x0, y1 - y0, t0.width(), t0.height(),
                         g_gxState.numTevStages, (int)g_gxState.blendMode,
                         (int)g_gxState.blendFacSrc, (int)g_gxState.blendFacDst,
                         g_gxState.colorUpdate ? 1 : 0, g_gxState.alphaUpdate ? 1 : 0);
        }
      }
    }
  }

  // Set by the pixel-watch block below when THIS draw's screen box contains the watch point.
  // SB_SKIP_COVERING then drops exactly those draws, so identification and causality come from
  // ONE instrument in ONE run — no comparing draw indices between two different counters.
  bool sb_covers_watch = false;
  // A prim with EVERY vertex behind the eye is clipped away by the GPU and is visually inert;
  // a prim with only SOME vertices behind rasterizes as a smear far outside the box of the ones
  // in front. Conflating the two made the eye-crossing skip over-broad — the oracle's crossing
  // prims are almost entirely the harmless full kind.
  bool sb_wneg_partial = false;
  bool sb_wneg_full = false;

  // SB_PIXEL_WATCH=<x>,<y>[,<frame>] — THE ATTRIBUTION HARNESS.
  //
  // Answers "which draws cover this pixel, in order, with what state" directly, instead of
  // bisecting by skipping draw groups and re-running (2.5 minutes per guess). Every draw's
  // vertices are transformed through the SAME matrices the GPU uses — per-vertex PNMTXIDX,
  // the position matrix, then the projection — divided by w, and mapped to screen pixels via
  // the logical viewport. A draw whose screen-space bounding box contains the watch point is
  // reported with the state that decides what it writes there.
  //
  // Coordinates are logical framebuffer pixels (the 640x448 the game draws in), origin top-left.
  {
    static int s_init = 0;
    static float s_wx = -1.f, s_wy = -1.f;
    static long s_wframe = -1;
    if (!s_init) {
      s_init = 1;
      if (const char* e = std::getenv("SB_PIXEL_WATCH"); e != nullptr && e[0] != '\0') {
        s_wx = (float)std::atof(e);
        const char* c1 = std::strchr(e, ',');
        if (c1 != nullptr) {
          s_wy = (float)std::atof(c1 + 1);
          const char* c2 = std::strchr(c1 + 1, ',');
          if (c2 != nullptr) s_wframe = std::atol(c2 + 1);
        }
      }
    }
    if (s_wx >= 0.f && s_wy >= 0.f) {
      const auto& pf = g_gxState.vtxFmts[fmt].attrs[GX_VA_POS];
      const auto pd2 = g_gxState.vtxDesc[GX_VA_POS];
      const auto& arr2 = g_gxState.arrays[GX_VA_POS];
      const bool idxed = (pd2 == GX_INDEX8 || pd2 == GX_INDEX16);
      const bool ok_fmt = (pf.type == GX_F32 || pf.type == GX_S16 || pf.type == GX_U16);
      const bool decodable = ok_fmt && (!idxed || arr2.data != nullptr);

      // A probe that silently ignores what it cannot parse reports a confident NOTHING. This
      // one accounts for every draw it saw: covered, not covered, or NOT DECODED — and prints
      // the undecoded tally with the formats responsible, so a null result can be trusted or
      // distrusted on the evidence rather than on faith.
      {
        static long s_seen = 0, s_undec = 0;
        static u32 s_badDesc = 0, s_badType = 0;
        ++s_seen;
        if (!decodable) {
          ++s_undec;
          s_badDesc = (u32)pd2;
          s_badType = (u32)pf.type;
        }
        // Report periodically rather than on a frame boundary: the watch is gated to ONE
        // frame, so a frame-change trigger inside that gate can never fire — the first
        // version of this accounting printed nothing at all, which is exactly the silence it
        // exists to prevent.
        if ((s_seen % 50) == 0)
          std::fprintf(stderr,
                       "[pixel-watch] coverage so far: %ld draws examined, %ld NOT DECODED "
                       "(last undecodable posDesc=%u posType=%u)\n",
                       s_seen, s_undec, s_badDesc, s_badType);
      }

      if (decodable) {
        // Offset of POS and of the matrix index within a vertex.
        u32 posOff2 = 0; int pnOff2 = -1;
        for (int a = GX_VA_PNMTXIDX; a < GX_VA_POS; ++a) {
          const auto d = g_gxState.vtxDesc[a];
          if (d == GX_NONE) continue;
          if (a == GX_VA_PNMTXIDX) pnOff2 = (int)posOff2;
          posOff2 += (d == GX_INDEX16) ? 2 : 1;
        }
        const float invFrac2 = 1.0f / (float)(1u << pf.frac);
        const float* P2 = reinterpret_cast<const float*>(&g_gxState.proj);
        const u32 vsz2 = g_gxState.lastVtxFmt == fmt ? g_gxState.lastVtxSize
                                                     : calculate_last_vtx_size(fmt);
        const auto& vp2 = g_gxState.logicalViewport;

        float sx0 = 1e30f, sx1 = -1e30f, sy0 = 1e30f, sy1 = -1e30f;
        bool any = false;
        // Clip-space positions, kept so coverage can be answered by CLIPPING the primitive the
        // way the GPU does, instead of by a bounding box of whichever vertices happened to be in
        // front. A box over the in-front vertices is not a coverage test: for a primitive that
        // straddles the eye plane the clipped polygon extends far outside it, so the box says
        // "does not cover" about a prim that rasterizes straight across the point.
        std::vector<std::array<float, 4>> clips;
        clips.reserve(vtxCount);
        u32 firstMtxIdx = 0xFFFF;
        // Vertices behind the eye plane. Dropping them silently was a false-negative mode:
        // with some vertices behind, the clipped polygon smears far outside the bounding box
        // of the ones in front, so the box says "does not cover" about a draw that does.
        u32 wneg = 0;
        for (u32 v = 0; v < vtxCount; ++v) {
          const u32 base = pos + v * vsz2;
          if (base + vsz2 > size) break;
          const u8* vp = data + base;
          u32 mtxIdx = g_gxState.currentPnMtx;
          if (pnOff2 >= 0) mtxIdx = vp[pnOff2] / 3u;
          if (firstMtxIdx == 0xFFFF) firstMtxIdx = mtxIdx;
          const u8* src;
          if (idxed) {
            const u32 i2 = (pd2 == GX_INDEX16) ? read_u16(vp + posOff2, true) : vp[posOff2];
            src = static_cast<const u8*>(arr2.data) + (size_t)i2 * arr2.stride;
          } else {
            src = vp + posOff2;
          }
          // Vertex-stream data is always big-endian (it came off the FIFO), but ARRAY data
          // carries its own endianness — a little-endian array read as big-endian yields
          // garbage positions and therefore garbage coverage answers, silently. This matters
          // the moment the harness is pointed at a runtime whose arrays are host-endian.
          const bool be = !idxed || !arr2.le;
          float x, y, z;
          if (pf.type == GX_F32) {
            x = read_f32(src, be); y = read_f32(src + 4, be);
            z = pf.cnt == GX_POS_XYZ ? read_f32(src + 8, be) : 0.f;
          } else {
            auto rs = [&](const u8* q) { return (float)(s16)read_u16(q, be) * invFrac2; };
            x = rs(src); y = rs(src + 2); z = pf.cnt == GX_POS_XYZ ? rs(src + 4) : 0.f;
          }
          const float* M = reinterpret_cast<const float*>(&g_gxState.pnMtx[mtxIdx % MaxPnMtx].pos);
          float mv[3];
          for (int c = 0; c < 3; ++c)
            mv[c] = M[4 * c] * x + M[4 * c + 1] * y + M[4 * c + 2] * z + M[4 * c + 3];
          float clip[4];
          for (int c = 0; c < 4; ++c)
            clip[c] = P2[4 * c] * mv[0] + P2[4 * c + 1] * mv[1] + P2[4 * c + 2] * mv[2] + P2[4 * c + 3];
          clips.push_back({clip[0], clip[1], clip[2], clip[3]});
          if (clip[3] <= 0.f) {
            ++wneg;
            continue;
          }
          const float nx = clip[0] / clip[3], ny = clip[1] / clip[3];
          // NDC -> screen pixels, origin top-left.
          const float px = vp2.left + (nx * 0.5f + 0.5f) * vp2.width;
          const float py = vp2.top + (0.5f - ny * 0.5f) * vp2.height;
          sx0 = std::min(sx0, px); sx1 = std::max(sx1, px);
          sy0 = std::min(sy0, py); sy1 = std::max(sy1, py);
          any = true;
        }

        // TRUE COVERAGE: triangulate the primitive, clip each triangle against the near plane
        // (w >= eps) in homogeneous space, and test the watch point against the resulting
        // polygon. This answers "does this draw rasterize this pixel" exactly, for straddling
        // primitives as well as wholly-visible ones — which a bounding box cannot do, and which
        // is the difference between attributing a pixel and merely correlating with one.
        bool hits = false;
        if (clips.size() >= 3) {
          const float kEps = 1e-5f;
          auto scr = [&](const std::array<float, 4>& c, float& px, float& py) {
            const float nx = c[0] / c[3], ny = c[1] / c[3];
            px = vp2.left + (nx * 0.5f + 0.5f) * vp2.width;
            py = vp2.top + (0.5f - ny * 0.5f) * vp2.height;
          };
          auto lerp4 = [](const std::array<float, 4>& a, const std::array<float, 4>& b, float t) {
            std::array<float, 4> r{};
            for (int i = 0; i < 4; ++i) r[i] = a[i] + (b[i] - a[i]) * t;
            return r;
          };
          auto triHits = [&](std::array<float, 4> a, std::array<float, 4> b, std::array<float, 4> c) {
            // Sutherland-Hodgman against the single plane w = eps.
            std::vector<std::array<float, 4>> poly{a, b, c}, out;
            for (size_t i = 0; i < poly.size(); ++i) {
              const auto& cur = poly[i];
              const auto& nxt = poly[(i + 1) % poly.size()];
              const bool curIn = cur[3] >= kEps, nxtIn = nxt[3] >= kEps;
              if (curIn) out.push_back(cur);
              if (curIn != nxtIn) {
                const float d = nxt[3] - cur[3];
                if (std::fabs(d) > 1e-20f) out.push_back(lerp4(cur, nxt, (kEps - cur[3]) / d));
              }
            }
            if (out.size() < 3) return false;
            // Fan-triangulate the clipped polygon and do an exact point-in-triangle test.
            float px0, py0;
            scr(out[0], px0, py0);
            for (size_t i = 1; i + 1 < out.size(); ++i) {
              float px1, py1, px2, py2;
              scr(out[i], px1, py1);
              scr(out[i + 1], px2, py2);
              const float d1 = (s_wx - px1) * (py0 - py1) - (px0 - px1) * (s_wy - py1);
              const float d2 = (s_wx - px2) * (py1 - py2) - (px1 - px2) * (s_wy - py2);
              const float d3 = (s_wx - px0) * (py2 - py0) - (px2 - px0) * (s_wy - py0);
              const bool neg = d1 < 0 || d2 < 0 || d3 < 0;
              const bool pos = d1 > 0 || d2 > 0 || d3 > 0;
              if (!(neg && pos)) return true;   // no sign mix -> inside
            }
            return false;
          };
          const size_t n = clips.size();
          switch (prim) {
          case GX_QUADS:
            for (size_t i = 0; i + 3 < n && !hits; i += 4)
              hits = triHits(clips[i], clips[i + 1], clips[i + 2]) ||
                     triHits(clips[i], clips[i + 2], clips[i + 3]);
            break;
          case GX_TRIANGLES:
            for (size_t i = 0; i + 2 < n && !hits; i += 3)
              hits = triHits(clips[i], clips[i + 1], clips[i + 2]);
            break;
          case GX_TRIANGLESTRIP:
            for (size_t i = 0; i + 2 < n && !hits; ++i)
              hits = triHits(clips[i], clips[i + 1], clips[i + 2]);
            break;
          case GX_TRIANGLEFAN:
            for (size_t i = 1; i + 1 < n && !hits; ++i)
              hits = triHits(clips[0], clips[i], clips[i + 1]);
            break;
          default:
            break;   // lines/points: no area, cannot cover a pixel
          }
        }
        // Report when the box covers the point OR when the quad crosses the eye plane (where
        // the box is not trustworthy). Also report EVERY 4-vertex prim, covered or not — the
        // culprit is known to be a 4-vertex draw, so an exhaustive list for one frame is
        // small and cannot hide it.
        const bool boxCovers = any && s_wx >= sx0 && s_wx <= sx1 && s_wy >= sy0 && s_wy <= sy1;
        const bool covers = hits;   // exact, clip-correct coverage
        (void)boxCovers;
        // A prim with vertices at or behind the eye plane has NO trustworthy screen box: the
        // in-front vertices alone give a box that can exclude the watch point while the clipped
        // triangle still rasterizes across it, and a prim with EVERY vertex behind the eye
        // produces no box at all (`any` stays false, so `covers` can never fire). Those are
        // invisible to a coverage test by construction, so they get their own class rather than
        // being silently counted as "does not cover".
        sb_covers_watch = covers;
        sb_wneg_partial = wneg > 0 && wneg < vtxCount;
        sb_wneg_full = wneg > 0 && wneg == vtxCount;
        // SB_DEGEN_DRAW=1: report draws whose transformed geometry COLLAPSES — a screen box
        // under 2px across with 3+ vertices. A skinned limb that renders as a sliver instead of
        // an arm is exactly this, and it names the position matrix responsible, which is the
        // thing to compare against the other runtime. Reports the matrix VALUES, because a
        // matrix can be non-zero (so a zero-row check passes) and still be garbage.
        {
          static int s_dinit = 0;
          static bool s_don = false;
          if (!s_dinit) { s_dinit = 1; s_don = std::getenv("SB_DEGEN_DRAW") != nullptr; }
          // Restricted to boxes that land ON SCREEN: a prim projecting to a single point far
          // outside the viewport is simply offscreen geometry, not a collapsed limb.
          const bool onScreen = sx0 > -64.f && sx1 < 704.f && sy0 > -64.f && sy1 < 512.f;
          if (s_don && any && onScreen && vtxCount >= 3 && (sx1 - sx0) < 8.f && (sy1 - sy0) < 8.f) {
            static long n = 0;
            if (++n <= 40) {
              const float* M = reinterpret_cast<const float*>(
                  &g_gxState.pnMtx[firstMtxIdx % MaxPnMtx].pos);
              std::fprintf(stderr,
                           "[degen] draw#%ld verts=%u mtx=%u box=[%.1f..%.1f x %.1f..%.1f] "
                           "M=[%.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f] "
                           "tex0=%ux%u mark='%s'\n",
                           g_sbPushedDrawCount, vtxCount, firstMtxIdx, sx0, sx1, sy0, sy1,
                           M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7],
                           M[8], M[9], M[10], M[11],
                           g_gxState.textures[0].texObj.width(),
                           g_gxState.textures[0].texObj.height(), g_sbLastMarker.c_str());
            }
          }
        }

        // The frame gate applies to REPORTING only. `covers` must be computed on every frame,
        // because SB_SKIP_COVERING has to drop the draw in every frame to change the picture —
        // gating the computation would skip in one frame and leave the dumped one untouched.
        const bool report = (s_wframe < 0 ||
                             static_cast<long>(sb_gx_vi_retrace_count()) == s_wframe);
        if (report && (covers || wneg > 0 || vtxCount == 4)) {
          const auto& t0 = g_gxState.textures[0].texObj;
          const auto& st = g_gxState.tevStages[g_gxState.numTevStages ? g_gxState.numTevStages - 1 : 0];
          std::fprintf(stderr,
                       "[pixel-watch] %s draw#%ld %s verts=%u wneg=%u box=[%.0f..%.0f x %.0f..%.0f] "
                       "tex0=%ux%u fmt=%u mips=%u hasMip=%d maxlod=%.1f minf=%d mode0=%08x id=%u "
                       "tev=%u bm=%d bf=%d/%d cU=%d aU=%d zc=%d zu=%d cull=%d "
                       "lastTEV=c(%d,%d,%d,%d)op=%d,%d,%d mark='%s'\n",
                       covers ? "COVERS"
                              : (wneg == 0 ? "       "
                                           : (wneg < vtxCount ? "PARTIAL" : "BEHIND ")),
                       g_sbPushedDrawCount, g_gxState.projType == GX_ORTHOGRAPHIC ? "O" : "P",
                       vtxCount, wneg, sx0, sx1, sy0, sy1, t0.width(), t0.height(),
                       // Mip state, because over-mipping is a KNOWN cause of exactly this
                       // symptom: a texture decoded with more levels than its dimensions
                       // support samples GARBAGE past the level-0 data and reads as white
                       // speckle (see the clamp in gfx/texture.hpp mip_count()).
                       t0.format(), t0.mip_count(), t0.has_mips() ? 1 : 0, (float)t0.max_lod(),
                       // Real HW has no 'has mips' bit: mip usage is encoded in TexMode0's
                       // min-filter field. If the recomp's registers say a MIP filter while
                       // aurora's flag says no mips, the flag simply isn't derived on the
                       // register path — which is a derivation gap, not missing game data.
                       (int)t0.min_filter(), t0.mode0,
                       t0.texObjId,
                       g_gxState.numTevStages, (int)g_gxState.blendMode,
                       (int)g_gxState.blendFacSrc, (int)g_gxState.blendFacDst,
                       g_gxState.colorUpdate ? 1 : 0, g_gxState.alphaUpdate ? 1 : 0,
                       (int)g_gxState.depthCompare, (int)g_gxState.depthUpdate,
                       (int)g_gxState.cullMode,
                       (int)st.colorPass.a, (int)st.colorPass.b, (int)st.colorPass.c,
                       (int)st.colorPass.d, (int)st.colorOp.op, (int)st.colorOp.bias,
                       (int)st.colorOp.scale, g_sbLastMarker.c_str());
        }
      }
    }
  }

  // SB_UV_PROBE=<W>x<H>: for draws binding a texture of those dimensions on texmap 0, decode
  // the first vertices' DIRECT texcoords straight out of the vertex stream. Configuration
  // comparisons (texgen, VAT, descriptors) can all match while the VALUES differ, and nothing
  // else reports what a draw actually samples.
  {
    static int s_uvInit = 0;
    static const char* s_uvWant = nullptr;
    if (!s_uvInit) { s_uvInit = 1; s_uvWant = std::getenv("SB_UV_PROBE"); }
    if (s_uvWant != nullptr && s_uvWant[0] != '\0' && vtxCount > 0) {
      const auto& t0obj = g_gxState.textures[0].texObj;
      char dims[32];
      std::snprintf(dims, sizeof(dims), "%ux%u", t0obj.width(), t0obj.height());
      if (std::strcmp(dims, s_uvWant) == 0) {
        static int n = 0;
        if (n < 12) {
          ++n;
          // Byte offset of each attribute within a vertex, in GX attribute order.
          const auto attrBytes = [&](int a) -> u32 {
            const auto d = g_gxState.vtxDesc[a];
            if (d == GX_NONE) return 0;
            if (d == GX_INDEX8) return 1;
            if (d == GX_INDEX16) return 2;
            if (a >= GX_VA_PNMTXIDX && a <= GX_VA_TEX7MTXIDX) return 1;
            const auto& af = g_gxState.vtxFmts[fmt].attrs[a];
            u32 comps = 0;
            if (a == GX_VA_POS) comps = af.cnt == GX_POS_XYZ ? 3 : 2;
            else if (a == GX_VA_NRM) comps = 3;
            else if (a == GX_VA_CLR0 || a == GX_VA_CLR1) comps = 0;  // handled below
            else comps = af.cnt == GX_TEX_ST ? 2 : 1;
            if (a == GX_VA_CLR0 || a == GX_VA_CLR1) {
              switch (af.type) {
              case GX_RGB565: case GX_RGBA4: return 2;
              case GX_RGB8: case GX_RGBA6: return 3;
              default: return 4;
              }
            }
            u32 csz = 0;
            switch (af.type) {
            case GX_U8: case GX_S8: csz = 1; break;
            case GX_U16: case GX_S16: csz = 2; break;
            default: csz = 4; break;
            }
            return comps * csz;
          };
          for (int which = 0; which < 2; ++which) {
            const int target = GX_VA_TEX0 + which;
            if (g_gxState.vtxDesc[target] != GX_DIRECT) continue;
            const auto& af = g_gxState.vtxFmts[fmt].attrs[target];
            if (af.type != GX_F32) continue;   // only decode what is unambiguous
            u32 off = 0;
            for (int a = GX_VA_PNMTXIDX; a < target; ++a) off += attrBytes(a);
            const u32 vsz = g_gxState.lastVtxFmt == fmt ? g_gxState.lastVtxSize
                                                        : calculate_last_vtx_size(fmt);
            // Stamp the frame ordinal: two probe lines could be consecutive FRAMES or two
            // draws within one frame, and a per-frame rate cannot be read off the sequence
            // without knowing which.
            std::fprintf(stderr, "[uv-probe] n=%d rc=%u tex%d verts=%u vsz=%u off=%u uv:", n,
                         sb_gx_vi_retrace_count(), which, vtxCount, vsz, off);
            const unsigned show = vtxCount < 4 ? vtxCount : 4;
            for (unsigned v = 0; v < show; ++v) {
              const u32 base = pos + v * vsz + off;
              if (base + 8 > size) break;
              const float u = read_f32(data + base, true);
              const float vv = read_f32(data + base + 4, true);
              std::fprintf(stderr, " (%.4f,%.4f)", u, vv);
            }
            std::fprintf(stderr, "\n");
          }
        }
      }
    }
  }

  // SB_NDC_PROBE=<minVerts> [+ SB_NDC_MARK=<marker substring>] [+ SB_NDC_PROBE_AFTER=<retraceCount>]:
  // CPU-side replication of the vertex shader transform for indexed-position draws —
  // projects EVERY vertex through the exact matrices the GPU will use
  // (per-vertex PNMTXIDX honored, same row-dot convention as the WGSL) and
  // histograms the clip results. Distinguishes "strip lands off-screen due to
  // a transform bug" from "strip genuinely tiny / genuinely missing".
  //
  // SB_NDC_PROBE_AFTER (2026-07-10, title-backdrop-black probe): the 400-print
  // budget below is a whole-run budget, not per-frame — at title, thousands of
  // draws happen before the stable PRESS-START window (present ~800-3200), so
  // an unwindowed probe exhausts its budget on the logo/fade frames and never
  // reaches the frame under investigation. Gate on the game's own VI retrace
  // counter (VIGetRetraceCount, defined in sms-boot/runtime/sdk_stubs.cpp and
  // advanced once per sb_frame_present — same counter SB_DUMP_FRAME_AFTER's
  // frame index tracks) so this can be pointed at the exact dumped present.
  {
    static int s_minVerts = -2;
    static const char* s_markFilter = nullptr;
    static long s_afterRetrace = -1;
    // SB_NDC_DRAW=<lo>[:<hi>] (2026-07-14, seagull localization): window the
    // probe by GLOBAL DRAW INDEX — the same "count every draw reaching this
    // function since process start" scheme SB_DRAW_DUMP's s_dumped uses, so
    // indices from a [draw-dump] log map 1:1. Windowed draws bypass the
    // minVerts/budget gates and print EVERY vertex (up to 32), so a specific
    // suspect draw (e.g. an invisible sprite) can be inspected at the vertex
    // level. If a windowed draw can't be walked (non-indexed pos), that is
    // REPORTED, never silently skipped.
    static long s_ndcDrawLo = -1, s_ndcDrawHi = -1;
    static long s_ndcDrawCounter = -1;
    if (s_minVerts == -2) {
      const char* e = std::getenv("SB_NDC_PROBE");
      s_minVerts = (e != nullptr && e[0] != '\0') ? std::atoi(e) : -1;
      if (s_minVerts == 0) s_minVerts = 1;
      s_markFilter = std::getenv("SB_NDC_MARK");
      if (const char* a = std::getenv("SB_NDC_PROBE_AFTER"); a != nullptr && a[0] != '\0') {
        s_afterRetrace = std::atol(a);
      }
      if (const char* w = std::getenv("SB_NDC_DRAW"); w != nullptr && w[0] != '\0') {
        char* endp = nullptr;
        s_ndcDrawLo = std::strtol(w, &endp, 0);
        s_ndcDrawHi = (endp != nullptr && *endp == ':') ? std::strtol(endp + 1, nullptr, 0) : s_ndcDrawLo;
      }
    }
    s_ndcDrawCounter = g_sbPushedDrawCount; // post-merge index of the draw this prim starts (or, if it
                                            // merges, one past the draw it extends — hence +1 slack below)
    const bool ndcDrawWindowed =
        s_ndcDrawLo >= 0 && s_ndcDrawCounter >= s_ndcDrawLo && s_ndcDrawCounter <= s_ndcDrawHi + 1;
    static int s_printed = 0;
    // SB_NDC_PROBE companion: the per-vertex breakdown below is normally gated to the
    // first 6 MATCHED draws, which (2026-07-10 title-backdrop probe) happened to all be
    // an orthographic pass. Also unlock the per-vertex dump for the first few PERSPECTIVE
    // draws specifically, so a fully w<=0 (behind-camera) perspective draw is visible at
    // the vertex level instead of only as a "wneg=vtxCount" summary line.
    static int s_printedP = 0;
    const auto posDesc = g_gxState.vtxDesc[GX_VA_POS];
    const auto& posFmt = g_gxState.vtxFmts[fmt].attrs[GX_VA_POS];
    const auto& arr = g_gxState.arrays[GX_VA_POS];
    const bool afterWindowOk = s_afterRetrace < 0 || static_cast<long>(sb_gx_vi_retrace_count()) >= s_afterRetrace;
    const bool walkable = (posDesc == GX_INDEX16 || posDesc == GX_INDEX8) &&
                          (posFmt.type == GX_F32 || posFmt.type == GX_S16) && arr.data != nullptr;
    if (ndcDrawWindowed && !walkable) {
      std::fprintf(stderr,
                   "[ndc-draw] #%ld NOT WALKABLE: posDesc=%d posType=%d arr=%p verts=%u -- extend the walker "
                   "before trusting this window\n",
                   s_ndcDrawCounter, static_cast<int>(posDesc), static_cast<int>(posFmt.type),
                   arr.data, vtxCount);
    }
    // SB_NDC_DRAW companion [tex-id]: texture-identity line for every windowed draw —
    // GC image address (image3), host data ptr, dims/format, data version, and an FNV-1a
    // hash + alpha summary of the SOURCE texels, so two draws can be compared at the
    // texture-content level (2026-07-14 seagull probe: same state, different per-bird
    // texture addresses; are the invisible birds' texels actually transparent?).
    if (ndcDrawWindowed) {
      const auto& tobj = g_gxState.textures[0].texObj;
      const u8* td = static_cast<const u8*>(tobj.data);
      u32 w = tobj.width(), h = tobj.height(), f = tobj.format();
      // Size of the base level in GC layout (fmt-dependent bpp); enough for identity.
      u32 bpp4 = (f == GX_TF_RGBA8) ? 128 : (f == GX_TF_RGB565 || f == GX_TF_RGB5A3 || f == GX_TF_IA8) ? 64
                 : (f == GX_TF_I8 || f == GX_TF_IA4 || f == GX_TF_C8)                                   ? 32
                                                                                                        : 16; // bits per 4 texels
      u64 nbytes = static_cast<u64>(w) * h * bpp4 / 32;
      if (nbytes > 0x4000) nbytes = 0x4000;
      u64 hash = 1469598103934665603ull;
      u32 aZero = 0, aFull = 0, aSamp = 0;
      if (td != nullptr) {
        for (u64 i = 0; i < nbytes; ++i) { hash ^= td[i]; hash *= 1099511628211ull; }
        if (f == GX_TF_RGB5A3) {
          // 16bpp BE: top bit 0 => 3-bit alpha in bits 12-14 (0 possible); top bit 1 => opaque.
          for (u64 i = 0; i + 1 < nbytes; i += 2) {
            u16 v = static_cast<u16>((td[i] << 8) | td[i + 1]);
            ++aSamp;
            if (v & 0x8000) ++aFull;
            else if ((v & 0x7000) == 0) ++aZero;
          }
        } else if (f == GX_TF_IA8) {
          for (u64 i = 0; i + 1 < nbytes; i += 2) { ++aSamp; if (td[i] == 0) ++aZero; else if (td[i] == 0xFF) ++aFull; }
        }
      }
      // Mip-level alpha summary (RGB5A3): the cutout (GEQUAL 128) samples the mip
      // matching on-screen size — a zero/stale mip chain makes SMALL instances
      // vanish while big ones only wash out. Level offsets follow GC layout
      // (levels packed contiguously after the base).
      char mipbuf[160]; mipbuf[0] = 0;
      if (td != nullptr && f == GX_TF_RGB5A3 && tobj.mip_count() > 1) {
        u64 off = static_cast<u64>(w) * h * 2;
        int mn = 0;
        for (u32 lvl = 1; lvl < tobj.mip_count() && lvl <= 4; ++lvl) {
          u32 lw = std::max(w >> lvl, 1u), lh = std::max(h >> lvl, 1u);
          u64 lb = static_cast<u64>(lw) * lh * 2;
          u32 z = 0, fu = 0, n = 0;
          for (u64 i = 0; i + 1 < lb; i += 2) {
            u16 v = static_cast<u16>((td[off + i] << 8) | td[off + i + 1]);
            ++n;
            if (v & 0x8000) ++fu;
            else if ((v & 0x7000) == 0) ++z;
          }
          mn += std::snprintf(mipbuf + mn, sizeof(mipbuf) - mn, " L%u[n=%u z=%u f=%u]", lvl, n, z, fu);
          off += lb;
        }
      }
      std::fprintf(stderr,
                   "[tex-id] #%ld image3=0x%x data=%p %ux%u fmt=%u ver=%u mips=%u hash=%016llx "
                   "alpha[samp=%u zero=%u full=%u]%s\n",
                   s_ndcDrawCounter, tobj.image3, static_cast<const void*>(td), w, h, f,
                   tobj.texDataVersion, tobj.mip_count(), static_cast<unsigned long long>(hash),
                   aSamp, aZero, aFull, mipbuf);
    }
    const bool probeMatch = s_minVerts > 0 && s_printed < 400 && afterWindowOk &&
                            vtxCount >= static_cast<u16>(s_minVerts) &&
                            (s_markFilter == nullptr || g_sbLastMarker.find(s_markFilter) != std::string::npos);
    if ((probeMatch || ndcDrawWindowed) && walkable) {
      ++s_printed;
      // Per-vertex offsets: attrs before POS are u8 matrix indices when DIRECT.
      u32 posOff = 0;
      int pnOff = -1;
      for (int a = GX_VA_PNMTXIDX; a < GX_VA_POS; ++a) {
        if (g_gxState.vtxDesc[a] == GX_DIRECT) {
          if (a == GX_VA_PNMTXIDX) pnOff = static_cast<int>(posOff);
          posOff += 1;
        }
      }
      const float invFrac = 1.f / static_cast<float>(1u << posFmt.frac);
      const bool le = arr.le;
      const float* P = reinterpret_cast<const float*>(&g_gxState.proj);
      u32 in = 0, zin = 0, wneg = 0;
      float xmin = 1e30f, xmax = -1e30f, ymin = 1e30f, ymax = -1e30f, zmin = 1e30f, zmax = -1e30f;
      u32 firstMtx = g_gxState.currentPnMtx;
      for (u32 v = 0; v < vtxCount; ++v) {
        const u8* vp = data + pos + v * vtxSize;
        u32 mtxIdx = g_gxState.currentPnMtx;
        if (pnOff >= 0) mtxIdx = vp[pnOff] / 3u;
        if (v == 0) firstMtx = mtxIdx;
        u32 idx = posDesc == GX_INDEX16 ? read_u16(vp + posOff, true) : vp[posOff];
        const u8* pd = static_cast<const u8*>(arr.data) + idx * arr.stride;
        float x, y, z;
        if (posFmt.type == GX_F32) {
          auto rf = [le](const u8* p) {
            u32 u;
            std::memcpy(&u, p, 4);
            if (!le) u = __builtin_bswap32(u);
            float f;
            std::memcpy(&f, &u, 4);
            return f;
          };
          x = rf(pd); y = rf(pd + 4); z = posFmt.cnt == GX_POS_XYZ ? rf(pd + 8) : 0.f;
        } else {
          auto rs = [le](const u8* p) {
            u16 u;
            std::memcpy(&u, p, 2);
            if (!le) u = static_cast<u16>((u << 8) | (u >> 8));
            return static_cast<float>(static_cast<s16>(u));
          };
          x = rs(pd) * invFrac; y = rs(pd + 2) * invFrac;
          z = posFmt.cnt == GX_POS_XYZ ? rs(pd + 4) * invFrac : 0.f;
        }
        const float* M = reinterpret_cast<const float*>(&g_gxState.pnMtx[mtxIdx % MaxPnMtx].pos);
        float mv[3];
        for (int c = 0; c < 3; ++c) mv[c] = M[4 * c] * x + M[4 * c + 1] * y + M[4 * c + 2] * z + M[4 * c + 3];
        float clip[4];
        for (int c = 0; c < 4; ++c)
          clip[c] = P[4 * c] * mv[0] + P[4 * c + 1] * mv[1] + P[4 * c + 2] * mv[2] + P[4 * c + 3];
        if (clip[3] <= 0.f) {
          ++wneg;
          if (ndcDrawWindowed ? v < 32 : (g_gxState.projType != GX_ORTHOGRAPHIC && s_printedP < 6 && v < 4)) {
            std::fprintf(stderr,
                         "[ndc-probe-behind]  v%u idx=%u pos=(%.1f,%.1f,%.1f) mtx=%u M=[%.3f %.3f %.3f %.3f | "
                         "%.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f] mv=(%.1f,%.1f,%.1f) clipW=%.3f mark='%s'\n",
                         v, idx, x, y, z, mtxIdx, M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8], M[9],
                         M[10], M[11], mv[0], mv[1], mv[2], clip[3], g_sbLastMarker.c_str());
          }
          continue;
        }
        const float nx = clip[0] / clip[3], ny = clip[1] / clip[3], nz = clip[2] / clip[3];
        xmin = std::min(xmin, nx); xmax = std::max(xmax, nx);
        ymin = std::min(ymin, ny); ymax = std::max(ymax, ny);
        zmin = std::min(zmin, nz); zmax = std::max(zmax, nz);
        // GC clip z convention: visible depth is z/w in [-1, 0]. Aurora runs
        // with unclippedDepth, so XY containment alone decides rasterization.
        if (nx >= -1.f && nx <= 1.f && ny >= -1.f && ny <= 1.f) ++in;
        if (nz >= -1.f && nz <= 0.f) ++zin;
        if (ndcDrawWindowed ? v < 32 : (s_printed <= 6 && v < 4)) {
          // Also fetch this vertex's CLR0 raw bytes (if indexed) — shading
          // ground truth for "geometry rasterizes but comes out black".
          u32 c0raw[4] = {0xAAAA, 0, 0, 0};
          const auto c0desc = g_gxState.vtxDesc[GX_VA_CLR0];
          if ((c0desc == GX_INDEX16 || c0desc == GX_INDEX8) && g_gxState.arrays[GX_VA_CLR0].data != nullptr) {
            u32 c0off = 0;
            for (int a = GX_VA_PNMTXIDX; a < GX_VA_CLR0; ++a) {
              switch (g_gxState.vtxDesc[a]) {
              case GX_NONE: break;
              case GX_DIRECT: c0off += a < GX_VA_POS ? 1 : comp_type_size(static_cast<GXAttr>(a), g_gxState.vtxFmts[fmt].attrs[a].type) * comp_cnt_count(static_cast<GXAttr>(a), g_gxState.vtxFmts[fmt].attrs[a].cnt); break;
              case GX_INDEX8: c0off += (a == GX_VA_NRM && g_gxState.vtxFmts[fmt].attrs[a].cnt == GX_NRM_NBT3) ? 3 : 1; break;
              case GX_INDEX16: c0off += (a == GX_VA_NRM && g_gxState.vtxFmts[fmt].attrs[a].cnt == GX_NRM_NBT3) ? 6 : 2; break;
              }
            }
            const u32 cidx = c0desc == GX_INDEX16 ? read_u16(vp + c0off, true) : vp[c0off];
            const auto& carr = g_gxState.arrays[GX_VA_CLR0];
            const u8* cd = static_cast<const u8*>(carr.data) + cidx * carr.stride;
            c0raw[0] = cd[0]; c0raw[1] = cd[1]; c0raw[2] = cd[2]; c0raw[3] = carr.stride > 3 ? cd[3] : 0;
          }
          // TEX0 UVs (indexed) — which texture region this vertex samples. For an
          // alpha-cutout material (aComp GEQUAL) wrong UVs land on transparent
          // texels and the draw vanishes with every other state field correct
          // (2026-07-14 seagull probe).
          float t0u = -999.f, t0v = -999.f;
          const auto t0desc = g_gxState.vtxDesc[GX_VA_TEX0];
          if ((t0desc == GX_INDEX16 || t0desc == GX_INDEX8) && g_gxState.arrays[GX_VA_TEX0].data != nullptr) {
            u32 t0off = 0;
            for (int a = GX_VA_PNMTXIDX; a < GX_VA_TEX0; ++a) {
              switch (g_gxState.vtxDesc[a]) {
              case GX_NONE: break;
              case GX_DIRECT: t0off += a < GX_VA_POS ? 1 : comp_type_size(static_cast<GXAttr>(a), g_gxState.vtxFmts[fmt].attrs[a].type) * comp_cnt_count(static_cast<GXAttr>(a), g_gxState.vtxFmts[fmt].attrs[a].cnt); break;
              case GX_INDEX8: t0off += (a == GX_VA_NRM && g_gxState.vtxFmts[fmt].attrs[a].cnt == GX_NRM_NBT3) ? 3 : 1; break;
              case GX_INDEX16: t0off += (a == GX_VA_NRM && g_gxState.vtxFmts[fmt].attrs[a].cnt == GX_NRM_NBT3) ? 6 : 2; break;
              }
            }
            const u32 tidx = t0desc == GX_INDEX16 ? read_u16(vp + t0off, true) : vp[t0off];
            const auto& tarr = g_gxState.arrays[GX_VA_TEX0];
            const u8* tdp = static_cast<const u8*>(tarr.data) + tidx * tarr.stride;
            const auto& tfmt = g_gxState.vtxFmts[fmt].attrs[GX_VA_TEX0];
            const bool tle = tarr.le;
            if (tfmt.type == GX_F32) {
              auto rf = [tle](const u8* p) { u32 u; std::memcpy(&u, p, 4); if (!tle) u = __builtin_bswap32(u); float f; std::memcpy(&f, &u, 4); return f; };
              t0u = rf(tdp); t0v = rf(tdp + 4);
            } else if (tfmt.type == GX_S16 || tfmt.type == GX_U16) {
              auto rs = [tle, &tfmt](const u8* p) { u16 u; std::memcpy(&u, p, 2); if (!tle) u = static_cast<u16>((u << 8) | (u >> 8)); float f = tfmt.type == GX_S16 ? static_cast<float>(static_cast<s16>(u)) : static_cast<float>(u); return f / static_cast<float>(1u << tfmt.frac); };
              t0u = rs(tdp); t0v = rs(tdp + 2);
            }
          }
          std::fprintf(stderr,
                       "[ndc-probe]   v%u idx=%u pos=(%.1f,%.1f,%.1f) mv=(%.1f,%.1f,%.1f) ndc=(%.3f,%.3f,%.4f) "
                       "w=%.1f mtx=%u clr0=[%02x %02x %02x %02x] c0type=%d t0=(%.4f,%.4f)\n",
                       v, idx, x, y, z, mv[0], mv[1], mv[2], nx, ny, nz, clip[3], firstMtx, c0raw[0], c0raw[1],
                       c0raw[2], c0raw[3], static_cast<int>(g_gxState.vtxFmts[fmt].attrs[GX_VA_CLR0].type), t0u, t0v);
        }
      }
      // TEV stage 0 combiner state — decides "vertex colors right but TEV
      // outputs black".
      {
        const auto& t0 = g_gxState.tevStages[0];
        std::fprintf(stderr,
                     "[ndc-probe]  tev0 C[a=%d b=%d c=%d d=%d op=%d bias=%d scale=%d out=%d] "
                     "A[a=%d b=%d c=%d d=%d op=%d out=%d] ras=%d tc=%d tm=%d nStages=%u\n",
                     static_cast<int>(t0.colorPass.a), static_cast<int>(t0.colorPass.b),
                     static_cast<int>(t0.colorPass.c), static_cast<int>(t0.colorPass.d),
                     static_cast<int>(t0.colorOp.op), static_cast<int>(t0.colorOp.bias),
                     static_cast<int>(t0.colorOp.scale), static_cast<int>(t0.colorOp.outReg),
                     static_cast<int>(t0.alphaPass.a), static_cast<int>(t0.alphaPass.b),
                     static_cast<int>(t0.alphaPass.c), static_cast<int>(t0.alphaPass.d),
                     static_cast<int>(t0.alphaOp.op), static_cast<int>(t0.alphaOp.outReg),
                     static_cast<int>(t0.channelId), static_cast<int>(t0.texCoordId),
                     static_cast<int>(t0.texMapId), g_gxState.numTevStages);
      }
      std::fprintf(stderr,
                   "[ndc-probe] #%d verts=%u inXY=%u inZ=%u wneg=%u proj=%c ndcX=[%.2f..%.2f] ndcY=[%.2f..%.2f] "
                   "ndcZ=[%.4f..%.4f] mtx0=%u cull=%d aComp=%d/%u,%d/%u zc=%d zu=%d mark='%s'\n",
                   ndcDrawWindowed ? static_cast<int>(s_ndcDrawCounter) : s_printed,
                   vtxCount, in, zin, wneg, g_gxState.projType == GX_ORTHOGRAPHIC ? 'O' : 'P', xmin, xmax,
                   ymin, ymax, zmin, zmax, firstMtx, static_cast<int>(g_gxState.cullMode),
                   static_cast<int>(g_gxState.alphaCompare.comp0), g_gxState.alphaCompare.ref0,
                   static_cast<int>(g_gxState.alphaCompare.comp1), g_gxState.alphaCompare.ref1,
                   static_cast<int>(g_gxState.depthCompare), static_cast<int>(g_gxState.depthUpdate),
                   g_sbLastMarker.c_str());
      if (g_gxState.projType != GX_ORTHOGRAPHIC) ++s_printedP;
    }
  }

  // SB_LENS_UV_DBG=1 (diagnostic, title lens-flare-ghost crosshatch investigation):
  // dump the raw per-vertex TEX0 UV (as actually read from the vertex attribute
  // stream, direct or indexed) plus stage-0 texgen + bound texture size/wrap, for
  // every draw marked "LensFlare" (TDrawBufObj name, set synchronously by
  // J3DDrawBuffer::draw via GXInsertDebugMarker). Answers: do the big on-screen
  // ghosts (#9-11) sample their sprite atlas with 0..1 (single map) or 0..N
  // (tiling) UVs?
  // draw_prim runs ~46k times per frame, so this gate is on the hottest path in the renderer. It
  // used to be an uncached std::getenv — an environment scan per primitive for a diagnostic that is
  // off — which measured 2.9 ms/frame. A hoisted lucent::Channel is a relaxed atomic load and a
  // compare (0.66 ns/call per lucent's own benchmark), and it puts the diagnostic on the one logger
  // instead of leaving a gated fprintf behind.
  static const lucent::Channel chLensUv{"lensuv"};
  if (chLensUv && g_sbLastMarker.find("LensFlare") != std::string::npos) {
    static long s_n = 0;
    static long s_hit = 0;
    if (s_hit < 40) {
      ++s_hit;
      const auto& hstage0 = g_gxState.tevStages[0];
      int htexW = -1, htexH = -1, hwrapS = -1, hwrapT = -1;
      if (hstage0.texMapId >= 0 && static_cast<unsigned>(hstage0.texMapId) < aurora::gx::MaxTextures) {
        const auto& htobj = g_gxState.textures[hstage0.texMapId].texObj;
        htexW = htobj.width();
        htexH = htobj.height();
        hwrapS = static_cast<int>(htobj.wrap_s());
        hwrapT = static_cast<int>(htobj.wrap_t());
      }
      lucent::debug(chLensUv,
                    "hit #{} prim={} verts={} fmt={} t0desc={} t0type={} t0cnt={} t0frac={} "
                    "arrData={} arrStride={} numTevStages={} tev0[texCoordId={} texMapId={} chanId={}] "
                    "tex={}x{} wrap=({},{}) mark='{}'",
                    s_hit, static_cast<unsigned>(prim), vtxCount, static_cast<int>(fmt),
                    static_cast<int>(g_gxState.vtxDesc[GX_VA_TEX0]),
                    static_cast<int>(g_gxState.vtxFmts[fmt].attrs[GX_VA_TEX0].type),
                    static_cast<int>(g_gxState.vtxFmts[fmt].attrs[GX_VA_TEX0].cnt),
                    g_gxState.vtxFmts[fmt].attrs[GX_VA_TEX0].frac,
                    static_cast<const void*>(g_gxState.arrays[GX_VA_TEX0].data),
                    g_gxState.arrays[GX_VA_TEX0].stride, g_gxState.numTevStages,
                    static_cast<int>(hstage0.texCoordId), static_cast<int>(hstage0.texMapId),
                    static_cast<int>(hstage0.channelId), htexW, htexH, hwrapS, hwrapT, g_sbLastMarker);
    }
    if (s_n < 400) {
      const auto& vtxFmt = g_gxState.vtxFmts[fmt];
      const auto t0Desc = g_gxState.vtxDesc[GX_VA_TEX0];
      // Offset of TEX0 within a vertex (same accumulation pattern as the CLR0
      // probe above, extended up to GX_VA_TEX0).
      u32 off = 0;
      for (int a = GX_VA_PNMTXIDX; a < GX_VA_TEX0; ++a) {
        switch (g_gxState.vtxDesc[a]) {
        case GX_NONE:
          break;
        case GX_DIRECT:
          off += (a < GX_VA_POS) ? 1
                                  : comp_type_size(static_cast<GXAttr>(a), vtxFmt.attrs[a].type) *
                                        comp_cnt_count(static_cast<GXAttr>(a), vtxFmt.attrs[a].cnt);
          break;
        case GX_INDEX8:
          off += (a == GX_VA_NRM && vtxFmt.attrs[a].cnt == GX_NRM_NBT3) ? 3 : 1;
          break;
        case GX_INDEX16:
          off += (a == GX_VA_NRM && vtxFmt.attrs[a].cnt == GX_NRM_NBT3) ? 6 : 2;
          break;
        }
      }
      const auto& t0Fmt = vtxFmt.attrs[GX_VA_TEX0];
      const auto& tcg = g_gxState.tcgs[0];
      const auto& t0Stage = g_gxState.tevStages[0];
      int texW = -1, texH = -1, wrapS = -1, wrapT = -1;
      if (t0Stage.texMapId >= 0 && static_cast<unsigned>(t0Stage.texMapId) < aurora::gx::MaxTextures) {
        const auto& tobj = g_gxState.textures[t0Stage.texMapId].texObj;
        texW = tobj.width();
        texH = tobj.height();
        wrapS = static_cast<int>(tobj.wrap_s());
        wrapT = static_cast<int>(tobj.wrap_t());
      }
      const u8 csz = comp_type_size(GX_VA_TEX0, t0Fmt.type);
      const bool has2 = comp_cnt_count(GX_VA_TEX0, t0Fmt.cnt) > 1;
      for (u32 v = 0; v < std::min<u32>(vtxCount, 4); ++v) {
        const u8* vp = data + pos + v * vtxSize;
        const u8* src = nullptr;
        bool le = true;
        if (t0Desc == GX_DIRECT) {
          src = vp + off;
        } else if ((t0Desc == GX_INDEX8 || t0Desc == GX_INDEX16) && g_gxState.arrays[GX_VA_TEX0].data != nullptr) {
          const u32 idx = t0Desc == GX_INDEX16 ? read_u16(vp + off, true) : vp[off];
          const auto& arr = g_gxState.arrays[GX_VA_TEX0];
          src = static_cast<const u8*>(arr.data) + idx * arr.stride;
          le = arr.le;
        }
        if (src == nullptr) continue;
        auto readComp = [&](const u8* p) -> float {
          switch (t0Fmt.type) {
          case GX_U8:
            return static_cast<float>(p[0]);
          case GX_S8:
            return static_cast<float>(static_cast<s8>(p[0]));
          case GX_U16: {
            u16 x;
            std::memcpy(&x, p, 2);
            if (!le) x = static_cast<u16>((x << 8) | (x >> 8));
            return static_cast<float>(x);
          }
          case GX_S16: {
            u16 x;
            std::memcpy(&x, p, 2);
            if (!le) x = static_cast<u16>((x << 8) | (x >> 8));
            return static_cast<float>(static_cast<s16>(x));
          }
          case GX_F32: {
            u32 xi;
            std::memcpy(&xi, p, 4);
            if (!le) xi = __builtin_bswap32(xi);
            float f;
            std::memcpy(&f, &xi, 4);
            return f;
          }
          default:
            return 0.f;
          }
        };
        const float invFrac = 1.f / static_cast<float>(1u << t0Fmt.frac);
        const float u = readComp(src) * invFrac;
        const float w = has2 ? readComp(src + csz) * invFrac : 0.f;
        ++s_n;
        lucent::debug(chLensUv,
                      "#{} v={} desc={} type={} cnt={} frac={} uv=({:.4f},{:.4f}) tex={}x{} wrap=({},{}) "
                      "tcg[src={} mtx={} type={}] verts={} proj={} mark='{}'",
                      s_n, v, static_cast<int>(t0Desc), static_cast<int>(t0Fmt.type), static_cast<int>(t0Fmt.cnt),
                      t0Fmt.frac, u, w, texW, texH, wrapS, wrapT, static_cast<int>(tcg.src), static_cast<int>(tcg.mtx),
                      static_cast<int>(tcg.type), vtxCount, g_gxState.projType == GX_ORTHOGRAPHIC ? 'O' : 'P',
                      g_sbLastMarker);
      }
    }
  }

  // AUTO ARRAY SIZING: arrays registered with size 0 (J3D's "trust" contract)
  // previously uploaded ZERO bytes — no indexed vertex attribute data ever
  // reached the GPU, so every INDEX8/INDEX16 J3D model rendered from an empty
  // storage buffer (invisible sky/map/sea at title). GC hardware reads guest
  // RAM directly; wgpu needs an explicit upload, so derive the required size
  // from the maximum index this draw actually references and grow the
  // per-array upload budget. A growth invalidates the cached upload and
  // blocks merging (the merge head's uniform holds the old storage offset).
  DP_PHASE(SB_DP_DIAG_PRE);
  {
    const auto& vtxFmt = g_gxState.vtxFmts[fmt];
    struct IdxField {
      u16 off;
      u8 wide; // index width in bytes (1 or 2)
      u8 n;    // consecutive indices (3 for NBT3)
      u8 attr;
    };
    IdxField fields[GX_VA_TEX7 + 1];
    int nFields = 0;
    u32 off = 0;
    for (int a = GX_VA_PNMTXIDX; a <= GX_VA_TEX7; ++a) {
      const auto desc = g_gxState.vtxDesc[a];
      if (desc == GX_NONE) continue;
      if (desc == GX_DIRECT) {
        off += comp_type_size(static_cast<GXAttr>(a), vtxFmt.attrs[a].type) *
               comp_cnt_count(static_cast<GXAttr>(a), vtxFmt.attrs[a].cnt);
        continue;
      }
      const bool nbt3 = a == GX_VA_NRM && vtxFmt.attrs[a].cnt == GX_NRM_NBT3;
      const u8 wide = desc == GX_INDEX16 ? 2 : 1;
      const u8 cnt = nbt3 ? 3 : 1;
      if (g_gxState.arrays[a].size == 0 && g_gxState.arrays[a].data != nullptr) {
        fields[nFields++] = {static_cast<u16>(off), wide, cnt, static_cast<u8>(a)};
      }
      off += static_cast<u32>(wide) * cnt;
    }
    DP_PHASE(SB_DP_ATTRENUM);
    if (nFields > 0) {
      // The scan is per-vertex, per-indexed-attribute and runs on EVERY draw, so it is the obvious
      // suspect for draw_prim's 45% share of render time — but "obvious suspect" is not a
      // measurement, and the last two attributions in this arc were both wrong. It is now timed by
      // the SB_DP_IDXSCAN phase probe rather than by a clock_gettime pair straddling this loop:
      // that pair cost ~40 ns of the ~340 ns body and was charged to draw_prim's own total, so the
      // old "scan = 14%" reading was in part the probe measuring itself.
      u32 maxIdx[GX_VA_TEX7 + 1] = {};
      for (u32 v = 0; v < vtxCount; ++v) {
        const u8* vp = data + pos + v * vtxSize;
        for (int f = 0; f < nFields; ++f) {
          const auto& fl = fields[f];
          for (u8 k = 0; k < fl.n; ++k) {
            const u32 idx = fl.wide == 2 ? read_u16(vp + fl.off + k * 2, true) : vp[fl.off + k];
            if (idx > maxIdx[fl.attr]) maxIdx[fl.attr] = idx;
          }
        }
      }
      for (int f = 0; f < nFields; ++f) {
        auto& arr = g_gxState.arrays[fields[f].attr];
        const u32 need = (maxIdx[fields[f].attr] + 1) * arr.stride;
        if (need > arr.sizeAuto) arr.sizeAuto = need;
        if (arr.cachedRange.size < arr.sizeAuto) {
          arr.cachedRange = {};
          g_gxState.stateDirty = true;
        }
      }
      DP_PHASE(SB_DP_IDXSCAN);
    }
  }

  // SB_SKIP_COVERING=1 (diagnostic): drop precisely the draws whose transformed screen box
  // contains the SB_PIXEL_WATCH point. This is the direct attribution test — the same code that
  // NAMES a covering draw also removes it, so "which draw paints this pixel" is answered by one
  // run, with no index translation between two instruments and no state-predicate guessing.
  //
  // It must live HERE, in draw_prim, not at the push_gx_draw skip site: a prim that merges into
  // the previous draw returns before ever reaching push_gx_draw, so a skip down there silently
  // misses the merged majority. The stream is still consumed (pos advanced) — a skip must drop
  // the DRAW, never desynchronize the FIFO parse.
  {
    static int s_init = 0;
    static bool s_on = false;
    static int s_mode = 1;
    if (!s_init) {
      s_init = 1;
      const char* e = std::getenv("SB_SKIP_COVERING");
      s_on = e != nullptr && e[0] != '\0';
      if (s_on) s_mode = std::atoi(e);
    }
    // =1 drops only draws whose box contains the point; =2 ALSO drops eye-plane-crossing
    // draws, whose box is not a valid coverage answer at all.
    //
    // SB_SKIP_WNEG_BF=<src>,<dst> narrows =2 to eye-crossing draws using that blend pair, so a
    // family proven guilty in bulk can be split by the state that decides what it writes.
    static int s_bfSrc = -1, s_bfDst = -1;
    static int s_bfInit = 0;
    if (!s_bfInit) {
      s_bfInit = 1;
      if (const char* e = std::getenv("SB_SKIP_WNEG_BF"); e != nullptr && e[0] != '\0')
        std::sscanf(e, "%d,%d", &s_bfSrc, &s_bfDst);
    }
    // The blend FACTORS are stale whenever blending is off — they keep whatever was last set.
    // Matching on them without checking blendMode silently sweeps in opaque draws, which is
    // how two supposedly disjoint blend-pair filters ended up selecting overlapping sets.
    const bool bfMatch = s_bfSrc < 0 || (g_gxState.blendMode == GX_BM_BLEND &&
                                         (int)g_gxState.blendFacSrc == s_bfSrc &&
                                         (int)g_gxState.blendFacDst == s_bfDst);

    // SB_SKIP_WNEG_KIND=partial|full selects which half of the crossing class to drop.
    // 'full' is the SHAM SURGERY control: those prims are clipped away by the GPU anyway, so
    // dropping them must be a visual no-op. If the image moves, the skip mechanism itself
    // perturbs unrelated rendering and every skip result here is contaminated.
    static int s_kindInit = 0;
    static int s_kind = 0;   // 0 = both, 1 = partial only, 2 = full only
    if (!s_kindInit) {
      s_kindInit = 1;
      if (const char* e = std::getenv("SB_SKIP_WNEG_KIND"); e != nullptr && e[0] != '\0')
        s_kind = (e[0] == 'p') ? 1 : (e[0] == 'f') ? 2 : 0;
    }
    const bool wnegMatch = s_kind == 1 ? sb_wneg_partial
                         : s_kind == 2 ? sb_wneg_full
                                       : (sb_wneg_partial || sb_wneg_full);
    // With a blend filter set the test is about THAT family alone, so the coverage clause is
    // dropped — otherwise the 393k harmless covering draws ride along and blur the answer.
    // SB_SKIP_BF=<src>,<dst> (diagnostic): drop EVERY draw using that blend pair, with no
    // eye-crossing precondition. Separates "this blend family paints it" from "the crossing
    // geometry paints it" — the two were confounded because the guilty family is mostly
    // crossing geometry.
    static int s_allBfInit = 0;
    static int s_allBfSrc = -1, s_allBfDst = -1;
    if (!s_allBfInit) {
      s_allBfInit = 1;
      if (const char* e = std::getenv("SB_SKIP_BF"); e != nullptr && e[0] != '\0')
        std::sscanf(e, "%d,%d", &s_allBfSrc, &s_allBfDst);
    }
    if (s_allBfSrc >= 0 && g_gxState.blendMode == GX_BM_BLEND &&
        (int)g_gxState.blendFacSrc == s_allBfSrc && (int)g_gxState.blendFacDst == s_allBfDst) {
      static long nbf = 0;
      if ((++nbf % 500) == 1) std::fprintf(stderr, "[skip-bf] dropped %ld draws\n", nbf);
      pos += totalVtxBytes;
      return;
    }

    // With a blend filter or a kind filter set the test is about THAT family alone, so the
    // coverage clause is dropped — otherwise the harmless covering draws ride along and blur
    // the answer.
    const bool coversClause = sb_covers_watch && s_bfSrc < 0 && s_kind == 0;
    if (s_on && (coversClause || (s_mode >= 2 && wnegMatch && bfMatch))) {
      static long n = 0;
      if ((++n % 100) == 1) std::fprintf(stderr, "[skip-covering] dropped %ld covering draws\n", n);
      pos += totalVtxBytes;
      if (dpProf) {
        ++g_dpEarlyReturns; // no phase claims this exit; it shows up as unattributed
      }
      return;
    }
  }

  // SB_NO_MERGE=1 (diagnostic): never merge draws. A merged draw reuses the
  // merge-head's uniform (array_start offsets) + bind groups; if the head's
  // indexed-array upload doesn't cover a later merged prim's indices, the GPU
  // fetches past the upload -> origin/garbage positions -> the geometry collapses
  // (candidate for the seagull zero-fragments defect). This isolates that.
  static int s_noMerge = -1;
  if (s_noMerge < 0) {
    const char* e = std::getenv("SB_NO_MERGE");
    s_noMerge = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
  }
  auto* lastDraw = (!g_gxState.stateDirty && s_noMerge == 0) ? gfx::get_last_draw_command<DrawData>() : nullptr;
  const bool canMerge = lastDraw != nullptr && prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS &&
                        lastDraw->instanceCount == 1;
  DP_PHASE(SB_DP_DIAG_POST);

  // Push raw vertex data to buffer. Merged draws must remain byte-contiguous with the previous range.
  gfx::Range vertRange = gfx::push_verts(data + pos, totalVtxBytes, canMerge ? 0 : 4);
  pos += totalVtxBytes;
  DP_PHASE(SB_DP_PUSHVERTS);

  // Try to merge with previous draw call
  if (canMerge) {
    // Only if the previous draw call was a single instance draw (no lines/points handling)
    u32 numIndices = 0;
    gfx::Range idxRange;
    const bool hadIndexRange = lastDraw->idxRange.size != 0;
    if (lastDraw->indexCount == 0 && prim != GX_TRIANGLES) {
      // Generate triangle index buffer for previous draw
      lastDraw->indexCount = prepare_idx_buffer(handle_draw_idx_buf, GX_TRIANGLES, 0, lastDraw->vtxCount);
    }
    if (lastDraw->indexCount != 0) {
      numIndices += prepare_idx_buffer(handle_draw_idx_buf, prim, lastDraw->vtxCount, vtxCount);
      idxRange = gfx::push_indices(handle_draw_idx_buf.data(), handle_draw_idx_buf.size(), hadIndexRange ? 0 : 4);
      handle_draw_idx_buf.clear();
    }
    CHECK(lastDraw->vertRange.offset + lastDraw->vertRange.size == vertRange.offset,
          "Non-consecutive vertex ranges ({} < {})", lastDraw->vertRange.offset + lastDraw->vertRange.size,
          vertRange.offset);
    if (hadIndexRange) {
      CHECK(lastDraw->idxRange.offset + lastDraw->idxRange.size == idxRange.offset,
            "Non-consecutive index ranges ({} < {})", lastDraw->idxRange.offset + lastDraw->idxRange.size,
            idxRange.offset);
    }
    lastDraw->vertRange.size += vertRange.size;
    if (lastDraw->idxRange.size == 0) {
      lastDraw->idxRange = idxRange;
    } else {
      lastDraw->idxRange.size += idxRange.size;
    }
    lastDraw->vtxCount += vtxCount;
    lastDraw->indexCount += numIndices;
    ++gfx::g_mergedDrawCallCount;
    DP_PHASE(SB_DP_MERGEIDX);
    if (dpProf) {
      ++g_dpMergedCalls;
    }
    return;
  }

  const uint64_t dpUnmT0 = dpProf ? dpTick : 0;
  handle_draw_unmerged(prim, fmt, vtxCount, vertRange);
  DP_PHASE(SB_DP_UNMERGED);
  if (dpProf) {
    ++g_dpUnmergedCalls;
    const uint64_t d = dpTick - dpUnmT0;
    if (g_dpUnmergedSampleCount < (long)(sizeof(g_dpUnmergedSamples) / sizeof(g_dpUnmergedSamples[0]))) {
      g_dpUnmergedSamples[g_dpUnmergedSampleCount++] = (uint32_t)(d > 0xFFFFFFFFull ? 0xFFFFFFFFull : d);
    } else {
      ++g_dpUnmergedSampleDropped; // reported, so a truncated distribution never reads as complete
    }
  }
}
#undef DP_PHASE

static void handle_draw(u8 cmd, const u8* data, u32& pos, u32 size, bool bigEndian) {
  GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
  GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
  const u32 cmdPos = pos - 1;

  CHECK(pos + 2 <= size, "draw vtxCount read overrun");
  u16 vtxCount = read_u16(data + pos, bigEndian);
  pos += 2;

  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt) vtxSize = g_gxState.lastVtxSize;
  else vtxSize = calculate_last_vtx_size(fmt);

  // PER-DRAW STATE ORACLE (sms-recomp/runtime/state_oracle.h). Aurora renders this stream
  // correctly, so its state at each draw is the reference the native path is checked against.
  // Weak symbols: aurora still links standalone, where these do nothing.
  if (sbr_state_diff_enabled != nullptr && sbr_state_oracle_aurora_raw != nullptr &&
      sbr_state_diff_enabled()) {
    // Aurora processes one contiguous buffer per frame, so a position that goes BACKWARDS is the
    // start of the next frame's buffer — the only frame boundary visible from inside this layer.
    static thread_local u32 s_lastCmdPos = 0;
    if (cmdPos < s_lastCmdPos && sbr_state_oracle_aurora_frame_end != nullptr)
      sbr_state_oracle_aurora_frame_end();
    s_lastCmdPos = cmdPos;
    unsigned char texmap[16]{}, texcoord[16]{}, texEnable[16]{};
    unsigned unitId[8]{};
    // The pixel-state block, reconstructed into the same RAW hardware encodings the recomp's
    // sbr_draw_state_fill packs (state_oracle.h documents the layouts), so the comparison is
    // encoding-for-encoding rather than enum-for-enum.
    unsigned char rasChannel[16]{};
    unsigned cWord[16]{}, aWord[16]{};
    unsigned short kSel[16]{};
    unsigned short chanCtrl[4]{};
    unsigned ambColor[2]{}, matColor[2]{}, konst[4]{};
    unsigned long long tevReg[4]{};
    const auto q8 = [](const Vec4<float>& v) {
      unsigned r = 0;
      for (int c = 0; c < 4; ++c) {
        long x = std::lround((double)v[c] * 255.0);
        if (x < 0) x = 0;
        if (x > 255) x = 255;
        r = (r << 8) | (unsigned)x;
      }
      return r;
    };
    const auto rawOp = [](const TevOp& o, unsigned& bias, unsigned& sub, unsigned& scale) {
      if (static_cast<unsigned>(o.op) >= 8) {   // compare mode: bias field is 3 on the wire
        bias = 3;
        sub = (static_cast<unsigned>(o.op) - 8) & 1;
        scale = ((static_cast<unsigned>(o.op) - 8) >> 1) & 3;
      } else {
        bias = static_cast<unsigned>(o.bias);
        sub = static_cast<unsigned>(o.op) & 1;
        scale = static_cast<unsigned>(o.scale);
      }
    };
    for (u32 k = 0; k < 16 && k < g_gxState.tevStages.size(); ++k) {
      const auto& ts = g_gxState.tevStages[k];
      const bool enabled = ts.texMapId != GX_TEXMAP_NULL &&
                           (static_cast<u32>(ts.texMapId) & 0x100u) == 0;
      texEnable[k] = enabled ? 1 : 0;
      texmap[k] = static_cast<unsigned char>(static_cast<u32>(ts.texMapId) & 7);
      texcoord[k] = static_cast<unsigned char>(static_cast<u32>(ts.texCoordId) & 7);
      switch (ts.channelId) {   // canonical hw ras values {0,1,5,6,7}
      case GX_COLOR0A0: rasChannel[k] = 0; break;
      case GX_COLOR1A1: rasChannel[k] = 1; break;
      case GX_ALPHA_BUMP: rasChannel[k] = 5; break;
      case GX_ALPHA_BUMPN: rasChannel[k] = 6; break;
      case GX_COLOR_ZERO: rasChannel[k] = 7; break;
      default: rasChannel[k] = 7; break;
      }
      unsigned bias, sub, scale;
      rawOp(ts.colorOp, bias, sub, scale);
      cWord[k] = static_cast<unsigned>(ts.colorPass.d) | static_cast<unsigned>(ts.colorPass.c) << 4 |
                 static_cast<unsigned>(ts.colorPass.b) << 8 |
                 static_cast<unsigned>(ts.colorPass.a) << 12 | bias << 16 | sub << 18 |
                 (ts.colorOp.clamp ? 1u : 0u) << 19 | scale << 20 |
                 static_cast<unsigned>(ts.colorOp.outReg) << 22;
      rawOp(ts.alphaOp, bias, sub, scale);
      aWord[k] = static_cast<unsigned>(ts.alphaPass.d) << 4 |
                 static_cast<unsigned>(ts.alphaPass.c) << 7 |
                 static_cast<unsigned>(ts.alphaPass.b) << 10 |
                 static_cast<unsigned>(ts.alphaPass.a) << 13 | bias << 16 | sub << 18 |
                 (ts.alphaOp.clamp ? 1u : 0u) << 19 | scale << 20 |
                 static_cast<unsigned>(ts.alphaOp.outReg) << 22;
      kSel[k] = static_cast<unsigned short>(static_cast<unsigned>(ts.kcSel) |
                                            static_cast<unsigned>(ts.kaSel) << 8);
    }
    // Report the BP-REGISTER image base, not the SDK texObj slot. `textures[m].texObj` is set by
    // GXLoadTexObj, which J3D almost never calls — it binds by replaying display lists that write
    // TX_SETIMAGE3 directly — so that slot is stale by construction and comparing it against the
    // recomp's BP-derived address compares two different quantities. That mismatch reads exactly
    // like "aurora holds the previous texture" and sent an investigation after a bind-timing bug
    // that does not exist. Same encoding as the recomp side: image3 bits 0-23 are the base in
    // 32-byte units, masked to the physical address it packs.
    for (u32 m = 0; m < 8 && m < g_gxState.loadedTextures.size(); ++m)
      unitId[m] = ((g_gxState.loadedTextures[m].image3 & 0x00FFFFFFu) << 5) & 0x01FFFFFFu;
    for (u32 c = 0; c < 4 && c < MaxColorChannels; ++c) {
      const auto& cfg = g_gxState.colorChannelConfig[c];
      const unsigned attnFn = cfg.attnFn == GX_AF_NONE ? 0u : (cfg.attnFn == GX_AF_SPEC ? 1u : 2u);
      const unsigned mask =
          static_cast<unsigned>(g_gxState.colorChannelState[c].lightMask.to_ulong() & 0xFFu);
      chanCtrl[c] = static_cast<unsigned short>(
          (cfg.matSrc == GX_SRC_VTX ? 1u : 0u) | (cfg.lightingEnabled ? 2u : 0u) |
          (cfg.ambSrc == GX_SRC_VTX ? 4u : 0u) | ((static_cast<unsigned>(cfg.diffFn) & 3u) << 3) |
          (attnFn << 5) | (mask << 8));
    }
    for (u32 c = 0; c < 2; ++c) {
      ambColor[c] = q8(g_gxState.colorChannelState[c].ambColor);
      matColor[c] = q8(g_gxState.colorChannelState[c].matColor);
    }
    for (u32 j = 0; j < 4; ++j) {
      konst[j] = q8(g_gxState.kcolors[j]);
      unsigned long long r = 0;
      for (int c = 0; c < 4; ++c)
        r = (r << 16) |
            (unsigned short)(short)std::lround((double)g_gxState.colorRegs[j][c] * 255.0);
      tevReg[j] = r;
    }
    // Raster state, packed identically to the recomp side (state_oracle.h documents the layout).
    // This is the state that decides whether a draw COVERS what is behind it, and it was never
    // part of the comparison.
    const unsigned rasterBits = (unsigned)(g_gxState.depthCompare ? 1u : 0u) |
                                ((unsigned)(g_gxState.depthUpdate ? 1u : 0u) << 1) |
                                (((unsigned)g_gxState.depthFunc & 7u) << 2);
    const unsigned blendMode = g_gxState.blendMode == GX_BM_BLEND      ? 1u
                               : g_gxState.blendMode == GX_BM_LOGIC    ? 2u
                               : g_gxState.blendMode == GX_BM_SUBTRACT ? 3u
                                                                       : 0u;
    const unsigned blendBits = blendMode | (((unsigned)g_gxState.blendFacSrc & 15u) << 3) |
                               (((unsigned)g_gxState.blendFacDst & 15u) << 7);
    // SCISSOR and CULL — the two pieces of per-draw raster state the oracle never carried. Aurora
    // confines every draw to this rect (BP 0x20/0x21) and applies GENMODE's cull mode; the recomp
    // renderer does neither, so comparing them tells us whether the hardware is CLIPPING draws that
    // this port paints across the whole target.
    const int scissorRect[4] = {g_gxState.logicalScissor.x, g_gxState.logicalScissor.y,
                                g_gxState.logicalScissor.width, g_gxState.logicalScissor.height};
    sbr_state_oracle_aurora_raw(cmdPos, g_gxState.numTevStages, g_gxState.numTexGens, texmap,
                                texcoord, texEnable, unitId, g_gxState.numChans, chanCtrl,
                                ambColor, matColor, rasChannel, cWord, aWord, kSel, konst, tevReg,
                                rasterBits, blendBits, scissorRect,
                                (unsigned)g_gxState.cullMode);
  }

  s_recentDraws[s_recentDrawHead] = {cmdPos, cmd, vtxCount, vtxSize};
  s_recentDrawHead = (s_recentDrawHead + 1) % kRecentDrawN;

  static thread_local int s_traceEnabled = -1;
  if (s_traceEnabled < 0) {
    const char* env = std::getenv("AURORA_DRAW_TRACE");
    s_traceEnabled = env && env[0] && env[0] != '0' ? 1 : 0;
  }
  if (s_traceEnabled) {
    Log.warn("[draw trace] pos={} cmd=0x{:02X} prim=0x{:02X} fmt={} vtxCount={} vtxSize={} totalBytes={} nextPos={}",
             cmdPos, cmd, static_cast<u32>(prim), static_cast<u32>(fmt),
             vtxCount, vtxSize, vtxCount * vtxSize, pos + vtxCount * vtxSize);
  }

  draw_prim(prim, fmt, vtxCount, data, pos, size);
}

static ByteBuffer handle_draw_unmerged_idxBuf;

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange) {
  ZoneScoped;
  u32 numIndices = 0;
  gfx::Range idxRange;

  if (prim != GX_TRIANGLES) {
    ZoneScopedN("build idx buffer");
    auto& realBuf = handle_draw_unmerged_idxBuf;
    numIndices = prepare_idx_buffer(realBuf, prim, 0, vtxCount);
    idxRange = gfx::push_indices(realBuf.data(), realBuf.size(), 4);
    realBuf.clear();
  }

  push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, numIndices);
}

// Identity of the most recent draws, for the staging-overflow fatal in gfx/common.hpp (it names
// the runaway upload instead of leaving it anonymous, and prints a ring so one bad draw is
// distinguishable from death-by-1000).
//
// RECORDED, NOT FORMATTED. This used to snprintf a 160-char line per draw, plus a second snprintf
// to copy it into the ring. Measured at ~500 ns per draw — 10.5% of the whole per-draw path — to
// produce text that is read only when the fatal fires, which in a healthy run is never.
//
// The fields are captured instead and formatted on demand in sb_last_draw_desc(). The diagnostic
// is unchanged: same fields, same ring depth, same OVERFLOWED marker. The marker string is COPIED
// (bounded) rather than pointed at, because g_sbLastMarker is reassigned as the frame proceeds and
// a pointer would render whatever the marker happened to be at fatal time, not at draw time.
struct DrawDescRec {
  bool used;
  unsigned prim;
  int fmt;
  unsigned vtxCount;
  unsigned numIndices;
  unsigned vertBytes;
  long drawIdx;
  char mark[72];
};
static DrawDescRec s_lastDrawRec{};
static DrawDescRec s_drawDescRing[16]{};
static unsigned s_drawDescRingPos = 0;

static void sb_record_draw_desc(unsigned prim, int fmt, unsigned vtxCount, unsigned numIndices,
                                unsigned vertBytes, const std::string& mark, long drawIdx) {
  DrawDescRec& r = s_lastDrawRec;
  r.used = true;
  r.prim = prim;
  r.fmt = fmt;
  r.vtxCount = vtxCount;
  r.numIndices = numIndices;
  r.vertBytes = vertBytes;
  r.drawIdx = drawIdx;
  const size_t n = mark.size() < sizeof(r.mark) - 1 ? mark.size() : sizeof(r.mark) - 1;
  std::memcpy(r.mark, mark.data(), n);
  r.mark[n] = '\0';
  s_drawDescRing[s_drawDescRingPos] = r;
  s_drawDescRingPos = (s_drawDescRingPos + 1) % 16;
}

static int sb_format_draw_desc(char* w, size_t cap, const DrawDescRec& r, const char* suffix) {
  return std::snprintf(w, cap, "\n  prim=0x%02x fmt=%d verts=%u idx=%u vertBytes=%u mark='%s' drawIdx=%ld%s",
                       r.prim, r.fmt, r.vtxCount, r.numIndices, r.vertBytes, r.mark, r.drawIdx,
                       suffix);
}

const char* sb_last_draw_desc()
{
  static char all[16 * 200];
  char* w = all;
  *w = '\0';
  for (unsigned i = 0; i < 16; ++i) {
    const DrawDescRec& r = s_drawDescRing[(s_drawDescRingPos + i) % 16];
    if (r.used) {
      w += sb_format_draw_desc(w, 200, r, "");
    }
  }
  if (s_lastDrawRec.used) {
    sb_format_draw_desc(w, 200, s_lastDrawRec, " <- OVERFLOWED");
  } else {
    // Never say nothing. An empty report here is indistinguishable from "no draws were recorded",
    // which is itself the interesting case when the overflow happens before the first draw.
    std::snprintf(w, 200, "\n  (no draws recorded before the overflow)");
  }
  return all;
}

uint64_t g_dpDescTicks = 0;

static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices) {
  const bool dpProf = sb_drawprim_profile();
  const uint64_t dpDesc0 = dpProf ? sb_tsc() : 0;
  sb_record_draw_desc((unsigned)prim, (int)fmt, (unsigned)vtxCount, (unsigned)numIndices,
                      (unsigned)vertRange.size, g_sbLastMarker, (long)g_sbPushedDrawCount);
  if (dpProf) {
    g_dpDescTicks += sb_tsc() - dpDesc0;
  }
  // GX_CULL_ALL culls both faces: the draw produces no fragments at all. WebGPU has no
  // equivalent rasterizer state, so it cannot be expressed in the pipeline — drop the draw
  // here instead, which is what the hardware does with it.
  //
  // Reached via the FIFO path with masked BP writes: a genMode write that sets only bit 15,
  // merged with a cached bit 14, yields cull 3 without any raw write containing 3. Before
  // this, to_primitive_state FATAL'd on it.
  if (g_gxState.cullMode == GX_CULL_ALL) {
    // Dropping a draw is invisible by construction, so count it: a runtime whose cull state
    // is wrong loses geometry silently and the result looks like "the game never drew it".
    // SB_CULL_STATS=1 reports the running total per frame ordinal.
    static const bool s_stats = std::getenv("SB_CULL_STATS") != nullptr;
    if (s_stats) {
      static long n = 0, lastFrame = -1;
      ++n;
      const long frame = static_cast<long>(sb_gx_vi_retrace_count());
      if (frame != lastFrame && frame % 200 == 0) {
        lastFrame = frame;
        std::fprintf(stderr, "[cull-all] %ld draws dropped by frame %ld\n", n, frame);
      }
    }
    return;
  }

  // SB_SKIP_ADD_OVERLAY=1 (diagnostic): drop the single additive orthographic quad that a
  // dump diff showed exists in this runtime and NOT in the reference one — orthographic,
  // 4 vertices, destination factor ONE (additive), alpha writes off. That combination is
  // unique in the frame, so this isolates one draw without needing a stable draw index.
  {
    static int s_init = 0;
    static bool s_on = false;
    if (!s_init) { s_init = 1; s_on = std::getenv("SB_SKIP_ADD_OVERLAY") != nullptr; }
    if (s_on && vtxCount == 4 && g_gxState.projType == GX_ORTHOGRAPHIC &&
        g_gxState.blendFacDst == GX_BL_ONE && !g_gxState.alphaUpdate) {
      return;
    }
  }

  // SB_SKIP_OPAQUE_P4=1 (diagnostic): drop OPAQUE perspective 4-vertex quads (blending off).
  // Neither previous targeted skip could touch this family, and an opaque quad partially
  // behind the eye plane rasterizes as a large smear — the one mechanism that both buries the
  // sea regardless of alpha and evades a bounding-box coverage test.
  {
    static int s_init = 0;
    static bool s_on = false;
    if (!s_init) { s_init = 1; s_on = std::getenv("SB_SKIP_OPAQUE_P4") != nullptr; }
    if (s_on && vtxCount == 4 && g_gxState.projType != GX_ORTHOGRAPHIC &&
        g_gxState.blendMode == GX_BM_NONE) {
      static long n = 0;
      if ((++n % 200) == 1) std::fprintf(stderr, "[skip-opaque-p4] skipped %ld draws\n", n);
      return;
    }
  }

  // SB_SKIP_SCENEQUAD=1 (diagnostic): drop the scene-covering PERSPECTIVE 4-vertex quad —
  // additive both factors (ONE/ONE), colour writes off, alpha writes on. The pixel-attribution
  // harness shows only two 4-vertex draws cover the washed pixel, and this is the other one.
  {
    static int s_init = 0;
    static bool s_on = false;
    if (!s_init) { s_init = 1; s_on = std::getenv("SB_SKIP_SCENEQUAD") != nullptr; }
    if (s_on && vtxCount == 4 && g_gxState.projType != GX_ORTHOGRAPHIC &&
        g_gxState.blendFacSrc == GX_BL_ONE && g_gxState.blendFacDst == GX_BL_ONE &&
        !g_gxState.colorUpdate) {
      static long n = 0;
      if ((++n % 200) == 1) std::fprintf(stderr, "[skip-scenequad] skipped %ld draws\n", n);
      return;
    }
  }

  // SB_SKIP_FADER=1 (diagnostic): drop the untextured full-screen orthographic quad — no
  // texgens, TEXMAP_NULL, GX_PASSCLR — which is the screen fader's fill_rect signature.
  {
    static int s_init = 0;
    static bool s_on = false;
    if (!s_init) { s_init = 1; s_on = std::getenv("SB_SKIP_FADER") != nullptr; }
    if (s_on && vtxCount == 4 && g_gxState.projType == GX_ORTHOGRAPHIC &&
        g_gxState.numTexGens == 0) {
      // A skip that matches nothing produces a null result indistinguishable from "this draw
      // is not the cause". Count and report, so the null can be told apart from the no-op.
      static long n = 0;
      if ((++n % 200) == 1) std::fprintf(stderr, "[skip-fader] skipped %ld draws\n", n);
      return;
    }
  }

  // SB_SKIP_VERTS=<n>[,<n>...] (diagnostic): drop draws with exactly those vertex counts.
  // A draw that writes only alpha (colorUpdate off) is invisible on its own, so its effect on
  // the frame can only be established by removing it and seeing what changes.
  {
    static int s_init = 0;
    static const char* s_want = nullptr;
    if (!s_init) { s_init = 1; s_want = std::getenv("SB_SKIP_VERTS"); }
    if (s_want != nullptr && s_want[0] != '\0') {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%u", vtxCount);
      const std::string_view want(s_want);
      size_t start = 0;
      while (start <= want.size()) {
        const size_t comma = want.find(',', start);
        const auto tok = want.substr(start, comma == std::string_view::npos
                                                ? std::string_view::npos : comma - start);
        if (!tok.empty() && tok == buf) return;
        if (comma == std::string_view::npos) break;
        start = comma + 1;
      }
    }
  }

  // SB_SKIP_TEX=<W>x<H>[,...] (diagnostic): drop draws binding a texture of those dimensions
  // on texmap 0. Ungated, unlike SB_SKIP_TEXDIM which only applies to draws whose marker
  // contains "Sky" — a marker the FIFO path does not set at all.
  {
    static int s_init = 0;
    static const char* s_want = nullptr;
    if (!s_init) { s_init = 1; s_want = std::getenv("SB_SKIP_TEX"); }
    if (s_want != nullptr && s_want[0] != '\0') {
      const auto& t0 = g_gxState.textures[0].texObj;
      char buf[24];
      std::snprintf(buf, sizeof(buf), "%ux%u", t0.width(), t0.height());
      const std::string_view want(s_want);
      size_t start = 0;
      while (start <= want.size()) {
        const size_t comma = want.find(',', start);
        const auto tok = want.substr(start, comma == std::string_view::npos
                                                ? std::string_view::npos : comma - start);
        if (!tok.empty() && tok == buf) return;
        if (comma == std::string_view::npos) break;
        start = comma + 1;
      }
    }
  }

  // Per-drain draw/vertex tally for SB_DRAW_STATS (reported from fifo::drain).
  detail::sDrainDraws += 1;
  detail::sDrainVerts += vtxCount;
  ++g_sbPushedDrawCount; // see decl above draw_prim: SB_NDC_DRAW window alignment
  // SB_ONLY_DRAW=<lo>[:<hi>] (diagnostic): drop every draw OUTSIDE the given
  // post-merge index window — isolates one suspect draw against the clear
  // color (a vanishing draw either appears alone => killed by earlier frame
  // state; or stays gone => its own GPU path is broken). Stream parsing and
  // state tracking are untouched; only the GPU push is skipped.
  {
    static long s_onlyLo = -2, s_onlyHi = -2;
    if (s_onlyLo == -2) {
      s_onlyLo = -1; s_onlyHi = -1;
      if (const char* w = std::getenv("SB_ONLY_DRAW"); w != nullptr && w[0] != '\0') {
        char* endp = nullptr;
        s_onlyLo = std::strtol(w, &endp, 0);
        s_onlyHi = (endp != nullptr && *endp == ':') ? std::strtol(endp + 1, nullptr, 0) : s_onlyLo;
      }
    }
    if (s_onlyLo >= 0 && (g_sbPushedDrawCount < s_onlyLo || g_sbPushedDrawCount > s_onlyHi)) {
      return;
    }
  }
  // SB_LIGHT_DBG=1: dump the ch0 lighting setup (diffFn/attnFn/mask/amb/mat) + each
  // active light's color/pos/attn for LIT high-vertex draws (Mario's skinned strips
  // are ~20-23 verts, lit). Used to find the ~2x dynamic-diffuse over-brighten on Mario.
  {
    static const bool s_lightDbg = std::getenv("SB_LIGHT_DBG") != nullptr;
    if (s_lightDbg) {
      const auto& lc = g_gxState.colorChannelConfig[0];
      const auto& ls = g_gxState.colorChannelState[0];
      if (lc.lightingEnabled && vtxCount >= 15) {
        static int s_ln = 0;
        if (s_ln++ < 16) {
          std::fprintf(stderr,
              "[light-dbg] verts=%u diffFn=%d attnFn=%d matSrc=%d ambSrc=%d mask=%02x amb=(%.2f,%.2f,%.2f) mat=(%.2f,%.2f,%.2f)\n",
              vtxCount, static_cast<int>(lc.diffFn), static_cast<int>(lc.attnFn), static_cast<int>(lc.matSrc),
              static_cast<int>(lc.ambSrc), static_cast<unsigned>(ls.lightMask.to_ulong() & 0xff),
              ls.ambColor.x(), ls.ambColor.y(), ls.ambColor.z(), ls.matColor.x(), ls.matColor.y(), ls.matColor.z());
          for (u32 i = 0; i < GX::MaxLights; ++i) {
            if (!ls.lightMask.test(i)) continue;
            const auto& L = g_gxState.lights[i];
            std::fprintf(stderr,
                "   L%u color=(%.3f,%.3f,%.3f) pos=(%.0f,%.0f,%.0f) dir=(%.2f,%.2f,%.2f) cosAtt=(%.3f,%.3f,%.3f) distAtt=(%.4f,%.4f,%.4f)\n",
                i, L.color.x(), L.color.y(), L.color.z(), L.pos.x(), L.pos.y(), L.pos.z(),
                L.dir.x(), L.dir.y(), L.dir.z(), L.cosAtt.x(), L.cosAtt.y(), L.cosAtt.z(),
                L.distAtt.x(), L.distAtt.y(), L.distAtt.z());
          }
        }
      }
    }
  }
  // SB_DRAW_DUMP=1: one-shot per-draw identity dump for the first drain past
  // draw #200 — prim/verts/texture/position-matrix translation, enough to
  // recognize which shapes the frame contains (e.g. the 752-vert sky dome).
  //
  // SB_DRAW_DUMP_AFTER=<retraceCount> (2026-07-10, title-backdrop-black probe):
  // the draw-index heuristic below (~160 draws/frame) undercounts badly once a
  // marker's packet-per-strip breakdown is this fine (DrawBuf MapOpa alone is
  // 200+ packets at title) — start/duration in draw-index space lands on the
  // wrong present entirely. Gate on VIGetRetraceCount instead (same counter
  // SB_DUMP_FRAME_AFTER / SB_NDC_PROBE_AFTER use) so this can target the exact
  // dumped present directly.
  // SB_DRAW_DUMP_FRAME=<retraceCount>: dump EVERY draw of exactly ONE frame —
  // the first present whose VIGetRetraceCount() has reached the target — with
  // no 200-draw cap (the fix for the draw-count-window/200-cap undercounting
  // whole late-boot frames; DrawBuf MapOpa alone is 200+ packets at title, so
  // the old windows landed on the wrong present or truncated mid-frame).
  // Retrace is incremented AFTER a frame's draws drain in sb_frame_present, so
  // every draw of "frame N" emits while VIGetRetraceCount()==N. NOTE: retrace
  // does NOT advance by 1 per present — sb_frame_present adds `retraces` (often
  // 2+ NTSC fields) each time, so it routinely JUMPS OVER an exact target (a
  // `== target` test then never fires — the 2026-07-10 "0 draws dumped despite
  // retrace passing 1000" bug). Latch onto the first retrace value >= target
  // and dump only that latched frame — robust to the step size.
  //
  // s_ddFrameActive: true ONLY during the SB_DRAW_DUMP_FRAME target frame's draws.
  // Shared with the SB_DRAW_DUMP block below so `SB_DRAW_DUMP=1 SB_DRAW_DUMP_FRAME=<N>`
  // emits the FULL ch0 [draw-dump] line for exactly frame N (uncapped, ONE frame) —
  // the previous s_fullFrame path uncapped but then firehosed EVERY frame after the
  // retrace threshold, so a clean full-frame render-state dump (for draw_diff) wasn't
  // possible. Declared at function scope so both blocks see it.
  static bool s_ddFrameActive = false;
  static const char* const s_ddFrameEnv = std::getenv("SB_DRAW_DUMP_FRAME");
  if (const char* fe = s_ddFrameEnv; fe != nullptr && fe[0] != '\0') {
    // SB_DRAW_DUMP_FRAME=<N>: dump every draw of the Nth RENDERED FRAME. "Frame"
    // is counted by retrace-value CHANGES, not by an absolute retrace target:
    // VIGetRetraceCount neither steps by 1 nor advances at a fixed rate vs
    // presents at the title (it can jump 2+ per present and stall during load
    // loops), so both `== target` and `>= target` on the absolute value are
    // unreliable (the 2026-07-10 "0 draws dumped" bug — retrace never landed on
    // 1000/3000). Counting distinct retrace values gives a robust ordinal frame
    // index regardless of step size or rate.
    static long s_targetFrame = -2;
    static long s_frameIdx = -1;       // # of retrace changes seen so far
    static long s_prevRetrace = -1;
    if (s_targetFrame == -2) s_targetFrame = std::atol(fe);
    const long rc = static_cast<long>(sb_gx_vi_retrace_count());
    if (rc != s_prevRetrace) { s_prevRetrace = rc; ++s_frameIdx; }
    s_ddFrameActive = (s_frameIdx == s_targetFrame);
    if (s_frameIdx == s_targetFrame) {
      static long s_frameDumped = 0;
      const auto& vp = g_gxState.logicalViewport;
      std::fprintf(stderr, "[draw-dump-frame] #%ld frame=%ld retrace=%ld prim=%u verts=%u proj=%c vp=%.0fx%.0f bm=%d sf=%d df=%d cU=%d aU=%d mark='%s'\n",
                   s_frameDumped++, s_targetFrame, rc, static_cast<unsigned>(prim), vtxCount,
                   g_gxState.projType == GX_ORTHOGRAPHIC ? 'O' : 'P', vp.width, vp.height,
                   static_cast<int>(g_gxState.blendMode), static_cast<int>(g_gxState.blendFacSrc),
                   static_cast<int>(g_gxState.blendFacDst), g_gxState.colorUpdate ? 1 : 0,
                   g_gxState.alphaUpdate ? 1 : 0, g_sbLastMarker.c_str());
    }
  }
  static const char* const s_ddEnv = std::getenv("SB_DRAW_DUMP");
  if (const char* e = s_ddEnv; e != nullptr) {
    // SB_DRAW_DUMP=<startDraw>: dump 200 draws starting at that global draw
    // index (draw counts run ~160/frame at title; pick start = frame*160).
    // SB_DRAW_DUMP=0 dumps from the very first draw.
    static int s_dumped = 0;
    static int s_start = -1;
    static long s_afterRetrace = -1;
    static int s_windowDumped = 0; // draws emitted since the retrace window opened
    static bool s_fullFrame = false; // SB_DRAW_DUMP_FRAME set: no 200-draw cap
    if (s_start < 0) {
      s_start = std::atoi(e);
      if (const char* a = std::getenv("SB_DRAW_DUMP_AFTER"); a != nullptr && a[0] != '\0') {
        s_afterRetrace = std::atol(a);
      }
      s_fullFrame = std::getenv("SB_DRAW_DUMP_FRAME") != nullptr;
    }
    // SB_DRAW_DUMP_ALL=1: dump EVERY draw, uncapped, no frame gate. The only
    // frame-targeting scheme that works in BOTH boot and .dff-replay: replay
    // bypasses the game's waitForRetrace, so sb_gx_vi_retrace_count() never
    // advances and the retrace-ordinal SB_DRAW_DUMP_FRAME gate never fires
    // (0 draws dumped). This mode depends on no counter — for a short replay
    // (3 frames) it dumps each settled frame's full draw list, and every draw
    // is prefixed with the emitting frame's marker so consumers can split.
    static const bool s_dumpAll = std::getenv("SB_DRAW_DUMP_ALL") != nullptr;
    // SB_DRAW_DUMP_AFTER present: replace the draw-index window with "first 200
    // draws once the retrace counter clears the threshold" — the two schemes
    // are mutually exclusive (a present's absolute draw index is unrelated to
    // its retrace count once boot has run for a while).
    const bool windowed = s_afterRetrace >= 0;
    const bool afterWindowOk = !windowed || static_cast<long>(sb_gx_vi_retrace_count()) >= s_afterRetrace;
    // s_fullFrame (SB_DRAW_DUMP_FRAME set): dump the full ch0 line for exactly the
    // target frame's draws (gated by s_ddFrameActive) — uncapped, one frame. Else the
    // legacy 200-cap windowed / draw-index behavior.
    const bool inRange = s_dumpAll ? true
                       : s_fullFrame ? s_ddFrameActive
                       : windowed ? (afterWindowOk && s_windowDumped < 200)
                                   : (s_dumped >= s_start && s_dumped < s_start + 200);
    if (inRange) {
      if (windowed) ++s_windowDumped;
      const auto& obj = g_gxState.textures[0].texObj;
      const auto* pn = reinterpret_cast<const float*>(&g_gxState.pnMtx[g_gxState.currentPnMtx].pos);
      const auto& vp = g_gxState.logicalViewport;
      const auto& sc = g_gxState.logicalScissor;
      const auto& cc = g_gxState.colorChannelConfig[0];
      const auto& cs = g_gxState.colorChannelState[0];
      // ALPHA0 (GX_ALPHA0 == 2): independent chanCtrl/ambient/material state
      // from COLOR0 — same REG/lightMask machinery, but never dumped before.
      // A lit-but-near-black material with a bright COLOR0 (mat/amb both
      // ~white, per oracle ground-truth scratch/oracle/title_light_ground_truth.txt)
      // is consistent with the geometry being nearly TRANSPARENT rather than
      // dark: XLU blend reads ALPHA0's REG output, which this printf never
      // exposed, so a wrong ALPHA0 ambSrc/matSrc/lightMask/register value
      // could sink alpha near 0 and blend the map/sea to the black EFB clear
      // while COLOR0 alone looks perfectly correct. See
      // debug_journal/2026-07-07_title_backdrop_and_indexed_mtx.md
      // Continuation 16/17 for the standing "map/sea near-black" defect.
      const auto& cca = g_gxState.colorChannelConfig[GX_ALPHA0];
      const auto& csa = g_gxState.colorChannelState[GX_ALPHA0];
      const auto& posFmt = g_gxState.vtxFmts[fmt].attrs[GX_VA_POS];
      const auto posDesc = g_gxState.vtxDesc[GX_VA_POS];
      // ALL bound texmaps, not just tex0. A draw routinely samples several maps, and the
      // single tex0 field cannot answer "which draw samples texture X" — asking that of a
      // render-to-texture result (an EFB copy) and seeing no match on tex0 says nothing,
      // since the copy may well be bound to a higher map.
      // The TEV PROGRAM, not just the stage count. Two draws can both report tev=2 and
      // compute entirely different things; without the per-stage args and ops there is no way
      // to tell whether a suspect draw's shading matches a reference runtime's.
      char tevbuf[512];
      {
        int o = 0;
        tevbuf[0] = '\0';
        const unsigned nst = g_gxState.numTevStages < MaxTevStages ? g_gxState.numTevStages
                                                                   : MaxTevStages;
        for (unsigned st = 0; st < nst && o < static_cast<int>(sizeof(tevbuf)) - 72; ++st) {
          const auto& t = g_gxState.tevStages[st];
          o += std::snprintf(tevbuf + o, sizeof(tevbuf) - o,
                             "%s%u:c(%d,%d,%d,%d)o=%d,%d,%d,r%d a(%d,%d,%d,%d)o=%d,%d,%d,r%d "
                             "tm=%d tc=%d ch=%d k=%d/%d",
                             o ? " | " : "", st,
                             (int)t.colorPass.a, (int)t.colorPass.b, (int)t.colorPass.c,
                             (int)t.colorPass.d, (int)t.colorOp.op, (int)t.colorOp.bias,
                             (int)t.colorOp.scale, (int)t.colorOp.outReg,
                             (int)t.alphaPass.a, (int)t.alphaPass.b, (int)t.alphaPass.c,
                             (int)t.alphaPass.d, (int)t.alphaOp.op, (int)t.alphaOp.bias,
                             (int)t.alphaOp.scale, (int)t.alphaOp.outReg,
                             (int)t.texMapId, (int)t.texCoordId, (int)t.channelId,
                             (int)t.kcSel, (int)t.kaSel);
        }
      }
      // Texgen configuration. A stage naming a GENERATED texcoord (tc=N) says nothing about
      // WHERE it samples without this: the type, source, and matrices are what map a vertex
      // onto the texture, and a draw can match in every other respect while sampling an
      // entirely different part of the image.
      char tcgbuf[256];
      {
        int o = 0;
        tcgbuf[0] = '\0';
        const unsigned ntc = g_gxState.numTexGens < MaxTexCoord ? g_gxState.numTexGens
                                                                : MaxTexCoord;
        for (unsigned t = 0; t < ntc && o < static_cast<int>(sizeof(tcgbuf)) - 40; ++t) {
          const auto& c = g_gxState.tcgs[t];
          o += std::snprintf(tcgbuf + o, sizeof(tcgbuf) - o, "%s%u:ty=%d src=%d mtx=%d pm=%d n=%d",
                             o ? "," : "", t, (int)c.type, (int)c.src, (int)c.mtx,
                             (int)c.postMtx, c.normalize ? 1 : 0);
        }
      }
      char texbuf[256];
      {
        int o = 0;
        texbuf[0] = '\0';
        for (int m = 0; m < GX_MAX_TEXMAP && o < static_cast<int>(sizeof(texbuf)) - 16; ++m) {
          const auto& to = g_gxState.textures[m];
          if (!to.ref) continue;   // not bound on this map
          // Include the texel pointer: two textures of identical dimensions can be entirely
          // different resources (a raw-RAM texture vs an EFB-copy result), and dimensions
          // alone cannot tell them apart.
          o += std::snprintf(texbuf + o, sizeof(texbuf) - o, "%s%d:%ux%u@%p", o ? "," : "", m,
                             to.texObj.width(), to.texObj.height(), to.texObj.data);
        }
      }
      // clr0Desc/clr1Desc: whether this draw's VCD actually supplies CLR0/CLR1
      // (GX_NONE=0 -> vtx_attr() default-white fallback fires; GX_DIRECT=1/
      // GX_INDEX8=2/GX_INDEX16=3 -> real per-vertex color stream is bound).
      const auto clr0Desc = g_gxState.vtxDesc[GX_VA_CLR0];
      const auto clr1Desc = g_gxState.vtxDesc[GX_VA_CLR1];
      // Texcoord descriptors and formats. Whether a texcoord is DIRECT or INDEXED decides
      // which path supplies its values - inline vertex bytes, or an array base that has to be
      // registered separately - and the two fail in completely different ways.
      char tcvbuf[192];
      {
        int o = 0;
        tcvbuf[0] = '\0';
        for (int t = 0; t < 4 && o < static_cast<int>(sizeof(tcvbuf)) - 40; ++t) {
          const auto d = g_gxState.vtxDesc[GX_VA_TEX0 + t];
          if (d == GX_NONE) continue;
          const auto& a = g_gxState.vtxFmts[fmt].attrs[GX_VA_TEX0 + t];
          o += std::snprintf(tcvbuf + o, sizeof(tcvbuf) - o, "%st%d:d=%d cnt=%d ty=%d fr=%u",
                             o ? "," : "", t, (int)d, (int)a.cnt, (int)a.type, a.frac);
        }
      }
      std::fprintf(stderr,
                   "[draw-dump] #%d prim=%u verts=%u tex0=%ux%u texs=[%s] tevp=[%s] tcg=[%s] tcv=[%s] zcmp=%d zupd=%d trans=(%.1f,%.1f,%.1f) "
                   "proj=%c blend=%u vp=(%.0f,%.0f %.0fx%.0f) sc=(%d,%d %ux%u) "
                   "tev=%u ch0[light=%d matSrc=%d ambSrc=%d attnFn=%d diffFn=%d mat=(%.2f,%.2f,%.2f,%.2f) amb=(%.2f,%.2f,%.2f) mask=%02x] "
                   "a0[light=%d matSrc=%d ambSrc=%d mat=%.2f amb=%.2f mask=%02x] "
                   "prj=[%.4f %.4f %.4f %.4f cx=%.4f cy=%.4f] cU=%d aU=%d bm=%d bf=%d/%d pos[desc=%d cnt=%d type=%d frac=%u] clr0=%d clr1=%d mtxIdx=%u "
                   "cull=%d zfunc=%d acmp=[c0=%d r0=%u op=%d c1=%d r1=%u] "
                   "posmtx=[%.2f %.2f %.2f %.2f | %.2f %.2f %.2f %.2f | %.2f %.2f %.2f %.2f] mark='%s'\n",
                   s_dumped, static_cast<unsigned>(prim), vtxCount, obj.width(), obj.height(), texbuf, tevbuf, tcgbuf, tcvbuf,
                   static_cast<int>(g_gxState.depthCompare), static_cast<int>(g_gxState.depthUpdate),
                   pn[3], pn[7], pn[11], g_gxState.projType == GX_ORTHOGRAPHIC ? 'O' : 'P',
                   static_cast<unsigned>(g_gxState.blendMode), vp.left, vp.top, vp.width, vp.height,
                   sc.x, sc.y, sc.width, sc.height, g_gxState.numTevStages,
                   static_cast<int>(cc.lightingEnabled), static_cast<int>(cc.matSrc), static_cast<int>(cc.ambSrc),
                   static_cast<int>(cc.attnFn), static_cast<int>(cc.diffFn),
                   cs.matColor.x(),
                   cs.matColor.y(), cs.matColor.z(), cs.matColor.w(), cs.ambColor.x(), cs.ambColor.y(),
                   cs.ambColor.z(), static_cast<unsigned>(cs.lightMask.to_ulong() & 0xff),
                   static_cast<int>(cca.lightingEnabled), static_cast<int>(cca.matSrc), static_cast<int>(cca.ambSrc),
                   csa.matColor.w(), csa.ambColor.w(), static_cast<unsigned>(csa.lightMask.to_ulong() & 0xff),
                   reinterpret_cast<const float*>(&g_gxState.proj)[0],
                   reinterpret_cast<const float*>(&g_gxState.proj)[5],
                   reinterpret_cast<const float*>(&g_gxState.proj)[10],
                   reinterpret_cast<const float*>(&g_gxState.proj)[11],
                   reinterpret_cast<const float*>(&g_gxState.proj)[2],
                   reinterpret_cast<const float*>(&g_gxState.proj)[6],
                   g_gxState.colorUpdate ? 1 : 0, g_gxState.alphaUpdate ? 1 : 0,
                   static_cast<int>(g_gxState.blendMode), static_cast<int>(g_gxState.blendFacSrc),
                   static_cast<int>(g_gxState.blendFacDst), static_cast<int>(posDesc),
                   static_cast<int>(posFmt.cnt), static_cast<int>(posFmt.type), posFmt.frac,
                   static_cast<int>(clr0Desc), static_cast<int>(clr1Desc),
                   g_gxState.currentPnMtx,
                   static_cast<int>(g_gxState.cullMode),
                   static_cast<int>(g_gxState.depthFunc),
                   static_cast<int>(g_gxState.alphaCompare.comp0), g_gxState.alphaCompare.ref0,
                   static_cast<int>(g_gxState.alphaCompare.op),
                   static_cast<int>(g_gxState.alphaCompare.comp1), g_gxState.alphaCompare.ref1,
                   pn[0], pn[1], pn[2], pn[3], pn[4], pn[5], pn[6], pn[7],
                   pn[8], pn[9], pn[10], pn[11], g_sbLastMarker.c_str());
      // Active-light colors/positions for this draw (the fields the [draw-dump]
      // line above can't fit) — needed to compare a lit draw's SHADING between
      // native boot and .dff replay (e.g. the file-select Mario overalls washout,
      // once ambient/matColor/texdims were shown to match). Emitted for every
      // lit draw of the dumped frame; also dumps the KONST color regs.
      // COLOR1 (GX_COLOR1==1) + ALPHA1 (GX_ALPHA1==3): the SECOND lit channel,
      // never dumped before. Mario's material pulls channel-1 RASC into TEV
      // stage 4 (chan=5=COLOR1A1); if ch1 lighting is over-bright it blows the
      // final to white while ch0 looks fine. (2026-07-15 Mario paleness.)
      {
        const auto& c1 = g_gxState.colorChannelConfig[GX_COLOR1];
        const auto& s1 = g_gxState.colorChannelState[GX_COLOR1];
        const auto& c1a = g_gxState.colorChannelConfig[GX_ALPHA1];
        const auto& s1a = g_gxState.colorChannelState[GX_ALPHA1];
        std::fprintf(stderr,
                     "   [dd-ch1] #%d ch1[light=%d matSrc=%d ambSrc=%d attnFn=%d diffFn=%d mat=(%.2f,%.2f,%.2f) amb=(%.2f,%.2f,%.2f) mask=%02x] "
                     "a1[light=%d matSrc=%d ambSrc=%d mask=%02x]\n",
                     s_dumped, static_cast<int>(c1.lightingEnabled), static_cast<int>(c1.matSrc),
                     static_cast<int>(c1.ambSrc), static_cast<int>(c1.attnFn), static_cast<int>(c1.diffFn),
                     s1.matColor.x(), s1.matColor.y(), s1.matColor.z(), s1.ambColor.x(), s1.ambColor.y(),
                     s1.ambColor.z(), static_cast<unsigned>(s1.lightMask.to_ulong() & 0xff),
                     static_cast<int>(c1a.lightingEnabled), static_cast<int>(c1a.matSrc),
                     static_cast<int>(c1a.ambSrc), static_cast<unsigned>(s1a.lightMask.to_ulong() & 0xff));
      }
      // Dump lights referenced by EITHER color channel (ch0 mask=cs, ch1 mask):
      // Mario's ch1 uses L2 (mask=04) which the ch0-only loop below never showed.
      const auto ch1Mask = g_gxState.colorChannelState[GX_COLOR1].lightMask;
      if (cs.lightMask.any() || ch1Mask.any()) {
        for (u32 li = 0; li < GX::MaxLights; ++li) {
          if (!cs.lightMask.test(li) && !ch1Mask.test(li)) continue;
          const auto& L = g_gxState.lights[li];
          std::fprintf(stderr, "   [dd-light] #%d L%u col=(%.3f,%.3f,%.3f) pos=(%.0f,%.0f,%.0f) cosAtt=(%.3f,%.3f,%.3f) distAtt=(%.4f,%.4f,%.4f)\n",
                       s_dumped, li, L.color.x(), L.color.y(), L.color.z(), L.pos.x(), L.pos.y(), L.pos.z(),
                       L.cosAtt.x(), L.cosAtt.y(), L.cosAtt.z(), L.distAtt.x(), L.distAtt.y(), L.distAtt.z());
        }
        for (u32 k = 0; k < 4; ++k) {
          const auto& kc = g_gxState.kcolors[k];
          std::fprintf(stderr, "   [dd-konst] #%d K%u=(%.3f,%.3f,%.3f,%.3f)\n",
                       s_dumped, k, kc.x(), kc.y(), kc.z(), kc.w());
        }
      }
      // SB_LOG=pn: full pos/nrm matrix PALETTE (all 10 PnMtx slots) for this
      // same window. Skinned draws (GX_VA_PNMTXIDX per-vertex) index into
      // this palette; a replay carries retail's recorded XF loads = ground
      // truth, while live-native computes the palette in J3D — a per-slot
      // diff localizes wrong skinned normals (2026-07-16 file-select Mario
      // black-patch investigation) to the exact matrix slot.
      if (sb_gx_log_on("pn")) {
        const bool skinned = g_gxState.vtxDesc[GX_VA_PNMTXIDX] != GX_NONE;
        sb_logf("pn", "#%d skinned=%d cur=%u", s_dumped, skinned ? 1 : 0, g_gxState.currentPnMtx);
        for (u32 mi = 0; mi < MaxPnMtx; ++mi) {
          const auto* p = reinterpret_cast<const float*>(&g_gxState.pnMtx[mi].pos);
          const auto* n = reinterpret_cast<const float*>(&g_gxState.pnMtx[mi].nrm);
          sb_logf("pn", "pos #%d %u [%.4f %.4f %.4f %.2f | %.4f %.4f %.4f %.2f | %.4f %.4f %.4f %.2f]",
                  s_dumped, mi, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]);
          sb_logf("pn", "nrm #%d %u [%.4f %.4f %.4f | %.4f %.4f %.4f | %.4f %.4f %.4f]",
                  s_dumped, mi, n[0], n[1], n[2], n[4], n[5], n[6], n[8], n[9], n[10]);
        }
      }
      // SB_LOG=vtxarr: bound vertex-array identity per dumped draw — attr,
      // stride, endianness, and the first 16 bytes of the array. A replay
      // carries retail's recorded arrays; live-native binds game memory. A
      // content mismatch here (with identical draw STATE) localizes a
      // BE-swap/layout gap in a specific attribute stream (2026-07-16 Mario
      // black patches: suspect TEX1 marking-overlay UVs).
      if (sb_gx_log_on("vtxarr")) {
        for (u32 at = GX_VA_POS; at <= GX_VA_TEX7; ++at) {
          if (g_gxState.vtxDesc[at] != GX_INDEX8 && g_gxState.vtxDesc[at] != GX_INDEX16) continue;
          const auto& arr = g_gxState.arrays[at];
          if (arr.data == nullptr) continue;
          const u8* d = (const u8*)arr.data;
          // Full-array FNV hash, endianness-normalized to BE so a native
          // (host-LE) array and a replay (BE) array with the same VALUES hash
          // the same — a mismatch means content divergence ANYWHERE in the
          // array (e.g. a partially-bounded byteswap), not just the head.
          const u32 nbytes = arr.size != 0 ? arr.size : arr.sizeAuto;
          u64 hash = 1469598103934665603ull;
          if (nbytes != 0) {
            const u32 esz = (arr.stride >= 4 && (arr.stride % 4) == 0) ? 4u : 2u;
            for (u32 off = 0; off + esz <= nbytes; off += esz) {
              u8 b[4];
              for (u32 k = 0; k < esz; ++k) b[k] = d[off + (arr.le ? esz - 1 - k : k)];
              for (u32 k = 0; k < esz; ++k) { hash ^= b[k]; hash *= 1099511628211ull; }
            }
          }
          sb_logf("vtxarr", "#%d attr=%u stride=%u le=%d n=%u hash=%016llx first16=%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
                  s_dumped, at, arr.stride, arr.le ? 1 : 0, nbytes, (unsigned long long)hash,
                  d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                  d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
        }
      }
      // SB_LOG=texgen: per-coord texgen config + the referenced texmatrix rows
      // for this same window — the state dimension the [draw-dump]/[tev] lines
      // never covered (wrong texgen src/mtx moves an overlay texture across
      // the model: 2026-07-16 Mario marking-layer patch suspect).
      if (sb_gx_log_on("texgen")) {
        for (u32 tc = 0; tc < aurora::gx::MaxTexCoord; ++tc) {
          const auto& t = g_gxState.tcgs[tc];
          if (t.src == GX_MAX_TEXGENSRC) continue;
          sb_logf("texgen", "#%d coord=%u type=%d src=%d mtx=%d post=%d norm=%d",
                  s_dumped, tc, (int)t.type, (int)t.src, (int)t.mtx, (int)t.postMtx,
                  t.normalize ? 1 : 0);
          if (t.mtx != GX_IDENTITY) {
            const u32 mi = ((u32)t.mtx - GX_TEXMTX0) / 3;
            if (mi < aurora::gx::MaxTexMtx) {
              const float* m = reinterpret_cast<const float*>(&g_gxState.texMtxs[mi]);
              sb_logf("texgen", "#%d   texmtx%u=[%.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f]",
                      s_dumped, mi, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
            }
          }
        }
      }
      // SB_TEV_DUMP=1: full TEV combiner state + texture format for this same
      // window, to pin down the exact stage math on a specific overbright
      // draw (e.g. the title logo quad) once SB_DRAW_DUMP has located it.
      if (std::getenv("SB_TEV_DUMP") != nullptr) {
        for (unsigned st = 0; st < g_gxState.numTevStages; ++st) {
          const auto& s = g_gxState.tevStages[st];
          std::fprintf(stderr,
                       "  [tev] #%d stage=%u texMap=%d texCoord=%d chan=%d "
                       "colorPass(a=%d b=%d c=%d d=%d op=%d bias=%d scale=%d clamp_outReg=%d) "
                       "alphaPass(a=%d b=%d c=%d d=%d op=%d bias=%d scale=%d outReg=%d) "
                       "kcSel=%d kaSel=%d swapRas=%d swapTex=%d\n",
                       s_dumped - 1, st, static_cast<int>(s.texMapId), static_cast<int>(s.texCoordId),
                       static_cast<int>(s.channelId), static_cast<int>(s.colorPass.a),
                       static_cast<int>(s.colorPass.b), static_cast<int>(s.colorPass.c),
                       static_cast<int>(s.colorPass.d), static_cast<int>(s.colorOp.op),
                       static_cast<int>(s.colorOp.bias), static_cast<int>(s.colorOp.scale),
                       static_cast<int>(s.colorOp.outReg), static_cast<int>(s.alphaPass.a),
                       static_cast<int>(s.alphaPass.b), static_cast<int>(s.alphaPass.c),
                       static_cast<int>(s.alphaPass.d), static_cast<int>(s.alphaOp.op),
                       static_cast<int>(s.alphaOp.bias), static_cast<int>(s.alphaOp.scale),
                       static_cast<int>(s.alphaOp.outReg), static_cast<int>(s.kcSel),
                       static_cast<int>(s.kaSel), static_cast<int>(s.tevSwapRas),
                       static_cast<int>(s.tevSwapTex));
        }
        // Report ALL bound texmaps (0-7), not just texMap0: a multi-texmap
        // material (Mario overalls = 4 texmaps) can wash if any texmap binds a
        // wrong/empty texture. `hasTex` = a texture handle is actually bound.
        for (unsigned tm = 0; tm < aurora::gx::MaxTextures; ++tm) {
          const auto& tu = g_gxState.textures[tm];
          const auto& tobj = tu.texObj;
          std::fprintf(stderr, "  [tex] texMap=%u hasTex=%d fmt=%d %ux%u minFilt=%d magFilt=%d wrapS=%d wrapT=%d\n",
                       tm, tu.ref ? 1 : 0, static_cast<int>(tobj.format()),
                       tobj.width(), tobj.height(),
                       static_cast<int>(tobj.min_filter()), static_cast<int>(tobj.mag_filter()),
                       static_cast<int>(tobj.wrap_s()), static_cast<int>(tobj.wrap_t()));
        }
        for (unsigned r = 0; r < aurora::gx::MaxTevRegs; ++r) {
          const auto& c = g_gxState.colorRegs[r];
          std::fprintf(stderr, "  [tevreg] %u = (%.3f, %.3f, %.3f, %.3f)\n", r, c.x(), c.y(), c.z(), c.w());
        }
        // Swap-table CONTENTS (the per-stage swapRas/swapTex indices above
        // are meaningless without them).
        for (unsigned sw = 0; sw < aurora::gx::MaxTevSwap; ++sw) {
          const auto& t = g_gxState.tevSwapTable[sw];
          std::fprintf(stderr, "  [tevswap] %u = r%d g%d b%d a%d\n", sw, (int)t.red, (int)t.green,
                       (int)t.blue, (int)t.alpha);
        }
      }
    }
    ++s_dumped;
  }
  // SB_CLOUD_TC_DBG=1 (diagnostic, title sky crosshatch investigation): fire
  // whenever ANY bound texture in ANY tev stage is 8x8 (the cloud-puff
  // texture per scratch/oracle/cloud_ground_truth.txt), regardless of which
  // TDrawBufObj marker is active. Dumps that stage's texgen (src/mtx/type)
  // and the texture's actual wrap mode, to compare against the oracle's
  // "identity texgen, raw-UV passthrough, wrap Clamp/Clamp" ground truth.
  static const bool s_cloudTcDbg = std::getenv("SB_CLOUD_TC_DBG") != nullptr;
  if (s_cloudTcDbg) {
    static long s_nSky = 0, s_nOther = 0;
    bool isSky = g_sbLastMarker.find("Sky") != std::string::npos;
    if (isSky ? s_nSky < 4000 : s_nOther < 5) {
    for (unsigned st = 0; st < g_gxState.numTevStages; ++st) {
      const auto& s = g_gxState.tevStages[st];
      if (s.texMapId < 0 || static_cast<unsigned>(s.texMapId) >= aurora::gx::MaxTextures)
        continue;
      const auto& tobj = g_gxState.textures[s.texMapId].texObj;
      if (tobj.width() == 8 && tobj.height() == 8) {
        long& s_n = isSky ? s_nSky : s_nOther;
        ++s_n;
        unsigned tc = static_cast<unsigned>(s.texCoordId);
        const auto& tcg = (tc < aurora::gx::MaxTexCoord) ? g_gxState.tcgs[tc] : g_gxState.tcgs[0];
        std::fprintf(stderr,
                     "[cloud-tc] #%ld mark='%s' stage=%u texMap=%d texCoord=%u fmt=%d wrapS=%d wrapT=%d "
                     "numTexGens=%u tcg[src=%d mtx=%d type=%d post=%d norm=%d]\n",
                     s_n, g_sbLastMarker.c_str(), st, static_cast<int>(s.texMapId), tc,
                     static_cast<int>(tobj.format()), static_cast<int>(tobj.wrap_s()),
                     static_cast<int>(tobj.wrap_t()), g_gxState.numTexGens,
                     static_cast<int>(tcg.src), static_cast<int>(tcg.mtx), static_cast<int>(tcg.type),
                     static_cast<int>(tcg.postMtx), static_cast<int>(tcg.normalize));
      }
    }
    }
  }
  // TEMP crosshatch-hunt: dump EVERY texture bound on EVERY draw marked
  // "Sky Xlu" (not just 8x8) so we can see the 16x16/64x64 phantom texobjs
  // directly at draw time (bake-time inspection at the CPU J3DTevs.cpp side
  // proved TSky's OWN materials bind the correct sky.bmt textures, so
  // whatever draws the crosshatch must be a DIFFERENT material entering the
  // same shared "DrawBuf Sky Xlu" — this dump identifies it by its GPU-side
  // texture binding, independent of which CPU material baked it).
  static const bool s_xhGpuDbg = std::getenv("SB_XH_GPU_DBG") != nullptr;
  if (s_xhGpuDbg
      && g_sbLastMarker.find("Sky Xlu") != std::string::npos) {
    static long s_n = 0;
    if (s_n < 200) {
      for (unsigned st = 0; st < g_gxState.numTevStages; ++st) {
        const auto& s = g_gxState.tevStages[st];
        if (s.texMapId < 0 || static_cast<unsigned>(s.texMapId) >= aurora::gx::MaxTextures)
          continue;
        const auto& tobj = g_gxState.textures[s.texMapId].texObj;
        ++s_n;
        const auto& vp = g_gxState.logicalViewport;
        std::fprintf(stderr,
                     "[xh-gpu] #%ld prim=%u verts=%u stage=%u texMap=%d fmt=%d %ux%u "
                     "wrapS=%d wrapT=%d proj=%c vp=(%.0f,%.0f %.0fx%.0f) numTexGens=%u\n",
                     s_n, static_cast<unsigned>(prim), vtxCount, st,
                     static_cast<int>(s.texMapId), static_cast<int>(tobj.format()), tobj.width(),
                     tobj.height(), static_cast<int>(tobj.wrap_s()), static_cast<int>(tobj.wrap_t()),
                     g_gxState.projType == GX_ORTHOGRAPHIC ? 'O' : 'P', vp.left, vp.top, vp.width,
                     vp.height, g_gxState.numTexGens);
      }
    }
  }
  // SB_SKIP_TEV3=1 (diagnostic, title-logo wash investigation): drop any draw
  // configured with >=3 TEV stages. Isolates whether the J2DPicture
  // black/white "duotone" recolor stage (title_parts_NN.bti overlay quads,
  // forced to flat white when mBlack.rgb==mWhite.rgb==white) is what's
  // washing out the title logo, without touching TEV/shader math.
  {
    static int s_skipTev3 = -1;
    if (s_skipTev3 < 0) {
      const char* e = std::getenv("SB_SKIP_TEV3");
      s_skipTev3 = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    if (s_skipTev3 == 1 && g_gxState.numTevStages >= 3) {
      return;
    }
  }
  // Build pipeline, bind groups, and push draw command
  static const bool s_profArr = std::getenv("SB_PROFILE_GFX") != nullptr;
  auto _pArr = s_profArr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  BindGroupRanges ranges{};
  static int s_arrDbg = -1;
  if (s_arrDbg < 0) {
    const char* e = std::getenv("SB_ARR_DBG");
    s_arrDbg = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
  }
  for (int i = GX_VA_POS; i <= GX_VA_TEX7; ++i) {
    if (g_gxState.vtxDesc[i] != GX_INDEX8 && g_gxState.vtxDesc[i] != GX_INDEX16) {
      continue;
    }
    auto& array = g_gxState.arrays[i];
    // SB_NO_ARRCACHE=1 (diagnostic): re-upload every indexed array on every
    // draw — bisects "GPU reads a stale cached array upload" (content changed
    // under an unchanged base/size, e.g. via replay memupdates) from other
    // vertex-path causes.
    static int s_noArrCache = -1;
    if (s_noArrCache < 0) {
      const char* e = std::getenv("SB_NO_ARRCACHE");
      s_noArrCache = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    bool cached = s_noArrCache == 0 && array.cachedRange.size > 0;
    if (cached) {
      ranges.vaRanges[i - GX_VA_POS] = array.cachedRange;
      if (sb_drawprim_profile()) {
        ++g_arrCachedHits;
      }
    } else {
      // size 0 = "trust" registration (J3D): upload the auto-derived extent
      // (max referenced index, maintained in draw_prim).
      const u32 effSize = array.size != 0 ? array.size : array.sizeAuto;
      // The slot's own cachedRange was dropped by GXSetArray re-registering this attribute, but
      // these exact bytes may already have been uploaded this frame under a different slot state
      // — the game re-points a slot at A, then B, then back at A. The upload belongs to the data,
      // so look it up by data identity before paying for another copy. See gx.hpp.
      const ArrayUploadKey upKey{array.data, effSize};
      const gfx::Range* hit = array_upload_lookup(upKey);
      gfx::Range range;
      if (hit != nullptr) {
        range = *hit;
        if (sb_drawprim_profile()) {
          ++g_arrDataCacheHits;
        }
      } else {
        // Persistent arena first: this array is almost certainly byte-identical to the copy the
        // GPU already holds from last frame (measured: 100% of 20.44 MB/frame is unchanged), in
        // which case the hash is the whole cost and no upload happens at all. The hash is the
        // change detector — there is no way to observe the game rewriting an array otherwise, and
        // a stale binding here is silent geometry corruption.
        const uint64_t contentHash =
            static_cast<uint64_t>(xxh3_hash_s(static_cast<const uint8_t*>(array.data), effSize));
        bool uploaded = false;
        range = gfx::push_storage_persistent(static_cast<const uint8_t*>(array.data), effSize, upKey,
                                             contentHash, &uploaded);
        if (range.size == 0) {
          // Arena full — fall back to the per-frame path rather than binding a bogus range.
          range = gfx::push_storage(static_cast<const uint8_t*>(array.data), effSize);
          if (sb_drawprim_profile()) {
            ++g_arrArenaFull;
          }
        } else if (sb_drawprim_profile()) {
          if (uploaded) {
            ++g_arrPersistUploads;
            g_arrPersistUploadBytes += effSize;
          } else {
            ++g_arrPersistHits;
            g_arrPersistHitBytes += effSize;
          }
        }
        array_upload_store(upKey, range);
      }
      ranges.vaRanges[i - GX_VA_POS] = range;
      array.cachedRange = range;
      // SB_PROFILE_DRAWPRIM=1: volume, not just call count. SB_PROFILE_GFX says arrayUpload is
      // the single largest per-draw item, and the only way to tell "uploads a lot of distinct
      // data" from "uploads the SAME array repeatedly" is to measure both totals and compare.
      // A call count alone cannot distinguish them, and the fix differs completely.
      if (sb_drawprim_profile() && hit == nullptr) {
        ++g_arrUploadCount;
        g_arrUploadBytes += effSize;
        const uint64_t key =
            (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(array.data)) << 8) ^ effSize;
        // SAFETY MEASUREMENT for keying the upload cache on data identity instead of on the
        // slot's current registration. That is only sound if a given (ptr,size) always holds the
        // same BYTES for the whole frame; if the game rewrites an array in place between two
        // draws, a data-keyed cache would serve the stale upload. Hash the uploaded bytes and
        // report any key whose content changed within a frame. A count of 0 is the precondition;
        // anything else falsifies the optimisation outright.
        const uint64_t h =
            static_cast<uint64_t>(xxh3_hash_s(static_cast<const uint8_t*>(array.data), effSize));
        // Compare against the SAME key's hash in the previous frame.
        {
          const auto pit = g_arrHashPrevFrame.find(key);
          if (pit == g_arrHashPrevFrame.end()) {
            g_arrNewVsPrevBytes += effSize;
          } else if (pit->second == h) {
            g_arrSameAsPrevBytes += effSize;
          } else {
            g_arrChangedVsPrevBytes += effSize;
          }
        }
        {
          const auto oit = g_arrOffsetPrevFrame.find(key);
          if (oit != g_arrOffsetPrevFrame.end()) {
            if (oit->second == range.offset) {
              ++g_arrOffsetStable;
            } else {
              ++g_arrOffsetMoved;
            }
          }
          g_arrOffsetThisFrame[key] = range.offset;
        }
        const auto it = g_arrUploadHash.find(key);
        if (it == g_arrUploadHash.end()) {
          g_arrUploadHash.emplace(key, h);
          g_arrUploadDistinctBytes += effSize; // only the FIRST upload of a given (ptr,size)
        } else if (it->second != h) {
          ++g_arrContentChanged;
        }
        g_arrUploadDistinct.insert(key);
      }
    }
    // SB_ARR_DBG=1: per-draw indexed-array binding trace — the shader reads
    // the shared storage buffer at vaRange.offset, so a zero-size push means
    // the array DATA never reached the GPU for this frame.
    if (s_arrDbg == 1) {
      static long nZero = 0, nReal = 0, nCached = 0;
      static long n = 0;
      if (cached) ++nCached;
      else if (array.size == 0) ++nZero;
      else ++nReal;
      ++n;
      if ((array.size > 0 && !cached && nReal <= 40) || (n % 2000) == 0)
        std::fprintf(stderr,
                     "[arr-dbg] n=%ld zero=%ld real=%ld cached=%ld | attr=%d data=%p size=%u le=%d cached=%d "
                     "range=(%u+%u) mark='%s'\n",
                     n, nZero, nReal, nCached, i, array.data, array.size, static_cast<int>(array.le), cached ? 1 : 0,
                     ranges.vaRanges[i - GX_VA_POS].offset, ranges.vaRanges[i - GX_VA_POS].size,
                     g_sbLastMarker.c_str());
    }
  }

  // SB_PROFILE_GFX perf breakdown of the per-draw GPU-command build (sb_gx_prof_add defined above).
  static const bool s_prof = std::getenv("SB_PROFILE_GFX") != nullptr;
  auto _pt = [] { return std::chrono::steady_clock::now(); };
  auto _pa = s_prof ? _pt() : std::chrono::steady_clock::time_point{};

  if (s_profArr) sb_gx_prof_add(5, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - _pArr).count());

  PipelineConfig config{};
  populate_pipeline_config(config, prim, fmt);
  const auto info = build_shader_info(config.shaderConfig);
  if (s_prof) { auto n = _pt(); sb_gx_prof_add(0, std::chrono::duration<double, std::micro>(n - _pa).count()); _pa = n; }
  // SB_SKIP_HASH=<hex>: drop every draw whose shader-config hash matches (the same
  // hash SB_SHADER_HASH prints). The only way to isolate a specific draw on the raw
  // .dff replay path, where game-side markers are absent (mark=''). Multiple hashes
  // may be given comma-separated. Used to pin the sea-surface speckle draw.
  {
    static bool s_init = false;
    static uint64_t s_skip[8] = {0};
    static int s_skipN = 0;
    if (!s_init) {
      s_init = true;
      if (const char* e = std::getenv("SB_SKIP_HASH"); e != nullptr && e[0] != '\0') {
        const char* p = e;
        while (*p != '\0' && s_skipN < 8) {
          s_skip[s_skipN++] = std::strtoull(p, nullptr, 16);
          const char* comma = std::strchr(p, ',');
          if (comma == nullptr) break;
          p = comma + 1;
        }
      }
    }
    if (s_skipN > 0) {
      const uint64_t h = static_cast<uint64_t>(xxh3_hash(config.shaderConfig));
      for (int k = 0; k < s_skipN; ++k) {
        if (h == s_skip[k]) return;
      }
    }
  }
  // SB_SHADER_HASH=1: per-draw shader-config hash (pairs with SB_SHADER_DUMP
  // to locate the exact WGSL a given marked draw uses).
  {
    static int s_shHash = -1;
    if (s_shHash < 0) {
      const char* e = std::getenv("SB_SHADER_HASH");
      s_shHash = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    if (s_shHash == 1) {
      static long n = 0;
      if (++n <= 30000)
        std::fprintf(stderr, "[draw-shader] n=%ld hash=%zx proj=%c verts=%u mark='%s'\n", n,
                     static_cast<size_t>(xxh3_hash(config.shaderConfig)),
                     g_gxState.projType == GX_ORTHOGRAPHIC ? 'O' : 'P', vtxCount, g_sbLastMarker.c_str());
    }
  }
  g_sbDrawSamplesCopy = false;
  if (s_prof) _pa = _pt();
  resolve_sampled_textures(info);
  if (s_prof) { auto n = _pt(); sb_gx_prof_add(6, std::chrono::duration<double, std::micro>(n - _pa).count()); _pa = n; }
  // SB_SKIP_COPY_QUAD=1 (diagnostic): drop draws that sample an EFB-copy
  // texture (the screen-texture repaint quads) — separates "scene hidden
  // under the quad overdraw" from "scene never rendered".
  static int s_skipCopyQuad = -1;
  if (s_skipCopyQuad < 0) {
    const char* e = std::getenv("SB_SKIP_COPY_QUAD");
    s_skipCopyQuad = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
  }
  if (s_skipCopyQuad == 1 && g_sbDrawSamplesCopy) {
    return;
  }
  // SB_PERSP_ZONLY=1 (diagnostic): force color_update=false on every PERSPECTIVE draw,
  // mimicking retail's title where the whole 3D scene is drawn Z-only (cU=0) and only the
  // 2D/EFB-snapshot composite paints color. If the frame then shows a crisp composited scene,
  // native's over-bright is the missing Z-only discipline; if it goes black, native has no
  // composite painting the backdrop. Restores after the draw so state isn't corrupted.
  {
    static int s_perspZonly = -1;
    if (s_perspZonly < 0) { const char* e = std::getenv("SB_PERSP_ZONLY"); s_perspZonly = (e && e[0] && e[0] != '0') ? 1 : 0; }
    if (s_perspZonly == 1 && g_gxState.projType != GX_ORTHOGRAPHIC) {
      g_gxState.colorUpdate = false;
    }
  }
  // SB_SKIP_ORTHO=1 (diagnostic): drop every orthographic-projection draw —
  // strips all 2D overlays (logo art, faders, letterbox) to expose what the
  // 3D scene alone contributes to the framebuffer.
  static int s_skipOrtho = -1;
  if (s_skipOrtho < 0) {
    const char* e = std::getenv("SB_SKIP_ORTHO");
    s_skipOrtho = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
  }
  if (s_skipOrtho == 1 && g_gxState.projType == GX_ORTHOGRAPHIC) {
    return;
  }
  // SB_SKIP_PERSP=1: complement of SKIP_ORTHO — drop perspective draws to see
  // ONLY the 2D/ortho layer's contribution.
  static int s_skipPersp = -1;
  if (s_skipPersp < 0) {
    const char* e = std::getenv("SB_SKIP_PERSP");
    s_skipPersp = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
  }
  if (s_skipPersp == 1 && g_gxState.projType != GX_ORTHOGRAPHIC) {
    return;
  }
  // SB_SKIP_MARK=<substr>[,<substr>...]: drop every draw whose current
  // draw-identity marker contains any listed substring — isolate which J3D
  // buffer (Mirror / LensFlare / MapOpa / Sky...) causes an artifact by
  // removing it. SB_SKIP_ADD=1: drop additive draws (src=SRCALPHA dst=ONE),
  // the glow/flare accumulation class.
  {
    static int s_init = 0;
    static const char* s_skipMark = nullptr;
    static int s_skipAdd = 0;
    if (!s_init) {
      s_init = 1;
      s_skipMark = std::getenv("SB_SKIP_MARK");
      const char* a = std::getenv("SB_SKIP_ADD");
      s_skipAdd = (a != nullptr && a[0] != '\0' && a[0] != '0') ? 1 : 0;
    }
    if (s_skipMark != nullptr && !g_sbLastMarker.empty()) {
      const std::string_view marks(s_skipMark);
      size_t start = 0;
      while (start <= marks.size()) {
        size_t comma = marks.find(',', start);
        const auto tok = marks.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        if (!tok.empty() && g_sbLastMarker.find(tok) != std::string::npos) {
          return;
        }
        if (comma == std::string_view::npos) break;
        start = comma + 1;
      }
    }
    if (s_skipAdd == 1 && g_gxState.blendMode == GX_BM_BLEND &&
        g_gxState.blendFacSrc == GX_BL_SRCALPHA && g_gxState.blendFacDst == GX_BL_ONE) {
      return;
    }
  }
  // SB_SKIP_TEXDIM=WxH[,WxH...] (diagnostic, title sky crosshatch isolation):
  // drop any draw whose tex0 dims match one of the listed sizes, regardless
  // of marker. Used to bisect which of the small Sky-Xlu quads (16x16,
  // 64x64, 128x128, 16x64, 256x256) produces the moire.
  {
    static int s_init = 0;
    static const char* s_skipTexDim = nullptr;
    if (!s_init) {
      s_init = 1;
      s_skipTexDim = std::getenv("SB_SKIP_TEXDIM");
    }
    static const bool s_skyDimDbg = std::getenv("SB_SKY_DIM_DBG") != nullptr;
    if (s_skyDimDbg && g_sbLastMarker.find("Sky") != std::string::npos) {
      static long n = 0;
      const auto& obj0 = g_gxState.textures[0].texObj;
      if (++n <= 4000000)
        std::fprintf(stderr, "[sky-dim] n=%ld tex0=%ux%u mark='%s'\n", n, obj0.width(), obj0.height(),
                     g_sbLastMarker.c_str());
    }
    if (s_skipTexDim != nullptr && g_sbLastMarker.find("Sky") != std::string::npos) {
      const auto& obj0 = g_gxState.textures[0].texObj;
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%ux%u", obj0.width(), obj0.height());
      const std::string_view dims(s_skipTexDim);
      size_t start = 0;
      while (start <= dims.size()) {
        size_t comma = dims.find(',', start);
        const auto tok = dims.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        if (!tok.empty() && tok == buf) {
          if (std::getenv("SB_SKIP_TEXDIM_DBG")) {
            static long n = 0;
            if (++n <= 50)
              std::fprintf(stderr, "[skip-texdim] matched %s mark='%s'\n", buf, g_sbLastMarker.c_str());
          }
          return;
        }
        if (comma == std::string_view::npos) break;
        start = comma + 1;
      }
    }
  }
  // SB_SKIP_SKY_IDX=<n>[,<n>...] (diagnostic, title sky crosshatch
  // bisection): skip the Nth draw (0-based) since 'DrawBuf Sky Xlu' began as
  // a marker group, regardless of texture dims/wrap — texture dims reported
  // here can diverge from the dims at draw-dump time (resolve_sampled_textures
  // may rebind/resize the slot), so counting draws within the marker is the
  // only reliable way to name a specific quad. The Sky Xlu group is 9 draws:
  // 0=16x16(far,Mirror) 1,2=128x128(far) 3=64x64(near,Mirror) 4,5,6=16x64(near)
  // 7=256x256(near) 8=202vert dome(untextured).
  {
    static const char* s_skipIdx = nullptr;
    static int s_init = 0;
    if (!s_init) {
      s_init = 1;
      s_skipIdx = std::getenv("SB_SKIP_SKY_IDX");
    }
    bool matched = false;
    if (s_skipIdx != nullptr && g_sbLastMarker == "DrawBuf Sky Xlu") {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%d", g_sbMarkerDrawIdx);
      const std::string_view idxs(s_skipIdx);
      size_t start = 0;
      while (start <= idxs.size()) {
        size_t comma = idxs.find(',', start);
        const auto tok = idxs.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        if (!tok.empty() && tok == buf) { matched = true; break; }
        if (comma == std::string_view::npos) break;
        start = comma + 1;
      }
    }
    ++g_sbMarkerDrawIdx;
    if (matched) return;
  }
  // SB_SKIP_MIRROR_FAR / SB_SKIP_MIRROR_NEAR (diagnostic, title sky
  // crosshatch bisection): the sky material draws two Mirror-wrapped quads
  // (per SB_SKIP_TEXDIM survey) under two different translation clusters —
  // a "far" quad (posmtx translate.x > 10000, huge world-space offset,
  // 16x16 tex) and a "near" quad (translate near origin, 64x64 tex, part of
  // the sun/glow group). Skip each independently to see which one is the
  // crosshatch source.
  //
  // These three used to be UNCACHED getenv on this per-draw path: 1,677,795 calls EACH in a 30 s
  // Delfino run (measured with an LD_PRELOAD getenv counter), 5.03M of the run's 7.0M total. Each
  // one is a linear scan of environ for a diagnostic that is off. The two SKIP switches are
  // behaviour toggles, so they stay env vars — but hoisted into statics; the _DBG one is a logging
  // gate, so it becomes a lucent channel like the rest of the printing in this file.
  {
    static const char* far = std::getenv("SB_SKIP_MIRROR_FAR");
    static const char* near = std::getenv("SB_SKIP_MIRROR_NEAR");
    static const lucent::Channel chSkipMirror{"skipmirror"};
    const bool dbg = static_cast<bool>(chSkipMirror);
    if ((far != nullptr || near != nullptr || dbg) && g_sbLastMarker.find("Sky") != std::string::npos) {
      const auto& obj0 = g_gxState.textures[0].texObj;
      if (obj0.wrap_s() == GX_MIRROR || obj0.wrap_t() == GX_MIRROR) {
        const auto* pn = reinterpret_cast<const float*>(&g_gxState.pnMtx[g_gxState.currentPnMtx].pos);
        bool isFar = std::fabs(pn[3]) > 10000.0f;
        if (dbg) {
          static long n = 0;
          if (++n <= 200)
            lucent::debug(chSkipMirror, "far={} tex={}x{} trans={:.1f}", isFar ? 1 : 0, obj0.width(),
                          obj0.height(), pn[3]);
        }
        if ((isFar && far != nullptr) || (!isFar && near != nullptr)) {
          return;
        }
      }
    }
  }
  // SB_ORTHO_DBG=1: per ORTHO draw, log verts + tex dims + TEV/blend/channel
  // + whether it samples a copy — to identify the fullscreen white 2D quad
  // washing the scene. Prints the FULL final-pass 2D overlay set of a frame.
  {
    static int s_orthoDbg = -1;
    if (s_orthoDbg < 0) {
      const char* e = std::getenv("SB_ORTHO_DBG");
      s_orthoDbg = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    if (s_orthoDbg == 1 && g_gxState.projType == GX_ORTHOGRAPHIC) {
      static long n = 0;
      if (++n <= 600) {
        const auto& obj = g_gxState.textures[0].texObj;
        const auto& t0 = g_gxState.tevStages[0];
        std::fprintf(stderr,
                     "[ortho] n=%ld verts=%u tex0=%ux%u tev=%u samplesCopy=%d bm=%d bf=%d/%d cU=%d aU=%d "
                     "ch0[light=%d matSrc=%d] tev0C[a=%d b=%d c=%d d=%d] mark='%s'\n",
                     n, vtxCount, obj.width(), obj.height(), g_gxState.numTevStages,
                     g_sbDrawSamplesCopy ? 1 : 0, static_cast<int>(g_gxState.blendMode),
                     static_cast<int>(g_gxState.blendFacSrc), static_cast<int>(g_gxState.blendFacDst),
                     g_gxState.colorUpdate ? 1 : 0, g_gxState.alphaUpdate ? 1 : 0,
                     static_cast<int>(g_gxState.colorChannelConfig[0].lightingEnabled),
                     static_cast<int>(g_gxState.colorChannelConfig[0].matSrc),
                     static_cast<int>(t0.colorPass.a), static_cast<int>(t0.colorPass.b),
                     static_cast<int>(t0.colorPass.c), static_cast<int>(t0.colorPass.d), g_sbLastMarker.c_str());
      }
    }
  }
  if (s_prof) _pa = _pt();
  const auto bindGroups = build_bind_groups(info);
  if (s_prof) { auto n = _pt(); sb_gx_prof_add(1, std::chrono::duration<double, std::micro>(n - _pa).count()); _pa = n; }
  const auto pipeline = gfx::pipeline_ref(config);
  if (s_prof) { auto n = _pt(); sb_gx_prof_add(2, std::chrono::duration<double, std::micro>(n - _pa).count()); _pa = n; }

  uint32_t instanceCount = 1;
  if (prim == GX_LINES) {
    instanceCount = vtxCount / 2;
  } else if (prim == GX_LINESTRIP) {
    instanceCount = vtxCount - 1;
  } else if (prim == GX_POINTS) {
    instanceCount = vtxCount;
  }
  if (s_prof) _pa = _pt();
  // Where POS lives inside this draw's vertex record. Computed here because the descriptor state
  // that determines it is current NOW and gone by the time the recorded frame is interpolated.
  u16 posOffset = 0;
  u8 posF32XYZ = 0;
  calculate_pos_layout(fmt, posOffset, posF32XYZ);
  const uint32_t texMtxCamMask = eye_space_texgen_mask(info);
  auto uniformRange = build_uniform(info, vertRange.offset, ranges);
  if (s_prof) { auto n = _pt(); sb_gx_prof_add(3, std::chrono::duration<double, std::micro>(n - _pa).count()); _pa = n; }
  gfx::push_draw_command(DrawData{
      .pipeline = pipeline,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = uniformRange,
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = instanceCount,
      .bindGroups = bindGroups,
      .dstAlpha = g_gxState.dstAlpha,
      .tag = g_pendingDrawTag,
      .pop = g_pendingDrawPop,
      .exact = g_pendingDrawExact,
      // The block order is pnMtx.pos, then the texture matrices, then pnMtx.nrm — see build_uniform.
      // Spelled out with both counts rather than doubling one: MaxPnMtx and MaxTexMtx happen to be
      // equal today, and a change to either would silently misplace the normal matrices.
      .mtxPosOffset = g_lastUniformMtxOffset,
      .mtxNrmOffset = g_lastUniformMtxOffset +
                      (u32)((MaxPnMtx + MaxTexMtx) * sizeof(Mat3x4<float>)),
      .ortho = g_gxState.projType == GX_ORTHOGRAPHIC ? (u8)1 : (u8)0,
      .vtxStride = static_cast<u16>(g_gxState.lastVtxSize),
      .posOffset = posOffset,
      .posF32XYZ = posF32XYZ,
      .texMtxCamMask = texMtxCamMask,
      .pnMtxSlot = static_cast<uint8_t>(g_gxState.currentPnMtx),
  });
  // ONE-SHOT, unlike the tag. A tag identifies an OBJECT and legitimately covers every draw that
  // object emits; "present this exactly" is a property of ONE primitive, and the function that
  // emits an exact screen quad often goes on to emit ordinary interpolating geometry from the same
  // call (TModelWaterManager::drawShineShadowVolume emits both its screen quads and its sphere-slice
  // display lists). A latch there would freeze the slices too, and the emitter has no seam at which
  // to clear it — GXEnd is not a function in this build. Consuming it here is the semantic that
  // matches what the flag means.
  g_pendingDrawExact = 0;
  if (g_pendingDrawTag != 0) {
    ++g_taggedDrawCount;
  } else {
    ++g_untaggedDrawCount;
    if (g_gxState.projType == GX_ORTHOGRAPHIC) {
      ++g_untaggedOrthoDrawCount;
    } else {
      ++g_untaggedPerspDrawCount;
      const auto posDesc = g_gxState.vtxDesc[GX_VA_POS];
      if (posDesc == GX_INDEX8 || posDesc == GX_INDEX16) {
        ++g_untaggedPerspIndexedCount;
      } else {
        ++g_untaggedPerspDirectCount;
      }
    }
  }
  if (s_prof) { auto n = _pt(); sb_gx_prof_add(4, std::chrono::duration<double, std::micro>(n - _pa).count()); }
}

std::string read_string(const u8* data, u32& pos, u32 size, bool bigEndian) {
  CHECK(pos + 2 <= size, "Aurora string length read overrun");
  const u16 length = read_u16(data + pos, bigEndian);
  pos += 2;

  CHECK(pos + length <= size, "Aurora string read overrun");
  std::string str(reinterpret_cast<const char*>(data) + pos, length);
  pos += length;
  return str;
}

void handle_aurora(const u8* data, u32& pos, u32 size, bool bigEndian) {
  ZoneScoped;
  CHECK(pos + 2 <= size, "Aurora cmd read overrun");
  u16 subCmd = read_u16(data + pos, bigEndian);
  pos += 2;

  // Setting of vertex array bases.
  if (subCmd == GX_AURORA_LOAD_VIEWPORT_RENDER) {
    CHECK(pos + 24 <= size, "GX_AURORA_LOAD_VIEWPORT_RENDER read overrun");
    const f32 left = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 top = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 width = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 height = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 nearZ = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 farZ = read_f32(data + pos, bigEndian);
    pos += 4;
    set_render_viewport({
        .left = left,
        .top = top,
        .width = width,
        .height = height,
        .znear = nearZ,
        .zfar = farZ,
    });
  } else if (subCmd == GX_AURORA_LOAD_SCISSOR_RENDER) {
    CHECK(pos + 16 <= size, "GX_AURORA_LOAD_SCISSOR_RENDER read overrun");
    const int32_t left = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t top = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t width = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t height = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    set_render_scissor({left, top, width, height});
  } else if (subCmd >= GX_AURORA_LOAD_ARRAYBASE && subCmd <= (GX_AURORA_LOAD_ARRAYBASE | 0x0f)) {
    CHECK(pos + 13 <= size, "GX_AURORA_LOAD_ARRAYBASE read overrun");
    u32 attrIdx = subCmd - GX_AURORA_LOAD_ARRAYBASE + GX_VA_POS;

    u64 arrayAddr = read_u64(data + pos, bigEndian);
    pos += 8;
    u32 arraySize = read_u32(data + pos, bigEndian);
    pos += 4;
    bool le = data[pos] == 1;
    pos += 1;

    auto& array = g_gxState.arrays[attrIdx];
    const auto newData = reinterpret_cast<void*>(arrayAddr);
    if (array.data != newData || array.size != arraySize || array.le != le) {
      const bool sameBacking = array.data == newData;
      array.data = newData;
      array.size = arraySize;
      array.le = le;
      // Only drop the cached upload when the backing array actually changes.
      array.cachedRange = {};
      // The auto-derived extent belongs to the backing memory, not the slot:
      // keep it when only flags change, reset it when the pointer moves.
      if (!sameBacking) array.sizeAuto = 0;
      g_gxState.stateDirty = true;
    }
  } else if (subCmd == GX_AURORA_LOAD_TEXOBJ) {
    CHECK(pos + 34 <= size, "GX_AURORA_LOAD_TEXOBJ read overrun");
    const auto texMapId = data[pos];
    pos += 1;
    CHECK(texMapId < MaxTextures, "invalid texture map id {}", texMapId);
    auto& slot = g_gxState.loadedTextures[texMapId];
    slot.data = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    pos += 8;
    slot.mWidth = read_u32(data + pos, bigEndian);
    pos += 4;
    slot.mHeight = read_u32(data + pos, bigEndian);
    pos += 4;
    slot.mFormat = static_cast<GXTexFmt>(read_u32(data + pos, bigEndian));
    pos += 4;
    slot.tlut = static_cast<GXTlut>(read_u32(data + pos, bigEndian));
    pos += 4;
    // Real mip level count (1 == base level only). GXTexObj_::mip_count()
    // gates entirely on flags-bit0 (has_mips); slot.mode1's max_lod field
    // (bits [8:15], Q4.4) is populated separately and earlier in this same
    // command stream by the real TexMode1 BP register write that
    // J3DGDSetTexLookupMode already emits per bind (decode_tex_bp_reg's
    // Kind::Mode1 case above does `slot.mode1 = value` verbatim from actual
    // hardware register state) -- this opcode must not clobber it, only
    // gate whether it's honored. Previously this byte was hardcoded to 0
    // ("no mips") for every GD-context texture bind regardless of the
    // source asset's real mip chain, so mip_count() always collapsed to 1
    // even when mode1's max_lod was already correct. Minifying a repeating
    // pattern with no mip filtering aliases into a fine regular
    // moire/crosshatch; the sky's 8x8 I4 cloud texture (tiled many times
    // across the dome) is the worst-case instance of this general gap.
    const u8 mipCount = data[pos];
    if (mipCount > 1) {
      slot.flags |= 1u;
    } else {
      slot.flags &= ~1u;
    }
    pos += 1;
    slot.texObjId = read_u32(data + pos, bigEndian);
    pos += 4;
    slot.texDataVersion = read_u32(data + pos, bigEndian);
    pos += 4;
    slot.set_no_cache(false); // Reset no-cache flag
    g_gxState.stateDirty = true;
  } else if (subCmd == GX_AURORA_LOAD_TLUT) {
    CHECK(pos + 23 <= size, "GX_AURORA_LOAD_TLUT read overrun");
    const auto idx = data[pos];
    pos += 1;
    CHECK(idx < MaxTluts, "invalid tlut slot {}", idx);
    auto& slot = g_gxState.loadedTluts[idx];
    slot.data = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    pos += 8;
    slot.format = static_cast<GXTlutFmt>(read_u32(data + pos, bigEndian));
    pos += 4;
    slot.numEntries = read_u16(data + pos, bigEndian);
    pos += 2;
    slot.tlutObjId = read_u32(data + pos, bigEndian);
    pos += 4;
    slot.tlutDataVersion = read_u32(data + pos, bigEndian);
    pos += 4;
    slot.set_no_cache(false); // Reset no-cache flag
    g_gxState.stateDirty = true;
  } else if (subCmd == GX2_SET_POLYGON_OFFSET) {
    CHECK(pos + 20 <= size, "GX2_SET_POLYGON_OFFSET read overrun");
    g_gxState.frontOffset = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.frontScale = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.backOffset = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.backScale = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.clamp = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.stateDirty = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_SRC) {
    CHECK(pos + 16 <= size, "GX_AURORA_LOAD_COPY_SRC read overrun");
    const int32_t left = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t top = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t width = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t height = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    g_gxState.texCopySrc = {left, top, width, height};
  } else if (subCmd == GX_AURORA_LOAD_COPY_DST) {
    CHECK(pos + 13 <= size, "GX_AURORA_LOAD_COPY_DST read overrun");
    g_gxState.texCopyDstWidth = read_u32(data + pos, bigEndian);
    pos += 4;
    g_gxState.texCopyDstHeight = read_u32(data + pos, bigEndian);
    pos += 4;
    g_gxState.texCopyFmt = static_cast<GXTexFmt>(read_u32(data + pos, bigEndian));
    pos += 4;
    g_gxState.texCopyDstWide = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_DEST) {
    CHECK(pos + 8 <= size, "GX_AURORA_LOAD_COPY_DEST read overrun");
    g_gxState.texCopyDest = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    pos += 8;
  } else if (subCmd == GX_AURORA_REQUEST_DEPTH_SNAPSHOT) {
    gfx::depth_peek::request_snapshot();
  } else if (subCmd == GX_AURORA_BEGIN_OFFSCREEN) {
    CHECK(pos + 8 <= size, "GX_AURORA_BEGIN_OFFSCREEN read overrun");
    const u32 width = read_u32(data + pos, bigEndian);
    pos += 4;
    const u32 height = read_u32(data + pos, bigEndian);
    pos += 4;
    gfx::begin_offscreen(width, height);
  } else if (subCmd == GX_AURORA_END_OFFSCREEN) {
    gfx::end_offscreen();
  } else if (subCmd == GX_AURORA_VIEW_MTX) {
    CHECK(pos + 48 <= size, "GX_AURORA_VIEW_MTX read overrun");
    float m[12];
    for (int i = 0; i < 12; ++i) {
      m[i] = read_f32(data + pos + i * 4, bigEndian);
    }
    pos += 48;
    gfx::interp::set_view_matrix(m);
  } else if (subCmd == GX_AURORA_DRAW_TAG) {
    CHECK(pos + 8 <= size, "GX_AURORA_DRAW_TAG read overrun");
    // Latched, not consumed: one tag covers every draw the tagged object emits, however many
    // elements and material passes that turns out to be. The emitter is responsible for tagging
    // again when it moves to the next object.
    g_pendingDrawTag = read_u64(data + pos, bigEndian);
    pos += 8;
  } else if (subCmd == GX_AURORA_DRAW_POP) {
    CHECK(pos + 1 <= size, "GX_AURORA_DRAW_POP read overrun");
    g_pendingDrawPop = data[pos];
    pos += 1;
  } else if (subCmd == GX_AURORA_DRAW_EXACT) {
    CHECK(pos + 1 <= size, "GX_AURORA_DRAW_EXACT read overrun");
    g_pendingDrawExact = data[pos];
    pos += 1;
  } else if (subCmd == GX_AURORA_DESTROY_TEXOBJ) {
    CHECK(pos + 4 <= size, "GX_AURORA_DESTROY_TEXOBJ read overrun");
    evict_texture_object(read_u32(data + pos, bigEndian));
    pos += 4;
  } else if (subCmd == GX_AURORA_DESTROY_TLUT) {
    CHECK(pos + 4 <= size, "GX_AURORA_DESTROY_TLUT read overrun");
    evict_tlut_object(read_u32(data + pos, bigEndian));
    pos += 4;
  } else if (subCmd == GX_AURORA_DESTROY_COPY_TEX) {
    CHECK(pos + 8 <= size, "GX_AURORA_DESTROY_COPY_TEX read overrun");
    evict_copy_texture(reinterpret_cast<const void*>(read_u64(data + pos, bigEndian)));
    pos += 8;
  } else if (subCmd == GX_AURORA_DRAW_SIZED) {
    CHECK(pos + 5 <= size, "GX_AURORA_DRAW_SIZED read overrun");
    u8 cmd = data[pos];
    pos += 1;
    u32 byteLen = read_u32(data + pos, bigEndian);
    pos += 4;
    GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    if (byteLen != 0) {
      u32 vtxSize;
      if (g_gxState.lastVtxFmt == fmt) {
        vtxSize = g_gxState.lastVtxSize;
      } else {
        vtxSize = calculate_last_vtx_size(fmt);
      }
      ASSERT(vtxSize != 0 && byteLen % vtxSize == 0,
             "GX_AURORA_DRAW_SIZED: {} bytes is not a whole number of size-{} vertices", byteLen, vtxSize);
      u32 vtxCount = byteLen / vtxSize;
      ASSERT(vtxCount <= 0xFFFF, "GX_AURORA_DRAW_SIZED: too many vertices ({})", vtxCount);
      draw_prim(prim, fmt, static_cast<u16>(vtxCount), data, pos, size);
    }
  } else if (subCmd == GX_AURORA_DRAW_INDEXED) {
    ZoneScopedN("DRAW_INDEXED");
    CHECK(pos + 7 <= size, "GX_AURORA_DRAW_INDEXED read overrun");
    const u8 cmd = data[pos];
    pos += 1;
    const u16 vtxCount = read_u16(data + pos, bigEndian);
    pos += 2;
    const u32 indexCount = read_u32(data + pos, bigEndian);
    pos += 4;
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    ASSERT(prim == GX_TRIANGLES, "GX_AURORA_DRAW_INDEXED: primitive must be GX_TRIANGLES, got {}",
           static_cast<u32>(prim));
    const u32 idxBytes = indexCount * static_cast<u32>(sizeof(u16));
    CHECK(pos + idxBytes <= size, "GX_AURORA_DRAW_INDEXED index data overrun");
    // Index data is always host-endian; push it to the GPU buffer as-is
    const gfx::Range idxRange = gfx::push_indices(data + pos, idxBytes, 4);
    pos += idxBytes;
    u32 vtxSize;
    if (g_gxState.lastVtxFmt == fmt) {
      vtxSize = g_gxState.lastVtxSize;
    } else {
      vtxSize = calculate_last_vtx_size(fmt);
    }
    const u32 totalVtxBytes = vtxCount * vtxSize;
    CHECK(pos + totalVtxBytes <= size, "GX_AURORA_DRAW_INDEXED vertex data overrun");
    const gfx::Range vertRange = gfx::push_verts(data + pos, totalVtxBytes, 4);
    pos += totalVtxBytes;
    if (indexCount != 0) {
      push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, indexCount);
    }
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_PUSH) {
    auto label = read_string(data, pos, size, bigEndian);
    gfx::push_debug_group(std::move(label));
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_POP) {
    pop_debug_group();
  } else if (subCmd == GX_AURORA_DEBUG_MARKER_INSERT) {
    auto label = read_string(data, pos, size, bigEndian);
    // SB_TIMELINE: ordered per-frame event log (marker changes + copies +
    // clears) to reconstruct the exact GC multi-pass sequence.
    if (sb_timeline_enabled() && label != g_sbLastMarker) {
      sb_timeline_log("draws '%s'", label.c_str());
    }
    g_sbLastMarker = label; // draw-identity for SB_DRAW_DUMP
    g_sbMarkerDrawIdx = 0;  // reset per-marker draw counter (SB_SKIP_SKY_IDX)
    gfx::insert_debug_marker(std::move(label));
  }

  else {
    u32 dumpStart = (pos > 33) ? pos - 33 : 0;
    u32 dumpEnd = (pos + 32 < size) ? pos + 32 : size;
    std::string hex;
    for (u32 i = dumpStart; i < dumpEnd; i++) {
      if (i == pos - 3 || i == pos - 2)
        hex += fmt::format("[{:02x}]", data[i]);
      else
        hex += fmt::format(" {:02x}", data[i]);
    }
    Log.error("Unknown Aurora subcommand: 0x{:04X} at pos {} -- caller likely mis-encoded "
              "the GX_AURORA (0x50) opcode payload or fell out of frame from an earlier "
              "extension. Hex dump (pos {}-{}, [] marks the subCmd bytes):{}",
              subCmd, pos - 2, dumpStart, dumpEnd - 1, hex);
  }
}

} // namespace aurora::gx::fifo

// Structural display-list scanner for build-time validation (SB_DL_VALIDATE):
// walks the same opcode grammar as the drain WITHOUT executing anything and
// returns the offset of the first invalid byte, or 0xFFFFFFFF if the stream
// is well-formed. allowDraws=0 is for J3D MATERIAL DLs, which must never
// contain geometry; a draw opcode there means the buffer is corrupt/stale.
// Draw opcodes can't be validated structurally anyway (their payload size
// depends on live VCD/VAT state), so allowDraws=1 stops the scan (returns
// clean) at the first draw instead.
extern "C" uint32_t aurora_gx_scan_dl(const uint8_t* data, uint32_t size, uint32_t allowDraws) {
  using namespace aurora::gx::fifo;
  u32 pos = 0;
  auto rd16 = [&](u32 p) { return static_cast<u32>(data[p]) << 8 | data[p + 1]; };
  while (pos < size) {
    const u8 cmd = data[pos];
    const u8 opcode = cmd & 0xF8; // CP_OPCODE_MASK
    if (cmd == 0x00) { // NOP
      pos += 1;
      continue;
    }
    if (cmd == 0x08) { // CP reg: opcode + addr + u32
      if (pos + 6 > size) return pos;
      pos += 6;
      continue;
    }
    if (cmd == 0x10) { // XF: opcode + u32 header + count*4
      if (pos + 5 > size) return pos;
      const u32 count = (rd16(pos + 1) & 0xFFFF) + 1;
      if (pos + 5 + count * 4 > size) return pos;
      pos += 5 + count * 4;
      continue;
    }
    if (cmd == 0x61) { // BP: opcode + u32
      if (pos + 5 > size) return pos;
      pos += 5;
      continue;
    }
    if (opcode == 0x20 || opcode == 0x28 || opcode == 0x30 || opcode == 0x38) { // indexed XF
      if (pos + 5 > size) return pos;
      pos += 5;
      continue;
    }
    if (cmd == 0x40) { // CALL_DL
      if (pos + 9 > size) return pos;
      pos += 9;
      continue;
    }
    if (cmd == 0x48) { // INVAL_VTX
      pos += 1;
      continue;
    }
    if (cmd == 0x50) { // GX_AURORA extension
      if (pos + 3 > size) return pos;
      const u32 sub = rd16(pos + 1);
      u32 payload;
      if (sub == 0x0001) payload = 24;                       // LOAD_VIEWPORT_RENDER
      else if (sub == 0x0002) payload = 16;                  // LOAD_SCISSOR_RENDER
      else if (sub >= 0x0010 && sub <= 0x001F) payload = 13; // LOAD_ARRAYBASE
      else if (sub == 0x0020 || sub == 0x0022) {             // debug group push / marker
        if (pos + 5 > size) return pos;
        payload = 2 + rd16(pos + 3);
      } else if (sub == 0x0021) payload = 0;                 // debug group pop
      else if (sub == 0x0030) payload = 34;                  // LOAD_TEXOBJ
      else return pos;                                       // DRAW_SIZED/INDEXED etc: not in material DLs
      if (pos + 3 + payload > size) return pos;
      pos += 3 + payload;
      continue;
    }
    if (opcode >= 0x80 && opcode < 0xC0) { // draw
      return allowDraws != 0 ? 0xFFFFFFFFu : pos;
    }
    return pos;
  }
  return 0xFFFFFFFFu;
}

// Accessor for gfx/common.hpp's staging-overflow fatal (declared there inside
// namespace aurora, so define it in that namespace).
namespace aurora {
const char* aurora_gfx_last_draw_desc() { return aurora::gx::fifo::sb_last_draw_desc(); }
} // namespace aurora
