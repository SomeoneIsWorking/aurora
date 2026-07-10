#include "command_processor.hpp"
#include <cstdarg>

#include "fifo.hpp"

#include "../gfx/common.hpp"
#include "../gfx/depth_peek.hpp"
#include "dolphin/gx/GXAurora.h"
#include "gx.hpp"
#include "gx_fmt.hpp"
#include "pipeline.hpp"
#include "shader_info.hpp"
#include "../internal.hpp"

#include <absl/container/flat_hash_map.h>
#include <tracy/Tracy.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>

namespace aurora::gx::fifo {
static Module Log("aurora::gx::fifo");

// Last debug marker seen in the stream — names the draw-buffer/pass each
// subsequent draw belongs to (printed by SB_DRAW_DUMP).
static thread_local std::string g_sbLastMarker;
static thread_local int g_sbMarkerDrawIdx = 0; // Nth draw since the current marker began (SB_SKIP_SKY_IDX)
// Exposed for cross-TU diagnostics (e.g. copy_tex logging which J3D buffer/2D
// element a GXCopyTex follows).
extern "C" const char* sb_gx_last_marker() { return g_sbLastMarker.c_str(); }

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
  u16 numIndices = 0;
  if (prim == GX_QUADS) {
    buf.reserve_extra((vtxCount / 4) * 6 * sizeof(u16));

    for (u16 v = 0; v < vtxCount; v += 4) {
      u16 idx0 = vtxStart + v;
      u16 idx1 = vtxStart + v + 1;
      u16 idx2 = vtxStart + v + 2;
      u16 idx3 = vtxStart + v + 3;

      buf.append(idx0);
      buf.append(idx1);
      buf.append(idx2);
      numIndices += 3;

      buf.append(idx2);
      buf.append(idx3);
      buf.append(idx0);
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLES) {
    buf.reserve_extra(vtxCount * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      buf.append(idx);
      ++numIndices;
    }
  } else if (prim == GX_TRIANGLEFAN) {
    buf.reserve_extra(((u32(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      buf.append(std::array{vtxStart, static_cast<u16>(idx - 1), idx});
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLESTRIP) {
    buf.reserve_extra(((static_cast<u32>(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      if ((v & 1) == 0) {
        buf.append(std::array{static_cast<u16>(idx - 2), static_cast<u16>(idx - 1), idx});
      } else {
        buf.append(std::array{static_cast<u16>(idx - 1), static_cast<u16>(idx - 2), idx});
      }
      numIndices += 3;
    }
  } else if (prim == GX_LINES || prim == GX_LINESTRIP || prim == GX_POINTS) {
    buf.reserve_extra(6 * sizeof(u16));
    buf.append<u16>(0);
    buf.append<u16>(1);
    buf.append<u16>(3);
    buf.append<u16>(3);
    buf.append<u16>(2);
    buf.append<u16>(0);
    numIndices = 6;
  } else
    UNLIKELY FATAL("unsupported primitive type {}", static_cast<u32>(prim));
  return numIndices;
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
static void handle_aurora(const u8* data, u32& pos, u32 size, bool bigEndian);

// Ring buffer of recent draws — dumped alongside the opcode ring buffer at
// unknown-opcode FATAL so a fifo desync's originating draw is visible without
// enabling AURORA_DRAW_TRACE (which spams stderr fast enough to skew timing).
struct RecentDraw { u32 pos; u8 cmd; u16 vtxCount; u32 vtxSize; };
static constexpr size_t kRecentDrawN = 16;
static thread_local RecentDraw s_recentDraws[kRecentDrawN];
struct RecentCmd { u32 pos; u8 cmd; };
static constexpr size_t kRecentN = 32;
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
      if (!copy_xf_data(dstAddr, srcData, len, !array.le)) {
#ifndef NDEBUG
        Log.debug("Unimplemented indexed XF load (opcode 0x{:02X}, dstAddr={:04x})", opcode, dstAddr);
#endif
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
          u32 dumpStart = (pos > 161) ? pos - 161 : 0;
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
        {
          std::string trail;
          for (size_t i = 0; i < kRecentN; ++i) {
            const auto& r = s_recent[(s_recentHead + i) % kRecentN];
            if (r.pos == 0 && r.cmd == 0 && i == 0) continue;
            trail += fmt::format(" [pos={} cmd=0x{:02X}]", r.pos, r.cmd);
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

  // Texture copy
  case 0x52: {
    const bool clear = bp_get(value, 1, 11) != 0;
    if (bp_get(value, 1, 14) != 0) {
      Log.warn("STUB: display copy is not implemented");
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
    if (std::getenv("SB_COPY_DBG") != nullptr) {
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
    if (std::getenv("SB_COPY_DBG") != nullptr) {
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
        g_gxState.colorChannelState[GX_COLOR0].ambColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA0].ambColor = unpack_color(val);
        g_gxState.stateDirty = true;
        break;
      case 0x0B:
        // Ambient color 1
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
          // Encoding: bit 9 = (attnFn != GX_AF_SPEC), bit 10 = (attnFn != GX_AF_NONE)
          bool bit9 = bp_get(val, 1, 9) != 0;
          bool bit10 = bp_get(val, 1, 10) != 0;
          u32 lightsHi = bp_get(val, 4, 11);
          if (!bit10) {
            chan.attnFn = GX_AF_NONE;
          } else if (!bit9) {
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
          set_logical_viewport({
              .left = ox - 340.0f - width / 2.0f,
              .top = oy - 340.0f - height / 2.0f,
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
          if (std::getenv("SB_DRAW_DUMP") != nullptr) {
            std::fprintf(stderr, "[proj-set] type=%c p=(%.4f %.4f %.4f %.4f %.4f %.4f) mark='%s'\n",
                         projType == GX_ORTHOGRAPHIC ? 'O' : 'P', p0, p1, p2, p3, p4, p5,
                         g_sbLastMarker.c_str());
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

// Draw command handler - parses vertices inline and caches results
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

static void draw_prim(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* data, u32& pos, u32 size) {
  ZoneScoped;
  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calculate_last_vtx_size(fmt); }

  u32 totalVtxBytes = vtxCount * vtxSize;
  if (pos + totalVtxBytes > size)
    UNLIKELY { handle_draw_overrun(totalVtxBytes, data, pos, size); }

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

  // SB_NDC_PROBE=<minVerts> [+ SB_NDC_MARK=<marker substring>]: CPU-side
  // replication of the vertex shader transform for indexed-position draws —
  // projects EVERY vertex through the exact matrices the GPU will use
  // (per-vertex PNMTXIDX honored, same row-dot convention as the WGSL) and
  // histograms the clip results. Distinguishes "strip lands off-screen due to
  // a transform bug" from "strip genuinely tiny / genuinely missing".
  {
    static int s_minVerts = -2;
    static const char* s_markFilter = nullptr;
    if (s_minVerts == -2) {
      const char* e = std::getenv("SB_NDC_PROBE");
      s_minVerts = (e != nullptr && e[0] != '\0') ? std::atoi(e) : -1;
      if (s_minVerts == 0) s_minVerts = 1;
      s_markFilter = std::getenv("SB_NDC_MARK");
    }
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
    if (s_minVerts > 0 && s_printed < 400 && vtxCount >= static_cast<u16>(s_minVerts) &&
        (posDesc == GX_INDEX16 || posDesc == GX_INDEX8) &&
        (posFmt.type == GX_F32 || posFmt.type == GX_S16) && arr.data != nullptr &&
        (s_markFilter == nullptr || g_sbLastMarker.find(s_markFilter) != std::string::npos)) {
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
          if (g_gxState.projType != GX_ORTHOGRAPHIC && s_printedP < 6 && v < 4) {
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
        if (s_printed <= 6 && v < 4) {
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
          std::fprintf(stderr,
                       "[ndc-probe]   v%u idx=%u pos=(%.1f,%.1f,%.1f) mv=(%.1f,%.1f,%.1f) ndc=(%.3f,%.3f,%.4f) "
                       "w=%.1f mtx=%u clr0=[%02x %02x %02x %02x] c0type=%d\n",
                       v, idx, x, y, z, mv[0], mv[1], mv[2], nx, ny, nz, clip[3], firstMtx, c0raw[0], c0raw[1],
                       c0raw[2], c0raw[3], static_cast<int>(g_gxState.vtxFmts[fmt].attrs[GX_VA_CLR0].type));
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
                   s_printed, vtxCount, in, zin, wneg, g_gxState.projType == GX_ORTHOGRAPHIC ? 'O' : 'P', xmin, xmax,
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
  if (std::getenv("SB_LENS_UV_DBG") != nullptr && g_sbLastMarker.find("LensFlare") != std::string::npos) {
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
      std::fprintf(stderr,
                   "[lens-uv-hit] #%ld prim=%u verts=%u fmt=%d t0desc=%d t0type=%d t0cnt=%d t0frac=%u "
                   "arrData=%p arrStride=%u numTevStages=%u tev0[texCoordId=%d texMapId=%d chanId=%d] "
                   "tex=%dx%d wrap=(%d,%d) mark='%s'\n",
                   s_hit, static_cast<unsigned>(prim), vtxCount, static_cast<int>(fmt),
                   static_cast<int>(g_gxState.vtxDesc[GX_VA_TEX0]),
                   static_cast<int>(g_gxState.vtxFmts[fmt].attrs[GX_VA_TEX0].type),
                   static_cast<int>(g_gxState.vtxFmts[fmt].attrs[GX_VA_TEX0].cnt),
                   g_gxState.vtxFmts[fmt].attrs[GX_VA_TEX0].frac, g_gxState.arrays[GX_VA_TEX0].data,
                   g_gxState.arrays[GX_VA_TEX0].stride, g_gxState.numTevStages,
                   static_cast<int>(hstage0.texCoordId), static_cast<int>(hstage0.texMapId),
                   static_cast<int>(hstage0.channelId), htexW, htexH, hwrapS, hwrapT, g_sbLastMarker.c_str());
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
        std::fprintf(stderr,
                     "[lens-uv] #%ld v=%u desc=%d type=%d cnt=%d frac=%u uv=(%.4f,%.4f) tex=%dx%d wrap=(%d,%d) "
                     "tcg[src=%d mtx=%d type=%d] verts=%u proj=%c mark='%s'\n",
                     s_n, v, static_cast<int>(t0Desc), static_cast<int>(t0Fmt.type), static_cast<int>(t0Fmt.cnt),
                     t0Fmt.frac, u, w, texW, texH, wrapS, wrapT, static_cast<int>(tcg.src), static_cast<int>(tcg.mtx),
                     static_cast<int>(tcg.type), vtxCount, g_gxState.projType == GX_ORTHOGRAPHIC ? 'O' : 'P',
                     g_sbLastMarker.c_str());
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
    if (nFields > 0) {
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
    }
  }

  auto* lastDraw = !g_gxState.stateDirty ? gfx::get_last_draw_command<DrawData>() : nullptr;
  const bool canMerge = lastDraw != nullptr && prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS &&
                        lastDraw->instanceCount == 1;

  // Push raw vertex data to buffer. Merged draws must remain byte-contiguous with the previous range.
  gfx::Range vertRange = gfx::push_verts(data + pos, totalVtxBytes, canMerge ? 0 : 4);
  pos += totalVtxBytes;

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
    return;
  }

  handle_draw_unmerged(prim, fmt, vtxCount, vertRange);
}

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

static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices) {
  // Per-drain draw/vertex tally for SB_DRAW_STATS (reported from fifo::drain).
  detail::sDrainDraws += 1;
  detail::sDrainVerts += vtxCount;
  // SB_DRAW_DUMP=1: one-shot per-draw identity dump for the first drain past
  // draw #200 — prim/verts/texture/position-matrix translation, enough to
  // recognize which shapes the frame contains (e.g. the 752-vert sky dome).
  if (const char* e = std::getenv("SB_DRAW_DUMP"); e != nullptr) {
    // SB_DRAW_DUMP=<startDraw>: dump 200 draws starting at that global draw
    // index (draw counts run ~160/frame at title; pick start = frame*160).
    // SB_DRAW_DUMP=1 keeps the old early-boot window.
    static int s_dumped = 0;
    static int s_start = -1;
    if (s_start < 0) {
      s_start = std::atoi(e);
      if (s_start < 200) s_start = 200;
    }
    if (s_dumped >= s_start && s_dumped < s_start + 200) {
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
      // clr0Desc/clr1Desc: whether this draw's VCD actually supplies CLR0/CLR1
      // (GX_NONE=0 -> vtx_attr() default-white fallback fires; GX_DIRECT=1/
      // GX_INDEX8=2/GX_INDEX16=3 -> real per-vertex color stream is bound).
      const auto clr0Desc = g_gxState.vtxDesc[GX_VA_CLR0];
      const auto clr1Desc = g_gxState.vtxDesc[GX_VA_CLR1];
      std::fprintf(stderr,
                   "[draw-dump] #%d prim=%u verts=%u tex0=%ux%u zcmp=%d zupd=%d trans=(%.1f,%.1f,%.1f) "
                   "proj=%c blend=%u vp=(%.0f,%.0f %.0fx%.0f) sc=(%d,%d %ux%u) "
                   "tev=%u ch0[light=%d matSrc=%d ambSrc=%d mat=(%.2f,%.2f,%.2f,%.2f) amb=(%.2f,%.2f,%.2f) mask=%02x] "
                   "a0[light=%d matSrc=%d ambSrc=%d mat=%.2f amb=%.2f mask=%02x] "
                   "prj=[%.4f %.4f %.4f %.4f] cU=%d aU=%d bm=%d bf=%d/%d pos[desc=%d cnt=%d type=%d frac=%u] clr0=%d clr1=%d mtxIdx=%u "
                   "posmtx=[%.2f %.2f %.2f %.2f | %.2f %.2f %.2f %.2f | %.2f %.2f %.2f %.2f] mark='%s'\n",
                   s_dumped, static_cast<unsigned>(prim), vtxCount, obj.width(), obj.height(),
                   static_cast<int>(g_gxState.depthCompare), static_cast<int>(g_gxState.depthUpdate),
                   pn[3], pn[7], pn[11], g_gxState.projType == GX_ORTHOGRAPHIC ? 'O' : 'P',
                   static_cast<unsigned>(g_gxState.blendMode), vp.left, vp.top, vp.width, vp.height,
                   sc.x, sc.y, sc.width, sc.height, g_gxState.numTevStages,
                   static_cast<int>(cc.lightingEnabled), static_cast<int>(cc.matSrc), static_cast<int>(cc.ambSrc),
                   cs.matColor.x(),
                   cs.matColor.y(), cs.matColor.z(), cs.matColor.w(), cs.ambColor.x(), cs.ambColor.y(),
                   cs.ambColor.z(), static_cast<unsigned>(cs.lightMask.to_ulong() & 0xff),
                   static_cast<int>(cca.lightingEnabled), static_cast<int>(cca.matSrc), static_cast<int>(cca.ambSrc),
                   csa.matColor.w(), csa.ambColor.w(), static_cast<unsigned>(csa.lightMask.to_ulong() & 0xff),
                   reinterpret_cast<const float*>(&g_gxState.proj)[0],
                   reinterpret_cast<const float*>(&g_gxState.proj)[5],
                   reinterpret_cast<const float*>(&g_gxState.proj)[10],
                   reinterpret_cast<const float*>(&g_gxState.proj)[11],
                   g_gxState.colorUpdate ? 1 : 0, g_gxState.alphaUpdate ? 1 : 0,
                   static_cast<int>(g_gxState.blendMode), static_cast<int>(g_gxState.blendFacSrc),
                   static_cast<int>(g_gxState.blendFacDst), static_cast<int>(posDesc),
                   static_cast<int>(posFmt.cnt), static_cast<int>(posFmt.type), posFmt.frac,
                   static_cast<int>(clr0Desc), static_cast<int>(clr1Desc),
                   g_gxState.currentPnMtx, pn[0], pn[1], pn[2], pn[3], pn[4], pn[5], pn[6], pn[7],
                   pn[8], pn[9], pn[10], pn[11], g_sbLastMarker.c_str());
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
        const auto& tobj = g_gxState.textures[0].texObj;
        std::fprintf(stderr, "  [tex] fmt=%d minFilt=%d magFilt=%d wrapS=%d wrapT=%d\n",
                     static_cast<int>(tobj.format()), static_cast<int>(tobj.min_filter()),
                     static_cast<int>(tobj.mag_filter()), static_cast<int>(tobj.wrap_s()),
                     static_cast<int>(tobj.wrap_t()));
        for (unsigned r = 0; r < aurora::gx::MaxTevRegs; ++r) {
          const auto& c = g_gxState.colorRegs[r];
          std::fprintf(stderr, "  [tevreg] %u = (%.3f, %.3f, %.3f, %.3f)\n", r, c.x(), c.y(), c.z(), c.w());
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
  if (std::getenv("SB_CLOUD_TC_DBG") != nullptr) {
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
  if (std::getenv("SB_XH_GPU_DBG") != nullptr
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
    bool cached = array.cachedRange.size > 0;
    if (cached) {
      ranges.vaRanges[i - GX_VA_POS] = array.cachedRange;
    } else {
      // size 0 = "trust" registration (J3D): upload the auto-derived extent
      // (max referenced index, maintained in draw_prim).
      const u32 effSize = array.size != 0 ? array.size : array.sizeAuto;
      const auto range = gfx::push_storage(static_cast<const uint8_t*>(array.data), effSize);
      ranges.vaRanges[i - GX_VA_POS] = range;
      array.cachedRange = range;
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

  PipelineConfig config{};
  populate_pipeline_config(config, prim, fmt);
  const auto info = build_shader_info(config.shaderConfig);
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
  resolve_sampled_textures(info);
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
    if (std::getenv("SB_SKY_DIM_DBG") != nullptr && g_sbLastMarker.find("Sky") != std::string::npos) {
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
  {
    const char* far = std::getenv("SB_SKIP_MIRROR_FAR");
    const char* near = std::getenv("SB_SKIP_MIRROR_NEAR");
    const char* dbg = std::getenv("SB_SKIP_MIRROR_DBG");
    if ((far != nullptr || near != nullptr || dbg != nullptr) && g_sbLastMarker.find("Sky") != std::string::npos) {
      const auto& obj0 = g_gxState.textures[0].texObj;
      if (obj0.wrap_s() == GX_MIRROR || obj0.wrap_t() == GX_MIRROR) {
        const auto* pn = reinterpret_cast<const float*>(&g_gxState.pnMtx[g_gxState.currentPnMtx].pos);
        bool isFar = std::fabs(pn[3]) > 10000.0f;
        if (dbg != nullptr) {
          static long n = 0;
          if (++n <= 200)
            std::fprintf(stderr, "[skip-mirror] far=%d tex=%ux%u trans=%.1f\n", isFar ? 1 : 0, obj0.width(),
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
  const auto bindGroups = build_bind_groups(info);
  const auto pipeline = gfx::pipeline_ref(config);

  uint32_t instanceCount = 1;
  if (prim == GX_LINES) {
    instanceCount = vtxCount / 2;
  } else if (prim == GX_LINESTRIP) {
    instanceCount = vtxCount - 1;
  } else if (prim == GX_POINTS) {
    instanceCount = vtxCount;
  }
  gfx::push_draw_command(DrawData{
      .pipeline = pipeline,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = build_uniform(info, vertRange.offset, ranges),
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = instanceCount,
      .bindGroups = bindGroups,
      .dstAlpha = g_gxState.dstAlpha,
  });
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
