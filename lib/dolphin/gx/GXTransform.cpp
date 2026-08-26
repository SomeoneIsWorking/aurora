#include "gx.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <execinfo.h>
#include "__gx.h"
#include "dolphin/mtx/GeoTypes.h"

static inline void CacheProjectionVector(const f32* ptr, GXProjectionType type) {
  __gx->projType = type;
  for (int i = 0; i < 6; ++i) {
    __gx->projMtx[i] = ptr[i + 1];
  }
}

// VIGetRetraceCount is defined game-side (sms-boot/runtime/sdk_stubs.cpp),
// advanced once per sb_frame_present (sms-boot/runtime/frame_seam.cpp).
// Declared weak so aurora's standalone unit tests (which don't link the
// game) still build. Same pattern as lib/gx/command_processor.cpp.
extern "C" unsigned VIGetRetraceCount(void) __attribute__((weak));
static unsigned sb_proj_vi_retrace_count() { return (&VIGetRetraceCount) ? VIGetRetraceCount() : 0; }

// Shared cross-instrument sequence counter (sms-boot/runtime/trace_seq.cpp).
// SB_TRACE_SEQ=1: prefix proj-dbg (and pos-mtx-slot0) lines with seq=N so
// this family interleaves exactly with the present-boundary/plist-order/
// drawbuf-flush logs on one global order (retrace stamps alone straddle
// present boundaries and can't be trusted to interleave — see trace_seq.cpp).
extern "C" uint64_t sb_trace_seq(void) __attribute__((weak));
static bool sb_trace_seq_on() {
  static int v = -1;
  if (v < 0) {
    const char* e = std::getenv("SB_TRACE_SEQ");
    v = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
  }
  return v == 1 && &sb_trace_seq;
}

// SB_PROJ_DBG_AFTER=<retrace> also gates [posmtx0-dbg]: unthresholded, this
// print fires on every GX_PNMTX0 load (every draw call), which floods stderr
// heavily enough from frame 0 to dominate a paced run's wall-clock (observed:
// 75s of paced runtime only reached retrace 788 with this unconditional).
// Reuses the existing SB_PROJ_DBG_AFTER threshold rather than adding a new
// var, since posmtx0 is diagnostic sibling data to the projection trace.
static long sb_proj_dbg_after_thresh() {
  static long t = -2;
  if (t == -2) {
    const char* e = std::getenv("SB_PROJ_DBG_AFTER");
    t = (e != nullptr && e[0] != '\0') ? std::atol(e) : -1;
  }
  return t;
}

// SB_PROJ_DBG=1: per-call log of every GXSetProjection[v] — type, the 4
// "diagonal" scale/translate terms (mtx00, mtx11, mtx22, mtx23 — the values
// that distinguish a calibrated perspective/ortho from a garbage one), and
// the current VIGetRetraceCount so calls can be correlated to a specific
// present. SB_PROJ_DBG_AFTER=<retraceCount>: once VIGetRetraceCount clears
// this threshold, also dump a caller backtrace for the first K calls of
// EACH projection type (K=5) so the binder can be named, not just counted.
static void sb_proj_dbg_log(const char* fn, GXProjectionType type, const f32* projVec) {
  static int dbgAll = -1;
  static long afterThresh = -1;
  if (dbgAll < 0) {
    const char* e = std::getenv("SB_PROJ_DBG");
    dbgAll = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    const char* eAfter = std::getenv("SB_PROJ_DBG_AFTER");
    afterThresh = (eAfter != nullptr && eAfter[0] != '\0') ? std::atol(eAfter) : -1;
  }
  if (dbgAll != 1 && afterThresh < 0) {
    return;
  }
  const unsigned retrace = sb_proj_vi_retrace_count();
  static long nCalls = 0;
  ++nCalls;
  const bool seqOn = sb_trace_seq_on();
  const uint64_t seq = seqOn ? sb_trace_seq() : 0;
  if (dbgAll == 1) {
    if (seqOn) {
      std::fprintf(stderr, "[proj-dbg] seq=%lu n=%ld fn=%s type=%c retrace=%u diag=[%.6f,%.6f,%.6f,%.6f]\n",
                   (unsigned long)seq, nCalls, fn, type == GX_ORTHOGRAPHIC ? 'O' : 'P', retrace, projVec[1],
                   projVec[3], projVec[5], projVec[6]);
    } else {
      std::fprintf(stderr, "[proj-dbg] n=%ld fn=%s type=%c retrace=%u diag=[%.6f,%.6f,%.6f,%.6f]\n", nCalls, fn,
                   type == GX_ORTHOGRAPHIC ? 'O' : 'P', retrace, projVec[1], projVec[3], projVec[5], projVec[6]);
    }
  }
  if (afterThresh >= 0 && static_cast<long>(retrace) >= afterThresh) {
    static int nOrtho = 0;
    static int nPersp = 0;
    int* counter = type == GX_ORTHOGRAPHIC ? &nOrtho : &nPersp;
    if (*counter < 5) {
      ++*counter;
      if (seqOn) {
        std::fprintf(stderr,
                      "[proj-dbg-after] seq=%lu n=%ld fn=%s type=%c retrace=%u diag=[%.6f,%.6f,%.6f,%.6f] (case %d/5 of type)\n",
                      (unsigned long)seq, nCalls, fn, type == GX_ORTHOGRAPHIC ? 'O' : 'P', retrace, projVec[1],
                      projVec[3], projVec[5], projVec[6], *counter);
      } else {
        std::fprintf(stderr,
                      "[proj-dbg-after] n=%ld fn=%s type=%c retrace=%u diag=[%.6f,%.6f,%.6f,%.6f] (case %d/5 of type)\n",
                      nCalls, fn, type == GX_ORTHOGRAPHIC ? 'O' : 'P', retrace, projVec[1], projVec[3], projVec[5],
                      projVec[6], *counter);
      }
      void* fr[16];
      int nf = backtrace(fr, 16);
      backtrace_symbols_fd(fr, nf, 2);
    }
  }
}

extern "C" {

void GXSetProjection(const void* mtx_, GXProjectionType type) {
  // SB_PROJ_BT=1: sequence-numbered caller backtrace per projection set.
  // Fifo order is FIFO, so call #N here == the Nth [proj-set] at drain.
  {
    static int dbg = -1;
    if (dbg < 0) {
      const char* e = std::getenv("SB_PROJ_BT");
      dbg = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    if (dbg == 1) {
      static long n = 0;
      ++n;
      std::fprintf(stderr, "[proj-call] n=%ld type=%c\n", n, type == GX_ORTHOGRAPHIC ? 'O' : 'P');
      void* fr[8];
      int nf = backtrace(fr, 8);
      backtrace_symbols_fd(fr, nf, 2);
    }
  }
  const auto& mtx = *reinterpret_cast<const aurora::Mat4x4<float>*>(mtx_);
  const f32 projVec[] = {
      static_cast<f32>(type == GX_ORTHOGRAPHIC),
      mtx[0][0],
      type == GX_ORTHOGRAPHIC ? mtx[0][3] : mtx[0][2],
      mtx[1][1],
      type == GX_ORTHOGRAPHIC ? mtx[1][3] : mtx[1][2],
      mtx[2][2],
      mtx[2][3],
  };
  CacheProjectionVector(projVec, type);
  sb_proj_dbg_log("GXSetProjection", type, projVec);

  // XF bulk write: 6 params + projection type at 0x1020-0x1026
  GX_WRITE_U8(0x10);
  GX_WRITE_U32(0x00061020);
  GX_WRITE_XF_REG_F(32, __gx->projMtx[0]);
  GX_WRITE_XF_REG_F(33, __gx->projMtx[1]);
  GX_WRITE_XF_REG_F(34, __gx->projMtx[2]);
  GX_WRITE_XF_REG_F(35, __gx->projMtx[3]);
  GX_WRITE_XF_REG_F(36, __gx->projMtx[4]);
  GX_WRITE_XF_REG_F(37, __gx->projMtx[5]);
  GX_WRITE_XF_REG_2(38, __gx->projType);
  __gx->bpSent = 0;
}

void GXSetProjectionv(const f32* ptr) {
  CHECK(ptr != nullptr, "null projection vector");

  const GXProjectionType type = ptr[0] == 0.0f ? GX_PERSPECTIVE : GX_ORTHOGRAPHIC;
  {
    static int dbg = -1;
    if (dbg < 0) {
      const char* e = std::getenv("SB_PROJ_BT");
      dbg = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    if (dbg == 1) {
      static long n = 0;
      ++n;
      std::fprintf(stderr, "[proj-callv] n=%ld type=%c\n", n, type == GX_ORTHOGRAPHIC ? 'O' : 'P');
      void* fr[8];
      int nf = backtrace(fr, 8);
      backtrace_symbols_fd(fr, nf, 2);
    }
  }
  CacheProjectionVector(ptr, type);
  sb_proj_dbg_log("GXSetProjectionv", type, ptr);

  // XF bulk write: 6 params + projection type at 0x1020-0x1026
  GX_WRITE_U8(0x10);
  GX_WRITE_U32(0x00061020);
  for (int i = 0; i < 6; ++i) {
    GX_WRITE_F32(__gx->projMtx[i]);
  }
  GX_WRITE_U32(__gx->projType);
  __gx->bpSent = 0;
}

void GXLoadPosMtxImm(const void* mtx_, u32 id) {
  CHECK(id >= GX_PNMTX0 && id <= GX_PNMTX9, "invalid pn mtx {}", static_cast<int>(id));
  const auto* mtx = reinterpret_cast<const f32*>(mtx_);

  // SB_TRACE_SEQ=1: log every load to slot 0 (GX_PNMTX0, the slot every
  // unskinned draw call binds) so the unified trace shows the pos-mtx write
  // that pairs with each drawbuf flush, not just the projection write.
  if (id == GX_PNMTX0 && sb_trace_seq_on()) {
    long after = sb_proj_dbg_after_thresh();
    unsigned retrace = sb_proj_vi_retrace_count();
    if (after < 0 || static_cast<long>(retrace) >= after) {
      std::fprintf(stderr, "[posmtx0-dbg] seq=%lu retrace=%u m=[%.3f,%.3f,%.3f,%.3f]\n",
                   (unsigned long)sb_trace_seq(), retrace, mtx[3], mtx[7], mtx[11], mtx[0]);
    }
  }

  GX_WRITE_U8(0x10);
  GX_WRITE_U32((id * 4) | 0xB0000);
  for (int i = 0; i < 12; i++) {
    GX_WRITE_F32(mtx[i]);
  }
}

void GXLoadNrmMtxImm(const void* mtx_, u32 id) {
  CHECK(id >= GX_PNMTX0 && id <= GX_PNMTX9, "invalid pn mtx {}", static_cast<int>(id));
  const auto* mtx = reinterpret_cast<const f32*>(mtx_);

  GX_WRITE_U8(0x10);
  GX_WRITE_U32((id * 3 + 0x400) | 0x80000);
  // Write 3x3 from 3x4 matrix (skip translation column)
  GX_WRITE_F32(mtx[0]);
  GX_WRITE_F32(mtx[1]);
  GX_WRITE_F32(mtx[2]);
  GX_WRITE_F32(mtx[4]);
  GX_WRITE_F32(mtx[5]);
  GX_WRITE_F32(mtx[6]);
  GX_WRITE_F32(mtx[8]);
  GX_WRITE_F32(mtx[9]);
  GX_WRITE_F32(mtx[10]);
}

void GXSetCurrentMtx(u32 id) {
  CHECK(id >= GX_PNMTX0 && id <= GX_PNMTX9, "invalid pn mtx {}", id);
  SET_REG_FIELD(0, __gx->matIdxA, 6, 0, id);
  __GXSetMatrixIndex(GX_VA_PNMTXIDX);
}

void GXLoadTexMtxImm(const void* mtx_, u32 id, GXTexMtxType type) {
  CHECK((id >= GX_TEXMTX0 && id <= GX_IDENTITY) || (id >= GX_PTTEXMTX0 && id <= GX_PTIDENTITY), "invalid tex mtx {}",
        id);

  u32 addr;
  if (id >= GX_PTTEXMTX0) {
    addr = (id - GX_PTTEXMTX0) * 4 + 0x500;
    CHECK(type == GX_MTX3x4, "invalid pt mtx type {}", underlying(type));
  } else {
    addr = id * 4;
  }

  u32 count = (type == GX_MTX2x4) ? 8 : 12;
  u32 reg = addr | ((count - 1) << 16);

  GX_WRITE_U8(0x10);
  GX_WRITE_U32(reg);

  const auto* mtx = reinterpret_cast<const f32*>(mtx_);
  for (u32 i = 0; i < count; i++) {
    GX_WRITE_F32(mtx[i]);
  }
}

void GXSetViewport(float left, float top, float width, float height, float nearZ, float farZ) {
  GXSetViewportJitter(left, top, width, height, nearZ, farZ, 1);
}

void GXSetViewportJitter(float left, float top, float width, float height, float nearZ, float farZ, u32 field) {
  float sx;
  float sy;
  float sz;
  float ox;
  float oy;
  float oz;
  float zmin;
  float zmax;

  if (field == 0) {
    top -= 0.5f;
  }

  sx = width / 2.0f;
  sy = -height / 2.0f;
  // The retail SDK encodes the XF viewport origin relative to 342. Keep this
  // paired with command_processor.cpp's FIFO decode so SDK calls and replayed
  // command streams reconstruct the same logical viewport.
  constexpr float kViewportOriginBias = 342.0f;
  ox = kViewportOriginBias + (left + width / 2.0f);
  oy = kViewportOriginBias + (top + height / 2.0f);
  zmin = 1.6777215e7f * nearZ;
  zmax = 1.6777215e7f * farZ;
  sz = zmax - zmin;
  oz = zmax;

  __gx->vpLeft = left;
  __gx->vpTop = top;
  __gx->vpWd = width;
  __gx->vpHt = height;
  __gx->vpNearz = nearZ;
  __gx->vpFarz = farZ;

  GX_WRITE_U8(0x10);
  GX_WRITE_U32(0x0005101A);
  GX_WRITE_XF_REG_F(26, sx);
  GX_WRITE_XF_REG_F(27, sy);
  GX_WRITE_XF_REG_F(28, sz);
  GX_WRITE_XF_REG_F(29, ox);
  GX_WRITE_XF_REG_F(30, oy);
  GX_WRITE_XF_REG_F(31, oz);
  __gx->bpSent = 0;
}

void GXProject(f32 x, f32 y, f32 z, const f32 mtx[3][4], const f32* pm, const f32* vp, f32* sx,
               f32* sy, f32* sz) {
  Vec peye;
  f32 xc;
  f32 yc;
  f32 zc;
  f32 wc;

  peye.x = mtx[0][3] + ((mtx[0][2] * z) + ((mtx[0][0] * x) + (mtx[0][1] * y)));
  peye.y = mtx[1][3] + ((mtx[1][2] * z) + ((mtx[1][0] * x) + (mtx[1][1] * y)));
  peye.z = mtx[2][3] + ((mtx[2][2] * z) + ((mtx[2][0] * x) + (mtx[2][1] * y)));
  if (pm[0] == 0.0f) {
    xc = (peye.x * pm[1]) + (peye.z * pm[2]);
    yc = (peye.y * pm[3]) + (peye.z * pm[4]);
    zc = pm[6] + (peye.z * pm[5]);
    wc = 1.0f / -peye.z;
  } else {
    xc = pm[2] + (peye.x * pm[1]);
    yc = pm[4] + (peye.y * pm[3]);
    zc = pm[6] + (peye.z * pm[5]);
    wc = 1.0f;
  }
  *sx = (vp[2] / 2.0f) + (vp[0] + (wc * (xc * vp[2] / 2.0f)));
  *sy = (vp[3] / 2.0f) + (vp[1] + (wc * (-yc * vp[3] / 2.0f)));
  *sz = vp[5] + (wc * (zc * (vp[5] - vp[4])));
}

// Indexed matrix loads: emit the GC CP LOAD_INDX commands; the fifo processor
// fetches from the bound GX_POS_MTX_ARRAY / GX_NRM_MTX_ARRAY at drain time —
// the same deferred-fetch semantics as the GC command processor, so callers
// may keep mutating the pool between call and drain, and only the final
// values are consumed. Encoding: u32 = index<<16 | (len-1)<<12 | xfAddr.
void GXLoadPosMtxIndx(u16 mtx_indx, u32 id) {
  CHECK(id >= GX_PNMTX0 && id <= GX_PNMTX9, "invalid pn mtx {}", static_cast<int>(id));
  GX_WRITE_U8(GX_LOAD_INDX_A);
  GX_WRITE_U32((static_cast<u32>(mtx_indx) << 16) | (11u << 12) | (id * 4));
}

void GXLoadNrmMtxIndx3x3(u16 mtx_indx, u32 id) {
  CHECK(id >= GX_PNMTX0 && id <= GX_PNMTX9, "invalid pn mtx {}", static_cast<int>(id));
  GX_WRITE_U8(GX_LOAD_INDX_B);
  GX_WRITE_U32((static_cast<u32>(mtx_indx) << 16) | (8u << 12) | (id * 3 + 0x400));
}

// TODO GXLoadNrmMtxImm3x3
// TODO GXLoadTexMtxIndx
// TODO GXSetZScaleOffset
}
