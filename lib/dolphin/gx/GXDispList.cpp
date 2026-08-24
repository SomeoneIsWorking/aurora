#include "gx.hpp"
#include "__gx.h"

#include "../../gx/fifo.hpp"

#include <cstring>
#include <execinfo.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static __GXData_struct sSavedGXData;

extern "C" {
void GXBeginDisplayList(void* list, u32 size) {
  CHECK(!aurora::gx::fifo::in_display_list(), "Display list began twice!");

  // Flush any pending dirty state before recording
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // Save current shadow register state if requested
  if (__gx->dlSaveContext != 0) {
    std::memcpy(&sSavedGXData, __gx, sizeof(sSavedGXData));
  }

  __gx->inDispList = 1;

  // Redirect FIFO writes to the user-provided buffer
  aurora::gx::fifo::begin_display_list(static_cast<u8*>(list), size);
}

u32 GXEndDisplayList() {
  // Flush any pending dirty state into the display list
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // End FIFO redirection and get the byte count (ROUNDUP32)
  u32 bytesWritten = aurora::gx::fifo::end_display_list();

  // Restore saved shadow register state
  if (__gx->dlSaveContext != 0) {
    std::memcpy(__gx, &sSavedGXData, sizeof(*__gx));
  }

  __gx->inDispList = 0;

  return bytesWritten;
}

// Sanitize a display-list byte stream before pushing to the FIFO. GC-baked
// display lists carry raw GC opcodes verbatim, several of which Aurora
// cannot honour on a 64-bit host:
//
//   CP_LOAD_REG(0x08) + addr in 0xA0..0xAF (CP_REG_ARRAYBASE_ID) is a
//   32-bit array-base pointer -- truncated on a 64-bit host. Aurora's fifo
//   processor already accepts and ignores these silently, but stripping
//   them here keeps the trace clean and saves the processor a lookup. The
//   correct 64-bit base pointer is supplied out-of-band via
//   GX_AURORA_LOAD_ARRAYBASE (see GXSetArray / J3D's J3DLoadArrayBasePtr).
//
// Returns a pair (sanitized_ptr, sanitized_nbytes). Reuses a scratch buffer
// grown on demand.
static const u8* sanitize_dl(const u8* src, u32 nbytes, u32& outLen) {
  static thread_local std::vector<u8> scratch;
  scratch.clear();
  scratch.reserve(nbytes);
  u32 pos = 0;
  bool dirty = false;
  // STRUCTURE-AWARE walk. The original byte-by-byte scan matched the
  // [08][A0..AF] pattern inside command PAYLOADS too (e.g. a BP value whose
  // low byte is 0x08) and deleted 6 bytes from the middle of the stream,
  // desyncing everything after — a proven title-screen corruption. Commands
  // are stepped whole; the arraybase test only runs at command boundaries.
  // Once a draw opcode appears the remainder is copied verbatim: draw payload
  // sizes depend on live VCD/VAT state, and real DLs never place arraybase
  // writes after geometry.
  const auto copyRest = [&] {
    scratch.insert(scratch.end(), src + pos, src + nbytes);
    pos = nbytes;
  };
  const auto copyCmd = [&](u32 n) {
    if (n > nbytes - pos)
      n = nbytes - pos; // truncated tail: keep bytes, drain diagnoses
    scratch.insert(scratch.end(), src + pos, src + pos + n);
    pos += n;
  };
  while (pos < nbytes) {
    const u8 cmd = src[pos];
    const u8 opcode = cmd & 0xF8;
    if (cmd == 0x00) { // NOP
      copyCmd(1);
    } else if (cmd == 0x08) { // CP reg: opcode + addr + u32
      if (pos + 6 <= nbytes && src[pos + 1] >= 0xA0 && src[pos + 1] <= 0xAF) {
        // Drop CP_REG_ARRAYBASE_ID writes entirely.
        pos += 6;
        dirty = true;
      } else {
        copyCmd(6);
      }
    } else if (cmd == 0x10) { // XF: opcode + u32 header + count*4
      const u32 count = pos + 3 <= nbytes ? ((static_cast<u32>(src[pos + 1]) << 8 | src[pos + 2]) + 1) : 1;
      copyCmd(5 + count * 4);
    } else if (cmd == 0x61) { // BP: opcode + u32
      copyCmd(5);
    } else if (opcode == 0x20 || opcode == 0x28 || opcode == 0x30 || opcode == 0x38) { // indexed XF
      copyCmd(5);
    } else if (cmd == 0x40) { // CALL_DL
      copyCmd(9);
    } else if (cmd == 0x48) { // INVAL_VTX
      copyCmd(1);
    } else if (cmd == 0x50) { // GX_AURORA extension
      if (pos + 3 > nbytes) {
        copyRest();
        break;
      }
      const u32 sub = static_cast<u32>(src[pos + 1]) << 8 | src[pos + 2];
      u32 payload;
      if (sub == 0x0001) payload = 24;                       // LOAD_VIEWPORT_RENDER
      else if (sub == 0x0002) payload = 16;                  // LOAD_SCISSOR_RENDER
      else if (sub >= 0x0010 && sub <= 0x001F) payload = 17; // LOAD_ARRAYBASE
      else if (sub == 0x0020 || sub == 0x0022) {             // debug group push / marker
        payload = pos + 5 <= nbytes ? 2 + (static_cast<u32>(src[pos + 3]) << 8 | src[pos + 4]) : 2;
      } else if (sub == 0x0021) payload = 0;                 // debug group pop
      else if (sub == 0x0030) payload = 34;                  // LOAD_TEXOBJ
      else {
        // DRAW_SIZED/INDEXED or unknown: stop filtering, copy verbatim.
        copyRest();
        break;
      }
      copyCmd(3 + payload);
    } else {
      // Draw opcode (0x80-0xBF) or unknown: stop filtering, copy verbatim.
      copyRest();
      break;
    }
  }
  outLen = static_cast<u32>(scratch.size());
  // SB_DL_SANITIZE_DBG=1: report every drop — a drop from the middle of a
  // baked BMD display list is almost certainly a FALSE MATCH on vertex index
  // bytes (0x08 0xA0..0xAF inside payload), which desyncs the whole stream.
  if (dirty) {
    static int dbg = -1;
    if (dbg < 0) {
      const char* e = getenv("SB_DL_SANITIZE_DBG");
      dbg = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    if (dbg) {
      static long n = 0;
      std::fprintf(stderr, "[dl-sanitize] n=%ld dropped %u bytes of %u\n", ++n, nbytes - outLen, nbytes);
    }
  }
  return dirty ? scratch.data() : src;
}

void GXCallDisplayList(const void* data, u32 nbytes) {
  // Flush any pending dirty state before calling
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // SB_DL_TAG=1: insert a debug marker naming the DL buffer before its bytes
  // enter the fifo — a drain-side desync FATAL then reports which buffer (or
  // that it was in the immediate-write region between tags).
  {
    static int s_tag = -1;
    if (s_tag < 0) {
      const char* e = getenv("SB_DL_TAG");
      s_tag = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    if (s_tag == 1) {
      char label[48];
      std::snprintf(label, sizeof(label), "dl@%p+%u", data, nbytes);
      GXInsertDebugMarker(label);
    }
  }

  // SB_DL_VALIDATE=1: structural scan at the chokepoint EVERY display list
  // passes through. allowDraws=1 (geometry DLs are legal here; the scan stops
  // at the first draw opcode). On failure: hex, caller backtrace, abort —
  // this names the caller that submitted the malformed buffer.
  {
    static int s_val = -1;
    if (s_val < 0) {
      const char* e = getenv("SB_DL_VALIDATE");
      s_val = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    if (s_val == 1) {
      extern u32 aurora_gx_scan_dl(const u8*, u32, u32);
      const u32 bad = aurora_gx_scan_dl(static_cast<const u8*>(data), nbytes, 1);
      if (bad != 0xFFFFFFFFu) {
        const u8* d = static_cast<const u8*>(data);
        std::fprintf(stderr, "[dl-validate] MALFORMED DL at CALL dl=%p size=%u off=%u:", data, nbytes, bad);
        for (u32 i = (bad > 16 ? bad - 16 : 0); i < bad + 16 && i < nbytes; ++i)
          std::fprintf(stderr, " %02x%s", d[i], i == bad ? "<" : "");
        std::fprintf(stderr, "\n[dl-validate] caller backtrace:\n");
        void* fr[16];
        int nf = backtrace(fr, 16);
        backtrace_symbols_fd(fr, nf, 2);
        abort();
      }
    }
  }

  // Flush pending primitives
  if (*reinterpret_cast<u32*>(&__gx->vNum) != 0) {
    __GXSendFlushPrim();
  }

  // Filter unsupported raw opcodes out of the stream (CP_REG_ARRAYBASE_ID
  // and friends -- see sanitize_dl comment).
  u32 outLen = nbytes;
  const u8* src = static_cast<const u8*>(data);
  const u8* sanitized = sanitize_dl(src, nbytes, outLen);

  // SB_DL_VALIDATE=1 (post-sanitize): if the pre-sanitize scan above passed
  // and THIS one fails, sanitize_dl corrupted the stream (false-positive
  // 6-byte drop inside a payload).
  {
    static int s_val = -1;
    if (s_val < 0) {
      const char* e = getenv("SB_DL_VALIDATE");
      s_val = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    if (s_val == 1 && sanitized != src) {
      extern u32 aurora_gx_scan_dl(const u8*, u32, u32);
      const u32 bad = aurora_gx_scan_dl(sanitized, outLen, 1);
      if (bad != 0xFFFFFFFFu) {
        std::fprintf(stderr,
                     "[dl-validate] SANITIZE CORRUPTED DL dl=%p size=%u->%u off=%u (pre-sanitize scan was clean)\n",
                     data, nbytes, outLen, bad);
        for (u32 i = (bad > 16 ? bad - 16 : 0); i < bad + 16 && i < outLen; ++i)
          std::fprintf(stderr, " %02x%s", sanitized[i], i == bad ? "<" : "");
        std::fprintf(stderr, "\n");
        abort();
      }
    }
  }

  // Write display list contents to the FIFO
  aurora::gx::fifo::write_data(sanitized, outLen);
}

}
