#include "gx.hpp"
#include "__gx.h"

#include "../../gx/fifo.hpp"

#include <cstring>
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
  while (pos < nbytes) {
    const u8 cmd = src[pos];
    // GX_CMD_LOAD_CP_REG (0x08): 1 opcode + 1 addr + 4 value = 6 bytes.
    if (cmd == 0x08 && pos + 6 <= nbytes) {
      const u8 addr = src[pos + 1];
      if (addr >= 0xA0 && addr <= 0xAF) {
        // Drop CP_REG_ARRAYBASE_ID writes entirely.
        pos += 6;
        dirty = true;
        continue;
      }
    }
    // Everything else falls through: copy remaining bytes as-is. We
    // conservatively copy byte-by-byte since we don't know the exact length
    // of every opcode from here without duplicating the processor.
    scratch.push_back(cmd);
    pos += 1;
  }
  outLen = static_cast<u32>(scratch.size());
  return dirty ? scratch.data() : src;
}

void GXCallDisplayList(const void* data, u32 nbytes) {
  // Flush any pending dirty state before calling
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
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

  // Write display list contents to the FIFO
  aurora::gx::fifo::write_data(sanitized, outLen);
}

}
