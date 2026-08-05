#pragma once

// Triangle-index generation for GX primitives, split out of command_processor.cpp so it can be
// tested directly. It was `static` inside a 4,700-line TU that pulls in the whole WebGPU renderer,
// which is why the hot path that produces every index in the frame had no unit test.
//
// Templated on the buffer so a test can supply a trivial stand-in and still exercise THIS code
// rather than a copy of it. The buffer must provide:
//   uint8_t* append_uninitialized(size_t)   -- reserve n bytes, advance length, return write ptr
//   void append<T>(const T&)                -- append one object
//
// Cost note (SB_PROFILE_DRAWPRIM, Delfino): this ran at 64 ns per primitive across ~45k primitives
// a frame, 12% of draw_prim. The cost was structural rather than arithmetic — every index went in
// through its own capacity-checked append to move two bytes, so a quad (53% of this game's
// primitives) paid six of them to emit twelve. Each branch now sizes its output exactly, takes the
// tail pointer ONCE, and writes straight through it: N capacity checks per primitive become 1.
// The emitted indices are unchanged, in the same order — asserted byte-for-byte against the
// previous implementation in tests/gx_prim_index_test.cpp.

#include <cstdint>
#include <cstring>

namespace aurora::gx::fifo {

// Writes go through memcpy rather than a uint16_t* cast: the buffer is byte-addressed and nothing
// guarantees an even offset, so a u16 store would be undefined behaviour.
inline void idx_put(uint8_t*& out, uint16_t a, uint16_t b, uint16_t c) {
  const uint16_t tri[3] = {a, b, c};
  std::memcpy(out, tri, sizeof(tri));
  out += sizeof(tri);
}

inline void idx_put1(uint8_t*& out, uint16_t a) {
  std::memcpy(out, &a, sizeof(a));
  out += sizeof(a);
}

// `prim` is the GXPrimitive opcode; taken as a plain integer so this header does not drag in the
// GX headers. FatalFn is invoked for an unsupported primitive (fail fast — never emit nothing).
template <typename Buffer, typename FatalFn>
uint16_t prepare_idx_buffer_impl(Buffer& buf, uint32_t prim, uint16_t vtxStart, uint16_t vtxCount,
                                 uint32_t kQuads, uint32_t kTriangles, uint32_t kTriangleFan,
                                 uint32_t kTriangleStrip, uint32_t kLines, uint32_t kLineStrip,
                                 uint32_t kPoints, FatalFn&& fatal) {
  uint16_t numIndices = 0;
  if (prim == kQuads) {
    // CEILING, not `vtxCount / 4`. The loop below steps by 4 and runs once more for a trailing
    // partial group, emitting a full six indices for it — so a floor here under-reserves by 12
    // bytes and the last idx_put writes past the end of the buffer. The old code could not have
    // this bug: it reserved as a hint and each append re-checked capacity. Sizing up front means
    // the size has to match what the loop actually writes, and this is the failure mode of that
    // trade. Caught by the byte-for-byte test, invisible in a rendered frame because this game
    // only ever emits quads in multiples of 4.
    const uint32_t quads = (uint32_t(vtxCount) + 3u) / 4u;
    uint8_t* out = buf.append_uninitialized(quads * 6 * sizeof(uint16_t));
    for (uint16_t v = 0; v < vtxCount; v += 4) {
      const uint16_t idx0 = vtxStart + v;
      const uint16_t idx1 = vtxStart + v + 1;
      const uint16_t idx2 = vtxStart + v + 2;
      const uint16_t idx3 = vtxStart + v + 3;
      idx_put(out, idx0, idx1, idx2);
      idx_put(out, idx2, idx3, idx0);
      numIndices += 6;
    }
  } else if (prim == kTriangles) {
    uint8_t* out = buf.append_uninitialized(vtxCount * sizeof(uint16_t));
    for (uint16_t v = 0; v < vtxCount; ++v) {
      idx_put1(out, static_cast<uint16_t>(vtxStart + v));
      ++numIndices;
    }
  } else if (prim == kTriangleFan || prim == kTriangleStrip) {
    // Both emit the first three vertices as one triangle, then three indices per further vertex.
    // Sized exactly: the old `(vtxCount - 3) * 3 + 3` underflows for vtxCount < 3, which stayed
    // harmless only because over-reserving is invisible. append_uninitialized advances the
    // length, so its size has to be right.
    const uint32_t n = vtxCount <= 3 ? vtxCount : 3u + (uint32_t(vtxCount) - 3u) * 3u;
    uint8_t* out = buf.append_uninitialized(n * sizeof(uint16_t));
    const bool fan = prim == kTriangleFan;
    for (uint16_t v = 0; v < vtxCount; ++v) {
      const uint16_t idx = vtxStart + v;
      if (v < 3) {
        idx_put1(out, idx);
        ++numIndices;
        continue;
      }
      if (fan) {
        idx_put(out, vtxStart, static_cast<uint16_t>(idx - 1), idx);
      } else if ((v & 1) == 0) {
        idx_put(out, static_cast<uint16_t>(idx - 2), static_cast<uint16_t>(idx - 1), idx);
      } else {
        idx_put(out, static_cast<uint16_t>(idx - 1), static_cast<uint16_t>(idx - 2), idx);
      }
      numIndices += 3;
    }
  } else if (prim == kLines || prim == kLineStrip || prim == kPoints) {
    uint8_t* out = buf.append_uninitialized(6 * sizeof(uint16_t));
    idx_put(out, 0, 1, 3);
    idx_put(out, 3, 2, 0);
    numIndices = 6;
  } else {
    fatal(prim);
  }
  return numIndices;
}

} // namespace aurora::gx::fifo
