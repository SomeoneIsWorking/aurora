// Equivalence test for triangle-index generation (lib/gx/prim_index.hpp).
//
// prepare_idx_buffer was rewritten for speed: it used to append every index through its own
// capacity-checked ByteBuffer::append (six calls to emit a quad's twelve bytes) and now sizes the
// output, takes the tail pointer once, and writes straight through. That is a pure performance
// change, so the bar is not "the frame still looks right" — it is that the emitted bytes are
// IDENTICAL. A frame mean can absorb a scrambled triangle; this cannot.
//
// The reference below is the previous implementation, transcribed verbatim from
// command_processor.cpp before the rewrite. The subject is the SHIPPING header, included directly
// rather than copied, so this test fails if the real code drifts from it.

#include <gtest/gtest.h>

#include "../lib/gx/prim_index.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// GXPrimitive opcode values (GXEnum.h), passed as plain integers by the header.
constexpr uint32_t kQuads = 0x80;
constexpr uint32_t kTriangles = 0x90;
constexpr uint32_t kTriangleStrip = 0x98;
constexpr uint32_t kTriangleFan = 0xA0;
constexpr uint32_t kLines = 0xA8;
constexpr uint32_t kLineStrip = 0xB0;
constexpr uint32_t kPoints = 0xB8;

// Minimal stand-in for ByteBuffer exposing just the two operations the header needs.
struct TestBuffer {
  std::vector<uint8_t> bytes;

  uint8_t* append_uninitialized(size_t size) {
    const size_t at = bytes.size();
    bytes.resize(at + size);
    return bytes.data() + at;
  }
  template <typename T>
  void append(const T& obj) {
    const size_t at = bytes.size();
    bytes.resize(at + sizeof(T));
    std::memcpy(bytes.data() + at, &obj, sizeof(T));
  }
};

// ---------------------------------------------------------------------------
// REFERENCE: the implementation as it stood before the rewrite, unchanged apart
// from using TestBuffer and integer primitive constants.
// ---------------------------------------------------------------------------
uint16_t prepare_idx_buffer_reference(TestBuffer& buf, uint32_t prim, uint16_t vtxStart,
                                      uint16_t vtxCount) {
  uint16_t numIndices = 0;
  if (prim == kQuads) {
    for (uint16_t v = 0; v < vtxCount; v += 4) {
      uint16_t idx0 = vtxStart + v;
      uint16_t idx1 = vtxStart + v + 1;
      uint16_t idx2 = vtxStart + v + 2;
      uint16_t idx3 = vtxStart + v + 3;

      buf.append(idx0);
      buf.append(idx1);
      buf.append(idx2);
      numIndices += 3;

      buf.append(idx2);
      buf.append(idx3);
      buf.append(idx0);
      numIndices += 3;
    }
  } else if (prim == kTriangles) {
    for (uint16_t v = 0; v < vtxCount; ++v) {
      const uint16_t idx = vtxStart + v;
      buf.append(idx);
      ++numIndices;
    }
  } else if (prim == kTriangleFan) {
    for (uint16_t v = 0; v < vtxCount; ++v) {
      const uint16_t idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      buf.append(vtxStart);
      buf.append(static_cast<uint16_t>(idx - 1));
      buf.append(idx);
      numIndices += 3;
    }
  } else if (prim == kTriangleStrip) {
    for (uint16_t v = 0; v < vtxCount; ++v) {
      const uint16_t idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      if ((v & 1) == 0) {
        buf.append(static_cast<uint16_t>(idx - 2));
        buf.append(static_cast<uint16_t>(idx - 1));
        buf.append(idx);
      } else {
        buf.append(static_cast<uint16_t>(idx - 1));
        buf.append(static_cast<uint16_t>(idx - 2));
        buf.append(idx);
      }
      numIndices += 3;
    }
  } else if (prim == kLines || prim == kLineStrip || prim == kPoints) {
    buf.append<uint16_t>(0);
    buf.append<uint16_t>(1);
    buf.append<uint16_t>(3);
    buf.append<uint16_t>(3);
    buf.append<uint16_t>(2);
    buf.append<uint16_t>(0);
    numIndices = 6;
  }
  return numIndices;
}

uint16_t run_subject(TestBuffer& buf, uint32_t prim, uint16_t vtxStart, uint16_t vtxCount) {
  return aurora::gx::fifo::prepare_idx_buffer_impl(
      buf, prim, vtxStart, vtxCount, kQuads, kTriangles, kTriangleFan, kTriangleStrip, kLines,
      kLineStrip, kPoints, [](uint32_t p) { ADD_FAILURE() << "unsupported primitive " << p; });
}

struct Case {
  uint32_t prim;
  const char* name;
};
constexpr Case kCases[] = {
    {kQuads, "QUADS"},       {kTriangles, "TRIANGLES"}, {kTriangleFan, "TRIANGLEFAN"},
    {kTriangleStrip, "TRIANGLESTRIP"}, {kLines, "LINES"}, {kLineStrip, "LINESTRIP"},
    {kPoints, "POINTS"},
};

} // namespace

// The whole point: byte-for-byte agreement across every primitive type and a vertex-count sweep
// that covers the sizes this game actually emits (3, 4 and 5-6 verts are 87% of primitives) as
// well as the boundaries where the two implementations size their output differently.
TEST(GxPrimIndex, MatchesPreviousImplementationByteForByte) {
  for (const auto& c : kCases) {
    // Quads step by 4; the others are meaningful from 0 up. Include 0/1/2 because the old
    // fan/strip size expression `(vtxCount - 3) * 3 + 3` underflows there.
    for (uint16_t n = 0; n <= 64; ++n) {
      for (uint16_t start : {uint16_t{0}, uint16_t{1}, uint16_t{1000}, uint16_t{65000}}) {
        TestBuffer refBuf, subBuf;
        const uint16_t refN = prepare_idx_buffer_reference(refBuf, c.prim, start, n);
        const uint16_t subN = run_subject(subBuf, c.prim, start, n);

        EXPECT_EQ(refN, subN) << c.name << " index count differs at vtxCount=" << n
                              << " vtxStart=" << start;
        EXPECT_EQ(refBuf.bytes, subBuf.bytes)
            << c.name << " emitted bytes differ at vtxCount=" << n << " vtxStart=" << start;
      }
    }
  }
}

// The buffer length must equal the reported index count times two. The rewrite advances the
// length up front from a size it computes itself, so a size expression that disagrees with the
// loop would silently leave uninitialised bytes in the index stream -- garbage triangles from
// whatever the allocator last held, which is exactly the kind of defect a frame mean hides.
TEST(GxPrimIndex, ReportedCountMatchesBytesWritten) {
  for (const auto& c : kCases) {
    for (uint16_t n = 0; n <= 64; ++n) {
      TestBuffer buf;
      const uint16_t count = run_subject(buf, c.prim, 0, n);
      EXPECT_EQ(buf.bytes.size(), static_cast<size_t>(count) * sizeof(uint16_t))
          << c.name << " wrote " << buf.bytes.size() << " bytes but reported " << count
          << " indices at vtxCount=" << n;
    }
  }
}

// Appending into a buffer that is already at an ODD offset must still work. The rewrite writes
// through a uint8_t* with memcpy for exactly this reason; a uint16_t* store would be undefined
// behaviour here and would crash or silently misalign on a strict-alignment target.
TEST(GxPrimIndex, HandlesUnalignedBufferOffset) {
  for (const auto& c : kCases) {
    TestBuffer refBuf, subBuf;
    refBuf.bytes.push_back(0xAB); // one byte -> every following u16 sits at an odd offset
    subBuf.bytes.push_back(0xAB);
    const uint16_t refN = prepare_idx_buffer_reference(refBuf, c.prim, 7, 12);
    const uint16_t subN = run_subject(subBuf, c.prim, 7, 12);
    EXPECT_EQ(refN, subN) << c.name;
    EXPECT_EQ(refBuf.bytes, subBuf.bytes) << c.name << " differs when starting at an odd offset";
  }
}

// A POSITIVE CONTROL for the comparison itself. The two tests above assert that reference and
// subject agree; if the harness were wired so that both sides ran the same code, or the buffers
// were compared before being filled, they would pass while checking nothing. Deliberately perturb
// one index and require the comparison to NOTICE.
TEST(GxPrimIndex, ComparisonDetectsADeliberateDifference) {
  TestBuffer refBuf, subBuf;
  prepare_idx_buffer_reference(refBuf, kQuads, 0, 8);
  run_subject(subBuf, kQuads, 0, 8);
  ASSERT_EQ(refBuf.bytes, subBuf.bytes) << "precondition: unperturbed output must match";
  ASSERT_FALSE(subBuf.bytes.empty());

  subBuf.bytes[0] ^= 0xFF;
  EXPECT_NE(refBuf.bytes, subBuf.bytes)
      << "the byte comparison cannot detect a corrupted index, so the tests above prove nothing";
}
