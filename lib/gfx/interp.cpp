// interp — see interp.hpp for the design and for the two approximations it makes.

#include "interp.hpp"

#include "../internal.hpp"

#include <cstring>
#include <unordered_map>
#include <vector>

namespace aurora::gfx::interp {
namespace {
static Module Log("aurora::interp");

// The two matrix spans a draw carries: 10 position matrices then, 480 bytes later, 10 normal
// matrices, each Mat3x4<float> = 12 floats.
constexpr uint32_t kMtxFloats = 10 * 12;
constexpr uint32_t kMtxBytes = kMtxFloats * sizeof(float);

// One draw's transform state, keyed by (tag, ordinal within that tag).
struct Entry {
  float pos[kMtxFloats];
  float nrm[kMtxFloats];
  uint32_t vtxCount;   // the consistency check — see below
};

// Two generations. `prev` is what the last tick recorded; `cur` is being filled by this tick and
// becomes `prev` at end_tick.
std::unordered_map<uint64_t, std::vector<Entry>> g_prev;
std::unordered_map<uint64_t, std::vector<Entry>> g_cur;

// How far into each tag we are THIS tick. Reset every tick; this is the ordinal that is scoped to a
// tag rather than global.
std::unordered_map<uint64_t, uint32_t> g_cursor;

long g_paired = 0;
long g_unpaired = 0;
long g_mismatched = 0;
} // namespace

void begin_tick() {
  g_cur.clear();
  g_cursor.clear();
}

void patch_draw(uint64_t tag, uint32_t vtxCount, const uint8_t* src, uint8_t* dst,
                uint32_t uniformSize, uint32_t mtxPosOffset, uint32_t mtxNrmOffset, float alpha) {
  if (tag == 0 || dst == nullptr || src == nullptr) {
    return; // untagged: no identity, so it snaps. Correct for 2D and immediate-mode geometry.
  }
  // A bounds failure here means the recorded offset does not describe this block — writing anyway
  // would scribble over the projection or past the block entirely. Refuse, loudly, once.
  if (mtxPosOffset + kMtxBytes > uniformSize || mtxNrmOffset + kMtxBytes > uniformSize) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      Log.error("matrix offsets do not fit the uniform block: pos {} nrm {} span {} but block is {} "
                "bytes. Refusing to patch; every draw will snap. This means the recorded offset and "
                "the recorded block disagree, which is a layout bug, not a tuning problem.",
                mtxPosOffset, mtxNrmOffset, kMtxBytes, uniformSize);
    }
    return;
  }

  const uint32_t ordinal = g_cursor[tag]++;

  auto& curVec = g_cur[tag];
  if (curVec.size() <= ordinal) {
    curVec.resize(ordinal + 1);
  }
  Entry& mine = curVec[ordinal];
  std::memcpy(mine.pos, src + mtxPosOffset, kMtxBytes);
  std::memcpy(mine.nrm, src + mtxNrmOffset, kMtxBytes);
  mine.vtxCount = vtxCount;

  const auto it = g_prev.find(tag);
  if (it == g_prev.end() || it->second.size() <= ordinal) {
    ++g_unpaired;   // new object this tick, or it drew fewer parts last tick: snap.
    return;
  }
  const Entry& was = it->second[ordinal];

  // THE CHECK THAT KEEPS THE ORDINAL HONEST. Pairing within a tag assumes the object replays the
  // same display list each tick. If it did not, the vertex counts differ, and interpolating between
  // two unrelated poses would smear the object across the screen — far worse than not interpolating
  // it. Snap instead, and count it, so a systematic mismatch shows up as a number rather than as an
  // unexplained visual artefact.
  if (was.vtxCount != vtxCount) {
    ++g_mismatched;
    return;
  }

  ++g_paired;
  const float a = alpha;
  const float b = 1.0f - alpha;
  auto* outPos = reinterpret_cast<float*>(dst + mtxPosOffset);
  auto* outNrm = reinterpret_cast<float*>(dst + mtxNrmOffset);
  for (uint32_t i = 0; i < kMtxFloats; ++i) {
    outPos[i] = was.pos[i] * b + mine.pos[i] * a;
    outNrm[i] = was.nrm[i] * b + mine.nrm[i] * a;
  }
}

void end_tick() { g_prev.swap(g_cur); }

void report() {
  const long total = g_paired + g_unpaired + g_mismatched;
  // Refuse rather than divide by nothing: "0% paired" from an empty run and "0% paired" from a
  // broken pairing are the same number, and only one of them is a defect.
  if (total == 0) {
    Log.warn("interpolation pairing: NO TAGGED DRAWS AT ALL. This says nothing about pairing — "
             "there was nothing to pair. Check that draw tags are being emitted.");
    return;
  }
  Log.info("interpolation pairing: {} of {} tagged draws paired with the previous tick ({:.1f}%); "
           "{} unpaired (new or newly-split object — snaps, correct); {} MISMATCHED vertex counts "
           "(snapped deliberately rather than smeared between two unrelated poses — a nonzero "
           "count here means some object does not replay a stable display list)",
           g_paired, total, 100.0 * (double)g_paired / (double)total, g_unpaired, g_mismatched);
}

} // namespace aurora::gfx::interp
