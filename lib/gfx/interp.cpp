// interp — see interp.hpp for the design and for the two approximations it makes.

#include "interp.hpp"

#include "../internal.hpp"

#include <cmath>
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

// How far a paired draw's TRANSLATION moved between the two ticks, in world units. This is the
// direct check on whether "the previous tick's matrices" really are that object's previous
// matrices: a real object moves a fraction of a unit per 1/30 s, so a mean in the hundreds means the
// pairing is handing back some other object's transform. The smoothness metric cannot see this — a
// large but CONSISTENT displacement reads as perfectly even motion, which is precisely its declared
// blind spot.
double g_transDeltaSum = 0.0;
double g_transDeltaMax = 0.0;
long g_transDeltaN = 0;

// ---- CAMERA ------------------------------------------------------------------------------------
// A GC Mtx as it arrives: 3 rows of 4 floats, p' = R*p + t with R the leading 3x3 and t the last
// column. Aurora's Mat3x4 stores the same rows in the same order (see the XF load in
// command_processor.cpp), so these arrays and the uniform bytes are the same layout.
float g_viewCur[12];
float g_viewPrev[12];
bool g_haveViewCur = false;
bool g_haveViewPrev = false;
float g_camDelta[12];      // V_lerp * V_cur^-1
bool g_camDeltaValid = false;
long g_cameraPatched = 0;

// out = a ∘ b, i.e. apply b then a. Rows of a select rows of b.
void compose(const float* a, const float* b, float* out) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out[r * 4 + c] = a[r * 4 + 0] * b[0 * 4 + c] + a[r * 4 + 1] * b[1 * 4 + c] +
                       a[r * 4 + 2] * b[2 * 4 + c];
    }
    out[r * 4 + 3] = a[r * 4 + 0] * b[0 * 4 + 3] + a[r * 4 + 1] * b[1 * 4 + 3] +
                     a[r * 4 + 2] * b[2 * 4 + 3] + a[r * 4 + 3];
  }
}

// Affine inverse: [R|t] -> [R^-1 | -R^-1 t]. A general 3x3 inverse rather than a transpose, because
// a view matrix is only a rotation if nothing upstream has scaled it, and assuming that would fail
// silently and subtly rather than loudly. False if singular.
bool affine_inverse(const float* m, float* out) {
  const float a = m[0], b = m[1], c = m[2];
  const float d = m[4], e = m[5], f = m[6];
  const float g = m[8], h = m[9], i = m[10];
  const float A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const float det = a * A + b * B + c * C;
  if (det > -1e-12f && det < 1e-12f) {
    return false;
  }
  const float inv = 1.0f / det;
  float r[9];
  r[0] = A * inv;                    r[1] = -(b * i - c * h) * inv;  r[2] = (b * f - c * e) * inv;
  r[3] = B * inv;                    r[4] = (a * i - c * g) * inv;   r[5] = -(a * f - c * d) * inv;
  r[6] = C * inv;                    r[7] = -(a * h - b * g) * inv;  r[8] = (a * e - b * d) * inv;
  const float tx = m[3], ty = m[7], tz = m[11];
  for (int row = 0; row < 3; ++row) {
    out[row * 4 + 0] = r[row * 3 + 0];
    out[row * 4 + 1] = r[row * 3 + 1];
    out[row * 4 + 2] = r[row * 3 + 2];
    out[row * 4 + 3] = -(r[row * 3 + 0] * tx + r[row * 3 + 1] * ty + r[row * 3 + 2] * tz);
  }
  return true;
}
} // namespace

void begin_tick() {
  g_cur.clear();
  g_cursor.clear();
}

bool patch_draw(uint64_t tag, uint32_t vtxCount, const uint8_t* src, uint8_t* dst,
                uint32_t uniformSize, uint32_t mtxPosOffset, uint32_t mtxNrmOffset, float alpha) {
  if (tag == 0 || dst == nullptr || src == nullptr) {
    return false; // untagged: no identity of its own. The caller applies the camera delta instead.
  }
  // A matrix block NEVER starts at offset 0 — vtxStart, the active matrix index, the viewport and
  // the vertex-array ranges all precede it. So 0 means the offset was never recorded, and patching
  // there rewrites the block header while READING the viewport as a matrix. That is exactly what
  // happened when this assignment was silently lost in an edit: every draw "interpolated" plausible
  // constants, the pairing statistics read 99.5%, and only a translation-delta of exactly 0.000
  // across 500k samples gave it away. A default that renders as corruption must be loud.
  if (mtxPosOffset == 0 || mtxNrmOffset == 0) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      Log.error("matrix offset is 0, which cannot be a real uniform-block matrix offset — the "
                "block header always precedes it. build_uniform is not recording it. Interpolation "
                "is OFF rather than corrupting every draw.");
    }
    return false;
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
    return false;
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
    ++g_unpaired;   // new object this tick, or it drew fewer parts last tick.
    return false;
  }
  const Entry& was = it->second[ordinal];

  // THE CHECK THAT KEEPS THE ORDINAL HONEST. Pairing within a tag assumes the object replays the
  // same display list each tick. If it did not, the vertex counts differ, and interpolating between
  // two unrelated poses would smear the object across the screen — far worse than not interpolating
  // it. Snap instead, and count it, so a systematic mismatch shows up as a number rather than as an
  // unexplained visual artefact.
  if (was.vtxCount != vtxCount) {
    ++g_mismatched;
    return false;
  }

  ++g_paired;
  {
    const float dx = mine.pos[3] - was.pos[3];
    const float dy = mine.pos[7] - was.pos[7];
    const float dz = mine.pos[11] - was.pos[11];
    const double d = std::sqrt((double)(dx * dx + dy * dy + dz * dz));
    g_transDeltaSum += d;
    if (d > g_transDeltaMax) { g_transDeltaMax = d; }
    ++g_transDeltaN;
  }
  const float a = alpha;
  const float b = 1.0f - alpha;
  auto* outPos = reinterpret_cast<float*>(dst + mtxPosOffset);
  auto* outNrm = reinterpret_cast<float*>(dst + mtxNrmOffset);
  for (uint32_t i = 0; i < kMtxFloats; ++i) {
    outPos[i] = was.pos[i] * b + mine.pos[i] * a;
    outNrm[i] = was.nrm[i] * b + mine.nrm[i] * a;
  }
  return true;
}

void set_view_matrix(const float m[12]) {
  std::memcpy(g_viewCur, m, sizeof(g_viewCur));
  g_haveViewCur = true;
}

bool begin_camera_delta(float alpha) {
  g_camDeltaValid = false;
  if (!g_haveViewCur || !g_haveViewPrev) {
    return false;   // first tick, or the emitter is not supplying a view: nothing to interpolate
  }
  float lerped[12];
  for (int i = 0; i < 12; ++i) {
    lerped[i] = g_viewPrev[i] * (1.0f - alpha) + g_viewCur[i] * alpha;
  }
  float invCur[12];
  if (!affine_inverse(g_viewCur, invCur)) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      Log.error("this tick's view matrix is singular, so V_lerp*V_cur^-1 does not exist. Camera "
                "interpolation is OFF for unpaired draws, which means they will render from the "
                "current viewpoint while paired ones do not — expect the frame to tear.");
    }
    return false;
  }
  compose(lerped, invCur, g_camDelta);
  g_camDeltaValid = true;
  return true;
}

void patch_camera_only(uint8_t* dst, uint32_t uniformSize, uint32_t mtxPosOffset,
                       uint32_t mtxNrmOffset) {
  if (!g_camDeltaValid || dst == nullptr) {
    return;
  }
  if (mtxPosOffset == 0 || mtxNrmOffset == 0 || mtxPosOffset + kMtxBytes > uniformSize ||
      mtxNrmOffset + kMtxBytes > uniformSize) {
    return;   // already reported by patch_draw's identical checks
  }
  auto* pos = reinterpret_cast<float*>(dst + mtxPosOffset);
  auto* nrm = reinterpret_cast<float*>(dst + mtxNrmOffset);
  for (int slot = 0; slot < 10; ++slot) {
    float out[12];
    compose(g_camDelta, pos + slot * 12, out);
    std::memcpy(pos + slot * 12, out, sizeof(out));
    // Normals take the rotation only — no translation. Correct for a rigid view delta, because the
    // inverse-transpose of a rotation is the rotation itself.
    compose(g_camDelta, nrm + slot * 12, out);
    out[3] = nrm[slot * 12 + 3];
    out[7] = nrm[slot * 12 + 7];
    out[11] = nrm[slot * 12 + 11];
    std::memcpy(nrm + slot * 12, out, sizeof(out));
  }
  ++g_cameraPatched;
}

void end_tick() {
  g_prev.swap(g_cur);
  if (g_haveViewCur) {
    std::memcpy(g_viewPrev, g_viewCur, sizeof(g_viewPrev));
    g_haveViewPrev = true;
  }
}

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
  Log.info("paired-draw translation delta between ticks: mean {:.3f} max {:.3f} world units over {} "
           "samples. A real object moves a fraction of a unit per 1/30 s — a mean in the hundreds "
           "means pairing is returning some OTHER transform, which the smoothness metric cannot "
           "see because a large consistent displacement still reads as even motion.",
           g_transDeltaN ? g_transDeltaSum / (double)g_transDeltaN : 0.0, g_transDeltaMax,
           g_transDeltaN);
  // The camera line is separate because its failure is separate: pairing can be perfect while the
  // frame still tears, if unpaired draws are left at the current viewpoint.
  if (!g_haveViewCur) {
    Log.warn("camera interpolation: NO VIEW MATRIX SUPPLIED. Every unpaired draw keeps the CURRENT "
             "viewpoint while paired draws move to the in-between one, so the frame renders from "
             "two viewpoints at once. This is the failure that makes partial coverage worse than "
             "none — emit GX_AURORA_VIEW_MTX.");
  } else {
    Log.info("camera interpolation: view matrix supplied, {} draw uniforms carried the camera delta",
             g_cameraPatched);
  }
}

} // namespace aurora::gfx::interp
