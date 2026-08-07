// interp — see interp.hpp for the design and for the two approximations it makes.

#include "interp.hpp"

#include <cstdlib>

#include "../internal.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// The denominators for the eye-space texgen gate. Declared rather than included: gx.hpp pulls in the
// whole GX state machine, and this file deliberately depends on nothing but the uniform bytes it is
// handed. They are DEFINED beside the gate that increments them (gx/command_processor.cpp) so the
// numbers and the decision they describe cannot drift apart.
namespace aurora::gx::fifo {
extern long g_texgenPosSourced;
extern long g_texgenRejectIndexed;
extern long g_texgenEyeSpace;
} // namespace aurora::gx::fifo

namespace aurora::gfx::interp {
namespace {
static Module Log("aurora::interp");

// The two matrix spans a draw carries: 10 position matrices then, 480 bytes later, 10 normal
// matrices, each Mat3x4<float> = 12 floats.
constexpr uint32_t kMtxFloats = 10 * 12;
constexpr uint32_t kMtxBytes = kMtxFloats * sizeof(float);
// ONE matrix, as opposed to kMtxBytes which is the whole ten-slot block. The texture matrices are
// addressed individually.
constexpr uint32_t kOneMtxBytes = 12 * sizeof(float);

// One draw's transform state, keyed by (tag, ordinal within that tag).
struct Entry {
  float pos[kMtxFloats];
  float nrm[kMtxFloats];
  uint32_t vtxCount;   // the consistency check — see below
  // A position-sourced texture matrix with this draw's model-view divided back out — see the
  // texgen block in patch_draw. Kept across ticks because whether it is CONSTANT is the whole
  // discriminator between a camera projection (which must follow the interpolated pose) and an
  // object-locked mapping (which must not).
  float texA[12];
  uint32_t texIdx = 0;      // which postex_mtx entry texA came from; 0 = none recorded
  bool texValid = false;
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

// THE SAME DELTA WITH THE CAMERA REMOVED. The number above is measured on pnMtx, which is
// model x view, so it cannot tell "this object moved 26911 units" from "the camera cut and every
// object in the frame moved with it". Those have opposite fixes — the first is a pairing defect
// (a tag collision handing back another object's transform), the second is a continuity defect
// (a tick with no meaningful in-between, which must snap rather than lerp) — so an instrument that
// reports their SUM points at neither. Removing the view from both sides gives the object's own
// world transform, and the two numbers side by side name which one is happening.
double g_objDeltaSum = 0.0;
double g_objDeltaMax = 0.0;
long g_objDeltaN = 0;
// A mean and a max cannot tell "every object is slightly wrong" from "one object is catastrophically
// wrong and the rest are fine", and those have different causes. The histogram separates them, and
// the worst-offender list carries the TAG, which names the guest shape and instance so the object
// can actually be identified rather than merely counted.
constexpr int kObjBuckets = 7;   // [0,0.1) [0.1,1) [1,10) [10,100) [100,1k) [1k,10k) [10k,inf)
long g_objHist[kObjBuckets] = {};
constexpr int kWorstDraws = 6;
struct WorstDraw { double delta = -1.0; uint64_t tag = 0; uint32_t ordinal = 0; long tick = -1; };
WorstDraw g_worstDraw[kWorstDraws];
// Counted separately because "attribution was not available" and "attribution says zero" must not
// look alike: on the first tick, or any tick with no previous view, the camera cannot be removed.
long g_objDeltaUnavailable = 0;

// ---- CAMERA ------------------------------------------------------------------------------------
// A GC Mtx as it arrives: 3 rows of 4 floats, p' = R*p + t with R the leading 3x3 and t the last
// column. Aurora's Mat3x4 stores the same rows in the same order (see the XF load in
// command_processor.cpp), so these arrays and the uniform bytes are the same layout.
float g_viewCur[12];
float g_viewPrev[12];
bool g_haveViewCur = false;
bool g_haveViewPrev = false;
float g_camDelta[12];      // V_lerp * V_cur^-1
float g_viewLerp[12];      // V_lerp itself; needed to rotate a WORLD-space delta into eye space

// ── BILLBOARD POSITIONS ─────────────────────────────────────────────────────────────────────────
//
// A JPA particle's draw bakes its position into the VERTEX stream — the CPU does
// getGlobalPosition() then MTXMultVec(view) and emits eye-space corners — so there is no matrix for
// patch_draw to interpolate and tagging one achieves nothing on its own. But a billboard does not
// DEFORM between ticks, it translates: the quad's shape comes from the particle's scale and the
// position is one point added to every corner. So the whole correction is a translation, and the
// position matrix for these draws is identity and otherwise unused.
//
// The recomp records each tagged object's WORLD position once per tick (set_tag_world_pos). World,
// not eye: an eye-space pair would be expressed in two different view transforms and their
// difference would fold camera motion into the object's own motion.
struct BillboardPos {
  float cur[3]{};
  float prev[3]{};
  uint64_t stampCur = 0;
  uint64_t stampPrev = 0;
};
std::unordered_map<uint64_t, BillboardPos> g_billboard;
bool g_camDeltaValid = false;
long g_cameraPatched = 0;
// Why a camera patch did NOT happen. patch_draw returns at its tag check for an untagged draw, so
// its offset diagnostic is never reached for exactly the draws that rely on this path — the skip
// below was therefore completely silent, and a silently-skipped draw renders from the CURRENT
// viewpoint while the rest of the frame is at the in-between one.
long g_billboardPatched = 0;
long g_billboardUnpaired = 0;
long g_camRefusedNoDelta = 0;
long g_camRefusedBadOffset = 0;
// Eye-space texture matrices carried to the interpolated viewpoint, and the offset refusals. These
// are the numerator; the denominators live in command_processor.cpp with the gate that produces the
// mask, because "0 patched" means one thing when no draw was a candidate and the opposite when many
// were and the gate rejected them all.
long g_texMtxPatched = 0;
long g_texMtxRefusedBadOffset = 0;
// The paired path's texture-matrix split. Stable vs unstable is the measured answer to "does this
// texture matrix contain the camera", taken per draw across ticks rather than assumed; both are
// printed because a run that is all-unstable and a run that is all-stable would otherwise look the
// same from the numerator alone.
long g_texStable = 0;
long g_texUnstable = 0;
long g_texNoPrev = 0;
long g_texSingular = 0;
long g_texMultiSlot = 0;

// The inverse views, kept for ATTRIBUTION rather than for patching: M = V^-1 * (V*M) recovers an
// object's own transform from the matrix the draw actually carries.
float g_invViewCur[12];
float g_invViewPrev[12];
bool g_haveInvCur = false;
bool g_haveInvPrev = false;

// ---- PER-TICK CAMERA MOTION --------------------------------------------------------------------
// The camera's own step between ticks, which is the quantity the cut hypothesis is about. Kept as a
// DISTRIBUTION, not a mean: a cut is by definition rare, and a rare huge value is invisible in a
// mean taken over hundreds of ticks. If cuts are real the histogram is bimodal — a dense bulk of
// ordinary camera motion and a handful of entries decades away. If it is unimodal, there were no
// cuts in this run and the doubled-energy residual has some other cause.
constexpr int kCamBuckets = 6;   // [0,1) [1,10) [10,100) [100,1k) [1k,10k) [10k,inf)
long g_camHist[kCamBuckets] = {};
double g_camEyeSum = 0.0;
double g_camEyeMax = 0.0;
double g_camRotSumDeg = 0.0;
double g_camRotMaxDeg = 0.0;
long g_camN = 0;
long g_tickIndex = 0;
// The worst few ticks by camera step, kept with their tick index so a cut can be located in a run
// rather than merely known to exist.
constexpr int kWorstTicks = 5;
struct WorstTick { long tick = -1; double eye = -1.0; double rotDeg = 0.0; };
WorstTick g_worst[kWorstTicks];

// A short history of eye positions, so a large step can be shown IN CONTEXT rather than as a bare
// magnitude. See the use site for why the context is the whole point.
constexpr int kEyeRing = 3;
struct EyeSample { long tick = -1; float x = 0, y = 0, z = 0; };
EyeSample g_eyeRing[kEyeRing];
int g_eyeRingPos = 0;
int g_eyeFollow = 0;
int g_eyeCasesPrinted = 0;

void note_worst_tick(long tick, double eye, double rotDeg) {
  int slot = -1;
  double lowest = eye;
  for (int i = 0; i < kWorstTicks; ++i) {
    if (g_worst[i].eye < lowest) {
      lowest = g_worst[i].eye;
      slot = i;
    }
  }
  if (slot >= 0) {
    g_worst[slot] = WorstTick{tick, eye, rotDeg};
  }
}

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
                uint32_t uniformSize, uint32_t mtxPosOffset, uint32_t mtxNrmOffset, float alpha,
                uint32_t texMtxCamMask, uint32_t pnMtxSlot) {
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
  // ATTRIBUTION: the same delta with the camera divided out. pnMtx is V*M, so M = V^-1*(V*M), and
  // the translation of M is the object's own world position. Comparing THOSE across the two ticks
  // separates "this object moved" from "the whole world moved because the camera did", which the
  // delta above cannot do. Slot 0 on both sides, matching the delta above.
  if (g_haveInvCur && g_haveInvPrev) {
    float objCur[12];
    float objPrev[12];
    compose(g_invViewCur, mine.pos, objCur);
    compose(g_invViewPrev, was.pos, objPrev);
    const double ox = objCur[3] - objPrev[3];
    const double oy = objCur[7] - objPrev[7];
    const double oz = objCur[11] - objPrev[11];
    const double od = std::sqrt(ox * ox + oy * oy + oz * oz);
    g_objDeltaSum += od;
    if (od > g_objDeltaMax) { g_objDeltaMax = od; }
    ++g_objDeltaN;
    {
      int bucket = 0;
      for (double edge = 0.1; bucket < kObjBuckets - 1 && od >= edge; edge *= 10.0) {
        ++bucket;
      }
      ++g_objHist[bucket];
      int slot = -1;
      double lowest = od;
      for (int i = 0; i < kWorstDraws; ++i) {
        if (g_worstDraw[i].delta < lowest) {
          lowest = g_worstDraw[i].delta;
          slot = i;
        }
      }
      if (slot >= 0) {
        g_worstDraw[slot] = WorstDraw{od, tag, ordinal, g_tickIndex};
      }
    }
  } else {
    ++g_objDeltaUnavailable;
  }
  const float a = alpha;
  const float b = 1.0f - alpha;
  auto* outPos = reinterpret_cast<float*>(dst + mtxPosOffset);
  auto* outNrm = reinterpret_cast<float*>(dst + mtxNrmOffset);
  for (uint32_t i = 0; i < kMtxFloats; ++i) {
    outPos[i] = was.pos[i] * b + mine.pos[i] * a;
    outNrm[i] = was.nrm[i] * b + mine.nrm[i] * a;
  }

  // ── POSITION-SOURCED TEXTURE MATRICES ────────────────────────────────────────────────────────
  //
  // The block above moved this draw's geometry to its in-between pose. A texgen sourced from
  // GX_TG_POS reads the RAW vertex, so its UVs did not move with it. Where the texture matrix is a
  // projection through the camera, the two are now a full tick apart and the projected image slides
  // across the surface while the camera moves.
  //
  // The correction needs no camera delta, because for a PAIRED draw the interpolated model-view is
  // already computed. Write the texture matrix as
  //
  //     texmtx = A * pnMtx          so     A = texmtx * pnMtx^-1
  //
  // and the in-between value is `A * pnMtx_lerp` — the same A, re-composed with the pose the
  // geometry is actually being drawn at. Exact, not an approximation: if the decomposition holds,
  // this is what the game itself would have built at the in-between camera.
  //
  // IT DOES NOT ALWAYS HOLD, AND THAT IS THE POINT. An object-locked projection — a decal, a shadow
  // map baked in object space — has texmtx = A' * M with no view in it, and its UVs are CORRECT
  // unchanged when the camera moves; rewriting it would be the corruption rather than the fix.
  // The two cases are indistinguishable from one frame's state, so this does not guess: it recovers
  // A each tick and applies the correction only where A came out THE SAME as last tick. A really is
  // constant for a camera projection (A is the projection, which does not change) and provably is
  // not for an object-locked mapping (there A = A' * M * pnMtx^-1 = A' * V^-1, which moves with the
  // camera). Both classes are counted below, so the split is visible rather than assumed.
  // SBR_INTERP_TEXMTX=0 — leave position-sourced texture matrices on the TICK's pose. The A/B for
  // this correction: with it off the projected image is frozen while the surface under it moves,
  // which is the defect; with it on the two travel together. Kept because a correction that cannot
  // be switched off cannot be shown to have done anything.
  static const bool texMtxEnabled = [] {
    const char* e = std::getenv("SBR_INTERP_TEXMTX");
    return e == nullptr || e[0] != '0';
  }();
  if (texMtxEnabled && texMtxCamMask != 0 && pnMtxSlot < 10) {
    const uint32_t bits = texMtxCamMask & ~((1u << 10) - 1u);
    if (bits != 0 && (bits & (bits - 1)) != 0) {
      ++g_texMultiSlot; // more than one; this handles the first and says so rather than silently
    }
    int idx = -1;
    for (int k = 10; k < 32; ++k) {
      if ((bits & (1u << k)) != 0) { idx = k; break; }
    }
    const uint32_t off = mtxPosOffset + static_cast<uint32_t>(idx) * kOneMtxBytes;
    if (idx >= 0 && off + kOneMtxBytes <= uniformSize) {
      float invPn[12];
      const float* pnCur = mine.pos + pnMtxSlot * 12;
      if (!affine_inverse(pnCur, invPn)) {
        ++g_texSingular; // a degenerate model-view; nothing to divide out
      } else {
        compose(reinterpret_cast<const float*>(src + off), invPn, mine.texA);
        mine.texIdx = static_cast<uint32_t>(idx);
        mine.texValid = true;
        if (!was.texValid || was.texIdx != mine.texIdx) {
          ++g_texNoPrev;
        } else {
          // Relative tolerance: A carries a projection whose entries are order 1 but need not be,
          // and an absolute epsilon would classify by scale rather than by stability.
          float scale = 1.0f;
          float diff = 0.0f;
          for (int e = 0; e < 12; ++e) {
            const float m0 = std::fabs(mine.texA[e]);
            if (m0 > scale) { scale = m0; }
            const float d = std::fabs(mine.texA[e] - was.texA[e]);
            if (d > diff) { diff = d; }
          }
          if (diff / scale > 1e-3f) {
            ++g_texUnstable; // object-locked, or A genuinely animating: leave it alone
          } else {
            float lerpPn[12];
            for (int e = 0; e < 12; ++e) {
              lerpPn[e] = was.pos[pnMtxSlot * 12 + e] * b + mine.pos[pnMtxSlot * 12 + e] * a;
            }
            float out[12];
            compose(mine.texA, lerpPn, out);
            std::memcpy(dst + off, out, sizeof(out));
            ++g_texStable;
          }
        }
      }
    }
  }
  return true;
}

// ── THE AUDIT ───────────────────────────────────────────────────────────────────────────────────
namespace {
constexpr int kMaxPop = 16;
long g_audit[kMaxPop][(int)Disposition::Count] = {};
std::string g_popName[kMaxPop];
// A draw is noted Pending first and then, if something claims it, noted again. Resolving Pending
// into SnappedNoIdentity at report time (rather than counting it twice) is what keeps the columns
// summing to the draw count.
} // namespace

void name_population(uint8_t pop, const char* name) {
  if (pop < kMaxPop && name != nullptr) {
    g_popName[pop] = name;
  }
}

void note_disposition(uint8_t pop, Disposition d) {
  if (pop >= kMaxPop) {
    pop = 0;
  }
  if (d != Disposition::Pending) {
    // This draw was claimed, so retire the Pending note taken a moment ago.
    if (g_audit[pop][(int)Disposition::Pending] > 0) {
      --g_audit[pop][(int)Disposition::Pending];
    }
  }
  ++g_audit[pop][(int)d];
}

void report_audit() {
  static const char* kName[(int)Disposition::Count] = {
      "unclaimed", "PAIRED", "billboard", "camera-only", "snap:2D", "snap:NO-ID"};
  long total = 0;
  for (int p = 0; p < kMaxPop; ++p) {
    for (int d = 0; d < (int)Disposition::Count; ++d) {
      total += g_audit[p][d];
    }
  }
  if (total == 0) {
    Log.warn("interpolation audit: NO DRAWS WERE FILED AT ALL. This says nothing about coverage — "
             "nothing was classified. Check the run is rendering with interpolation on.");
    return;
  }
  Log.info("INTERPOLATION AUDIT — every draw, by the system that emitted it and the fate it got. "
           "PAIRED and billboard interpolate; snap:2D is CORRECT (a screen-space element has no "
           "in-between); camera-only and snap:NO-ID are the defects — geometry that follows the "
           "camera but not its own motion, or nothing at all.");
  Log.info("  {:<22} {:>10} {:>11} {:>12} {:>10} {:>11}  {}", "population", "PAIRED", "billboard",
           "camera-only", "snap:2D", "snap:NO-ID", "verdict");
  for (int p = 0; p < kMaxPop; ++p) {
    long sum = 0;
    for (int d = 0; d < (int)Disposition::Count; ++d) {
      sum += g_audit[p][d];
    }
    if (sum == 0) {
      continue;
    }
    // Anything still Pending was never claimed by any patch: perspective, no identity.
    const long noId = g_audit[p][(int)Disposition::SnappedNoIdentity] +
                      g_audit[p][(int)Disposition::Pending];
    const long good = g_audit[p][(int)Disposition::Paired] + g_audit[p][(int)Disposition::Billboard];
    // CameraOnlyStatic is CORRECT, not a shortfall: geometry that did not move needs the camera
    // delta and nothing else. Counting it as a defect is what made "97.3% PARTIAL" understate the
    // world-geometry row.
    const long bad = g_audit[p][(int)Disposition::CameraOnly] + noId;
    // A 2D population with no interpolated draws is CORRECT, not a failure — a screen-space
    // element has no meaningful in-between. Saying "interpolates (0.0% move)" of it, as the first
    // version did, is the report contradicting itself in one line.
    const char* verdict = (bad == 0 && good == 0) ? "CORRECT (2D: no in-between exists)"
                          : bad == 0              ? "interpolates"
                          : good == 0             ? "SNAPS ENTIRELY"
                                                  : "PARTIAL";
    Log.info("  {:<22} {:>10} {:>11} {:>12} {:>10} {:>11}  {} ({:.1f}% interpolate; camera-only is "
             "an UPPER BOUND on the defect, not a measurement — see the note in interp.cpp)",
             g_popName[p].empty() ? (p == 0 ? "(unlabelled)" : "pop " + std::to_string(p))
                                  : g_popName[p],
             g_audit[p][(int)Disposition::Paired], g_audit[p][(int)Disposition::Billboard],
             g_audit[p][(int)Disposition::CameraOnly], g_audit[p][(int)Disposition::SnappedOrtho],
             noId, verdict,
             // Denominator is what OUGHT to move: 2D and provably-static draws are excluded,
             // because a percentage that counts them as failures cannot reach 100 even when the
             // path is perfect.
             (good + bad) > 0 ? 100.0 * (double)good / (double)(good + bad) : 100.0);
  }
  Log.info("  A population labelled (unlabelled) is one no seam claims — those are the draws whose "
           "emitter is still unknown, and they are the honest edge of this audit rather than a "
           "clean bill of health.");
}

// ── VERTEX INTERPOLATION ────────────────────────────────────────────────────────────────────────
namespace {
struct VertRec {
  std::vector<float> pos;   // vtxCount * 3, host order
  uint64_t stamp = 0;
};
std::unordered_map<uint64_t, VertRec> g_vertPrev;
long g_vtxPatched = 0, g_vtxUnpaired = 0, g_vtxCountChanged = 0;

// The buffer holds raw GC vertex records: big-endian floats, byte-swapped by the shader when it
// reads them (gx/shader.cpp bswap32). So every read swaps in and every write swaps back out — a
// lerp done on the native interpretation of those bytes would produce garbage that still renders.
inline float be_f32(const uint8_t* p) {
  const uint32_t w = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
  float f;
  std::memcpy(&f, &w, sizeof f);
  return f;
}
inline void put_be_f32(uint8_t* p, float f) {
  uint32_t w;
  std::memcpy(&w, &f, sizeof w);
  p[0] = (uint8_t)(w >> 24);
  p[1] = (uint8_t)(w >> 16);
  p[2] = (uint8_t)(w >> 8);
  p[3] = (uint8_t)w;
}
} // namespace

bool patch_vertices(uint64_t tag, uint32_t vtxCount, uint16_t stride, uint16_t posOffset,
                    const uint8_t* src, uint8_t* dst, float alpha) {
  if (tag == 0 || src == nullptr || dst == nullptr || vtxCount == 0 || stride == 0) {
    return false;
  }
  auto& rec = g_vertPrev[tag];
  const size_t need = (size_t)vtxCount * 3;

  // Read this tick's positions first — they become `prev` for the next tick either way, so a draw
  // that cannot be interpolated this time still seeds the pair for next time.
  std::vector<float> cur(need);
  for (uint32_t v = 0; v < vtxCount; ++v) {
    const uint8_t* p = src + (size_t)v * stride + posOffset;
    cur[v * 3 + 0] = be_f32(p);
    cur[v * 3 + 1] = be_f32(p + 4);
    cur[v * 3 + 2] = be_f32(p + 8);
  }

  const bool consecutive = rec.stamp + 1 == g_tickIndex;
  const bool sameCount = rec.pos.size() == need;
  bool patched = false;
  if (consecutive && sameCount) {
    for (uint32_t v = 0; v < vtxCount; ++v) {
      uint8_t* q = dst + (size_t)v * stride + posOffset;
      for (int c = 0; c < 3; ++c) {
        const float a = rec.pos[v * 3 + c];
        put_be_f32(q + c * 4, a + (cur[v * 3 + c] - a) * alpha);
      }
    }
    ++g_vtxPatched;
    patched = true;
  } else if (!consecutive) {
    ++g_vtxUnpaired;
  } else {
    ++g_vtxCountChanged;
  }

  rec.pos.swap(cur);
  rec.stamp = g_tickIndex;
  return patched;
}

void report_vertex_interp() {
  const long total = g_vtxPatched + g_vtxUnpaired + g_vtxCountChanged;
  if (total == 0) {
    Log.warn("vertex interpolation: NO deforming draw was ever offered to it. That is not 'nothing "
             "deforms' — it means no draw reached the seam, so this says nothing about coverage.");
    return;
  }
  Log.info("vertex interpolation: {} of {} draw(s) had their POSITIONS lerped ({:.1f}%); {} had no "
           "consecutive previous tick (new object, or one that skipped a tick — correctly snaps); "
           "{} changed VERTEX COUNT and were snapped deliberately rather than smeared between two "
           "unrelated meshes.",
           g_vtxPatched, total, 100.0 * (double)g_vtxPatched / (double)total, g_vtxUnpaired,
           g_vtxCountChanged);
}

void set_tag_world_pos(uint64_t tag, float x, float y, float z) {
  if (tag == 0) {
    return;
  }
  auto& b = g_billboard[tag];
  // Rotate cur -> prev only when this is the FIRST sighting of the tag in a new tick. A particle is
  // drawn once per tick, but the guard costs nothing and makes a second draw idempotent rather than
  // shifting prev onto cur and destroying the pair.
  if (b.stampCur != g_tickIndex) {
    b.stampPrev = b.stampCur;
    b.prev[0] = b.cur[0];
    b.prev[1] = b.cur[1];
    b.prev[2] = b.cur[2];
    b.stampCur = g_tickIndex;
  }
  b.cur[0] = x;
  b.cur[1] = y;
  b.cur[2] = z;
}

bool patch_billboard(uint64_t tag, const uint8_t* src, uint8_t* dst, uint32_t uniformSize,
                     uint32_t mtxPosOffset, uint32_t mtxNrmOffset, float alpha) {
  if (tag == 0 || !g_camDeltaValid || src == nullptr || dst == nullptr) {
    return false;
  }
  auto it = g_billboard.find(tag);
  if (it == g_billboard.end()) {
    return false;
  }
  const BillboardPos& b = it->second;
  // Both samples must be from consecutive ticks. A particle that skipped a tick — off screen,
  // culled — would otherwise be interpolated across a gap its alpha was never scaled for, which is a
  // plausible-looking frame computed from the wrong pair.
  //
  // THE STAMP IS ONE BEHIND g_tickIndex, and getting this wrong made the whole path silently inert:
  // set_tag_world_pos is called by the host while the GUEST is drawing, and g_tickIndex is not
  // incremented until begin_camera_delta at the frame seam AFTER that. So a position recorded during
  // the tick now being patched carries stamp g_tickIndex - 1. The first version required
  // stampCur == g_tickIndex, which is never true; it reported 0 patched and 375,451 "unpaired",
  // which the report's own denominator caught in one run.
  if (b.stampCur + 1 != g_tickIndex || b.stampPrev + 1 != b.stampCur) {
    ++g_billboardUnpaired;
    return false;
  }
  if (mtxPosOffset == 0 || mtxNrmOffset == 0 || mtxPosOffset + kMtxBytes > uniformSize ||
      mtxNrmOffset + kMtxBytes > uniformSize) {
    return false;
  }
  // The vertices sit at V_cur * P_cur. The camera delta alone would show the particle at its
  // CURRENT world position from the interpolated viewpoint; what is wanted is its INTERPOLATED world
  // position from that viewpoint. The difference is a world-space displacement, rotated into eye
  // space by the interpolated view.
  const float k = alpha - 1.0f;   // -(1 - alpha)
  const float dw[3] = {k * (b.cur[0] - b.prev[0]), k * (b.cur[1] - b.prev[1]),
                       k * (b.cur[2] - b.prev[2])};
  float de[3];
  for (int r = 0; r < 3; ++r) {
    de[r] = g_viewLerp[r * 4 + 0] * dw[0] + g_viewLerp[r * 4 + 1] * dw[1] +
            g_viewLerp[r * 4 + 2] * dw[2];
  }
  const auto* srcPos = reinterpret_cast<const float*>(src + mtxPosOffset);
  const auto* srcNrm = reinterpret_cast<const float*>(src + mtxNrmOffset);
  auto* dstPos = reinterpret_cast<float*>(dst + mtxPosOffset);
  auto* dstNrm = reinterpret_cast<float*>(dst + mtxNrmOffset);
  for (int slot = 0; slot < 10; ++slot) {
    float out[12];
    compose(g_camDelta, srcPos + slot * 12, out);
    out[3] += de[0];
    out[7] += de[1];
    out[11] += de[2];
    std::memcpy(dstPos + slot * 12, out, sizeof(out));
    // Normals take the rotation only; a translation does not affect them.
    float outN[12];
    compose(g_camDelta, srcNrm + slot * 12, outN);
    outN[3] = outN[7] = outN[11] = 0.0f;
    std::memcpy(dstNrm + slot * 12, outN, sizeof(outN));
  }
  ++g_billboardPatched;
  return true;
}

void set_view_matrix(const float m[12]) {
  std::memcpy(g_viewCur, m, sizeof(g_viewCur));
  g_haveViewCur = true;
}

bool begin_camera_delta(float alpha) {
  g_camDeltaValid = false;
  ++g_tickIndex;
  // The inverses are computed here whether or not interpolation can proceed, because attribution
  // wants them independently of patching: a tick that cannot be interpolated is exactly the kind of
  // tick worth measuring.
  g_haveInvCur = g_haveViewCur && affine_inverse(g_viewCur, g_invViewCur);
  g_haveInvPrev = g_haveViewPrev && affine_inverse(g_viewPrev, g_invViewPrev);
  if (g_haveInvCur && g_haveInvPrev) {
    // The camera's own step: eye position (the INVERSE view's translation — the view's own
    // translation is -R*eye and would report rotation as motion) and the rotation angle between the
    // two orientations.
    const double dx = g_invViewCur[3] - g_invViewPrev[3];
    const double dy = g_invViewCur[7] - g_invViewPrev[7];
    const double dz = g_invViewCur[11] - g_invViewPrev[11];
    const double eye = std::sqrt(dx * dx + dy * dy + dz * dz);
    // trace(R_cur * R_prev^T) = 1 + 2cos(theta) for a rotation.
    double tr = 0.0;
    for (int r = 0; r < 3; ++r) {
      for (int k = 0; k < 3; ++k) {
        tr += (double)g_viewCur[r * 4 + k] * (double)g_viewPrev[r * 4 + k];
      }
    }
    double c = (tr - 1.0) * 0.5;
    if (c > 1.0) { c = 1.0; }
    if (c < -1.0) { c = -1.0; }
    const double rotDeg = std::acos(c) * (180.0 / 3.14159265358979323846);
    g_camEyeSum += eye;
    if (eye > g_camEyeMax) { g_camEyeMax = eye; }
    g_camRotSumDeg += rotDeg;
    if (rotDeg > g_camRotMaxDeg) { g_camRotMaxDeg = rotDeg; }
    ++g_camN;
    int bucket = 0;
    for (double edge = 1.0; bucket < kCamBuckets - 1 && eye >= edge; edge *= 10.0) {
      ++bucket;
    }
    ++g_camHist[bucket];
    note_worst_tick(g_tickIndex, eye, rotDeg);

    // LOCALISATION, not a behavioural threshold: dump the eye positions AROUND a large step so its
    // shape can be read. A step that jumps away and comes straight back is not a cut at all — it is
    // this measurement aliasing between two cameras, because j3dSys.mViewMtx is a single global and
    // the emitter samples it at end of tick, so a tick that renders a second camera (a mirror or
    // reflection pass) hands over that camera's view instead. A step that jumps and STAYS is a real
    // discontinuity. The two demand opposite responses and the histogram cannot tell them apart.
    // The 1000 only decides what gets PRINTED; nothing branches on it.
    if (eye > 1000.0 && g_eyeCasesPrinted < 4) {
      ++g_eyeCasesPrinted;
      Log.info("large camera step at tick {} ({:.1f} units) — preceding eye positions:", g_tickIndex,
               eye);
      for (int i = 0; i < kEyeRing; ++i) {
        const EyeSample& s = g_eyeRing[(g_eyeRingPos + i) % kEyeRing];
        if (s.tick >= 0) {
          Log.info("    tick {}: eye ({:.1f}, {:.1f}, {:.1f})", s.tick, s.x, s.y, s.z);
        }
      }
      Log.info("    tick {}: eye ({:.1f}, {:.1f}, {:.1f})  <-- the step", g_tickIndex,
               g_invViewCur[3], g_invViewCur[7], g_invViewCur[11]);
      g_eyeFollow = 3;   // and the ticks after it, which is what distinguishes the two shapes
    } else if (g_eyeFollow > 0) {
      --g_eyeFollow;
      Log.info("    tick {}: eye ({:.1f}, {:.1f}, {:.1f})  <-- after (returns => camera aliasing, "
               "stays => real cut)",
               g_tickIndex, g_invViewCur[3], g_invViewCur[7], g_invViewCur[11]);
    }
    g_eyeRing[g_eyeRingPos] = EyeSample{g_tickIndex, (float)g_invViewCur[3], (float)g_invViewCur[7],
                                        (float)g_invViewCur[11]};
    g_eyeRingPos = (g_eyeRingPos + 1) % kEyeRing;
  }
  if (!g_haveViewCur || !g_haveViewPrev) {
    return false;   // first tick, or the emitter is not supplying a view: nothing to interpolate
  }
  float lerped[12];
  for (int i = 0; i < 12; ++i) {
    lerped[i] = g_viewPrev[i] * (1.0f - alpha) + g_viewCur[i] * alpha;
    g_viewLerp[i] = lerped[i];
  }
  if (!g_haveInvCur) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      Log.error("this tick's view matrix is singular, so V_lerp*V_cur^-1 does not exist. Camera "
                "interpolation is OFF for unpaired draws, which means they will render from the "
                "current viewpoint while paired ones do not — expect the frame to tear.");
    }
    return false;
  }
  compose(lerped, g_invViewCur, g_camDelta);
  g_camDeltaValid = true;
  return true;
}

// SBR_INTERP_CAMONLY=0 — leave unpaired draws on the TICK's viewpoint instead of moving them to the
// interpolated one. An A/B for a specific, code-verified defect, not a general knob.
//
// THE DEFECT. GX texgen sourced from GX_TG_POS uses the RAW vertex attribute, not the position after
// the position matrix (lib/gx/shader.cpp, `vtx_attr(config, GX_VA_POS)`). SMS's water refraction
// exploits exactly that: it builds its quad in EYE space, loads an identity PNMTX and a view-less
// texture matrix (C_MTXLightPerspective in slot 0x1e), so the screen UV is `texmtx * eye_position`.
//
// This function moves the POSITION matrices to the interpolated camera and never touches the
// TEXTURE matrices — no interpolation path does. For ordinary geometry that is right. For a
// position-sourced texgen it is not: the quad is drawn at the N-and-a-half viewpoint while its UVs
// still map to N, so the reflection is sampled at the old screen mapping and drawn at the new
// position. That reads as the reflection sitting in the WRONG PLACE rather than juddering, which is
// how it was reported.
//
// The principled fix is to compose the delta into the texture matrix as well — UV should be
// `texmtx * (camDelta * pos)`, i.e. `texmtx' = texmtx * camDelta` — but ONLY for texgens sourced
// from position; doing it to a texgen sourced from a UV attribute would corrupt ordinary texturing.
// That needs the texgen source at this seam, which it does not currently have. Until then this
// switch makes the trade visible: off, the water is self-consistent and unpaired geometry snaps.
bool camera_patch_enabled() {
  static const bool v = [] {
    const char* e = std::getenv("SBR_INTERP_CAMONLY");
    return e == nullptr || e[0] != '0';
  }();
  return v;
}

// A STATIC/MOVED SPLIT FOR camera-only DRAWS WAS TRIED TWICE AND BOTH DESIGNS WERE UNSOUND.
// Recorded here so it is not attempted a third time the same way.
//
// The idea is right: a draw receiving the camera delta alone is CORRECT when its own transform did
// not change (static world geometry needs exactly that) and defective when it did, and reporting
// them as one number understates the path. The implementations were the problem.
//
//  1. Hash the draw's pnMtx and compare with last tick's. pnMtx is model x view, so it changes the
//     instant the CAMERA moves — every draw reads "moved" in any run worth measuring. Worse, the
//     draws it did call static were the ones whose raw matrix is bit-identical every tick, which
//     for JPA particles is the IDENTITY matrix they always load. "Static" meant "identity pnMtx",
//     not "did not move", and it inflated particles to 99.4% and the shine slices to 100%.
//
//  2. Divide the camera out first: hash V_cur^-1 * pnMtx, which is the object's own world
//     transform. Mathematically right and numerically useless — the product is not bit-exact, so
//     the round-off differs as the view changes and NOTHING ever compares equal. Measured: the
//     static column went to zero for every population.
//
// A sound version needs a TOLERANCE on the world transform, which is defensible for a statistic
// (a mis-classified draw adds noise to a percentage and cannot produce an artefact) but has to be
// validated against both classes before it is believed — the discipline this file already applies
// to the pairing itself. Not done; camera-only is reported whole and read as an UPPER BOUND on the
// defect rather than a measurement of it.

void patch_camera_only(const uint8_t* src, uint8_t* dst, uint32_t uniformSize,
                       uint32_t mtxPosOffset, uint32_t mtxNrmOffset, uint32_t texMtxCamMask) {
  if (!camera_patch_enabled()) {
    return;
  }
  if (!g_camDeltaValid || dst == nullptr || src == nullptr) {
    ++g_camRefusedNoDelta;
    return;
  }
  if (mtxPosOffset == 0 || mtxNrmOffset == 0 || mtxPosOffset + kMtxBytes > uniformSize ||
      mtxNrmOffset + kMtxBytes > uniformSize) {
    ++g_camRefusedBadOffset;
    return;
  }
  // EVERY READ FROM src, EVERY WRITE TO dst — the same rule patch_draw follows, and for the same
  // reason. dst is GPU staging, which is write-combined: writing it is cheap, reading it back is
  // uncached and roughly two orders of magnitude slower. This function used to read its input from
  // dst and write the result back over it, so each unpaired draw performed ~960 bytes of uncached
  // reads. That was the dominant cost of interpolation — the feature spent more time reading GPU
  // memory back than the rest of the frame took to build. src holds the same bytes in ordinary RAM.
  const auto* srcPos = reinterpret_cast<const float*>(src + mtxPosOffset);
  const auto* srcNrm = reinterpret_cast<const float*>(src + mtxNrmOffset);
  auto* dstPos = reinterpret_cast<float*>(dst + mtxPosOffset);
  auto* dstNrm = reinterpret_cast<float*>(dst + mtxNrmOffset);
  for (int slot = 0; slot < 10; ++slot) {
    float out[12];
    compose(g_camDelta, srcPos + slot * 12, out);
    std::memcpy(dstPos + slot * 12, out, sizeof(out));
    // Normals take the rotation only — no translation. Correct for a rigid view delta, because the
    // inverse-transpose of a rotation is the rotation itself.
    compose(g_camDelta, srcNrm + slot * 12, out);
    out[3] = srcNrm[slot * 12 + 3];
    out[7] = srcNrm[slot * 12 + 7];
    out[11] = srcNrm[slot * 12 + 11];
    std::memcpy(dstNrm + slot * 12, out, sizeof(out));
  }
  ++g_cameraPatched;

  // EYE-SPACE TEXTURE MATRICES. The position block above moved this draw's vertices to the
  // interpolated viewpoint. For a texgen sourced from GX_TG_POS the UV is computed from the RAW
  // vertex, so it did NOT move with them, and the two are now inconsistent by exactly the delta.
  //
  //   real frame:  uv = texmtx * p           position = p                (consistent)
  //   in-between:  uv = texmtx * p           position = camDelta * p     (off by camDelta)
  //   wanted:      uv = texmtx * camDelta * p
  //
  // so texmtx' = texmtx * camDelta — the delta on the RIGHT, because it acts on the vertex before
  // the texture matrix does, which is the opposite side from the position matrices above (there the
  // delta acts on the result of the model-view, hence camDelta * pnMtx). Getting the side wrong
  // would produce a plausible, smoothly-wrong mapping rather than an obvious failure, which is why
  // it is spelled out here next to the code.
  //
  // Valid only because the mask's gate established that this draw's position matrix is the identity
  // — see eye_space_texgen_mask. Under that condition object space IS eye space and the two patches
  // are the same reprojection.
  if (texMtxCamMask != 0) {
    // UNPAIRED, so there is no cross-tick identity and the stability test patch_draw uses is not
    // available. The one case that can be settled without it is an IDENTITY model-view: the
    // vertices are then already in eye space — the game did the view transform itself and baked the
    // result into the stream — so the texture matrix maps eye space to UV by construction and
    // cannot be object-locked. That is SMS's water refraction, and composing the delta on the RIGHT
    // (it acts on the vertex before the texture matrix does, the opposite side from the position
    // matrices above) is the same reprojection the geometry just received. Any other model-view is
    // left alone rather than guessed at.
    const float* pn = reinterpret_cast<const float*>(src + mtxPosOffset);
    static constexpr float kI[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    bool identity = true;
    for (int e = 0; e < 12; ++e) {
      const float d = pn[e] - kI[e];
      if (d < -1e-4f || d > 1e-4f) { identity = false; break; }
    }
    for (int idx = 0; identity && idx < 32; ++idx) {
      if ((texMtxCamMask & (1u << idx)) == 0) {
        continue;
      }
      const uint32_t off = mtxPosOffset + static_cast<uint32_t>(idx) * kOneMtxBytes;
      if (off + kOneMtxBytes > uniformSize) {
        ++g_texMtxRefusedBadOffset;
        continue;
      }
      float out[12];
      compose(reinterpret_cast<const float*>(src + off), g_camDelta, out);
      std::memcpy(dst + off, out, sizeof(out));
      ++g_texMtxPatched;
    }
  }
}

namespace {
// Zero every accumulator. Used by the self-test so it cannot leave its synthetic samples in the
// numbers a real run reports.
void reset_stats() {
  g_prev.clear();
  g_cur.clear();
  g_cursor.clear();
  g_paired = g_unpaired = g_mismatched = 0;
  g_transDeltaSum = g_transDeltaMax = 0.0;
  g_transDeltaN = 0;
  g_objDeltaSum = g_objDeltaMax = 0.0;
  g_objDeltaN = g_objDeltaUnavailable = 0;
  for (int i = 0; i < kObjBuckets; ++i) { g_objHist[i] = 0; }
  for (int i = 0; i < kWorstDraws; ++i) { g_worstDraw[i] = WorstDraw{}; }
  for (int i = 0; i < kCamBuckets; ++i) { g_camHist[i] = 0; }
  g_camEyeSum = g_camEyeMax = g_camRotSumDeg = g_camRotMaxDeg = 0.0;
  g_camN = 0;
  g_tickIndex = 0;
  for (int i = 0; i < kWorstTicks; ++i) { g_worst[i] = WorstTick{}; }
  g_haveViewCur = g_haveViewPrev = g_haveInvCur = g_haveInvPrev = false;
  g_camDeltaValid = false;
  g_cameraPatched = 0;
  g_camRefusedNoDelta = g_camRefusedBadOffset = 0;
  g_texMtxPatched = g_texMtxRefusedBadOffset = 0;
  g_texStable = g_texUnstable = g_texNoPrev = g_texSingular = g_texMultiSlot = 0;
}

// Build a view matrix for a camera at world position `eye`, axis-aligned: V = [I | -eye].
void make_view_at(float x, float y, float z, float* out) {
  const float m[12] = {1, 0, 0, -x, 0, 1, 0, -y, 0, 0, 1, -z};
  std::memcpy(out, m, sizeof(m));
}

// Write V*M for an object at world position (ox,oy,oz) into slot 0 of a uniform block, both the
// pos and nrm spans. M is a pure translation, so V*M = [I | obj - eye].
void write_draw_block(uint8_t* buf, uint32_t posOff, uint32_t nrmOff, const float* view, float ox,
                      float oy, float oz) {
  float a[12] = {1, 0, 0, view[3] + ox, 0, 1, 0, view[7] + oy, 0, 0, 1, view[11] + oz};
  std::memcpy(buf + posOff, a, sizeof(a));
  std::memcpy(buf + nrmOff, a, sizeof(a));
}
} // namespace

bool selftest() {
  constexpr uint32_t kSize = 2048, kPos = 144, kNrm = 1024;
  static_assert(kNrm + kMtxBytes <= kSize, "self-test block must hold both matrix spans");
  std::vector<uint8_t> src(kSize), dst(kSize);

  // Each case runs two ticks and reads back the two deltas. The point of running BOTH is that a
  // discriminator which has only ever seen one class is not known to discriminate: an attribution
  // that always returned zero would pass case A and fail case B, and one that ignored the view
  // entirely would pass B and fail A.
  struct Case {
    const char* name;
    float eye0[3], eye1[3];   // camera world position, tick 0 -> tick 1
    float obj0[3], obj1[3];   // object world position, tick 0 -> tick 1
    double wantTotal, wantObj;
  };
  const Case cases[] = {
      {"camera moves 1000, object static", {0, 0, 0}, {1000, 0, 0}, {0, 0, 0}, {0, 0, 0}, 1000.0, 0.0},
      {"camera static, object moves 1000", {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1000, 0, 0}, 1000.0, 1000.0},
  };

  bool ok = true;
  for (const Case& c : cases) {
    reset_stats();
    float v0[12], v1[12];
    make_view_at(c.eye0[0], c.eye0[1], c.eye0[2], v0);
    make_view_at(c.eye1[0], c.eye1[1], c.eye1[2], v1);

    // begin_tick() is not optional bookkeeping: it resets the per-tag ordinal cursor. Without it the
    // second tick's draw takes ordinal 1, finds nothing at that ordinal in the previous tick, and
    // reports unpaired — which is how the first version of this self-test failed, and is a fair
    // model of how the real path would fail if begin_tick were ever dropped.
    begin_tick();
    set_view_matrix(v0);
    begin_camera_delta(0.5f);
    write_draw_block(src.data(), kPos, kNrm, v0, c.obj0[0], c.obj0[1], c.obj0[2]);
    patch_draw(1, 3, src.data(), dst.data(), kSize, kPos, kNrm, 0.5f, 0, 0);
    end_tick();

    begin_tick();
    set_view_matrix(v1);
    begin_camera_delta(0.5f);
    write_draw_block(src.data(), kPos, kNrm, v1, c.obj1[0], c.obj1[1], c.obj1[2]);
    const bool paired = patch_draw(1, 3, src.data(), dst.data(), kSize, kPos, kNrm, 0.5f, 0, 0);

    const double gotTotal = g_transDeltaN ? g_transDeltaSum / (double)g_transDeltaN : -1.0;
    const double gotObj = g_objDeltaN ? g_objDeltaSum / (double)g_objDeltaN : -1.0;
    const bool pass = paired && std::fabs(gotTotal - c.wantTotal) < 0.01 &&
                      std::fabs(gotObj - c.wantObj) < 0.01;
    if (!pass) {
      ok = false;
      Log.error("SELFTEST FAILED [{}]: paired={} total delta {:.3f} (want {:.3f}) object delta "
                "{:.3f} (want {:.3f}). The camera/object attribution does not discriminate, so any "
                "conclusion drawn from those two numbers is unfounded.",
                c.name, paired, gotTotal, c.wantTotal, gotObj, c.wantObj);
    }
  }
  reset_stats();
  if (ok) {
    Log.info("interp selftest PASSED: camera/object attribution separates a 1000-unit camera move "
             "(object delta 0) from a 1000-unit object move (object delta 1000) — it has been run "
             "against both classes, not just the one it is expected to find.");
  }
  return ok;
}

long tick_index() { return g_tickIndex; }

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
  // BILLBOARDS. Printed with BOTH numbers because "0 patched" has two opposite causes: no billboard
  // was tagged at all, or every tagged one failed to find a consecutive-tick partner.
  Log.info("billboard interpolation: {} draw(s) had their own displacement applied as a translation, "
           "{} were tagged but had no usable prev/cur pair (new particle, or one that skipped a "
           "tick — those correctly fall back to the camera delta alone).{}",
           g_billboardPatched, g_billboardUnpaired,
           g_billboardPatched == 0
               ? "  <-- NONE PATCHED. Either nothing recorded a world position for a tagged draw, or "
                 "the tags never reached the draw. A particle-heavy scene reporting 0 here means the "
                 "seam is not connected, NOT that particles do not move."
               : "");
  Log.info("paired-draw translation delta between ticks: mean {:.3f} max {:.3f} world units over {} "
           "samples. A real object moves a fraction of a unit per 1/30 s — a mean in the hundreds "
           "means pairing is returning some OTHER transform, which the smoothness metric cannot "
           "see because a large consistent displacement still reads as even motion.",
           g_transDeltaN ? g_transDeltaSum / (double)g_transDeltaN : 0.0, g_transDeltaMax,
           g_transDeltaN);
  // ATTRIBUTION. The line above bundles object motion and camera motion, because pnMtx is
  // model x view. This one divides the camera out, and the PAIR of numbers is what identifies the
  // defect: a large total with a SMALL object delta is the camera moving the whole world (and a
  // large max there is a camera CUT, which must snap rather than lerp); a large object delta is a
  // pairing defect, some other object's transform coming back from the table.
  if (g_objDeltaN == 0) {
    Log.warn("paired-draw delta ATTRIBUTION unavailable for all {} samples — no invertible view for "
             "one of the two ticks, so the camera could not be divided out. The total-delta line "
             "above therefore says NOTHING about whether the motion was the object or the camera.",
             g_objDeltaUnavailable);
  } else {
    Log.info("paired-draw delta with the CAMERA REMOVED (object's own world motion): mean {:.3f} max "
             "{:.3f} over {} samples ({} samples had no invertible view and are excluded). Compare "
             "with the total above: total >> object means the camera moved the world; object large "
             "means pairing returned another object's transform.",
             g_objDeltaSum / (double)g_objDeltaN, g_objDeltaMax, g_objDeltaN, g_objDeltaUnavailable);
    Log.info("  object-motion distribution (world units/tick): [0,0.1) {} | [0.1,1) {} | [1,10) {} | "
             "[10,100) {} | [100,1k) {} | [1k,10k) {} | [10k,inf) {}. Ordinary animation lives in "
             "the first two buckets; anything from [10,100) up is a pose no object reaches in 1/30 "
             "s, so those counts are the mispairings, and their SHARE says whether this is a broad "
             "defect or a few pathological objects.",
             g_objHist[0], g_objHist[1], g_objHist[2], g_objHist[3], g_objHist[4], g_objHist[5],
             g_objHist[6]);
    for (int i = 0; i < kWorstDraws; ++i) {
      if (g_worstDraw[i].tick >= 0) {
        // The tag is (guest J3DShape << 32 | instance draw-matrix pointer), so both halves are
        // printable guest addresses — the object is identifiable, not just countable.
        Log.info("  worst draw: {:.3f} units, shape {:#010x} instance {:#010x} ordinal {} on tick {}",
                 g_worstDraw[i].delta, (uint32_t)(g_worstDraw[i].tag >> 32),
                 (uint32_t)(g_worstDraw[i].tag & 0xffffffffu), g_worstDraw[i].ordinal,
                 g_worstDraw[i].tick);
      }
    }
  }
  // The camera's own step, as a DISTRIBUTION. A cut is rare by definition, so a mean cannot show
  // one; the histogram can, and it can also show the OTHER answer — a unimodal distribution means
  // this run contained no cuts at all, and any residual blamed on them is misattributed.
  if (g_camN == 0) {
    Log.warn("camera step: NO TICK had two invertible views, so camera motion was never measured. "
             "This is not 'the camera did not move' — it is 'this instrument saw nothing'.");
  } else {
    Log.info("camera step per tick over {} ticks: eye translation mean {:.3f} max {:.3f} world "
             "units; rotation mean {:.3f} max {:.3f} deg",
             g_camN, g_camEyeSum / (double)g_camN, g_camEyeMax, g_camRotSumDeg / (double)g_camN,
             g_camRotMaxDeg);
    Log.info("  eye-step distribution (world units/tick): [0,1) {} | [1,10) {} | [10,100) {} | "
             "[100,1k) {} | [1k,10k) {} | [10k,inf) {}. BIMODAL — a dense bulk plus a few entries "
             "decades away — is a camera CUT, a tick with no meaningful in-between that must snap. "
             "UNIMODAL means this run contained no cuts and they cannot explain any residual.",
             g_camHist[0], g_camHist[1], g_camHist[2], g_camHist[3], g_camHist[4], g_camHist[5]);
    for (int i = 0; i < kWorstTicks; ++i) {
      if (g_worst[i].tick >= 0) {
        Log.info("  worst tick #{}: eye step {:.3f} units, rotation {:.3f} deg", g_worst[i].tick,
                 g_worst[i].eye, g_worst[i].rotDeg);
      }
    }
  }
  // The camera line is separate because its failure is separate: pairing can be perfect while the
  // frame still tears, if unpaired draws are left at the current viewpoint.
  if (!g_haveViewCur) {
    Log.warn("camera interpolation: NO VIEW MATRIX SUPPLIED. Every unpaired draw keeps the CURRENT "
             "viewpoint while paired draws move to the in-between one, so the frame renders from "
             "two viewpoints at once. This is the failure that makes partial coverage worse than "
             "none — emit GX_AURORA_VIEW_MTX.");
  } else {
    Log.info("camera interpolation: {} draw uniforms carried the camera delta; REFUSED {} for no "
             "usable camera delta and {} for an unusable matrix offset. A refused draw keeps the "
             "CURRENT viewpoint while the rest of the frame moves to the in-between one, so a large "
             "refusal count is the frame being drawn from two viewpoints at once.",
             g_cameraPatched, g_camRefusedNoDelta, g_camRefusedBadOffset);
  }

  // The eye-space texgen line, printed with its full denominator chain so that every zero in it
  // says WHICH zero it is. Reading left to right: how many draws used a position-sourced matrix
  // texgen at all, how the gate disposed of them, and how many texture matrices were rewritten.
  // Position-sourced texgens, with the full denominator chain so every zero says WHICH zero it is:
  // how many draws used the construct, how the record-time gate disposed of them, and how the
  // cross-tick stability test then split the survivors into camera projections and object-locked
  // mappings.
  Log.info("position-sourced texgens: {} draw(s) used one; {} rejected at record time (per-vertex "
           "matrix index, or no matrix). Of the rest, the paired path measured {} STABLE (the "
           "texture matrix tracks this draw's model-view, so it was re-composed with the "
           "interpolated pose) and {} UNSTABLE (object-locked or animating, left alone); {} had no "
           "previous tick to compare against, {} a singular model-view, {} carried more than one "
           "such matrix and only the first was handled. Unpaired eye-space draws: {} texture "
           "matrix write(s) carried the camera delta, {} refused for a bad offset.{}",
           aurora::gx::fifo::g_texgenPosSourced, aurora::gx::fifo::g_texgenRejectIndexed,
           g_texStable, g_texUnstable, g_texNoPrev, g_texSingular, g_texMultiSlot, g_texMtxPatched,
           g_texMtxRefusedBadOffset,
           aurora::gx::fifo::g_texgenPosSourced == 0
               ? "   <-- NO DRAW IN THIS RUN USED ONE. That is not 'the fix works': it means this "
                 "scene never exercised the construct, so this run says NOTHING about it."
           : (g_texStable == 0 && g_texUnstable == 0 && g_texMtxPatched == 0)
               ? "   <-- candidates existed but NOTHING reached the stability test, so the mask is "
                 "not arriving at the patching code. Check DrawData::texMtxCamMask is passed on."
               : "");
}

} // namespace aurora::gfx::interp
