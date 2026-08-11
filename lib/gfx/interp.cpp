// interp — see interp.hpp for the design and for the two approximations it makes.

#include "interp.hpp"

#include <cstdlib>

#include "../internal.hpp"

#include <algorithm>
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
  // WHICH TICK THIS SAMPLE IS FROM, and the answer is always "the previous one" — which is worth
  // measuring precisely because it did not have to be. The tables are a prev/cur swap, and if `cur`
  // were merely overwritten rather than cleared, a tag that skipped a tick would find a TWO-tick-old
  // entry sitting where the previous tick's should be. Pairing against that while `alpha` assumes a
  // one-tick spacing moves the object by the wrong fraction of a step, and it reads as a successful
  // pairing in every count there is. begin_tick() clears g_cur, so it cannot happen — and the stamp
  // is what turns that from a property of the code into a number (see g_pairedStale, measured 0 of
  // 333,348 pairings).
  long stamp = -1;
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
  // How far this draw's object moved in the tick that produced this entry, camera removed. It is
  // the object's OWN scale for what "far" means — see the discontinuity test.
  double objDelta = -1.0;
};

// Two generations. `prev` is what the last tick recorded; `cur` is being filled by this tick and
// becomes `prev` at end_tick.
// ONE table, not a prev/cur swap. A swap can only ever offer the immediately previous tick, so an
// object that drew, skipped a tick and drew again had nothing to pair with and snapped — 562 of J3D
// world geometry's 822 misses on a Delfino run, 90 of 90 for the shadow alpha cubes, 293 of 304 for
// the screen wipes. Keeping the last sample per (tag, ordinal) with the tick it came from lets those
// pair against a sample that is genuinely theirs, with `alpha` reweighted for the spacing — the same
// generalisation the vertex path already makes, and for the same reason: interpolating between the
// two most recent samples of one object, weighted by how far apart they are, is the correct answer,
// and the one-tick rule was approximating a spacing that was not there.
std::unordered_map<uint64_t, std::vector<Entry>> g_ent;

// How stale a sample may be before the older pose stops describing anywhere the object has recently
// been. Past this a snap is safer than a sweep across whatever happened while nobody was looking.
// The same bound the vertex path uses, deliberately: two paths disagreeing about how old is too old
// would show up as one part of an object interpolating while another snapped.
constexpr long kMaxGapMtx = 4;

// The camera, per tick, for as far back as a pair may reach. The object-motion attribution and the
// discontinuity gate both divide the camera out of a sample, and the camera that belongs to a sample
// is the one from ITS tick — using the previous tick's for a three-tick-old sample would charge the
// object with the camera's motion and refuse it as a teleport. A pair whose camera is not in this
// ring is refused rather than paired against the wrong one.
struct InvViewSample {
  long stamp = -1;
  float m[12] = {};
};
InvViewSample g_invHist[kMaxGapMtx + 2];
long g_gapPaired = 0, g_gapRefusedNoCamera = 0, g_gapTooStale = 0;
// How long the gaps actually ARE. Deciding a bound from a count of refusals is guessing; this says
// whether the misses are near-misses or absences of a wholly different order.
long g_mtxGapHist[10] = {};

// How far into each tag we are THIS tick. Reset every tick; this is the ordinal that is scoped to a
// tag rather than global.
std::unordered_map<uint64_t, uint32_t> g_cursor;
// How many parts each tag has EVER drawn, across the whole run — not just last tick. This is what
// separates "this object is new" (nothing to pair with, correct) from "this object drew before and
// we lost it" (a real gap). One entry per object, not per part.
std::unordered_map<uint64_t, uint32_t> g_everSeenParts;

long g_paired = 0;
long g_unpaired = 0;
long g_mismatched = 0;
// The matrix path's misses, BY POPULATION and by cause. The audit's camera-only column is the sum
// of three different things — an object that skipped a tick, one whose display list changed length,
// and one refused as discontinuous — and a single number cannot say which, so a residual of 832 out
// of 293,000 was unattributable. The vertex path already splits its misses this way; this is the
// same split for the matrices.
long g_popGap[256] = {}, g_popMismatch[256] = {}, g_popRefused[256] = {};
// Pairings made against a sample that is NOT one tick old. See Entry::stamp.
long g_pairedFresh = 0, g_pairedStale = 0, g_pairedStaleMaxAge = 0;

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
// THE SAME HISTOGRAM, PER POPULATION. The global one says 21,866 paired draws moved 10-100 world
// units in a 30th of a second and calls them mispairings; that sentence is a CLAIM about what
// objects can do, and it was never checked against an object whose speed is known. Attributing it
// says WHICH systems are in the tail, which is the first thing a reader needs in order to decide
// whether the tail is Mario running or pairing returning another object's transform.
long g_objHistPop[256][kObjBuckets] = {};
constexpr int kWorstDraws = 6;
struct WorstDraw { double delta = -1.0; uint64_t tag = 0; uint32_t ordinal = 0; long tick = -1; };
WorstDraw g_worstDraw[kWorstDraws];
// Counted separately because "attribution was not available" and "attribution says zero" must not
// look alike: on the first tick, or any tick with no previous view, the camera cannot be removed.
long g_objDeltaUnavailable = 0;
// Beyond this, a paired delta is not motion. See the note at the use site for why a threshold is
// defensible here and was not for the camera cut: a three-decade gap in the distribution, and a
// measured ground truth (a running Mario, 58.5 units/tick max) far below it.
constexpr double kDiscontinuity = 100.0;
long g_snappedDiscontinuity = 0;
double g_snappedDiscontinuityMax = 0.0;
double g_acceptedMax = 0.0;
// THE REFUSED DELTAS, BUCKETED. Without this the distribution above is a lie by construction: a
// refused draw returns before it is bucketed, so the accepted histogram reads "[100,inf) 0" no
// matter how many draws were cut there, and the bound looks vindicated by data it removed. What
// decides whether 100 is the right bound is the SHAPE of what it cuts — a far cluster three decades
// out is a mispair, a continuous run from 100 upward is real motion being severed.
// PER-POPULATION vertex-path outcomes. "particle stripe interpolates 63%" is not a number anyone
// can act on: a chain that gains a particle CHANGES VERTEX COUNT and must snap, which is a ceiling,
// while a chain that skipped a tick is a seam problem. Same percentage, opposite conclusions.
long g_vtxFirstSight = 0;
long g_vtxGapHist[9] = {};   // index = gap in ticks, 8 = "8 or more"
long g_vtxGapPatched = 0;   // interpolated across a skipped tick, alpha scaled by the spacing
long g_vtxTooStale = 0;     // last sample older than the bound: snapped rather than swept
long g_vtxPopPatched[256] = {};
long g_vtxPopUnpaired[256] = {};
long g_vtxPopCountChanged[256] = {};
long g_objHistRefused[kObjBuckets] = {};
long g_refusedPop[256] = {};
// WHO gets refused, by count rather than by size. The largest single refusal names the most extreme
// event; the most FREQUENT one names the systematic defect, and 45 refusals per tick sustained is a
// systematic defect. Keyed by the tag's shape half, because a mispair against a pooled instance
// changes the instance half every tick while the shape stays put — grouping by the full tag would
// scatter one cause across hundreds of rows.
std::unordered_map<uint32_t, long> g_refusedShape;
int g_refusedShown = 0;
long g_refusedLastTick = -1;

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
long g_billboardBirth = 0, g_billboardGap = 0;
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
  // The sample table is NOT cleared: it is the per-object history a gapped pair reads from. Only the
  // per-tick ordinal cursor resets.
  g_cursor.clear();
}

bool patch_draw(uint64_t tag, uint32_t vtxCount, const uint8_t* src, uint8_t* dst,
                uint32_t uniformSize, uint32_t mtxPosOffset, uint32_t mtxNrmOffset, float alpha,
                uint32_t texMtxCamMask, uint32_t pnMtxSlot, uint8_t pop,
                bool* outFirstEverSighting) {
  // Default it on EVERY path, including the early refusals below. A draw with no tag at all, or one
  // refused for a bad matrix offset, is not a birth — it is a draw this path cannot identify, and
  // letting a stale `true` leak out of a previous call would file it as correct-by-construction.
  if (outFirstEverSighting != nullptr) {
    *outFirstEverSighting = false;
  }
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

  // HAS THIS (tag, ordinal) EVER BEEN DRAWN BEFORE IN THIS RUN? Recorded as a high-water mark per
  // tag, which costs one map entry per object rather than one per part. It is updated on every
  // call regardless of what happens below, so a draw that fails to pair for some other reason is
  // still marked as seen and cannot be reported as a birth on a later tick.
  uint32_t& everSeenParts = g_everSeenParts[tag];
  const bool firstEverSighting = ordinal >= everSeenParts;
  if (ordinal + 1 > everSeenParts) {
    everSeenParts = ordinal + 1;
  }

  auto& vec = g_ent[tag];
  if (vec.size() <= ordinal) {
    vec.resize(ordinal + 1);
  }
  // The PREVIOUS sample, taken by value before this tick's overwrites it. One table means `mine` and
  // `was` are the same slot, so the copy is what keeps them distinct — writing first and reading
  // after would pair every draw with itself and read as 100% interpolation of nothing.
  const Entry was = vec[ordinal];
  Entry& mine = vec[ordinal];
  std::memcpy(mine.pos, src + mtxPosOffset, kMtxBytes);
  std::memcpy(mine.nrm, src + mtxNrmOffset, kMtxBytes);
  mine.vtxCount = vtxCount;
  mine.stamp = g_tickIndex;
  mine.texValid = false;
  mine.texIdx = 0;
  mine.objDelta = was.objDelta;

  const long gap = (was.stamp >= 0) ? (g_tickIndex - was.stamp) : -1;
  // BUCKET EVERY GAP, not only the refused ones. The first version incremented this inside the
  // refusal branch, so buckets 1-4 — exactly the recoverable range the histogram exists to size —
  // could never be reached, and the output read "no near-misses" when it meant "not counted".
  if (gap >= 1) ++g_mtxGapHist[gap < 9 ? (size_t)gap : 9];
  if (gap < 1 || gap > kMaxGapMtx) {
    ++g_unpaired;   // never drawn before, or its last sample is too old to describe it.
    if (!firstEverSighting) ++g_popGap[pop];
    if (gap > kMaxGapMtx) ++g_gapTooStale;
    // A BIRTH IS NOT A DEFECT, and it is the only kind of miss that can never be fixed by better
    // pairing: there is no previous pose to interpolate from. Reported separately so a population
    // that behaves perfectly does not sit at "99.7% PARTIAL" for the run's whole length because of
    // its own first tick. A miss on an object that HAS been seen before is a real gap and stays in
    // the defect column.
    if (outFirstEverSighting != nullptr) {
      *outFirstEverSighting = firstEverSighting;
    }
    return false;
  }

  // The camera that belongs to `was`. Refuse rather than substitute: pairing a three-tick-old pose
  // against this tick's camera would attribute the camera's whole motion to the object.
  const float* invViewWas = nullptr;
  for (const InvViewSample& v : g_invHist) {
    if (v.stamp == was.stamp) { invViewWas = v.m; break; }
  }
  if (gap > 1 && invViewWas == nullptr) {
    ++g_unpaired;
    ++g_popGap[pop];
    ++g_gapRefusedNoCamera;
    return false;
  }
  if (gap > 1) ++g_gapPaired;

  // THE CHECK THAT KEEPS THE ORDINAL HONEST. Pairing within a tag assumes the object replays the
  // same display list each tick. If it did not, the vertex counts differ, and interpolating between
  // two unrelated poses would smear the object across the screen — far worse than not interpolating
  // it. Snap instead, and count it, so a systematic mismatch shows up as a number rather than as an
  // unexplained visual artefact.
  if (was.vtxCount != vtxCount) {
    ++g_mismatched;
    ++g_popMismatch[pop];
    return false;
  }

  // Is this pairing against LAST tick's sample, or an older one the swap left behind?
  {
    const long age = (was.stamp >= 0) ? (g_tickIndex - was.stamp) : -1;
    if (age == 1) {
      ++g_pairedFresh;
    } else {
      ++g_pairedStale;
      if (age > g_pairedStaleMaxAge) g_pairedStaleMaxAge = age;
    }
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
  if (g_haveInvCur && (invViewWas != nullptr || g_haveInvPrev)) {
    float objCur[12];
    float objPrev[12];
    compose(g_invViewCur, mine.pos, objCur);
    compose(invViewWas != nullptr ? invViewWas : g_invViewPrev, was.pos, objPrev);
    const double ox = objCur[3] - objPrev[3];
    const double oy = objCur[7] - objPrev[7];
    const double oz = objCur[11] - objPrev[11];
    const double od = std::sqrt(ox * ox + oy * oy + oz * oz);
    g_objDeltaSum += od;
    if (od > g_objDeltaMax) { g_objDeltaMax = od; }
    ++g_objDeltaN;
    // ── THE DISCONTINUITY SNAP ──────────────────────────────────────────────────────────────────
    //
    // A pair this far apart is not an object moving, whatever produced it. Either two different
    // objects shared a tag for one tick, or the object genuinely teleported — and the correct
    // frame is the same in both cases: show it where it is, do not sweep it across the screen.
    // Interpolating one of these puts a whole model in mid-air for half a frame.
    //
    // A FIXED SPEED BOUND WAS TRIED HERE FIRST AND IS GONE; the paragraph below says why, because
    // the reasoning that produced it was careful and still wrong, which is the kind worth keeping.
    // It refused any delta >= 100 units/tick, justified on a 593-tick plaza run (669,750 paired
    // draws below the bound, 182 above) plus an independent ground truth inside the bulk (gpMario
    // running peaks at 58.5 units/tick — motion_truth.cpp, claim C034, which still holds as an
    // observation). What that evidence could not contain was a scene faster than the plaza, and
    // the gate's own margin line was what caught it: in Pianta Village the largest ACCEPTED delta
    // sat at 99.9 against a bound of 100, and 12,791 draws were being refused per run.
    // ── IS THIS A TELEPORT, OR JUST SOMETHING FAST? ──────────────────────────────────────────
    //
    // The rule used to be a constant: refuse any delta >= 100 world units in a tick. It was
    // measured — on the PLAZA, where the distribution had 669,750 draws below 100 and 182 above,
    // three decades out. Pianta Village falsified it. There the refused deltas are contiguous with
    // the accepted bulk (12,657 in [100,1k) against 41,939 accepted in [10,100)) and printing the
    // coordinates settled it in one run: a single object walking 323.8 -> 318.3 -> 313.0 -> 307.7
    // units per tick along a smooth decelerating arc. Correctly paired, genuinely fast, and the
    // constant was snapping it every frame for 282 ticks.
    //
    // So the test is now the object's own CONTINUITY rather than a global speed limit. A teleport
    // is a STEP: this tick's motion has no relation to last tick's. Real motion, however fast, is
    // smooth at 30 Hz — the sequence above changes by ~5 units per tick. An object is therefore
    // allowed to move as far as it was ALREADY moving, times a generous factor, and the constant
    // survives only as the floor for an object with no history yet.
    //
    // WHY THE HISTORY IS STORED EVEN WHEN THE DRAW IS REFUSED: otherwise nothing that starts moving
    // fast can ever be accepted — its history stays at zero and every tick refuses on the floor,
    // forever, which is precisely the bug being fixed. Storing it costs one snapped frame at the
    // moment fast motion begins, and that frame is the one where the object's speed genuinely was
    // discontinuous.
    //
    // WHAT THIS STILL DOES NOT CATCH, stated rather than discovered later: a mispair that is
    // SUSTAINED — a tag aliased to another object of similarly steady motion — will legitimise
    // itself after one tick. The magnitude rule did not catch that either (it accepted anything
    // under 100 forever); what bounds it is the tag's own quality, measured separately, and within
    // a tick tags were found not to collide at all (frame_interp/shape_identity.cpp).
    // SCALED BY THE SPACING. `od` is the distance over `gap` ticks, so comparing it against a
    // per-tick allowance would refuse every gapped pair as a teleport purely for having waited.
    constexpr double kContinuityRatio = 4.0;
    const double allowed = ((was.objDelta > 0.0)
                                ? std::max(kDiscontinuity, kContinuityRatio * was.objDelta)
                                : kDiscontinuity) *
                           (double)gap;
    mine.objDelta = od;
    if (od >= allowed) {
      // The first few refusals, with both world positions. A count and a magnitude cannot tell a
      // rotating sub-part (positions tracing an arc around a fixed centre) from an alias (two fixed
      // points alternating) from a genuine mispair (unrelated positions) — the coordinates can, and
      // there is no way to reason to the answer from the magnitude alone.
      // Cap by NOVELTY, not by count: one line per tick, so a sample is not eight lines from the
      // same tick describing one event. (The first version printed the first eight refusals and all
      // eight came from tick 10 — a run's whole tail summarised by a single moment.)
      if (g_refusedShown < 20 && g_tickIndex != g_refusedLastTick) {
        g_refusedLastTick = g_tickIndex;
        ++g_refusedShown;
        Log.info("    refusal #{}: shape {:#010x} instance {:#010x} ordinal {} tick {} — world "
                 "{:.1f},{:.1f},{:.1f} -> {:.1f},{:.1f},{:.1f}  (delta {:.1f})",
                 g_refusedShown, (uint32_t)(tag >> 32), (uint32_t)(tag & 0xffffffffu), ordinal,
                 g_tickIndex, objPrev[3], objPrev[7], objPrev[11], objCur[3], objCur[7], objCur[11],
                 od);
      }
      ++g_snappedDiscontinuity;
      ++g_popRefused[pop];
      if (od > g_snappedDiscontinuityMax) { g_snappedDiscontinuityMax = od; }
      {
        int rb = 0;
        for (double edge = 0.1; rb < kObjBuckets - 1 && od >= edge; edge *= 10.0) { ++rb; }
        ++g_objHistRefused[rb];
        ++g_refusedPop[pop];
        ++g_refusedShape[(uint32_t)(tag >> 32)];
      }
      return false;   // the caller falls back to the camera delta alone, which is correct here
    }
    if (od > g_acceptedMax) { g_acceptedMax = od; }
    {
      int bucket = 0;
      for (double edge = 0.1; bucket < kObjBuckets - 1 && od >= edge; edge *= 10.0) {
        ++bucket;
      }
      ++g_objHist[bucket];
      ++g_objHistPop[pop][bucket];
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
  // ALPHA, REWEIGHTED FOR THE SPACING. `alpha` says where the presentation frame sits between the
  // PREVIOUS TICK and this one. When the paired sample is `gap` ticks old the same moment sits at
  // 1 - (1 - alpha)/gap of the way from it to here — for gap 1 this is alpha unchanged, so every
  // consecutive pair behaves exactly as before. Identical to the vertex path's rule, on purpose.
  const float a = 1.0f - (1.0f - alpha) / (float)gap;
  const float b = 1.0f - a;
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
// The full u8 id space, not the 11 hand-written labels it started as. The host allocates the ids
// above the curated block to EMITTER SITES it discovers at runtime, so this array is the audit's
// per-emitter table and its size is the ceiling on how many distinct emitters can be told apart.
// Anything beyond it is reported as an overflow by the host rather than folded into pop 0 — which
// is why this is 256 (the whole space the stream byte can carry) instead of a round number.
constexpr int kMaxPop = 256;
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

int max_populations() { return kMaxPop; }

int audit_disposition_count() { return (int)Disposition::Count; }

void audit_row(uint8_t pop, long* out, int outLen) {
  if (out == nullptr) {
    return;
  }
  for (int d = 0; d < outLen; ++d) {
    out[d] = (pop < kMaxPop && d < (int)Disposition::Count) ? g_audit[pop][d] : 0;
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

// ── IS THE SCREEN-SPACE GEOMETRY ACTUALLY STILL? ────────────────────────────────────────────────
namespace {
struct OrthoTick {
  uint64_t sum = 0;     // commutative over the tick's draws
  long count = 0;
  long stamp = -1;      // tick this belongs to
};
OrthoTick g_orthoCur[kMaxPop], g_orthoPrev[kMaxPop];
long g_orthoTicks[kMaxPop] = {};      // ticks with a previous tick to compare against
long g_orthoChanged[kMaxPop] = {};    // of those, how many differed
} // namespace

void note_ortho_geometry(uint8_t pop, const uint8_t* src, uint32_t uniformSize,
                         uint32_t mtxPosOffset) {
  if (pop >= kMaxPop || src == nullptr || mtxPosOffset == 0 ||
      mtxPosOffset + kMtxBytes > uniformSize) {
    return;
  }
  OrthoTick& cur = g_orthoCur[pop];
  if (cur.stamp != g_tickIndex) {
    // A tick boundary for THIS population. Compare against the last tick it drew in rather than
    // the numerically previous one: a 2D element that draws every other tick is not "changing"
    // just because it was absent in between.
    if (cur.stamp >= 0) {
      const OrthoTick& was = g_orthoPrev[pop];
      if (was.stamp >= 0) {
        ++g_orthoTicks[pop];
        if (was.sum != cur.sum || was.count != cur.count) ++g_orthoChanged[pop];
      }
      g_orthoPrev[pop] = cur;
    }
    cur = OrthoTick{};
    cur.stamp = g_tickIndex;
  }
  uint64_t h = 1469598103934665603ull;
  for (uint32_t i = 0; i < kMtxBytes; ++i) {
    h = (h ^ src[mtxPosOffset + i]) * 1099511628211ull;
  }
  h ^= h >> 29;
  h *= 0xbf58476d1ce4e5b9ull;
  h ^= h >> 32;
  cur.sum += h;
  ++cur.count;
}

void report_ortho_motion() {
  long anyMeasured = 0;
  for (int p = 0; p < kMaxPop; ++p) anyMeasured += g_orthoTicks[p];
  if (anyMeasured == 0) {
    Log.info("screen-space motion: NOTHING was measured — no orthographic draw was seen on two "
             "ticks this run. This says nothing about whether 2D snapping is correct.");
    return;
  }
  Log.info("screen-space motion — whether `snap:2D` is PROVABLY correct or merely unexamined. A "
           "static element has no in-between and snapping it is right. This measures a commutative "
           "hash of every ortho draw's position matrix, per population, per tick, and it can only "
           "answer one of the two questions: a population that never differs is provably still, so "
           "snapping it is certainly right. A population that DIFFERS has not been shown to judder "
           "— the difference may be smooth motion, which does have an in-between this path does "
           "not produce, or a discrete content change (a different glyph, a meter reading, a "
           "different number of elements), which does not. Separating those needs a per-element "
           "identity that 2D draws do not carry.");
  for (int p = 0; p < kMaxPop; ++p) {
    if (g_orthoTicks[p] == 0) continue;
    const double pct = 100.0 * (double)g_orthoChanged[p] / (double)g_orthoTicks[p];
    Log.info("  {:<22} {:>7} of {:>7} tick(s) differed from the previous one ({:.1f}%) — {}",
             g_popName[p].empty() ? (p == 0 ? "(unlabelled)" : "pop " + std::to_string(p))
                                  : g_popName[p],
             g_orthoChanged[p], g_orthoTicks[p], pct,
             g_orthoChanged[p] == 0
                 ? "PROVABLY STILL, so snapping it is exactly right"
                 : "NOT still — could be smooth motion or a discrete content change, and this "
                   "measure cannot tell them apart, so `snap:2D` here is a description of what "
                   "happens rather than a verdict that nothing was lost");
  }
}

void report_audit() {
  static const char* kName[(int)Disposition::Count] = {
      "unclaimed", "PAIRED",     "billboard",  "camera-only",
      "snap:2D",   "snap:EXACT", "snap:NO-ID", "birth"};
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
           "in-between); birth is CORRECT too (nothing existed to interpolate from); camera-only "
           "and snap:NO-ID are the defects — geometry that follows the camera but not its own "
           "motion, or nothing at all.");
  Log.info("  {:<22} {:>10} {:>11} {:>12} {:>10} {:>11} {:>11} {:>8}  {}", "population", "PAIRED",
           "billboard", "camera-only", "snap:2D", "snap:EXACT", "snap:NO-ID", "birth", "verdict");
  // Ordered by size, and CAPPED — the population space is now the whole u8 range because the host
  // allocates ids to emitter sites it discovers at runtime, so a run can legitimately fill dozens of
  // rows. The cap is on the SMALL rows, never the large ones, and what it drops is stated with its
  // draw count: a truncated list that does not say it truncated reads as a complete one.
  std::vector<std::pair<int, long>> rows;
  for (int p = 0; p < kMaxPop; ++p) {
    long sum = 0;
    for (int d = 0; d < (int)Disposition::Count; ++d) {
      sum += g_audit[p][d];
    }
    if (sum > 0) {
      rows.emplace_back(p, sum);
    }
  }
  std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second > b.second; });
  constexpr size_t kMaxRows = 32;
  long omittedDraws = 0;
  size_t omittedRows = 0;
  for (size_t i = kMaxRows; i < rows.size(); ++i) {
    ++omittedRows;
    omittedDraws += rows[i].second;
  }
  for (size_t i = 0; i < rows.size() && i < kMaxRows; ++i) {
    const int p = rows[i].first;
    // Anything still Pending was never claimed by any patch: perspective, no identity.
    const long noId = g_audit[p][(int)Disposition::SnappedNoIdentity] +
                      g_audit[p][(int)Disposition::Pending];
    const long good = g_audit[p][(int)Disposition::Paired] + g_audit[p][(int)Disposition::Billboard];
    // CameraOnlyStatic is CORRECT, not a shortfall: geometry that did not move needs the camera
    // delta and nothing else. Counting it as a defect is what made "97.3% PARTIAL" understate the
    // world-geometry row.
    const long bad = g_audit[p][(int)Disposition::CameraOnly] + noId;
    // Births are in NEITHER column. A draw whose object is being seen for the first time has no
    // previous pose to interpolate from, so counting it as a defect makes a perfect population
    // permanently imperfect — every once-per-tick emitter used to sit at exactly 99.7% for the
    // whole run because of its own first tick. Counting it as a success would be the opposite lie.
    const long birth = g_audit[p][(int)Disposition::CameraOnlyBirth];
    // A 2D population with no interpolated draws is CORRECT, not a failure — a screen-space
    // element has no meaningful in-between. Saying "interpolates (0.0% move)" of it, as the first
    // version did, is the report contradicting itself in one line.
    const char* verdict =
        // All births and nothing else: the population drew once and never again, so there was
        // never a pair to make. Saying "2D" of it would be a claim about a projection nobody
        // measured.
        (bad == 0 && good == 0 && birth > 0 && g_audit[p][(int)Disposition::SnappedOrtho] == 0 &&
         g_audit[p][(int)Disposition::SnappedExact] == 0)
            ? "CORRECT (drew once — a first sighting has nothing to pair with)"
        : (bad == 0 && good == 0 && g_audit[p][(int)Disposition::SnappedExact] > 0 &&
           g_audit[p][(int)Disposition::SnappedOrtho] == 0)
            ? "CORRECT (screen-space: must NOT move)"
        : (bad == 0 && good == 0) ? "CORRECT (2D: no in-between exists)"
        : bad == 0                ? "interpolates"
        : good == 0               ? "SNAPS ENTIRELY"
                                  : "PARTIAL";
    Log.info("  {:<22} {:>10} {:>11} {:>12} {:>10} {:>11} {:>11} {:>8}  {} ({:.1f}% interpolate, "
             "excluding {} first-ever sighting(s) which had nothing to pair with; camera-only is "
             "an UPPER BOUND on the defect, not a measurement — see interp.cpp)",
             g_popName[p].empty() ? (p == 0 ? "(unlabelled)" : "pop " + std::to_string(p))
                                  : g_popName[p],
             g_audit[p][(int)Disposition::Paired], g_audit[p][(int)Disposition::Billboard],
             g_audit[p][(int)Disposition::CameraOnly], g_audit[p][(int)Disposition::SnappedOrtho],
             g_audit[p][(int)Disposition::SnappedExact], noId, birth, verdict,
             // Denominator is what OUGHT to move: 2D and provably-static draws are excluded,
             // because a percentage that counts them as failures cannot reach 100 even when the
             // path is perfect.
             (good + bad) > 0 ? 100.0 * (double)good / (double)(good + bad) : 100.0, birth);
  }
  if (omittedRows > 0) {
    Log.info("  ... and {} further population(s) accounting for {} draw(s), not shown here. Every "
             "one of them IS recorded — the graphics registry file holds the full list "
             "(tools/gfx/graphics_db.py list).",
             omittedRows, omittedDraws);
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
// Count-change alignment: of the draws whose vertex count changed, how many had their shared
// segment at the FRONT and how many at the BACK, plus the mean per-coordinate distance of the
// winning and losing alignment. Both sums exist so the win rate can be read against whether the
// winner is actually close — one of two numbers is always smaller, which makes a bare win rate
// unfalsifiable.
long g_vtxAlignPrefix = 0, g_vtxAlignSuffix = 0;
double g_vtxAlignWinSum = 0.0, g_vtxAlignLoseSum = 0.0;

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
                    const uint8_t* src, uint8_t* dst, float alpha, uint8_t pop) {
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

  // ── AN OBJECT THAT SKIPS TICKS MAY STILL INTERPOLATE, IF THE ALPHA IS SCALED ────────────────
  //
  // The rule was "consecutive ticks or snap", and it cost more than it looked like: the shadow
  // alpha cubes are drawn by about sixty groups of which only ~18 draw on any given tick, so 2,675
  // of 5,162 cube draws had no consecutive predecessor and snapped. Their identities are fine —
  // 62 first sightings in 290 ticks, keys stable — they simply do not draw every tick.
  //
  // The generalisation is exact rather than a loosening. `alpha` says where the presentation frame
  // sits between the PREVIOUS tick and this one. When the last sample is `gap` ticks old, the same
  // moment sits at 1 - (1 - alpha)/gap of the way from that sample to this one — for gap 1 this is
  // alpha unchanged, so consecutive pairs behave exactly as before, and for gap 3 the in-between
  // frame moves a sixth of the way back instead of a half. Interpolating between the two most
  // recent samples of the same object, weighted by how far apart they are, is the correct answer;
  // the old rule was approximating a 1-tick spacing that was not there.
  //
  // A GAP IS STILL BOUNDED. Past a few ticks the older sample stops describing anything nearby —
  // the object may have been off-screen, culled, or somewhere else entirely — and a snap is safer
  // than a sweep across whatever happened while nobody was looking. Draws refused for that are
  // counted, not silently dropped, so the bound can be judged rather than assumed.
  constexpr long kMaxGap = 4;
  const long gap = g_tickIndex - rec.stamp;
  const bool consecutive = gap >= 1 && gap <= kMaxGap && rec.stamp != 0;
  const bool sameCount = rec.pos.size() == need;
  bool patched = false;
  if (consecutive && sameCount) {
    const float ga = 1.0f - (1.0f - alpha) / (float)gap;
    for (uint32_t v = 0; v < vtxCount; ++v) {
      uint8_t* q = dst + (size_t)v * stride + posOffset;
      for (int c = 0; c < 3; ++c) {
        const float a = rec.pos[v * 3 + c];
        put_be_f32(q + c * 4, a + (cur[v * 3 + c] - a) * ga);
      }
    }
    ++g_vtxPatched;
    ++g_vtxPopPatched[pop];
    if (gap > 1) { ++g_vtxGapPatched; }
    patched = true;
  } else if (!consecutive) {
    ++g_vtxUnpaired;
    ++g_vtxPopUnpaired[pop];
    if (gap > kMaxGap) { ++g_vtxTooStale; }
    // The GAP ITSELF, bucketed. Raising a bound because a recovery was disappointing is guessing;
    // this says whether the misses are near-misses at all. `rec.stamp == 0` is the first sighting
    // of a tag and is filed separately — it is not a gap, it is an object that has never drawn.
    if (rec.stamp == 0) {
      ++g_vtxFirstSight;
    } else {
      const long g = gap < 0 ? 0 : gap;
      ++g_vtxGapHist[g < 8 ? g : 8];
    }
  } else {
    ++g_vtxCountChanged;
    ++g_vtxPopCountChanged[pop];
    // IS THE SHARED PART A PREFIX OR A SUFFIX? A count change is currently a hard snap, and for a
    // particle chain that gained a link that is heavier than it needs to be: the links that already
    // existed still correspond, and only the new one has no partner. But WHICH end grows is a
    // property of how the chain is built, and guessing it would lerp two unrelated segments
    // together — worse than snapping.
    //
    // So this measures it instead of assuming. Both alignments are scored on the same min(N) shared
    // vertices, and BOTH means are reported: if the winner's mean is not far below the loser's, the
    // shared segment is not really shared and neither alignment is usable. A win rate alone would
    // hide that, because one of two numbers is always smaller.
    const size_t prevN = rec.pos.size() / 3;
    const size_t m = prevN < vtxCount ? prevN : vtxCount;
    if (m > 0) {
      double pre = 0.0;
      double suf = 0.0;
      for (size_t v = 0; v < m; ++v) {
        for (int c = 0; c < 3; ++c) {
          const double dp = (double)cur[v * 3 + c] - (double)rec.pos[v * 3 + c];
          pre += dp < 0 ? -dp : dp;
          const double ds = (double)cur[(vtxCount - m + v) * 3 + c] -
                            (double)rec.pos[(prevN - m + v) * 3 + c];
          suf += ds < 0 ? -ds : ds;
        }
      }
      pre /= (double)(m * 3);
      suf /= (double)(m * 3);
      if (pre <= suf) {
        ++g_vtxAlignPrefix;
        g_vtxAlignWinSum += pre;
        g_vtxAlignLoseSum += suf;
      } else {
        ++g_vtxAlignSuffix;
        g_vtxAlignWinSum += suf;
        g_vtxAlignLoseSum += pre;
      }
    }
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
  Log.info("  of those, {} were interpolated ACROSS a skipped tick with alpha scaled by the spacing "
           "(an object that does not draw every tick still has two real samples); {} were refused "
           "for a last sample older than 4 ticks, where the older pose no longer describes anywhere "
           "the object has recently been.",
           g_vtxGapPatched, g_vtxTooStale);
  Log.info("  why the rest missed: {} were a tag's FIRST sighting (never drawn before — no pair can "
           "exist); gaps for the others, in ticks: 0 {} | 1 {} | 2 {} | 3 {} | 4 {} | 5 {} | 6 {} | "
           "7 {} | 8+ {}. A gap of 0 means the same tag drew TWICE in one tick, which is a tagging "
           "collision rather than a spacing problem.",
           g_vtxFirstSight, g_vtxGapHist[0], g_vtxGapHist[1], g_vtxGapHist[2], g_vtxGapHist[3],
           g_vtxGapHist[4], g_vtxGapHist[5], g_vtxGapHist[6], g_vtxGapHist[7], g_vtxGapHist[8]);
  // HOW OLD THE SAMPLE EACH PAIRING USED ACTUALLY WAS. A pairing counts as a success whatever the
  // spacing, so this is the only place a wrong-spacing pairing can show up at all.
  {
    const long total = g_pairedFresh + g_pairedStale;
    if (total == 0) {
      Log.info("  pairing freshness: NOTHING paired, so nothing was measured here.");
    } else {
      Log.info("  pairing freshness: {} of {} pairing(s) used the PREVIOUS tick's sample ({:.2f}%); "
               "{} used an older one, up to {} tick(s) old. `alpha` places the in-between frame "
               "assuming a one-tick spacing, so every stale pairing moves its object by the wrong "
               "fraction of a step — and it is counted as a success everywhere else.{}",
               g_pairedFresh, total, 100.0 * (double)g_pairedFresh / (double)total, g_pairedStale,
               g_pairedStaleMaxAge,
               g_pairedStale == 0
                   ? "  Zero here means the tables never hand back a stale sample, which is what "
                     "begin_tick()'s clear of g_cur is supposed to guarantee. It is corroborated "
                     "rather than merely asserted: the self-test's gap case draws a tag, skips a "
                     "tick and draws again, and the table REFUSES that pairing instead of "
                     "returning the older sample."
                   : "  A NONZERO count means the swap is handing back samples it should not.");
    }
  }
  // WHAT THE GAP TOLERANCE ACTUALLY DID. Three numbers, because "0 recovered" has three different
  // causes and the fix for each is different: the gaps are longer than the bound, the camera for the
  // older sample was not retained, or there were no gaps to recover in the first place.
  Log.info("  gap tolerance (bound {} tick(s)): {} pairing(s) made across a skipped tick, {} "
           "refused for a sample older than the bound, {} refused because the camera from that "
           "sample's tick was no longer retained.{}",
           kMaxGapMtx, g_gapPaired, g_gapTooStale, g_gapRefusedNoCamera,
           (g_gapPaired == 0 && g_gapTooStale == 0 && g_gapRefusedNoCamera == 0)
               ? "  All three zero means no draw ever skipped a tick and came back — the tolerance "
                 "was never exercised, so this run says nothing about whether it works."
               : "");
  Log.info("  gap LENGTHS, in ticks: 1 {} | 2 {} | 3 {} | 4 {} | 5 {} | 6 {} | 7 {} | 8 {} | 9+ {}. "
           "This is what decides whether raising the bound would help: near-misses clustered at 2-4 "
           "are recoverable, a tail at 9+ is an object that was CULLED and came back, and sweeping "
           "that one across wherever it went while off-screen is exactly what snapping prevents.",
           g_mtxGapHist[1], g_mtxGapHist[2], g_mtxGapHist[3], g_mtxGapHist[4], g_mtxGapHist[5],
           g_mtxGapHist[6], g_mtxGapHist[7], g_mtxGapHist[8], g_mtxGapHist[9]);
  // WHY EACH POPULATION'S MATRIX RESIDUAL EXISTS. Printed only for populations that HAVE one, and
  // with all three causes even when two are zero: "12 gaps" alone leaves the reader to assume the
  // other two were zero rather than unmeasured.
  {
    bool any = false;
    for (int p = 0; p < kMaxPop; ++p) {
      const long total = g_popGap[p] + g_popMismatch[p] + g_popRefused[p];
      if (total == 0) continue;
      any = true;
      Log.info("  matrix residual: {:<22} {} draw(s) did not pair — {} skipped a tick and came "
               "back (the object was there before, so this is a gap the pairing table lost), {} "
               "changed display-list length (no vertex correspondence; snapping is correct), {} "
               "refused as discontinuous (a step change, deliberately not smeared).",
               g_popName[p].empty() ? (p == 0 ? "(unlabelled)" : "pop " + std::to_string(p))
                                    : g_popName[p],
               total, g_popGap[p], g_popMismatch[p], g_popRefused[p]);
    }
    if (!any) {
      Log.info("  matrix residual: NONE — every tagged draw in every population either paired or "
               "was a first sighting. That is the line to watch: it reading nothing is a result, "
               "not a missing report.");
    }
  }
  // WHERE THE SHARED SEGMENT SITS on a count change. This does not change behaviour — a count
  // change still snaps — it says whether lerping the shared part would be sound, and for which end.
  if (g_vtxCountChanged > 0) {
    const long aligned = g_vtxAlignPrefix + g_vtxAlignSuffix;
    if (aligned == 0) {
      Log.info("  count-change alignment: {} draw(s) changed vertex count but NONE could be "
               "scored (no shared vertices at all). Nothing can be said about which end grows.",
               g_vtxCountChanged);
    } else {
      const double win = g_vtxAlignWinSum / (double)aligned;
      const double lose = g_vtxAlignLoseSum / (double)aligned;
      Log.info("  count-change alignment: of {} scored draw(s), {} matched better as a PREFIX (the "
               "shared vertices are at the front, growth at the back) and {} as a SUFFIX. Mean "
               "per-coordinate distance {:.3f} for the winning alignment against {:.3f} for the "
               "losing one — a ratio near 1 means NEITHER end really corresponds and lerping the "
               "shared segment would smear two unrelated shapes, whatever the win rate says.",
               aligned, g_vtxAlignPrefix, g_vtxAlignSuffix, win, lose);
    }
  }
  // BY POPULATION, because the two failure modes mean opposite things and the total cannot separate
  // them. A COUNT CHANGE is a ceiling: a particle chain that gained a link, or a mesh rebuilt at a
  // different resolution, has no vertex correspondence and snapping is the correct answer. A
  // NON-CONSECUTIVE tick is a seam question: the object drew, then did not, then drew again, and
  // whether that is legitimate (it left the scene) or a tagging gap is worth knowing per system.
  for (int p = 0; p < kMaxPop; ++p) {
    const long t = g_vtxPopPatched[p] + g_vtxPopUnpaired[p] + g_vtxPopCountChanged[p];
    if (t == 0) continue;
    Log.info("  vertex path: {:<28} {} of {} lerped ({:.1f}%) — {} not consecutive, {} changed "
             "vertex count",
             g_popName[p].empty() ? (p == 0 ? std::string("(unlabelled)")
                                            : "pop " + std::to_string(p))
                                  : g_popName[p],
             g_vtxPopPatched[p], t, 100.0 * (double)g_vtxPopPatched[p] / (double)t,
             g_vtxPopUnpaired[p], g_vtxPopCountChanged[p]);
  }
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
    // WHICH KIND OF MISS, because "new particle, or one that skipped a tick" is two facts and only
    // one of them is a ceiling. A particle that has never had a previous position is a BIRTH and no
    // pairing can exist for it — in a system that spawns and kills continuously that is simply the
    // rate. A particle whose samples exist but are not adjacent has drawn, stopped and drawn again,
    // and that is a question about the seam or about tag reuse. Counting them together makes a
    // birth rate and a defect indistinguishable.
    if (b.stampPrev == 0) {
      ++g_billboardBirth;
    } else {
      ++g_billboardGap;
    }
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
  // Keep this tick's inverse view for as long as a gapped pair may reach back for it. Written here,
  // where it is computed, so the ring cannot hold a matrix that was never valid.
  if (g_haveInvCur) {
    InvViewSample& slot = g_invHist[(size_t)(g_tickIndex % (long)(sizeof(g_invHist) / sizeof(g_invHist[0])))];
    slot.stamp = g_tickIndex;
    std::memcpy(slot.m, g_invViewCur, sizeof(slot.m));
  }
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
  g_ent.clear();
  g_cursor.clear();
  g_paired = g_unpaired = g_mismatched = 0;
  g_transDeltaSum = g_transDeltaMax = 0.0;
  g_transDeltaN = 0;
  g_objDeltaSum = g_objDeltaMax = 0.0;
  g_objDeltaN = g_objDeltaUnavailable = 0;
  g_snappedDiscontinuity = 0;
  g_snappedDiscontinuityMax = g_acceptedMax = 0.0;
  for (int i = 0; i < kObjBuckets; ++i) { g_objHist[i] = 0; g_objHistRefused[i] = 0; }
  for (int p = 0; p < 256; ++p) { g_refusedPop[p] = 0; }
  g_refusedShape.clear();
  g_refusedShown = 0;
  g_refusedLastTick = -1;
  for (int p = 0; p < kMaxPop; ++p) {
    for (int i = 0; i < kObjBuckets; ++i) { g_objHistPop[p][i] = 0; }
  }
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
    bool wantPaired;          // false = the discontinuity gate is expected to REFUSE this one
    double wantTotal, wantObj;
  };
  // THE OBJECT CASE MOVES 50 UNITS, NOT 1000, AND THAT IS THE POINT OF THE THIRD CASE.
  //
  // It used to move 1000, and when the discontinuity gate landed (>= 100 units/tick refuses to pair)
  // this self-test started failing on every run — correctly, because a 1000-unit step is exactly
  // what the gate exists to refuse. The attribution being tested here is a separate question from
  // the gate, so the attribution cases now sit INSIDE the accepted range (50 units/tick is ordinary
  // motion: gpMario peaks at 58.5, claim C034), and the gate gets a case of its own that must be
  // REFUSED. Two mechanisms, two controls — collapsing them is how the failure went unnoticed for a
  // whole session of runs that each printed the error.
  const Case cases[] = {
      {"camera moves 1000, object static", {0, 0, 0}, {1000, 0, 0}, {0, 0, 0}, {0, 0, 0}, true, 1000.0, 0.0},
      {"camera static, object moves 50", {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {50, 0, 0}, true, 50.0, 50.0},
      // The gate's own control: it must FIRE on a step three decades past the bound. Without this,
      // a gate that never fired (kDiscontinuity set to infinity, the comparison inverted) would pass
      // every other check in this file, and its report line would read "0 refused" — which is
      // indistinguishable from a scene that simply contains no discontinuity.
      {"object teleports 1000 -- must be REFUSED", {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1000, 0, 0}, false, 0.0, 0.0},
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
    patch_draw(1, 3, src.data(), dst.data(), kSize, kPos, kNrm, 0.5f, 0, 0, 0);
    end_tick();

    begin_tick();
    set_view_matrix(v1);
    begin_camera_delta(0.5f);
    write_draw_block(src.data(), kPos, kNrm, v1, c.obj1[0], c.obj1[1], c.obj1[2]);
    const bool paired = patch_draw(1, 3, src.data(), dst.data(), kSize, kPos, kNrm, 0.5f, 0, 0, 0);

    const double gotTotal = g_transDeltaN ? g_transDeltaSum / (double)g_transDeltaN : -1.0;
    const double gotObj = g_objDeltaN ? g_objDeltaSum / (double)g_objDeltaN : -1.0;
    bool pass;
    if (c.wantPaired) {
      pass = paired && std::fabs(gotTotal - c.wantTotal) < 0.01 &&
             std::fabs(gotObj - c.wantObj) < 0.01;
    } else {
      // Refused for the RIGHT REASON: not paired, and the discontinuity counter is the thing that
      // moved. A draw refused by the vertex-count gate or by a missing previous tick would also
      // report paired=false, and that would not be this gate working.
      pass = !paired && g_snappedDiscontinuity == 1;
    }
    if (!pass) {
      ok = false;
      if (c.wantPaired) {
        Log.error("SELFTEST FAILED [{}]: paired={} total delta {:.3f} (want {:.3f}) object delta "
                  "{:.3f} (want {:.3f}). The camera/object attribution does not discriminate, so any "
                  "conclusion drawn from those two numbers is unfounded.",
                  c.name, paired, gotTotal, c.wantTotal, gotObj, c.wantObj);
      } else {
        Log.error("SELFTEST FAILED [{}]: paired={} (want false), discontinuity refusals {} (want 1). "
                  "The gate that is supposed to refuse a teleport did not fire on one, so its "
                  "\"0 refused\" in the audit means nothing.",
                  c.name, paired, g_snappedDiscontinuity);
      }
    }
  }
  // ── THE BIRTH FLAG'S OWN CONTROL ────────────────────────────────────────────────────────────
  //
  // The audit now excludes first-ever sightings from the pass/fail denominator, which means a bug
  // that reported EVERY unpaired draw as a birth would show every population at 100% and look like
  // success. So the flag is run against both classes: a tag that has never been seen (must read
  // true) and a tag that drew, went away, and came back (must read FALSE — that is a real gap, and
  // it is the answer a broken flag cannot give).
  //
  // Distinct tags, deliberately: reusing tag 1 would inherit the ever-seen state the cases above
  // leave behind, and the test would pass or fail on ordering rather than on the mechanism.
  {
    float v[12];
    make_view_at(0, 0, 0, v);
    constexpr uint64_t kNewTag = 0xB100u;
    constexpr uint64_t kGapTag = 0xB200u;

    reset_stats();
    bool birthNew = false;
    begin_tick();
    set_view_matrix(v);
    begin_camera_delta(0.5f);
    write_draw_block(src.data(), kPos, kNrm, v, 0, 0, 0);
    const bool pairedFirst =
        patch_draw(kNewTag, 3, src.data(), dst.data(), kSize, kPos, kNrm, 0.5f, 0, 0, 0, &birthNew);
    // Same tick, and the gap tag draws for the first time here so it has a history to lose.
    write_draw_block(src.data(), kPos, kNrm, v, 0, 0, 0);
    patch_draw(kGapTag, 3, src.data(), dst.data(), kSize, kPos, kNrm, 0.5f, 0, 0, 0);
    end_tick();

    // A tick in which kGapTag does not draw at all. kNewTag draws again and must now PAIR — a flag
    // stuck at true would be caught here as well.
    bool birthSecond = false;
    begin_tick();
    set_view_matrix(v);
    begin_camera_delta(0.5f);
    write_draw_block(src.data(), kPos, kNrm, v, 0, 0, 0);
    const bool pairedSecond =
        patch_draw(kNewTag, 3, src.data(), dst.data(), kSize, kPos, kNrm, 0.5f, 0, 0, 0,
                   &birthSecond);
    end_tick();

    // kGapTag returns after ONE skipped tick. It must now PAIR — that is the gap tolerance — and it
    // must not be a birth. Before the tolerance landed this case asserted the opposite, and the
    // self-test failing when the behaviour changed is the control doing its job rather than a
    // regression.
    bool birthReturn = true;
    begin_tick();
    set_view_matrix(v);
    begin_camera_delta(0.5f);
    write_draw_block(src.data(), kPos, kNrm, v, 0, 0, 0);
    const bool pairedReturn =
        patch_draw(kGapTag, 3, src.data(), dst.data(), kSize, kPos, kNrm, 0.5f, 0, 0, 0,
                   &birthReturn);
    end_tick();

    if (pairedFirst || !birthNew) {
      ok = false;
      Log.error("SELFTEST FAILED [birth flag]: a tag never seen before reported paired={} birth={} "
                "(want false/true). Every first sighting would be filed as a DEFECT, and every "
                "once-per-tick population would read PARTIAL forever.",
                pairedFirst, birthNew);
    }
    if (!pairedSecond || birthSecond) {
      ok = false;
      Log.error("SELFTEST FAILED [birth flag]: the same tag one tick later reported paired={} "
                "birth={} (want true/false). A flag that stays set would exclude real draws from "
                "the audit denominator.",
                pairedSecond, birthSecond);
    }
    if (!pairedReturn || birthReturn) {
      ok = false;
      Log.error("SELFTEST FAILED [gap tolerance]: a tag that drew, skipped ONE tick and came back "
                "reported paired={} birth={} (want true/false). Either the gap tolerance is not "
                "reaching a two-tick-old sample, or an unpaired draw is being called a birth — the "
                "second would make the audit's 100% rows a tautology.",
                pairedReturn, birthReturn);
    }

    // AND THE OTHER SIDE OF THE BOUND. A tolerance with no upper limit is not a tolerance, and a
    // limit nobody has seen refuse anything is indistinguishable from an infinite one. This tag
    // waits kMaxGapMtx + 1 ticks, which MUST be refused.
    constexpr uint64_t kStaleTag = 0xB300u;
    bool birthStale = true;
    begin_tick();
    set_view_matrix(v);
    begin_camera_delta(0.5f);
    write_draw_block(src.data(), kPos, kNrm, v, 0, 0, 0);
    patch_draw(kStaleTag, 3, src.data(), dst.data(), kSize, kPos, kNrm, 0.5f, 0, 0, 0);
    end_tick();
    for (long i = 0; i < kMaxGapMtx + 1; ++i) {
      begin_tick();
      set_view_matrix(v);
      begin_camera_delta(0.5f);
      end_tick();
    }
    begin_tick();
    set_view_matrix(v);
    begin_camera_delta(0.5f);
    write_draw_block(src.data(), kPos, kNrm, v, 0, 0, 0);
    const bool pairedStale =
        patch_draw(kStaleTag, 3, src.data(), dst.data(), kSize, kPos, kNrm, 0.5f, 0, 0, 0,
                   &birthStale);
    end_tick();
    if (pairedStale || birthStale) {
      ok = false;
      Log.error("SELFTEST FAILED [gap bound]: a sample {} tick(s) old reported paired={} birth={} "
                "(want false/false). A tolerance with no upper bound sweeps an object across "
                "wherever it went while nobody was looking.",
                kMaxGapMtx + 2, pairedStale, birthStale);
    }
  }

  reset_stats();
  if (ok) {
    Log.info("interp selftest PASSED: camera/object attribution separates a 1000-unit camera move "
             "(object delta 0) from a 50-unit object move (object delta 50) — it has been run "
             "against both classes, not just the one it is expected to find — and the discontinuity "
             "gate demonstrably FIRES on a 1000-unit teleport, so its refusal count is a real "
             "measurement rather than a switch nobody has seen move. The birth flag was run "
             "against both classes too: a never-seen tag reads birth, and a tag that drew, skipped "
             "a tick and returned reads NOT a birth — so the audit's birth column cannot be "
             "swallowing real gaps. The gap tolerance is bounded at both ends: a one-tick gap "
             "PAIRS and a gap past the limit is REFUSED, both demonstrated rather than assumed.");
  }
  return ok;
}

long tick_index() { return g_tickIndex; }

void end_tick() {
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
  if (g_billboardUnpaired != 0) {
    Log.info("  of those unpaired: {} were a particle's FIRST sighting (a birth — no pair can exist, "
             "and in a system that spawns continuously this is the spawn rate, not a defect) and {} "
             "had samples that were not adjacent (drew, stopped, drew again — that one is a question "
             "about the seam or about tag reuse).",
             g_billboardBirth, g_billboardGap);
  }
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
    // WHAT THIS LINE USED TO SAY WAS WRONG, and it is worth the space to say so where the number is
    // printed. It read "anything from [10,100) up is a pose no object reaches in 1/30 s, so those
    // counts are the mispairings" — a threshold nobody had measured, condemning 84,507 draws on a
    // plaza run. gpMario, walking and running under a pad script, spends 90 of 593 ticks in
    // [10,100) and peaks at 58.5 units/tick (sms-recomp/frame_interp/motion_truth.cpp, claim C034).
    // [10,100) is what MOTION looks like in this game's units.
    //
    // The boundary the data actually supports is 100, and it is supported by a GAP rather than by a
    // preference: the same run has 84,507 draws below it and 182 at or above, three decades away.
    Log.info("  object-motion distribution (world units/tick): [0,0.1) {} | [0.1,1) {} | [1,10) {} | "
             "[10,100) {} | [100,1k) {} | [1k,10k) {} | [10k,inf) {}. The first FOUR buckets are "
             "ordinary motion — a running Mario measures up to 58 units/tick. Only [100,inf) is "
             "beyond anything the game's own objects were observed to do, and those are the "
             "mispairings (or genuine teleports, which must snap either way).",
             g_objHist[0], g_objHist[1], g_objHist[2], g_objHist[3], g_objHist[4], g_objHist[5],
             g_objHist[6]);
    // WHO IS IN THE TAIL. Printed for every population with any draw at 10 units/tick or more,
    // because "6.8% of paired draws moved impossibly far" and "Mario and the particles moved fast"
    // are the same number until it is attributed.
    Log.info("  discontinuity snap: {} paired draw(s) refused as DISCONTINUOUS — each moved further "
             "in one tick than 4x what that same object moved in the tick before, with a floor of "
             "{:.0f} units for an object with no history yet (largest refused {:.1f}); largest "
             "delta ACCEPTED was {:.1f}, which is now allowed to exceed the floor because a fast "
             "object earns its own speed. A count that climbs with scene SPEED rather than with "
             "scene chaos would mean the ratio is too tight.",
             g_snappedDiscontinuity, kDiscontinuity, g_snappedDiscontinuityMax, g_acceptedMax);
    if (g_snappedDiscontinuity != 0) {
      // The refused deltas, on the SAME buckets. The accepted histogram above cannot show these —
      // a refusal returns before bucketing, so it reads [100,inf) 0 however many were cut — and the
      // shape here is what says whether the bound is right. Contiguous with the accepted bulk means
      // it is severing real motion; a cluster decades out means it is catching mispairs.
      Log.info("    refused-delta distribution (same buckets): [0,0.1) {} | [0.1,1) {} | [1,10) {} | "
               "[10,100) {} | [100,1k) {} | [1k,10k) {} | [10k,inf) {}",
               g_objHistRefused[0], g_objHistRefused[1], g_objHistRefused[2], g_objHistRefused[3],
               g_objHistRefused[4], g_objHistRefused[5], g_objHistRefused[6]);
      {
        std::vector<std::pair<uint32_t, long>> v(g_refusedShape.begin(), g_refusedShape.end());
        std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
        const size_t n = v.size() < 6 ? v.size() : 6;
        for (size_t i = 0; i < n; ++i) {
          Log.info("    refused most often: shape {:#010x} — {} draw(s) of {} total, over {} "
                   "distinct shape(s)",
                   v[i].first, v[i].second, g_snappedDiscontinuity, g_refusedShape.size());
        }
      }
      for (int p = 0; p < kMaxPop; ++p) {
        if (g_refusedPop[p] == 0) continue;
        Log.info("    refused by population: {:<28} {} draw(s)",
                 g_popName[p].empty() ? (p == 0 ? std::string("(unlabelled)")
                                               : "pop " + std::to_string(p))
                                      : g_popName[p],
                 g_refusedPop[p]);
      }
    }
    for (int p = 0; p < kMaxPop; ++p) {
      long tail = 0, total = 0;
      for (int b = 0; b < kObjBuckets; ++b) {
        total += g_objHistPop[p][b];
        if (b >= 3) { tail += g_objHistPop[p][b]; }
      }
      if (tail == 0) {
        continue;
      }
      Log.info("    motion tail by population: {:<24} {} of {} paired draw(s) at >=10 units/tick "
               "({:.1f}%) — buckets [10,100) {} | [100,1k) {} | [1k,10k) {} | [10k,inf) {}. The "
               "first of those is ordinary motion; the rest are the suspect ones.",
               g_popName[p].empty() ? (p == 0 ? "(unlabelled)" : "pop " + std::to_string(p))
                                    : g_popName[p],
               tail, total, total ? 100.0 * (double)tail / (double)total : 0.0,
               g_objHistPop[p][3], g_objHistPop[p][4], g_objHistPop[p][5], g_objHistPop[p][6]);
    }
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
