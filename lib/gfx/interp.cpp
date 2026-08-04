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
bool g_camDeltaValid = false;
long g_cameraPatched = 0;

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

void patch_camera_only(const uint8_t* src, uint8_t* dst, uint32_t uniformSize,
                       uint32_t mtxPosOffset, uint32_t mtxNrmOffset) {
  if (!g_camDeltaValid || dst == nullptr || src == nullptr) {
    return;
  }
  if (mtxPosOffset == 0 || mtxNrmOffset == 0 || mtxPosOffset + kMtxBytes > uniformSize ||
      mtxNrmOffset + kMtxBytes > uniformSize) {
    return;   // already reported by patch_draw's identical checks
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
    patch_draw(1, 3, src.data(), dst.data(), kSize, kPos, kNrm, 0.5f);
    end_tick();

    begin_tick();
    set_view_matrix(v1);
    begin_camera_delta(0.5f);
    write_draw_block(src.data(), kPos, kNrm, v1, c.obj1[0], c.obj1[1], c.obj1[2]);
    const bool paired = patch_draw(1, 3, src.data(), dst.data(), kSize, kPos, kNrm, 0.5f);

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
    Log.info("camera interpolation: view matrix supplied, {} draw uniforms carried the camera delta",
             g_cameraPatched);
  }
}

} // namespace aurora::gfx::interp
