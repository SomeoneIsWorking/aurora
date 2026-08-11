#pragma once
// interp — the matrix interpolation half of 60fps-from-a-30Hz-tick.
//
// The frame machinery lives in common.cpp (capture_replay_snapshot / install_replay_snapshot):
// one recorded frame, presented twice, the second emission reusing the first's geometry upload and
// re-pushing only the uniform block. That alone gives two IDENTICAL presents, which is the control,
// not the feature.
//
// This is what makes the first of the two presents show an in-between moment. It rewrites, in the
// recorded frame's live uniform staging, the two 480-byte matrix spans of each tagged draw, with
// each matrix lerped toward the value the SAME OBJECT had on the previous tick. The second emission
// still carries the tick's true matrices, because its uniform bytes were shadowed before any of
// this ran. So the pair presents (t-0.5) then (t) — standard interpolation, smoothness bought at
// half a tick of latency.
//
// WHY THIS IS SOUND HERE, when "patch the matrices" was rightly rejected for the retired attempt:
// that one paired draws by SEQUENCE POSITION and patched a stream in which matrices were already
// baked into display lists. Here the pairing key is supplied by the emitter (GX_AURORA_DRAW_TAG,
// the guest scene-graph pointer) and the patch target is the uniform block aurora itself built from
// g_gxState.pnMtx — by which point every display list has already been executed, so there is
// nothing baked left to fight.
//
// TWO APPROXIMATIONS, NAMED RATHER THAN HIDDEN:
//
//  * Matrices are lerped COMPONENTWISE. That is not a correct interpolation of a rotation — the
//    result is slightly non-orthonormal, shrinking toward the chord — but over a 1/30 s interval at
//    alpha 0.5 the angle is small and the error is far below a pixel. A slerp would need the
//    rotation decomposed per matrix per draw, which is not affordable here. This is the standard
//    trade; it is recorded so it can be revisited if something visibly shears.
//  * A draw is paired with the previous tick's draw at the same ORDINAL WITHIN ITS TAG. That is an
//    ordinal, which this project bans in general and for good reason — but it is scoped to a single
//    tagged object and reset at every tag, and within one object the same display list is replayed
//    each tick, so the sequence is the same by construction. It is checked rather than trusted: the
//    vertex count is stored alongside and a mismatch SNAPS that draw instead of smearing it between
//    two unrelated poses, and the mismatches are counted.

#include <cstdint>

namespace aurora::gfx::interp {

// Reset the per-tick pairing table's write cursor. Called once per recorded frame.
void begin_tick();

// Record this draw's true matrices for the next tick, and, if the same object drew last tick, write
// the interpolated matrices into `dst`. `alpha` is 0 at the previous tick's pose and 1 at this one.
//
// `src` and `dst` point at the SAME draw's uniform block in two places: `src` in the snapshot, which
// is ordinary cached RAM, and `dst` in the live GPU staging, which is write-combined. Every read
// comes from `src` and every write goes to `dst`, deliberately — reading back out of write-combined
// memory is uncached and slow, and doing it once per draw would put the cost of interpolation in
// exactly the wrong place.
//
// `uniformSize` is the block's size, used purely as a bounds check: writing 960 bytes at an offset
// that does not fit is a corruption, and it must fail loudly rather than scribble.
// Returns TRUE only if the draw was actually interpolated. A tagged draw that fails to pair (new
// object, or a vertex-count mismatch) is left untouched, and the caller MUST then give it the
// camera-only treatment — otherwise it sits at the current viewpoint while the rest of the frame
// does not, which is the same tearing failure as an untagged draw.
bool patch_draw(uint64_t tag, uint32_t vtxCount, const uint8_t* src, uint8_t* dst,
                uint32_t uniformSize, uint32_t mtxPosOffset, uint32_t mtxNrmOffset, float alpha,
                uint32_t texMtxCamMask, uint32_t pnMtxSlot);

// The view matrix in force for this tick, as the game built it (GC Mtx: 3 rows of 4 floats,
// p' = M*p). Supplied by the emitter through GX_AURORA_VIEW_MTX, because aurora cannot recover it:
// J3D concatenates the camera into every draw matrix in viewCalc, so what reaches pnMtx is
// model x view with no seam between them.
void set_view_matrix(const float m[12]);

// Apply the tick's CAMERA interpolation to a draw that has no identity of its own.
//
// This is what stops partial coverage from tearing the frame in two. A draw that cannot be paired
// still has the camera baked into its matrix, so leaving it alone renders it from the CURRENT
// viewpoint while every interpolated object is at the in-between one. Measured, that is worse than
// not interpolating at all: frame energy p90 went 15x when 69% of the scene interpolated and the
// rest did not.
//
// With column convention A = V*M, the wanted matrix is V_lerp*M = (V_lerp * V_cur^-1) * A, so one
// 4x4 per tick left-multiplied into every unpaired draw gives the whole frame a single coherent
// viewpoint while its object motion still snaps. Normals take only the rotation part, which is
// correct for a rigid view delta because the inverse-transpose of a rotation is itself.
//
// Does nothing if no view matrix has been supplied, or if the tick has no previous view to
// interpolate from.
// `src` is the cached-RAM copy of this draw's uniform block and `dst` the live GPU staging: every
// read comes from src, every write goes to dst, because reading write-combined memory back is
// uncached and was the dominant cost of interpolation until it was removed.
// ── THE AUDIT ───────────────────────────────────────────────────────────────────────────────────
//
// One global "77.8% of draws carry an identity" cannot distinguish a correctly-snapping HUD from
// world geometry stuttering, and it cannot say WHICH system is which. So every draw is filed under
// the population that emitted it (GX_AURORA_DRAW_POP) and the fate it actually received.
//
// The five outcomes are exhaustive by construction — every draw is noted once as Pending or
// SnappedOrtho and then, if it is patched, once more with its real outcome — so the columns sum to
// the draw count and a population cannot fall between them unnoticed.
enum class Disposition : uint8_t {
  Pending = 0,      // perspective, not yet decided; becomes SnappedNoIdentity if nothing claims it
  Paired,           // matrix lerped against the same object's previous tick — full interpolation
  Billboard,        // position carried in vertices; its own displacement applied as a translation
  CameraOnly,       // follows the camera but not its own motion. An UPPER BOUND on the defect:
                    // for STATIC world geometry this is exactly correct, and no sound test to
                    // separate the two has been built — see the note in interp.cpp
  SnappedOrtho,     // 2D/HUD; correct, a screen-space element has no meaningful in-between
  SnappedExact,     // declared screen-space-under-perspective by the emitter (GX_AURORA_DRAW_EXACT):
                    // correct, and correct for a reason the ortho test cannot see
  SnappedNoIdentity,// perspective with nothing to pair on — the honest remaining gap
  Count
};
void note_disposition(uint8_t pop, Disposition d);

// ── VERTEX INTERPOLATION, for geometry that DEFORMS ─────────────────────────────────────────────
//
// Flags and the sea ripple grid rebuild their mesh every tick. Their motion is in the VERTICES, not
// in any matrix, so no amount of matrix work reaches them — this is the only thing that does.
//
// `src` is the tick's raw vertex bytes (big-endian GC floats, as the shader reads them) and `dst` a
// SEPARATE buffer the caller has allocated for the interpolated emission. It must be separate: both
// emissions replay the same recorded passes and therefore the same vertRange, so patching in place
// would corrupt the tick's OWN frame. Uniforms escape this because the snapshot re-pushes them;
// vertices have no such path.
//
// Returns false — and touches nothing — when the tag has no usable previous tick, when the vertex
// COUNT changed (a mesh rebuilt at a different resolution has no correspondence, and smearing
// between two unrelated shapes is worse than snapping), or when the layout is not the one shape it
// understands.
bool patch_vertices(uint64_t tag, uint32_t vtxCount, uint16_t stride, uint16_t posOffset,
                    const uint8_t* src, uint8_t* dst, float alpha);
void report_vertex_interp();
// Population names, registered by the host so the report reads as systems rather than numbers.
void name_population(uint8_t pop, const char* name);
void report_audit();

// How many population slots exist. The host allocates ids out of this space at runtime (one per
// newly-discovered emitter site), so it needs the ceiling rather than assuming one.
int max_populations();

// How many outcomes Disposition has. The host duplicates the enum's ORDER across this boundary (the
// enum is in an internal header), so it needs to be able to check that assumption rather than hold
// it silently — a new outcome on one side only shifts every column.
int audit_disposition_count();

// Read one population's whole audit row into `out`, which must hold (int)Disposition::Count longs,
// indexed by Disposition. The host's graphics registry persists these per emitter across runs,
// which a log line cannot do: a population that appears in one stage and not the next is invisible
// in a per-run report and obvious in a file that accumulates.
//
// The whole row rather than one cell, because every consumer wants the denominator too — a PAIRED
// count on its own cannot say whether a population interpolates.
void audit_row(uint8_t pop, long* out, int outLen);

// ── BILLBOARDS ──────────────────────────────────────────────────────────────────────────────────
//
// Record a tagged object's WORLD position for this tick. Called by the host from the draw seam that
// knows the object (JPA's two billboard exec entries). World rather than eye space on purpose: an
// eye-space pair is expressed in two different view transforms, so its difference would fold camera
// motion into the object's own.
void set_tag_world_pos(uint64_t tag, float x, float y, float z);

// Patch a draw whose geometry carries its position in the VERTEX data rather than in a matrix — a
// JPA billboard being the case this exists for. Applies the camera delta AND the object's own
// interpolated displacement, as a translation, which is exact for geometry that translates without
// deforming. Returns false when the tag has no usable prev/cur pair, in which case the caller falls
// back to the camera delta alone exactly as before.
bool patch_billboard(uint64_t tag, const uint8_t* src, uint8_t* dst, uint32_t uniformSize,
                     uint32_t mtxPosOffset, uint32_t mtxNrmOffset, float alpha);

// texMtxCamMask: DrawData::texMtxCamMask — the texture matrices that live in EYE space and so must
// receive the same delta as the position matrices, or 0. Without it, a position-sourced texgen
// (SMS's water refraction) is drawn at the interpolated viewpoint while its UVs still map to the
// tick's, so the reflection sits in the wrong place for as long as the camera is moving.
void patch_camera_only(const uint8_t* src, uint8_t* dst, uint32_t uniformSize,
                       uint32_t mtxPosOffset, uint32_t mtxNrmOffset, uint32_t texMtxCamMask);

// Compute this tick's camera delta from the supplied view matrices. Called once per tick, before
// the per-draw patching. Returns false if there is no usable delta (no view supplied, or the view
// is not invertible), in which case patch_camera_only is inert.
bool begin_camera_delta(float alpha);

// Roll this tick's recorded matrices over to become the previous tick's. Called after the frame is
// recorded and patched.
void end_tick();

// This tick's index, so an event reported elsewhere can be lined up against the per-tick camera
// measurements in report(). Valid after begin_camera_delta.
long tick_index();

// Pairing statistics, so "interpolating" can be told apart from "silently snapping everything".
// Reports the share of tagged draws that found a partner, the mismatch count, and the two-number
// attribution that says whether a large inter-tick delta was the OBJECT moving or the CAMERA.
void report();

// Run the attribution discriminator against BOTH classes it is supposed to tell apart — a camera
// that moves while the object is static, and an object that moves while the camera is static — and
// return false if it fails either. Called once before the first real tick, because a discriminator
// that has only been reasoned about is not known to discriminate: this project has shipped one that
// scored backwards on both classes. Leaves no synthetic samples in the run's statistics.
bool selftest();

} // namespace aurora::gfx::interp
