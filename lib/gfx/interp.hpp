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
                uint32_t uniformSize, uint32_t mtxPosOffset, uint32_t mtxNrmOffset, float alpha);

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
void patch_camera_only(uint8_t* dst, uint32_t uniformSize, uint32_t mtxPosOffset,
                       uint32_t mtxNrmOffset);

// Compute this tick's camera delta from the supplied view matrices. Called once per tick, before
// the per-draw patching. Returns false if there is no usable delta (no view supplied, or the view
// is not invertible), in which case patch_camera_only is inert.
bool begin_camera_delta(float alpha);

// Roll this tick's recorded matrices over to become the previous tick's. Called after the frame is
// recorded and patched.
void end_tick();

// Pairing statistics, so "interpolating" can be told apart from "silently snapping everything".
// Reports the share of tagged draws that found a partner, and the mismatch count.
void report();

} // namespace aurora::gfx::interp
