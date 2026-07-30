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
void patch_draw(uint64_t tag, uint32_t vtxCount, const uint8_t* src, uint8_t* dst,
                uint32_t uniformSize, uint32_t mtxPosOffset, uint32_t mtxNrmOffset, float alpha);

// Roll this tick's recorded matrices over to become the previous tick's. Called after the frame is
// recorded and patched.
void end_tick();

// Pairing statistics, so "interpolating" can be told apart from "silently snapping everything".
// Reports the share of tagged draws that found a partner, and the mismatch count.
void report();

} // namespace aurora::gfx::interp
