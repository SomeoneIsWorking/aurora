#pragma once

#include "../gfx/common.hpp"
#include "gx.hpp"

namespace aurora::gx {
struct DrawData {
  gfx::PipelineRef pipeline;
  gfx::Range vertRange;
  gfx::Range idxRange;
  gfx::Range uniformRange;
  uint32_t vtxCount;
  uint32_t indexCount;
  uint32_t instanceCount;
  GXBindGroups bindGroups;
  uint32_t dstAlpha;
  // Caller-supplied identity for this draw, from GX_AURORA_DRAW_TAG; 0 when untagged. Used to pair
  // a draw with the same object's draw in the previous tick for interpolation. Never derived from a
  // draw ordinal — see the sub-opcode's comment in GXAurora.h for why that cannot work.
  uint64_t tag;
  // Which game-side system emitted this draw (GX_AURORA_DRAW_POP). Not an identity — an audit
  // label, so the interpolation report can say WHICH populations interpolate rather than quoting
  // one global percentage that cannot separate a correctly-snapping HUD from stuttering geometry.
  uint8_t pop;
  // Present this draw EXACTLY on an interpolated frame (GX_AURORA_DRAW_EXACT): screen-space
  // geometry under a perspective projection, which the ortho test cannot see and the camera delta
  // must not touch.
  uint8_t exact;
  // Byte offset of pnMtx[0].pos within this draw's uniform block, and of pnMtx[0].nrm. Recorded at
  // build time because the layout is not fixed — an optional lineMode block shifts everything after
  // it by 16 bytes, and lineMode is not otherwise recoverable from a DrawData. Interpolation writes
  // exactly these two 480-byte spans; a wrong offset would write matrices over the projection.
  uint32_t mtxPosOffset;
  uint32_t mtxNrmOffset;
  // Whether this draw was issued under an ORTHOGRAPHIC projection. Interpolation needs it: the
  // camera delta may only be applied to draws whose matrix is model x view. A 2D pane's matrix is
  // not — applying a viewpoint change to it displaces the HUD bodily every other frame, which reads
  // as perfectly EVEN motion to a smoothness metric and so hides behind a good-looking score.
  uint8_t ortho;
  // Vertex layout, recorded so interpolation can find the POSITION floats inside each raw GC vertex
  // record without re-deriving the descriptor state, which is long gone by the time the recorded
  // frame is patched. posOffset is the byte offset of GX_VA_POS within a vertex. The vertex path
  // accepts DIRECT XYZ f32 and s16 positions; the latter needs its VAT fractional shift preserved
  // so an in-between pose is quantised back into the same GC representation the shader consumes.
  uint16_t vtxStride;
  uint16_t posOffset;
  uint8_t posF32XYZ;
  uint8_t posS16XYZ;
  uint8_t posFrac;
  // Which entries of the shader's `postex_mtx` array hold a TEXTURE matrix that must receive the
  // interpolated camera delta alongside the position matrices. Bit k = postex_mtx[k], so the bits
  // that can ever be set are MaxPnMtx..MaxPnMtx+MaxTexMtx-1 (the texture block; the low bits are
  // the position matrices, which patch_camera_only handles unconditionally).
  //
  // WHY THIS EXISTS. A GX texgen sourced from GX_TG_POS reads the RAW vertex attribute, not the
  // position after the position matrix. SMS's water refraction is built on that: the quad is
  // authored in EYE space, drawn with an identity PNMTX and a view-less projection texture matrix,
  // so its screen UV is `texmtx * eye_position`. Moving only the position matrix to the
  // interpolated viewpoint draws that quad in the right place while its UVs still map to the
  // previous viewpoint — the reflection sits in the WRONG PLACE, and only while the camera moves,
  // which is exactly how it was reported.
  //
  // The bit is set ONLY where composing the delta is provably the same reprojection as the one the
  // position matrix receives: see the gate in command_processor.cpp. Blanket-patching every
  // position-sourced texgen would corrupt object-locked projections, whose UVs are correct
  // unchanged when the camera moves.
  uint32_t texMtxCamMask;
  // Which position matrix this draw uses, needed to divide the model-view back out of a
  // position-sourced texture matrix. Only meaningful when the mask is non-zero, which the
  // record-time gate only allows for draws whose matrix index is NOT per-vertex.
  uint8_t pnMtxSlot;
};

constexpr uint32_t GXPipelineConfigVersion = 13;
struct PipelineConfig {
  uint32_t version = GXPipelineConfigVersion;
  uint32_t msaaSamples = 1;
  ShaderConfig shaderConfig;
  GXCompare depthFunc;
  GXCullMode cullMode;
  GXBlendMode blendMode;
  GXBlendFactor blendFacSrc, blendFacDst;
  GXLogicOp blendOp;
  uint32_t dstAlpha;
  uint32_t polygonOffsetBits;
  uint32_t polygonOffsetScaleBits;
  uint32_t polygonOffsetClampBits;
  bool depthCompare, depthUpdate, alphaUpdate, colorUpdate;
};
static_assert(std::has_unique_object_representations_v<PipelineConfig>);

wgpu::RenderPipeline create_pipeline([[maybe_unused]] const PipelineConfig& config);
void render(const DrawData& data, const wgpu::RenderPassEncoder& pass);

void queue_surface(const u8* dlStart, uint32_t dlSize, bool bigEndian) noexcept;
} // namespace aurora::gx
