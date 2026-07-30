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
