#include "gfx/texture_usage.hpp"

#include <gtest/gtest.h>

namespace aurora::gfx {
namespace {

constexpr auto kRenderTextureUsage =
    wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment;

TEST(TextureUsage, UnsupportedReadbackDoesNotWidenTheDescriptor) {
  EXPECT_EQ(with_texture_readback(kRenderTextureUsage, TextureReadback::Unsupported), kRenderTextureUsage);
  EXPECT_EQ(with_texture_readback(kRenderTextureUsage, TextureReadback::Unsupported) & wgpu::TextureUsage::CopySrc,
            wgpu::TextureUsage::None);
}

TEST(TextureUsage, SupportedReadbackAddsOnlyCopySource) {
  EXPECT_EQ(with_texture_readback(kRenderTextureUsage, TextureReadback::Supported),
            kRenderTextureUsage | wgpu::TextureUsage::CopySrc);
}

} // namespace
} // namespace aurora::gfx
