#pragma once

#include <webgpu/webgpu_cpp.h>

namespace aurora::gfx {

enum class TextureReadback {
  Unsupported,
  Supported,
};

constexpr wgpu::TextureUsage with_texture_readback(wgpu::TextureUsage usage, TextureReadback readback) noexcept {
  return readback == TextureReadback::Supported ? usage | wgpu::TextureUsage::CopySrc : usage;
}

} // namespace aurora::gfx
