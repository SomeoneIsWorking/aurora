#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace aurora::gfx::persistent_upload {

using WriteSink = std::function<void(uint64_t offset, const uint8_t* data, size_t length)>;

// Copies the producer-owned bytes and schedules the write through the render worker's FIFO. Sharing
// that FIFO with end-frame submission preserves older submit -> write -> current submit without a
// GPU wait. The sink is injectable so CPU-only tests exercise this exact production seam without
// Dawn.
void schedule(uint64_t offset, const uint8_t* data, size_t length, WriteSink sink);

} // namespace aurora::gfx::persistent_upload
