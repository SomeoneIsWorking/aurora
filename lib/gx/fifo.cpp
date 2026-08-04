#include "fifo.hpp"
#include "command_processor.hpp"
#include "../internal.hpp"

#include <cstdlib>
#include <cstring>

namespace aurora::gx::fifo {
static Module Log("aurora::gx::fifo");

namespace detail {
uint8_t* sBufferData = nullptr;
uint32_t sBufferSize = 0;
uint32_t sBufferCapacity = 0;
bool sInDisplayList = false;
uint8_t* sDlBuffer = nullptr;
uint32_t sDlSize = 0;
uint32_t sDlWritePos = 0;
uint32_t sDrainDraws = 0;
uint32_t sDrainVerts = 0;
} // namespace detail

void init() {
  constexpr uint32_t initialCapacity = 64 * 1024;
  free(detail::sBufferData);
  detail::sBufferData = static_cast<uint8_t*>(malloc(initialCapacity));
  detail::sBufferSize = 0;
  detail::sBufferCapacity = initialCapacity;
  detail::sInDisplayList = false;
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
}

void write_data_grow(const void* data, uint32_t length) {
  uint32_t needed = detail::sBufferSize + length;
  uint32_t newCap = std::max(detail::sBufferCapacity * 2, needed);
  detail::sBufferData = static_cast<uint8_t*>(realloc(detail::sBufferData, newCap));
  std::memcpy(detail::sBufferData + detail::sBufferSize, data, length);
  detail::sBufferSize = needed;
  detail::sBufferCapacity = newCap;
}

void patch_u32(const uint32_t offset, const uint32_t val) {
  ASSERT(!detail::sInDisplayList && offset + sizeof(uint32_t) <= detail::sBufferSize,
         "fifo::patch_u32: invalid patch offset {} (buffer size {})", offset, detail::sBufferSize);
  const auto out = bswap(val);
  std::memcpy(detail::sBufferData + offset, &out, sizeof(out));
}

void begin_display_list(uint8_t* buf, uint32_t size) {
  detail::sInDisplayList = true;
  detail::sDlBuffer = buf;
  detail::sDlSize = size;
  detail::sDlWritePos = 0;
}

uint32_t end_display_list() {
  detail::sInDisplayList = false;
  uint32_t bytesWritten = detail::sDlWritePos;
  uint32_t padded = (bytesWritten + 31) & ~31u;
  while (detail::sDlWritePos < padded && detail::sDlWritePos < detail::sDlSize) {
    detail::sDlBuffer[detail::sDlWritePos++] = 0;
  }
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
  return padded;
}

bool in_display_list() { return detail::sInDisplayList; }

void drain() {
  if (detail::sBufferSize == 0) {
    return;
  }
  process(detail::sBufferData, detail::sBufferSize, true);
  // SB_DRAW_STATS=1: one line per drain (== one presented frame) with the
  // draw/vertex count — the cheap triage between "scene not drawn" (count
  // too low) and "drawn but invisible" (counts present; chase state).
  if (std::getenv("SB_DRAW_STATS") != nullptr) {
    static uint32_t s_frame = 0;
    std::fprintf(stderr, "[draw-stats] frame=%u bytes=%u draws=%u verts=%u\n",
                 s_frame++, detail::sBufferSize, detail::sDrainDraws, detail::sDrainVerts);
    std::fflush(stderr);
  }
  // SB_PROFILE_DRAWPRIM=1: report draw_prim's own share of the drain, and how much of THAT is the
  // per-vertex max-index scan. Printed per drain (one frame) so it lines up with SB_PROFILE_GFX.
  {
    extern int64_t g_dpTotalNs, g_dpScanNs;
    extern long g_dpCalls;
    if (g_dpCalls > 0) {
      std::fprintf(stderr, "[drawprim] calls=%ld total=%.2fms scan=%.2fms (%.0f%% of draw_prim)\n",
                   g_dpCalls, g_dpTotalNs / 1e6, g_dpScanNs / 1e6,
                   g_dpTotalNs ? 100.0 * (double)g_dpScanNs / (double)g_dpTotalNs : 0.0);
      std::fflush(stderr);
      g_dpTotalNs = g_dpScanNs = 0;
      g_dpCalls = 0;
    }
  }
  detail::sDrainDraws = 0;
  detail::sDrainVerts = 0;
  detail::sBufferSize = 0;
}

const uint8_t* get_buffer_data() { return detail::sBufferData; }
uint32_t get_buffer_size() { return detail::sBufferSize; }
void clear_buffer() { detail::sBufferSize = 0; }

} // namespace aurora::gx::fifo
