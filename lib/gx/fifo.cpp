#include "fifo.hpp"
#include "command_processor.hpp"
#include "../internal.hpp"

#include <cstdlib>
#include <cstring>
#include <algorithm>

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
  // SB_PROFILE_DRAWPRIM=1: report draw_prim's own share of the drain, broken down by phase.
  // Printed per drain (one frame) so it lines up with SB_PROFILE_GFX.
  {
    extern int64_t g_dpTotalNs;
    extern long g_dpCalls, g_dpVerts[8], g_dpVertTotal, g_dpPrimKind[8];
    extern uint64_t g_dpPhase[8], g_dpWholeTicks;
    extern long g_dpPhaseCalls[8];
    extern long g_dpMergedCalls, g_dpUnmergedCalls, g_dpEarlyReturns;
    extern long g_dpUnmergedSampleCount, g_dpUnmergedSampleDropped;
    extern double sb_dp_ns_per_tick_pub();
    extern uint64_t sb_dp_probe_cost_ticks_pub();
    extern int sb_dp_probes_per_call_pub();
    if (g_dpCalls > 0) {
      std::fprintf(stderr, "[drawprim] calls=%ld total=%.2fms\n", g_dpCalls, g_dpTotalNs / 1e6);
      std::fprintf(stderr,
                   "[drawprim] verts/prim: 1-2:%ld 3:%ld 4:%ld 5-6:%ld 7-12:%ld 13-24:%ld "
                   "25-48:%ld 49+:%ld | total verts=%ld mean=%.1f\n",
                   g_dpVerts[0], g_dpVerts[1], g_dpVerts[2], g_dpVerts[3], g_dpVerts[4],
                   g_dpVerts[5], g_dpVerts[6], g_dpVerts[7], g_dpVertTotal,
                   (double)g_dpVertTotal / (double)g_dpCalls);

      // Phase breakdown. The phases partition the body, so anything they do not claim is a region
      // with no probe on some path — printed as `unattributed` rather than folded into a neighbour.
      static const char* kPhaseName[8] = {"prologue",   "diag-pre",  "attr-enum", "idx-scan",
                                          "diag-post",  "push-verts", "merge-idx", "unmerged"};
      const double nsPerTick = sb_dp_ns_per_tick_pub();
      uint64_t sum = 0;
      for (int i = 0; i < 8; ++i) {
        sum += g_dpPhase[i];
      }
      const double wholeNs = (double)g_dpWholeTicks * nsPerTick;
      for (int i = 0; i < 8; ++i) {
        const double ns = (double)g_dpPhase[i] * nsPerTick;
        // ns/call is over the calls that ENTERED this phase, and the entry count is printed
        // beside it: a phase that runs on 3% of calls has a per-call cost 35x its share of the
        // average, and averaging over all calls hides exactly the expensive-but-rare case.
        const long n = g_dpPhaseCalls[i];
        std::fprintf(stderr, "[drawprim]   %-10s %7.3fms  %5.1f%%  %8.1f ns/call  n=%ld (%.0f%% of calls)\n",
                     kPhaseName[i], ns / 1e6, wholeNs > 0 ? 100.0 * ns / wholeNs : 0.0,
                     n > 0 ? ns / (double)n : 0.0, n,
                     g_dpCalls > 0 ? 100.0 * (double)n / (double)g_dpCalls : 0.0);
      }
      const double unattrNs = wholeNs - (double)sum * nsPerTick;
      std::fprintf(stderr, "[drawprim]   %-10s %7.3fms  %5.1f%%   <- CONTROL: regions no phase claimed\n",
                   "unattr", unattrNs / 1e6, wholeNs > 0 ? 100.0 * unattrNs / wholeNs : 0.0);

      // CONTROL: what the probes themselves cost. A phase split whose probe overhead is comparable
      // to the phases is measuring itself, exactly as the clock_gettime version did. Say so rather
      // than let the percentages be read as fact.
      const double probeNs = (double)sb_dp_probe_cost_ticks_pub() * nsPerTick;
      const double probeTotalNs = probeNs * (double)sb_dp_probes_per_call_pub() * (double)g_dpCalls;
      const double probePct = wholeNs > 0 ? 100.0 * probeTotalNs / wholeNs : 0.0;
      std::fprintf(stderr,
                   "[drawprim]   probe cost %.1f ns x %d probes/call = %.1f%% of the measured body%s\n",
                   probeNs, sb_dp_probes_per_call_pub(), probePct,
                   probePct > 25.0 ? "  <- TOO HIGH: phase split is NOT admissible" : "");
      std::fprintf(stderr, "[drawprim]   paths: merged=%ld unmerged=%ld early-return=%ld (sum must equal calls=%ld)\n",
                   g_dpMergedCalls, g_dpUnmergedCalls, g_dpEarlyReturns, g_dpCalls);

      // Distribution of the unmerged (per-draw) cost. See the declaration: the mean alone cannot
      // tell a uniformly expensive build path from a few pipeline compiles.
      if (g_dpUnmergedSampleCount > 0) {
        extern uint32_t g_dpUnmergedSamples[4096];
        std::sort(g_dpUnmergedSamples, g_dpUnmergedSamples + g_dpUnmergedSampleCount);
        const auto pct = [&](double p) {
          long i = (long)(p * (double)(g_dpUnmergedSampleCount - 1));
          return (double)g_dpUnmergedSamples[i] * nsPerTick;
        };
        std::fprintf(stderr,
                     "[drawprim]   unmerged ns: p50=%.0f p90=%.0f p99=%.0f max=%.0f  (n=%ld%s)\n",
                     pct(0.50), pct(0.90), pct(0.99),
                     (double)g_dpUnmergedSamples[g_dpUnmergedSampleCount - 1] * nsPerTick,
                     g_dpUnmergedSampleCount,
                     g_dpUnmergedSampleDropped > 0 ? " TRUNCATED - distribution incomplete" : "");
      }
      g_dpUnmergedSampleCount = 0;
      g_dpUnmergedSampleDropped = 0;
      std::fflush(stderr);
      g_dpTotalNs = 0;
      g_dpCalls = 0;
      g_dpVertTotal = 0;
      g_dpWholeTicks = 0;
      g_dpMergedCalls = g_dpUnmergedCalls = g_dpEarlyReturns = 0;
      for (int i = 0; i < 8; ++i) { g_dpPhase[i] = 0; g_dpPhaseCalls[i] = 0; }
      for (int i = 0; i < 8; ++i) { g_dpVerts[i] = 0; g_dpPrimKind[i] = 0; }
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
