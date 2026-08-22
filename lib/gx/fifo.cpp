#include "fifo.hpp"
#include "command_processor.hpp"
#include "../internal.hpp"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <lucent/log.h>

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

// --- Draw-pass idempotence probe (SB_DOUBLE_DRAW in MarDirectorDirect.cpp) -------------------
//
// Game-native 60fps interpolation renders the draw phases twice per logic tick, so those passes
// must emit the SAME commands both times — anything that differs is state the pass mutated as it
// ran. These let the game mark the fifo, run a pass, hash what it emitted, REWIND, and run it
// again, so the second pass replaces the first rather than adding to it (no doubled geometry, no
// staging overflow, and the frame still renders normally from the final pass).
extern "C" uint32_t sb_gx_fifo_mark(void) { return detail::sBufferSize; }
extern "C" void sb_gx_fifo_rewind(uint32_t mark) {
  // Only ever shrinks. Growing here would expose uninitialised bytes to the parser.
  if (mark <= detail::sBufferSize) {
    detail::sBufferSize = mark;
  }
}
// Snapshot / compare, to LOCALISE a divergence rather than just detect one. A byte count says
// two passes differ; the first differing offset says where, which is what identifies the culprit
// command in the stream.
static uint8_t* sSnap = nullptr;
static uint32_t sSnapSize = 0, sSnapCap = 0;
extern "C" void sb_gx_fifo_snapshot(uint32_t from, uint32_t to) {
  sSnapSize = (detail::sBufferData != nullptr && to > from) ? (to - from) : 0;
  if (sSnapSize > sSnapCap) {
    sSnap = static_cast<uint8_t*>(realloc(sSnap, sSnapSize));
    sSnapCap = sSnapSize;
  }
  if (sSnapSize > 0) {
    std::memcpy(sSnap, detail::sBufferData + from, sSnapSize);
  }
}
// Returns the first byte offset (relative to `from`) at which the current fifo contents differ
// from the snapshot, or -1 if the common prefix matches AND the lengths match. If the lengths
// differ but the common prefix is identical, returns the common length — the divergence is "one
// stream kept going", which is a different fault from "a byte changed".
extern "C" long sb_gx_fifo_compare(uint32_t from, uint32_t to) {
  const uint32_t cur = (to > from) ? (to - from) : 0;
  const uint32_t n = cur < sSnapSize ? cur : sSnapSize;
  for (uint32_t i = 0; i < n; ++i) {
    if (detail::sBufferData[from + i] != sSnap[i]) {
      return static_cast<long>(i);
    }
  }
  return (cur == sSnapSize) ? -1L : static_cast<long>(n);
}
// Hex of the first `n` bytes of the snapshot (pass A) and of the live fifo (pass B), so a
// divergence at offset 0 can be READ rather than guessed at. GX commands are opcode-led, so the
// first bytes name the command that differs.
extern "C" void sb_gx_fifo_dump_heads(uint32_t from, uint32_t n) {
  char a[160] = {0}, b[160] = {0};
  const uint32_t na = n < sSnapSize ? n : sSnapSize;
  for (uint32_t i = 0; i < na && i * 3 + 3 < sizeof(a); ++i) {
    std::snprintf(a + i * 3, 4, "%02x ", sSnap[i]);
  }
  const uint32_t avail = detail::sBufferSize > from ? detail::sBufferSize - from : 0;
  const uint32_t nb = n < avail ? n : avail;
  for (uint32_t i = 0; i < nb && i * 3 + 3 < sizeof(b); ++i) {
    std::snprintf(b + i * 3, 4, "%02x ", detail::sBufferData[from + i]);
  }
  std::fprintf(stderr, "[dbl-draw]   passA head: %s\n[dbl-draw]   passB head: %s\n", a, b);
}
extern "C" uint64_t sb_gx_fifo_hash(uint32_t from, uint32_t to) {
  if (detail::sBufferData == nullptr || to <= from || to > detail::sBufferSize) {
    return 0; // caller checks the byte count separately; 0 here means "nothing to hash"
  }
  // FNV-1a; no dependency on aurora's xxhash from this TU.
  uint64_t h = 1469598103934665603ull;
  for (uint32_t i = from; i < to; ++i) {
    h ^= detail::sBufferData[i];
    h *= 1099511628211ull;
  }
  return h;
}

void drain() {
  if (detail::sBufferSize != 0) {
    process(detail::sBufferData, detail::sBufferSize, true);
  }
  // drain() is also the end-of-frame accounting boundary for command streams processed
  // synchronously through aurora_fifo_replay(). The recomp runtime uses that path, so its live
  // FIFO is intentionally empty here while the draw counters are not. Returning on an empty live
  // buffer used to suppress the report and carry replay counters across frames.
  // SB_DRAW_STATS=1: one line per drain (== one presented frame) with the
  // draw/vertex count — the cheap triage between "scene not drawn" (count
  // too low) and "drawn but invisible" (counts present; chase state).
  if (std::getenv("SB_DRAW_STATS") != nullptr) {
    static uint32_t s_frame = 0;
    std::fprintf(stderr, "[draw-stats] frame=%u bytes=%u draws=%u verts=%u\n", s_frame++, detail::sBufferSize,
                 detail::sDrainDraws, detail::sDrainVerts);
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
                   g_dpVerts[0], g_dpVerts[1], g_dpVerts[2], g_dpVerts[3], g_dpVerts[4], g_dpVerts[5], g_dpVerts[6],
                   g_dpVerts[7], g_dpVertTotal, (double)g_dpVertTotal / (double)g_dpCalls);

      // Phase breakdown. The phases partition the body, so anything they do not claim is a region
      // with no probe on some path — printed as `unattributed` rather than folded into a neighbour.
      static const char* kPhaseName[8] = {"prologue",  "diag-pre",   "attr-enum", "idx-scan",
                                          "diag-post", "push-verts", "merge-idx", "unmerged"};
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
                     kPhaseName[i], ns / 1e6, wholeNs > 0 ? 100.0 * ns / wholeNs : 0.0, n > 0 ? ns / (double)n : 0.0, n,
                     g_dpCalls > 0 ? 100.0 * (double)n / (double)g_dpCalls : 0.0);
      }
      const double unattrNs = wholeNs - (double)sum * nsPerTick;
      std::fprintf(stderr, "[drawprim]   %-10s %7.3fms  %5.1f%%   <- CONTROL: regions no phase claimed\n", "unattr",
                   unattrNs / 1e6, wholeNs > 0 ? 100.0 * unattrNs / wholeNs : 0.0);

      // CONTROL: what the probes themselves cost. A phase split whose probe overhead is comparable
      // to the phases is measuring itself, exactly as the clock_gettime version did. Say so rather
      // than let the percentages be read as fact.
      const double probeNs = (double)sb_dp_probe_cost_ticks_pub() * nsPerTick;
      const double probeTotalNs = probeNs * (double)sb_dp_probes_per_call_pub() * (double)g_dpCalls;
      const double probePct = wholeNs > 0 ? 100.0 * probeTotalNs / wholeNs : 0.0;
      std::fprintf(stderr, "[drawprim]   probe cost %.1f ns x %d probes/call = %.1f%% of the measured body%s\n",
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
        std::fprintf(stderr, "[drawprim]   unmerged ns: p50=%.0f p90=%.0f p99=%.0f max=%.0f  (n=%ld%s)\n", pct(0.50),
                     pct(0.90), pct(0.99), (double)g_dpUnmergedSamples[g_dpUnmergedSampleCount - 1] * nsPerTick,
                     g_dpUnmergedSampleCount,
                     g_dpUnmergedSampleDropped > 0 ? " TRUNCATED - distribution incomplete" : "");
      }
      // Sub-measurement: the two snprintf calls at the head of push_gx_draw, which run on EVERY
      // draw to build a description used only by the staging-overflow fatal message.
      {
        extern uint64_t g_dpDescTicks;
        const double ns = (double)g_dpDescTicks * nsPerTick;
        std::fprintf(stderr, "[drawprim]   of which draw-desc snprintf: %.3fms  %.1f%% of unmerged  %.0f ns/draw\n",
                     ns / 1e6, g_dpPhase[7] > 0 ? 100.0 * ns / ((double)g_dpPhase[7] * nsPerTick) : 0.0,
                     g_dpUnmergedCalls > 0 ? ns / (double)g_dpUnmergedCalls : 0.0);
        g_dpDescTicks = 0;
      }
      // Indexed-array storage upload volume. SB_PROFILE_GFX puts arrayUpload at ~63% of the
      // per-draw build; this says whether that is distinct geometry or the same arrays re-pushed.
      {
        extern uint64_t g_arrUploadCount, g_arrUploadBytes, g_arrUploadDistinctBytes, g_arrCachedHits;
        extern std::unordered_set<uint64_t> g_arrUploadDistinct;
        extern std::unordered_map<uint64_t, uint64_t> g_arrUploadHash;
        extern uint64_t g_arrContentChanged, g_arrDataCacheHits;
        std::fprintf(stderr,
                     "[drawprim]   arrays: uploads=%llu (%.2f MB)  distinct=%zu (%.2f MB)  "
                     "redundancy=%.1fx  cache-hits=%llu\n",
                     (unsigned long long)g_arrUploadCount, (double)g_arrUploadBytes / 1048576.0,
                     g_arrUploadDistinct.size(), (double)g_arrUploadDistinctBytes / 1048576.0,
                     g_arrUploadDistinctBytes > 0 ? (double)g_arrUploadBytes / (double)g_arrUploadDistinctBytes : 0.0,
                     (unsigned long long)g_arrCachedHits);
        std::fprintf(stderr, "[drawprim]   arrays: data-keyed cache hits (uploads avoided)=%llu\n",
                     (unsigned long long)g_arrDataCacheHits);
        g_arrDataCacheHits = 0;
        {
          extern uint64_t g_arrPersistUploads, g_arrPersistHits, g_arrArenaFull;
          extern uint64_t g_arrPersistUploadBytes, g_arrPersistHitBytes;
          // fifo.cpp does not pull in the gfx headers; forward-declare what the report needs.
          extern uint64_t sb_arena_used();
          extern size_t sb_arena_entries();

          std::fprintf(stderr,
                       "[drawprim]   arena: reused=%llu (%.2f MB, NOT uploaded)  uploaded=%llu (%.2f MB)  "
                       "full-fallbacks=%llu  arena used=%.2f MB in %zu entries\n",
                       (unsigned long long)g_arrPersistHits, (double)g_arrPersistHitBytes / 1048576.0,
                       (unsigned long long)g_arrPersistUploads, (double)g_arrPersistUploadBytes / 1048576.0,
                       (unsigned long long)g_arrArenaFull, (double)sb_arena_used() / 1048576.0, sb_arena_entries());
          g_arrPersistUploads = g_arrPersistHits = g_arrArenaFull = 0;
          g_arrPersistUploadBytes = g_arrPersistHitBytes = 0;
        }
        // The precondition for caching uploads by DATA identity rather than by slot registration.
        std::fprintf(stderr, "[drawprim]   arrays: in-frame content changes under an unchanged (ptr,size): %llu%s\n",
                     (unsigned long long)g_arrContentChanged,
                     g_arrContentChanged == 0 ? "  <- a data-keyed upload cache is SAFE"
                                              : "  <- a data-keyed upload cache would serve STALE data");
        // Ceiling on what a persistent (cross-frame) geometry buffer could remove.
        {
          extern std::unordered_map<uint64_t, uint64_t> g_arrHashPrevFrame;
          extern uint64_t g_arrSameAsPrevBytes, g_arrChangedVsPrevBytes, g_arrNewVsPrevBytes;
          const double tot = (double)(g_arrSameAsPrevBytes + g_arrChangedVsPrevBytes + g_arrNewVsPrevBytes);
          std::fprintf(
              stderr,
              "[drawprim]   arrays vs PREVIOUS frame: unchanged=%.2f MB (%.0f%%)  changed=%.2f MB  new=%.2f MB\n",
              (double)g_arrSameAsPrevBytes / 1048576.0, tot > 0 ? 100.0 * (double)g_arrSameAsPrevBytes / tot : 0.0,
              (double)g_arrChangedVsPrevBytes / 1048576.0, (double)g_arrNewVsPrevBytes / 1048576.0);
          extern std::unordered_map<uint64_t, uint32_t> g_arrOffsetPrevFrame, g_arrOffsetThisFrame;
          extern uint64_t g_arrOffsetStable, g_arrOffsetMoved;
          std::fprintf(stderr, "[drawprim]   arrays: storage offset vs prev frame: stable=%llu moved=%llu%s\n",
                       (unsigned long long)g_arrOffsetStable, (unsigned long long)g_arrOffsetMoved,
                       g_arrOffsetMoved == 0 ? "  <- deterministic; the GPU buffer already holds these bytes"
                                             : "  <- offsets move; skipping the copy needs a persistent allocator");
          g_arrOffsetPrevFrame = g_arrOffsetThisFrame;
          g_arrOffsetThisFrame.clear();
          g_arrOffsetStable = g_arrOffsetMoved = 0;
          g_arrHashPrevFrame = g_arrUploadHash; // carry this frame's hashes into the next
          g_arrSameAsPrevBytes = g_arrChangedVsPrevBytes = g_arrNewVsPrevBytes = 0;
        }
        g_arrContentChanged = 0;
        g_arrUploadHash.clear();
        g_arrUploadCount = g_arrUploadBytes = g_arrUploadDistinctBytes = g_arrCachedHits = 0;
        g_arrUploadDistinct.clear();
      }
      g_dpUnmergedSampleCount = 0;
      g_dpUnmergedSampleDropped = 0;
      std::fflush(stderr);
      g_dpTotalNs = 0;
      g_dpCalls = 0;
      g_dpVertTotal = 0;
      g_dpWholeTicks = 0;
      g_dpMergedCalls = g_dpUnmergedCalls = g_dpEarlyReturns = 0;
      for (int i = 0; i < 8; ++i) {
        g_dpPhase[i] = 0;
        g_dpPhaseCalls[i] = 0;
      }
      for (int i = 0; i < 8; ++i) {
        g_dpVerts[i] = 0;
        g_dpPrimKind[i] = 0;
      }
    }
  }
  // The staging-overflow fatal in gfx/common.hpp names the runaway draw by calling
  // aurora_gfx_last_draw_desc(). That path is (correctly) almost never taken, so the recorder and
  // formatter behind it would otherwise go unexercised — and it was rewritten to defer formatting
  // off the per-draw hot path. This channel prints the SAME function's output once per frame, so
  // the text the fatal would show is verifiable on real data without provoking an overflow.
  {
    static const lucent::Channel chDrawDesc{"drawdesc"};
    if (chDrawDesc) {
      // Same namespace (aurora::gx::fifo) and the same function the fatal reaches through
      // aurora_gfx_last_draw_desc().
      extern const char* sb_last_draw_desc();
      lucent::debug(chDrawDesc, "last draws (as the overflow fatal would print them):{}", sb_last_draw_desc());
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
