#include "fifo.hpp"
#include "command_processor.hpp"
#include "../internal.hpp"

#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include <lucent/log.h>

namespace aurora::gx::fifo {
static Module Log("aurora::gx::fifo");
extern "C" void sbr_gxfifo_work_report() __attribute__((weak));

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
  // SB_DRAW_STATS=1 reports deterministic units of work at the drain boundary. These counters
  // describe game/FIFO inputs and branch populations, so CPU/GPU contention cannot change their
  // meaning. Every counter is reset at this same boundary, including empty frames.
  static const bool drawStats = [] {
    const char* value = std::getenv("SB_DRAW_STATS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  if (drawStats) {
    static uint32_t s_frame = 0;
    extern uint64_t g_autoScanDraws, g_autoScanVertices, g_autoScanFieldVisits, g_autoScanIndexBytes;
    extern uint64_t g_autoLayoutFieldsChecked;
    extern uint64_t g_fifoProcessCalls, g_fifoInputBytes;
    extern long g_dpCalls, g_dpVerts[8], g_dpVertTotal, g_dpPrimKind[8];
    extern long g_dpMergedCalls, g_dpUnmergedCalls, g_dpEarlyReturns;
    extern uint64_t g_arrUploadCount, g_arrUploadBytes, g_arrUploadDistinctBytes, g_arrCachedHits;
    extern std::unordered_set<uint64_t> g_arrUploadDistinct;
    extern std::unordered_map<uint64_t, uint64_t> g_arrUploadHash;
    extern uint64_t g_arrContentChanged, g_arrDataCacheHits;
    extern uint64_t g_arrPersistUploads, g_arrPersistHits, g_arrArenaFull;
    extern uint64_t g_arrPersistUploadBytes, g_arrPersistHitBytes;
    extern uint64_t sb_arena_used();
    extern size_t sb_arena_entries();
    extern std::unordered_map<uint64_t, uint64_t> g_arrHashPrevFrame;
    extern uint64_t g_arrSameAsPrevBytes, g_arrChangedVsPrevBytes, g_arrNewVsPrevBytes;
    extern std::unordered_map<uint64_t, uint32_t> g_arrOffsetPrevFrame, g_arrOffsetThisFrame;
    extern uint64_t g_arrOffsetStable, g_arrOffsetMoved;

    const uint32_t frame = s_frame++;
    const long classifiedPaths = g_dpMergedCalls + g_dpUnmergedCalls + g_dpEarlyReturns;
    const long unclassifiedPaths = g_dpCalls - classifiedPaths;
    const uint64_t crossFrameBytes = g_arrSameAsPrevBytes + g_arrChangedVsPrevBytes + g_arrNewVsPrevBytes;
    Log.info("[draw-stats] frame={} fifo_calls={} bytes={} draws={} verts={}", frame, g_fifoProcessCalls,
             g_fifoInputBytes, detail::sDrainDraws, detail::sDrainVerts);
    Log.info(
        "[draw-work] frame={} auto_scan_draws={} vertices={} field_visits={} index_bytes={} "
        "layout_fields={} prims={} prim_verts={} size_buckets={},{},{},{},{},{},{},{} "
        "kinds={},{},{},{},{},{},{},{} paths={},{},{} classified={} unclassified={}",
        frame, g_autoScanDraws, g_autoScanVertices, g_autoScanFieldVisits, g_autoScanIndexBytes,
        g_autoLayoutFieldsChecked, g_dpCalls, g_dpVertTotal, g_dpVerts[0], g_dpVerts[1], g_dpVerts[2], g_dpVerts[3],
        g_dpVerts[4], g_dpVerts[5], g_dpVerts[6], g_dpVerts[7], g_dpPrimKind[0], g_dpPrimKind[1], g_dpPrimKind[2],
        g_dpPrimKind[3], g_dpPrimKind[4], g_dpPrimKind[5], g_dpPrimKind[6], g_dpPrimKind[7], g_dpMergedCalls,
        g_dpUnmergedCalls, g_dpEarlyReturns, classifiedPaths, unclassifiedPaths);
    Log.info(
        "[draw-storage] frame={} uploads={} upload_bytes={} distinct={} distinct_bytes={} cached_hits={} "
        "data_cache_hits={} arena_reused={} arena_reused_bytes={} arena_uploaded={} arena_uploaded_bytes={} "
        "arena_full={} arena_used={} arena_entries={} content_changes={} prev_same_bytes={} prev_changed_bytes={} "
        "prev_new_bytes={} prev_total_bytes={} offsets_stable={} offsets_moved={}",
        frame, g_arrUploadCount, g_arrUploadBytes, g_arrUploadDistinct.size(), g_arrUploadDistinctBytes,
        g_arrCachedHits, g_arrDataCacheHits, g_arrPersistHits, g_arrPersistHitBytes, g_arrPersistUploads,
        g_arrPersistUploadBytes, g_arrArenaFull, sb_arena_used(), sb_arena_entries(), g_arrContentChanged,
        g_arrSameAsPrevBytes, g_arrChangedVsPrevBytes, g_arrNewVsPrevBytes, crossFrameBytes, g_arrOffsetStable,
        g_arrOffsetMoved);
    if (sbr_gxfifo_work_report != nullptr) {
      sbr_gxfifo_work_report();
    }

    g_arrOffsetPrevFrame = g_arrOffsetThisFrame;
    g_arrOffsetThisFrame.clear();
    g_arrHashPrevFrame = g_arrUploadHash;
    g_arrUploadHash.clear();
    g_arrUploadDistinct.clear();

    g_autoScanDraws = g_autoScanVertices = g_autoScanFieldVisits = g_autoScanIndexBytes = 0;
    g_autoLayoutFieldsChecked = 0;
    g_fifoProcessCalls = g_fifoInputBytes = 0;
    g_dpCalls = 0;
    g_dpVertTotal = 0;
    g_dpMergedCalls = g_dpUnmergedCalls = g_dpEarlyReturns = 0;
    for (int i = 0; i < 8; ++i) {
      g_dpVerts[i] = 0;
      g_dpPrimKind[i] = 0;
    }
    g_arrUploadCount = g_arrUploadBytes = g_arrUploadDistinctBytes = g_arrCachedHits = 0;
    g_arrDataCacheHits = g_arrContentChanged = 0;
    g_arrPersistUploads = g_arrPersistHits = g_arrArenaFull = 0;
    g_arrPersistUploadBytes = g_arrPersistHitBytes = 0;
    g_arrSameAsPrevBytes = g_arrChangedVsPrevBytes = g_arrNewVsPrevBytes = 0;
    g_arrOffsetStable = g_arrOffsetMoved = 0;
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
