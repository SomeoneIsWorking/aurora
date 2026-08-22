#include "indexed_interp.hpp"

#include "interp.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "../internal.hpp"

namespace aurora::gfx::indexed_interp {
namespace {
Module Log("aurora::indexed_interp");

constexpr long kMaxGap = 4;

struct History {
  long stamp = -1;
  std::vector<float> positions;
};

struct Sample {
  std::vector<uint8_t> bytes;
  std::vector<float> previous;
  std::vector<float> current;
  uint16_t stride = 0;
  uint8_t population = 0;
  long gap = -1;
  bool paired = false;
};

std::unordered_map<uint64_t, History> g_history;
std::unordered_map<uint32_t, Sample> g_samples;
std::unordered_map<uint64_t, long> g_seenThisTick;
long g_captureTick = -1;
uint32_t g_nextHandle = 1;

long g_patched = 0;
long g_first = 0;
long g_tooStale = 0;
long g_mismatched = 0;
long g_patchedByPopulation[256] = {};

float read_be_f32(const uint8_t* src) {
  const uint32_t bits = static_cast<uint32_t>(src[0]) << 24 | static_cast<uint32_t>(src[1]) << 16 |
                        static_cast<uint32_t>(src[2]) << 8 | static_cast<uint32_t>(src[3]);
  return std::bit_cast<float>(bits);
}

void write_be_f32(uint8_t* dst, float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  dst[0] = static_cast<uint8_t>(bits >> 24);
  dst[1] = static_cast<uint8_t>(bits >> 16);
  dst[2] = static_cast<uint8_t>(bits >> 8);
  dst[3] = static_cast<uint8_t>(bits);
}

std::vector<float> read_positions(const uint8_t* src, uint32_t byteSize, uint16_t stride) {
  const uint32_t count = byteSize / stride;
  std::vector<float> out(static_cast<size_t>(count) * 3);
  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t* pos = src + static_cast<size_t>(i) * stride;
    for (uint32_t component = 0; component < 3; ++component) {
      out[static_cast<size_t>(i) * 3 + component] = read_be_f32(pos + component * 4);
    }
  }
  return out;
}

void apply_positions(const Sample& sample, float alpha, std::vector<uint8_t>& out) {
  const float adjusted = 1.0f - (1.0f - alpha) / static_cast<float>(sample.gap);
  const uint32_t count = static_cast<uint32_t>(sample.current.size() / 3);
  for (uint32_t i = 0; i < count; ++i) {
    uint8_t* pos = out.data() + static_cast<size_t>(i) * sample.stride;
    for (uint32_t component = 0; component < 3; ++component) {
      const size_t index = static_cast<size_t>(i) * 3 + component;
      const float value = sample.previous[index] + (sample.current[index] - sample.previous[index]) * adjusted;
      write_be_f32(pos + component * 4, value);
    }
  }
}

bool close(float a, float b) { return std::fabs(a - b) < 0.0001f; }
} // namespace

uint32_t capture(uint64_t tag, const uint8_t* src, uint32_t byteSize, uint16_t stride, uint8_t population) {
  if (tag == 0 || src == nullptr || byteSize == 0 || stride < 12 || byteSize % stride != 0) {
    return 0;
  }
  static const bool controlPassed = selftest();
  ASSERT(controlPassed, "indexed interpolation control failed");

  // GX parsing precedes interpolate_recorded_frame(), whose camera setup advances the tick.
  const long tick = interp::tick_index() + 1;
  if (g_captureTick != tick) {
    g_captureTick = tick;
    g_samples.clear();
    g_seenThisTick.clear();
    g_nextHandle = 1;
  }
  if (g_seenThisTick.contains(tag)) {
    Log.fatal(
        "indexed interpolation tag 0x{:016x} captured more than once in tick {}; the emitter "
        "contract is one array draw per tag, so pairing it would be ambiguous",
        tag, tick);
  }
  g_seenThisTick[tag] = tick;

  Sample sample;
  sample.bytes.assign(src, src + byteSize);
  sample.current = read_positions(src, byteSize, stride);
  sample.stride = stride;
  sample.population = population;

  History& history = g_history[tag];
  sample.gap = history.stamp >= 0 ? tick - history.stamp : -1;
  sample.paired = sample.gap >= 1 && sample.gap <= kMaxGap && history.positions.size() == sample.current.size();
  if (sample.paired) {
    sample.previous = history.positions;
  } else if (history.stamp < 0) {
    ++g_first;
  } else if (sample.gap > kMaxGap) {
    ++g_tooStale;
  } else {
    ++g_mismatched;
  }
  history.stamp = tick;
  history.positions = sample.current;

  const uint32_t handle = g_nextHandle++;
  g_samples.emplace(handle, std::move(sample));
  return handle;
}

bool patch(uint32_t handle, float alpha, bool resampling, std::vector<uint8_t>& out) {
  const auto it = g_samples.find(handle);
  if (it == g_samples.end() || !it->second.paired) {
    return false;
  }
  const Sample& sample = it->second;
  out = sample.bytes;
  apply_positions(sample, std::clamp(alpha, 0.0f, 1.0f), out);
  if (!resampling) {
    ++g_patched;
    ++g_patchedByPopulation[sample.population];
  }
  return true;
}

bool selftest() {
  static const bool ok = [] {
    Sample sample;
    sample.stride = 16;
    sample.gap = 1;
    sample.paired = true;
    sample.previous = {0.0f, 10.0f, 20.0f};
    sample.current = {20.0f, 30.0f, 40.0f};
    sample.bytes.resize(16, 0xa5);
    write_be_f32(sample.bytes.data(), 20.0f);
    write_be_f32(sample.bytes.data() + 4, 30.0f);
    write_be_f32(sample.bytes.data() + 8, 40.0f);

    std::vector<uint8_t> midpoint = sample.bytes;
    apply_positions(sample, 0.5f, midpoint);
    const bool midpointMoved = close(read_be_f32(midpoint.data()), 10.0f) &&
                               close(read_be_f32(midpoint.data() + 4), 20.0f) &&
                               close(read_be_f32(midpoint.data() + 8), 30.0f);
    const bool tailPreserved = std::memcmp(midpoint.data() + 12, sample.bytes.data() + 12, 4) == 0;

    std::vector<uint8_t> quarter = sample.bytes;
    std::vector<uint8_t> threeQuarter = sample.bytes;
    apply_positions(sample, 0.25f, quarter);
    apply_positions(sample, 0.75f, threeQuarter);
    const bool samplesDiffer =
        close(read_be_f32(quarter.data()), 5.0f) && close(read_be_f32(threeQuarter.data()), 15.0f);
    const bool passed = midpointMoved && tailPreserved && samplesDiffer;
    if (!passed) {
      Log.error(
          "indexed interpolation control FAILED: midpointMoved={} tailPreserved={} "
          "samplesDiffer={}",
          midpointMoved, tailPreserved, samplesDiffer);
    } else {
      Log.info(
          "indexed interpolation control passed: known 0->20 motion produced 5/10/15 at "
          "alpha .25/.5/.75 and preserved non-position bytes");
    }
    return passed;
  }();
  return ok;
}

void report() {
  Log.info(
      "indexed position interpolation: {} paired, {} first sightings, {} stale, {} size "
      "mismatches{}",
      g_patched, g_first, g_tooStale, g_mismatched,
      g_patched + g_first + g_tooStale + g_mismatched == 0
          ? " — NO MARKED INDEXED DRAW REACHED THE CAPTURE; this run does not verify the live seam"
          : "");
  for (int population = 0; population < 256; ++population) {
    if (g_patchedByPopulation[population] != 0) {
      Log.info("  population {}: {} indexed arrays paired", population, g_patchedByPopulation[population]);
    }
  }
}

} // namespace aurora::gfx::indexed_interp
