#include "indexed_interp.hpp"

#include "interp.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <span>
#include <unordered_map>
#include <unordered_set>

#include "../internal.hpp"

namespace aurora::gfx::indexed_interp {
namespace {
Module Log("aurora::indexed_interp");

constexpr long kMaxGap = 4;

struct History {
  long stamp = -1;
  std::vector<float> positions;
};

struct HistoryKey {
  uint64_t tag = 0;
  uint64_t member = 0;

  bool operator==(const HistoryKey&) const = default;
};

struct HistoryKeyHash {
  size_t operator()(const HistoryKey& key) const {
    const size_t tagHash = std::hash<uint64_t>{}(key.tag);
    const size_t memberHash = std::hash<uint64_t>{}(key.member);
    return tagHash ^ (memberHash + 0x9e3779b97f4a7c15ULL + (tagHash << 6) + (tagHash >> 2));
  }
};

struct Sample {
  std::vector<uint8_t> bytes;
  std::vector<float> previous;
  std::vector<float> current;
  std::vector<uint64_t> keys;
  std::vector<long> gaps;
  std::vector<uint8_t> pairedGroups;
  uint16_t stride = 0;
  uint8_t population = 0;
  uint32_t verticesPerGroup = 0;
  uint32_t firstGroups = 0;
  uint32_t staleGroups = 0;
  uint32_t mismatchedGroups = 0;
  uint32_t pairedGroupCount = 0;
};

struct CaptureState {
  std::unordered_map<HistoryKey, History, HistoryKeyHash> history;
  std::unordered_map<uint32_t, Sample> samples;
  std::unordered_map<uint64_t, uint32_t> handlesThisTick;
  long captureTick = -1;
  uint32_t nextHandle = 1;
};

enum class CaptureKind : uint8_t { NewSample, Alias, Conflict };

struct CaptureResult {
  uint32_t handle = 0;
  CaptureKind kind = CaptureKind::Conflict;
};

CaptureState g_state;

long g_patched = 0;
long g_first = 0;
long g_tooStale = 0;
long g_mismatched = 0;
long g_aliases = 0;
long g_patchedGroups = 0;
long g_keyedSamples = 0;
long g_keyedGroups = 0;
long g_keyedPairedGroups = 0;
long g_unkeyedSamples = 0;
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
  for (size_t group = 0; group < sample.pairedGroups.size(); ++group) {
    if (sample.pairedGroups[group] == 0) {
      continue;
    }
    const float adjusted = 1.0f - (1.0f - alpha) / static_cast<float>(sample.gaps[group]);
    const uint32_t firstVertex = static_cast<uint32_t>(group) * sample.verticesPerGroup;
    for (uint32_t withinGroup = 0; withinGroup < sample.verticesPerGroup; ++withinGroup) {
      const uint32_t vertex = firstVertex + withinGroup;
      uint8_t* pos = out.data() + static_cast<size_t>(vertex) * sample.stride;
      for (uint32_t component = 0; component < 3; ++component) {
        const size_t index = static_cast<size_t>(vertex) * 3 + component;
        const float value = sample.previous[index] + (sample.current[index] - sample.previous[index]) * adjusted;
        write_be_f32(pos + component * 4, value);
      }
    }
  }
}

bool matches_input(const Sample& sample, const uint8_t* src, uint32_t byteSize, uint16_t stride, uint8_t population,
                   std::span<const uint64_t> quadKeys) {
  return sample.bytes.size() == byteSize && sample.stride == stride && sample.population == population &&
         std::ranges::equal(sample.keys, quadKeys) && std::memcmp(sample.bytes.data(), src, byteSize) == 0;
}

CaptureResult capture_sample(CaptureState& state, long tick, uint64_t tag, const uint8_t* src, uint32_t byteSize,
                             uint16_t stride, uint8_t population, std::span<const uint64_t> quadKeys) {
  if (state.captureTick != tick) {
    state.captureTick = tick;
    state.samples.clear();
    state.handlesThisTick.clear();
    state.nextHandle = 1;
  }

  if (const auto found = state.handlesThisTick.find(tag); found != state.handlesThisTick.end()) {
    const Sample& sample = state.samples.at(found->second);
    return CaptureResult{found->second, matches_input(sample, src, byteSize, stride, population, quadKeys)
                                            ? CaptureKind::Alias
                                            : CaptureKind::Conflict};
  }

  Sample sample;
  sample.bytes.assign(src, src + byteSize);
  sample.current = read_positions(src, byteSize, stride);
  sample.previous = sample.current;
  sample.keys.assign(quadKeys.begin(), quadKeys.end());
  sample.stride = stride;
  sample.population = population;
  sample.verticesPerGroup = quadKeys.empty() ? static_cast<uint32_t>(sample.current.size() / 3) : 4;
  const size_t groupCount = quadKeys.empty() ? 1 : quadKeys.size();
  sample.gaps.resize(groupCount, -1);
  sample.pairedGroups.resize(groupCount, 0);

  for (size_t group = 0; group < groupCount; ++group) {
    const uint64_t member = quadKeys.empty() ? 0 : quadKeys[group];
    History& history = state.history[HistoryKey{tag, member}];
    const long gap = history.stamp >= 0 ? tick - history.stamp : -1;
    sample.gaps[group] = gap;
    const size_t firstFloat = group * static_cast<size_t>(sample.verticesPerGroup) * 3;
    const size_t groupFloatCount = static_cast<size_t>(sample.verticesPerGroup) * 3;
    const bool paired = gap >= 1 && gap <= kMaxGap && history.positions.size() == groupFloatCount;
    if (paired) {
      std::copy(history.positions.begin(), history.positions.end(), sample.previous.begin() + firstFloat);
      sample.pairedGroups[group] = 1;
      ++sample.pairedGroupCount;
    } else if (gap < 0) {
      ++sample.firstGroups;
    } else if (gap > kMaxGap) {
      ++sample.staleGroups;
    } else {
      ++sample.mismatchedGroups;
    }
    history.stamp = tick;
    history.positions.assign(sample.current.begin() + firstFloat,
                             sample.current.begin() + firstFloat + groupFloatCount);
  }

  const uint32_t handle = state.nextHandle++;
  state.samples.emplace(handle, std::move(sample));
  state.handlesThisTick.emplace(tag, handle);
  return CaptureResult{handle, CaptureKind::NewSample};
}

void note_capture_outcome(const Sample& sample) {
  if (sample.keys.empty()) {
    ++g_unkeyedSamples;
  } else {
    ++g_keyedSamples;
    g_keyedGroups += sample.keys.size();
    g_keyedPairedGroups += sample.pairedGroupCount;
  }
  g_first += sample.firstGroups;
  g_tooStale += sample.staleGroups;
  g_mismatched += sample.mismatchedGroups;
}

bool is_birth_only(const Sample& sample) {
  return sample.pairedGroupCount == 0 && sample.firstGroups == sample.pairedGroups.size();
}

bool close(float a, float b) { return std::fabs(a - b) < 0.0001f; }
} // namespace

uint32_t capture(uint64_t tag, const uint8_t* src, uint32_t byteSize, uint16_t stride, uint8_t population,
                 std::span<const uint64_t> quadKeys) {
  if (tag == 0 || src == nullptr || byteSize == 0 || stride < 12 || byteSize % stride != 0) {
    return 0;
  }
  const uint32_t positionCount = byteSize / stride;
  ASSERT(quadKeys.empty() || positionCount == quadKeys.size() * 4,
         "indexed interpolation received {} quad keys for {} positions", quadKeys.size(), positionCount);
  if (!quadKeys.empty()) {
    const std::unordered_set<uint64_t> uniqueKeys(quadKeys.begin(), quadKeys.end());
    ASSERT(uniqueKeys.size() == quadKeys.size(), "indexed interpolation received duplicate stable quad keys");
    ASSERT(!uniqueKeys.contains(0), "indexed interpolation received a zero stable quad key");
  }
  static const bool controlPassed = selftest();
  ASSERT(controlPassed, "indexed interpolation control failed");

  // GX parsing precedes interpolate_recorded_frame(), whose camera setup advances the tick.
  const long tick = interp::tick_index() + 1;
  const CaptureResult result = capture_sample(g_state, tick, tag, src, byteSize, stride, population, quadKeys);
  if (result.kind == CaptureKind::Conflict) {
    Log.fatal(
        "indexed interpolation tag 0x{:016x} supplied different position-array bytes, layout, or "
        "population within tick {}; multiple render passes may alias one sample, but one object "
        "cannot have two poses in the same simulation tick",
        tag, tick);
  }
  const Sample& sample = g_state.samples.at(result.handle);
  if (result.kind == CaptureKind::Alias) {
    ++g_aliases;
  } else {
    note_capture_outcome(sample);
  }
  return result.handle;
}

bool patch(uint32_t handle, float alpha, bool resampling, std::vector<uint8_t>& out) {
  const auto it = g_state.samples.find(handle);
  if (it == g_state.samples.end() || it->second.pairedGroupCount == 0) {
    return false;
  }
  const Sample& sample = it->second;
  out = sample.bytes;
  apply_positions(sample, std::clamp(alpha, 0.0f, 1.0f), out);
  if (!resampling) {
    ++g_patched;
    g_patchedGroups += sample.pairedGroupCount;
    ++g_patchedByPopulation[sample.population];
  }
  return true;
}

bool birth_only(uint32_t handle) {
  const auto it = g_state.samples.find(handle);
  if (it == g_state.samples.end()) {
    return false;
  }
  const Sample& sample = it->second;
  return is_birth_only(sample);
}

bool reappearance_only(uint32_t handle) {
  const auto it = g_state.samples.find(handle);
  if (it == g_state.samples.end()) {
    return false;
  }
  const Sample& sample = it->second;
  return sample.pairedGroupCount == 0 && sample.staleGroups > 0 && sample.mismatchedGroups == 0;
}

bool selftest() {
  static const bool ok = [] {
    constexpr uint64_t kTag = 0x54444c5400000001ULL;
    CaptureState state;
    std::vector<uint8_t> previous(16, 0xa5);
    write_be_f32(previous.data(), 0.0f);
    write_be_f32(previous.data() + 4, 10.0f);
    write_be_f32(previous.data() + 8, 20.0f);
    const CaptureResult first = capture_sample(state, 1, kTag, previous.data(), previous.size(), 16, 7, {});
    const CaptureResult alias = capture_sample(state, 1, kTag, previous.data(), previous.size(), 16, 7, {});
    std::vector<uint8_t> conflict = previous;
    write_be_f32(conflict.data(), 1.0f);
    const CaptureResult rejected = capture_sample(state, 1, kTag, conflict.data(), conflict.size(), 16, 7, {});
    const bool firstClassifiedAsBirth = is_birth_only(state.samples.at(first.handle));

    std::vector<uint8_t> current(16, 0xa5);
    write_be_f32(current.data(), 20.0f);
    write_be_f32(current.data() + 4, 30.0f);
    write_be_f32(current.data() + 8, 40.0f);
    const CaptureResult second = capture_sample(state, 2, kTag, current.data(), current.size(), 16, 7, {});
    const Sample& sample = state.samples.at(second.handle);

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
    const bool aliasReused =
        first.kind == CaptureKind::NewSample && alias.kind == CaptureKind::Alias && first.handle == alias.handle;
    const bool conflictRefused = rejected.kind == CaptureKind::Conflict;
    const bool nextTickPaired = second.kind == CaptureKind::NewSample && sample.pairedGroupCount == 1 &&
                                sample.gaps.size() == 1 && sample.gaps[0] == 1;

    constexpr uint64_t kMembershipTag = 0x54444c5400000002ULL;
    constexpr uint64_t kA = 0x4101;
    constexpr uint64_t kB = 0x4202;
    constexpr uint64_t kC = 0x4303;
    const std::array<uint64_t, 2> keysAB{kA, kB};
    const std::array<uint64_t, 2> keysBC{kB, kC};
    std::vector<uint8_t> membersPrevious(8 * 16, 0);
    std::vector<uint8_t> membersCurrent(8 * 16, 0);
    for (uint32_t vertex = 0; vertex < 4; ++vertex) {
      write_be_f32(membersPrevious.data() + static_cast<size_t>(vertex) * 16, 10.0f);
      write_be_f32(membersPrevious.data() + static_cast<size_t>(vertex + 4) * 16, 20.0f);
      write_be_f32(membersCurrent.data() + static_cast<size_t>(vertex) * 16, 40.0f);
      write_be_f32(membersCurrent.data() + static_cast<size_t>(vertex + 4) * 16, 80.0f);
    }
    capture_sample(state, 10, kMembershipTag, membersPrevious.data(), membersPrevious.size(), 16, 7, keysAB);
    const CaptureResult membership =
        capture_sample(state, 11, kMembershipTag, membersCurrent.data(), membersCurrent.size(), 16, 7, keysBC);
    const Sample& membershipSample = state.samples.at(membership.handle);
    std::vector<uint8_t> membershipMidpoint = membershipSample.bytes;
    apply_positions(membershipSample, 0.5f, membershipMidpoint);
    const bool survivorFollowedIdentity = close(read_be_f32(membershipMidpoint.data()), 30.0f);
    const bool newbornStayedCurrent = close(read_be_f32(membershipMidpoint.data() + 4 * 16), 80.0f);
    const bool membershipChangedSafely = membershipSample.pairedGroupCount == 1 && membershipSample.firstGroups == 1 &&
                                         survivorFollowedIdentity && newbornStayedCurrent;
    const bool passed = midpointMoved && tailPreserved && samplesDiffer && aliasReused && conflictRefused &&
                        firstClassifiedAsBirth && nextTickPaired && membershipChangedSafely;
    if (!passed) {
      Log.error(
          "indexed interpolation control FAILED: midpointMoved={} tailPreserved={} "
          "samplesDiffer={} aliasReused={} conflictRefused={} firstClassifiedAsBirth={} nextTickPaired={} "
          "membershipChangedSafely={}",
          midpointMoved, tailPreserved, samplesDiffer, aliasReused, conflictRefused, firstClassifiedAsBirth,
          nextTickPaired, membershipChangedSafely);
    } else {
      Log.info(
          "indexed interpolation control passed: known 0->20 motion produced 5/10/15 at "
          "alpha .25/.5/.75, identical same-tick passes reused one sample, changed bytes were "
          "refused, and a surviving quad followed its stable identity across an A,B -> B,C membership change");
    }
    return passed;
  }();
  return ok;
}

void report() {
  Log.info(
      "indexed position interpolation: {} patched arrays / {} paired identity groups, {} first-sighting groups, "
      "{} stale groups, {} layout-mismatch groups, {} same-tick draw aliases{}",
      g_patched, g_patchedGroups, g_first, g_tooStale, g_mismatched, g_aliases,
      g_patched + g_first + g_tooStale + g_mismatched == 0
          ? " — NO MARKED INDEXED DRAW REACHED THE CAPTURE; this run does not verify the live seam"
          : "");
  Log.info(
      "  stable-key capture: {} of {} quad groups pairable across {} keyed array sample(s); {} unkeyed "
      "whole-array sample(s)",
      g_keyedPairedGroups, g_keyedGroups, g_keyedSamples, g_unkeyedSamples);
  for (int population = 0; population < 256; ++population) {
    if (g_patchedByPopulation[population] != 0) {
      Log.info("  population {}: {} indexed arrays paired", population, g_patchedByPopulation[population]);
    }
  }
}

} // namespace aurora::gfx::indexed_interp
