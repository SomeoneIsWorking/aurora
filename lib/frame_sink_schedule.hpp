#pragma once

#include <aurora/aurora.h>

#include <algorithm>

namespace aurora {

// Calling-thread cadence for the asynchronous frame sink. Selection happens before the packet is
// handed to the render worker, so both the caller and the eventual MapAsync job can name the same
// packet without consulting mutable callback-time state.
class FrameSinkSchedule {
public:
  void configure(int everyNFrames) noexcept {
    every_ = std::max(0, everyNFrames);
    remaining_ = 0;
  }

  [[nodiscard]] bool will_capture() const noexcept { return every_ > 0 && remaining_ == 0; }

  [[nodiscard]] bool consume() noexcept {
    if (every_ <= 0) {
      return false;
    }
    if (remaining_ == 0) {
      remaining_ = every_ - 1;
      return true;
    }
    --remaining_;
    return false;
  }

private:
  int every_ = 0;
  int remaining_ = 0;
};

inline AuroraFrameSinkInfo make_frame_sink_info(const AuroraGpuSubmitInfo& submit) noexcept {
  return {
      .structSize = sizeof(AuroraFrameSinkInfo),
      .version = AURORA_FRAME_SINK_INFO_VERSION,
      .frameId = submit.frameId,
      .replaySourceFrameId = submit.replaySourceFrameId,
      .replayEmission = submit.replayEmission,
      .reserved = 0,
  };
}

} // namespace aurora
