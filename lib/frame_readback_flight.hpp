#pragma once

#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace aurora {

struct FrameReadbackFlightSnapshot {
  uint32_t mapsPending;
  uint32_t mapsCompleted;
  uint32_t mapsFailed;
};

// Tracks the part of asynchronous framebuffer readback that remains live after a queue submit.
// MapAsync callbacks may run on a Dawn callback thread, so the counters must not rely on the render
// worker's serialization.
class FrameReadbackFlight {
public:
  class Callback {
  public:
    explicit Callback(FrameReadbackFlight& owner) noexcept : m_owner(owner) {}
    Callback(const Callback&) = delete;
    Callback& operator=(const Callback&) = delete;
    ~Callback() { m_owner.map_finished(m_mapSucceeded); }

    void mark_map_success() noexcept { m_mapSucceeded = true; }

  private:
    FrameReadbackFlight& m_owner;
    bool m_mapSucceeded = false;
  };

  void map_requested() noexcept {
    std::lock_guard lock{m_mutex};
    ++m_mapsPending;
  }

  [[nodiscard]] Callback callback_started() noexcept { return Callback{*this}; }

  [[nodiscard]] FrameReadbackFlightSnapshot snapshot() const noexcept {
    std::lock_guard lock{m_mutex};
    return {
        .mapsPending = m_mapsPending,
        .mapsCompleted = m_mapsCompleted,
        .mapsFailed = m_mapsFailed,
    };
  }

private:
  void map_finished(bool success) noexcept {
    std::lock_guard lock{m_mutex};
    if (m_mapsPending == 0) {
      // A callback without a matching request makes every later flight report false. This is an
      // internal lifecycle violation, not a recoverable diagnostic failure.
      std::abort();
    }
    --m_mapsPending;
    ++(success ? m_mapsCompleted : m_mapsFailed);
  }

  mutable std::mutex m_mutex;
  uint32_t m_mapsPending = 0;
  uint32_t m_mapsCompleted = 0;
  uint32_t m_mapsFailed = 0;
};

} // namespace aurora
