#include "frame_readback_flight.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

namespace aurora {
namespace {

TEST(FrameReadbackFlight, ReportsPendingSuccessfulAndFailedMapsIndependently) {
  FrameReadbackFlight flight;

  EXPECT_EQ(flight.snapshot().mapsPending, 0u);
  flight.map_requested();
  flight.map_requested();

  auto snapshot = flight.snapshot();
  EXPECT_EQ(snapshot.mapsPending, 2u);
  EXPECT_EQ(snapshot.mapsCompleted, 0u);
  EXPECT_EQ(snapshot.mapsFailed, 0u);

  {
    auto callback = flight.callback_started();
    callback.mark_map_success();
    snapshot = flight.snapshot();
    EXPECT_EQ(snapshot.mapsPending, 2u);
    EXPECT_EQ(snapshot.mapsCompleted, 0u);
  }
  snapshot = flight.snapshot();
  EXPECT_EQ(snapshot.mapsPending, 1u);
  EXPECT_EQ(snapshot.mapsCompleted, 1u);
  EXPECT_EQ(snapshot.mapsFailed, 0u);

  { auto callback = flight.callback_started(); }
  snapshot = flight.snapshot();
  EXPECT_EQ(snapshot.mapsPending, 0u);
  EXPECT_EQ(snapshot.mapsCompleted, 1u);
  EXPECT_EQ(snapshot.mapsFailed, 1u);
}

TEST(FrameReadbackFlight, RejectsCompletionWithoutARequest) {
  FrameReadbackFlight flight;
  EXPECT_DEATH({ auto callback = flight.callback_started(); }, "");
}

TEST(FrameReadbackFlight, SnapshotRemainsCoherentWhileCallbacksFinishConcurrently) {
  constexpr uint32_t WorkerCount = 8;
  constexpr uint32_t CallbacksPerWorker = 2000;
  constexpr uint32_t CallbackCount = WorkerCount * CallbacksPerWorker;
  FrameReadbackFlight flight;
  for (uint32_t index = 0; index < CallbackCount; ++index) {
    flight.map_requested();
  }

  std::atomic_bool start = false;
  std::atomic_uint32_t workersRunning = WorkerCount;
  std::vector<std::thread> workers;
  workers.reserve(WorkerCount);
  for (uint32_t worker = 0; worker < WorkerCount; ++worker) {
    workers.emplace_back([&, worker] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (uint32_t index = 0; index < CallbacksPerWorker; ++index) {
        auto callback = flight.callback_started();
        if ((worker + index) % 2 == 0) {
          callback.mark_map_success();
        }
      }
      workersRunning.fetch_sub(1, std::memory_order_release);
    });
  }

  start.store(true, std::memory_order_release);
  while (workersRunning.load(std::memory_order_acquire) != 0) {
    const auto snapshot = flight.snapshot();
    EXPECT_EQ(snapshot.mapsPending + snapshot.mapsCompleted + snapshot.mapsFailed, CallbackCount);
    std::this_thread::yield();
  }
  for (auto& worker : workers) {
    worker.join();
  }

  const auto snapshot = flight.snapshot();
  EXPECT_EQ(snapshot.mapsPending, 0u);
  EXPECT_EQ(snapshot.mapsCompleted, CallbackCount / 2);
  EXPECT_EQ(snapshot.mapsFailed, CallbackCount / 2);
}

} // namespace
} // namespace aurora
