#include "gfx/pipeline_worker_gate.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <optional>
#include <thread>

#include <gtest/gtest.h>

namespace aurora::gfx {
namespace {

TEST(PipelineWorkerGate, PauseWaitsForInFlightWorkAndBlocksNewWork) {
  PipelineWorkerGate gate;
  std::optional<PipelineWorkerGate::WorkLease> inFlight;
  inFlight.emplace(gate.enter_work());

  auto pauseFinished = std::async(std::launch::async, [&gate] { gate.pause_and_wait(); });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
  while (!gate.is_paused() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  const bool pauseObserved = gate.is_paused();
  if (!pauseObserved) {
    inFlight.reset();
    gate.resume();
  }
  ASSERT_TRUE(pauseObserved);
  EXPECT_EQ(pauseFinished.wait_for(std::chrono::milliseconds{0}), std::future_status::timeout);

  inFlight.reset();
  ASSERT_EQ(pauseFinished.wait_for(std::chrono::seconds{1}), std::future_status::ready);

  std::promise<void> attemptedPromise;
  auto attempted = attemptedPromise.get_future();
  std::atomic_bool entered = false;
  std::thread nextWorker([&] {
    attemptedPromise.set_value();
    auto work = gate.enter_work();
    entered.store(true, std::memory_order_release);
  });
  ASSERT_EQ(attempted.wait_for(std::chrono::seconds{1}), std::future_status::ready);
  EXPECT_FALSE(entered.load(std::memory_order_acquire));

  gate.resume();
  nextWorker.join();
  EXPECT_TRUE(entered.load(std::memory_order_acquire));
}

TEST(PipelineWorkerGate, UnpausedGateAdmitsWork) {
  PipelineWorkerGate gate;
  auto work = gate.enter_work();
  EXPECT_FALSE(gate.is_paused());
}

} // namespace
} // namespace aurora::gfx
