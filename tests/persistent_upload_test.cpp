#include "../lib/gfx/persistent_upload.hpp"
#include "../lib/gfx/render_worker.hpp"

#include <cstdint>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {
using aurora::gfx::persistent_upload::schedule;
namespace render_worker = aurora::gfx::render_worker;

class PersistentUploadTest : public testing::Test {
protected:
  void TearDown() override { render_worker::shutdown(); }
};

TEST_F(PersistentUploadTest, OrdersOwnedWriteBetweenSubmitsOnWorker) {
  render_worker::initialize();

  std::promise<void> workerBlocked;
  std::promise<void> releaseWorker;
  const std::shared_future<void> release = releaseWorker.get_future().share();
  render_worker::enqueue_work([&] {
    workerBlocked.set_value();
    release.wait();
  });
  workerBlocked.get_future().wait();

  const std::thread::id producerThread = std::this_thread::get_id();
  std::vector<std::string> order;
  std::vector<uint8_t> observedBytes;
  std::thread::id writeThread;
  uint64_t observedOffset = 0;
  bool sinkSawWorker = false;

  render_worker::enqueue_end_frame(40, [&] { order.emplace_back("older-submit"); });
  std::vector<uint8_t> source{0x10, 0x20, 0x30, 0x40, 0x50};
  schedule(96, source.data(), source.size(), [&](uint64_t offset, const uint8_t* data, size_t length) {
    order.emplace_back("persistent-write");
    observedBytes.assign(data, data + length);
    writeThread = std::this_thread::get_id();
    observedOffset = offset;
    sinkSawWorker = render_worker::is_worker_thread();
  });
  source.assign(source.size(), 0xff);
  render_worker::enqueue_end_frame(41, [&] { order.emplace_back("current-submit"); });

  releaseWorker.set_value();
  render_worker::synchronize();

  EXPECT_EQ(order, (std::vector<std::string>{"older-submit", "persistent-write", "current-submit"}));
  EXPECT_EQ(observedBytes, (std::vector<uint8_t>{0x10, 0x20, 0x30, 0x40, 0x50}));
  EXPECT_EQ(observedOffset, 96u);
  EXPECT_TRUE(sinkSawWorker);
  EXPECT_NE(writeThread, producerThread);
}

TEST_F(PersistentUploadTest, RefusesInlineWriteWithoutWorker) {
  const std::vector<uint8_t> source{0x10, 0x20, 0x30, 0x40};
  EXPECT_DEATH(schedule(96, source.data(), source.size(), [](uint64_t, const uint8_t*, size_t) {}),
               "Persistent GPU upload at offset 96 bypassed the render worker");
}

} // namespace
