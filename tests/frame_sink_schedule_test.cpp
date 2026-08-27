#include "frame_sink_schedule.hpp"

#include <gtest/gtest.h>

namespace aurora {
namespace {

TEST(FrameSinkSchedule, FirstFrameAndConfiguredCadenceAreSelectedWithoutQueryMutation) {
  FrameSinkSchedule schedule;
  schedule.configure(3);

  EXPECT_TRUE(schedule.will_capture());
  EXPECT_TRUE(schedule.will_capture());
  EXPECT_TRUE(schedule.consume());
  EXPECT_FALSE(schedule.will_capture());
  EXPECT_FALSE(schedule.consume());
  EXPECT_FALSE(schedule.will_capture());
  EXPECT_FALSE(schedule.consume());
  EXPECT_TRUE(schedule.will_capture());
  EXPECT_TRUE(schedule.consume());
}

TEST(FrameSinkSchedule, ReconfigureResetsCadenceAndDisableRefusesCapture) {
  FrameSinkSchedule schedule;
  schedule.configure(2);
  EXPECT_TRUE(schedule.consume());
  EXPECT_FALSE(schedule.will_capture());

  schedule.configure(1);
  EXPECT_TRUE(schedule.will_capture());
  EXPECT_TRUE(schedule.consume());
  EXPECT_TRUE(schedule.will_capture());

  schedule.configure(0);
  EXPECT_FALSE(schedule.will_capture());
  EXPECT_FALSE(schedule.consume());
}

TEST(FrameSinkSchedule, MetadataCopiesExactEmissionIdentityAndReplayLineage) {
  AuroraGpuSubmitInfo submit{};
  submit.frameId = 101;
  submit.replaySourceFrameId = 77;
  submit.replayEmission = 1;

  const AuroraFrameSinkInfo info = make_frame_sink_info(submit);
  submit.frameId = 999;
  submit.replaySourceFrameId = 998;

  EXPECT_EQ(info.structSize, sizeof(AuroraFrameSinkInfo));
  EXPECT_EQ(info.version, AURORA_FRAME_SINK_INFO_VERSION);
  EXPECT_EQ(info.frameId, 101u);
  EXPECT_EQ(info.replaySourceFrameId, 77u);
  EXPECT_EQ(info.replayEmission, 1u);
}

} // namespace
} // namespace aurora
