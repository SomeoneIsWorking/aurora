#include "../lib/gfx/debug_markers.hpp"

#include <gtest/gtest.h>

#include <string>

namespace aurora::gfx {
namespace {

TEST(DebugMarkers, ReplayCopyKeepsOriginalLongLabelAcrossLaterFrameRepopulation) {
  const std::string original(4096, 'A');
  DebugMarkers queuedRealPass;
  const DebugMarkers::Id realCommand = queuedRealPass.record(original);

  // Replay capture deep-copies the pass-owned store before either command is consumed.
  DebugMarkers queuedReplayPass = queuedRealPass;
  const DebugMarkers::Id replayCommand = realCommand;

  // This models the exact state transition that broke the former global table: end_frame cleared
  // it and the next frame repopulated index zero before the queued command was consumed. A distinct
  // pass-owned store cannot retarget either copied command. Worker queue ordering is covered by the
  // live multi-presentation control, not reimplemented in this value-semantics test.
  DebugMarkers laterFrame;
  const DebugMarkers::Id laterCommand = laterFrame.record(std::string(4096, 'B'));

  EXPECT_EQ(queuedRealPass.label(realCommand), original);
  EXPECT_EQ(queuedReplayPass.label(replayCommand), original);
  EXPECT_NE(laterFrame.label(laterCommand), original);
}

} // namespace
} // namespace aurora::gfx
