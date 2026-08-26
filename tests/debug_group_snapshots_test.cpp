#include "../lib/gfx/debug_group_snapshots.hpp"

#include <gtest/gtest.h>

namespace aurora::gfx {
namespace {

TEST(DebugGroupSnapshots, ReusesOneSnapshotForEveryCommandAtTheSameRevision) {
  DebugGroupSnapshots snapshots;
  const std::vector<std::string> stack{"scene", "mario"};
  const uint32_t first = snapshots.capture(stack, 7);
  for (int command = 0; command < 100000; ++command) {
    EXPECT_EQ(snapshots.capture(stack, 7), first);
  }
  EXPECT_EQ(snapshots.size(), 1u);
  EXPECT_EQ(snapshots.resolve(first), stack);
}

TEST(DebugGroupSnapshots, RecordsChangesAndUsesASentinelForTheEmptyStack) {
  DebugGroupSnapshots snapshots;
  const std::vector<std::string> outer{"scene"};
  const std::vector<std::string> nested{"scene", "mario"};
  const uint32_t outerIndex = snapshots.capture(outer, 1);
  const uint32_t nestedIndex = snapshots.capture(nested, 2);
  const uint32_t emptyIndex = snapshots.capture({}, 3);

  EXPECT_NE(outerIndex, nestedIndex);
  EXPECT_EQ(snapshots.resolve(outerIndex), outer);
  EXPECT_EQ(snapshots.resolve(nestedIndex), nested);
  EXPECT_EQ(emptyIndex, DebugGroupSnapshots::NoStack);
  EXPECT_TRUE(snapshots.resolve(emptyIndex).empty());
  EXPECT_EQ(snapshots.size(), 2u);
}

} // namespace
} // namespace aurora::gfx
