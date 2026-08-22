#include <gtest/gtest.h>

#include "../lib/gx/auto_array_sizing.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace {

using aurora::gx::IndexedAttrLayout;
using aurora::gx::fifo::scan_auto_array_max_indices;

std::array<uint32_t, 8> scan(const std::vector<IndexedAttrLayout>& fields, const std::vector<uint8_t>& vertices,
                             uint32_t vertexStride) {
  std::array<uint32_t, 8> maxima{};
  scan_auto_array_max_indices(fields.data(), static_cast<uint8_t>(fields.size()), vertices.data(),
                              static_cast<uint32_t>(vertices.size() / vertexStride), vertexStride, maxima.data());
  return maxima;
}

TEST(GxAutoArraySizing, ScansMixedEightAndSixteenBitIndices) {
  const std::vector<IndexedAttrLayout> fields{
      {.offset = 0, .width = 1, .attr = 1}, {.offset = 1, .width = 2, .attr = 3}, {.offset = 3, .width = 1, .attr = 6}};
  const std::vector<uint8_t> vertices{
      2, 0x01, 0x23, 9, 7, 0x00, 0x42, 3, 5, 0x02, 0x01, 12,
  };

  const auto maxima = scan(fields, vertices, 4);
  EXPECT_EQ(maxima[1], 7u);
  EXPECT_EQ(maxima[3], 0x201u);
  EXPECT_EQ(maxima[6], 12u);
}

TEST(GxAutoArraySizing, CombinesFlattenedNbt3FieldsForOneAttribute) {
  const std::vector<IndexedAttrLayout> fields{
      {.offset = 0, .width = 2, .attr = 2}, {.offset = 2, .width = 2, .attr = 2}, {.offset = 4, .width = 2, .attr = 2}};
  const std::vector<uint8_t> vertices{
      0x00, 0x02, 0x00, 0x40, 0x00, 0x08, 0x01, 0x00, 0x00, 0x03, 0x00, 0x20,
  };

  const auto maxima = scan(fields, vertices, 6);
  EXPECT_EQ(maxima[2], 0x100u);
}

TEST(GxAutoArraySizing, ComparisonDetectsChangedIndex) {
  const std::vector<IndexedAttrLayout> fields{{.offset = 0, .width = 1, .attr = 0}};
  std::vector<uint8_t> vertices{1, 3, 2};
  const auto baseline = scan(fields, vertices, 1);
  vertices[2] = 9;
  const auto changed = scan(fields, vertices, 1);

  ASSERT_EQ(baseline[0], 3u);
  EXPECT_EQ(changed[0], 9u);
  EXPECT_NE(baseline, changed) << "the scan comparison failed its known-difference control";
}

TEST(GxAutoArraySizing, ScansGenericFallbackBeyondSpecializedFieldCounts) {
  std::vector<IndexedAttrLayout> fields;
  std::vector<uint8_t> vertices;
  for (uint8_t attr = 0; attr < 7; ++attr) {
    fields.push_back({.offset = attr, .width = 1, .attr = attr});
    vertices.push_back(static_cast<uint8_t>(attr + 10));
  }

  const auto maxima = scan(fields, vertices, 7);
  for (uint8_t attr = 0; attr < 7; ++attr) {
    EXPECT_EQ(maxima[attr], static_cast<uint32_t>(attr + 10));
  }
}

TEST(GxAutoArraySizing, EverySpecializedDispatchCountProducesTheSameMaximaRule) {
  for (uint8_t fieldCount = 1; fieldCount <= 6; ++fieldCount) {
    std::vector<IndexedAttrLayout> fields;
    std::vector<uint8_t> vertices;
    for (uint8_t attr = 0; attr < fieldCount; ++attr) {
      fields.push_back({.offset = attr, .width = 1, .attr = attr});
    }
    for (uint8_t base : {uint8_t{1}, uint8_t{11}}) {
      for (uint8_t attr = 0; attr < fieldCount; ++attr) {
        vertices.push_back(static_cast<uint8_t>(base + attr));
      }
    }

    const auto maxima = scan(fields, vertices, fieldCount);
    for (uint8_t attr = 0; attr < fieldCount; ++attr) {
      EXPECT_EQ(maxima[attr], static_cast<uint32_t>(attr + 11)) << "field count " << +fieldCount;
    }
  }
}

TEST(GxAutoArraySizing, EmptyLayoutPerformsNoScan) {
  const auto maxima = scan({}, {1, 2, 3}, 1);
  EXPECT_EQ(maxima, (std::array<uint32_t, 8>{}));
}

} // namespace
