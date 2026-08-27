#include "../lib/gfx/gpu_submit_probe.hpp"
#include "../lib/gfx/replay_lineage.hpp"

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace aurora::gfx::replay_lineage {
namespace {

constexpr uint64_t SourceFrame = 1607;
constexpr BufferPrefixes SourcePrefixes{.vertices = 96, .indices = 48, .storage = 160};

uint64_t command_hash(const std::vector<uint64_t>& pipelines) {
  gpu_submit_probe::Builder builder{gpu_submit_probe::FrameInput{.passCount = 1, .replaySourceFrameId = SourceFrame}};
  builder.begin_pass({.label = "source", .commandCount = static_cast<uint32_t>(pipelines.size())});
  for (uint64_t pipeline : pipelines) {
    builder.add_draw(gpu_submit_probe::GxDrawInput{.pipeline = pipeline});
  }
  builder.end_pass();
  return builder.finish().commandHash;
}

WriterSnapshot source_writers() {
  WriterEpochs writers;
  writers.note_write(Buffer::Vertices, SourceFrame, 0, SourcePrefixes.vertices);
  writers.note_write(Buffer::Indices, SourceFrame, 0, SourcePrefixes.indices);
  writers.note_write(Buffer::Storage, SourceFrame, 0, SourcePrefixes.storage);
  return writers.snapshot();
}

TEST(ReplayLineage, MatchingObservationPassesPureValidation) {
  const std::array<uint8_t, 8> uniforms{0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<uint64_t> pipelines{10, 11, 12};
  const Source source = capture(SourceFrame, command_hash(pipelines), uniforms,
                                {static_cast<uint32_t>(pipelines.size())}, SourcePrefixes);
  const Observation replay{
      .replayEmission = true,
      .commandHash = source.commandHash,
      .uniformHash = source.uniformHash,
      .writers = source_writers(),
  };

  EXPECT_EQ(validate(&source, replay), ValidationStatus::Valid);
  EXPECT_TRUE(passed(validate(&source, replay)));
}

TEST(ReplayLineage, IntendedInterpolationOnlyOnFirstEmissionPasses) {
  const std::array<uint8_t, 4> sourceUniforms{1, 2, 3, 4};
  const std::array<uint8_t, 4> interpolatedUniforms{9, 2, 3, 4};
  const Source source = capture(SourceFrame, command_hash({1, 2, 3}), sourceUniforms, {3}, SourcePrefixes);
  const Observation firstEmission{
      .replayEmission = false,
      .commandHash = command_hash({1, 99, 3}),
      .uniformHash = hash_uniforms(interpolatedUniforms),
  };

  EXPECT_EQ(validate(&source, firstEmission), ValidationStatus::NotReplay);
  EXPECT_TRUE(passed(validate(&source, firstEmission)));
}

TEST(ReplayLineage, RetainedIntermediateSampleValidatesThePreInterpolationInstallation) {
  const std::array<uint8_t, 4> sourceUniforms{1, 2, 3, 4};
  const Source source = capture(SourceFrame, command_hash({1, 2, 3}), sourceUniforms, {3}, SourcePrefixes);
  const Installation installation = observe_installation(source.commandHash, sourceUniforms);

  // A retained presentation between the first and final sample intentionally interpolates its
  // installed command/uniform copy. The worker first observes the untouched installation, while
  // the final pre-submit uniform gate uses a new expected hash for the intentional result.
  std::array<uint8_t, 4> interpolatedUniforms = sourceUniforms;
  interpolatedUniforms[0] = 9;
  const uint64_t interpolatedCommandHash = command_hash({1, 99, 3});
  ASSERT_NE(hash_uniforms(interpolatedUniforms), installation.uniformHash);
  ASSERT_NE(interpolatedCommandHash, installation.commandHash);

  EXPECT_EQ(validate_installation(&source, installation), ValidationStatus::Valid);
}

TEST(ReplayLineage, PreSubmitUniformCheckReobservesTheFinalBytes) {
  const std::array<uint8_t, 4> sourceUniforms{1, 2, 3, 4};
  std::array<uint8_t, 4> interpolatedUniforms{9, 2, 3, 4};
  const uint64_t expected = hash_uniforms(interpolatedUniforms);

  EXPECT_EQ(validate_uniforms(expected, interpolatedUniforms), ValidationStatus::Valid);
  interpolatedUniforms[2] ^= 0x80;
  EXPECT_EQ(validate_uniforms(expected, interpolatedUniforms), ValidationStatus::UniformMismatch);
}

TEST(ReplayLineage, MutatedMiddleDrawOutsideBoundedTailFails) {
  std::vector<uint64_t> pipelines(12);
  for (size_t index = 0; index < pipelines.size(); ++index) {
    pipelines[index] = 100 + index;
  }
  const std::array<uint8_t, 4> uniforms{1, 2, 3, 4};
  const Source source = capture(SourceFrame, command_hash(pipelines), uniforms,
                                {static_cast<uint32_t>(pipelines.size())}, SourcePrefixes);
  pipelines[1] = 0xdeadbeef;
  const Observation replay{
      .replayEmission = true,
      .commandHash = command_hash(pipelines),
      .uniformHash = source.uniformHash,
      .writers = source_writers(),
  };

  EXPECT_EQ(validate(&source, replay), ValidationStatus::CommandMismatch);
  EXPECT_FALSE(passed(validate(&source, replay)));
}

TEST(ReplayLineage, MutatedSourceUniformByteFails) {
  const std::array<uint8_t, 4> uniforms{1, 2, 3, 4};
  auto mutated = uniforms;
  mutated[2] ^= 0x80;
  const Source source = capture(SourceFrame, command_hash({1}), uniforms, {1}, SourcePrefixes);
  const Observation replay{
      .replayEmission = true,
      .commandHash = source.commandHash,
      .uniformHash = hash_uniforms(mutated),
      .writers = source_writers(),
  };

  EXPECT_EQ(validate(&source, replay), ValidationStatus::UniformMismatch);
  EXPECT_FALSE(passed(validate(&source, replay)));
}

TEST(ReplayLineage, SourcePrefixWriterMustSurviveUntilReplay) {
  const std::array<uint8_t, 1> uniforms{1};
  const Source source = capture(SourceFrame, command_hash({1}), uniforms, {1}, SourcePrefixes);
  WriterEpochs writers;
  writers.note_write(Buffer::Vertices, SourceFrame, 0, SourcePrefixes.vertices);
  writers.note_write(Buffer::Indices, SourceFrame, 0, SourcePrefixes.indices);
  writers.note_write(Buffer::Storage, SourceFrame, 0, SourcePrefixes.storage);
  writers.note_write(Buffer::Vertices, SourceFrame + 1, 0, 4);
  const Observation replay{
      .replayEmission = true,
      .commandHash = source.commandHash,
      .uniformHash = source.uniformHash,
      .writers = writers.snapshot(),
  };

  EXPECT_EQ(validate(&source, replay), ValidationStatus::VertexWriterMismatch);
}

TEST(ReplayLineage, OverlayWriteAboveProtectedPrefixPreservesOwner) {
  const std::array<uint8_t, 1> uniforms{1};
  const Source source = capture(SourceFrame, command_hash({1}), uniforms, {1}, SourcePrefixes);
  WriterEpochs writers;
  writers.note_write(Buffer::Vertices, SourceFrame, 0, SourcePrefixes.vertices);
  writers.note_write(Buffer::Indices, SourceFrame, 0, SourcePrefixes.indices);
  writers.note_write(Buffer::Storage, SourceFrame, 0, SourcePrefixes.storage);
  writers.note_write(Buffer::Vertices, SourceFrame + 1, SourcePrefixes.vertices, SourcePrefixes.vertices + 64);
  const Observation replay{
      .replayEmission = true,
      .commandHash = source.commandHash,
      .uniformHash = source.uniformHash,
      .writers = writers.snapshot(),
  };

  EXPECT_EQ(validate(&source, replay), ValidationStatus::Valid);
}

TEST(ReplayLineage, PartialOverwriteRetainsOnlyTheStillOwnedLowerPrefix) {
  constexpr BufferPrefixes ShortPrefixes{.vertices = 32, .indices = 48, .storage = 160};
  const std::array<uint8_t, 1> uniforms{1};
  const Source fullSource = capture(SourceFrame, command_hash({1}), uniforms, {1}, SourcePrefixes);
  const Source shortSource = capture(SourceFrame, command_hash({1}), uniforms, {1}, ShortPrefixes);
  WriterEpochs writers;
  writers.note_write(Buffer::Vertices, SourceFrame, 0, SourcePrefixes.vertices);
  writers.note_write(Buffer::Indices, SourceFrame, 0, SourcePrefixes.indices);
  writers.note_write(Buffer::Storage, SourceFrame, 0, SourcePrefixes.storage);
  writers.note_write(Buffer::Vertices, SourceFrame + 1, 32, 64);
  const Observation observation{
      .replayEmission = true,
      .commandHash = fullSource.commandHash,
      .uniformHash = fullSource.uniformHash,
      .writers = writers.snapshot(),
  };

  EXPECT_EQ(validate(&fullSource, observation), ValidationStatus::VertexWriterMismatch);
  EXPECT_EQ(validate(&shortSource, observation), ValidationStatus::Valid);
}

TEST(ReplayLineage, ProbeV3CarriesSourceTripleAtAppendOnlyBoundary) {
  constexpr uint64_t CommandHash = 0x1111222233334444ull;
  constexpr uint64_t UniformHash = 0x5555666677778888ull;
  gpu_submit_probe::Builder builder{gpu_submit_probe::FrameInput{
      .replaySourceFrameId = SourceFrame,
      .replaySourceCommandHash = CommandHash,
      .replaySourceUniformHash = UniformHash,
  }};
  const AuroraGpuSubmitInfo probe = builder.finish();

  EXPECT_EQ(probe.version, 3u);
  EXPECT_EQ(probe.structSize, 1536u);
  EXPECT_EQ(probe.replaySourceFrameId, SourceFrame);
  EXPECT_EQ(probe.replaySourceCommandHash, CommandHash);
  EXPECT_EQ(probe.replaySourceUniformHash, UniformHash);
}

} // namespace
} // namespace aurora::gfx::replay_lineage
