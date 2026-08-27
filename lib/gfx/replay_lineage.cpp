#include "replay_lineage.hpp"

#define XXH_STATIC_LINKING_ONLY
#include <xxhash.h>

#include <algorithm>
#include <utility>

namespace aurora::gfx::replay_lineage {
namespace {

constexpr size_t buffer_index(Buffer buffer) noexcept { return static_cast<size_t>(buffer); }

bool owns_prefix(const WriterSnapshot& writers, Buffer buffer, uint64_t frameId, uint32_t bytes) noexcept {
  if (bytes == 0) {
    return true;
  }
  const PrefixWriter& writer = writers.buffers[buffer_index(buffer)];
  return writer.frameId == frameId && writer.bytes >= bytes;
}

} // namespace

void WriterEpochs::note_write(Buffer buffer, uint64_t frameId, uint32_t begin, uint32_t end) {
  if (begin >= end) {
    return;
  }

  PrefixWriter& writer = m_snapshot.buffers[buffer_index(buffer)];
  if (begin == 0) {
    writer = {.frameId = frameId, .bytes = end};
    return;
  }
  if (writer.frameId == frameId && begin <= writer.bytes) {
    writer.bytes = std::max(writer.bytes, end);
    return;
  }
  if (begin < writer.bytes) {
    // The range below begin is still a homogeneous prefix owned by the previous frame. Retaining
    // that exact shorter prefix keeps the model truthful without letting a replay whose required
    // bytes cross the overwrite pass validation.
    writer.bytes = begin;
  }
  // A write wholly above the known prefix is an overlay and does not change prefix ownership.
}

WriterSnapshot WriterEpochs::snapshot() const noexcept { return m_snapshot; }

uint64_t hash_uniforms(std::span<const uint8_t> uniforms) { return XXH3_64bits(uniforms.data(), uniforms.size()); }

Installation observe_installation(uint64_t commandHash, std::span<const uint8_t> uniforms) {
  return {.commandHash = commandHash, .uniformHash = hash_uniforms(uniforms)};
}

ValidationStatus validate_installation(const Source* source, const Installation& installation) noexcept {
  if (source == nullptr || source->frameId == 0) {
    return ValidationStatus::MissingSource;
  }
  if (installation.commandHash != source->commandHash) {
    return ValidationStatus::CommandMismatch;
  }
  if (installation.uniformHash != source->uniformHash) {
    return ValidationStatus::UniformMismatch;
  }
  return ValidationStatus::Valid;
}

ValidationStatus validate_command(uint64_t expectedHash, uint64_t observedHash) noexcept {
  return observedHash == expectedHash ? ValidationStatus::Valid : ValidationStatus::CommandMismatch;
}

ValidationStatus validate_uniforms(uint64_t expectedHash, std::span<const uint8_t> uniforms) {
  return hash_uniforms(uniforms) == expectedHash ? ValidationStatus::Valid : ValidationStatus::UniformMismatch;
}

ValidationStatus validate_writers(const Source* source, const WriterSnapshot& writers) noexcept {
  if (source == nullptr || source->frameId == 0) {
    return ValidationStatus::MissingSource;
  }
  if (!owns_prefix(writers, Buffer::Vertices, source->frameId, source->buffers.vertices)) {
    return ValidationStatus::VertexWriterMismatch;
  }
  if (!owns_prefix(writers, Buffer::Indices, source->frameId, source->buffers.indices)) {
    return ValidationStatus::IndexWriterMismatch;
  }
  if (!owns_prefix(writers, Buffer::Storage, source->frameId, source->buffers.storage)) {
    return ValidationStatus::StorageWriterMismatch;
  }
  return ValidationStatus::Valid;
}

Source capture(uint64_t frameId, uint64_t commandHash, std::span<const uint8_t> uniforms,
               std::vector<uint32_t> passCommandCounts, BufferPrefixes buffers) {
  return {
      .frameId = frameId,
      .commandHash = commandHash,
      .uniformHash = hash_uniforms(uniforms),
      .uniformBytes = static_cast<uint32_t>(uniforms.size()),
      .buffers = buffers,
      .passCommandCounts = std::move(passCommandCounts),
  };
}

ValidationStatus validate(const Source* source, const Observation& observation) noexcept {
  if (!observation.replayEmission) {
    return ValidationStatus::NotReplay;
  }
  if (source == nullptr || source->frameId == 0) {
    return ValidationStatus::MissingSource;
  }
  const ValidationStatus installationStatus =
      validate_installation(source, {.commandHash = observation.commandHash, .uniformHash = observation.uniformHash});
  if (installationStatus != ValidationStatus::Valid) {
    return installationStatus;
  }
  return validate_writers(source, observation.writers);
}

bool passed(ValidationStatus status) noexcept {
  return status == ValidationStatus::NotReplay || status == ValidationStatus::Valid;
}

const char* status_name(ValidationStatus status) noexcept {
  switch (status) {
  case ValidationStatus::NotReplay:
    return "not-replay";
  case ValidationStatus::Valid:
    return "valid";
  case ValidationStatus::MissingSource:
    return "missing-source";
  case ValidationStatus::CommandMismatch:
    return "command-mismatch";
  case ValidationStatus::UniformMismatch:
    return "uniform-mismatch";
  case ValidationStatus::VertexWriterMismatch:
    return "vertex-writer-mismatch";
  case ValidationStatus::IndexWriterMismatch:
    return "index-writer-mismatch";
  case ValidationStatus::StorageWriterMismatch:
    return "storage-writer-mismatch";
  }
  return "unknown";
}

} // namespace aurora::gfx::replay_lineage
