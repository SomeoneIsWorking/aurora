#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace aurora::gfx::replay_lineage {

enum class Buffer : uint8_t {
  Vertices,
  Indices,
  Storage,
  Count,
};

struct BufferPrefixes {
  uint32_t vertices = 0;
  uint32_t indices = 0;
  uint32_t storage = 0;
};

struct Source {
  uint64_t frameId = 0;
  uint64_t commandHash = 0;
  uint64_t uniformHash = 0;
  uint32_t uniformBytes = 0;
  BufferPrefixes buffers;
  std::vector<uint32_t> passCommandCounts;
};

struct PrefixWriter {
  uint64_t frameId = 0;
  uint32_t bytes = 0;
};

struct WriterSnapshot {
  std::array<PrefixWriter, static_cast<size_t>(Buffer::Count)> buffers{};
};

class WriterEpochs {
public:
  void note_write(Buffer buffer, uint64_t frameId, uint32_t begin, uint32_t end);
  [[nodiscard]] WriterSnapshot snapshot() const noexcept;

private:
  WriterSnapshot m_snapshot;
};

enum class ValidationStatus : uint8_t {
  NotReplay,
  Valid,
  MissingSource,
  CommandMismatch,
  UniformMismatch,
  VertexWriterMismatch,
  IndexWriterMismatch,
  StorageWriterMismatch,
};

struct Observation {
  bool replayEmission = false;
  uint64_t commandHash = 0;
  uint64_t uniformHash = 0;
  WriterSnapshot writers;
};

struct Installation {
  uint64_t commandHash = 0;
  uint64_t uniformHash = 0;
};

[[nodiscard]] uint64_t hash_uniforms(std::span<const uint8_t> uniforms);
[[nodiscard]] Installation observe_installation(uint64_t commandHash, std::span<const uint8_t> uniforms);
[[nodiscard]] ValidationStatus validate_installation(const Source* source, const Installation& installation) noexcept;
[[nodiscard]] ValidationStatus validate_uniforms(uint64_t expectedHash, std::span<const uint8_t> uniforms);
[[nodiscard]] ValidationStatus validate_writers(const Source* source, const WriterSnapshot& writers) noexcept;
[[nodiscard]] Source capture(uint64_t frameId, uint64_t commandHash, std::span<const uint8_t> uniforms,
                             std::vector<uint32_t> passCommandCounts, BufferPrefixes buffers);
[[nodiscard]] ValidationStatus validate(const Source* source, const Observation& observation) noexcept;
[[nodiscard]] bool passed(ValidationStatus status) noexcept;
[[nodiscard]] const char* status_name(ValidationStatus status) noexcept;

} // namespace aurora::gfx::replay_lineage
