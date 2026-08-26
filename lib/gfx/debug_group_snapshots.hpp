#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace aurora::gfx {

class DebugGroupSnapshots {
public:
  static constexpr uint32_t NoStack = std::numeric_limits<uint32_t>::max();

  uint32_t capture(const std::vector<std::string>& stack, uint64_t revision) {
    if (revision == m_revision) {
      return m_current;
    }
    m_revision = revision;
    if (stack.empty()) {
      m_current = NoStack;
      return m_current;
    }
    m_snapshots.push_back(stack);
    m_current = static_cast<uint32_t>(m_snapshots.size() - 1);
    return m_current;
  }

  const std::vector<std::string>& resolve(uint32_t index) const {
    static const std::vector<std::string> empty;
    if (index == NoStack) {
      return empty;
    }
    assert(index < m_snapshots.size());
    return m_snapshots[index];
  }

  bool contains(uint32_t index) const noexcept { return index == NoStack || index < m_snapshots.size(); }
  std::size_t size() const noexcept { return m_snapshots.size(); }

private:
  uint64_t m_revision = std::numeric_limits<uint64_t>::max();
  uint32_t m_current = NoStack;
  std::vector<std::vector<std::string>> m_snapshots;
};

} // namespace aurora::gfx
