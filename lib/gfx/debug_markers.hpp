#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aurora::gfx {

// A render pass owns the strings named by its marker commands. Copying the pass for replay deep
// copies this store, so neither a delayed render worker nor a later frame can retarget a command.
class DebugMarkers {
public:
  using Id = uint32_t;

  [[nodiscard]] Id record(std::string label);
  [[nodiscard]] const std::string& label(Id id) const;

private:
  std::vector<std::string> m_labels;
};

} // namespace aurora::gfx
