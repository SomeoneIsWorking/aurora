#include "debug_markers.hpp"

#include "../internal.hpp"

#include <limits>
#include <utility>

namespace aurora::gfx {
namespace {

Module Log("aurora::gfx::debug_markers");

} // namespace

DebugMarkers::Id DebugMarkers::record(std::string label) {
  CHECK(m_labels.size() < std::numeric_limits<Id>::max(), "Render pass exceeded the debug-marker ID space");
  const Id id = static_cast<Id>(m_labels.size());
  m_labels.emplace_back(std::move(label));
  return id;
}

const std::string& DebugMarkers::label(Id id) const {
  CHECK(id < m_labels.size(), "Render pass debug marker {} is outside its {} owned labels", id, m_labels.size());
  return m_labels[id];
}

} // namespace aurora::gfx
