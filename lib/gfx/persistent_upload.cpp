#include "persistent_upload.hpp"

#include "../internal.hpp"
#include "render_worker.hpp"

#include <utility>
#include <vector>

namespace aurora::gfx::persistent_upload {
static Module Log("aurora::gfx::persistent_upload");

void schedule(uint64_t offset, const uint8_t* data, size_t length, WriteSink sink) {
  CHECK(data != nullptr, "Persistent upload at offset {} has a null source", offset);
  CHECK(length > 0, "Persistent upload at offset {} has no bytes", offset);
  CHECK(static_cast<bool>(sink), "Persistent upload at offset {} has no write sink", offset);

  std::vector<uint8_t> ownedBytes(data, data + length);
  render_worker::enqueue_persistent_upload([offset, bytes = std::move(ownedBytes), sink = std::move(sink)]() mutable {
    CHECK(render_worker::is_worker_thread(), "Persistent GPU upload at offset {} bypassed the render worker", offset);
    sink(offset, bytes.data(), bytes.size());
  });
}

} // namespace aurora::gfx::persistent_upload
