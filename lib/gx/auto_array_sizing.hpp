#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace aurora::gx {

// Byte layout of one indexed field in a FIFO vertex. NBT3's three consecutive normal indices are
// flattened into three fields so the hot scan has no per-field inner loop.
struct IndexedAttrLayout {
  uint16_t offset = 0;
  uint8_t width = 0;
  uint8_t attr = 0;
};

namespace fifo {

inline uint32_t read_index(const uint8_t* data, const IndexedAttrLayout& field) {
  if (field.width == 1) {
    return data[field.offset];
  }
  uint16_t value;
  std::memcpy(&value, data + field.offset, sizeof(value));
  return static_cast<uint16_t>((value << 8) | (value >> 8));
}

template <size_t FieldCount>
inline void scan_auto_array_max_indices_fixed(const IndexedAttrLayout* fields, const uint8_t* vertices,
                                              uint32_t vertexCount, uint32_t vertexStride, uint32_t* maxIndices) {
  for (uint32_t vertex = 0; vertex < vertexCount; ++vertex) {
    const uint8_t* data = vertices + vertex * vertexStride;
    for (size_t fieldIndex = 0; fieldIndex < FieldCount; ++fieldIndex) {
      const auto& field = fields[fieldIndex];
      const uint32_t index = read_index(data, field);
      if (index > maxIndices[field.attr]) {
        maxIndices[field.attr] = index;
      }
    }
  }
}

inline void scan_auto_array_max_indices(const IndexedAttrLayout* fields, uint8_t fieldCount, const uint8_t* vertices,
                                        uint32_t vertexCount, uint32_t vertexStride, uint32_t* maxIndices) {
  switch (fieldCount) {
  case 0:
    return;
  case 1:
    return scan_auto_array_max_indices_fixed<1>(fields, vertices, vertexCount, vertexStride, maxIndices);
  case 2:
    return scan_auto_array_max_indices_fixed<2>(fields, vertices, vertexCount, vertexStride, maxIndices);
  case 3:
    return scan_auto_array_max_indices_fixed<3>(fields, vertices, vertexCount, vertexStride, maxIndices);
  case 4:
    return scan_auto_array_max_indices_fixed<4>(fields, vertices, vertexCount, vertexStride, maxIndices);
  case 5:
    return scan_auto_array_max_indices_fixed<5>(fields, vertices, vertexCount, vertexStride, maxIndices);
  case 6:
    return scan_auto_array_max_indices_fixed<6>(fields, vertices, vertexCount, vertexStride, maxIndices);
  default:
    break;
  }

  for (uint32_t vertex = 0; vertex < vertexCount; ++vertex) {
    const uint8_t* data = vertices + vertex * vertexStride;
    for (uint8_t fieldIndex = 0; fieldIndex < fieldCount; ++fieldIndex) {
      const auto& field = fields[fieldIndex];
      const uint32_t index = read_index(data, field);
      if (index > maxIndices[field.attr]) {
        maxIndices[field.attr] = index;
      }
    }
  }
}

} // namespace fifo
} // namespace aurora::gx
