#pragma once

#include <cstdint>
#include <vector>

namespace aurora::gfx::indexed_interp {

uint32_t capture(uint64_t tag, const uint8_t* src, uint32_t byteSize, uint16_t stride, uint8_t population);
bool patch(uint32_t handle, float alpha, bool resampling, std::vector<uint8_t>& out);
bool selftest();
void report();

} // namespace aurora::gfx::indexed_interp
