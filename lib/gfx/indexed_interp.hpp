#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace aurora::gfx::indexed_interp {

uint32_t capture(uint64_t tag, const uint8_t* src, uint32_t byteSize, uint16_t stride, uint8_t population,
                 std::span<const uint64_t> quadKeys = {});
bool patch(uint32_t handle, float alpha, bool resampling, std::vector<uint8_t>& out);
bool birth_only(uint32_t handle);
bool reappearance_only(uint32_t handle);
bool selftest();
void report();

} // namespace aurora::gfx::indexed_interp
