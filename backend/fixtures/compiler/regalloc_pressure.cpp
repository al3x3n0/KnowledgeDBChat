#include <cstddef>
#include <cstdint>

extern "C" std::uint64_t register_pressure(
    const std::uint64_t *left,
    const std::uint64_t *right,
    std::size_t count
) {
    std::uint64_t a = 1, b = 3, c = 5, d = 7;
    std::uint64_t e = 11, f = 13, g = 17, h = 19;

    for (std::size_t index = 0; index < count; ++index) {
        a += left[index] ^ h;
        b ^= right[index] + a;
        c += (left[index] << 1) ^ b;
        d ^= (right[index] >> 1) + c;
        e += a ^ d;
        f ^= b + e;
        g += c ^ f;
        h ^= d + g;
    }
    return a ^ b ^ c ^ d ^ e ^ f ^ g ^ h;
}
