/*
 * Next-N-lines prefetching, as a plugin -- a deliberate reimplementation of
 * gem5's own TaggedPrefetcher. If the two agree, the shim is faithful; a
 * plugin whose first outing is a novel algorithm can only be checked against
 * an opinion.
 *
 * Includes the ABI header and nothing else.
 */
#include <gem5_pf_plugin_abi.h>

#include <cstdlib>
#include <cstring>
#include <new>

namespace {

struct Tagged
{
    uint32_t block_size;
    uint32_t degree;
};

uint64_t
read_config(const char *config, const char *key, uint64_t fallback)
{
    if (!config) return fallback;
    const char *found = std::strstr(config, key);
    if (!found) return fallback;
    const char *eq = std::strchr(found, '=');
    if (!eq) return fallback;
    char *end = nullptr;
    unsigned long long value = std::strtoull(eq + 1, &end, 10);
    return (end == eq + 1) ? fallback : (uint64_t)value;
}

Gem5PfPrefetcher *
create(const char *config, uint32_t block_size)
{
    Tagged *self = new (std::nothrow) Tagged();
    if (!self) return nullptr;
    self->block_size = block_size ? block_size : 64;
    self->degree = (uint32_t)read_config(config, "degree", 2);
    if (self->degree == 0) self->degree = 1;
    return reinterpret_cast<Gem5PfPrefetcher *>(self);
}

void destroy(Gem5PfPrefetcher *s) { delete reinterpret_cast<Tagged *>(s); }

size_t
calculate(Gem5PfPrefetcher *s, const Gem5PfAccess *access,
          Gem5PfRequest *out, size_t max_out)
{
    const Tagged *self = reinterpret_cast<const Tagged *>(s);
    uint64_t block = access->address & ~(uint64_t)(self->block_size - 1);

    size_t n = 0;
    for (uint32_t i = 1; i <= self->degree && n < max_out; ++i) {
        out[n].address = block + (uint64_t)i * self->block_size;
        out[n].priority = 0;
        ++n;
    }
    return n;
}

const Gem5PfApiV1 API = {
    GEM5_PF_ABI_VERSION, create, destroy, calculate,
};

} // namespace

extern "C" const Gem5PfApiV1 *gem5_pf_api_v1(void) { return &API; }
