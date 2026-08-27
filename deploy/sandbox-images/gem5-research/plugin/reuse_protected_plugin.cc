/*
 * Reuse-protected LRU, as a plugin. The same algorithm as the version
 * compiled into gem5, so the two are a null control on each other: identical
 * statistics prove the plugin boundary changes nothing.
 *
 * Includes the ABI header and nothing else -- no gem5 source, no gem5
 * headers, no link against gem5.
 */
#include "gem5_rp_plugin_abi.h"

#include <cstdlib>
#include <cstring>
#include <new>

namespace {

struct Policy
{
    uint64_t protect_after_touches;
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

Gem5RpPolicy *
create(const char *config)
{
    Policy *self = new (std::nothrow) Policy();
    if (!self) return nullptr;
    self->protect_after_touches = read_config(config, "protect_after_touches", 2);
    if (self->protect_after_touches == 0) self->protect_after_touches = 1;
    return reinterpret_cast<Gem5RpPolicy *>(self);
}

void destroy(Gem5RpPolicy *s) { delete reinterpret_cast<Policy *>(s); }

void
invalidate(Gem5RpPolicy *, Gem5RpEntry *e)
{
    e->last_touch_tick = 0;
    e->touches = 0;
}

void
touch(Gem5RpPolicy *, Gem5RpEntry *e, uint64_t tick)
{
    e->last_touch_tick = tick;
    e->touches++;
}

void
reset(Gem5RpPolicy *, Gem5RpEntry *e, uint64_t tick)
{
    /* Insertion, not reuse: a line that was protected before eviction must
     * not arrive protected. */
    e->last_touch_tick = tick;
    e->touches = 1;
}

size_t
get_victim(Gem5RpPolicy *s, Gem5RpEntry *const *entries, size_t count)
{
    const Policy *self = reinterpret_cast<const Policy *>(s);
    size_t oldest = 0;
    bool have_unprotected = false;
    size_t unprotected = 0;

    for (size_t i = 0; i < count; ++i) {
        /* An invalid entry is always the best victim. */
        if (entries[i]->last_touch_tick == 0) return i;

        if (entries[i]->last_touch_tick < entries[oldest]->last_touch_tick) {
            oldest = i;
        }
        if (entries[i]->touches < self->protect_after_touches) {
            if (!have_unprotected ||
                entries[i]->last_touch_tick <
                    entries[unprotected]->last_touch_tick) {
                unprotected = i;
                have_unprotected = true;
            }
        }
    }
    return have_unprotected ? unprotected : oldest;
}

const Gem5RpApiV1 API = {
    GEM5_RP_ABI_VERSION,
    create, destroy, invalidate, touch, reset, get_victim,
};

} // namespace

extern "C" const Gem5RpApiV1 *gem5_rp_api_v1(void) { return &API; }
