/*
 * The contract between gem5 and a replacement-policy plugin.
 *
 * Deliberately free of every gem5 type. A plugin includes this header and
 * nothing else, which is what makes it compile in seconds, in an image that
 * carries no gem5 source, and keeps compiling when gem5 is upgraded -- the
 * whole reason to have a plugin boundary rather than a rebuild.
 */
#ifndef GEM5_RP_PLUGIN_ABI_H
#define GEM5_RP_PLUGIN_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEM5_RP_ABI_VERSION 1u
#define GEM5_RP_SCRATCH_WORDS 4

/* What a plugin may know and remember about one cache entry. gem5 owns the
 * storage; the plugin owns the meaning of `scratch`. */
typedef struct
{
    uint64_t last_touch_tick;   /* 0 exactly when the entry is invalid */
    uint64_t touches;           /* touches since the last reset */
    uint64_t scratch[GEM5_RP_SCRATCH_WORDS];
} Gem5RpEntry;

typedef struct Gem5RpPolicy Gem5RpPolicy;   /* opaque, the plugin's own state */

typedef struct
{
    uint32_t abi_version;

    Gem5RpPolicy *(*create)(const char *config);
    void (*destroy)(Gem5RpPolicy *self);

    void (*invalidate)(Gem5RpPolicy *self, Gem5RpEntry *entry);
    void (*touch)(Gem5RpPolicy *self, Gem5RpEntry *entry, uint64_t tick);
    void (*reset)(Gem5RpPolicy *self, Gem5RpEntry *entry, uint64_t tick);

    /* Index into `entries` of the one to evict. Returning >= count is a bug
     * in the plugin and gem5 refuses the run rather than evicting at random. */
    size_t (*get_victim)(Gem5RpPolicy *self,
                         Gem5RpEntry *const *entries,
                         size_t count);
} Gem5RpApiV1;

/* Every plugin exports exactly one symbol of this name and shape. */
#define GEM5_RP_ENTRY_SYMBOL "gem5_rp_api_v1"
typedef const Gem5RpApiV1 *(*Gem5RpEntryPoint)(void);

#ifdef __cplusplus
}
#endif

#endif /* GEM5_RP_PLUGIN_ABI_H */
