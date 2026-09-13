/*
 * The contract between gem5 and a prefetcher plugin.
 *
 * Same shape and the same reasoning as gem5_rp_plugin_abi.h: no gem5 type
 * crosses it, so a plugin compiles in under a second, against one header, in
 * an image carrying no gem5 source, and survives a gem5 upgrade.
 *
 * A prefetcher is a harder fit than a replacement policy, because gem5 hands
 * it a queue to fill rather than a choice to make. The shim keeps the queue --
 * throttling, page-crossing checks and latency are gem5's job and getting them
 * wrong is how a prefetcher measures the harness instead of the algorithm --
 * and asks the plugin only the question that is actually the algorithm: given
 * this access, which addresses would you fetch?
 */
#ifndef GEM5_PF_PLUGIN_ABI_H
#define GEM5_PF_PLUGIN_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEM5_PF_ABI_VERSION 1u

/* One access the prefetcher was told about. */
typedef struct
{
    uint64_t address;      /* the address accessed */
    uint64_t pc;           /* the instruction, when has_pc */
    uint32_t size;         /* bytes */
    uint32_t requestor_id;
    uint8_t has_pc;
    uint8_t secure;
    uint8_t write;
    uint8_t cache_miss;    /* whether this access missed */
} Gem5PfAccess;

/* One address the plugin wants fetched. Priority orders the queue; 0 is the
 * usual answer and larger is more urgent. */
typedef struct
{
    uint64_t address;
    int32_t priority;
} Gem5PfRequest;

typedef struct Gem5PfPrefetcher Gem5PfPrefetcher;

typedef struct
{
    uint32_t abi_version;

    Gem5PfPrefetcher *(*create)(const char *config, uint32_t block_size);
    void (*destroy)(Gem5PfPrefetcher *self);

    /*
     * The algorithm. Fill `out` with at most `max_out` requests and return how
     * many were written; returning more than max_out is a bug in the plugin
     * and gem5 refuses the run rather than reading past the buffer.
     *
     * Addresses need not be block-aligned and need not be in any order; gem5
     * still decides what is worth issuing.
     */
    size_t (*calculate)(Gem5PfPrefetcher *self,
                        const Gem5PfAccess *access,
                        Gem5PfRequest *out,
                        size_t max_out);
} Gem5PfApiV1;

#define GEM5_PF_ENTRY_SYMBOL "gem5_pf_api_v1"
typedef const Gem5PfApiV1 *(*Gem5PfEntryPoint)(void);

#ifdef __cplusplus
}
#endif

#endif /* GEM5_PF_PLUGIN_ABI_H */
