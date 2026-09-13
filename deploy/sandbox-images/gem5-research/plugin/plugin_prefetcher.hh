/*
 * A prefetcher loaded from a shared object at run time.
 *
 * Derives from Queued, not from Base: the queue, the throttling and the
 * page-crossing checks stay gem5's. A prefetcher that reimplements those
 * measures the harness -- at gem5's default L1 mshrs=4 a stride prefetcher
 * issues 35 of 503,959 identified candidates and reads as no prefetcher at
 * all, which is a property of the queue and not of any algorithm.
 */
#ifndef __MEM_CACHE_PREFETCH_PLUGIN_PREFETCHER_HH__
#define __MEM_CACHE_PREFETCH_PLUGIN_PREFETCHER_HH__

#include <string>
#include <vector>

#include "mem/cache/prefetch/gem5_pf_plugin_abi.h"
#include "mem/cache/prefetch/queued.hh"
#include "mem/packet.hh"

namespace gem5
{

struct PluginPrefetcherParams;

namespace prefetch
{

class PluginPrefetcher : public Queued
{
  protected:
    void *handle;
    const Gem5PfApiV1 *api;
    Gem5PfPrefetcher *impl;

    /** Most requests one access may produce. */
    const size_t maxRequests;

    /** Reused across calls so the hot path does not allocate. */
    mutable std::vector<Gem5PfRequest> scratch;

  public:
    PluginPrefetcher(const PluginPrefetcherParams &p);
    ~PluginPrefetcher();

    void calculatePrefetch(const PrefetchInfo &pfi,
                           std::vector<AddrPriority> &addresses,
                           const CacheAccessor &cache) override;
};

} // namespace prefetch
} // namespace gem5

#endif // __MEM_CACHE_PREFETCH_PLUGIN_PREFETCHER_HH__
