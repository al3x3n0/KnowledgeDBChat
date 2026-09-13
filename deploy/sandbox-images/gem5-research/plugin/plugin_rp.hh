/*
 * A replacement policy loaded from a shared object at run time.
 *
 * gem5 has no plugin interface: a mechanism is compiled in, and on this
 * machine adding one costs 7m49s of rebuild and cannot be done while the
 * application stack is running, because the final link needs more memory than
 * the Docker VM has left. This SimObject is compiled in ONCE and turns every
 * later mechanism into a `g++ -shared` of a few seconds, with no link step and
 * therefore no memory wall.
 *
 * It forwards to a plugin across a C ABI that mentions no gem5 type, so the
 * plugin needs this project's ABI header and nothing else.
 */
#ifndef __MEM_CACHE_REPLACEMENT_POLICIES_PLUGIN_RP_HH__
#define __MEM_CACHE_REPLACEMENT_POLICIES_PLUGIN_RP_HH__

#include <string>

#include "base/types.hh"
#include "mem/cache/replacement_policies/base.hh"
#include "mem/cache/replacement_policies/gem5_rp_plugin_abi.h"

namespace gem5
{

struct PluginRPParams;

namespace replacement_policy
{

class PluginPolicy : public Base
{
  protected:
    /** gem5 owns the storage the plugin reads and writes. */
    struct PluginReplData : ReplacementData
    {
        Gem5RpEntry entry;
        PluginReplData() : entry() {}
    };

    /** dlopen handle, kept so the library outlives every entry. */
    void *handle;

    /** The plugin's function table and its instance. */
    const Gem5RpApiV1 *api;
    Gem5RpPolicy *impl;

  public:
    typedef PluginRPParams Params;
    PluginPolicy(const Params &p);
    ~PluginPolicy();

    void invalidate(const std::shared_ptr<ReplacementData>& replacement_data)
                                                                    override;
    void touch(const std::shared_ptr<ReplacementData>& replacement_data) const
                                                                     override;
    void reset(const std::shared_ptr<ReplacementData>& replacement_data) const
                                                                     override;
    ReplaceableEntry* getVictim(const ReplacementCandidates& candidates) const
                                                                     override;
    std::shared_ptr<ReplacementData> instantiateEntry() override;
};

} // namespace replacement_policy
} // namespace gem5

#endif // __MEM_CACHE_REPLACEMENT_POLICIES_PLUGIN_RP_HH__
