#include "mem/cache/replacement_policies/plugin_rp.hh"

#include <dlfcn.h>

#include <cassert>
#include <memory>
#include <vector>

#include "base/logging.hh"
#include "params/PluginRP.hh"
#include "sim/cur_tick.hh"

namespace gem5
{

namespace replacement_policy
{

PluginPolicy::PluginPolicy(const Params &p)
  : Base(p), handle(nullptr), api(nullptr), impl(nullptr)
{
    fatal_if(p.library.empty(),
             "PluginRP needs a library to load; none was given.");

    // RTLD_NOW so an unresolved symbol is a load-time error naming the symbol,
    // rather than a crash mid-simulation with a stack that says nothing.
    handle = dlopen(p.library.c_str(), RTLD_NOW | RTLD_LOCAL);
    fatal_if(handle == nullptr,
             "PluginRP could not load %s: %s", p.library, dlerror());

    dlerror();
    void *symbol = dlsym(handle, p.entry_symbol.c_str());
    const char *symbol_error = dlerror();
    fatal_if(symbol == nullptr || symbol_error != nullptr,
             "PluginRP found no %s in %s: %s. A plugin exports exactly one "
             "function of that name returning a const Gem5RpApiV1*.",
             p.entry_symbol, p.library,
             symbol_error ? symbol_error : "symbol is null");

    auto entry_point = reinterpret_cast<Gem5RpEntryPoint>(symbol);
    api = entry_point();
    fatal_if(api == nullptr, "PluginRP: %s returned no API table",
             p.entry_symbol);
    fatal_if(api->abi_version != GEM5_RP_ABI_VERSION,
             "PluginRP: %s was built against ABI version %u, this gem5 "
             "speaks version %u. Rebuild the plugin against the current "
             "gem5_rp_plugin_abi.h.",
             p.library, api->abi_version, GEM5_RP_ABI_VERSION);
    fatal_if(!api->create || !api->destroy || !api->invalidate || !api->touch
             || !api->reset || !api->get_victim,
             "PluginRP: %s left an entry in its API table null. A missing "
             "hook would be called as a null pointer mid-simulation.",
             p.library);

    impl = api->create(p.config.c_str());
    fatal_if(impl == nullptr,
             "PluginRP: %s refused its configuration %s",
             p.library, p.config.empty() ? "(none)" : p.config);
}

PluginPolicy::~PluginPolicy()
{
    if (api && impl) {
        api->destroy(impl);
    }
    if (handle) {
        dlclose(handle);
    }
}

void
PluginPolicy::invalidate(
    const std::shared_ptr<ReplacementData>& replacement_data)
{
    auto data = std::static_pointer_cast<PluginReplData>(replacement_data);
    api->invalidate(impl, &data->entry);
}

void
PluginPolicy::touch(
    const std::shared_ptr<ReplacementData>& replacement_data) const
{
    auto data = std::static_pointer_cast<PluginReplData>(replacement_data);
    api->touch(impl, &data->entry, static_cast<uint64_t>(curTick()));
}

void
PluginPolicy::reset(
    const std::shared_ptr<ReplacementData>& replacement_data) const
{
    auto data = std::static_pointer_cast<PluginReplData>(replacement_data);
    api->reset(impl, &data->entry, static_cast<uint64_t>(curTick()));
}

ReplaceableEntry*
PluginPolicy::getVictim(const ReplacementCandidates& candidates) const
{
    assert(candidates.size() > 0);

    std::vector<Gem5RpEntry *> entries;
    entries.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        entries.push_back(&std::static_pointer_cast<PluginReplData>(
            candidate->replacementData)->entry);
    }

    size_t victim = api->get_victim(impl, entries.data(), entries.size());

    // An out-of-range answer is a plugin bug, and evicting something arbitrary
    // instead would make it a silently wrong measurement rather than an error.
    fatal_if(victim >= candidates.size(),
             "PluginRP: the plugin chose victim %llu of %llu candidates.",
             (unsigned long long)victim,
             (unsigned long long)candidates.size());

    return candidates[victim];
}

std::shared_ptr<ReplacementData>
PluginPolicy::instantiateEntry()
{
    return std::shared_ptr<ReplacementData>(new PluginReplData());
}

} // namespace replacement_policy
} // namespace gem5
