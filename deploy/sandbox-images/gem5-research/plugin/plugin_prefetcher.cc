#include "mem/cache/prefetch/plugin_prefetcher.hh"

#include <dlfcn.h>

#include "base/logging.hh"
#include "params/PluginPrefetcher.hh"

namespace gem5
{

namespace prefetch
{

PluginPrefetcher::PluginPrefetcher(const PluginPrefetcherParams &p)
  : Queued(p), handle(nullptr), api(nullptr), impl(nullptr),
    maxRequests(p.max_requests_per_access)
{
    fatal_if(p.library.empty(),
             "PluginPrefetcher needs a library to load; none was given.");
    fatal_if(maxRequests == 0,
             "PluginPrefetcher: max_requests_per_access is 0, so the plugin "
             "could never ask for anything and the run would measure no "
             "prefetcher while reporting one.");

    handle = dlopen(p.library.c_str(), RTLD_NOW | RTLD_LOCAL);
    fatal_if(handle == nullptr,
             "PluginPrefetcher could not load %s: %s", p.library, dlerror());

    dlerror();
    void *symbol = dlsym(handle, p.entry_symbol.c_str());
    const char *symbol_error = dlerror();
    fatal_if(symbol == nullptr || symbol_error != nullptr,
             "PluginPrefetcher found no %s in %s: %s. A plugin exports exactly "
             "one function of that name returning a const Gem5PfApiV1*, and it "
             "must be extern \"C\" or the name will be mangled out of reach.",
             p.entry_symbol, p.library,
             symbol_error ? symbol_error : "symbol is null");

    auto entry_point = reinterpret_cast<Gem5PfEntryPoint>(symbol);
    api = entry_point();
    fatal_if(api == nullptr, "PluginPrefetcher: %s returned no API table",
             p.entry_symbol);
    fatal_if(api->abi_version != GEM5_PF_ABI_VERSION,
             "PluginPrefetcher: %s was built against ABI version %u, this "
             "gem5 speaks version %u. Rebuild it against the current "
             "gem5_pf_plugin_abi.h.",
             p.library, api->abi_version, GEM5_PF_ABI_VERSION);
    fatal_if(!api->create || !api->destroy || !api->calculate,
             "PluginPrefetcher: %s left an entry in its API table null.",
             p.library);

    impl = api->create(p.config.c_str(), (uint32_t)blkSize);
    fatal_if(impl == nullptr,
             "PluginPrefetcher: %s refused its configuration %s",
             p.library, p.config.empty() ? "(none)" : p.config);

    scratch.resize(maxRequests);
}

PluginPrefetcher::~PluginPrefetcher()
{
    if (api && impl) {
        api->destroy(impl);
    }
    if (handle) {
        dlclose(handle);
    }
}

void
PluginPrefetcher::calculatePrefetch(const PrefetchInfo &pfi,
                                    std::vector<AddrPriority> &addresses,
                                    const CacheAccessor &cache)
{
    Gem5PfAccess access;
    access.address = pfi.getAddr();
    access.pc = pfi.hasPC() ? pfi.getPC() : 0;
    access.size = pfi.getSize();
    access.requestor_id = 0;
    access.has_pc = pfi.hasPC() ? 1 : 0;
    access.secure = pfi.isSecure() ? 1 : 0;
    access.write = pfi.isWrite() ? 1 : 0;
    access.cache_miss = pfi.isCacheMiss() ? 1 : 0;

    size_t produced = api->calculate(impl, &access, scratch.data(), maxRequests);

    // Reading past the buffer would be memory corruption inside the simulator,
    // and the run would carry on producing numbers afterwards.
    fatal_if(produced > maxRequests,
             "PluginPrefetcher: the plugin reported %llu requests for a buffer "
             "of %llu.",
             (unsigned long long)produced, (unsigned long long)maxRequests);

    for (size_t i = 0; i < produced; ++i) {
        addresses.push_back(
            AddrPriority(scratch[i].address, scratch[i].priority));
    }
}

} // namespace prefetch
} // namespace gem5
