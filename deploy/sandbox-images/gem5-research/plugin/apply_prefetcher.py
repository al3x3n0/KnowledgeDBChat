import io
base = "/src/gem5/src/mem/cache/prefetch/"

p = base + "Prefetcher.py"
s = io.open(p).read()
if "PluginPrefetcher" not in s:
    s = s.rstrip() + '''


class PluginPrefetcher(QueuedPrefetcher):
    type = "PluginPrefetcher"
    cxx_class = "gem5::prefetch::PluginPrefetcher"
    cxx_header = "mem/cache/prefetch/plugin_prefetcher.hh"
    library = Param.String("", "Path to the plugin shared object")
    entry_symbol = Param.String(
        "gem5_pf_api_v1", "Symbol exporting the plugin's API table"
    )
    config = Param.String("", "Opaque configuration handed to the plugin")
    max_requests_per_access = Param.Unsigned(
        8, "Most prefetch requests one access may produce"
    )
'''
    io.open(p, "w").write(s)
    print("python decl added")
else:
    print("python decl already present")

p = base + "SConscript"
s = io.open(p).read()
if "PluginPrefetcher" not in s:
    a = "Source('queued.cc')"
    assert s.count(a) == 1, f"Source anchor count {s.count(a)}"
    s = s.replace(a, a + "\nSource('plugin_prefetcher.cc')", 1)
    marker = "'TaggedPrefetcher'"
    assert s.count(marker) >= 1, "sim_objects anchor missing"
    s = s.replace(marker, marker + ", 'PluginPrefetcher'", 1)
    io.open(p, "w").write(s)
    print("sconscript updated")
else:
    print("sconscript already updated")
