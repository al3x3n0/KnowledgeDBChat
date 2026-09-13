import io
base = "/src/gem5/src/mem/cache/replacement_policies/"

p = base + "ReplacementPolicies.py"
s = io.open(p).read()
if "PluginRP" not in s:
    s = s.rstrip() + '''


class PluginRP(BaseReplacementPolicy):
    type = "PluginRP"
    cxx_class = "gem5::replacement_policy::PluginPolicy"
    cxx_header = "mem/cache/replacement_policies/plugin_rp.hh"
    library = Param.String("", "Path to the plugin shared object")
    entry_symbol = Param.String(
        "gem5_rp_api_v1", "Symbol exporting the plugin's API table"
    )
    config = Param.String("", "Opaque configuration handed to the plugin")
'''
    io.open(p, "w").write(s)
    print("python decl added")
else:
    print("python decl already present")

p = base + "SConscript"
s = io.open(p).read()
if "PluginRP" not in s:
    a = "'ReuseProtectedRP'])"
    assert s.count(a) == 1, s.count(a)
    s = s.replace(a, "'ReuseProtectedRP', 'PluginRP'])", 1)
    c = "Source('bip_rp.cc')"
    assert s.count(c) == 1
    s = s.replace(c, c + "\nSource('plugin_rp.cc')", 1)
    io.open(p, "w").write(s)
    print("sconscript updated")
else:
    print("sconscript already updated")
