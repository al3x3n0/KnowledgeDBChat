"""Emit the Cargo manifest and extern list from the toolchain registry.

The crate set has two consumers -- this image, which builds the rlibs, and
`agent_toolchains.RUST_CRATES`, which is what the model is told it may use --
and they must not drift. Being told about a crate the image does not carry is
worse than not being told at all: the run writes code against it and gets an
unresolved-import error that looks like its own mistake.

Keeping two lists in step by hand does not work, and a test that compares them
is worse than it looks: it can only run where both files are present, so in the
usual backend-only test container it skips and guards nothing. So there is one
list, in the Python module, and the image derives from it here.

`agent_toolchains` imports nothing but the standard library precisely so this
can load it without the application around it.
"""

import argparse
import importlib.util
import sys


def load_toolchains(path):
    spec = importlib.util.spec_from_file_location("agent_toolchains", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: @dataclass resolves its own module out of
    # sys.modules, and without this the import dies inside dataclasses with an
    # AttributeError that says nothing about the cause.
    sys.modules["agent_toolchains"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("toolchains", help="path to agent_toolchains.py")
    parser.add_argument(
        "--emit", choices=("manifest", "extern-names"), default="manifest"
    )
    args = parser.parse_args()
    crates = load_toolchains(args.toolchains).RUST_CRATES
    if not crates:
        raise SystemExit("RUST_CRATES is empty; refusing to build an empty crate set")

    if args.emit == "extern-names":
        # The direct crates only. Transitive dependencies are reachable through
        # -L dependency, but putting them in scope would invite use of an API
        # the model was never told about and that no version pin covers.
        print(" ".join(crate.extern_name for crate in crates))
        return

    print("[package]")
    print('name = "sandbox-crateset"')
    print('version = "0.0.0"')
    print('edition = "2021"')
    print()
    print("[dependencies]")
    for crate in crates:
        # Pinned exactly: a caret range makes the dependency graph a function
        # of the image build date, which is the opposite of what a
        # reproduction study needs.
        print(f'{crate.name} = "={crate.version}"')


if __name__ == "__main__":
    main()
