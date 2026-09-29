"""Each language through its own parser, and never a pattern in its place.

What these pin is what a regex gets wrong: C++ methods inside a class and a
namespace, a template, a function a macro expands to, TypeScript interfaces
and methods, and a test() call's title.
"""

from pathlib import Path

import pytest

from app.services import repo_symbol_parsers as parsers
from app.services.repo_symbol_index_service import RepoSymbolIndexService


def _names(path: Path):
    return {
        (name, kind): (start, end)
        for name, kind, start, end in parsers.symbols_in(path)
    }


CPP = """#include <vector>
namespace gfx {
class Image {
public:
    Image(int w, int h);
    int width() const { return w_; }
    void tint(int r, int g, int b);
private:
    int w_;
};

Image::Image(int w, int h) : w_(w) {}

void Image::tint(int r, int g, int b)
{
    (void)r; (void)g; (void)b;
}

template <typename T>
T clampv(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }
}  // namespace gfx
"""

C_MACRO = """#define DEFINE_GETTER(name) int get_##name(void) { return 1; }
DEFINE_GETTER(width)
int prototype_only(int x);
"""

TS = """export interface Tint { r: number; g: number }
export class Painter {
  apply(t: Tint): void {
    return;
  }
}
export const blend = (a: number, b: number): number => a + b;
function helper() {}
test('blends two colours', () => {
  expect(blend(1, 2)).toBe(3);
});
"""


@pytest.fixture(autouse=True)
def _fresh_cache():
    parsers._CACHE.clear()


def test_cpp_methods_constructors_and_templates(tmp_path):
    path = tmp_path / "image.cpp"
    path.write_text(CPP)
    found = _names(path)
    assert ("Image", "class") in found
    assert found[("width", "function")] == (6, 6)  # defined inside the class
    assert found[("tint", "function")] == (14, 17)  # out-of-line member definition
    assert ("clampv", "function") in found  # a template
    # The in-class DECLARATIONS of the constructor and tint are not
    # definitions; only the out-of-line ones are.
    assert sum(1 for (n, k) in found if n == "tint") == 1


def test_a_macro_expansion_is_a_definition_and_a_prototype_is_not(tmp_path):
    path = tmp_path / "getters.c"
    path.write_text(C_MACRO)
    found = _names(path)
    assert ("get_width", "function") in found
    assert ("prototype_only", "function") not in found


def test_typescript_interfaces_classes_methods_arrows_and_tests(tmp_path):
    path = tmp_path / "painter.ts"
    path.write_text(TS)
    found = _names(path)
    assert ("Tint", "class") in found
    assert found[("Painter", "class")] == (2, 6)
    assert found[("apply", "function")] == (3, 5)
    assert ("blend", "function") in found
    assert ("helper", "function") in found
    assert ("blends two colours", "test") in found


def test_no_parser_means_unreadable_not_empty(tmp_path, monkeypatch):
    """A missing parser is reported, never replaced by a pattern."""
    path = tmp_path / "src" / "a.c"
    path.parent.mkdir()
    path.write_text("int visible(void) { return 0; }\n")
    monkeypatch.setattr(parsers, "_clang_index", lambda: None)
    assert not RepoSymbolIndexService.reads("src/a.c")
    result = RepoSymbolIndexService().retrieve(
        repo_root=tmp_path, query_keywords=["visible"], include_paths=["src/a.c"]
    )
    assert result["symbol_scan_files"] == 0


def test_a_parse_is_cached_per_file_version(tmp_path, monkeypatch):
    path = tmp_path / "k.c"
    path.write_text("int one(void) { return 1; }\n")
    calls = []
    real = parsers.clang_symbols

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(parsers, "clang_symbols", counting)
    parsers.symbols_in(path)
    parsers.symbols_in(path)
    assert len(calls) == 1
    path.write_text("int one(void) { return 1; }\nint two(void) { return 2; }\n")
    assert ("two", "function", 2, 2) in parsers.symbols_in(path)
    assert len(calls) == 2
