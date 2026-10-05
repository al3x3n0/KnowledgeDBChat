"""C symbols and single-file searches: two silent "not found"s from one run.

A run asked get_symbol_context for ImageColorTint in raylib's rtextures.c and
was told it did not exist (the index read only Python and JS), then asked
search_code with path='src/rtextures.c' and got 0 matches (rglob on a file
yields nothing). Both answers read as facts about the file.
"""

from pathlib import Path

from app.services.coding_workspace_manager import (
    CodingWorkspace,
    CodingWorkspaceManager,
)
from app.services.repo_symbol_index_service import RepoSymbolIndexService

C_SOURCE = """#include "raylib.h"

// A prototype, not a definition
void ImageColorTint(Image *image, Color color);

static inline int clampi(int v) { return v < 0 ? 0 : v; }

void ImageColorTint(Image *image, Color color)
{
    if (image == 0) return;
    for (int i = 0; i < 4; i++)
    {
        helper(i);
    }
}

Image GenImageColor(int width, int height,
                    Color color)
{
    Image image = { 0 };
    return image;
}
"""


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "rtextures.c").write_text(C_SOURCE)
    return tmp_path


def test_c_function_definitions_are_found_with_their_extent(tmp_path):
    root = _repo(tmp_path)
    svc = RepoSymbolIndexService()
    found = {
        m["symbol"]: (m["start_line"], m["end_line"])
        for sym in ("ImageColorTint", "GenImageColor", "clampi")
        for m in svc.retrieve(
            repo_root=root, query_keywords=[sym], include_paths=["src/rtextures.c"]
        )["symbol_matches"]
        if m["symbol"] == sym
    }
    assert found["ImageColorTint"] == (
        8,
        15,
    )  # the definition, not the prototype on line 4
    assert found["GenImageColor"] == (17, 22)  # a parameter list split over two lines
    assert found["clampi"][0] == 6


def test_calls_inside_bodies_are_not_symbols(tmp_path):
    root = _repo(tmp_path)
    rows = RepoSymbolIndexService().retrieve(
        repo_root=root, query_keywords=["helper"], include_paths=["src/rtextures.c"]
    )["symbol_matches"]
    assert all(r["symbol"] != "helper" for r in rows)


def test_the_index_says_what_it_cannot_read():
    assert RepoSymbolIndexService.reads("src/rtextures.c")
    assert not RepoSymbolIndexService.reads("src/shader.glsl")


def test_search_code_scoped_to_one_file_searches_that_file(tmp_path):
    root = _repo(tmp_path)
    ws = CodingWorkspace(workspace_id="w", base_path=root)
    hits = CodingWorkspaceManager().search_code(
        ws, r"void ImageColorTint", path="src/rtextures.c"
    )
    assert [h["line"] for h in hits] == [4, 8]
