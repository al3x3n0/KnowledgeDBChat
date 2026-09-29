"""Symbol definitions from source files, each language through its own parser.

`RepoSymbolIndexService` answers "where is X defined?". A regex cannot answer
that for any language worth asking about: it cannot tell a definition from a
prototype split across lines, a macro that expands to a function, a method
inside a class, or the end of a body. The index used to read only Python and,
by regex, JS/TS -- and for a C file it read nothing and reported "not found",
which a run took as a fact about raylib's rtextures.c.

So each language is read by a real parser:

  Python      `ast`, the interpreter's own.
  C / C++     libclang (`clang.cindex`), the compiler's own front end. It
              knows a definition from a declaration, sees through macros and
              templates, and reports exact extents. Its error recovery still
              yields every definition when headers are missing: on raylib's
              rtextures.c, 108 definitions with or without include paths.
  JS / TS     tree-sitter with the JavaScript and TypeScript grammars.

Nothing falls back to a regex. libclang is ESSENTIAL: if it cannot be
loaded, `LibclangMissing` is raised where it is first needed, and the image
build asserts it parses. The tree-sitter grammars are reported as unreadable
when missing, so a caller is told "cannot read this file" rather than "the
symbol is not there".

Each result is (name, kind, start_line, end_line), 1-based and inclusive.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

Symbol = Tuple[str, str, int, int]

PYTHON_EXTS = frozenset({".py"})
C_EXTS = frozenset({".c"})
CPP_EXTS = frozenset({".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx"})
#: `.h` is either language; the parser is chosen from the file's contents.
HEADER_EXTS = frozenset({".h"})
JS_EXTS = frozenset({".js", ".jsx", ".mjs", ".cjs"})
TS_EXTS = frozenset({".ts"})
TSX_EXTS = frozenset({".tsx"})

CLANG_EXTS = C_EXTS | CPP_EXTS | HEADER_EXTS
TREE_SITTER_EXTS = JS_EXTS | TS_EXTS | TSX_EXTS
ALL_EXTS = PYTHON_EXTS | CLANG_EXTS | TREE_SITTER_EXTS


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


class LibclangMissing(RuntimeError):
    """libclang is essential here; its absence is a broken install."""


@lru_cache(maxsize=1)
def _clang_index():
    # Not optional. A missing libclang used to make C/C++ files "unreadable",
    # which quietly turned symbol lookup off for the languages this project's
    # optimisation work is about. It is a hard dependency, and a broken install
    # says so where it is first needed.
    try:
        from clang import cindex

        # The bindings import without the shared library they drive; only
        # creating an index proves libclang is actually loadable.
        return cindex.Index.create()
    except Exception as exc:
        raise LibclangMissing(
            "libclang is required for C/C++ symbol lookup and could not be "
            f"loaded ({exc.__class__.__name__}: {exc}). Install the pinned "
            "`libclang` from backend/requirements.txt, or rebuild the image."
        ) from exc


@lru_cache(maxsize=4)
def _ts_parser(kind: str):
    try:
        from tree_sitter import Language, Parser

        if kind == "js":
            import tree_sitter_javascript as grammar

            language = Language(grammar.language())
        else:
            import tree_sitter_typescript as grammar

            language = Language(
                grammar.language_tsx()
                if kind == "tsx"
                else grammar.language_typescript()
            )
        return Parser(language)
    except Exception:
        return None


def _release_clang_index() -> None:
    # libclang's Index.__del__ calls into the library; at interpreter exit
    # the module holding it may already be gone, which prints a traceback.
    _clang_index.cache_clear()


import atexit  # noqa: E402

atexit.register(_release_clang_index)


def parser_available(path: str) -> bool:
    """Whether a real parser for this file's language is installed."""
    ext = Path(path).suffix.lower()
    if ext in PYTHON_EXTS:
        return True
    if ext in CLANG_EXTS:
        _clang_index()  # raises LibclangMissing rather than answering False
        return True
    if ext in TREE_SITTER_EXTS:
        return _ts_parser(_ts_kind(ext)) is not None
    return False


def _ts_kind(ext: str) -> str:
    return "tsx" if ext in TSX_EXTS else "ts" if ext in TS_EXTS else "js"


# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #


def python_symbols(text: str) -> List[Symbol]:
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: List[Symbol] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "function"
        elif isinstance(node, ast.ClassDef):
            kind = "class"
        else:
            continue
        start = int(getattr(node, "lineno", 1) or 1)
        out.append(
            (node.name, kind, start, int(getattr(node, "end_lineno", start) or start))
        )
    return out


# --------------------------------------------------------------------------- #
# C / C++ through libclang
# --------------------------------------------------------------------------- #

_CPP_MARKERS = ("class ", "namespace ", "template<", "template <", "::", "public:")


def _clang_args(path: Path, repo_root: Optional[Path], text: str) -> List[str]:
    ext = path.suffix.lower()
    is_cpp = ext in CPP_EXTS or (
        ext in HEADER_EXTS and any(marker in text for marker in _CPP_MARKERS)
    )
    if is_cpp:
        args = [
            "-x",
            "c++-header" if ext in HEADER_EXTS | {".hh", ".hpp", ".hxx"} else "c++",
        ]
        args.append("-std=c++17")
    else:
        args = ["-x", "c-header" if ext in HEADER_EXTS else "c", "-std=gnu11"]
    # Where headers usually are. Missing ones cost accuracy only inside
    # expressions: clang's recovery keeps every definition regardless.
    include_dirs = [path.parent]
    if repo_root is not None:
        include_dirs += [repo_root, repo_root / "include", repo_root / "src"]
    args += [f"-I{d}" for d in include_dirs if d.is_dir()]
    return args


def clang_symbols(
    path: Path, text: str, repo_root: Optional[Path] = None
) -> List[Symbol]:
    from clang import cindex

    index = _clang_index()
    try:
        tu = index.parse(
            str(path),
            args=_clang_args(path, repo_root, text),
            unsaved_files=[(str(path), text)],
            options=cindex.TranslationUnit.PARSE_INCOMPLETE,
        )
    except cindex.TranslationUnitLoadError:
        return []

    K = cindex.CursorKind
    functions = {
        K.FUNCTION_DECL,
        K.CXX_METHOD,
        K.CONSTRUCTOR,
        K.DESTRUCTOR,
        K.CONVERSION_FUNCTION,
        K.FUNCTION_TEMPLATE,
    }
    classes = {K.CLASS_DECL, K.STRUCT_DECL, K.UNION_DECL, K.CLASS_TEMPLATE}
    scopes = {K.NAMESPACE, K.LINKAGE_SPEC} | classes
    here = os.path.realpath(str(path))

    out: List[Symbol] = []

    def visit(cursor) -> None:
        for child in cursor.get_children():
            location = child.location
            if location.file is None or os.path.realpath(location.file.name) != here:
                continue  # declared in an included header, not in this file
            kind = child.kind
            if kind in functions and child.is_definition():
                out.append(
                    (
                        child.spelling,
                        "function",
                        child.extent.start.line,
                        child.extent.end.line,
                    )
                )
            elif kind in classes and child.is_definition() and child.spelling:
                out.append(
                    (
                        child.spelling,
                        "class",
                        child.extent.start.line,
                        child.extent.end.line,
                    )
                )
            if kind in scopes:
                visit(child)  # methods defined inside a class or namespace

    visit(tu.cursor)
    return out


# --------------------------------------------------------------------------- #
# JavaScript / TypeScript through tree-sitter
# --------------------------------------------------------------------------- #

_TS_FUNCTIONS = {
    "function_declaration",
    "generator_function_declaration",
    "method_definition",
}
_TS_CLASSES = {
    "class_declaration",
    "abstract_class_declaration",
    "interface_declaration",
}
_TS_FUNCTION_VALUES = {"arrow_function", "function_expression", "function"}
_TEST_CALLS = {"test", "it"}


def tree_sitter_symbols(path: Path, text: str) -> List[Symbol]:
    parser = _ts_parser(_ts_kind(path.suffix.lower()))
    if parser is None:
        return []
    source = text.encode("utf-8", errors="replace")
    tree = parser.parse(source)

    def name_of(node) -> str:
        named = node.child_by_field_name("name")
        return (
            source[named.start_byte : named.end_byte].decode("utf-8", "replace")
            if named
            else ""
        )

    def lines(node) -> Tuple[int, int]:
        return node.start_point[0] + 1, node.end_point[0] + 1

    out: List[Symbol] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        kind = node.type
        if kind in _TS_FUNCTIONS or kind in _TS_CLASSES:
            name = name_of(node)
            if name:
                out.append(
                    (
                        name,
                        "function" if kind in _TS_FUNCTIONS else "class",
                        *lines(node),
                    )
                )
        elif kind == "variable_declarator":
            value = node.child_by_field_name("value")
            if value is not None and value.type in _TS_FUNCTION_VALUES:
                name = name_of(node)
                if name:
                    out.append((name, "function", *lines(node)))
        elif kind == "call_expression":
            callee = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            callee_name = (
                source[callee.start_byte : callee.end_byte].decode("utf-8", "replace")
                if callee is not None
                else ""
            )
            if (
                callee_name in _TEST_CALLS
                and args is not None
                and args.named_child_count
            ):
                first = args.named_children[0]
                if first.type in ("string", "template_string"):
                    title = source[first.start_byte : first.end_byte].decode(
                        "utf-8", "replace"
                    )
                    out.append((title.strip("'\"`"), "test", *lines(node)))
        stack.extend(reversed(node.children))
    return out


# --------------------------------------------------------------------------- #
# Dispatch, cached per file version
# --------------------------------------------------------------------------- #

_CACHE: dict = {}
_CACHE_LIMIT = 4096


def symbols_in(path: Path, repo_root: Optional[Path] = None) -> List[Symbol]:
    """Every definition in one file, parsed once per version of the file.

    The index re-scans a repository on every call, and a real parse is not
    free (libclang: ~0.5 s for a 200 KB C file), so results are cached by
    path, modification time and size.
    """
    try:
        stat = path.stat()
    except OSError:
        return []
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    ext = path.suffix.lower()
    if ext in PYTHON_EXTS:
        result = python_symbols(text)
    elif ext in CLANG_EXTS:
        result = clang_symbols(path, text, repo_root)
    elif ext in TREE_SITTER_EXTS:
        result = tree_sitter_symbols(path, text)
    else:
        result = []
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.clear()
    _CACHE[key] = result
    return result
