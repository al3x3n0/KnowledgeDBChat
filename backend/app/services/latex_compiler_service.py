"""
LaTeX compilation service for LaTeX Studio.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class LatexSafetyError(ValueError):
    def __init__(self, message: str, violations: Optional[List[str]] = None):
        super().__init__(message)
        self.violations = violations or []


@dataclass(frozen=True)
class LatexCompileResult:
    success: bool
    engine: Optional[str]
    pdf_bytes: Optional[bytes]
    log: str
    violations: List[str]


_FORBIDDEN_PATTERNS: List[Tuple[str, str]] = [
    (r"\\write18\b", "Disallowed: \\write18"),
    (r"\\(?:openin|openout|read|write)\b", "Disallowed: low-level I/O (\\openin/\\openout/\\read/\\write)"),
]

_INCLUDE_COMMANDS = ("input", "include", "includegraphics", "bibliography", "addbibresource")


def _check_safe_mode(tex_source: str) -> List[str]:
    violations: List[str] = []
    for pattern, reason in _FORBIDDEN_PATTERNS:
        if re.search(pattern, tex_source):
            violations.append(reason)
    return violations


_INCLUDE_RE = re.compile(
    r"\\(?P<cmd>input|include|includegraphics|bibliography|addbibresource)\s*"
    r"(?:\[[^\]]*\]\s*)?"
    r"\{(?P<arg>[^}]+)\}",
    re.IGNORECASE,
)


def _is_safe_relative_path(name: str) -> bool:
    s = (name or "").strip()
    if not s:
        return False
    if s.startswith(("/", "~")):
        return False
    if re.match(r"^[a-zA-Z]:", s):
        return False
    if ".." in s.replace("\\", "/").split("/"):
        return False
    if "|" in s:
        return False
    if "\\" in s:
        return False
    return True


def _possible_graphics_names(base: str) -> List[str]:
    b = base.strip()
    if not b:
        return []
    if "." in Path(b).name:
        return [b]
    # Common extensions supported by pdflatex/tectonic.
    exts = [".pdf", ".png", ".jpg", ".jpeg", ".eps"]
    return [f"{b}{ext}" for ext in exts]


def _normalize_tex_input_name(name: str) -> str:
    n = name.strip()
    if not n:
        return n
    if "." in Path(n).name:
        return n
    return f"{n}.tex"


def _extract_includes(tex_source: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for m in _INCLUDE_RE.finditer(tex_source or ""):
        cmd = (m.group("cmd") or "").strip().lower()
        arg = (m.group("arg") or "").strip()
        pairs.append((cmd, arg))
    return pairs


def _check_includes_allowed(
    tex_source: str,
    *,
    allowed_files: Iterable[str],
) -> List[str]:
    allowed_set = {str(x) for x in (allowed_files or [])}
    violations: List[str] = []
    for cmd, arg in _extract_includes(tex_source):
        # Some commands accept comma-separated lists.
        parts = [p.strip() for p in arg.split(",") if p.strip()] if cmd in ("bibliography",) else [arg]
        for part in parts:
            if not _is_safe_relative_path(part):
                violations.append(f"Disallowed path in \\{cmd}{{...}}: {part}")
                continue
            if cmd in ("input", "include"):
                target = _normalize_tex_input_name(part)
                if target not in allowed_set:
                    violations.append(f"Missing project file for \\{cmd}{{{part}}} (expected {target})")
            elif cmd == "includegraphics":
                candidates = _possible_graphics_names(part)
                if not candidates or not any(c in allowed_set for c in candidates):
                    violations.append(f"Missing project image for \\includegraphics{{{part}}}")
            elif cmd in ("bibliography", "addbibresource"):
                target = part
                if cmd == "bibliography" and "." not in Path(target).name:
                    target = f"{target}.bib"
                if target not in allowed_set:
                    violations.append(f"Missing project bib file for \\{cmd}{{{part}}}")
    return violations


class LatexCompilerService:
    def available_engines(self) -> Dict[str, bool]:
        return {
            "tectonic": shutil.which("tectonic") is not None,
            "pdflatex": shutil.which("pdflatex") is not None,
        }

    def available_tools(self) -> Dict[str, bool]:
        return {
            "bibtex": shutil.which("bibtex") is not None,
        }

    def check_safe_mode(self, tex_source: str) -> List[str]:
        return _check_safe_mode(tex_source or "")

    def _run(
        self,
        cmd: List[str],
        *,
        cwd: Path,
        timeout_seconds: int,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, str]:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
        return int(proc.returncode), proc.stdout or ""

    def compile_to_pdf(
        self,
        *,
        tex_source: str,
        timeout_seconds: int,
        max_source_chars: int,
        safe_mode: bool = True,
        preferred_engine: Optional[str] = None,
        additional_files: Optional[Dict[str, bytes]] = None,
    ) -> LatexCompileResult:
        tex_source = (tex_source or "").replace("\r\n", "\n")
        if not tex_source.strip():
            return LatexCompileResult(
                success=False,
                engine=None,
                pdf_bytes=None,
                log="Empty LaTeX source.",
                violations=[],
            )

        if len(tex_source) > max_source_chars:
            return LatexCompileResult(
                success=False,
                engine=None,
                pdf_bytes=None,
                log=f"LaTeX source too large ({len(tex_source)} chars; max {max_source_chars}).",
                violations=[],
            )

        violations: List[str] = []
        if safe_mode:
            violations = _check_safe_mode(tex_source)
            if violations:
                raise LatexSafetyError("Unsafe LaTeX detected in safe_mode.", violations=violations)

        additional_files = additional_files or {}
        if safe_mode:
            # In safe_mode, disallow including any external/local files unless they are provided
            # explicitly as project files (additional_files).
            include_violations = _check_includes_allowed(
                tex_source,
                allowed_files=additional_files.keys(),
            )
            # Also scan included .tex files for forbidden primitives.
            for name, data in additional_files.items():
                if not name.lower().endswith(".tex"):
                    continue
                try:
                    included_text = (data or b"").decode("utf-8", errors="replace")
                except Exception:
                    included_text = ""
                include_violations.extend(_check_safe_mode(included_text))

            if include_violations:
                raise LatexSafetyError("Unsafe or missing project file references detected in safe_mode.", violations=include_violations)

        engines = self.available_engines()
        engine = preferred_engine or ("tectonic" if engines.get("tectonic") else "pdflatex" if engines.get("pdflatex") else None)
        if engine not in ("tectonic", "pdflatex"):
            return LatexCompileResult(
                success=False,
                engine=None,
                pdf_bytes=None,
                log="No LaTeX compiler available on server. Install `tectonic` or `pdflatex`.",
                violations=violations,
            )

        with tempfile.TemporaryDirectory(prefix="latex_studio_") as tmp:
            tmp_path = Path(tmp)
            out_dir = tmp_path / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            main_tex = tmp_path / "main.tex"
            main_tex.write_text(tex_source + ("\n" if not tex_source.endswith("\n") else ""), encoding="utf-8")

            # Write additional project files (no subdirectories for now).
            for name, data in additional_files.items():
                safe_name = (name or "").strip()
                if not safe_name or "/" in safe_name or "\\" in safe_name:
                    continue
                try:
                    (tmp_path / safe_name).write_bytes(data or b"")
                except Exception:
                    continue
                try:
                    (out_dir / safe_name).write_bytes(data or b"")
                except Exception:
                    pass

            if engine == "tectonic":
                cmd = ["tectonic", "--outdir", str(out_dir), str(main_tex.name)]
                rc, output = self._run(cmd, cwd=tmp_path, timeout_seconds=timeout_seconds)
                pdf_path = out_dir / "main.pdf"
            else:
                cmd = [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-file-line-error",
                    "-no-shell-escape",
                    f"-output-directory={out_dir}",
                    str(main_tex.name),
                ]
                rc, output = self._run(cmd, cwd=tmp_path, timeout_seconds=timeout_seconds)
                pdf_path = out_dir / "main.pdf"

            log_parts = [output.strip()]

            # Optional: BibTeX passes for pdflatex when requested and available.
            if engine == "pdflatex":
                try:
                    from app.core.config import settings
                except Exception:
                    settings = None

                want_bibtex = True
                if settings is not None:
                    try:
                        want_bibtex = bool(getattr(settings, "LATEX_COMPILER_RUN_BIBTEX", True))
                    except Exception:
                        want_bibtex = True

                if want_bibtex and shutil.which("bibtex"):
                    aux_path = out_dir / "main.aux"
                    needs_bib = False
                    try:
                        if re.search(r"\\bibliography\\s*\\{", tex_source):
                            needs_bib = True
                        elif aux_path.exists():
                            aux_text = aux_path.read_text(encoding="utf-8", errors="replace")
                            if "\\bibdata" in aux_text or "\\citation" in aux_text:
                                needs_bib = True
                    except Exception:
                        needs_bib = False

                    if needs_bib and rc == 0:
                        env = dict(os.environ or {})
                        # Help bibtex find .bib files stored alongside the project.
                        env.setdefault("BIBINPUTS", f"{tmp_path}:{out_dir}:")
                        bib_rc, bib_out = self._run(["bibtex", "main"], cwd=out_dir, timeout_seconds=timeout_seconds, env=env)
                        log_parts.append((bib_out or "").strip())
                        if bib_rc == 0:
                            # Rerun pdflatex twice to resolve refs/citations.
                            for _ in range(2):
                                rc, out = self._run(cmd, cwd=tmp_path, timeout_seconds=timeout_seconds)
                                log_parts.append((out or "").strip())
                        else:
                            # Keep rc non-zero if bibtex failed; the PDF may still exist but citations won't resolve.
                            rc = bib_rc
                elif want_bibtex and re.search(r"\\bibliography\\s*\\{", tex_source):
                    log_parts.append("BibTeX requested but `bibtex` binary is not available on the server.")

            # Some compilers write a .log file; append if present.
            log_file = out_dir / "main.log"
            if log_file.exists():
                try:
                    log_parts.append(log_file.read_text(encoding="utf-8", errors="replace").strip())
                except Exception:
                    pass

            log = "\n\n".join([p for p in log_parts if p])

            if rc != 0 or not pdf_path.exists():
                return LatexCompileResult(
                    success=False,
                    engine=engine,
                    pdf_bytes=None,
                    log=log or f"Compilation failed (exit code {rc}).",
                    violations=violations,
                )

            pdf_bytes = pdf_path.read_bytes()
            return LatexCompileResult(
                success=True,
                engine=engine,
                pdf_bytes=pdf_bytes,
                log=log,
                violations=violations,
            )


latex_compiler_service = LatexCompilerService()
