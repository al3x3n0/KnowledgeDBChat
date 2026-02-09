"""
Pydantic schemas for LaTeX Studio endpoints.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LatexStatusResponse(BaseModel):
    enabled: bool
    admin_only: bool
    use_celery_worker: bool = False
    celery_queue: Optional[str] = None
    timeout_seconds: int
    max_source_chars: int
    available_engines: Dict[str, bool]
    available_tools: Dict[str, bool] = {}


class LatexCompileRequest(BaseModel):
    tex_source: str = Field(..., min_length=1, description="LaTeX source (single-file) to compile")
    safe_mode: bool = Field(default=True, description="Restrict dangerous TeX features (recommended)")
    preferred_engine: Optional[str] = Field(default=None, description="Optional: 'tectonic' or 'pdflatex'")


class LatexCompileResponse(BaseModel):
    success: bool
    engine: Optional[str] = None
    pdf_base64: Optional[str] = None
    log: str = ""
    violations: List[str] = []


class LatexCopilotSectionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="What to write in LaTeX")
    search_query: Optional[str] = Field(default=None, max_length=4000, description="Optional KB search query")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional explicit document IDs to ground on")
    max_sources: int = Field(default=8, ge=1, le=20)
    use_vector_snippets: bool = True
    chunks_per_source: int = Field(default=3, ge=1, le=8)
    chunk_max_chars: int = Field(default=500, ge=100, le=2000)
    max_source_chars: int = Field(default=2000, ge=200, le=8000)
    citation_mode: str = Field(default="thebibliography", pattern="^(thebibliography|bibtex)$")


class LatexCopilotSectionResponse(BaseModel):
    tex_snippet: str
    # Back-compat: historically this field carried a LaTeX thebibliography block.
    bibtex: str = ""
    references_tex: str = ""
    bibtex_entries: str = ""
    sources: List[Dict[str, str]] = []


class LatexCopilotFixRequest(BaseModel):
    tex_source: str = Field(..., min_length=1, description="Current LaTeX source")
    compile_log: str = Field(..., min_length=1, description="Compiler log / error output")
    safe_mode: bool = Field(default=True, description="Avoid introducing unsafe TeX primitives")


class LatexCopilotFixResponse(BaseModel):
    tex_source_fixed: str
    notes: str = ""
    unsafe_warnings: List[str] = []


class LatexMathCopilotRequest(BaseModel):
    tex_source: str = Field(..., min_length=1, description="Current LaTeX source (paper.tex)")
    mode: str = Field(default="analyze", pattern="^(analyze|autocomplete)$")
    goal: str = Field(
        default="Standardize math notation and fix equation references.",
        max_length=2000,
        description="What to improve (math style, units, shapes, refs)",
    )
    selection: Optional[str] = Field(default=None, max_length=8000, description="Optional: selected LaTeX fragment")
    cursor_context: Optional[str] = Field(default=None, max_length=8000, description="Optional: local context around cursor")

    enforce_siunitx: bool = Field(default=True, description="Prefer siunitx for units formatting")
    enforce_shapes: bool = Field(default=True, description="Prefer consistent tensor/matrix/vector shape notation")
    enforce_bold_italic_conventions: bool = Field(default=True, description="Prefer consistent scalar/vector/matrix styling")
    enforce_equation_labels: bool = Field(default=True, description="Prefer labeling displayed equations and using (eq)refs")

    max_source_chars: int = Field(default=60000, ge=500, le=120000)
    return_patched_source: bool = Field(default=True, description="If diff applies, include patched tex_source in response")


class LatexMathCopilotResponse(BaseModel):
    conventions: Dict[str, str] = {}
    suggestions: List[Dict[str, str]] = []
    diff_unified: str = ""
    notes: str = ""
    base_sha256: str
    diff_applies: bool = False
    patched_sha256: Optional[str] = None
    tex_source_patched: Optional[str] = None
    diff_warnings: List[str] = []


class LatexCitationsRequest(BaseModel):
    document_ids: List[str] = Field(..., min_length=1, max_length=50)
    mode: str = Field(default="bibtex", pattern="^(bibtex|thebibliography)$")
    bib_filename: str = Field(default="refs.bib", max_length=100)


class LatexCitationsResponse(BaseModel):
    mode: str
    cite_keys_by_doc_id: Dict[str, str]
    cite_command: str
    bibliography_scaffold: str = ""
    bibtex_entries: str = ""
    references_tex: str = ""


class LatexApplyUnifiedDiffRequest(BaseModel):
    diff_unified: str = Field(..., min_length=1, max_length=200000, description="Unified diff that patches paper.tex")
    expected_base_sha256: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional: sha256 of current paper.tex to prevent applying to stale content",
    )


class LatexApplyUnifiedDiffResponse(BaseModel):
    applied: bool
    tex_source: str
    base_sha256: str
    new_sha256: str
    warnings: List[str] = []
