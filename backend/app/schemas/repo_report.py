"""
Pydantic schemas for repository report and presentation generation.
"""

from datetime import datetime
from typing import Optional, List, Literal, Any, Dict
from uuid import UUID
from pydantic import BaseModel, Field, validator


# =============================================================================
# Section Types
# =============================================================================

AVAILABLE_SECTIONS = [
    "overview",           # Repository overview (name, description, stars, forks, license)
    "readme",             # README content
    "file_structure",     # File tree and architecture diagram
    "commits",            # Recent commits
    "issues",             # Open issues
    "pull_requests",      # Open pull requests
    "code_stats",         # Language breakdown, line counts
    "contributors",       # Top contributors
    "architecture",       # LLM-generated architecture summary
    "technology_stack",   # LLM-detected technology stack
]

DEFAULT_SECTIONS = [
    "overview",
    "readme",
    "file_structure",
    "commits",
    "code_stats",
    "architecture",
]


# =============================================================================
# Repository Analysis Data Schemas
# =============================================================================

class RepoInfo(BaseModel):
    """Basic repository information."""
    name: str
    full_name: str
    description: Optional[str] = None
    url: str
    default_branch: str = "main"
    stars: int = 0
    forks: int = 0
    watchers: int = 0
    license: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    language: Optional[str] = None  # Primary language


class CommitInfo(BaseModel):
    """Information about a commit."""
    sha: str
    message: str
    author: str
    date: datetime
    url: Optional[str] = None


class IssueInfo(BaseModel):
    """Information about an issue."""
    number: int
    title: str
    state: str
    author: str
    created_at: datetime
    labels: List[str] = Field(default_factory=list)
    url: Optional[str] = None


class PullRequestInfo(BaseModel):
    """Information about a pull request."""
    number: int
    title: str
    state: str
    author: str
    created_at: datetime
    labels: List[str] = Field(default_factory=list)
    source_branch: str
    target_branch: str
    url: Optional[str] = None


class ContributorInfo(BaseModel):
    """Information about a contributor."""
    username: str
    name: Optional[str] = None
    contributions: int = 0
    avatar_url: Optional[str] = None


class LanguageStats(BaseModel):
    """Language statistics for a repository."""
    languages: Dict[str, int] = Field(default_factory=dict)  # language -> bytes
    total_bytes: int = 0
    percentages: Dict[str, float] = Field(default_factory=dict)  # language -> percentage


class FileTreeNode(BaseModel):
    """Node in the file tree structure."""
    name: str
    type: Literal["file", "directory"]
    path: str
    children: List["FileTreeNode"] = Field(default_factory=list)
    size: Optional[int] = None
    language: Optional[str] = None


FileTreeNode.model_rebuild()


class RepoInsights(BaseModel):
    """LLM-generated insights about the repository."""
    architecture_summary: Optional[str] = None
    key_features: List[str] = Field(default_factory=list)
    technology_stack: List[str] = Field(default_factory=list)
    tech_stack_details: Dict[str, str] = Field(default_factory=dict)  # tech -> description


class RepoAnalysisResult(BaseModel):
    """Complete repository analysis result."""
    repo_info: RepoInfo
    readme_content: Optional[str] = None
    readme_html: Optional[str] = None
    file_tree: Optional[FileTreeNode] = None
    file_tree_text: Optional[str] = None  # ASCII tree representation
    commits: List[CommitInfo] = Field(default_factory=list)
    issues: List[IssueInfo] = Field(default_factory=list)
    pull_requests: List[PullRequestInfo] = Field(default_factory=list)
    contributors: List[ContributorInfo] = Field(default_factory=list)
    language_stats: Optional[LanguageStats] = None
    insights: Optional[RepoInsights] = None
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Theme Configuration (reuse from presentation schemas)
# =============================================================================

class ThemeColors(BaseModel):
    """Color configuration for a theme."""
    title_color: str = Field(default="#1a365d", description="Title text color (hex)")
    accent_color: str = Field(default="#2e86ab", description="Accent/highlight color (hex)")
    text_color: str = Field(default="#333333", description="Body text color (hex)")
    bg_color: str = Field(default="#ffffff", description="Background color (hex)")


class ThemeFonts(BaseModel):
    """Font configuration."""
    title_font: str = Field(default="Calibri", description="Font for titles")
    body_font: str = Field(default="Calibri", description="Font for body text")


class ThemeSizes(BaseModel):
    """Font size configuration."""
    title_size: int = Field(default=44, ge=20, le=72)
    subtitle_size: int = Field(default=24, ge=12, le=48)
    heading_size: int = Field(default=36, ge=16, le=60)
    body_size: int = Field(default=20, ge=10, le=36)


class ThemeConfig(BaseModel):
    """Complete theme configuration."""
    colors: ThemeColors = Field(default_factory=ThemeColors)
    fonts: ThemeFonts = Field(default_factory=ThemeFonts)
    sizes: ThemeSizes = Field(default_factory=ThemeSizes)


# =============================================================================
# Request Schemas
# =============================================================================

class RepoReportJobCreate(BaseModel):
    """Request schema for creating a repository report job."""
    # Source - one of source_id or repo_url is required
    source_id: Optional[UUID] = Field(
        default=None,
        description="ID of an existing DocumentSource (GitHub/GitLab repo)"
    )
    repo_url: Optional[str] = Field(
        default=None,
        description="Repository URL for ad-hoc analysis (e.g., https://github.com/owner/repo)"
    )
    repo_token: Optional[str] = Field(
        default=None,
        description="Access token for private repos (ad-hoc mode only)"
    )

    # Output configuration
    output_format: Literal["docx", "pdf", "pptx"] = Field(
        default="docx",
        description="Output format"
    )
    title: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Report title. Defaults to repository name."
    )
    sections: List[str] = Field(
        default_factory=lambda: DEFAULT_SECTIONS.copy(),
        description=f"Sections to include. Options: {AVAILABLE_SECTIONS}"
    )

    # PPTX-specific options
    slide_count: int = Field(
        default=10,
        ge=3,
        le=30,
        description="Target slide count (PPTX only)"
    )
    include_diagrams: bool = Field(
        default=True,
        description="Include architecture diagrams"
    )

    # Style
    style: Literal["professional", "technical", "casual", "modern", "minimal", "corporate"] = Field(
        default="professional",
        description="Built-in style preset"
    )
    custom_theme: Optional[ThemeConfig] = Field(
        default=None,
        description="Custom theme configuration (overrides style)"
    )

    @validator('sections')
    def validate_sections(cls, v):
        invalid = [s for s in v if s not in AVAILABLE_SECTIONS]
        if invalid:
            raise ValueError(f"Invalid sections: {invalid}. Valid options: {AVAILABLE_SECTIONS}")
        return v

    @validator('repo_url')
    def validate_source(cls, v, values):
        source_id = values.get('source_id')
        if not source_id and not v:
            raise ValueError("Either source_id or repo_url must be provided")
        return v


class RepoReportJobUpdate(BaseModel):
    """Schema for updating a repo report job (internal use)."""
    status: Optional[str] = None
    progress: Optional[int] = None
    current_stage: Optional[str] = None
    analysis_data: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# =============================================================================
# Response Schemas
# =============================================================================

class RepoReportJobResponse(BaseModel):
    """Response schema for repository report jobs."""
    id: UUID
    user_id: UUID

    # Source reference
    source_id: Optional[UUID] = None
    adhoc_url: Optional[str] = None

    # Repository info
    repo_name: str
    repo_url: str
    repo_type: str

    # Output configuration
    output_format: str
    title: str
    sections: List[str]
    slide_count: Optional[int] = None
    include_diagrams: bool

    # Style
    style: str
    custom_theme: Optional[Dict[str, Any]] = None

    # Status
    status: str
    progress: int
    current_stage: Optional[str] = None

    # Output
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    download_url: Optional[str] = None

    # Error
    error: Optional[str] = None

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Analysis data (optional, for detail view)
    analysis_data: Optional[RepoAnalysisResult] = None

    class Config:
        from_attributes = True


class RepoReportJobListItem(BaseModel):
    """Compact response schema for listing jobs."""
    id: UUID
    user_id: UUID
    repo_name: str
    repo_url: str
    repo_type: str
    output_format: str
    title: str
    status: str
    progress: int
    file_size: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class RepoReportJobListResponse(BaseModel):
    """Response schema for listing repository report jobs."""
    jobs: List[RepoReportJobListItem]
    total: int


# =============================================================================
# Progress WebSocket Messages
# =============================================================================

class RepoReportProgressMessage(BaseModel):
    """WebSocket message for report generation progress."""
    type: Literal["progress", "stage", "complete", "error"]
    job_id: UUID
    status: Optional[str] = None
    progress: Optional[int] = None
    stage: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# Section Configuration
# =============================================================================

class SectionInfo(BaseModel):
    """Information about an available section."""
    id: str
    name: str
    description: str
    default: bool
    supports_formats: List[str]


class AvailableSectionsResponse(BaseModel):
    """Response schema for available sections."""
    sections: List[SectionInfo] = Field(default_factory=lambda: [
        SectionInfo(
            id="overview",
            name="Repository Overview",
            description="Basic info: name, description, stars, forks, license",
            default=True,
            supports_formats=["docx", "pdf", "pptx"]
        ),
        SectionInfo(
            id="readme",
            name="README",
            description="README content rendered as documentation",
            default=True,
            supports_formats=["docx", "pdf", "pptx"]
        ),
        SectionInfo(
            id="file_structure",
            name="File Structure",
            description="Repository file tree and architecture diagram",
            default=True,
            supports_formats=["docx", "pdf", "pptx"]
        ),
        SectionInfo(
            id="commits",
            name="Recent Commits",
            description="Recent commit history with authors and messages",
            default=True,
            supports_formats=["docx", "pdf", "pptx"]
        ),
        SectionInfo(
            id="issues",
            name="Issues",
            description="Open issues list",
            default=False,
            supports_formats=["docx", "pdf", "pptx"]
        ),
        SectionInfo(
            id="pull_requests",
            name="Pull Requests",
            description="Open pull requests list",
            default=False,
            supports_formats=["docx", "pdf", "pptx"]
        ),
        SectionInfo(
            id="code_stats",
            name="Code Statistics",
            description="Language breakdown and line counts",
            default=True,
            supports_formats=["docx", "pdf", "pptx"]
        ),
        SectionInfo(
            id="contributors",
            name="Contributors",
            description="Top contributors list",
            default=False,
            supports_formats=["docx", "pdf", "pptx"]
        ),
        SectionInfo(
            id="architecture",
            name="Architecture Analysis",
            description="LLM-generated architecture summary and insights",
            default=True,
            supports_formats=["docx", "pdf", "pptx"]
        ),
        SectionInfo(
            id="technology_stack",
            name="Technology Stack",
            description="Detected technologies and frameworks",
            default=False,
            supports_formats=["docx", "pdf", "pptx"]
        ),
    ])
