"""
Repository Report Generator Service.

Converts repository analysis data into DOCX/PDF content items.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from io import BytesIO
from loguru import logger

from app.schemas.repo_report import RepoAnalysisResult
from app.services.docx_builder import DOCXBuilder
from app.services.pdf_builder import PDFBuilder


class RepoReportGenerator:
    """
    Generates DOCX/PDF reports from repository analysis data.

    Converts RepoAnalysisResult into structured content items
    and renders them using DOCXBuilder or PDFBuilder.
    """

    def __init__(
        self,
        style: str = "professional",
        custom_theme: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the report generator.

        Args:
            style: Built-in style name
            custom_theme: Custom theme configuration
        """
        self.style = style
        self.custom_theme = custom_theme

    async def generate_docx(
        self,
        analysis: RepoAnalysisResult,
        title: str,
        sections: List[str],
        progress_callback: Optional[callable] = None
    ) -> bytes:
        """
        Generate a DOCX report from analysis data.

        Args:
            analysis: Repository analysis result
            title: Report title
            sections: List of sections to include
            progress_callback: Optional callback for progress updates

        Returns:
            DOCX file as bytes
        """
        if progress_callback:
            await progress_callback(50, "Building document content")

        content_items = self._build_content_items(analysis, sections)

        if progress_callback:
            await progress_callback(70, "Rendering DOCX document")

        builder = DOCXBuilder(style=self.style, custom_theme=self.custom_theme)
        docx_bytes = builder.build(
            title=title,
            content_items=content_items,
            author="KnowledgeDB Repository Report",
            subject=f"Repository analysis for {analysis.repo_info.full_name}"
        )

        if progress_callback:
            await progress_callback(85, "DOCX generation complete")

        return docx_bytes

    async def generate_pdf(
        self,
        analysis: RepoAnalysisResult,
        title: str,
        sections: List[str],
        progress_callback: Optional[callable] = None
    ) -> bytes:
        """
        Generate a PDF report from analysis data.

        Args:
            analysis: Repository analysis result
            title: Report title
            sections: List of sections to include
            progress_callback: Optional callback for progress updates

        Returns:
            PDF file as bytes
        """
        if progress_callback:
            await progress_callback(50, "Building document content")

        content_items = self._build_content_items(analysis, sections)

        if progress_callback:
            await progress_callback(70, "Rendering PDF document")

        builder = PDFBuilder(style=self.style, custom_theme=self.custom_theme)
        pdf_bytes = builder.build(
            title=title,
            content_items=content_items,
            author="KnowledgeDB Repository Report",
            subject=f"Repository analysis for {analysis.repo_info.full_name}"
        )

        if progress_callback:
            await progress_callback(85, "PDF generation complete")

        return pdf_bytes

    def _build_content_items(
        self,
        analysis: RepoAnalysisResult,
        sections: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Build content items from analysis data.

        Args:
            analysis: Repository analysis result
            sections: List of sections to include

        Returns:
            List of content items for document builder
        """
        items: List[Dict[str, Any]] = []

        # Overview section
        if "overview" in sections:
            items.extend(self._build_overview_section(analysis))

        # README section
        if "readme" in sections and analysis.readme_content:
            items.extend(self._build_readme_section(analysis))

        # File structure section
        if "file_structure" in sections and analysis.file_tree_text:
            items.extend(self._build_file_structure_section(analysis))

        # Commits section
        if "commits" in sections and analysis.commits:
            items.extend(self._build_commits_section(analysis))

        # Issues section
        if "issues" in sections and analysis.issues:
            items.extend(self._build_issues_section(analysis))

        # Pull requests section
        if "pull_requests" in sections and analysis.pull_requests:
            items.extend(self._build_pull_requests_section(analysis))

        # Code statistics section
        if "code_stats" in sections and analysis.language_stats:
            items.extend(self._build_code_stats_section(analysis))

        # Contributors section
        if "contributors" in sections and analysis.contributors:
            items.extend(self._build_contributors_section(analysis))

        # Architecture section
        if "architecture" in sections and analysis.insights:
            items.extend(self._build_architecture_section(analysis))

        # Technology stack section
        if "technology_stack" in sections and analysis.insights:
            items.extend(self._build_tech_stack_section(analysis))

        # Add generation timestamp at the end
        items.append({"type": "horizontal_rule"})
        items.append({
            "type": "paragraph",
            "text": f"Report generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} by KnowledgeDB"
        })

        return items

    def _build_overview_section(self, analysis: RepoAnalysisResult) -> List[Dict[str, Any]]:
        """Build the overview section."""
        items = []
        repo = analysis.repo_info

        items.append({"type": "heading", "level": 1, "text": "Repository Overview"})

        # Description
        if repo.description:
            items.append({"type": "paragraph", "text": repo.description})

        # Stats table
        stats_rows = [
            ["Repository", repo.full_name],
            ["URL", repo.url],
            ["Default Branch", repo.default_branch],
        ]

        if repo.stars > 0:
            stats_rows.append(["Stars", str(repo.stars)])
        if repo.forks > 0:
            stats_rows.append(["Forks", str(repo.forks)])
        if repo.watchers > 0:
            stats_rows.append(["Watchers", str(repo.watchers)])
        if repo.license:
            stats_rows.append(["License", repo.license])
        if repo.language:
            stats_rows.append(["Primary Language", repo.language])
        if repo.created_at:
            stats_rows.append(["Created", repo.created_at.strftime("%Y-%m-%d")])
        if repo.updated_at:
            stats_rows.append(["Last Updated", repo.updated_at.strftime("%Y-%m-%d")])

        items.append({
            "type": "table",
            "headers": ["Property", "Value"],
            "rows": stats_rows
        })

        items.append({"type": "paragraph", "text": ""})  # Spacer
        return items

    def _build_readme_section(self, analysis: RepoAnalysisResult) -> List[Dict[str, Any]]:
        """Build the README section."""
        items = []
        items.append({"type": "page_break"})
        items.append({"type": "heading", "level": 1, "text": "README"})

        # Parse and convert README markdown to content items
        readme_lines = analysis.readme_content.split("\n")
        current_paragraph = []

        for line in readme_lines:
            stripped = line.strip()

            # Heading detection
            if stripped.startswith("# "):
                if current_paragraph:
                    items.append({"type": "paragraph", "text": " ".join(current_paragraph)})
                    current_paragraph = []
                items.append({"type": "heading", "level": 2, "text": stripped[2:]})
            elif stripped.startswith("## "):
                if current_paragraph:
                    items.append({"type": "paragraph", "text": " ".join(current_paragraph)})
                    current_paragraph = []
                items.append({"type": "heading", "level": 2, "text": stripped[3:]})
            elif stripped.startswith("### "):
                if current_paragraph:
                    items.append({"type": "paragraph", "text": " ".join(current_paragraph)})
                    current_paragraph = []
                items.append({"type": "heading", "level": 3, "text": stripped[4:]})
            elif stripped.startswith("- ") or stripped.startswith("* "):
                if current_paragraph:
                    items.append({"type": "paragraph", "text": " ".join(current_paragraph)})
                    current_paragraph = []
                items.append({"type": "bullet_list", "items": [stripped[2:]]})
            elif stripped.startswith("```"):
                if current_paragraph:
                    items.append({"type": "paragraph", "text": " ".join(current_paragraph)})
                    current_paragraph = []
                # We'll just note there's a code block - proper parsing would be more complex
            elif stripped == "":
                if current_paragraph:
                    items.append({"type": "paragraph", "text": " ".join(current_paragraph)})
                    current_paragraph = []
            else:
                current_paragraph.append(stripped)

        if current_paragraph:
            items.append({"type": "paragraph", "text": " ".join(current_paragraph)})

        return items

    def _build_file_structure_section(self, analysis: RepoAnalysisResult) -> List[Dict[str, Any]]:
        """Build the file structure section."""
        items = []
        items.append({"type": "page_break"})
        items.append({"type": "heading", "level": 1, "text": "File Structure"})

        items.append({
            "type": "paragraph",
            "text": "The repository has the following directory structure:"
        })

        # Truncate tree if too long
        tree_text = analysis.file_tree_text
        if len(tree_text) > 5000:
            lines = tree_text.split("\n")[:100]
            tree_text = "\n".join(lines) + "\n... (truncated)"

        items.append({
            "type": "code_block",
            "code": tree_text,
            "language": "text"
        })

        return items

    def _build_commits_section(self, analysis: RepoAnalysisResult) -> List[Dict[str, Any]]:
        """Build the commits section."""
        items = []
        items.append({"type": "page_break"})
        items.append({"type": "heading", "level": 1, "text": "Recent Commits"})

        items.append({
            "type": "paragraph",
            "text": f"The following table shows the {len(analysis.commits)} most recent commits:"
        })

        commit_rows = []
        for commit in analysis.commits[:20]:
            date_str = commit.date.strftime("%Y-%m-%d") if commit.date else "N/A"
            message = commit.message[:60] + "..." if len(commit.message) > 60 else commit.message
            commit_rows.append([
                commit.sha[:8],
                message,
                commit.author,
                date_str
            ])

        items.append({
            "type": "table",
            "headers": ["SHA", "Message", "Author", "Date"],
            "rows": commit_rows
        })

        return items

    def _build_issues_section(self, analysis: RepoAnalysisResult) -> List[Dict[str, Any]]:
        """Build the issues section."""
        items = []
        items.append({"type": "page_break"})
        items.append({"type": "heading", "level": 1, "text": "Open Issues"})

        items.append({
            "type": "paragraph",
            "text": f"There are currently {len(analysis.issues)} open issues:"
        })

        issue_rows = []
        for issue in analysis.issues[:20]:
            date_str = issue.created_at.strftime("%Y-%m-%d") if issue.created_at else "N/A"
            title = issue.title[:50] + "..." if len(issue.title) > 50 else issue.title
            labels = ", ".join(issue.labels[:3]) if issue.labels else "-"
            issue_rows.append([
                f"#{issue.number}",
                title,
                issue.author,
                labels,
                date_str
            ])

        items.append({
            "type": "table",
            "headers": ["#", "Title", "Author", "Labels", "Created"],
            "rows": issue_rows
        })

        return items

    def _build_pull_requests_section(self, analysis: RepoAnalysisResult) -> List[Dict[str, Any]]:
        """Build the pull requests section."""
        items = []
        items.append({"type": "page_break"})
        items.append({"type": "heading", "level": 1, "text": "Open Pull Requests"})

        items.append({
            "type": "paragraph",
            "text": f"There are currently {len(analysis.pull_requests)} open pull requests:"
        })

        pr_rows = []
        for pr in analysis.pull_requests[:20]:
            date_str = pr.created_at.strftime("%Y-%m-%d") if pr.created_at else "N/A"
            title = pr.title[:50] + "..." if len(pr.title) > 50 else pr.title
            branches = f"{pr.source_branch} → {pr.target_branch}"
            if len(branches) > 30:
                branches = branches[:27] + "..."
            pr_rows.append([
                f"#{pr.number}",
                title,
                pr.author,
                branches,
                date_str
            ])

        items.append({
            "type": "table",
            "headers": ["#", "Title", "Author", "Branches", "Created"],
            "rows": pr_rows
        })

        return items

    def _build_code_stats_section(self, analysis: RepoAnalysisResult) -> List[Dict[str, Any]]:
        """Build the code statistics section."""
        items = []
        items.append({"type": "page_break"})
        items.append({"type": "heading", "level": 1, "text": "Code Statistics"})

        if not analysis.language_stats:
            items.append({"type": "paragraph", "text": "No language statistics available."})
            return items

        items.append({"type": "heading", "level": 2, "text": "Language Breakdown"})

        lang_rows = []
        percentages = analysis.language_stats.percentages
        sorted_langs = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

        for lang, pct in sorted_langs[:15]:
            # Create a simple text bar
            bar_length = int(pct / 5)  # Each 5% = 1 char
            bar = "█" * bar_length
            lang_rows.append([lang, f"{pct:.1f}%", bar])

        items.append({
            "type": "table",
            "headers": ["Language", "Percentage", "Distribution"],
            "rows": lang_rows
        })

        if analysis.language_stats.total_bytes > 0:
            total_kb = analysis.language_stats.total_bytes / 1024
            if total_kb > 1024:
                total_str = f"{total_kb / 1024:.1f} MB"
            else:
                total_str = f"{total_kb:.1f} KB"
            items.append({
                "type": "paragraph",
                "text": f"Total code size: {total_str}"
            })

        return items

    def _build_contributors_section(self, analysis: RepoAnalysisResult) -> List[Dict[str, Any]]:
        """Build the contributors section."""
        items = []
        items.append({"type": "page_break"})
        items.append({"type": "heading", "level": 1, "text": "Top Contributors"})

        contrib_rows = []
        for i, contrib in enumerate(analysis.contributors[:10], 1):
            name = contrib.name or contrib.username
            contrib_rows.append([
                str(i),
                name,
                str(contrib.contributions)
            ])

        items.append({
            "type": "table",
            "headers": ["Rank", "Contributor", "Contributions"],
            "rows": contrib_rows
        })

        return items

    def _build_architecture_section(self, analysis: RepoAnalysisResult) -> List[Dict[str, Any]]:
        """Build the architecture analysis section."""
        items = []

        if not analysis.insights:
            return items

        items.append({"type": "page_break"})
        items.append({"type": "heading", "level": 1, "text": "Architecture Analysis"})

        if analysis.insights.architecture_summary:
            items.append({
                "type": "paragraph",
                "text": analysis.insights.architecture_summary
            })

        if analysis.insights.key_features:
            items.append({"type": "heading", "level": 2, "text": "Key Features"})
            items.append({
                "type": "bullet_list",
                "items": analysis.insights.key_features
            })

        return items

    def _build_tech_stack_section(self, analysis: RepoAnalysisResult) -> List[Dict[str, Any]]:
        """Build the technology stack section."""
        items = []

        if not analysis.insights or not analysis.insights.technology_stack:
            return items

        items.append({"type": "page_break"})
        items.append({"type": "heading", "level": 1, "text": "Technology Stack"})

        items.append({
            "type": "paragraph",
            "text": "The following technologies, frameworks, and tools were detected in this repository:"
        })

        items.append({
            "type": "bullet_list",
            "items": analysis.insights.technology_stack
        })

        return items


# Singleton instance
repo_report_generator = RepoReportGenerator()
