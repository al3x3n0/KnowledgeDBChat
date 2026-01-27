"""
Repository Presentation Generator Service.

Converts repository analysis data into PPTX presentations.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from io import BytesIO
from loguru import logger

from app.schemas.repo_report import RepoAnalysisResult
from app.schemas.presentation import PresentationOutline, SlideContent
from app.services.pptx_builder import PPTXBuilder
from app.services.llm_service import LLMService


class RepoPresentationGenerator:
    """
    Generates PPTX presentations from repository analysis data.

    Uses LLM to generate compelling slide content and
    Mermaid diagrams for architecture visualization.
    """

    def __init__(
        self,
        style: str = "professional",
        custom_theme: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the presentation generator.

        Args:
            style: Built-in style name
            custom_theme: Custom theme configuration
        """
        self.style = style
        self.custom_theme = custom_theme
        self.llm_service = LLMService()

    async def generate_pptx(
        self,
        analysis: RepoAnalysisResult,
        title: str,
        sections: List[str],
        slide_count: int = 10,
        include_diagrams: bool = True,
        progress_callback: Optional[callable] = None
    ) -> bytes:
        """
        Generate a PPTX presentation from analysis data.

        Args:
            analysis: Repository analysis result
            title: Presentation title
            sections: List of sections to include
            slide_count: Target number of slides
            include_diagrams: Whether to include Mermaid diagrams
            progress_callback: Optional callback for progress updates

        Returns:
            PPTX file as bytes
        """
        if progress_callback:
            await progress_callback(50, "Generating presentation outline")

        # Generate outline using LLM
        outline = await self._generate_outline(
            analysis=analysis,
            title=title,
            sections=sections,
            slide_count=slide_count,
            include_diagrams=include_diagrams
        )

        if progress_callback:
            await progress_callback(65, "Generating diagrams")

        # Generate Mermaid diagrams if requested
        diagrams = {}
        if include_diagrams:
            diagrams = await self._generate_diagrams(outline, analysis)

        if progress_callback:
            await progress_callback(80, "Building PPTX presentation")

        # Build the presentation
        builder = PPTXBuilder(style=self.style, custom_theme=self.custom_theme)
        pptx_bytes = builder.build(outline, diagrams)

        if progress_callback:
            await progress_callback(90, "PPTX generation complete")

        return pptx_bytes

    async def _generate_outline(
        self,
        analysis: RepoAnalysisResult,
        title: str,
        sections: List[str],
        slide_count: int,
        include_diagrams: bool
    ) -> PresentationOutline:
        """
        Generate presentation outline using LLM.

        Args:
            analysis: Repository analysis result
            title: Presentation title
            sections: Sections to include
            slide_count: Target slide count
            include_diagrams: Whether to include diagram slides

        Returns:
            PresentationOutline with all slides
        """
        slides: List[SlideContent] = []
        slide_num = 1

        # Slide 1: Title slide
        slides.append(SlideContent(
            slide_number=slide_num,
            slide_type="title",
            title=title,
            subtitle=analysis.repo_info.description or f"Analysis of {analysis.repo_info.full_name}",
            content=[]
        ))
        slide_num += 1

        # Slide 2: Overview
        if "overview" in sections:
            overview_content = [
                f"Repository: {analysis.repo_info.full_name}",
                f"URL: {analysis.repo_info.url}",
            ]
            if analysis.repo_info.stars > 0:
                overview_content.append(f"Stars: {analysis.repo_info.stars:,}")
            if analysis.repo_info.forks > 0:
                overview_content.append(f"Forks: {analysis.repo_info.forks:,}")
            if analysis.repo_info.license:
                overview_content.append(f"License: {analysis.repo_info.license}")
            if analysis.repo_info.language:
                overview_content.append(f"Primary Language: {analysis.repo_info.language}")

            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Repository Overview",
                content=overview_content
            ))
            slide_num += 1

        # Slide: Architecture (if available)
        if "architecture" in sections and analysis.insights and analysis.insights.architecture_summary:
            # Split architecture summary into bullet points
            arch_summary = analysis.insights.architecture_summary
            arch_points = self._split_to_bullets(arch_summary, max_bullets=5)

            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Architecture Overview",
                content=arch_points
            ))
            slide_num += 1

            # Architecture diagram slide
            if include_diagrams:
                slides.append(SlideContent(
                    slide_number=slide_num,
                    slide_type="diagram",
                    title="Project Structure",
                    content=["Visual representation of the project architecture"],
                    diagram_description="architecture diagram showing main components"
                ))
                slide_num += 1

        # Slide: Key Features
        if "architecture" in sections and analysis.insights and analysis.insights.key_features:
            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Key Features",
                content=analysis.insights.key_features[:6]
            ))
            slide_num += 1

        # Slide: Technology Stack
        if "technology_stack" in sections and analysis.insights and analysis.insights.technology_stack:
            tech_list = analysis.insights.technology_stack[:8]
            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Technology Stack",
                content=tech_list
            ))
            slide_num += 1

        # Slide: Code Statistics
        if "code_stats" in sections and analysis.language_stats and analysis.language_stats.percentages:
            lang_items = []
            sorted_langs = sorted(
                analysis.language_stats.percentages.items(),
                key=lambda x: x[1],
                reverse=True
            )[:6]
            for lang, pct in sorted_langs:
                lang_items.append(f"{lang}: {pct:.1f}%")

            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Language Breakdown",
                content=lang_items
            ))
            slide_num += 1

        # Slide: File Structure
        if "file_structure" in sections and analysis.file_tree_text:
            # Extract key directories from file tree
            tree_lines = analysis.file_tree_text.split("\n")[:15]
            dir_structure = [line for line in tree_lines if line.strip()][:6]

            if dir_structure:
                slides.append(SlideContent(
                    slide_number=slide_num,
                    slide_type="content",
                    title="Project Structure",
                    content=dir_structure
                ))
                slide_num += 1

        # Slide: Recent Activity
        if "commits" in sections and analysis.commits:
            commit_items = []
            for commit in analysis.commits[:5]:
                msg = commit.message[:50] + "..." if len(commit.message) > 50 else commit.message
                commit_items.append(f"{commit.sha[:7]}: {msg}")

            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Recent Commits",
                content=commit_items
            ))
            slide_num += 1

        # Slide: Contributors
        if "contributors" in sections and analysis.contributors:
            contrib_items = []
            for contrib in analysis.contributors[:6]:
                name = contrib.name or contrib.username
                contrib_items.append(f"{name}: {contrib.contributions} contributions")

            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Top Contributors",
                content=contrib_items
            ))
            slide_num += 1

        # Slide: Issues Summary
        if "issues" in sections and analysis.issues:
            issue_items = [f"Open issues: {len(analysis.issues)}"]
            for issue in analysis.issues[:4]:
                title_short = issue.title[:40] + "..." if len(issue.title) > 40 else issue.title
                issue_items.append(f"#{issue.number}: {title_short}")

            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Open Issues",
                content=issue_items
            ))
            slide_num += 1

        # Slide: Pull Requests
        if "pull_requests" in sections and analysis.pull_requests:
            pr_items = [f"Open PRs: {len(analysis.pull_requests)}"]
            for pr in analysis.pull_requests[:4]:
                title_short = pr.title[:40] + "..." if len(pr.title) > 40 else pr.title
                pr_items.append(f"#{pr.number}: {title_short}")

            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Open Pull Requests",
                content=pr_items
            ))
            slide_num += 1

        # Summary slide
        summary_points = [
            f"Repository: {analysis.repo_info.full_name}",
        ]
        if analysis.repo_info.description:
            summary_points.append(analysis.repo_info.description[:80])
        if analysis.language_stats and analysis.language_stats.percentages:
            top_lang = max(analysis.language_stats.percentages.items(), key=lambda x: x[1])
            summary_points.append(f"Primary language: {top_lang[0]}")
        if analysis.insights and analysis.insights.technology_stack:
            summary_points.append(f"Key technologies: {', '.join(analysis.insights.technology_stack[:3])}")

        slides.append(SlideContent(
            slide_number=slide_num,
            slide_type="summary",
            title="Summary",
            content=summary_points
        ))

        return PresentationOutline(
            title=title,
            subtitle=analysis.repo_info.description,
            slides=slides
        )

    async def _generate_diagrams(
        self,
        outline: PresentationOutline,
        analysis: RepoAnalysisResult
    ) -> Dict[int, bytes]:
        """
        Generate Mermaid diagrams for diagram slides.

        Args:
            outline: Presentation outline
            analysis: Repository analysis

        Returns:
            Dict mapping slide number to PNG bytes
        """
        diagrams = {}

        for slide in outline.slides:
            if slide.slide_type != "diagram":
                continue

            try:
                # Generate Mermaid code for architecture diagram
                mermaid_code = self._generate_architecture_mermaid(analysis)

                if mermaid_code:
                    # Render Mermaid to PNG
                    png_bytes = await self._render_mermaid_to_png(mermaid_code)
                    if png_bytes:
                        diagrams[slide.slide_number] = png_bytes
            except Exception as e:
                logger.warning(f"Failed to generate diagram for slide {slide.slide_number}: {e}")

        return diagrams

    def _generate_architecture_mermaid(self, analysis: RepoAnalysisResult) -> str:
        """
        Generate a Mermaid diagram representing project architecture.

        Args:
            analysis: Repository analysis

        Returns:
            Mermaid diagram code
        """
        if not analysis.file_tree_text:
            return ""

        # Parse top-level directories from file tree
        tree_lines = analysis.file_tree_text.split("\n")
        top_dirs = []

        for line in tree_lines[1:20]:  # Skip root, check first 20 lines
            # Look for top-level directories (ones with single-level indentation)
            if line.startswith("├── ") or line.startswith("└── "):
                name = line.replace("├── ", "").replace("└── ", "").strip()
                if name.endswith("/"):
                    name = name[:-1]
                    top_dirs.append(name)

        if not top_dirs:
            return ""

        # Build a simple flowchart
        mermaid = "graph TD\n"
        mermaid += f"    Root[{analysis.repo_info.name}]\n"

        for i, dir_name in enumerate(top_dirs[:8]):  # Limit to 8 directories
            node_id = f"D{i}"
            display_name = dir_name.replace("-", " ").replace("_", " ")
            mermaid += f"    Root --> {node_id}[{display_name}]\n"

        return mermaid

    async def _render_mermaid_to_png(self, mermaid_code: str) -> Optional[bytes]:
        """
        Render Mermaid diagram to PNG.

        This is a placeholder - in production, you'd use:
        - mermaid-cli (mmdc)
        - A Mermaid API service
        - Kroki.io

        Args:
            mermaid_code: Mermaid diagram code

        Returns:
            PNG bytes or None if rendering fails
        """
        try:
            import httpx

            # Use Kroki.io for rendering (public service)
            # In production, consider self-hosting Kroki
            import base64
            import zlib

            # Compress and encode for Kroki
            compressed = zlib.compress(mermaid_code.encode("utf-8"), 9)
            encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")

            kroki_url = f"https://kroki.io/mermaid/png/{encoded}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(kroki_url)
                if response.status_code == 200:
                    return response.content
                else:
                    logger.warning(f"Kroki returned status {response.status_code}")
                    return None

        except Exception as e:
            logger.warning(f"Failed to render Mermaid diagram: {e}")
            return None

    def _split_to_bullets(self, text: str, max_bullets: int = 5) -> List[str]:
        """
        Split a paragraph into bullet points.

        Args:
            text: Paragraph text
            max_bullets: Maximum number of bullets

        Returns:
            List of bullet point strings
        """
        # Split on sentences
        sentences = text.replace("\n", " ").split(". ")
        bullets = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Clean up
            if not sentence.endswith("."):
                sentence += "."

            # Skip very short sentences
            if len(sentence) < 15:
                continue

            # Truncate long sentences
            if len(sentence) > 100:
                sentence = sentence[:97] + "..."

            bullets.append(sentence)

            if len(bullets) >= max_bullets:
                break

        return bullets if bullets else [text[:200]]


# Singleton instance
repo_presentation_generator = RepoPresentationGenerator()
