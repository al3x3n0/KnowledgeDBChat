#!/usr/bin/env python3
"""
Seed arXiv sources for Compiler Optimization R&D team.

Creates pre-configured arXiv sources with relevant categories and search queries
for LLVM, compiler optimizations, and AI/ML compilers.

Usage:
    docker compose exec backend python scripts/seed_compiler_sources.py
"""

import asyncio
import sys
from uuid import uuid4
from datetime import datetime

# Add the app to path
sys.path.insert(0, "/app")

from sqlalchemy import select
from app.core.database import AsyncSessionLocal


# ArXiv sources to create
ARXIV_SOURCES = [
    {
        "name": "Compiler Optimization Research",
        "description": "Core compiler optimization papers from arXiv (cs.PL, cs.PF)",
        "config": {
            "queries": [
                "LLVM optimization",
                "compiler optimization loop vectorization",
                "instruction selection code generation",
                "register allocation compiler",
                "loop transformation tiling unrolling",
            ],
            "categories": ["cs.PL", "cs.PF"],
            "paper_ids": [],
            "max_results": 50,
            "start": 0,
            "sort_by": "submittedDate",
            "sort_order": "descending",
            "auto_summarize": True,
            "auto_literature_review": False,
            "auto_enrich_metadata": True,
            "topic": "Compiler Optimization",
            "display": {
                "queries": ["LLVM optimization", "compiler optimization"],
                "categories": ["cs.PL", "cs.PF"],
                "max_results": 50,
            }
        }
    },
    {
        "name": "CPU Architecture & Microarchitecture",
        "description": "CPU architecture, SIMD, and microarchitecture optimization papers",
        "config": {
            "queries": [
                "SIMD vectorization performance",
                "CPU cache optimization",
                "branch prediction microarchitecture",
                "instruction level parallelism",
                "memory hierarchy optimization",
            ],
            "categories": ["cs.AR", "cs.PF"],
            "paper_ids": [],
            "max_results": 50,
            "start": 0,
            "sort_by": "submittedDate",
            "sort_order": "descending",
            "auto_summarize": True,
            "auto_literature_review": False,
            "auto_enrich_metadata": True,
            "topic": "CPU Architecture",
            "display": {
                "queries": ["SIMD vectorization", "cache optimization"],
                "categories": ["cs.AR", "cs.PF"],
                "max_results": 50,
            }
        }
    },
    {
        "name": "AI/ML Compilers & Tensor Optimization",
        "description": "Neural network compilers, MLIR, TVM, and tensor optimization",
        "config": {
            "queries": [
                "neural network compiler optimization",
                "MLIR machine learning",
                "tensor compiler TVM Halide",
                "deep learning inference optimization",
                "quantization neural network INT8",
                "operator fusion graph optimization",
            ],
            "categories": ["cs.LG", "cs.PL", "cs.AR"],
            "paper_ids": [],
            "max_results": 50,
            "start": 0,
            "sort_by": "submittedDate",
            "sort_order": "descending",
            "auto_summarize": True,
            "auto_literature_review": False,
            "auto_enrich_metadata": True,
            "topic": "AI/ML Compilers",
            "display": {
                "queries": ["neural network compiler", "MLIR", "tensor compiler"],
                "categories": ["cs.LG", "cs.PL"],
                "max_results": 50,
            }
        }
    },
    {
        "name": "Polyhedral & Auto-Parallelization",
        "description": "Polyhedral compilation, auto-parallelization, and loop analysis",
        "config": {
            "queries": [
                "polyhedral compilation optimization",
                "automatic parallelization compiler",
                "loop dependence analysis",
                "affine transformation scheduling",
                "data locality optimization",
            ],
            "categories": ["cs.PL", "cs.DC"],
            "paper_ids": [],
            "max_results": 30,
            "start": 0,
            "sort_by": "submittedDate",
            "sort_order": "descending",
            "auto_summarize": True,
            "auto_literature_review": False,
            "auto_enrich_metadata": True,
            "topic": "Polyhedral Compilation",
            "display": {
                "queries": ["polyhedral compilation", "auto-parallelization"],
                "categories": ["cs.PL", "cs.DC"],
                "max_results": 30,
            }
        }
    },
]


async def seed_sources():
    """Create arXiv sources for compiler optimization team."""
    from app.models.document import DocumentSource

    async with AsyncSessionLocal() as db:
        created = []
        skipped = []

        for source_def in ARXIV_SOURCES:
            # Check if source already exists
            result = await db.execute(
                select(DocumentSource).where(DocumentSource.name == source_def["name"])
            )
            existing = result.scalar_one_or_none()

            if existing:
                skipped.append(source_def["name"])
                continue

            # Create new source
            source = DocumentSource(
                id=uuid4(),
                name=source_def["name"],
                source_type="arxiv",
                config=source_def["config"],
                is_active=True,
                is_syncing=False,
            )
            db.add(source)
            created.append(source_def["name"])

        await db.commit()

        print("\n" + "=" * 60)
        print("ArXiv Sources Seeding Complete")
        print("=" * 60)

        if created:
            print(f"\n✓ Created {len(created)} sources:")
            for name in created:
                print(f"  - {name}")

        if skipped:
            print(f"\n⊘ Skipped {len(skipped)} (already exist):")
            for name in skipped:
                print(f"  - {name}")

        # List all arxiv sources with IDs for triggering sync
        result = await db.execute(
            select(DocumentSource).where(DocumentSource.source_type == "arxiv")
        )
        all_sources = result.scalars().all()

        print("\n" + "-" * 60)
        print("To trigger ingestion, run:")
        print("-" * 60)
        for src in all_sources:
            print(f"\n# {src.name}")
            print(f"curl -X POST http://localhost:8000/api/v1/admin/sync/source/{src.id} \\")
            print(f"  -H 'Authorization: Bearer <token>'")

        print("\n" + "-" * 60)
        print("Or trigger all at once via admin panel or:")
        print("  curl -X POST http://localhost:8000/api/v1/admin/sync/all")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(seed_sources())
