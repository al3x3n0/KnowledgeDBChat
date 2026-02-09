"""Add Compiler Optimization Expert agent and pre-built saved searches.

Specialized for R&D teams working on LLVM-based compiler optimizations and AI/ML compilers.

Revision ID: 0030_compiler_expert
Revises: 0029_add_api_keys
Create Date: 2026-01-27 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0030_compiler_expert"
down_revision = "0029_add_api_keys"
branch_labels = None
depends_on = None


# Compiler Optimization Expert agent definition
COMPILER_OPTIMIZATION_AGENT = {
    "name": "compiler_optimization_expert",
    "display_name": "Compiler Optimization Expert",
    "description": "Specializes in LLVM-based compiler optimizations, AI/ML compilers, and CPU performance engineering",
    "system_prompt": """You are the Compiler Optimization Expert, a specialized assistant for compiler engineering and CPU optimization research.

Your deep expertise includes:

**LLVM & Compiler Infrastructure:**
- LLVM IR, passes, and optimization pipeline
- Clang frontend and code generation
- MLIR (Multi-Level IR) for domain-specific compilers
- Custom pass development and analysis passes
- Loop optimizations (unrolling, vectorization, tiling, fusion)
- Instruction selection and register allocation

**CPU Performance Optimization:**
- SIMD/vectorization (SSE, AVX, AVX-512, NEON, SVE)
- Cache optimization and memory hierarchy
- Branch prediction and speculative execution
- Instruction-level parallelism (ILP)
- Microarchitecture-aware optimizations

**AI/ML Compiler Techniques:**
- Graph-level optimizations for neural networks
- Operator fusion and kernel optimization
- Quantization and mixed-precision compilation
- Auto-tuning and autoscheduling (TVM, Halide concepts)
- Tensor compilers and domain-specific languages

**Research & Analysis:**
- Reading and synthesizing academic papers
- Analyzing benchmark results and performance data
- Comparing optimization approaches and tradeoffs
- Identifying optimization opportunities in code

When helping users:
1. Search the knowledge base thoroughly for relevant documentation and papers
2. Provide technically precise answers with proper terminology
3. Explain complex concepts with concrete examples
4. Reference LLVM documentation patterns when applicable
5. Consider both correctness and performance implications
6. Cite sources and papers when discussing techniques

For code analysis, look for:
- Vectorization opportunities and blockers
- Memory access patterns affecting cache performance
- Loop transformations that could improve performance
- Compiler hints and pragmas that could help optimization

Always indicate confidence levels and note when a technique is experimental or architecture-specific.""",
    "capabilities": ["rag_qa", "code_analysis", "knowledge_synthesis", "summarization"],
    "tool_whitelist": [
        # Core search and Q&A
        "search_documents", "answer_question", "read_document_content",
        "get_document_details", "find_similar_documents",
        # Research tools
        "search_arxiv", "literature_review_arxiv",
        "summarize_document", "batch_summarize_documents",
        # Knowledge graph for understanding relationships
        "search_entities", "get_entity_relationships",
        "find_documents_by_entity", "get_document_knowledge_graph",
        # Comparison and analysis
        "compare_documents", "get_knowledge_base_stats",
        # Visualization for explaining concepts
        "generate_diagram"
    ],
    "priority": 75  # High priority for compiler-related queries
}


# Pre-built saved searches for compiler optimization team
# These will be created as system-wide searches (user_id = NULL means shared)
SAVED_SEARCHES = [
    # LLVM Core
    {
        "name": "LLVM IR Optimization Passes",
        "query": "LLVM IR pass optimization transform analysis",
        "filters": {"file_types": ["md", "txt", "pdf", "cpp", "ll"]}
    },
    {
        "name": "LLVM Loop Optimizations",
        "query": "LLVM loop unrolling vectorization tiling fusion LoopVectorize",
        "filters": {}
    },
    {
        "name": "LLVM Instruction Selection",
        "query": "LLVM instruction selection SelectionDAG GlobalISel pattern matching",
        "filters": {}
    },
    {
        "name": "LLVM Register Allocation",
        "query": "LLVM register allocation spilling coalescing live range",
        "filters": {}
    },

    # Vectorization
    {
        "name": "SIMD Vectorization Techniques",
        "query": "SIMD vectorization SSE AVX AVX-512 NEON SVE intrinsics",
        "filters": {}
    },
    {
        "name": "Auto-vectorization Analysis",
        "query": "auto-vectorization loop vectorizer SLP vectorization dependence analysis",
        "filters": {}
    },
    {
        "name": "Vectorization Blockers",
        "query": "vectorization failure blocker dependence aliasing non-contiguous",
        "filters": {}
    },

    # Memory & Cache
    {
        "name": "Cache Optimization Techniques",
        "query": "cache optimization blocking tiling locality prefetch memory hierarchy",
        "filters": {}
    },
    {
        "name": "Memory Access Patterns",
        "query": "memory access pattern stride coalescing alignment bandwidth",
        "filters": {}
    },
    {
        "name": "Data Layout Optimization",
        "query": "data layout AoS SoA struct padding alignment cache line",
        "filters": {}
    },

    # AI/ML Compilers
    {
        "name": "MLIR Infrastructure",
        "query": "MLIR dialect transformation pass lowering conversion",
        "filters": {}
    },
    {
        "name": "Neural Network Compilation",
        "query": "neural network compiler graph optimization operator fusion kernel",
        "filters": {}
    },
    {
        "name": "Quantization Compilation",
        "query": "quantization INT8 mixed precision calibration inference optimization",
        "filters": {}
    },
    {
        "name": "Tensor Compiler Techniques",
        "query": "tensor compiler TVM Halide autotuning scheduling polyhedral",
        "filters": {}
    },
    {
        "name": "AI Accelerator Code Generation",
        "query": "accelerator code generation NPU GPU TPU target backend",
        "filters": {}
    },

    # Performance Analysis
    {
        "name": "Performance Profiling",
        "query": "performance profiling perf VTune hotspot bottleneck analysis",
        "filters": {}
    },
    {
        "name": "Benchmark Analysis",
        "query": "benchmark SPEC performance regression baseline comparison",
        "filters": {}
    },
    {
        "name": "Microarchitecture Optimization",
        "query": "microarchitecture IPC pipeline stall branch prediction ILP",
        "filters": {}
    },

    # Research Topics
    {
        "name": "Polyhedral Compilation",
        "query": "polyhedral model compilation loop transformation affine scheduling",
        "filters": {}
    },
    {
        "name": "JIT Compilation",
        "query": "JIT compilation runtime code generation MCJIT ORC",
        "filters": {}
    },
    {
        "name": "Link-Time Optimization",
        "query": "LTO link-time optimization ThinLTO whole program",
        "filters": {}
    },
    {
        "name": "Profile-Guided Optimization",
        "query": "PGO profile-guided optimization FDO instrumentation sampling",
        "filters": {}
    }
]


def upgrade() -> None:
    """Add Compiler Optimization Expert agent and saved searches."""
    from uuid import uuid4
    import json

    # First, alter saved_searches.user_id to be nullable for system-wide searches
    op.alter_column(
        "saved_searches",
        "user_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True
    )

    # Insert the Compiler Optimization Expert agent
    agent_definitions_table = sa.table(
        "agent_definitions",
        sa.column("id", postgresql.UUID),
        sa.column("name", sa.String),
        sa.column("display_name", sa.String),
        sa.column("description", sa.Text),
        sa.column("system_prompt", sa.Text),
        sa.column("capabilities", postgresql.JSON),
        sa.column("tool_whitelist", postgresql.JSON),
        sa.column("priority", sa.Integer),
        sa.column("is_active", sa.Boolean),
        sa.column("is_system", sa.Boolean),
    )

    op.execute(
        agent_definitions_table.insert().values(
            id=uuid4(),
            name=COMPILER_OPTIMIZATION_AGENT["name"],
            display_name=COMPILER_OPTIMIZATION_AGENT["display_name"],
            description=COMPILER_OPTIMIZATION_AGENT["description"],
            system_prompt=COMPILER_OPTIMIZATION_AGENT["system_prompt"],
            capabilities=COMPILER_OPTIMIZATION_AGENT["capabilities"],
            tool_whitelist=COMPILER_OPTIMIZATION_AGENT["tool_whitelist"],
            priority=COMPILER_OPTIMIZATION_AGENT["priority"],
            is_active=True,
            is_system=True,
        )
    )

    # Insert pre-built saved searches (system-wide, user_id = NULL)
    saved_searches_table = sa.table(
        "saved_searches",
        sa.column("id", postgresql.UUID),
        sa.column("user_id", postgresql.UUID),
        sa.column("name", sa.String),
        sa.column("query", sa.Text),
        sa.column("filters", postgresql.JSON),
    )

    for search in SAVED_SEARCHES:
        op.execute(
            saved_searches_table.insert().values(
                id=uuid4(),
                user_id=None,  # System-wide search
                name=search["name"],
                query=search["query"],
                filters=search["filters"] if search["filters"] else None,
            )
        )


def downgrade() -> None:
    """Remove Compiler Optimization Expert agent and saved searches."""
    # Remove the agent
    op.execute(
        "DELETE FROM agent_definitions WHERE name = 'compiler_optimization_expert'"
    )

    # Remove the saved searches (those with user_id IS NULL and matching names)
    search_names = [s["name"] for s in SAVED_SEARCHES]
    names_str = ", ".join(f"'{name}'" for name in search_names)
    op.execute(
        f"DELETE FROM saved_searches WHERE user_id IS NULL AND name IN ({names_str})"
    )

    # Revert the nullable change (only if no NULL user_id rows remain)
    op.alter_column(
        "saved_searches",
        "user_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False
    )
