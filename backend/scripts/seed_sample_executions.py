#!/usr/bin/env python3
"""
Seed sample execution history for tools and workflows.

Creates realistic execution history demonstrating:
- Completed workflow runs with node-level results
- Tool execution audit logs
- Various status scenarios (completed, failed, in-progress)

Usage:
    docker compose exec backend python scripts/seed_sample_executions.py
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from uuid import uuid4
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.models.workflow import (
    Workflow, WorkflowNode, WorkflowExecution, WorkflowNodeExecution
)
from app.models.tool_audit import ToolExecutionAudit
from app.models.agent_definition import AgentDefinition
from app.models.memory import AgentConversation


# =============================================================================
# Sample Execution Data
# =============================================================================

WORKFLOW_EXECUTION_SCENARIOS = [
    # Weekly Compiler Research Digest - completed successfully
    {
        "workflow_name": "Weekly Compiler Research Digest",
        "trigger_type": "schedule",
        "status": "completed",
        "days_ago": 7,
        "context": {
            "trigger_data": {"scheduled_at": "2026-01-20T09:00:00Z"},
            "search_results": {
                "arxiv_papers": [
                    {"id": "2601.12345", "title": "MLIR-based Loop Optimization for Modern CPUs", "authors": ["J. Smith", "A. Chen"]},
                    {"id": "2601.12346", "title": "Auto-vectorization Improvements in LLVM 19", "authors": ["M. Johnson"]},
                    {"id": "2601.12347", "title": "Polyhedral Analysis for Tensor Compilers", "authors": ["L. Wang", "K. Lee"]},
                ],
                "internal_docs": [
                    {"id": "doc-001", "title": "Q4 Performance Review", "relevance": 0.92},
                    {"id": "doc-002", "title": "SIMD Optimization Guidelines", "relevance": 0.88},
                ]
            },
            "digest_content": """# Weekly Compiler Research Digest
## January 20, 2026

### Key Papers This Week

1. **MLIR-based Loop Optimization for Modern CPUs** (arXiv:2601.12345)
   - Authors: J. Smith, A. Chen
   - Key insight: New loop tiling strategy improves cache utilization by 35%

2. **Auto-vectorization Improvements in LLVM 19** (arXiv:2601.12346)
   - Author: M. Johnson
   - Relevance: Direct applicability to our vectorization pipeline

3. **Polyhedral Analysis for Tensor Compilers** (arXiv:2601.12347)
   - Authors: L. Wang, K. Lee
   - Application: Could enhance our tensor optimization passes

### Internal Updates
- Q4 Performance Review completed
- SIMD Optimization Guidelines updated

### Recommended Actions
- Review MLIR loop optimization techniques for integration
- Schedule deep-dive on LLVM 19 auto-vectorization changes
"""
        },
        "node_results": [
            {"node_id": "start", "status": "completed", "output": {}, "time_ms": 5},
            {"node_id": "search_internal", "status": "completed", "output": {"doc_count": 12, "top_score": 0.92}, "time_ms": 1250},
            {"node_id": "search_arxiv", "status": "completed", "output": {"paper_count": 8, "filtered": 3}, "time_ms": 2100},
            {"node_id": "summarize", "status": "completed", "output": {"summary_length": 1850}, "time_ms": 4500},
            {"node_id": "format_digest", "status": "completed", "output": {"format": "markdown", "sections": 4}, "time_ms": 320},
            {"node_id": "end", "status": "completed", "output": {"notification_sent": True}, "time_ms": 10},
        ]
    },
    # Benchmark Analysis Pipeline - completed
    {
        "workflow_name": "Benchmark Analysis Pipeline",
        "trigger_type": "manual",
        "status": "completed",
        "days_ago": 3,
        "context": {
            "trigger_data": {
                "benchmark_name": "SPEC CPU 2017",
                "comparison": "baseline_v2.1 vs optimized_v2.2"
            },
            "benchmark_results": {
                "baseline": {"geomean": 45.2, "peak": 62.1},
                "optimized": {"geomean": 52.8, "peak": 71.3},
                "improvement": {"geomean": "+16.8%", "peak": "+14.8%"}
            },
            "analysis_report": """# Benchmark Analysis Report
## SPEC CPU 2017: baseline_v2.1 vs optimized_v2.2

### Executive Summary
The optimized compiler version shows a **16.8% geomean improvement** across SPEC CPU 2017 benchmarks.

### Key Improvements
| Benchmark | Baseline | Optimized | Improvement |
|-----------|----------|-----------|-------------|
| 500.perlbench_r | 8.2 | 9.1 | +11.0% |
| 502.gcc_r | 12.4 | 15.8 | +27.4% |
| 505.mcf_r | 9.8 | 10.9 | +11.2% |
| 520.omnetpp_r | 7.1 | 8.2 | +15.5% |
| 525.x264_r | 18.5 | 21.2 | +14.6% |

### Regression Analysis
- No significant regressions detected
- 502.gcc_r shows highest improvement due to loop unrolling enhancements

### Recommendations
1. Proceed with release of v2.2
2. Investigate 502.gcc_r patterns for broader application
3. Continue monitoring floating-point benchmarks
"""
        },
        "node_results": [
            {"node_id": "start", "status": "completed", "output": {}, "time_ms": 3},
            {"node_id": "parse_input", "status": "completed", "output": {"benchmarks": 12, "metrics": 5}, "time_ms": 150},
            {"node_id": "fetch_baseline", "status": "completed", "output": {"records": 48}, "time_ms": 850},
            {"node_id": "fetch_comparison", "status": "completed", "output": {"records": 48}, "time_ms": 780},
            {"node_id": "calculate_deltas", "status": "completed", "output": {"improvements": 11, "regressions": 1}, "time_ms": 120},
            {"node_id": "generate_report", "status": "completed", "output": {"report_length": 1240, "charts": 3}, "time_ms": 3200},
            {"node_id": "end", "status": "completed", "output": {}, "time_ms": 5},
        ]
    },
    # Optimization Technique Research - completed
    {
        "workflow_name": "Optimization Technique Research",
        "trigger_type": "manual",
        "status": "completed",
        "days_ago": 5,
        "context": {
            "trigger_data": {
                "technique": "Loop Vectorization",
                "focus_areas": ["auto-vectorization", "SLP", "outer-loop vectorization"]
            },
            "research_summary": """# Loop Vectorization Research Summary

## Technique Overview
Loop vectorization transforms scalar loop operations into SIMD vector operations, enabling parallel processing of multiple data elements.

## Key Approaches

### 1. Auto-vectorization (Inner Loops)
- **Mechanism**: Compiler automatically detects vectorizable loops
- **Requirements**: No loop-carried dependencies, aligned memory access
- **LLVM Implementation**: LoopVectorize pass

### 2. SLP Vectorization (Superword Level Parallelism)
- **Mechanism**: Packs similar independent operations into vectors
- **Scope**: Basic blocks rather than loops
- **LLVM Implementation**: SLPVectorizer pass

### 3. Outer-loop Vectorization
- **Mechanism**: Vectorizes outer loops when inner loops are not profitable
- **Challenges**: More complex dependency analysis
- **Status**: Limited support in current compilers

## Recent Advances
1. VPlan-based vectorization in LLVM (more flexible cost modeling)
2. Predicated vectorization for irregular loops
3. Vector Function ABI for math library calls

## Relevant Papers
- "VPlan: Unified Vectorization in LLVM" (CGO 2023)
- "Outer-loop Vectorization Revisited" (PLDI 2024)
- "Efficient SLP Vectorization" (CC 2025)

## Recommendations
1. Investigate VPlan improvements in LLVM trunk
2. Evaluate outer-loop vectorization for our tensor kernels
3. Review math library vectorization opportunities
"""
        },
        "node_results": [
            {"node_id": "start", "status": "completed", "output": {}, "time_ms": 2},
            {"node_id": "search_internal", "status": "completed", "output": {"matches": 23}, "time_ms": 980},
            {"node_id": "search_arxiv", "status": "completed", "output": {"papers": 15}, "time_ms": 1850},
            {"node_id": "analyze", "status": "completed", "output": {"key_insights": 8}, "time_ms": 5200},
            {"node_id": "generate_report", "status": "completed", "output": {"sections": 6, "recommendations": 3}, "time_ms": 4100},
            {"node_id": "end", "status": "completed", "output": {}, "time_ms": 3},
        ]
    },
    # Prior Art Search - completed
    {
        "workflow_name": "Prior Art Search",
        "trigger_type": "manual",
        "status": "completed",
        "days_ago": 2,
        "context": {
            "trigger_data": {
                "topic": "Register Allocation with Machine Learning",
                "scope": "academic and patent"
            },
            "prior_art_results": {
                "academic_papers": 12,
                "patents": 5,
                "key_references": [
                    "Leather & Fursin, 'ML Compiler Optimization' (2020)",
                    "US Patent 10,234,567 - Neural Register Allocation",
                    "arXiv:2312.05678 - RL for Register Allocation"
                ]
            }
        },
        "node_results": [
            {"node_id": "start", "status": "completed", "output": {}, "time_ms": 2},
            {"node_id": "search_internal", "status": "completed", "output": {"matches": 5}, "time_ms": 720},
            {"node_id": "search_arxiv", "status": "completed", "output": {"papers": 12}, "time_ms": 1650},
            {"node_id": "check_papers", "status": "completed", "output": {"has_papers": True}, "time_ms": 15},
            {"node_id": "wait_ingestion", "status": "completed", "output": {"ingested": 3}, "time_ms": 8500},
            {"node_id": "generate_review", "status": "completed", "output": {"review_length": 2100}, "time_ms": 4800},
            {"node_id": "end", "status": "completed", "output": {}, "time_ms": 5},
        ]
    },
    # Failed workflow example
    {
        "workflow_name": "Weekly Compiler Research Digest",
        "trigger_type": "schedule",
        "status": "failed",
        "days_ago": 14,
        "error": "ArXiv API rate limit exceeded. Retry after 60 seconds.",
        "context": {
            "trigger_data": {"scheduled_at": "2026-01-13T09:00:00Z"},
        },
        "node_results": [
            {"node_id": "start", "status": "completed", "output": {}, "time_ms": 4},
            {"node_id": "search_internal", "status": "completed", "output": {"doc_count": 8}, "time_ms": 1100},
            {"node_id": "search_arxiv", "status": "failed", "error": "ArXiv API rate limit exceeded", "time_ms": 5200},
        ]
    },
    # In-progress workflow
    {
        "workflow_name": "Research Presentation Builder",
        "trigger_type": "manual",
        "status": "running",
        "days_ago": 0,
        "progress": 60,
        "current_node": "generate_slides",
        "context": {
            "trigger_data": {
                "topic": "LLVM MLIR for Domain-Specific Compilers",
                "slide_count": 12
            },
            "search_results": {"papers": 8, "internal_docs": 5},
        },
        "node_results": [
            {"node_id": "start", "status": "completed", "output": {}, "time_ms": 3},
            {"node_id": "search_internal", "status": "completed", "output": {"matches": 5}, "time_ms": 920},
            {"node_id": "search_arxiv", "status": "completed", "output": {"papers": 8}, "time_ms": 1780},
            {"node_id": "generate_slides", "status": "running", "output": None, "time_ms": None},
        ]
    },
]


TOOL_EXECUTION_SAMPLES = [
    # LLM Prompt tools
    {
        "tool_name": "explain_optimization",
        "input": {"optimization_name": "Loop Unrolling", "context": "LLVM backend"},
        "output": {
            "explanation": """## Loop Unrolling in LLVM

**What it does**: Loop unrolling duplicates the loop body multiple times, reducing loop overhead and enabling better instruction scheduling.

**How LLVM implements it**:
1. The `LoopUnrollPass` analyzes loop trip counts and costs
2. Full unrolling for small, known-count loops
3. Partial unrolling (unroll factor 2-8) for larger loops
4. Runtime unrolling when trip count is unknown

**When it helps**:
- Reduces branch mispredictions
- Enables SIMD opportunities
- Better register utilization

**Trade-offs**:
- Increased code size (I-cache pressure)
- May hurt if unroll factor is wrong
""",
            "tokens_used": 285
        },
        "status": "completed",
        "execution_time_ms": 2150,
        "hours_ago": 2
    },
    {
        "tool_name": "analyze_llvm_ir",
        "input": {
            "ir_snippet": """define i32 @sum_array(ptr %arr, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit
loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %add, %loop ]
  %ptr = getelementptr i32, ptr %arr, i32 %i
  %val = load i32, ptr %ptr
  %add = add nsw i32 %sum, %val
  %inc = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %inc, %n
  br i1 %done, label %exit, label %loop
exit:
  %result = phi i32 [ 0, %entry ], [ %add, %loop ]
  ret i32 %result
}""",
            "focus": "vectorization potential"
        },
        "output": {
            "analysis": """## LLVM IR Analysis: sum_array

### Vectorization Assessment: **HIGH POTENTIAL**

**Loop characteristics**:
- Simple reduction pattern (sum accumulation)
- Linear memory access with stride 1
- No loop-carried dependencies except reduction
- Trip count available at runtime (`%n`)

**Vectorization strategy**:
```
; Vector factor: 4 (SSE) or 8 (AVX2)
; Reduction: horizontal add at loop exit
; Remainder: scalar epilogue
```

**Optimization flags**:
- `-mllvm -force-vector-width=8` for AVX2
- Add `!llvm.loop.vectorize.enable` metadata

**Expected speedup**: 3-4x with AVX2
""",
            "vectorizable": True,
            "recommended_vf": 8
        },
        "status": "completed",
        "execution_time_ms": 1850,
        "hours_ago": 5
    },
    {
        "tool_name": "generate_benchmark_report",
        "input": {
            "benchmark_data": {
                "name": "matrix_multiply",
                "baseline_gflops": 45.2,
                "optimized_gflops": 78.5,
                "cpu": "Intel Xeon Gold 6248"
            }
        },
        "output": {
            "report": """## Benchmark Report: matrix_multiply

**Test Configuration**:
- CPU: Intel Xeon Gold 6248 (20 cores, 2.5 GHz)
- Compiler: Custom LLVM 18 fork

**Results**:
| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| GFLOPS | 45.2 | 78.5 | **+73.7%** |
| Efficiency | 28.3% | 49.1% | +20.8pp |

**Analysis**:
The optimized version achieves 49.1% of theoretical peak,
indicating effective vectorization and cache blocking.

**Key optimizations applied**:
1. AVX-512 vectorization (8-wide)
2. Loop tiling (32x32 blocks)
3. Register blocking (4x4 micro-kernel)
""",
            "improvement_pct": 73.7
        },
        "status": "completed",
        "execution_time_ms": 980,
        "hours_ago": 8
    },
    {
        "tool_name": "paper_key_points",
        "input": {
            "paper_title": "VPlan: Unified Vectorization in LLVM",
            "abstract": "We present VPlan, a new vectorization planning infrastructure..."
        },
        "output": {
            "key_points": [
                "VPlan introduces a unified representation for vectorization decisions",
                "Enables cost-model-driven exploration of multiple vectorization strategies",
                "Supports predicated vectorization for irregular loops",
                "Improves maintainability by separating planning from code generation",
                "Shows 5-15% performance improvement on SPEC benchmarks"
            ],
            "relevance_score": 0.92,
            "recommended_action": "Schedule team review - directly applicable to our vectorizer"
        },
        "status": "completed",
        "execution_time_ms": 1650,
        "hours_ago": 12
    },
    # Transform tools
    {
        "tool_name": "calculate_speedup",
        "input": {"baseline_time": 12.5, "optimized_time": 3.2},
        "output": {"speedup": 3.91, "improvement_pct": 74.4},
        "status": "completed",
        "execution_time_ms": 15,
        "hours_ago": 1
    },
    {
        "tool_name": "format_benchmark_json",
        "input": {
            "raw_data": "benchmark,baseline,optimized\nmatmul,45.2,78.5\nconv2d,32.1,58.9"
        },
        "output": {
            "formatted": {
                "benchmarks": [
                    {"name": "matmul", "baseline": 45.2, "optimized": 78.5, "speedup": 1.74},
                    {"name": "conv2d", "baseline": 32.1, "optimized": 58.9, "speedup": 1.83}
                ],
                "summary": {"geomean_speedup": 1.78}
            }
        },
        "status": "completed",
        "execution_time_ms": 45,
        "hours_ago": 3
    },
    # Failed tool execution
    {
        "tool_name": "notify_slack",
        "input": {
            "channel": "#compiler-team",
            "message": "Weekly digest ready"
        },
        "output": None,
        "error": "Webhook URL not configured. Please set SLACK_WEBHOOK_URL in secrets.",
        "status": "failed",
        "execution_time_ms": 120,
        "hours_ago": 24
    },
    # More recent executions
    {
        "tool_name": "compare_techniques",
        "input": {
            "technique_a": "Loop Fusion",
            "technique_b": "Loop Fission"
        },
        "output": {
            "comparison": """## Loop Fusion vs Loop Fission

| Aspect | Loop Fusion | Loop Fission |
|--------|-------------|--------------|
| **Purpose** | Merge adjacent loops | Split loop into multiple |
| **Cache benefit** | Better temporal locality | Better spatial locality |
| **Register pressure** | Higher | Lower |
| **Parallelization** | May inhibit | Often enables |
| **Best for** | Sequential access patterns | Large working sets |

**Recommendation**: Consider loop fusion when loops access the same data;
use fission when register pressure is high or to enable vectorization.
""",
            "recommended": "context-dependent"
        },
        "status": "completed",
        "execution_time_ms": 1920,
        "hours_ago": 6
    },
    {
        "tool_name": "extract_metrics",
        "input": {
            "log_content": """
[PERF] matmul_kernel: 1250 cycles, 128 cache_misses, IPC=2.3
[PERF] conv2d_kernel: 3200 cycles, 512 cache_misses, IPC=1.8
[PERF] relu_kernel: 180 cycles, 4 cache_misses, IPC=3.1
"""
        },
        "output": {
            "metrics": [
                {"kernel": "matmul_kernel", "cycles": 1250, "cache_misses": 128, "ipc": 2.3},
                {"kernel": "conv2d_kernel", "cycles": 3200, "cache_misses": 512, "ipc": 1.8},
                {"kernel": "relu_kernel", "cycles": 180, "cache_misses": 4, "ipc": 3.1}
            ],
            "summary": {
                "total_cycles": 4630,
                "avg_ipc": 2.4,
                "cache_miss_rate": "estimated 2.1%"
            }
        },
        "status": "completed",
        "execution_time_ms": 85,
        "hours_ago": 4
    },
]


async def seed_workflow_executions(db: AsyncSession, user_id, workflows: dict):
    """Create sample workflow execution history."""
    created_count = 0

    for scenario in WORKFLOW_EXECUTION_SCENARIOS:
        workflow_name = scenario["workflow_name"]
        if workflow_name not in workflows:
            print(f"  ⚠ Workflow '{workflow_name}' not found, skipping")
            continue

        workflow = workflows[workflow_name]

        # Calculate timestamps
        base_time = datetime.utcnow() - timedelta(days=scenario["days_ago"])
        started_at = base_time

        # Sum up node execution times for completion time
        total_time_ms = sum(
            nr.get("time_ms", 0) or 0
            for nr in scenario.get("node_results", [])
        )
        completed_at = base_time + timedelta(milliseconds=total_time_ms) if scenario["status"] in ("completed", "failed") else None

        # Create execution record
        execution = WorkflowExecution(
            id=uuid4(),
            workflow_id=workflow.id,
            user_id=user_id,
            trigger_type=scenario["trigger_type"],
            trigger_data=scenario.get("context", {}).get("trigger_data", {}),
            status=scenario["status"],
            progress=scenario.get("progress", 100 if scenario["status"] == "completed" else 0),
            current_node_id=scenario.get("current_node"),
            context=scenario.get("context", {}),
            error=scenario.get("error"),
            created_at=base_time - timedelta(seconds=5),
            started_at=started_at,
            completed_at=completed_at
        )
        db.add(execution)
        await db.flush()

        # Create node execution records
        node_time_offset = 0
        for node_result in scenario.get("node_results", []):
            node_started = started_at + timedelta(milliseconds=node_time_offset)
            exec_time = node_result.get("time_ms")
            node_completed = None
            if exec_time and node_result["status"] in ("completed", "failed"):
                node_completed = node_started + timedelta(milliseconds=exec_time)
                node_time_offset += exec_time

            node_exec = WorkflowNodeExecution(
                id=uuid4(),
                execution_id=execution.id,
                node_id=node_result["node_id"],
                status=node_result["status"],
                input_data=node_result.get("input"),
                output_data=node_result.get("output"),
                error=node_result.get("error"),
                execution_time_ms=exec_time,
                started_at=node_started,
                completed_at=node_completed
            )
            db.add(node_exec)

        created_count += 1
        print(f"  ✓ Created execution for '{workflow_name}' ({scenario['status']})")

    await db.commit()
    return created_count


async def seed_tool_executions(db: AsyncSession, user_id, agent_id=None, conversation_id=None):
    """Create sample tool execution audit logs."""
    created_count = 0

    for sample in TOOL_EXECUTION_SAMPLES:
        exec_time = datetime.utcnow() - timedelta(hours=sample["hours_ago"])

        audit = ToolExecutionAudit(
            id=uuid4(),
            user_id=user_id,
            agent_definition_id=agent_id,
            conversation_id=conversation_id,
            tool_name=sample["tool_name"],
            tool_input=sample["input"],
            tool_output=sample.get("output"),
            status=sample["status"],
            error=sample.get("error"),
            execution_time_ms=sample["execution_time_ms"],
            approval_required=False,
            created_at=exec_time,
            updated_at=exec_time
        )
        db.add(audit)
        created_count += 1

    await db.commit()
    print(f"  ✓ Created {created_count} tool execution audit records")
    return created_count


async def main():
    print("=" * 60)
    print("Seeding Sample Execution History")
    print("=" * 60)

    async with AsyncSessionLocal() as db:
        # Get admin user
        print("\n[1/4] Finding admin user...")
        result = await db.execute(
            select(User).where(User.username == "admin")
        )
        user = result.scalar_one_or_none()

        if not user:
            print("  ✗ Admin user not found. Run seed_tools_workflows.py first.")
            return
        print(f"  Using user: {user.username} ({user.id})")

        # Get workflows
        print("\n[2/4] Loading workflows...")
        result = await db.execute(
            select(Workflow).where(Workflow.user_id == user.id)
        )
        workflows_list = result.scalars().all()
        workflows = {w.name: w for w in workflows_list}
        print(f"  Found {len(workflows)} workflows")

        # Get compiler expert agent (if exists)
        print("\n[3/4] Finding Compiler Optimization Expert agent...")
        result = await db.execute(
            select(AgentDefinition).where(AgentDefinition.name == "compiler_optimization_expert")
        )
        agent = result.scalar_one_or_none()
        agent_id = agent.id if agent else None

        # Check for existing conversation
        conversation_id = None
        if agent_id:
            result = await db.execute(
                select(AgentConversation)
                .where(AgentConversation.active_agent_id == agent_id)
                .limit(1)
            )
            conv = result.scalar_one_or_none()
            conversation_id = conv.id if conv else None

        print(f"  Agent ID: {agent_id}")
        print(f"  Conversation ID: {conversation_id}")

        # Seed workflow executions
        print("\n[4/4] Creating sample data...")
        print("\nWorkflow Executions:")
        exec_count = await seed_workflow_executions(db, user.id, workflows)

        print("\nTool Execution Audits:")
        tool_count = await seed_tool_executions(db, user.id, agent_id, conversation_id)

        print("\n" + "-" * 60)
        print("Summary:")
        print(f"  Workflow Executions: {exec_count} created")
        print(f"  Tool Audit Records:  {tool_count} created")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
