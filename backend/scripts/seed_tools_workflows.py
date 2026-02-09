#!/usr/bin/env python3
"""
Seed useful tools and workflows for Compiler Optimization R&D team demo.

Creates:
- Custom user tools (LLM prompts, transforms, webhooks)
- Pre-configured workflows from templates

Usage:
    docker compose exec backend python scripts/seed_tools_workflows.py
"""

import asyncio
import sys
from uuid import uuid4
from datetime import datetime

sys.path.insert(0, "/app")

from sqlalchemy import select
from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.models.workflow import UserTool, Workflow, WorkflowNode, WorkflowEdge


# =============================================================================
# Custom User Tools
# =============================================================================

USER_TOOLS = [
    # -------------------------------------------------------------------------
    # LLM Prompt Tools
    # -------------------------------------------------------------------------
    {
        "name": "explain_optimization",
        "description": "Explain a compiler optimization technique in simple terms with examples",
        "tool_type": "llm_prompt",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "technique": {
                    "type": "string",
                    "description": "The optimization technique to explain"
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "expert"],
                    "default": "intermediate"
                }
            },
            "required": ["technique"]
        },
        "config": {
            "system_prompt": """You are a compiler optimization expert. Explain compiler optimizations clearly and accurately.
Include:
- What the optimization does
- When it's applied
- Performance benefits
- Example code showing before/after
- Any limitations or trade-offs""",
            "user_prompt": """Explain the {{technique}} optimization at a {{detail_level}} level.

Provide a clear explanation with concrete examples.""",
            "output_format": "text",
            "temperature": 0.3,
            "max_tokens": 1500
        }
    },
    {
        "name": "analyze_llvm_ir",
        "description": "Analyze LLVM IR code and suggest optimizations",
        "tool_type": "llm_prompt",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "llvm_ir": {
                    "type": "string",
                    "description": "LLVM IR code to analyze"
                },
                "optimization_goal": {
                    "type": "string",
                    "enum": ["performance", "size", "both"],
                    "default": "performance"
                }
            },
            "required": ["llvm_ir"]
        },
        "config": {
            "system_prompt": """You are an LLVM compiler expert. Analyze LLVM IR and provide optimization insights.

Focus on:
- Identifying optimization opportunities
- Suggesting LLVM passes that could help
- Explaining why certain patterns prevent optimization
- Recommending source-level changes""",
            "user_prompt": """Analyze this LLVM IR with a focus on {{optimization_goal}} optimization:

```llvm
{{llvm_ir}}
```

Provide specific optimization recommendations.""",
            "output_format": "text",
            "temperature": 0.2,
            "max_tokens": 2000
        }
    },
    {
        "name": "generate_benchmark_report",
        "description": "Generate a structured benchmark analysis report",
        "tool_type": "llm_prompt",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "benchmark_name": {"type": "string"},
                "results_summary": {"type": "string"},
                "baseline_comparison": {"type": "string"}
            },
            "required": ["benchmark_name", "results_summary"]
        },
        "config": {
            "system_prompt": """You are a performance analyst. Generate clear, actionable benchmark reports.""",
            "user_prompt": """Generate a benchmark analysis report:

Benchmark: {{benchmark_name}}
Results: {{results_summary}}
{% if baseline_comparison %}Baseline Comparison: {{baseline_comparison}}{% endif %}

Structure the report with:
1. Executive Summary
2. Key Metrics
3. Performance Analysis
4. Regression/Improvement Analysis
5. Recommendations""",
            "output_format": "text",
            "temperature": 0.3,
            "max_tokens": 2000
        }
    },
    {
        "name": "compare_techniques",
        "description": "Compare two or more optimization techniques",
        "tool_type": "llm_prompt",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "techniques": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of techniques to compare"
                },
                "criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["performance", "complexity", "applicability"]
                }
            },
            "required": ["techniques"]
        },
        "config": {
            "system_prompt": """You are a compiler optimization expert. Provide objective comparisons of optimization techniques.""",
            "user_prompt": """Compare these optimization techniques: {{techniques | join(', ')}}

Evaluation criteria: {{criteria | join(', ')}}

Provide a structured comparison with:
- Overview of each technique
- Comparison table
- Pros and cons
- Use case recommendations""",
            "output_format": "text",
            "temperature": 0.3,
            "max_tokens": 2000
        }
    },
    {
        "name": "paper_key_points",
        "description": "Extract key points from a research paper abstract",
        "tool_type": "llm_prompt",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "abstract": {"type": "string"},
                "focus_area": {"type": "string", "default": "compiler optimization"}
            },
            "required": ["title", "abstract"]
        },
        "config": {
            "system_prompt": """You are a research analyst specializing in {{focus_area}}. Extract actionable insights from papers.""",
            "user_prompt": """Extract key points from this paper:

Title: {{title}}
Abstract: {{abstract}}

Provide:
1. Main contribution (1-2 sentences)
2. Key techniques introduced
3. Reported results/improvements
4. Potential applications
5. Limitations mentioned""",
            "output_format": "json",
            "temperature": 0.2,
            "max_tokens": 1000
        }
    },

    # -------------------------------------------------------------------------
    # Transform Tools
    # -------------------------------------------------------------------------
    {
        "name": "format_benchmark_json",
        "description": "Transform benchmark results into a standardized JSON format",
        "tool_type": "transform",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "raw_results": {"type": "string"},
                "benchmark_name": {"type": "string"},
                "timestamp": {"type": "string"}
            },
            "required": ["raw_results", "benchmark_name"]
        },
        "config": {
            "transform_type": "jinja2",
            "template": """{
  "benchmark": "{{benchmark_name}}",
  "timestamp": "{{timestamp or 'now'}}",
  "results": {{raw_results}},
  "metadata": {
    "generated_by": "KnowledgeDB",
    "version": "1.0"
  }
}"""
        }
    },
    {
        "name": "extract_metrics",
        "description": "Extract specific metrics from benchmark output using JSONPath",
        "tool_type": "transform",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "data": {"type": "object"},
                "metric_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "JSONPath expressions for metrics to extract"
                }
            },
            "required": ["data", "metric_paths"]
        },
        "config": {
            "transform_type": "jsonpath",
            "template": "{{metric_paths}}"
        }
    },
    {
        "name": "markdown_to_slides",
        "description": "Transform markdown content into slide structure",
        "tool_type": "transform",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "markdown": {"type": "string"},
                "max_bullets_per_slide": {"type": "integer", "default": 5}
            },
            "required": ["markdown"]
        },
        "config": {
            "transform_type": "jinja2",
            "template": """{% set sections = markdown.split('## ') %}
{
  "slides": [
    {% for section in sections if section.strip() %}
    {
      "title": "{{ section.split('\\n')[0] }}",
      "bullets": [
        {% for line in section.split('\\n')[1:] if line.strip().startswith('-') %}
        "{{ line.strip()[2:] }}"{% if not loop.last %},{% endif %}
        {% endfor %}
      ]
    }{% if not loop.last %},{% endif %}
    {% endfor %}
  ]
}"""
        }
    },

    # -------------------------------------------------------------------------
    # Webhook Tools (examples - URLs are placeholders)
    # -------------------------------------------------------------------------
    {
        "name": "notify_slack",
        "description": "Send notification to Slack channel (configure webhook URL)",
        "tool_type": "webhook",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "channel": {"type": "string", "default": "#compiler-team"}
            },
            "required": ["message"]
        },
        "config": {
            "method": "POST",
            "url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            "headers": {"Content-Type": "application/json"},
            "body_template": """{
  "channel": "{{channel}}",
  "text": "{{message}}",
  "username": "KnowledgeDB Bot"
}""",
            "timeout_seconds": 10
        }
    },
    {
        "name": "trigger_ci_pipeline",
        "description": "Trigger CI/CD pipeline for benchmark run",
        "tool_type": "webhook",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "default": "main"},
                "benchmark_suite": {"type": "string"},
                "parameters": {"type": "object"}
            },
            "required": ["benchmark_suite"]
        },
        "config": {
            "method": "POST",
            "url": "https://gitlab.example.com/api/v4/projects/ID/trigger/pipeline",
            "headers": {
                "Content-Type": "application/json",
                "PRIVATE-TOKEN": "{{env.GITLAB_TOKEN}}"
            },
            "body_template": """{
  "ref": "{{branch}}",
  "variables": {
    "BENCHMARK_SUITE": "{{benchmark_suite}}",
    "EXTRA_PARAMS": "{{parameters | tojson}}"
  }
}""",
            "timeout_seconds": 30
        }
    },

    # -------------------------------------------------------------------------
    # Python Tools
    # -------------------------------------------------------------------------
    {
        "name": "calculate_speedup",
        "description": "Calculate speedup and efficiency metrics from benchmark data",
        "tool_type": "python",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "baseline_time": {"type": "number"},
                "optimized_time": {"type": "number"},
                "num_threads": {"type": "integer", "default": 1}
            },
            "required": ["baseline_time", "optimized_time"]
        },
        "config": {
            "code": """
# Calculate performance metrics
baseline = params.get('baseline_time')
optimized = params.get('optimized_time')
threads = params.get('num_threads', 1)

if optimized <= 0:
    return {"error": "Invalid optimized time"}

speedup = baseline / optimized
efficiency = speedup / threads if threads > 1 else 1.0
improvement_pct = ((baseline - optimized) / baseline) * 100

result = {
    "speedup": round(speedup, 2),
    "efficiency": round(efficiency, 2),
    "improvement_percent": round(improvement_pct, 1),
    "baseline_time": baseline,
    "optimized_time": optimized,
    "threads": threads
}
return result
""",
            "timeout_seconds": 5,
            "allowed_imports": ["json", "math"]
        }
    },
    {
        "name": "parse_perf_output",
        "description": "Parse Linux perf stat output into structured data",
        "tool_type": "python",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "perf_output": {"type": "string"}
            },
            "required": ["perf_output"]
        },
        "config": {
            "code": """
import re

output = params.get('perf_output', '')
metrics = {}

# Parse common perf metrics
patterns = {
    'cycles': r'([\\d,]+)\\s+cycles',
    'instructions': r'([\\d,]+)\\s+instructions',
    'cache_misses': r'([\\d,]+)\\s+cache-misses',
    'cache_references': r'([\\d,]+)\\s+cache-references',
    'branch_misses': r'([\\d,]+)\\s+branch-misses',
    'branches': r'([\\d,]+)\\s+branches',
    'time_elapsed': r'([\\d.]+)\\s+seconds time elapsed',
    'ipc': r'([\\d.]+)\\s+insn per cycle'
}

for name, pattern in patterns.items():
    match = re.search(pattern, output)
    if match:
        value = match.group(1).replace(',', '')
        metrics[name] = float(value) if '.' in value else int(value)

# Calculate derived metrics
if 'cycles' in metrics and 'instructions' in metrics:
    metrics['ipc_calculated'] = round(metrics['instructions'] / metrics['cycles'], 2)

if 'cache_misses' in metrics and 'cache_references' in metrics:
    metrics['cache_miss_rate'] = round(metrics['cache_misses'] / metrics['cache_references'] * 100, 2)

return metrics
""",
            "timeout_seconds": 5,
            "allowed_imports": ["re", "json"]
        }
    },
    {
        "name": "generate_comparison_table",
        "description": "Generate a markdown comparison table from benchmark data",
        "tool_type": "python",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "benchmarks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "baseline": {"type": "number"},
                            "optimized": {"type": "number"}
                        }
                    }
                },
                "metric_name": {"type": "string", "default": "Time (ms)"}
            },
            "required": ["benchmarks"]
        },
        "config": {
            "code": """
benchmarks = params.get('benchmarks', [])
metric = params.get('metric_name', 'Time (ms)')

lines = [
    f"| Benchmark | Baseline ({metric}) | Optimized ({metric}) | Speedup | Change |",
    "|-----------|---------------------|----------------------|---------|--------|"
]

for b in benchmarks:
    name = b.get('name', 'Unknown')
    baseline = b.get('baseline', 0)
    optimized = b.get('optimized', 0)

    if optimized > 0:
        speedup = baseline / optimized
        change = ((baseline - optimized) / baseline) * 100
        change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
    else:
        speedup = 0
        change_str = "N/A"

    lines.append(f"| {name} | {baseline:.2f} | {optimized:.2f} | {speedup:.2f}x | {change_str} |")

return "\\n".join(lines)
""",
            "timeout_seconds": 5,
            "allowed_imports": ["json"]
        }
    }
]


# =============================================================================
# Pre-configured Workflows
# =============================================================================

WORKFLOWS = [
    {
        "name": "Weekly Compiler Research Digest",
        "description": "Automated weekly digest of new compiler optimization research from arXiv and internal docs",
        "template_id": "cpu_research_monitor",
        "trigger_config": {
            "type": "schedule",
            "schedule": "0 9 * * 1"  # Monday 9 AM
        }
    },
    {
        "name": "Benchmark Analysis Pipeline",
        "description": "Analyze benchmark results and generate reports with regression detection",
        "template_id": "benchmark_analysis",
        "trigger_config": {"type": "manual"}
    },
    {
        "name": "Optimization Technique Research",
        "description": "Deep dive research on specific optimization techniques",
        "template_id": "optimization_deep_dive",
        "trigger_config": {"type": "manual"}
    },
    {
        "name": "Architecture Decision Record",
        "description": "Generate ADRs for optimization decisions with context and alternatives",
        "template_id": "adr_generator",
        "trigger_config": {"type": "manual"}
    },
    {
        "name": "Research Presentation Builder",
        "description": "Create presentations from research topics with arXiv integration",
        "template_id": "research_presentation",
        "trigger_config": {"type": "manual"}
    },
    {
        "name": "Prior Art Search",
        "description": "Search for prior art and existing research for patent/novelty analysis",
        "template_id": "prior_art_research",
        "trigger_config": {"type": "manual"}
    },
    {
        "name": "Quick Paper Brief",
        "description": "Generate a quick technical brief from an arXiv paper",
        "template_id": "quick_paper_brief",
        "trigger_config": {"type": "manual"}
    }
]


async def get_or_create_admin_user(db) -> User:
    """Get or create an admin user for owning system tools/workflows."""
    result = await db.execute(
        select(User).where(User.username == "admin")
    )
    admin = result.scalar_one_or_none()

    if not admin:
        # Try to find any admin user
        result = await db.execute(
            select(User).where(User.role == "admin").limit(1)
        )
        admin = result.scalar_one_or_none()

    if not admin:
        # Create admin user
        from app.core.security import get_password_hash
        admin = User(
            id=uuid4(),
            username="admin",
            email="admin@example.com",
            hashed_password=get_password_hash("admin123"),
            role="admin",
            is_active=True
        )
        db.add(admin)
        await db.commit()
        await db.refresh(admin)
        print("  Created admin user (admin / admin123)")

    return admin


async def seed_tools(db, user: User):
    """Seed user tools."""
    created = []
    skipped = []

    for tool_def in USER_TOOLS:
        # Check if exists
        result = await db.execute(
            select(UserTool).where(
                UserTool.user_id == user.id,
                UserTool.name == tool_def["name"]
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            skipped.append(tool_def["name"])
            continue

        tool = UserTool(
            id=uuid4(),
            user_id=user.id,
            name=tool_def["name"],
            description=tool_def["description"],
            tool_type=tool_def["tool_type"],
            parameters_schema=tool_def["parameters_schema"],
            config=tool_def["config"],
            is_enabled=True,
            version=1
        )
        db.add(tool)
        created.append(tool_def["name"])

    await db.commit()
    return created, skipped


async def seed_workflows(db, user: User):
    """Seed workflows from templates."""
    from app.services.workflow_templates import get_template_by_id

    created = []
    skipped = []

    for wf_def in WORKFLOWS:
        # Check if exists
        result = await db.execute(
            select(Workflow).where(
                Workflow.user_id == user.id,
                Workflow.name == wf_def["name"]
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            skipped.append(wf_def["name"])
            continue

        # Get template
        template = get_template_by_id(wf_def["template_id"])
        if not template:
            print(f"  Warning: Template {wf_def['template_id']} not found, skipping {wf_def['name']}")
            continue

        # Create workflow
        workflow = Workflow(
            id=uuid4(),
            user_id=user.id,
            name=wf_def["name"],
            description=wf_def["description"],
            is_active=True,
            trigger_config=wf_def["trigger_config"]
        )
        db.add(workflow)
        await db.flush()  # Get workflow ID

        # Create nodes
        node_map = {}  # template node_id -> db node id
        for node_def in template.get("nodes", []):
            node = WorkflowNode(
                id=uuid4(),
                workflow_id=workflow.id,
                node_id=node_def["node_id"],
                node_type=node_def["node_type"],
                builtin_tool=node_def.get("builtin_tool"),
                config=node_def.get("config", {}),
                position_x=node_def.get("position_x", 0),
                position_y=node_def.get("position_y", 0)
            )
            db.add(node)
            node_map[node_def["node_id"]] = node.id

        await db.flush()

        # Create edges
        for edge_def in template.get("edges", []):
            edge = WorkflowEdge(
                id=uuid4(),
                workflow_id=workflow.id,
                source_node_id=edge_def["source_node_id"],
                target_node_id=edge_def["target_node_id"],
                source_handle=edge_def.get("source_handle"),
                condition=edge_def.get("condition")
            )
            db.add(edge)

        created.append(wf_def["name"])

    await db.commit()
    return created, skipped


async def main():
    """Seed tools and workflows."""
    print("\n" + "=" * 60)
    print("Seeding Tools and Workflows")
    print("=" * 60)

    async with AsyncSessionLocal() as db:
        # Get or create admin user
        print("\n[1/3] Setting up admin user...")
        admin = await get_or_create_admin_user(db)
        print(f"  Using user: {admin.username} ({admin.id})")

        # Seed tools
        print("\n[2/3] Seeding user tools...")
        tools_created, tools_skipped = await seed_tools(db, admin)

        if tools_created:
            print(f"  ✓ Created {len(tools_created)} tools:")
            for name in tools_created:
                print(f"    - {name}")
        if tools_skipped:
            print(f"  ⊘ Skipped {len(tools_skipped)} (already exist)")

        # Seed workflows
        print("\n[3/3] Seeding workflows from templates...")
        wf_created, wf_skipped = await seed_workflows(db, admin)

        if wf_created:
            print(f"  ✓ Created {len(wf_created)} workflows:")
            for name in wf_created:
                print(f"    - {name}")
        if wf_skipped:
            print(f"  ⊘ Skipped {len(wf_skipped)} (already exist)")

        # Summary
        print("\n" + "-" * 60)
        print("Summary:")
        print(f"  Tools:     {len(tools_created)} created, {len(tools_skipped)} skipped")
        print(f"  Workflows: {len(wf_created)} created, {len(wf_skipped)} skipped")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
