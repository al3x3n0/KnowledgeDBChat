# KnowledgeDBChat — Architecture Diagrams

Implementation-faithful visual map of the system as it exists today. Four
views, from the outside in: deployment, backend subsystems, the autonomous
agent runtime, and the LLM provider stack.

Diagrams are Mermaid — GitHub renders them inline; locally use the Kroki
service (`http://localhost:8001`) or any Mermaid renderer.

---

## 1. Deployment view (docker-compose services)

```mermaid
flowchart LR
    subgraph Clients
        UI["React Frontend<br/>(35 pages, api.ts client)"]
        MCPC["External MCP Clients<br/>(API-key auth)"]
    end

    subgraph Core["Backend"]
        NGINX["nginx :3000"]
        API["FastAPI backend :8000<br/>~58 endpoint groups /api/v1"]
        CEL["Celery worker<br/>27 task modules"]
        CELTEX["celery_latex<br/>isolated LaTeX queue"]
        BEAT["celery_beat (prod)<br/>schedules"]
    end

    subgraph Data["State & Storage"]
        PG[("PostgreSQL :5432<br/>50+ models, 73 migrations")]
        REDIS[("Redis :6379<br/>cache, broker, pub/sub, flags")]
        QDR[("Qdrant :6333<br/>vector index")]
        MINIO[("MinIO :9000/:9001<br/>documents, media, exports")]
    end

    subgraph Aux["Auxiliary services"]
        OLL["Ollama :11434<br/>local LLM"]
        KROKI["Kroki :8001<br/>diagram rendering"]
        VID["Go video-streamer :8080"]
    end

    EXT["External LLM APIs<br/>DeepSeek | OpenAI | Anthropic | Qwen | Kimi"]

    UI --> NGINX --> API
    MCPC --> API
    API --> PG
    API --> REDIS
    API --> QDR
    API --> MINIO
    API --> OLL
    API --> EXT
    API --> KROKI
    REDIS --> CEL
    CEL --> PG
    CEL --> QDR
    CEL --> MINIO
    CEL --> OLL
    CEL --> EXT
    REDIS --> CELTEX
    BEAT --> REDIS
    UI --> VID --> MINIO
```

---

## 2. Backend subsystem view

```mermaid
flowchart TB
    ROUTES["api/routes.py — /api/v1<br/>auth · users · chat · documents · kg · admin"]

    subgraph Subsystems["Domain subsystems (endpoints + services + models share name prefixes)"]
        AGENTS["Autonomous agents<br/>agent-jobs · control-plane · workflows"]
        CODING["Coding swarm<br/>backlog · code-patches · patch-prs · git"]
        RESEARCH["Research suite<br/>papers · notes · portfolios · inbox · monitors"]
        GEN["Doc generation<br/>latex · presentations · docx · artifacts · export"]
        TRAIN["Training / AI Hub<br/>datasets · jobs · model registry · evals"]
        GOV["Tool governance<br/>policies · audit · secrets · user-tools · mcp-config"]
        OBS["Observability<br/>usage · retrieval-traces · llm-snapshots · analytics"]
    end

    subgraph Foundation["Foundation services"]
        RAG["RAG pipeline<br/>hybrid search + rerank + MMR + KG context"]
        LLM["LLMService + llm_providers/<br/>(view 4)"]
        MEM["Memory<br/>extraction · ranking · injection"]
        KGS["Knowledge graph<br/>Entity - Mention - Relationship"]
        CONN["Connectors<br/>GitLab · GitHub · Confluence · Web · ArXiv"]
        STORE["VectorStore · Storage · Transcription"]
    end

    ROUTES --> Subsystems
    Subsystems --> Foundation
    AGENTS -.->|"policy + audit on every tool call"| GOV
    Foundation --> INFRA["PostgreSQL · Redis · Qdrant · MinIO"]
```

---

## 3. Autonomous agent runtime (one job)

The observe→think→act→evaluate loop with the current think-phase pipeline.
Items marked ★ are recent additions.

```mermaid
flowchart TB
    START["execute_job"]
    DET{"deterministic_runner<br/>configured?"}
    RUNNER["Deterministic runner registry<br/>(30+ runners: research, coding, latex...)"]
    INIT["Loop init: checkpoint resume ·<br/>skill profile · project profile · memory injection"]
    OBS["OBSERVE<br/>AgentObservationService"]
    GATE{"goal_achieved claimed?"}
    CONTRACT["Goal contract check<br/>min findings / artifacts / progress<br/>blocks false completion"]
    EVAL["EVALUATE<br/>progress heuristics · critic pass ·<br/>checkpoint every 5 iters"]

    subgraph THINK["THINK — AgentThinkingService"]
        COMPACT["★ Auto-compaction<br/>state > threshold → summarize old actions<br/>into compressed_history (fast tier)"]
        PROMPT["★ Cache-friendly prompt split<br/>stable per-job prefix → system prompt<br/>volatile plan/critic/history → user message"]
        NTL{"★ native_tool_loop<br/>enabled?"}
        LOOP["★ Native tool loop<br/>model calls read-safe tools via native API<br/>(bounded; gated tools deferred)"]
        STRUCT["★ generate_structured<br/>schema-enforced decision JSON"]
        PARSE["Decision parser<br/>Pydantic validate → retry → repair"]
    end

    subgraph ACT["ACT — act_phase"]
        APPROVE{"approval checkpoint /<br/>dangerous tool?"}
        PAUSE["PAUSE job<br/>await operator approval"]
        DISPATCH["AgentActionService → AgentToolRegistry<br/>policy check → execute → audit log<br/>(173 tools)"]
    end

    subgraph FIN["FINALIZE"]
        RESULTS["persist results + digest"]
        MEMX["memory extraction → ConversationMemory"]
        CHAIN["trigger chained jobs<br/>(fan-out / fan-in via Celery)"]
    end

    subgraph GOLDEN["★ Golden-task regression suite"]
        G1["runs this real loop end-to-end with<br/>scripted LLM + scripted tools (5 contracts)"]
    end

    START --> DET
    DET -- yes --> RUNNER
    DET -- no --> INIT
    INIT --> OBS
    OBS --> COMPACT --> PROMPT --> NTL
    NTL -- yes --> LOOP --> PARSE
    NTL -- no --> STRUCT --> PARSE
    PARSE --> GATE
    GATE -- yes --> CONTRACT
    GATE -- no --> APPROVE
    APPROVE -- required --> PAUSE
    APPROVE -- clear --> DISPATCH
    DISPATCH --> EVAL
    EVAL -->|"budgets remain"| OBS
    CONTRACT -- satisfied --> RESULTS
    EVAL -->|"budget / stop"| RESULTS
    RESULTS --> MEMX --> CHAIN
```

---

## 4. LLM provider stack & observability

```mermaid
flowchart TB
    subgraph Callers
        CHAT["Chat / RAG"]
        THINKC["Agent think phase"]
        NTLC["★ Native tool loop"]
        COMPC["★ Auto-compaction"]
        MEMC["Memory extraction/ranking"]
    end

    subgraph SVC["LLMService"]
        GR["generate_response<br/>(legacy prompted text)"]
        GS["★ generate_structured<br/>(native tools + JSON schema)"]
        ROUTE["Tier routing: fast | balanced | deep<br/>feature flags → user/task overrides →<br/>fallback tiers + health cooldowns"]
        SEM["concurrency semaphore"]
    end

    subgraph PROV["★ app/services/llm_providers/"]
        OLLP["OllamaProvider<br/>/api/chat · tools · format"]
        OAIP["OpenAICompatibleProvider (openai SDK)<br/>openai · deepseek · qwen · kimi · custom"]
        ANTP["AnthropicProvider (anthropic SDK)<br/>tools · forced-tool schema output ·<br/>★ cache_control breakpoints · refusal handling"]
    end

    subgraph APIs["Model APIs"]
        OLLAMA["Ollama (local)"]
        OPENAI["OpenAI"]
        DS["DeepSeek"]
        QWEN["Qwen / DashScope"]
        KIMI["Kimi / Moonshot"]
        CLAUDE["Anthropic Claude"]
    end

    subgraph OBSV["Observability (per call)"]
        USAGE[("llm_usage_events<br/>tokens · latency · tier · cache hits")]
        SNAP[("★ llm_call_snapshots<br/>full prompts + responses<br/>job/iteration/phase correlation")]
    end

    Callers --> SVC
    GR --> ROUTE
    GS --> ROUTE
    ROUTE --> SEM --> PROV
    OLLP --> OLLAMA
    OAIP --> OPENAI
    OAIP --> DS
    OAIP --> QWEN
    OAIP --> KIMI
    ANTP --> CLAUDE
    SVC --> USAGE
    SVC --> SNAP
    SNAP -.->|"GET /api/v1/llm-snapshots"| REPLAY["Replay / debug"]
```

**Key properties encoded above**

- Every tool an agent executes — whether via the classic act phase or the
  native tool loop — passes through the same dispatch: policy engine →
  execution → audit log. Approval-gated and dangerous tools always route
  back to the act phase's pause-for-approval machinery.
- The think-phase system prompt is byte-stable per job; all per-iteration
  context rides in the user message. On Anthropic this makes the prefix a
  cache read (~0.1× cost) from iteration 2 onward; OpenAI/DeepSeek prefix
  caching engages automatically.
- Both LLM paths share tier routing, provider health cooldowns, usage
  accounting, and (opt-in) full call snapshots keyed by job/iteration/phase.
- The golden-task suite (`tests/test_golden_agent_tasks.py`) exercises the
  real loop in view 3 with only the LLM and tool seams scripted — it is the
  regression gate for changes to any starred component.
