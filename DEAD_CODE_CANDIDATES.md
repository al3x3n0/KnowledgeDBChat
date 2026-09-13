# Dead-code candidates, remaining after the de-slop pass

Remaining: 56 functions, 927 lines.

Was 106 functions / 1791 lines before the pass. Three whole modules and 26
functions were removed or wired; what is left is the long tail. Each is a
judgment about whether the capability is abandoned or unfinished, and none is
large enough to force the question on its own.

## Method

AST for definitions; identifier counts over every .py in app/, tests/, main.py,
scripts/, alembic/ and seed_data/, counting identifiers inside string literals
too, plus a scan of frontend/src for names that might be an API contract.
Excludes decorated functions (routes, Celery tasks, fixtures, computed fields),
`on_*` framework callbacks, and Protocol stubs.

The first version of this scan missed main.py, and so reported the FastAPI
exception handlers, a TrainerCallback and a Pydantic computed field as dead.
They are not. A name reached only by a getattr built from a fragment rather
than a whole literal would still evade this.

## Known keepers

- `core/database.py:create_tables` / `drop_tables` -- CLAUDE.md keeps these
  deliberately, for tests and throwaway databases.

| lines | location | function |
|------:|----------|----------|
| 59 | `app/services/analytics_service.py:592` | `get_search_analytics` |
| 53 | `app/services/diagram_service.py:837` | `create_data_flow_diagram` |
| 48 | `app/services/knowledge_graph_service.py:812` | `extract_entity_names_from_text` |
| 46 | `app/services/visualization_service.py:578` | `get_chart_data_for_frontend` |
| 45 | `app/services/search_service.py:615` | `get_search_history_suggestions` |
| 44 | `app/services/trainers/base_trainer.py:173` | `estimate_memory_requirements` |
| 41 | `app/services/agent_memory_integration.py:299` | `get_conversation_memory_injections` |
| 39 | `app/services/llm_service.py:1492` | `check_model_availability` |
| 36 | `app/services/text_processor.py:767` | `extract_metadata` |
| 35 | `app/core/logging.py:121` | `log_service_call` |
| 31 | `app/services/transcription/ssl_config.py:149` | `get_ssl_instructions` |
| 29 | `app/services/agent_job_memory_service.py:1431` | `get_memories_by_type` |
| 27 | `app/services/memory_service.py:336` | `create_memory_interaction` |
| 25 | `app/services/chat_service.py:776` | `_build_context` |
| 24 | `app/services/llm_service.py:1557` | `pull_model` |
| 22 | `app/models/agent_job.py:330` | `get_chain_data_for_child` |
| 20 | `app/api/endpoints/agent_jobs.py:422` | `_resolve_coding_swarm_profile` |
| 19 | `app/services/vector_store.py:1769` | `update_document_chunks` |
| 19 | `app/core/rate_limit.py:63` | `get_rate_limit_for_endpoint` |
| 18 | `app/core/database.py:109` | `create_tables` |
| 18 | `app/api/endpoints/agent_jobs.py:348` | `_merge_coding_swarm_request_with_profile` |
| 16 | `app/api/endpoints/agent_jobs.py:329` | `_build_coding_swarm_goal` |
| 15 | `app/api/endpoints/agent_jobs.py:301` | `_build_bug_triage_swarm_goal` |
| 14 | `app/models/model_registry.py:158` | `get_display_info` |
| 13 | `app/services/notification_service.py:416` | `cleanup_expired_notifications` |
| 12 | `app/services/research_monitor_profile_service.py:181` | `_monitor_follow_up_autonomy_from_policy` |
| 12 | `app/services/job_results_exporter.py:2108` | `export_job_results_enhanced` |
| 12 | `app/api/endpoints/coding_backlog.py:489` | `_refresh_waiting_metadata` |
| 12 | `app/api/endpoints/agent_control_plane.py:117` | `_normalize_bool` |
| 11 | `app/api/endpoints/agent_jobs.py:694` | `_normalize_datetime` |
| 9 | `app/services/visualization_service.py:626` | `save_chart_to_file` |
| 8 | `app/models/user.py:118` | `has_permission` |
| 8 | `app/models/training_dataset.py:220` | `to_alpaca_dict` |
| 8 | `app/models/training_dataset.py:142` | `get_token_statistics` |
| 8 | `app/models/agent_job.py:358` | `get_chain_hierarchy` |
| 7 | `app/api/endpoints/agent_jobs.py:319` | `_get_coding_swarm_preset_definition` |
| 6 | `app/api/endpoints/coding_swarm_profiles.py:129` | `_get_visible_profile_or_404` |
| 5 | `app/services/research_opportunity_service.py:889` | `get_opportunity_lookup` |
| 5 | `app/models/training_job.py:234` | `get_training_time_seconds` |
| 5 | `app/core/cache.py:41` | `close_redis_client` |
| 4 | `app/services/agent_router.py:49` | `get_generalist_agent` |
| 4 | `app/models/training_job.py:228` | `get_current_loss` |
| 4 | `app/models/training_dataset.py:214` | `get_output` |
| 4 | `app/models/training_dataset.py:208` | `get_instruction` |
| 4 | `app/models/model_registry.py:178` | `get_size_mb` |
| 4 | `app/models/agent_job.py:272` | `get_recent_log` |
| 4 | `app/api/endpoints/agent_jobs.py:1017` | `_get_autonomy_budget_from_job` |
| 2 | `app/services/docx_editor_service.py:282` | `calculate_content_hash` |
| 2 | `app/services/agent_decision_parser.py:185` | `reset_metrics` |
| 2 | `app/models/model_registry.py:148` | `can_undeploy` |
| 2 | `app/models/api_key.py:103` | `has_any_scope` |
| 2 | `app/models/agent_job.py:354` | `is_chain_root` |
| 2 | `app/models/agent_definition.py:96` | `has_capability` |
| 1 | `app/services/langgraph_issue_pr_service.py:960` | `_build_repo_cache_key` |
| 1 | `app/services/callgrind_profile.py:52` | `hottest_addresses` |
| 1 | `app/api/endpoints/experiments.py:155` | `_is_scientific_validation_run` |
