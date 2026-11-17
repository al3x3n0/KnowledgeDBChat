SELECT count(*) FROM agent_jobs
WHERE execution_lease_expires_at > now()
   OR (status = 'pending'
       AND (next_run_at IS NULL OR next_run_at <= now()
            OR COALESCE(results::jsonb #>> '{sandbox_wait,key}', '') <> '')
       AND (COALESCE(results::jsonb #>> '{execution_strategy,scheduler_state,queue_reason}', '') <> 'user_job_cap'
            OR COALESCE(results::jsonb #>> '{sandbox_wait,key}', '') <> ''))
