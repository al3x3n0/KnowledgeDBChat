import type { AgentJobMemoryExtractResponse } from '../types';

export type JobMemoryExtractionSummary = {
  status?: string;
  created_count?: number;
  parsed_count?: number;
  candidate_count?: number;
  skipped_duplicates?: number;
  dedup_existing_signature_count?: number;
  is_relaunch_chain?: boolean;
  relaunch_root_job_id?: string | null;
  error?: string | null;
};

export type JobMemoryRuntimeSummary = {
  profile?: string;
  role?: string;
  limit?: number;
};

export type JobMemoryPersistenceSummary = {
  enabled?: boolean;
  injected_count?: number;
  runtime?: JobMemoryRuntimeSummary | null;
  extraction?: JobMemoryExtractionSummary | null;
};

export const normalizeJobMemoryExtractionSummary = (
  value: unknown
): JobMemoryExtractionSummary | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const raw = value as Record<string, unknown>;
  const toFiniteNumber = (v: unknown): number | undefined => {
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  };
  const summary: JobMemoryExtractionSummary = {
    status: typeof raw.status === 'string' ? raw.status : undefined,
    created_count: toFiniteNumber(raw.created_count) ?? toFiniteNumber(raw.memories_created),
    parsed_count: toFiniteNumber(raw.parsed_count),
    candidate_count: toFiniteNumber(raw.candidate_count),
    skipped_duplicates: toFiniteNumber(raw.skipped_duplicates),
    dedup_existing_signature_count: toFiniteNumber(raw.dedup_existing_signature_count),
    is_relaunch_chain: raw.is_relaunch_chain === undefined ? undefined : Boolean(raw.is_relaunch_chain),
    relaunch_root_job_id: raw.relaunch_root_job_id == null
      ? undefined
      : String(raw.relaunch_root_job_id || '').trim() || null,
    error: raw.error == null ? undefined : String(raw.error || '').trim() || null,
  };
  const hasData =
    summary.status !== undefined
    || summary.created_count !== undefined
    || summary.parsed_count !== undefined
    || summary.candidate_count !== undefined
    || summary.skipped_duplicates !== undefined
    || summary.dedup_existing_signature_count !== undefined
    || summary.is_relaunch_chain !== undefined
    || summary.relaunch_root_job_id !== undefined
    || summary.error !== undefined;
  return hasData ? summary : null;
};

export const normalizeManualExtractionResult = (
  result: AgentJobMemoryExtractResponse
): JobMemoryExtractionSummary => (
  normalizeJobMemoryExtractionSummary({
    status: 'completed',
    ...result,
  }) || { status: 'completed', created_count: 0 }
);

export const normalizeJobMemoryPersistenceSummary = (
  value: unknown
): JobMemoryPersistenceSummary | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const raw = value as Record<string, unknown>;
  const toFiniteNumber = (v: unknown): number | undefined => {
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  };

  const runtimeRaw = raw.runtime && typeof raw.runtime === 'object'
    ? (raw.runtime as Record<string, unknown>)
    : null;
  const runtime: JobMemoryRuntimeSummary | null = runtimeRaw
    ? {
        profile: runtimeRaw.profile == null ? undefined : String(runtimeRaw.profile || '').trim() || undefined,
        role: runtimeRaw.role == null ? undefined : String(runtimeRaw.role || '').trim() || undefined,
        limit: toFiniteNumber(runtimeRaw.limit),
      }
    : null;

  const extraction = normalizeJobMemoryExtractionSummary(raw.extraction);

  const summary: JobMemoryPersistenceSummary = {
    enabled: raw.enabled === undefined ? undefined : Boolean(raw.enabled),
    injected_count: toFiniteNumber(raw.injected_count),
    runtime,
    extraction,
  };

  const hasData =
    summary.enabled !== undefined
    || summary.injected_count !== undefined
    || summary.runtime !== null
    || summary.extraction !== null;
  return hasData ? summary : null;
};
