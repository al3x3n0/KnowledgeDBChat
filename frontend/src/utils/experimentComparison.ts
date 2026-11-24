import type { ExperimentRun } from '../types';

const measurementFields: Record<string, [label: string, unit: string]> = {
  compile_time_ms: ['Compile time', 'ms'],
  runtime_ms: ['Runtime', 'ms'],
  binary_size_bytes: ['Binary size', 'bytes'],
  artifact_diff_score: ['Artifact diff score', ''],
  cycles: ['Cycles', ''],
  retired_instructions: ['Retired instructions', ''],
  load_queue_full_events: ['Load queue full events', ''],
  instructions_per_cycle: ['Instructions per cycle', ''],
};

interface Measurement {
  label: string;
  unit: string;
  value: number;
}

const isCompleted = (run: ExperimentRun) => (run.status === 'completed' || run.status === 'succeeded') &&
  run.experiment_run?.ok !== false && run.results?.ok !== false;

export function experimentComparisonUrl(planId: string, baselineId?: string, candidateId?: string) {
  const base = `/research-notes/experiments/${encodeURIComponent(planId)}/compare`;
  if (!baselineId && !candidateId) return base;
  const params = new URLSearchParams();
  if (baselineId) params.set('baseline', baselineId);
  if (candidateId) params.set('candidate', candidateId);
  return `${base}?${params}`;
}

function numericValue(value: unknown): number | null {
  if (typeof value === 'string' && !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim())) return null;
  if (typeof value !== 'number' && typeof value !== 'string') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function measurements(run: ExperimentRun) {
  const result = new Map<string, Measurement>();
  const summary = run.results?.measurement_summary ?? run.measurement_summary;
  const isPlanned = ['pending', 'planned'].includes(String(summary?.status ?? '').trim().toLowerCase());
  if (!isPlanned) {
    for (const [key, [label, unit]] of Object.entries(measurementFields)) {
      const value = numericValue(summary?.[key]);
      if (value !== null) result.set(key, { label, unit, value });
    }
  }
  const counters = run.results?.perf_counters ?? (isPlanned ? null : summary?.perf_counters ?? run.perf_counters);
  if (!counters || typeof counters !== 'object' || Array.isArray(counters)) return result;
  for (const [key, rawValue] of Object.entries(counters || {})) {
    const value = numericValue(rawValue);
    if (value !== null) {
      const label = key.replace(/_/g, ' ');
      result.set(`perf:${key}`, { label, unit: '', value });
    }
  }
  return result;
}

export function experimentComparisonIssues(baseline: ExperimentRun, candidate: ExperimentRun): string[] {
  const issues: string[] = [];
  if (!isCompleted(baseline) || !isCompleted(candidate)) {
    issues.push('Both runs need a successful completion for deltas.');
  }
  if (!baseline.benchmark_family || !candidate.benchmark_family || !baseline.benchmark_suite_id || !candidate.benchmark_suite_id) {
    issues.push('Benchmark family or suite is missing from at least one run.');
  } else if (baseline.benchmark_family !== candidate.benchmark_family || baseline.benchmark_suite_id !== candidate.benchmark_suite_id) {
    issues.push('Benchmark families or suites differ.');
  }
  const baselineCases = new Set((baseline.benchmark_case_ids || []).filter(Boolean));
  const candidateCases = new Set((candidate.benchmark_case_ids || []).filter(Boolean));
  if (!baselineCases.size || !candidateCases.size) {
    issues.push('Benchmark cases are missing from at least one run.');
  } else if (baselineCases.size !== candidateCases.size ||
    Array.from(baselineCases).some((id) => !candidateCases.has(id))) {
    issues.push('The runs cover different benchmark cases.');
  }
  if (baseline.sandbox_profile_id && candidate.sandbox_profile_id &&
    baseline.sandbox_profile_id !== candidate.sandbox_profile_id) {
    issues.push('The runs used different sandbox profiles.');
  }
  return issues;
}

export function compareExperimentMeasurements(baseline: ExperimentRun, candidate: ExperimentRun, includeDeltas: boolean) {
  const before = measurements(baseline);
  const after = measurements(candidate);
  const keys = Array.from(new Set([...Array.from(before.keys()), ...Array.from(after.keys())]));
  return keys.map((key) => {
    const prior = before.get(key);
    const next = after.get(key);
    const beforeValue = prior?.value ?? null;
    const afterValue = next?.value ?? null;
    const difference = includeDeltas && beforeValue !== null && afterValue !== null
      ? afterValue - beforeValue : null;
    const percent = difference !== null && beforeValue
      ? difference / Math.abs(beforeValue) * 100 : null;
    return {
      key,
      label: prior?.label ?? next?.label ?? key,
      unit: prior?.unit ?? next?.unit ?? '',
      baseline: beforeValue,
      candidate: afterValue,
      difference: difference !== null && Number.isFinite(difference) ? difference : null,
      percent: percent !== null && Number.isFinite(percent) ? percent : null,
    };
  });
}

export function suggestExperimentPair(runs: ExperimentRun[], candidateId?: string) {
  const candidate = candidateId === undefined
    ? runs.find((run) => isCompleted(run) && measurements(run).size > 0) ?? runs.find(isCompleted) ?? runs[0]
    : runs.find((run) => run.id === candidateId);
  const candidateIndex = runs.findIndex((run) => run.id === candidate?.id);
  const baseline = candidate && runs.slice(candidateIndex + 1).find((run) => (
    experimentComparisonIssues(run, candidate).length === 0 &&
    compareExperimentMeasurements(run, candidate, true).some((row) => row.difference !== null)
  ));
  return { baseline, candidate };
}
