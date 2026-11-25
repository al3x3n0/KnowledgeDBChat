import React, { useEffect, useMemo } from 'react';
import { ArrowLeft, ArrowLeftRight, ArrowUpRight } from 'lucide-react';
import { useQuery } from 'react-query';
import { Link, useParams, useSearchParams } from 'react-router-dom';

import Button from '../components/common/Button';
import { apiClient } from '../services/api';
import type { ExperimentPlan, ExperimentRun } from '../types';
import { compareExperimentMeasurements, experimentComparisonIssues, experimentRepeatCounts, suggestExperimentPair } from '../utils/experimentComparison';

const numberFormat = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 8 });
const ratioFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });

function formatMeasurement(value: number, unit: string) {
  return unit ? `${numberFormat.format(value)} ${unit}` : numberFormat.format(value);
}

function researchNoteUrl(plan: ExperimentPlan, runId?: string) {
  const params = new URLSearchParams({ note: plan.research_note_id, plan: plan.id });
  if (runId) params.set('run', runId);
  return `/research-notes?${params}`;
}

const RunEvidence: React.FC<{ label: string; run: ExperimentRun; plan: ExperimentPlan }> = ({ label, run, plan }) => {
  const inventory = run.artifact_inventory ?? run.compiler_artifacts?.artifact_inventory;
  const files = ['ir_paths', 'asm_paths', 'remark_paths', 'log_paths'].flatMap((key) => {
    const paths = run.compiler_artifacts?.[key];
    return Array.isArray(paths) ? paths.filter((path): path is string => typeof path === 'string' && Boolean(path)) : [];
  });
  const cases = run.benchmark_case_ids || [];
  const source = run.experiment_run?.source_name || run.experiment_run?.source_id || run.config?.source_id;
  const { actual: repeats, configured: configuredRepeats } = experimentRepeatCounts(run);
  return (
    <article className="min-w-0 space-y-3 rounded-lg border border-gray-200 bg-white p-3.5 sm:p-4" aria-label={`${label} evidence`}>
      <div>
        <p className="text-xs font-medium text-gray-600">{label}</p>
        <h3 className="break-words font-semibold text-gray-900">{run.name}</h3>
        <p className="text-xs capitalize text-gray-600">{run.status}</p>
      </div>
      <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1.5 break-words text-xs">
        <dt className="text-gray-600">Family</dt>
        <dd className="text-gray-900">{run.benchmark_family || 'Unrecorded'}</dd>
        <dt className="text-gray-600">Suite</dt>
        <dd className="text-gray-900">{run.benchmark_suite_id || 'Unrecorded'}</dd>
        <dt className="text-gray-600">Cases</dt>
        <dd className="text-gray-900">{cases.length ? cases.join(', ') : 'Unrecorded'}</dd>
        <dt className="text-gray-600">Sandbox</dt>
        <dd className="text-gray-900">{run.sandbox_profile_id || 'Unrecorded'}</dd>
        <dt className="text-gray-600">Recipe</dt>
        <dd className="text-gray-900">{run.recipe_id || 'Unrecorded'}{run.recipe_version != null && ` · v${run.recipe_version}`}</dd>
        {source && (
          <>
            <dt className="text-gray-600">Source</dt>
            <dd className="text-gray-900">{String(source)}</dd>
          </>
        )}
        <dt className="text-gray-600">Repeats</dt>
        <dd className="text-gray-900">
          {typeof repeats === 'number' && repeats > 0 ? repeats
            : typeof configuredRepeats === 'number' && configuredRepeats > 0 ? `${configuredRepeats} configured`
              : 'Unrecorded'}
        </dd>
        <dt className="text-gray-600">Artifact types</dt>
        <dd className="text-gray-900">{Array.isArray(inventory) && inventory.length ? inventory.join(', ') : 'Unrecorded'}</dd>
      </dl>
      {files.length > 0 && (
        <details className="break-words text-xs text-gray-700">
          <summary className="cursor-pointer font-medium">Recorded files</summary>
          <ul className="mt-1 space-y-1">
            {Array.from(new Set(files)).map((file) => <li key={file}>{file}</li>)}
          </ul>
        </details>
      )}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
        <Link className="text-primary-700 hover:underline" to={researchNoteUrl(plan, run.id)}>
          Open {label.toLowerCase()} run
        </Link>
        {run.agent_job_id && (
          <Link className="inline-flex items-center gap-1 text-primary-700 hover:underline" to={`/autonomous-agents?job=${encodeURIComponent(run.agent_job_id)}`}>
            Agent job <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" />
          </Link>
        )}
      </div>
    </article>
  );
};

const ExperimentComparisonPage: React.FC = () => {
  const { planId = '' } = useParams();
  const [params, setParams] = useSearchParams();
  const planQuery = useQuery(['experiment-plan', planId], () => apiClient.getExperimentPlan(planId), { retry: false });
  const runsQuery = useQuery(['experiment-runs', planId], () => apiClient.listExperimentRuns(planId), { retry: false });
  const plan = planQuery.data;
  const baselineId = params.get('baseline');
  const candidateId = params.get('candidate');
  const runs = runsQuery.data?.runs;
  const { baseline, candidate } = useMemo(() => {
    if (!runs) return {};
    const suggested = suggestExperimentPair(runs, candidateId ?? undefined);
    const baseline = baselineId === null ? suggested.baseline
      : runs.find((run) => run.id === baselineId && run.id !== suggested.candidate?.id);
    return { baseline, candidate: suggested.candidate };
  }, [runs, baselineId, candidateId]);
  useEffect(() => {
    if (!baseline || !candidate || (baselineId !== null && candidateId !== null)) return;
    setParams({ baseline: baseline.id, candidate: candidate.id }, { replace: true });
  }, [baseline, baselineId, candidate, candidateId, setParams]);
  const issues = baseline && candidate ? experimentComparisonIssues(baseline, candidate) : [];
  const rows = baseline && candidate ? compareExperimentMeasurements(baseline, candidate, issues.length === 0) : [];
  const setPair = (nextBaseline: string, nextCandidate: string) => setParams({ baseline: nextBaseline, candidate: nextCandidate });
  const linkedSelectionMissing = Boolean((baselineId && !baseline) || (candidateId && !candidate));

  return (
    <div className="w-full max-w-5xl space-y-5 p-4 pb-20 lg:p-6 lg:pb-20">
      <header>
        <Link to={plan ? researchNoteUrl(plan) : '/research-notes'} className="inline-flex items-center gap-1 text-sm text-primary-700 hover:underline">
          <ArrowLeft aria-hidden="true" className="h-4 w-4" /> Research Notes
        </Link>
        <h1 className="mt-2 text-2xl font-bold text-gray-900">Compare experiment runs</h1>
        {plan && <p className="mt-1 break-words text-sm text-gray-600">{plan.title}</p>}
      </header>

      {(planQuery.isError || runsQuery.isError) ? (
        <div role="alert" className="flex flex-wrap items-center gap-2 text-sm text-gray-700">
          <p>Could not load this experiment plan and its runs.</p>
          <Button type="button" size="sm" variant="secondary" onClick={() => {
            if (planQuery.isError) void planQuery.refetch();
            if (runsQuery.isError) void runsQuery.refetch();
          }}>Try again</Button>
        </div>
      ) : !plan || !runs ? (
        <p role="status" className="text-sm text-gray-600">Loading experiment runs…</p>
      ) : runs.length < 2 ? (
        <p className="text-sm text-gray-600">This experiment plan needs two runs to compare.</p>
      ) : (
        <>
          <div className="flex flex-wrap items-end gap-3 rounded-lg border border-gray-200 bg-white p-3.5">
            <label className="min-w-0 flex-1 basis-52 space-y-1 text-xs font-medium text-gray-700">
              <span>Baseline</span>
              <select value={baseline?.id ?? ''} onChange={(e) => setPair(e.target.value, candidate?.id ?? '')}
                className="block w-full truncate rounded-md border border-gray-300 bg-gray-50 px-2 py-2 text-sm text-gray-900">
                <option value="">Select baseline run</option>
                {runs.filter((run) => run.id !== candidate?.id).map((run) => <option key={run.id} value={run.id}>{run.name} · {run.status}</option>)}
              </select>
            </label>
            <label className="min-w-0 flex-1 basis-52 space-y-1 text-xs font-medium text-gray-700">
              <span>Candidate</span>
              <select value={candidate?.id ?? ''} onChange={(e) => setPair(baseline?.id ?? '', e.target.value)}
                className="block w-full truncate rounded-md border border-gray-300 bg-gray-50 px-2 py-2 text-sm text-gray-900">
                <option value="">Select candidate run</option>
                {runs.filter((run) => run.id !== baseline?.id).map((run) => <option key={run.id} value={run.id}>{run.name} · {run.status}</option>)}
              </select>
            </label>
            <Button type="button" size="sm" variant="secondary" disabled={!baseline || !candidate}
              icon={<ArrowLeftRight className="h-4 w-4" aria-hidden="true" />}
              onClick={() => baseline && candidate && setPair(candidate.id, baseline.id)}>
              Swap runs
            </Button>
          </div>
          {linkedSelectionMissing && <p role="status" className="text-sm text-gray-700">Choose two available runs from this plan.</p>}
          {!linkedSelectionMissing && (!baseline || !candidate) && <p className="text-sm text-gray-600">Select both runs to see their measurements and evidence.</p>}
          {baseline && candidate && (
            <>
              <section aria-labelledby="measurement-heading" className="space-y-2">
                <h2 id="measurement-heading" className="text-lg font-semibold text-gray-900">Measurements</h2>
                {issues.length ? (
                  <div className="rounded-lg border border-amber-400/40 bg-amber-500/10 p-3 text-sm text-gray-800">
                    <p className="font-medium">Deltas unavailable for this pair</p>
                    <ul className="mt-1 list-disc space-y-0.5 pl-5">
                      {issues.map((issue) => <li key={issue}>{issue}</li>)}
                    </ul>
                  </div>
                ) : (
                  <p className="text-sm text-gray-600">Recorded benchmark IDs match. Changes show candidate minus baseline.</p>
                )}
                {rows.length === 0 ? (
                  <p className="text-sm text-gray-600">These runs have no recorded numeric measurements.</p>
                ) : (
                  <table className="w-full table-fixed border-collapse text-left text-xs tabular-nums sm:text-sm" aria-label="Experiment measurements">
                    <thead className="border-b border-gray-300 text-gray-600">
                      <tr>
                        <th scope="col" className="w-[26%] py-2 pr-1 font-medium sm:w-[32%]">Metric</th>
                        <th scope="col" className="py-2 px-1 text-right font-medium">Baseline</th>
                        <th scope="col" className="py-2 px-1 text-right font-medium">Candidate</th>
                        <th scope="col" className="py-2 pl-1 text-right font-medium">Change</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 text-gray-900">
                      {rows.map((row) => (
                        <tr key={row.key}>
                          <th scope="row" className="break-words py-2 pr-1 font-medium">{row.label}</th>
                          <td className="break-words px-1 py-2 text-right">{row.baseline === null ? 'Missing' : formatMeasurement(row.baseline, row.unit)}</td>
                          <td className="break-words px-1 py-2 text-right">{row.candidate === null ? 'Missing' : formatMeasurement(row.candidate, row.unit)}</td>
                          <td className="break-words py-2 pl-1 text-right">
                            {row.difference === null ? <span aria-label="Change unavailable">—</span> : (
                              <>
                                <span>{row.difference > 0 ? '+' : ''}{formatMeasurement(row.difference, row.unit)}</span>
                                {row.percent !== null && <span className="block text-xs text-gray-600">{row.percent > 0 ? '+' : ''}{ratioFormat.format(row.percent)}%</span>}
                                {row.baseline === 0 && <span className="block text-xs text-gray-600">Zero baseline</span>}
                              </>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </section>
              <section aria-labelledby="evidence-heading" className="space-y-2">
                <h2 id="evidence-heading" className="text-lg font-semibold text-gray-900">Run evidence</h2>
                <div className="grid gap-3 sm:grid-cols-2">
                  <RunEvidence label="Baseline" run={baseline} plan={plan} />
                  <RunEvidence label="Candidate" run={candidate} plan={plan} />
                </div>
              </section>
            </>
          )}
        </>
      )}
    </div>
  );
};

export default ExperimentComparisonPage;
