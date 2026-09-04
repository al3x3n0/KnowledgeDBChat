/**
 * Author a research pipeline, and be told what is wrong with it before it runs.
 *
 * A pipeline is a DAG of stages, and a stage is not a list of steps — it is a
 * goal contract saying what must be true when the stage is done. The tools
 * that get there are *derived* from the contract, which is why this page shows
 * them rather than asking for them: if you have to name the tool yourself, you
 * can name one that cannot produce what you asked for.
 *
 * The checking happens before anything expensive starts. The four failures it
 * catches are the ones that otherwise cost a whole run to discover:
 *
 *   - a contract asking for evidence no tool produces
 *   - a stage built on a measurement the stage before it never takes
 *   - a budget that sounded generous and is two orders short
 *   - a loop with no bound
 *
 * Launch is the one thing here that spends anything. It is offered only for a
 * pipeline that is startable — valid, expressible as a chain, and inside its
 * budget — and it names the price and the stopping points before it starts.
 * The estimate on screen is sent with the request, so a spec edited since it
 * was priced is refused by the server rather than quietly costing more.
 */

import clsx from 'clsx';
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  GitBranch,
  Layers,
  PauseCircle,
  Play,
  Save,
  Trash2,
  Workflow,
} from 'lucide-react';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

import Button from '../components/common/Button';
import PipelineGraph from '../components/pipelines/PipelineGraph';
import { apiClient } from '../services/api';
import type { PipelineCheck, SavedPipeline } from '../types';

const STORAGE_KEY = 'pipeline_studio_draft_v1';

/** Worked examples rather than an empty box: the shape is unfamiliar enough
 *  that a blank editor is a worse starting point than something to edit.
 *
 *  Each one is a pipeline that validates as written -- a starter that fails
 *  its own check teaches the wrong thing about the format. */
const EXAMPLE = `{
  "name": "int8-attention-study",
  "stages": [
    {
      "id": "profile",
      "goal": "Find where the kernel actually spends its time",
      "contract": { "required_finding_types": ["dynamic_profile"] }
    },
    {
      "id": "measure",
      "goal": "Benchmark the hot path",
      "depends_on": ["profile"],
      "assumes": ["dynamic_profile"],
      "contract": { "required_finding_types": ["benchmark_measurement"] }
    },
    {
      "id": "attribute",
      "goal": "Say what the ceiling is and why",
      "depends_on": ["measure"],
      "assumes": ["benchmark_measurement"],
      "checkpoint": true,
      "contract": { "required_finding_types": ["bottleneck_attribution"] }
    }
  ]
}`;

/** Read a paper, implement the algorithm it describes, and find out whether the
 *  number reproduces. The `implement` stage loops because writing an algorithm
 *  from prose does not work first try, and its contract asks for a *verified*
 *  implementation: the fastest implementation of anything is one that returns
 *  garbage, so timing unchecked code measures work nobody looked at. */
const REPRODUCE_PAPER = `{
  "name": "reproduce-paper-algorithm",
  "stages": [
    {
      "id": "find",
      "goal": "Ingest the paper describing the algorithm to reproduce",
      "contract": { "required_finding_types": ["papers_ingested"] }
    },
    {
      "id": "specify",
      "goal": "Read the paper into an implementable specification: the steps, its worked examples, and the numbers it claims -- written down before any code exists",
      "depends_on": ["find"],
      "assumes": ["papers_ingested"],
      "contract": { "required_finding_types": ["algorithm_spec"] }
    },
    {
      "id": "implement",
      "goal": "Implement the algorithm in C or Rust and establish that it computes the paper's worked examples",
      "depends_on": ["specify"],
      "assumes": ["algorithm_spec"],
      "loop": { "max_iterations": 6, "until": "contract_satisfied" },
      "contract": { "required_finding_types": ["implementation_verified"] }
    },
    {
      "id": "measure",
      "goal": "Time the verified implementation",
      "depends_on": ["implement"],
      "assumes": ["implementation_verified"],
      "contract": {
        "required_finding_types": ["benchmark_measurement"],
        "validity": { "require_uncertainty": ["benchmark_measurement"] }
      }
    },
    {
      "id": "compare",
      "goal": "Score the measurement against the paper's claim, or say which condition makes them incomparable",
      "depends_on": ["measure"],
      "assumes": ["benchmark_measurement", "implementation_verified"],
      "checkpoint": true,
      "contract": { "required_finding_types": ["reproduction_verdict"] }
    }
  ]
}`;

const STARTERS: { label: string; source: string }[] = [
  { label: 'Bottleneck study', source: EXAMPLE },
  { label: 'Reproduce a paper', source: REPRODUCE_PAPER },
];

const readDraft = (): string => {
  try {
    return window.localStorage.getItem(STORAGE_KEY) || EXAMPLE;
  } catch {
    return EXAMPLE;
  }
};

const minutes = (seconds: number) =>
  seconds < 60 ? `${seconds}s` : `${Math.round(seconds / 60)} min`;

const PipelineStudioPage: React.FC = () => {
  const navigate = useNavigate();
  const [source, setSource] = useState<string>(readDraft);
  const [budget, setBudget] = useState<string>('');
  const [check, setCheck] = useState<PipelineCheck | null>(null);
  const [parseError, setParseError] = useState<string>('');
  const [serverError, setServerError] = useState<string>('');
  const [checking, setChecking] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [view, setView] = useState<'split' | 'text' | 'graph'>(() => {
    try {
      return (window.localStorage.getItem('pipeline_studio_view') as any) || 'split';
    } catch {
      return 'split';
    }
  });
  const [selectedStage, setSelectedStage] = useState<string | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem('pipeline_studio_view', view);
    } catch {
      // Not worth telling the user about.
    }
  }, [view]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, source);
    } catch {
      // A private window: the draft simply does not survive a reload.
    }
  }, [source]);

  const parsed = useMemo(() => {
    try {
      const value = JSON.parse(source);
      return { value, error: '' };
    } catch (error: any) {
      // Report the parse failure as its own thing. It is not a problem with
      // the pipeline — there is no pipeline yet — and calling the server with
      // unparseable text would only get a less specific version of this.
      return { value: null, error: error?.message || 'Not valid JSON' };
    }
  }, [source]);

  const runCheck = useCallback(async () => {
    setParseError(parsed.error);
    if (parsed.error || !parsed.value) {
      setCheck(null);
      return;
    }
    setChecking(true);
    setServerError('');
    try {
      const budgetSeconds = budget.trim() ? Number(budget.trim()) : undefined;
      const result = await apiClient.checkPipeline(
        parsed.value,
        Number.isFinite(budgetSeconds) && budgetSeconds ? budgetSeconds : undefined
      );
      setCheck(result);
    } catch (error: any) {
      // A 400 here means the shape is wrong in a way the parser accepted —
      // valid JSON that is not a pipeline. The server says which.
      setServerError(error?.response?.data?.detail || 'Could not check the pipeline');
      setCheck(null);
    } finally {
      setChecking(false);
    }
  }, [parsed, budget]);

  const [saved, setSaved] = useState<SavedPipeline[]>([]);
  // Which saved pipeline the editor is currently showing. null means the
  // draft has never been saved, which is why Save asks for a name and Update
  // does not.
  const [openId, setOpenId] = useState<string | null>(null);

  const refreshSaved = useCallback(async () => {
    try {
      setSaved(await apiClient.listSavedPipelines());
    } catch {
      // A failure to list is not worth a toast on every mount; the editor
      // works without the library.
    }
  }, []);

  useEffect(() => {
    refreshSaved();
  }, [refreshSaved]);

  const handleSave = useCallback(async () => {
    if (!parsed.value) {
      toast.error('Fix the JSON before saving');
      return;
    }
    try {
      if (openId) {
        const updated = await apiClient.updateSavedPipeline(openId, {
          spec: parsed.value,
        });
        toast.success(`Saved ${updated.name}`);
      } else {
        const name = window.prompt(
          'Name this pipeline',
          parsed.value.name || 'Untitled pipeline'
        );
        if (!name || !name.trim()) return;
        const created = await apiClient.saveSavedPipeline({
          name: name.trim(),
          spec: parsed.value,
        });
        setOpenId(created.id);
        toast.success(`Saved ${created.name}`);
      }
      refreshSaved();
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || 'Could not save');
    }
  }, [parsed, openId, refreshSaved]);

  const handleOpen = useCallback((pipeline: SavedPipeline) => {
    setOpenId(pipeline.id);
    setSource(JSON.stringify(pipeline.spec, null, 2));
    // Whatever verdict was cached when it was saved is not re-used: tools and
    // their costs move underneath a stored spec, so it is checked again.
    setCheck(null);
  }, []);

  const handleDeleteSaved = useCallback(
    async (pipeline: SavedPipeline) => {
      if (!window.confirm(`Delete "${pipeline.name}"? The runs it started stay.`)) return;
      try {
        await apiClient.deleteSavedPipeline(pipeline.id);
        if (openId === pipeline.id) setOpenId(null);
        refreshSaved();
      } catch (error: any) {
        toast.error(error?.response?.data?.detail || 'Could not delete');
      }
    },
    [openId, refreshSaved]
  );

  // Check as you type, once you stop. 600ms is long enough that a half-typed
  // stage id does not produce a wall of problems you did not ask about.
  useEffect(() => {
    const timer = window.setTimeout(runCheck, 600);
    return () => window.clearTimeout(timer);
  }, [runCheck]);

  // Startable, not merely valid: a pipeline that cannot be expressed as a
  // chain, or cannot afford its own budget, must not offer a Launch button
  // that the server is only going to refuse.
  const canLaunch = Boolean(
    check?.valid &&
      check.expressible &&
      check.plan &&
      (!check.budget || check.budget.affordable)
  );

  const handleLaunch = useCallback(async () => {
    if (!canLaunch || !parsed.value || !check?.plan) return;
    const total = check.plan.total_seconds;
    const stops = check.plan.checkpoints.length
      ? `\n\nIt will stop for you at: ${check.plan.checkpoints.join(', ')}.`
      : '';
    // The one place in this feature that spends anything, so the number and
    // the stopping points are said out loud before it does.
    const ok = window.confirm(
      `Start "${parsed.value.name || 'this pipeline'}"?\n\n` +
        `${check.plan.order.length} stages, about ${minutes(total)} of work.${stops}`
    );
    if (!ok) return;

    setLaunching(true);
    try {
      const budgetSeconds = budget.trim() ? Number(budget.trim()) : undefined;
      const started = await apiClient.launchPipeline(parsed.value, {
        budgetSeconds: budgetSeconds || undefined,
        // The estimate shown above is the estimate being agreed to. If the
        // spec changed since it was priced the server refuses rather than
        // running something nobody saw costed.
        acknowledgedSeconds: total,
        pipelineId: openId || undefined,
      });
      toast.success(`Started ${started.name}`);
      refreshSaved();
      navigate(`/autonomous-agents?job=${started.job_id}`);
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || 'Could not start the pipeline');
    } finally {
      setLaunching(false);
    }
    // openId so the run and the pipeline each know about the other.
  }, [canLaunch, parsed, check, budget, navigate, openId, refreshSaved]);

  // The graph never holds a model of its own: it hands back a whole spec and
  // that is serialised straight into the text. Two views of one document
  // cannot diverge, because there is nothing for them to diverge between.
  const applySpecFromGraph = useCallback(
    (next: Record<string, any>) => {
      setSource(JSON.stringify(next, null, 2));
    },
    []
  );

  const problems = check?.problems || [];
  const bindingProblems = check?.binding_problems || [];
  const plan = check?.plan;
  const budgetResult = check?.budget;

  return (
    <div className="p-6 h-full min-h-0 flex flex-col gap-4">
      <div className="flex items-center justify-between flex-none">
        <div className="flex items-center gap-2">
          <Workflow className="w-5 h-5 text-primary-600" />
          <h1 className="text-xl font-semibold text-gray-900">Pipeline Studio</h1>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-500" htmlFor="pipeline-budget">
            Budget (seconds)
          </label>
          <input
            id="pipeline-budget"
            value={budget}
            onChange={(e) => setBudget(e.target.value.replace(/[^0-9]/g, ''))}
            placeholder="optional"
            className="w-28 px-2 py-1 text-sm rounded-md bg-gray-50 border border-gray-300
              shadow-[inset_0_1px_2px_0_rgb(0_0_0_/_0.35)]
              focus:outline-none focus:border-primary-600 focus:shadow-accent-glow"
          />
          <div
            className="flex rounded-md border border-gray-300 overflow-hidden"
            role="group"
            aria-label="Editor view"
          >
            {(['text', 'split', 'graph'] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                aria-pressed={view === mode}
                className={`px-2.5 py-1 text-xs capitalize transition-colors duration-fast ${
                  view === mode
                    ? 'bg-primary-500/15 text-primary-700'
                    : 'text-gray-600 hover:bg-gray-200'
                }`}
                onClick={() => setView(mode)}
              >
                {mode}
              </button>
            ))}
          </div>
          <Button size="sm" variant="secondary" onClick={handleSave}>
            <Save className="w-4 h-4 mr-1" />
            {openId ? 'Save' : 'Save as…'}
          </Button>
          <select
            aria-label="Start from a worked example"
            className="h-8 rounded-md border border-gray-300 bg-gray-100 px-2 text-xs text-gray-700"
            value=""
            onChange={(e) => {
              const starter = STARTERS.find((s) => s.label === e.target.value);
              if (!starter) return;
              setOpenId(null);
              setSource(starter.source);
            }}
          >
            <option value="">New…</option>
            {STARTERS.map((s) => (
              <option key={s.label} value={s.label}>
                {s.label}
              </option>
            ))}
          </select>
          <Button
            size="sm"
            disabled={!canLaunch || launching}
            loading={launching}
            title={
              canLaunch
                ? 'Start this pipeline'
                : 'Fix the problems above before launching'
            }
            onClick={handleLaunch}
          >
            <Play className="w-4 h-4 mr-1" />
            Launch
          </Button>
        </div>
      </div>

      <p className="text-sm text-gray-500 flex-none max-w-3xl">
        A stage says what must be <em>true</em> when it is done, not which tools to
        run — those are derived from the contract. Everything below is decided
        before anything starts, and nothing here launches a run.
      </p>

      {saved.length > 0 && (
        <div className="flex-none flex flex-wrap gap-2" aria-label="Saved pipelines">
          {saved.map((pipeline) => (
            <div
              key={pipeline.id}
              className={`group flex items-center gap-2 pl-2.5 pr-1 py-1 rounded-full border text-xs
                transition-all duration-fast ease-ui
                ${
                  openId === pipeline.id
                    ? 'bg-primary-500/10 border-primary-500/60 text-primary-700'
                    : 'bg-gray-100 border-gray-300 text-gray-700 hover:border-gray-400'
                }`}
            >
              <button
                type="button"
                className="flex items-center gap-1.5"
                onClick={() => handleOpen(pipeline)}
                title={
                  pipeline.launch_count
                    ? `Run ${pipeline.launch_count} time(s)`
                    : 'Never run'
                }
              >
                {/* The verdict from when it was saved. A hint, not an answer:
                    the studio re-checks whatever it opens, because tools and
                    costs move underneath a stored spec. */}
                {pipeline.last_check_valid === 'valid' ? (
                  <CheckCircle2 className="w-3 h-3 text-primary-700" />
                ) : (
                  <AlertTriangle className="w-3 h-3 text-yellow-400" />
                )}
                {pipeline.name}
                {pipeline.launch_count > 0 && (
                  <span className="font-mono text-[10px] text-gray-500">
                    ×{pipeline.launch_count}
                  </span>
                )}
              </button>
              <button
                type="button"
                aria-label={`Delete ${pipeline.name}`}
                className="p-0.5 rounded opacity-0 group-hover:opacity-100 focus:opacity-100 text-gray-500 hover:text-red-300 transition-opacity duration-fast"
                onClick={() => handleDeleteSaved(pipeline)}
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-0">
        {/* The spec, as text and/or as a graph. Both edit the same document:
            the graph hands back a whole spec and it is serialised into the
            text, so neither view holds state the other cannot see. */}
        <div className="flex flex-col min-h-0 gap-3">
          <div className="flex items-center justify-between">
            <h2 className="section-heading mb-0">
              {view === 'graph' ? 'Shape' : 'Specification'}
            </h2>
            {checking && <span className="text-xs text-gray-500">checking…</span>}
          </div>

          {view !== 'graph' && (
            <textarea
              aria-label="Pipeline specification"
              spellCheck={false}
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className={`w-full rounded-lg bg-gray-50 border border-gray-300
                px-3 py-2 font-mono text-xs leading-relaxed text-gray-900
                shadow-[inset_0_1px_2px_0_rgb(0_0_0_/_0.35)]
                focus:outline-none focus:border-primary-600 focus:shadow-accent-glow
                ${view === 'split' ? 'h-1/2 min-h-[200px]' : 'flex-1 min-h-[320px]'}`}
            />
          )}

          {view !== 'text' && (
            <div
              className={`rounded-lg border border-gray-200 overflow-hidden bg-gray-50
                ${view === 'split' ? 'flex-1 min-h-[220px]' : 'flex-1 min-h-[320px]'}`}
            >
              {parsed.value && !parsed.error ? (
                <PipelineGraph
                  spec={parsed.value}
                  check={check}
                  onChange={applySpecFromGraph}
                  selectedStageId={selectedStage}
                  onSelectStage={setSelectedStage}
                />
              ) : (
                // No picture of a document that does not parse. Drawing the
                // last good shape would show a graph that is not the one being
                // edited, which is worse than showing none.
                <div className="h-full flex items-center justify-center p-4 text-center">
                  <p className="text-xs text-gray-500">
                    The graph appears once the specification parses.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* What is wrong with it, and what it will cost */}
        <div className="flex flex-col min-h-0 overflow-y-auto scrollbar-thin space-y-3">
          {parseError && (
            <div className="rounded-lg border border-red-500/60 bg-red-500/10 p-3">
              <h3 className="section-heading mb-1 text-red-300">Not valid JSON</h3>
              <p className="text-xs font-mono text-red-200">{parseError}</p>
            </div>
          )}

          {serverError && !parseError && (
            <div className="rounded-lg border border-red-500/60 bg-red-500/10 p-3">
              <h3 className="section-heading mb-1 text-red-300">Not a pipeline</h3>
              <p className="text-xs text-red-200">{serverError}</p>
            </div>
          )}

          {check && !parseError && (
            <>
              {/* Validity, expressibility and affordability are three answers,
                  shown as three, because a pipeline can be well formed,
                  compile to a chain, and still be unaffordable. */}
              <div className="flex flex-wrap gap-2">
                <Verdict ok={check.valid} label={check.valid ? 'Spec valid' : 'Spec invalid'} />
                {check.valid && (
                  <Verdict
                    ok={check.expressible}
                    label={check.expressible ? 'Runs as a chain' : 'Cannot run as a chain'}
                  />
                )}
                {budgetResult && (
                  <Verdict
                    ok={budgetResult.affordable}
                    label={
                      budgetResult.affordable
                        ? `Fits ${minutes(budgetResult.budget_seconds)}`
                        : `Needs ${minutes(budgetResult.estimated_seconds)}`
                    }
                  />
                )}
              </div>

              {problems.length > 0 && (
                <Panel title={`${problems.length} problem${problems.length === 1 ? '' : 's'}`} tone="bad">
                  <ul className="space-y-1.5">
                    {problems.map((p) => (
                      <li key={p} className="text-xs text-red-200 flex gap-2">
                        <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                        <span>{p}</span>
                      </li>
                    ))}
                  </ul>
                </Panel>
              )}

              {bindingProblems.length > 0 && (
                <Panel title="Cannot be expressed as a chain" tone="bad">
                  <ul className="space-y-1.5">
                    {bindingProblems.map((p) => (
                      <li key={p} className="text-xs text-red-200">
                        {p}
                      </li>
                    ))}
                  </ul>
                </Panel>
              )}

              {plan && (
                <Panel title="Plan" tone="ok">
                  <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-gray-600 mb-3">
                    <span className="flex items-center gap-1.5">
                      <Layers className="w-3.5 h-3.5" />
                      {plan.stages.length} stage{plan.stages.length === 1 ? '' : 's'}
                    </span>
                    <span className="flex items-center gap-1.5">
                      <Clock className="w-3.5 h-3.5" />
                      {minutes(plan.total_seconds)} total
                    </span>
                    <span className="flex items-center gap-1.5">
                      <GitBranch className="w-3.5 h-3.5" />
                      {minutes(plan.critical_path_seconds)} on the longest path
                    </span>
                    {plan.checkpoints.length > 0 && (
                      <span className="flex items-center gap-1.5 text-primary-700">
                        <PauseCircle className="w-3.5 h-3.5" />
                        stops at {plan.checkpoints.join(', ')}
                      </span>
                    )}
                  </div>

                  <ol className="space-y-2">
                    {plan.order.map((stageId, index) => {
                      const stage = plan.stages.find((s) => s.stage_id === stageId);
                      return (
                        <li
                          key={stageId}
                          className="bg-white border border-gray-200 rounded-lg p-2.5"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-xs font-medium text-gray-900">
                              <span className="font-mono text-gray-500 mr-2">
                                {index + 1}
                              </span>
                              {stageId}
                              {stage?.checkpoint && (
                                <span className="ml-2 text-[10px] uppercase tracking-wide text-primary-700">
                                  checkpoint
                                </span>
                              )}
                            </span>
                            <span className="text-xs font-mono text-gray-500">
                              {minutes(stage?.seconds || 0)}
                              {stage && stage.iterations > 1 && ` ×${stage.iterations}`}
                            </span>
                          </div>
                          {/* The derived tools. This is the part you cannot get
                              wrong by hand, because you never write it. */}
                          <div className="mt-1.5 flex flex-wrap gap-1">
                            {(stage?.tools || []).map((tool) => (
                              <span
                                key={tool}
                                className="px-1.5 py-0.5 rounded bg-gray-200 text-[10px] font-mono text-gray-700"
                              >
                                {tool}
                              </span>
                            ))}
                            {(stage?.unpriced || []).length > 0 && (
                              <span
                                className="px-1.5 py-0.5 rounded bg-yellow-500/15 text-[10px] text-yellow-300"
                                title="No recorded cost — counted as zero, which is not the same as free"
                              >
                                {stage!.unpriced.length} unpriced
                              </span>
                            )}
                          </div>
                        </li>
                      );
                    })}
                  </ol>
                </Panel>
              )}

              {budgetResult?.caveat && (
                <p className="text-xs text-yellow-300">{budgetResult.caveat}</p>
              )}

              {check.description.length > 0 && (
                <Panel title="In words" tone="plain">
                  <pre className="text-xs font-mono text-gray-600 whitespace-pre-wrap">
                    {check.description.join('\n')}
                  </pre>
                </Panel>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

const Verdict: React.FC<{ ok: boolean; label: string }> = ({ ok, label }) => (
  <span
    className={clsx(
      'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border',
      ok
        ? 'bg-primary-500/10 text-primary-700 border-primary-500/60'
        : 'bg-red-500/10 text-red-300 border-red-500/60'
    )}
  >
    {ok ? (
      <CheckCircle2 className="w-3.5 h-3.5" />
    ) : (
      <AlertTriangle className="w-3.5 h-3.5" />
    )}
    {label}
  </span>
);

const Panel: React.FC<{
  title: string;
  tone: 'ok' | 'bad' | 'plain';
  children: React.ReactNode;
}> = ({ title, tone, children }) => (
  <div
    className={clsx(
      'rounded-lg p-3 border',
      tone === 'bad' ? 'border-red-500/40 bg-red-500/5' : 'bg-white border-gray-200'
    )}
  >
    <h3 className="section-heading">{title}</h3>
    {children}
  </div>
);

export default PipelineStudioPage;
