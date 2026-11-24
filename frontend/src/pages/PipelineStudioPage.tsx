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
 * Checks belong to the exact spec and budget being edited. The estimate on
 * screen is also sent with launch so the server can refuse a changed price.
 *
 * A pipeline that has run also shows its last run: how far it got, which stage
 * it is on, and whether the stages that finished produced what they promised.
 * That belongs beside the editor rather than only on the job page, because the
 * question a spec raises after it has run once is what it did the last time.
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
  RefreshCw,
  Save,
  Sparkles,
  Trash2,
  Workflow,
} from 'lucide-react';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

import Button from '../components/common/Button';
import PipelineGraph from '../components/pipelines/PipelineGraph';
import PipelineLaunchReview, { formatPipelineTime } from '../components/pipelines/PipelineLaunchReview';
import PipelineRunProgress from '../components/pipelines/PipelineRunProgress';
import StageInspector from '../components/pipelines/StageInspector';
import BindingPreviewPanel from '../components/pipelines/BindingPreviewPanel';
import { CPU_CO_DESIGN_PIPELINE } from '../utils/cpuCoDesignStudy';
import { removePipelineStages, replacePipelineStage } from '../utils/pipelineEditing';
import { apiClient } from '../services/api';
import type {
  ChainImportCandidate,
  ContractSuggestion,
  PipelineCheck,
  PipelineVocabulary,
  SavedPipeline,
} from '../types';

const STORAGE_KEY = 'pipeline_studio_draft_v1';
const DRAFT_RECOVERY_KEY = 'pipeline_studio_recoverable_drafts_v1';

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
      "goal": "Score the measurement against the paper's claim, or say which condition makes them incomparable. Provide the source document ID and the claim's location in the paper when recorded. Record units, conditions, and the measurement source for the comparison.",
      "depends_on": ["measure"],
      "assumes": ["algorithm_spec", "benchmark_measurement", "implementation_verified"],
      "checkpoint": true,
      "contract": { "required_finding_types": ["reproduction_verdict"] }
    }
  ]
}`;

const STARTERS: { label: string; source: string; initialStage?: string }[] = [
  { label: 'Bottleneck study', source: EXAMPLE },
  { label: 'CPU co-design', source: JSON.stringify(CPU_CO_DESIGN_PIPELINE, null, 2), initialStage: 'validate-baseline' },
  { label: 'Reproduce a paper', source: REPRODUCE_PAPER },
];

const readDraft = (): string => {
  try {
    return window.localStorage.getItem(STORAGE_KEY) || EXAMPLE;
  } catch {
    return EXAMPLE;
  }
};

interface RecoverableDraft {
  source: string;
  budget: string;
  openId: string | null;
  storedAt: number;
}

function readRecoverableDrafts(): RecoverableDraft[] {
  try {
    const drafts: unknown = JSON.parse(window.localStorage.getItem(DRAFT_RECOVERY_KEY) || '[]');
    return Array.isArray(drafts) ? drafts.filter((draft): draft is RecoverableDraft =>
      draft && typeof draft.source === 'string' && typeof draft.budget === 'string' &&
      (draft.openId === null || typeof draft.openId === 'string') &&
      Number.isFinite(draft.storedAt)
    ) : [];
  } catch {
    return [];
  }
}

function draftName(source: string): string {
  try {
    const name: unknown = JSON.parse(source)?.name;
    return typeof name === 'string' && name.trim() ? name : 'Untitled pipeline';
  } catch {
    return 'Untitled pipeline';
  }
}

const minutes = (seconds: number) =>
  seconds < 60 ? `${seconds}s` : `${Math.round(seconds / 60)} min`;

const PipelineStudioPage: React.FC = () => {
  const navigate = useNavigate();
  const [source, setSource] = useState<string>(readDraft);
  const [budget, setBudget] = useState<string>('');
  const enteredBudget = budget.trim();
  const budgetNumber = Number(enteredBudget);
  const budgetError = Boolean(enteredBudget) &&
    (!/^\d+$/.test(enteredBudget) || !Number.isSafeInteger(budgetNumber) || budgetNumber < 1);
  const budgetSeconds = enteredBudget && !budgetError ? budgetNumber : undefined;
  const [recoverableDrafts, setRecoverableDrafts] = useState(readRecoverableDrafts);
  const [catalogRevision, setCatalogRevision] = useState(0);
  const checkKey = JSON.stringify([source, budget, catalogRevision]);
  const [checked, setChecked] = useState<{ key: string; result: PipelineCheck } | null>(null);
  const check = checked?.key === checkKey ? checked.result : null;
  const checkRequest = useRef(0);
  const currentCheckKey = useRef(checkKey);
  currentCheckKey.current = checkKey;
  const [parseError, setParseError] = useState<string>('');
  const [serverError, setServerError] = useState<string>('');
  const [checking, setChecking] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [launchReview, setLaunchReview] = useState<{ key: string; pipelineId: string | null } | null>(null);
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

  useEffect(() => {
    try {
      window.localStorage.setItem(DRAFT_RECOVERY_KEY, JSON.stringify(recoverableDrafts));
    } catch {
      // The drafts remain available for the current editor session.
    }
  }, [recoverableDrafts]);

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
    const request = ++checkRequest.current;
    const isCurrent = () => request === checkRequest.current && checkKey === currentCheckKey.current;
    setParseError(parsed.error);
    setServerError('');
    setChecked(null);
    if (parsed.error) {
      setChecking(false);
      return;
    }
    if (!parsed.value || typeof parsed.value !== 'object' || Array.isArray(parsed.value)) {
      setServerError('The pipeline specification must be a JSON object.');
      setChecking(false);
      return;
    }
    setChecking(true);
    try {
      const result = await apiClient.checkPipeline(parsed.value, budgetSeconds);
      if (isCurrent()) setChecked({ key: checkKey, result });
    } catch (error: any) {
      // A 400 here means the shape is wrong in a way the parser accepted —
      // valid JSON that is not a pipeline. The server says which.
      if (isCurrent()) {
        setServerError(error?.response?.data?.detail || 'Could not check the pipeline');
        setChecked(null);
      }
    } finally {
      if (request === checkRequest.current) setChecking(false);
    }
  }, [parsed, budgetSeconds, checkKey]);

  useEffect(() => () => { checkRequest.current += 1; }, []);
  useEffect(() => { setParseError(''); setServerError(''); }, [checkKey]);

  const [saved, setSaved] = useState<SavedPipeline[]>([]);
  // Which saved pipeline the editor is currently showing. null means the
  // draft has never been saved, which is why Save asks for a name and Update
  // does not.
  const [openId, setOpenId] = useState<string | null>(null);

  const rememberDraft = (restored?: RecoverableDraft) => {
    const uneditedExample = !budget && !openId && STARTERS.some((starter) => starter.source === source);
    const snapshot = { source, budget, openId, storedAt: Date.now() };
    setRecoverableDrafts((drafts) => {
      const remaining = drafts.filter((draft) => draft !== restored);
      if (uneditedExample || remaining.some((draft) =>
        draft.source === source && draft.budget === budget && draft.openId === openId
      )) return remaining;
      return [snapshot, ...remaining];
    });
  };

  const resumeDraft = (draft: RecoverableDraft) => {
    rememberDraft(draft);
    setOpenId(draft.openId);
    setSelectedStage(null);
    setSource(draft.source);
    setBudget(draft.budget);
  };

  /** The current owner's tools and active skills may change in another tab.
   *  Refresh metadata without replacing the spec or remounting its editors. */
  const [vocabulary, setVocabulary] = useState<PipelineVocabulary | null>(null);
  const [vocabularyLoading, setVocabularyLoading] = useState(true);
  const [vocabularyError, setVocabularyError] = useState('');
  const vocabularyRequest = useRef(0);
  const refreshVocabulary = useCallback(async () => {
    const request = ++vocabularyRequest.current;
    setVocabularyLoading(true);
    setVocabularyError('');
    // A catalog refresh must not leave Launch using an older capability check.
    checkRequest.current += 1;
    setChecked(null);
    setChecking(false);
    try {
      const result = await apiClient.getPipelineVocabulary();
      if (request === vocabularyRequest.current) setVocabulary(result);
    } catch {
      if (request === vocabularyRequest.current) {
        setVocabularyError('Could not refresh the tool catalog. Retry to load current tools and skills. Your stage settings are preserved.');
      }
    } finally {
      if (request === vocabularyRequest.current) {
        setVocabularyLoading(false);
        setCatalogRevision((revision) => revision + 1);
      }
    }
  }, []);
  useEffect(() => {
    refreshVocabulary();
    return () => { vocabularyRequest.current += 1; };
  }, [refreshVocabulary]);

  const [describing, setDescribing] = useState('');
  const [drafting, setDrafting] = useState(false);

  /** Describe what you want; get a pipeline to edit.
   *
   *  The blank state here is a JSON document in a format nobody knows, and the
   *  worked examples only help someone whose question resembles one of them.
   *  What comes back is a STARTING POINT: it lands in the editor and is
   *  checked there like anything typed by hand, and nothing is launched. */
  const handleDraft = useCallback(async () => {
    const wanted = describing.trim();
    if (!wanted) return;
    setDrafting(true);
    try {
      const result = await apiClient.draftPipeline(wanted, budgetSeconds);
      setOpenId(null);
      setSelectedStage(null);
      setSource(JSON.stringify(result.spec, null, 2));
      setDescribing('');
      if (result.problems.length) {
        // Said plainly rather than hidden: a draft that does not check is
        // still worth having, and the author needs to know it is theirs to
        // finish rather than ready to run.
        toast(
          `Drafted, with ${result.problems.length} thing(s) still to fix — see the checks.`,
          { icon: '\u26a0\ufe0f' }
        );
      } else {
        toast.success(result.repaired ? 'Drafted (repaired once)' : 'Drafted');
      }
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || 'Could not draft a pipeline');
    } finally {
      setDrafting(false);
    }
  }, [describing, budgetSeconds]);

  /** The saved pipeline currently in the editor, when there is one. Its last
   *  run is the only thing on this page that already cost something. */
  const openPipeline = useMemo(
    () => (openId ? saved.find((p) => p.id === openId) || null : null),
    [openId, saved]
  );

  /** Saved chains that could become pipelines.
   *
   *  Chains are being retired as a way to author work: a chain says *when* the
   *  next step fires, a pipeline says *what must be true* when a stage is done.
   *  They produce the same runtime, so this is a rewrite of the description
   *  rather than a migration of anything running. Empty for anyone with no
   *  chains, which is why the panel hides itself rather than explaining. */
  const [importable, setImportable] = useState<ChainImportCandidate[]>([]);
  const [importing, setImporting] = useState<string | null>(null);
  /** Evidence each contract-less stage might be asking for.
   *
   *  Naming the evidence is the hardest step in writing a pipeline: it means
   *  knowing which types exist, which tool makes each, and what the stage's job
   *  type may call. These are proposals with their reasons attached, never
   *  applied on their own. */
  const [contractHints, setContractHints] = useState<Record<string, ContractSuggestion[]>>({});
  /** Values for a chain's {placeholders}, keyed chain id then variable name.
   *  A chain is often a template and a pipeline goal is literal, so these have
   *  to be supplied at import or the goal keeps the braces. */
  const [chainVars, setChainVars] = useState<Record<string, Record<string, string>>>({});

  useEffect(() => {
    const spec = parsed.value;
    if (!spec) {
      setContractHints({});
      return;
    }
    let live = true;
    // Debounced: this follows the editor, and a request per keystroke would be
    // noise for an answer that only changes when a goal does.
    const timer = window.setTimeout(async () => {
      try {
        const result = await apiClient.suggestContracts(spec);
        if (live) setContractHints(result.suggestions || {});
      } catch {
        if (live) setContractHints({});
      }
    }, 400);
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [parsed.value]);

  /** Write a suggested evidence type into a stage's contract.
   *
   *  Edits the text the author is looking at rather than some parallel state:
   *  the spec in the editor stays the one true version of what they have said.
   */
  const applyContractHint = useCallback(
    (stageId: string, findingType: string) => {
      const spec = parsed.value as any;
      if (!spec) return;
      const next = JSON.parse(JSON.stringify(spec));
      const stage = (next.stages || []).find((s: any) => s?.id === stageId);
      if (!stage) return;
      const contract = (stage.contract && typeof stage.contract === 'object')
        ? stage.contract
        : {};
      const existing: string[] = Array.isArray(contract.required_finding_types)
        ? contract.required_finding_types
        : [];
      if (existing.includes(findingType)) return;
      stage.contract = { ...contract, required_finding_types: [...existing, findingType] };
      setSource(JSON.stringify(next, null, 2));
    },
    [parsed.value]
  );

  const refreshImportable = useCallback(async () => {
    try {
      const survey = await apiClient.surveyChainsForImport();
      setImportable(survey.candidates || []);
    } catch {
      // Nothing to say: the studio works without the panel.
    }
  }, []);

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
    refreshImportable();
  }, [refreshSaved, refreshImportable]);

  const handleImportChain = useCallback(
    async (candidate: ChainImportCandidate) => {
      setImporting(candidate.chain_id);
      try {
        const pipeline = await apiClient.importChainAsPipeline(
          candidate.chain_id,
          undefined,
          chainVars[candidate.chain_id]
        );
        toast.success(
          `Imported ${pipeline.name}. ${candidate.contracts_to_write} stage${
            candidate.contracts_to_write === 1 ? '' : 's'
          } still need a contract.`
        );
        await refreshSaved();
        await refreshImportable();
        setOpenId(pipeline.id);
        setSelectedStage(null);
        setSource(JSON.stringify(pipeline.spec, null, 2));
      } catch (error: any) {
        const detail = error?.response?.data?.detail;
        toast.error(
          typeof detail === 'object' && detail?.blockers?.length
            ? `${candidate.name}: ${detail.blockers[0].step} cannot be a stage`
            : `Could not import ${candidate.name}`
        );
      } finally {
        setImporting(null);
      }
    },
    [refreshSaved, refreshImportable, chainVars]
  );

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
    // Stage ids are local to a pipeline. Unmount its editors so unapplied
    // defaults and bindings cannot carry into another document with the same ids.
    setSelectedStage(null);
    setSource(JSON.stringify(pipeline.spec, null, 2));
    // Whatever verdict was cached when it was saved is not re-used: tools and
    // their costs move underneath a stored spec, so it is checked again.
    setChecked(null);
  }, []);

  const handleDeleteSaved = useCallback(
    async (pipeline: SavedPipeline) => {
      if (!window.confirm(`Delete "${pipeline.name}"? The runs it started stay.`)) return;
      try {
        await apiClient.deleteSavedPipeline(pipeline.id);
        if (openId === pipeline.id) setOpenId(null);
        setRecoverableDrafts((drafts) => drafts.map((draft) =>
          draft.openId === pipeline.id ? { ...draft, openId: null } : draft
        ));
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
    !budgetError && !checking && !vocabularyLoading && check?.valid &&
      check.expressible &&
      check.plan &&
      !check.launch_problems?.length &&
      (!check.budget || check.budget.affordable)
  );
  const reviewIsCurrent = launchReview?.key === checkKey && launchReview.pipelineId === openId;
  const closeLaunchReview = useCallback(() => setLaunchReview(null), []);
  useEffect(() => closeLaunchReview(), [checkKey, openId, closeLaunchReview]);

  const handleLaunch = useCallback(async () => {
    if (!reviewIsCurrent || !canLaunch || launching || !parsed.value || !check?.plan) return;
    const total = check.plan.total_seconds;
    setLaunching(true);
    try {
      const started = await apiClient.launchPipeline(parsed.value, {
        budgetSeconds,
        // The estimate shown above is the estimate being agreed to. If the
        // price changed on the server since the check, it refuses rather than
        // running something nobody saw costed.
        acknowledgedSeconds: total,
        pipelineId: openId || undefined,
      });
      closeLaunchReview();
      toast.success(`Started ${started.name}`);
      refreshSaved();
      navigate(`/autonomous-agents?job=${started.job_id}`);
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || 'Could not start the pipeline');
      if ([409, 422].includes(error?.response?.status)) {
        closeLaunchReview();
        void runCheck();
      }
    } finally {
      setLaunching(false);
    }
    // openId so the run and the pipeline each know about the other.
  }, [reviewIsCurrent, canLaunch, launching, parsed, check, budgetSeconds, navigate, openId, refreshSaved, closeLaunchReview, runCheck]);

  // The graph never holds a model of its own: it hands back a whole spec and
  // that is serialised straight into the text. Two views of one document
  // cannot diverge, because there is nothing for them to diverge between.
  const applySpecFromGraph = useCallback(
    (next: Record<string, any>) => {
      setSource(JSON.stringify(next, null, 2));
    },
    []
  );

  /** The stage the graph has selected, read out of the spec rather than
   *  stored: the spec is the only source of truth, so a selection that
   *  survived a rename or a delete would point at something gone. */
  const selectedStageObject = useMemo(() => {
    if (!selectedStage || !Array.isArray(parsed.value?.stages)) return null;
    const found = (parsed.value.stages || []).find(
      (st: any) => String(st?.id) === selectedStage
    );
    return found || null;
  }, [selectedStage, parsed]);

  const problems = check?.problems || [];
  const bindingProblems = check?.binding_problems || [];
  const launchProblems = check?.launch_problems || [];
  const plan = check?.plan;
  const budgetResult = check?.budget;
  let readiness = 'Checking launch requirements…';
  if (budgetError) {
    readiness = 'Enter a whole number of seconds greater than zero.';
  } else if (parsed.error) {
    readiness = 'Fix the specification JSON to review launch.';
  } else if (serverError) {
    readiness = 'The check could not evaluate this pipeline.';
  } else if (budgetResult && !budgetResult.affordable) {
    readiness = `Estimated work is ${formatPipelineTime(budgetResult.estimated_seconds)} against a ${formatPipelineTime(budgetResult.budget_seconds)} budget.`;
  } else if (canLaunch && plan) {
    const count = plan.order.length;
    readiness = `${count} ${count === 1 ? 'stage' : 'stages'} · ${formatPipelineTime(plan.total_seconds)} estimated work`;
  } else if (check && !vocabularyLoading) {
    readiness = 'Resolve the issues in the check results to review launch.';
  }

  return (
    <div className="flex min-h-full min-w-0 flex-col gap-4 p-4 lg:h-full lg:p-6">
      <div className="flex flex-none flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Workflow className="w-5 h-5 text-primary-600" />
          <h1 className="text-xl font-semibold text-gray-900">Pipeline Studio</h1>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="text-xs text-gray-500" htmlFor="pipeline-budget">
            Budget (seconds)
          </label>
          <input
            id="pipeline-budget"
            value={budget}
            inputMode="numeric"
            aria-invalid={budgetError}
            aria-describedby="pipeline-readiness"
            onChange={(e) => setBudget(e.target.value)}
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
              rememberDraft();
              setOpenId(null);
              setSelectedStage(starter.initialStage ?? null);
              setSource(starter.source);
              setBudget('');
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
            aria-describedby="pipeline-readiness"
            onClick={() => setLaunchReview({ key: checkKey, pipelineId: openId })}
          >
            <Play className="w-4 h-4 mr-1" />
            Review launch
          </Button>
        </div>
      </div>

      <div className="flex flex-none flex-wrap items-baseline gap-x-2 gap-y-1 rounded-md border border-gray-200 px-3 py-2 text-sm">
        <span className="font-medium text-gray-900">Launch readiness</span>
        <span id="pipeline-readiness" role="status" className="min-w-0 break-words text-gray-600">{readiness}</span>
      </div>
      {reviewIsCurrent && canLaunch && parsed.value && plan && (
        <PipelineLaunchReview
          spec={parsed.value}
          plan={plan}
          budgetSeconds={budgetSeconds}
          launching={launching}
          onClose={closeLaunchReview}
          onLaunch={handleLaunch}
        />
      )}
      {recoverableDrafts.length > 0 && (
        <details className="max-w-3xl flex-none rounded-md border border-gray-300 bg-gray-100 px-3 py-2 text-sm">
          <summary className="cursor-pointer font-medium text-primary-700">Recover drafts ({recoverableDrafts.length})</summary>
          <ul className="mt-2 max-h-40 space-y-1 overflow-y-auto">
            {recoverableDrafts.map((draft) => (
              <li key={JSON.stringify([draft.storedAt, draft.source, draft.budget, draft.openId])} className="flex flex-wrap items-center justify-between gap-2">
                <div className="min-w-0 break-words">
                  <span className="font-medium text-gray-900">{draftName(draft.source)}</span>
                  <span className="ml-2 text-xs text-gray-600">{new Date(draft.storedAt).toLocaleString()}</span>
                </div>
                <Button type="button" variant="secondary" size="sm" onClick={() => resumeDraft(draft)}>
                  Resume draft
                </Button>
              </li>
            ))}
          </ul>
        </details>
      )}
      {parsed.value?.name === CPU_CO_DESIGN_PIPELINE.name && (
        <div className="max-w-3xl flex-none space-y-1 rounded-md border border-gray-300 bg-gray-100 px-3 py-2 text-xs text-gray-600">
          <p>Edit the baseline stage with your kernel revisions and two input shapes. This example uses gem5 from the microarchitecture research program.</p>
          {vocabulary && !vocabulary.evidence_types.some((entry) => entry.name === 'simulated_measurement') && (
            <p className="text-amber-300">Simulation evidence is unavailable. Enable the microarchitecture research program before launching.</p>
          )}
        </div>
      )}

      {/* Say what you want. The other way into a format nobody knows: the
          starters only help an author whose question resembles one of them. */}
      <div className="flex-none flex items-center gap-2">
        <Sparkles className="w-4 h-4 text-primary-600 flex-none" />
        <input
          aria-label="Describe the pipeline you want"
          value={describing}
          onChange={(e) => setDescribing(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !drafting) handleDraft();
          }}
          placeholder="Describe what you want done — e.g. read the FlashAttention paper, implement it, and check the speedup reproduces"
          className="flex-1 px-2.5 py-1.5 text-sm rounded-md bg-gray-50 border border-gray-300
            text-gray-900 placeholder:text-gray-500
            shadow-[inset_0_1px_2px_0_rgb(0_0_0_/_0.35)]
            focus:outline-none focus:border-primary-600 focus:shadow-accent-glow"
        />
        <Button
          size="sm"
          variant="secondary"
          loading={drafting}
          disabled={drafting || !describing.trim()}
          onClick={handleDraft}
          title="Draft a pipeline from this description. Launches nothing."
        >
          Draft it
        </Button>
      </div>

      {importable.length > 0 && (
        <div
          className="flex-none rounded-lg border border-amber-500/40 bg-amber-500/5 p-3"
          aria-label="Job chains you can import"
        >
          <div className="text-xs font-medium text-gray-800 mb-1">
            Job chains you can bring here
          </div>
          <p className="text-xs text-gray-600 mb-2.5 max-w-prose">
            A chain says when the next step fires; a pipeline says what must be
            true when a stage is done. Importing rewrites the description — it
            copies, and leaves the chain running.
          </p>
          <div className="flex flex-wrap gap-2">
            {importable.map((candidate) => {
              // A server predating this field omits it; the page still works,
              // it just has no values to ask for.
              const variables = candidate.variables || [];
              return (
              <div
                key={candidate.chain_id}
                className={`flex items-center gap-2 pl-2.5 pr-1 py-1 rounded-full border text-xs ${
                  candidate.convertible
                    ? 'border-gray-300 bg-gray-50'
                    : 'border-rose-300 bg-rose-50'
                }`}
                title={
                  candidate.convertible
                    ? `${candidate.steps} steps · ${candidate.contracts_to_write} contract${
                        candidate.contracts_to_write === 1 ? '' : 's'
                      } to write after importing`
                    : candidate.blockers
                        .map((b) => `${b.step} (${b.trigger}): ${b.reason}`)
                        .join('; ')
                }
              >
                <span className="font-medium text-gray-800">{candidate.name}</span>
                {candidate.convertible ? (
                  <>
                    <span className="text-gray-500">
                      {candidate.contracts_to_write} contract
                      {candidate.contracts_to_write === 1 ? '' : 's'} to write
                    </span>
                    {/* A chain is often a template. Its goals keep the braces
                        unless the values are given here. */}
                    {variables.map((variable) => (
                      <input
                        key={variable}
                        className="w-28 px-1.5 py-0.5 rounded border border-gray-300 bg-white text-xs"
                        placeholder={variable}
                        aria-label={`${variable} for ${candidate.name}`}
                        value={chainVars[candidate.chain_id]?.[variable] || ''}
                        onChange={(e) =>
                          setChainVars((prev) => ({
                            ...prev,
                            [candidate.chain_id]: {
                              ...(prev[candidate.chain_id] || {}),
                              [variable]: e.target.value,
                            },
                          }))
                        }
                      />
                    ))}
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={
                        importing === candidate.chain_id ||
                        variables.some(
                          (v) => !(chainVars[candidate.chain_id]?.[v] || '').trim()
                        )
                      }
                      onClick={() => handleImportChain(candidate)}
                    >
                      {importing === candidate.chain_id ? 'Importing…' : 'Import'}
                    </Button>
                  </>
                ) : (
                  // Named rather than hidden: a chain that cannot be a pipeline
                  // is a thing to decide about, not an error to suppress.
                  <span className="text-rose-700">
                    {candidate.blockers[0]?.step} needs {candidate.blockers[0]?.trigger}
                  </span>
                )}
              </div>
              );
            })}
          </div>
        </div>
      )}

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

      <div className="grid grid-cols-1 gap-4 lg:min-h-0 lg:flex-1 lg:grid-cols-2">
        {/* The spec, as text and/or as a graph. Both edit the same document:
            the graph hands back a whole spec and it is serialised into the
            text, so neither view holds state the other cannot see. */}
        <div className="flex min-w-0 flex-col gap-3 lg:min-h-0">
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
                ${view === 'split' ? 'h-56 min-h-[200px] lg:h-1/2' : 'min-h-[320px] lg:flex-1'}`}
            />
          )}

          {view !== 'text' && (
            <div
              className={`rounded-lg border border-gray-200 overflow-hidden bg-gray-50
                ${view === 'split' ? 'h-64 min-h-[220px] lg:h-auto lg:flex-1' : 'h-[320px] min-h-[320px] lg:h-auto lg:flex-1'}`}
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
        <div className="flex min-w-0 flex-col space-y-3 scrollbar-thin lg:min-h-0 lg:overflow-y-auto">
          <section aria-label="Tool catalog" className="rounded-lg border border-gray-300 p-3 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-gray-700">Tool catalog</span>
              <button type="button" onClick={refreshVocabulary} disabled={vocabularyLoading}
                className="inline-flex items-center gap-1 text-xs text-primary-700 underline disabled:opacity-50">
                <RefreshCw className="w-3 h-3" />Refresh tools and skills
              </button>
            </div>
            {vocabularyLoading && <p role="status" className="text-xs text-gray-500">Loading tools and skills…</p>}
            {vocabularyError && <p role="alert" className="text-xs text-yellow-700">{vocabularyError}</p>}
            {vocabularyError && vocabulary && <p className="text-xs text-gray-500">Showing the last loaded catalog.</p>}
          </section>
          {/* The selected stage, editable. Selecting a node used to only
              highlight it, so everything a stage SAYS could still be changed
              only as JSON. Writes a whole new spec like every other gesture,
              so the text view cannot disagree with it. */}
          {selectedStageObject && (
            <StageInspector
              stage={selectedStageObject}
              allStageIds={parsed.value.stages.filter((st: any) => st?.id).map((st: any) => String(st.id))}
              vocabulary={vocabulary}
              vocabularyLoading={vocabularyLoading}
              onChange={(next) => {
                applySpecFromGraph(replacePipelineStage(parsed.value || {}, selectedStage!, next));
                // Renaming a stage moves the selection with it, or the panel
                // would close on the first keystroke in the id field.
                if (next.id !== selectedStage) setSelectedStage(next.id);
              }}
              onDelete={() => {
                applySpecFromGraph(removePipelineStages(parsed.value || {}, [selectedStage!]));
                setSelectedStage(null);
              }}
              onClose={() => setSelectedStage(null)}
            />
          )}

          {selectedStageObject && (
            <BindingPreviewPanel
              rootJobId={openPipeline?.last_job_id}
              stage={selectedStageObject}
              spec={parsed.value || {}}
              catalogRevision={catalogRevision}
              catalogRefreshing={vocabularyLoading}
              onOpenJob={(id) => navigate(`/autonomous-agents?job=${id}`)}
            />
          )}

          {/* And, for one that has run, what happened. Everything above this
              is decided before anything starts; the run is the only thing on
              the page that already spent something, so it goes first. */}
          {openPipeline?.last_job_id && (
            <PipelineRunProgress
              key={openPipeline.last_job_id}
              rootJobId={openPipeline.last_job_id}
              onOpenJob={(jobId) => navigate(`/autonomous-agents?job=${jobId}`)}
            />
          )}

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

              {Object.keys(contractHints).length > 0 && (
                <Panel title="Evidence these stages might need" tone="plain">
                  <p className="text-xs text-gray-600 mb-2.5">
                    Naming the evidence a stage must produce is the step that
                    decides whether it can stop honestly. These come from the
                    words the goal and the producing tool share — check the
                    reason before taking one.
                  </p>
                  <div className="space-y-2.5">
                    {Object.entries(contractHints).map(([stageId, hints]) => (
                      <div key={stageId}>
                        <div className="text-xs font-medium text-gray-800 mb-1">
                          {stageId}
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {hints.map((hint) => (
                            <button
                              key={hint.finding_type}
                              type="button"
                              onClick={() => applyContractHint(stageId, hint.finding_type)}
                              className="px-2 py-1 rounded border border-gray-300 bg-white text-xs
                                         text-left hover:border-primary-500 hover:bg-primary-500/5
                                         focus:outline-none focus:ring-2 focus:ring-primary-500/40"
                              /* The reason, not just the name: a suggestion
                                 matching one generic word is one to dismiss,
                                 and that is only visible if it is shown. */
                              title={`produced by ${hint.produced_by} · matched ${hint.matched.join(', ')}`}
                            >
                              <span className="font-mono">{hint.finding_type}</span>
                              <span className="text-gray-500"> · {hint.matched.join(' ')}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
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
              {launchProblems.length > 0 && (
                <Panel title="Launch blockers" tone="bad">
                  <ul className="space-y-1.5">
                    {launchProblems.map((problem) => (
                      <li key={problem} className="text-xs text-red-200">{problem}</li>
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
