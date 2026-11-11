/**
 * Edit one stage without writing JSON.
 *
 * The graph could wire stages and delete them; everything a stage actually
 * *says* — its goal, the evidence it must produce, whether it loops, whether a
 * person approves it — could only be changed in the text editor. So selecting
 * a node highlighted it and did nothing, and authoring was still a JSON job.
 *
 * The one field that matters most is the evidence, and it is a picker rather
 * than a text box on purpose. Requiring a finding type no tool produces is the
 * single most common way an authored pipeline fails its own check: the
 * contract reads perfectly and nothing in the system can satisfy it. A list
 * served from the tool specs makes that error unavailable rather than merely
 * reported — and it shows the two things an author is really choosing between,
 * which are what the evidence costs and which job type it forces.
 *
 * Like the graph, this holds no state of its own. Every gesture produces a new
 * stage object and hands it up; the spec stays the only source of truth, so
 * the text view can never disagree with this one.
 */

import clsx from 'clsx';
import { AlertTriangle, Check, Clock, Trash2, X } from 'lucide-react';
import React, { useEffect, useMemo, useState } from 'react';

import type { PipelineEvidenceType, PipelineVocabulary } from '../../types';
import { removeStageToolConfiguration, stageBindingSourceIssues } from '../../utils/pipelineEditing';
import type { PipelineStageSpec } from './PipelineGraph';
import StageToolPicker from './StageToolPicker';
import StageSkillDetails from './StageSkillDetails';

export interface StageInspectorProps {
  stage: PipelineStageSpec;
  /** Every stage id, so `depends_on` can only name one that exists. */
  allStageIds: string[];
  vocabulary: PipelineVocabulary | null;
  vocabularyLoading?: boolean;
  onChange: (next: PipelineStageSpec) => void;
  onDelete: () => void;
  onClose: () => void;
}

const FIELD =
  'w-full px-2 py-1 text-xs rounded-md bg-gray-50 border border-gray-300 ' +
  'text-gray-900 focus:outline-none focus:border-primary-600';

function seconds(n: number): string {
  if (!n) return '';
  return n < 60 ? `${n}s` : `${Math.round(n / 60)}m`;
}

export const StageInspector: React.FC<StageInspectorProps> = ({
  stage,
  allStageIds,
  vocabulary,
  vocabularyLoading,
  onChange,
  onDelete,
  onClose,
}) => {
  const [evidenceFilter, setEvidenceFilter] = useState('');
  const loadingVocabulary = vocabularyLoading ?? !vocabulary;
  const [idDraft, setIdDraft] = useState(stage.id);
  const [idError, setIdError] = useState('');
  useEffect(() => { setIdDraft(stage.id); setIdError(''); }, [stage.id]);

  const required: string[] = useMemo(
    () => (stage.contract?.required_finding_types as string[]) || [],
    [stage.contract]
  );

  const byName = useMemo(() => {
    const map: Record<string, PipelineEvidenceType> = {};
    (vocabulary?.evidence_types || []).forEach((e) => {
      map[e.name] = e;
    });
    return map;
  }, [vocabulary]);

  /** Job types every piece of required evidence permits. Empty means nothing
   *  constrains the choice; a non-empty list that excludes the current
   *  job_type is the mistake that plans tools a stage cannot call. */
  const forcedJobTypes = useMemo(() => {
    const constraints = required
      .map((name) => byName[name]?.job_types || [])
      .filter((list) => list.length > 0);
    if (!constraints.length) return [];
    return constraints.reduce((acc, list) => acc.filter((j) => list.includes(j)));
  }, [required, byName]);

  const jobType = String(stage.job_type || 'research');
  const jobTypeWrong = forcedJobTypes.length > 0 && !forcedJobTypes.includes(jobType);
  const bindingIssues = stageBindingSourceIssues(stage, allStageIds);

  const patch = (fields: Partial<PipelineStageSpec>) => onChange({ ...stage, ...fields });

  const toggleEvidence = (name: string) => {
    const next = required.includes(name)
      ? required.filter((n) => n !== name)
      : [...required, name];
    patch({ contract: { ...(stage.contract || {}), required_finding_types: next } });
  };

  const visibleEvidence = useMemo(() => {
    const all = vocabulary?.evidence_types || [];
    const needle = evidenceFilter.trim().toLowerCase();
    if (!needle) {
      // Unfiltered, the chosen ones first: fifty types is a scroll, and what
      // this stage already asks for is what you came to look at.
      return [...all].sort((a, b) => {
        const ax = required.includes(a.name) ? 0 : 1;
        const bx = required.includes(b.name) ? 0 : 1;
        return ax - bx || a.name.localeCompare(b.name);
      });
    }
    return all.filter((e) => [e.name, e.sandbox_skill?.name, e.sandbox_skill?.description, e.sandbox_skill?.image]
      .filter(Boolean).join(' ').toLowerCase().includes(needle));
  }, [vocabulary, evidenceFilter, required]);

  const totalSeconds = required.reduce(
    (sum, name) => sum + (byName[name]?.typical_seconds || 0),
    0
  );

  return (
    <div
      className="rounded-lg border border-primary-500/40 bg-gray-100 p-3 space-y-3"
      data-testid="stage-inspector"
    >
      <div className="flex items-center justify-between">
        <h3 className="section-heading mb-0">Stage</h3>
        <div className="flex items-center gap-1">
          <button
            type="button"
            aria-label={`Delete stage ${stage.id}`}
            className="p-1 rounded text-gray-500 hover:text-red-400 hover:bg-gray-200"
            onClick={onDelete}
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            aria-label="Close the stage editor"
            className="p-1 rounded text-gray-500 hover:text-gray-800 hover:bg-gray-200"
            onClick={onClose}
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <label className="block">
          <span className="text-[11px] text-gray-500">Id</span>
          <input
            aria-label="Stage id"
            className={FIELD}
            value={idDraft}
            onChange={(e) => { setIdDraft(e.target.value); setIdError(''); }}
            onBlur={() => {
              const id = idDraft.trim();
              if (!id || (id !== stage.id && allStageIds.includes(id))) {
                setIdError('Rename not applied: stage id must be nonempty and unique.');
                return;
              }
              if (id !== stage.id) patch({ id });
              else setIdDraft(id);
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur();
              if (e.key === 'Escape') { setIdDraft(stage.id); setIdError(''); }
            }}
          />
          {idError && <span role="alert" className="text-[11px] text-red-600">{idError}</span>}
        </label>
        <label className="block">
          <span className="text-[11px] text-gray-500">Job type</span>
          <select
            aria-label="Stage job type"
            className={clsx(FIELD, jobTypeWrong && 'border-yellow-400')}
            value={jobType}
            onChange={(e) => patch({ job_type: e.target.value })}
          >
            {(vocabulary?.job_types || ['research']).map((j) => (
              <option key={j} value={j}>
                {j}
              </option>
            ))}
          </select>
        </label>
      </div>

      {jobTypeWrong && (
        // Not merely a warning: this is the failure where a stage validates,
        // plans a set of tools, starts, and cannot see any of them.
        <p className="text-[11px] text-yellow-300 flex items-start gap-1">
          <AlertTriangle className="w-3 h-3 mt-0.5 flex-none" />
          <span>
            The evidence this stage requires can only be produced under{' '}
            <span className="font-medium">{forcedJobTypes.join(' or ')}</span>. Left as{' '}
            {jobType}, it will be planned with tools it cannot call.
          </span>
        </p>
      )}

      <label className="block">
        <span className="text-[11px] text-gray-500">
          Goal — what must be <em>true</em> when it is done
        </span>
        <textarea
          aria-label="Stage goal"
          rows={2}
          className={FIELD}
          value={String(stage.goal || '')}
          onChange={(e) => patch({ goal: e.target.value })}
        />
      </label>

      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-[11px] text-gray-500">
            Must produce {required.length > 0 && `(${required.length})`}
          </span>
          {totalSeconds > 0 && (
            <span className="text-[10px] text-gray-500 inline-flex items-center gap-0.5">
              <Clock className="w-3 h-3" />~{seconds(totalSeconds)}
            </span>
          )}
        </div>
        <input
          aria-label="Filter evidence types"
          className={clsx(FIELD, 'mb-1')}
          placeholder="filter evidence…"
          value={evidenceFilter}
          onChange={(e) => setEvidenceFilter(e.target.value)}
        />
        <div className="max-h-44 overflow-y-auto scrollbar-thin rounded-md border border-gray-300 divide-y divide-gray-200">
          {visibleEvidence.length === 0 && (
            <p className="text-[11px] text-gray-500 p-2">
              {vocabulary ? 'Nothing matches.' : loadingVocabulary
                ? 'Loading the evidence types…' : 'Tool catalog unavailable. Refresh tools and skills to retry.'}
            </p>
          )}
          {visibleEvidence.map((entry) => {
            const chosen = required.includes(entry.name);
            return (
              <button
                key={entry.name}
                type="button"
                onClick={() => toggleEvidence(entry.name)}
                className={clsx(
                  'w-full text-left px-2 py-1.5 flex items-start gap-2 transition-colors duration-fast',
                  chosen ? 'bg-primary-500/10' : 'hover:bg-gray-200'
                )}
              >
                <span className="w-3 flex-none pt-0.5">
                  {chosen && <Check className="w-3 h-3 text-primary-600" />}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block text-[11px] font-mono text-gray-800">
                    {entry.name}
                  </span>
                  <span className="block text-[10px] text-gray-500 truncate">
                    {entry.producers.join(', ')}
                    {entry.typical_seconds ? ` · ~${seconds(entry.typical_seconds)}` : ''}
                    {entry.job_types.length ? ` · ${entry.job_types.join('/')}` : ''}
                    {/* Perishable evidence is never inherited from an upstream
                        stage, so a looping stage cannot keep a verdict it
                        earned before its own edit. */}
                    {entry.perishable ? ' · perishable' : ''}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <StageSkillDetails required={required} evidence={byName} loading={loadingVocabulary} />

      {bindingIssues.length > 0 && <section aria-label="Stage binding issues" className="rounded border border-yellow-600 p-2 text-[11px] text-yellow-700">
        <h4 className="font-medium">Bindings need attention</h4>
        <ul className="list-disc pl-4 space-y-1">
          {bindingIssues.map((issue) => <li key={issue}>{issue}</li>)}
        </ul>
        <p className="mt-1">Bindings are preserved. Repair them under “Bind arguments to stage results” before launch.</p>
      </section>}

      <StageToolPicker
        key={stage.id}
        tools={vocabulary?.tools}
        selected={stage.tools}
        jobType={jobType}
        onChange={(tools) => patch({ tools })}
        onRemoveTool={(name) => onChange(removeStageToolConfiguration(stage, name))}
        defaults={stage.tool_defaults}
        bindings={stage.tool_bindings}
        sources={stage.depends_on || []}
        onBindingsChange={(name, values) => {
          const next = { ...(stage.tool_bindings || {}) };
          if (values) next[name] = values;
          else delete next[name];
          patch({
            tool_bindings: Object.keys(next).length ? next : undefined,
            tools: values && Array.isArray(stage.tools) && !stage.tools.includes(name)
              ? [...stage.tools, name] : stage.tools,
          });
        }}
        onDefaultsChange={(name, values) => {
          const next = { ...(stage.tool_defaults || {}) };
          if (values) next[name] = values;
          else delete next[name];
          patch({
            tool_defaults: Object.keys(next).length ? next : undefined,
            tools: values && Array.isArray(stage.tools) && !stage.tools.includes(name)
              ? [...stage.tools, name] : stage.tools,
          });
        }}
      />

      <div className="grid grid-cols-2 gap-2">
        <label className="block">
          <span className="text-[11px] text-gray-500">Depends on</span>
          <select
            aria-label="Stage dependencies"
            multiple
            size={Math.min(4, Math.max(2, allStageIds.length))}
            className={clsx(FIELD, 'h-auto')}
            value={(stage.depends_on || []) as string[]}
            onChange={(e) =>
              patch({
                depends_on: Array.from(e.target.selectedOptions).map((o) => o.value),
              })
            }
          >
            {allStageIds
              .filter((id) => id !== stage.id)
              .map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
          </select>
        </label>
        <div className="space-y-2">
          <label className="flex items-center gap-1.5 text-[11px] text-gray-600">
            <input
              type="checkbox"
              aria-label="Checkpoint"
              checked={Boolean(stage.checkpoint)}
              onChange={(e) => patch({ checkpoint: e.target.checked })}
            />
            Checkpoint — a person approves before the next stage
          </label>
          <label className="flex items-center gap-1.5 text-[11px] text-gray-600">
            <input
              type="checkbox"
              aria-label="Loop"
              checked={Boolean(stage.loop)}
              onChange={(e) =>
                patch({
                  // An unbounded loop is refused by the checker, so turning
                  // one on has to supply the bound rather than leave it out.
                  loop: e.target.checked
                    ? { max_iterations: 4, until: 'contract_satisfied' }
                    : undefined,
                })
              }
            />
            Repeat until the contract is met
          </label>
          {stage.loop && (
            <label className="block">
              <span className="text-[11px] text-gray-500">Max iterations</span>
              <input
                aria-label="Max iterations"
                type="number"
                min={1}
                className={FIELD}
                value={Number(stage.loop.max_iterations || 1)}
                onChange={(e) =>
                  patch({
                    loop: {
                      ...stage.loop,
                      max_iterations: Math.max(1, Number(e.target.value) || 1),
                    },
                  })
                }
              />
            </label>
          )}
        </div>
      </div>
    </div>
  );
};

export default StageInspector;
