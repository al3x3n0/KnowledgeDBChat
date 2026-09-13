/**
 * What a run found, and whether it is what was asked for.
 *
 * This replaces `Findings: 4` — a number that said how many findings existed
 * and nothing about whether any of them was worth believing. For a research
 * pipeline that is the difference between a run you can supervise and one you
 * can only watch finish.
 *
 * It is organised by REQUIREMENT rather than by finding, because the question
 * is never "how many findings" but "is the thing I asked for actually there".
 * A requirement with nothing under it is the most important row on the screen.
 *
 * Three distinctions are drawn that a count destroys, and each one is a reason
 * to disbelieve a number that otherwise looks fine:
 *
 *   - **a spread.** A single trial reports no dispersion, and a measurement
 *     without one is a value, not a result. Shown per finding, and called out
 *     on the requirement when the contract asked for it specifically.
 *   - **the machine it was taken on.** A wall-clock number measured on a
 *     saturated host is not a measurement. The producing tool records this;
 *     nothing showed it.
 *   - **perishability.** Evidence a later change invalidates reads very
 *     differently from evidence that keeps.
 *
 * Read-only, deliberately. The contract's own verdict is carried through and
 * never recomputed here: this shows what that verdict was reached from.
 */

import clsx from 'clsx';
import {
  AlertTriangle,
  Beaker,
  CheckCircle2,
  CircleDashed,
  Clock3,
  HelpCircle,
  ThumbsDown,
  Undo2,
  X,
} from 'lucide-react';
import React, { useCallback, useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import { useQuery } from 'react-query';

import { apiClient } from '../../services/api';
import type {
  AgentJob,
  AgentJobEvidence,
  AgentJobEvidenceItem,
} from '../../types';

export interface EvidenceSectionProps {
  job: AgentJob;
}

/** Environments the producing tool distinguishes, and how much they should
 *  worry a reader. `busy` is a caveat; `saturated` means the number is not a
 *  measurement at all. */
const ENVIRONMENT_CLASS: Record<string, string> = {
  quiet: 'border-primary-500/50 bg-primary-500/10 text-primary-700',
  busy: 'border-amber-300 bg-amber-50 text-amber-700',
  saturated: 'border-red-300 bg-red-50 text-red-700',
};

const EvidenceCard: React.FC<{
  item: AgentJobEvidenceItem;
  onDispute: (index: number, reason: string) => Promise<void>;
  onWithdraw: (index: number) => Promise<void>;
}> = ({ item, onDispute, onWithdraw }) => {
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!reason.trim()) return;
    setBusy(true);
    try {
      await onDispute(item.index, reason.trim());
      setOpen(false);
      setReason('');
    } finally {
      setBusy(false);
    }
  };

  return (
  <div
    className={clsx(
      'rounded-md border px-2.5 py-2',
      // A rejected result is still shown in full — it happened, and hiding it
      // would lose the run's own account. It just must not look believed.
      item.disputed
        ? 'border-amber-300 bg-amber-50/40'
        : 'border-gray-300 bg-gray-50'
    )}
  >
    <div className="flex flex-wrap items-center gap-1.5 mb-1">
      <span className="text-[11px] font-mono text-gray-800">{item.type}</span>
      {item.measurement_environment && (
        <span
          className={clsx(
            'text-[10px] px-1.5 py-0.5 rounded-full border',
            ENVIRONMENT_CLASS[item.measurement_environment] ||
              'border-gray-300 text-gray-600'
          )}
          title="The load on the machine while this was measured. A wall-clock number taken on a saturated host is not a measurement."
        >
          {item.measurement_environment}
        </span>
      )}
      {!item.has_uncertainty && (
        <span
          className="text-[10px] px-1.5 py-0.5 rounded-full border border-gray-300 text-gray-500 inline-flex items-center gap-0.5"
          title="No spread reported. A single trial says nothing about dispersion, so this is a value rather than a result."
        >
          <HelpCircle className="w-2.5 h-2.5" />
          no spread
        </span>
      )}
      {item.perishable && (
        <span
          className="text-[10px] px-1.5 py-0.5 rounded-full border border-gray-300 text-gray-500"
          title="Evidence a later change invalidates. Never inherited from an upstream stage, so a looping stage cannot keep a verdict it earned before its own edit."
        >
          perishable
        </span>
      )}
      {item.disputed && (
        <span className="text-[10px] px-1.5 py-0.5 rounded-full border border-amber-400 bg-amber-100 text-amber-800">
          rejected
        </span>
      )}
      <span className="ml-auto flex items-center gap-1">
        {item.disputed ? (
          <button
            type="button"
            aria-label={`Withdraw the rejection of finding ${item.index}`}
            title="Take the rejection back — a measurement re-taken and found good"
            className="p-0.5 rounded text-gray-500 hover:text-primary-700 hover:bg-gray-200"
            onClick={() => onWithdraw(item.index)}
          >
            <Undo2 className="w-3 h-3" />
          </button>
        ) : (
          <button
            type="button"
            aria-label={`Reject finding ${item.index}`}
            title="Reject this result. Advisory: no verdict changes, but a restart of this stage will be told why."
            className="p-0.5 rounded text-gray-500 hover:text-amber-700 hover:bg-gray-200"
            onClick={() => setOpen((v) => !v)}
          >
            <ThumbsDown className="w-3 h-3" />
          </button>
        )}
      </span>
    </div>

    {item.title && <p className="text-xs text-gray-700 mb-1">{item.title}</p>}

    {item.values.length > 0 && (
      <dl className="flex flex-wrap gap-x-3 gap-y-0.5">
        {item.values.map((value) => (
          <div key={value.label} className="flex items-baseline gap-1">
            <dt className="text-[10px] text-gray-500">{value.label}</dt>
            <dd className="text-[11px] font-mono text-gray-800">{value.value}</dd>
          </div>
        ))}
      </dl>
    )}

    {item.warning && (
      <p className="text-[11px] text-amber-700 mt-1 flex items-start gap-1">
        <AlertTriangle className="w-3 h-3 mt-0.5 flex-none" />
        {item.warning}
      </p>
    )}

    {item.disputed && item.dispute_reason && (
      <p className="text-[11px] text-amber-800 mt-1.5 pl-2 border-l-2 border-amber-300">
        Rejected: {item.dispute_reason}
      </p>
    )}

    {open && (
      <div className="mt-2 space-y-1.5">
        <label
          className="block text-[11px] text-gray-600"
          htmlFor={`dispute-${item.index}`}
        >
          Why is this result not to be believed? A restart of this stage begins
          with what you write here.
        </label>
        <textarea
          id={`dispute-${item.index}`}
          rows={2}
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="The host was saturated while this ran — retake it on a quiet machine"
          className="w-full px-2 py-1 text-xs rounded-md bg-gray-50 border border-gray-300
            focus:outline-none focus:border-primary-600"
        />
        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={busy || !reason.trim()}
            onClick={submit}
            className="inline-flex items-center gap-1 px-2 py-1 text-[11px] rounded-md
              border border-amber-400 bg-amber-50 text-amber-800
              hover:bg-amber-100 disabled:opacity-40"
          >
            <ThumbsDown className="w-3 h-3" />
            Reject
          </button>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="inline-flex items-center gap-1 px-2 py-1 text-[11px] text-gray-600 hover:text-gray-900"
          >
            <X className="w-3 h-3" />
            Cancel
          </button>
        </div>
      </div>
    )}
  </div>
  );
};

export const EvidenceSection: React.FC<EvidenceSectionProps> = ({ job }) => {
  const [showUnrequested, setShowUnrequested] = useState(false);

  const { data, isLoading, error, refetch } = useQuery<AgentJobEvidence>(
    ['job-evidence', job.id],
    () => apiClient.getJobEvidence(String(job.id)),
    {
      enabled: Boolean(job.id),
      // Rendered for every run, and a run this user cannot see 404s. Retrying
      // that is noise.
      retry: false,
      staleTime: 30_000,
    }
  );

  const handleDispute = useCallback(
    async (index: number, reason: string) => {
      try {
        await apiClient.disputeJobEvidence(String(job.id), index, reason);
        // Said out loud because it is the surprising half of the design: the
        // result stays, the verdict stands, and the reason travels.
        toast.success('Rejected — a restart of this stage will be told why');
        refetch();
      } catch (err: any) {
        toast.error(err?.response?.data?.detail || 'Could not record the rejection');
      }
    },
    [job.id, refetch]
  );

  const handleWithdraw = useCallback(
    async (index: number) => {
      try {
        await apiClient.withdrawJobEvidenceDispute(String(job.id), index);
        toast.success('Rejection withdrawn');
        refetch();
      } catch (err: any) {
        toast.error(err?.response?.data?.detail || 'Could not withdraw it');
      }
    },
    [job.id, refetch]
  );

  const byIndex = useMemo(() => {
    const map: Record<number, AgentJobEvidenceItem> = {};
    (data?.evidence || []).forEach((item) => {
      map[item.index] = item;
    });
    return map;
  }, [data]);

  if (error || isLoading || !data) return null;
  if (!data.evidence.length && !data.requirements.length) return null;

  const unrequested = data.unrequested
    .map((index) => byIndex[index])
    .filter(Boolean);

  return (
    <div className="mb-4" data-testid="evidence-section">
      <div className="flex items-center gap-2 mb-2">
        <h3 className="text-sm font-medium text-gray-700 flex items-center gap-1.5">
          <Beaker className="w-4 h-4" />
          Evidence
        </h3>
        <span className="text-xs text-gray-500">
          {data.evidence.length} finding{data.evidence.length === 1 ? '' : 's'}
        </span>
        {data.contract_enabled && (
          <span
            className={clsx(
              'text-[11px] px-2 py-0.5 rounded-full border font-medium',
              data.contract_satisfied
                ? 'border-primary-500/60 bg-primary-500/10 text-primary-700'
                : 'border-amber-300 bg-amber-50 text-amber-700'
            )}
          >
            {data.contract_satisfied ? 'Contract met' : 'Contract not met'}
          </span>
        )}
        {data.disputed_count > 0 && (
          // The contract may well be met. A run resting on rejected evidence
          // still must not read as clean.
          <span className="text-[11px] px-2 py-0.5 rounded-full border border-amber-400 bg-amber-100 text-amber-800 font-medium">
            {data.disputed_count} rejected
          </span>
        )}
      </div>

      <div className="bg-gray-100 border border-gray-300 rounded-lg p-3 space-y-3">
        {data.requirements.map((requirement) => {
          const found = requirement.satisfied_by
            .map((index) => byIndex[index])
            .filter(Boolean);
          const shortOfSpread = requirement.missing_uncertainty.length > 0;
          return (
            <div key={requirement.finding_type}>
              <div className="flex flex-wrap items-center gap-1.5 mb-1">
                {requirement.satisfied ? (
                  shortOfSpread ? (
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                  ) : (
                    <CheckCircle2 className="w-3.5 h-3.5 text-primary-600" />
                  )
                ) : (
                  <CircleDashed className="w-3.5 h-3.5 text-gray-400" />
                )}
                <span className="text-xs font-mono text-gray-800">
                  {requirement.finding_type}
                </span>
                {requirement.uncertainty_required && (
                  <span
                    className="text-[10px] px-1.5 py-0.5 rounded-full border border-gray-300 text-gray-600"
                    title="The contract requires this evidence to carry a spread."
                  >
                    spread required
                  </span>
                )}
              </div>

              {!requirement.satisfied && (
                // The most important row on the screen: the thing that was
                // asked for is not there.
                <p className="text-[11px] text-gray-500 pl-5">
                  Nothing of this type was produced.
                </p>
              )}

              {shortOfSpread && (
                <p className="text-[11px] text-amber-700 pl-5 mb-1">
                  {requirement.missing_uncertainty.length} of {found.length} arrived
                  without the spread the contract requires.
                </p>
              )}

              <div className="pl-5 space-y-1.5">
                {found.map((item) => (
                  <EvidenceCard
                    key={item.index}
                    item={item}
                    onDispute={handleDispute}
                    onWithdraw={handleWithdraw}
                  />
                ))}
              </div>
            </div>
          );
        })}

        {data.missing.length > 0 && (
          <div className="rounded-md border border-amber-200 bg-amber-50 p-2">
            <div className="text-[11px] font-medium text-amber-800 mb-1">
              Still outstanding
            </div>
            <ul className="text-[11px] text-amber-700 space-y-0.5">
              {data.missing.map((entry, index) => (
                <li key={`${index}-${entry.slice(0, 24)}`}>- {entry}</li>
              ))}
            </ul>
          </div>
        )}

        {data.unsettled_predictions.length > 0 && (
          // A claim the run made and never checked. Shown whether or not the
          // contract asked for `predictions_measured`, because an unsettled
          // prediction is a claim nobody settled either way.
          <p className="text-[11px] text-amber-700 flex items-start gap-1">
            <Clock3 className="w-3 h-3 mt-0.5 flex-none" />
            {data.unsettled_predictions.length} prediction
            {data.unsettled_predictions.length === 1 ? '' : 's'} recorded and never
            settled by a measurement.
          </p>
        )}

        {unrequested.length > 0 && (
          <div>
            <button
              type="button"
              className="text-[11px] text-gray-500 hover:text-gray-800"
              onClick={() => setShowUnrequested((open) => !open)}
            >
              {showUnrequested ? 'Hide' : 'Show'} {unrequested.length} finding
              {unrequested.length === 1 ? '' : 's'} nothing asked for
              {data.unrequested_total > unrequested.length &&
                ` (of ${data.unrequested_total})`}
            </button>
            {showUnrequested && (
              <div className="space-y-1.5 mt-1.5">
                {unrequested.map((item) => (
                  <EvidenceCard
                    key={item.index}
                    item={item}
                    onDispute={handleDispute}
                    onWithdraw={handleWithdraw}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default EvidenceSection;
