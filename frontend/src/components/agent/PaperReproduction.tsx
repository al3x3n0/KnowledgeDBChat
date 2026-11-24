import clsx from 'clsx';
import React from 'react';

import type { AgentJobEvidenceValue, AgentJobReproductionEvidence } from '../../types';

const valueFormat = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 8 });
const percentFormat = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 2 });

const verdictLabels = {
  reproduced: { label: 'Reproduced', tone: 'bg-primary-500/10 text-primary-700' },
  not_reproduced: { label: 'Outside tolerance', tone: 'bg-amber-500/10 text-amber-700 dark:text-amber-300' },
  incomparable: { label: 'Incomparable', tone: 'bg-amber-500/10 text-amber-700 dark:text-amber-300' },
};

const Result: React.FC<{ label: string; value?: number | null; unit: string }> = ({ label, value, unit }) => (
  <div className="min-w-0 rounded border border-gray-200 bg-white p-2">
    <dt className="text-xs font-medium text-gray-600">{label}</dt>
    <dd className="mt-0.5 break-words text-sm font-semibold tabular-nums text-gray-900">
      {value == null ? 'Unrecorded' : valueFormat.format(value)}
      {value != null && unit && <span className="ml-1 text-xs font-medium text-gray-700">{unit}</span>}
    </dd>
    {value != null && !unit && <dd className="text-xs text-gray-600">Unit unrecorded</dd>}
  </div>
);

const Conditions: React.FC<{ title: string; entries: AgentJobEvidenceValue[] }> = ({ title, entries }) => (
  <div className="min-w-0">
    <p className="font-medium text-gray-900">{title}</p>
    {entries.length ? (
      <dl className="mt-1 space-y-1">
        {entries.map(({ label, value }, index) => (
          <div key={`${index}-${label}`}>
            <dt className="break-words capitalize text-gray-600">{label.replace(/_/g, ' ')}</dt>
            <dd className="break-words text-gray-900">{value}</dd>
          </div>
        ))}
      </dl>
    ) : <p className="mt-1 text-gray-600">Unrecorded</p>}
  </div>
);

const PaperReproduction: React.FC<{
  result: AgentJobReproductionEvidence;
  disputed: boolean;
  onOpenDocument?: (id: string) => void;
}> = ({ result, disputed, onOpenDocument }) => {
  const verdict = verdictLabels[result.verdict];
  const documentId = result.claim_document_id;
  const showPercentages = result.verdict !== 'incomparable' &&
    (result.relative_error != null || result.tolerance != null);

  return (
    <div className="space-y-2.5 text-xs text-gray-700">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className={clsx('rounded-full px-2 py-0.5 font-semibold', disputed ? 'bg-gray-200 text-gray-700' : verdict.tone)}>{verdict.label}</span>
        {result.metric && <span className="break-words font-medium text-gray-800">{result.metric}</span>}
      </div>
      <dl className="grid grid-cols-2 gap-2">
        <Result label="Paper claim" value={result.claimed_value} unit={result.claimed_unit} />
        <Result label="Run measurement" value={result.measured_value} unit={result.measured_unit} />
      </dl>
      {showPercentages && (
        <dl className="flex flex-wrap gap-x-4 gap-y-1">
          {result.relative_error != null && <div className="flex gap-1">
            <dt>Relative error</dt><dd className="font-medium tabular-nums">{percentFormat.format(result.relative_error)}</dd>
          </div>}
          {result.tolerance != null && <div className="flex gap-1">
            <dt>Tolerance</dt><dd className="font-medium tabular-nums">{percentFormat.format(result.tolerance)}</dd>
          </div>}
        </dl>
      )}
      <div className="space-y-1 break-words">
        <div className="flex flex-wrap items-baseline gap-x-2">
          {documentId && onOpenDocument && (
            <button type="button" className="font-medium text-primary-700 hover:underline" onClick={() => onOpenDocument(documentId)}>
              Open source paper
            </button>
          )}
          {result.claim_location && <span>{result.claim_location}</span>}
          {!result.claim_document_id && !result.claim_location && <span className="text-gray-600">Paper source unrecorded</span>}
        </div>
        <p><span className="font-medium">Measurement source</span> · {result.measurement_source || 'Unrecorded'}</p>
      </div>
      {result.blockers.length > 0 && (
        <div className="break-words rounded border border-amber-400/40 bg-amber-500/10 p-2">
          <p className="font-medium text-gray-900">Comparison blockers</p>
          <ul className="mt-1 list-disc space-y-1 pl-4">
            {result.blockers.map((line, index) => <li key={index}>{line}</li>)}
          </ul>
        </div>
      )}
      {result.caveats.length > 0 && (
        <div className="break-words">
          <p className="font-medium text-gray-900">Caveats</p>
          <ul className="mt-1 list-disc space-y-1 pl-4">
            {result.caveats.map((line, index) => <li key={index}>{line}</li>)}
          </ul>
        </div>
      )}
      {result.verdict === 'incomparable' && result.blockers.length === 0 && result.summary && (
        <p className="break-words">{result.summary}</p>
      )}
      {(result.claimed_conditions.length > 0 || result.measured_conditions.length > 0) && (
        <details className="rounded border border-gray-200 p-2">
          <summary className="cursor-pointer font-medium text-primary-700">Recorded conditions</summary>
          <div className="mt-2 grid grid-cols-2 gap-3">
            <Conditions title="Paper" entries={result.claimed_conditions} />
            <Conditions title="Run" entries={result.measured_conditions} />
          </div>
        </details>
      )}
    </div>
  );
};

export default PaperReproduction;
