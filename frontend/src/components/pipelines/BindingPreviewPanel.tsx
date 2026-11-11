import React, { useEffect, useRef, useState } from 'react';

import { apiClient } from '../../services/api';
import type { PipelineBindingPreview } from '../../types';
import type { PipelineStageSpec } from './PipelineGraph';

interface Props {
  rootJobId?: string | null;
  stage: PipelineStageSpec;
  spec: Record<string, unknown>;
  catalogRevision?: number;
  catalogRefreshing?: boolean;
  onOpenJob?: (id: string) => void;
}

export default function BindingPreviewPanel({ rootJobId, stage, spec, onOpenJob, catalogRevision = 0, catalogRefreshing = false }: Props) {
  const signature = JSON.stringify([rootJobId, stage.id, spec, catalogRevision, catalogRefreshing]);
  const [data, setData] = useState<{ signature: string; result: PipelineBindingPreview } | null>(null);
  const [error, setError] = useState<{ signature: string; message: string } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const request = useRef(0);
  useEffect(() => {
    request.current += 1;
    setData(null);
    setError(null);
    setBusy(null);
  }, [signature]);
  useEffect(() => () => { request.current += 1; }, []);
  const preview = async () => {
    if (!rootJobId || catalogRefreshing) return;
    const id = ++request.current;
    setBusy(signature);
    setError(null);
    setData(null);
    try {
      const result = await apiClient.previewPipelineBindings(rootJobId, spec, stage.id);
      if (id === request.current) setData({ signature, result });
    } catch (caught: any) {
      const detail = caught?.response?.data?.detail;
      if (id === request.current) setError({ signature, message: typeof detail === 'string' ? detail : 'Could not preview bindings.' });
    } finally {
      if (id === request.current) setBusy(null);
    }
  };
  const result = data?.signature === signature ? data.result : null;
  if (!Object.keys(stage.tool_bindings || {}).length) return null;

  return <section className="rounded-lg border border-gray-300 bg-gray-100 p-3 space-y-2" aria-label="Binding preview">
    <h3 className="section-heading mb-0">Binding preview</h3>
    <p className="text-xs text-gray-500">Inspect applied bindings against the latest recorded dependency results. Runtime uses snapshots taken when each stage is created; future runs may produce different values.</p>
    {!rootJobId && <p className="text-xs text-gray-500">Run this saved pipeline to preview recorded results.</p>}
    {catalogRefreshing && <p className="text-xs text-gray-500">Waiting for the tool catalog refresh.</p>}
    <button type="button" className="text-xs underline text-primary-700 disabled:opacity-50" disabled={!rootJobId || catalogRefreshing || busy === signature} onClick={preview}>
      {busy === signature ? 'Resolving bindings…' : 'Preview applied bindings'}
    </button>
    {error?.signature === signature && <p role="alert" className="text-xs text-red-600">{error.message}</p>}
    {result && <div className="space-y-2">
      {!result.items.length && <p className="text-xs text-gray-500">No argument bindings to preview.</p>}
      {result.items.map((item) => <article key={`${item.tool}:${item.argument}`} className="rounded border border-gray-300 p-2 text-xs space-y-1">
        <h4 className="font-mono break-all text-gray-800">{item.tool}.{item.argument}</h4>
        <p className="text-gray-500 break-all">{item.source_stage} → {item.path || '(whole result)'}</p>
        {item.source_job_id && <p className="text-gray-500">
          Source status: {item.source_status} · {item.source_contract_satisfied ? 'contract met' : 'contract unmet'}{' '}
          {onOpenJob && <button type="button" className="underline" onClick={() => onOpenJob(item.source_job_id!)}>Open source job</button>}
        </p>}
        {item.status === 'resolved' ? <>
          <p className="text-gray-600">Resolved value · {item.value_type}</p>
          <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all rounded bg-gray-50 p-2 text-gray-800">{item.value_json}</pre>
          {item.truncated && <p className="text-gray-500">Value preview truncated.</p>}
        </> : <p className="text-red-600">{item.error || 'Unable to resolve this binding.'}</p>}
      </article>)}
      {result.omitted > 0 && <p className="text-xs text-gray-500">{result.omitted} additional bindings omitted from this preview.</p>}
    </div>}
  </section>;
}
