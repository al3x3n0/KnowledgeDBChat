import React, { useEffect, useMemo, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { useLocation } from 'react-router-dom';

import apiClient from '../services/api';
import {
  PatchPR,
  PatchPRFromChainRequest,
  PatchPRListItem,
  PatchPRMergeResponse,
} from '../types';
import Button from '../components/common/Button';
import LoadingSpinner from '../components/common/LoadingSpinner';

interface ExperimentRun {
  proposal_id?: string;
  ran?: boolean;
  ok?: boolean;
  backend?: string;
  commands?: unknown[];
}

const PatchPRsPage: React.FC = () => {
  const location = useLocation();
  const [items, setItems] = useState<PatchPRListItem[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);

  const [selected, setSelected] = useState<PatchPR | null>(null);
  const [selectedLoading, setSelectedLoading] = useState(false);
  const [actionPending, setActionPending] = useState(false);
  const detailRequest = useRef(0);

  const [rootJobId, setRootJobId] = useState<string>('');
  const [proposalStrategy, setProposalStrategy] = useState<'best_passing' | 'latest'>('best_passing');
  const deepLinkedPatchPrId = useMemo(() => String(new URLSearchParams(location.search).get('pr') || '').trim(), [location.search]);

  const canCreate = useMemo(() => rootJobId.trim().length > 0, [rootJobId]);

  const refresh = async () => {
    setLoading(true);
    try {
      const res = await apiClient.listPatchPRs({ limit: 50, offset: 0 });
      setItems(res.items || []);
      setTotal(res.total || 0);
    } catch (e: any) {
      toast.error(e?.message || 'Failed to load Patch PRs');
    } finally {
      setLoading(false);
    }
  };

  const loadDetail = async (id: string) => {
    const request = ++detailRequest.current;
    setSelectedLoading(true);
    setSelected(null);
    try {
      const pr = await apiClient.getPatchPR(id);
      if (detailRequest.current === request) setSelected(pr);
    } catch (e: any) {
      if (detailRequest.current === request) toast.error(e?.message || 'Failed to load Patch PR');
    } finally {
      if (detailRequest.current === request) setSelectedLoading(false);
    }
  };

  const createFromChain = async () => {
    if (!canCreate || actionPending) return;
    setActionPending(true);
    try {
      const payload: PatchPRFromChainRequest = {
        root_job_id: rootJobId.trim(),
        proposal_strategy: proposalStrategy,
        open_after_create: true,
      };
      const pr = await apiClient.createPatchPRFromChain(payload);
      toast.success('Created Patch PR');
      await refresh();
      await loadDetail(pr.id);
    } catch (e: any) {
      toast.error(e?.message || 'Failed to create Patch PR');
    } finally {
      setActionPending(false);
    }
  };

  const approveSelected = async () => {
    if (!selected?.diff_sha256 || actionPending) return;
    const request = detailRequest.current;
    setActionPending(true);
    try {
      const pr = await apiClient.approvePatchPR(selected.id, {
        diff_sha256: selected.diff_sha256,
      });
      if (detailRequest.current === request) setSelected(pr);
      toast.success('Approved');
      await refresh();
    } catch (e: any) {
      toast.error(e?.message || 'Approve failed');
      if (detailRequest.current === request) await loadDetail(selected.id);
    } finally {
      setActionPending(false);
    }
  };

  const mergeSelected = async (dryRun: boolean) => {
    if (!selected || actionPending) return;
    const request = detailRequest.current;
    setActionPending(true);
    try {
      const res: PatchPRMergeResponse = await apiClient.mergePatchPR(selected.id, {
        dry_run: dryRun,
        diff_sha256: selected.diff_sha256 || undefined,
      });
      if (res.ok) {
        toast.success(dryRun ? 'Patch can be applied' : 'Applied to Library');
      } else {
        toast.error('Patch has conflicts');
      }
      if (detailRequest.current === request) await loadDetail(selected.id);
      await refresh();
    } catch (e: any) {
      toast.error(e?.message || 'Patch application failed');
    } finally {
      setActionPending(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (!deepLinkedPatchPrId) return;
    loadDetail(deepLinkedPatchPrId);
  }, [deepLinkedPatchPrId]);

  const experiments: ExperimentRun[] = Array.isArray(selected?.checks?.experiment_runs)
    ? selected.checks.experiment_runs.filter(
        (run: ExperimentRun) => run.proposal_id === selected?.selected_proposal_id
      )
    : [];
  const patchCheck = [selected?.checks?.merge, selected?.checks?.dry_run]
    .find((check) => check && check.diff_sha256 === selected?.diff_sha256);
  const canReview = Boolean(selected?.diff_sha256
    && !['merged', 'rejected'].includes(selected.status));
  const canApply = canReview && Boolean(selected?.source_id);

  const downloadPatch = () => {
    if (!selected?.diff_unified) return;
    const url = URL.createObjectURL(new Blob([selected.diff_unified], { type: 'text/x-diff' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `patch-${selected.id.slice(0, 8)}.diff`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Patch PRs</h1>
          <p className="text-sm text-gray-700">Review coding-agent changes before applying them to the Library</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={refresh} disabled={loading}>
            Refresh
          </Button>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="min-w-0 space-y-4">
          <div className="bg-gray-100/60 border border-gray-300 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-800">Create from chain</div>
            </div>
            <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="md:col-span-2">
                <label className="text-xs text-gray-700">root_job_id</label>
                <input
                  className="mt-1 w-full bg-gray-50 border border-gray-300 rounded px-3 py-2 text-sm text-gray-900"
                  value={rootJobId}
                  onChange={(e) => setRootJobId(e.target.value)}
                  placeholder="UUID of chain root job"
                />
              </div>
              <div>
                <label className="text-xs text-gray-700">strategy</label>
                <select
                  className="mt-1 w-full bg-gray-50 border border-gray-300 rounded px-3 py-2 text-sm text-gray-900"
                  value={proposalStrategy}
                  onChange={(e) => setProposalStrategy(e.target.value as any)}
                >
                  <option value="best_passing">best_passing</option>
                  <option value="latest">latest</option>
                </select>
              </div>
            </div>
            <div className="mt-3">
              <Button onClick={createFromChain} disabled={!canCreate || actionPending}>
                Create Patch PR
              </Button>
            </div>
          </div>

          <div className="bg-gray-100/60 border border-gray-300 rounded-lg overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-300 flex items-center justify-between">
              <div className="text-sm text-gray-800">Recent ({total})</div>
              {loading && <LoadingSpinner size="sm" />}
            </div>
            <div className="divide-y divide-gray-300">
              {(items || []).map((p) => (
                <button
                  key={p.id}
                  className="w-full text-left px-4 py-3 hover:bg-gray-200/40"
                  onClick={() => loadDetail(p.id)}
                  disabled={actionPending}
                  aria-current={selected?.id === p.id ? 'true' : undefined}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-sm text-gray-900 truncate">{p.title}</div>
                    <div className="text-xs text-gray-700">{p.status}</div>
                  </div>
                  <div className="mt-1 text-xs text-gray-600 truncate">{p.id}</div>
                </button>
              ))}
              {!loading && (items || []).length === 0 && (
                <div className="px-4 py-6 text-sm text-gray-700">No Patch PRs yet.</div>
              )}
            </div>
          </div>
        </div>

        <div className="min-w-0 lg:col-span-2 bg-gray-100/60 border border-gray-300 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-800">Details</div>
            {selectedLoading && <LoadingSpinner size="sm" />}
          </div>

          {!selected && <div className="mt-3 text-sm text-gray-700">Select a Patch PR to view.</div>}

          {selected && (
            <div className="mt-3 space-y-3">
              <div>
                <div className="text-sm text-gray-900 font-semibold">{selected.title}</div>
                <div className="text-xs text-gray-700">
                  {selected.status === 'approved' && !selected.approval_valid ? 'Needs approval' : selected.status}
                </div>
                <div className="mt-1 text-xs text-gray-600 break-all">{selected.id}</div>
              </div>

              <section aria-label="Selected patch diff">
                <h2 className="text-sm font-semibold text-gray-900 mb-1">File changes</h2>
                {selected.diff_unified ? (
                  <pre className="max-h-[28rem] overflow-auto rounded border border-gray-300 bg-gray-50 py-2 text-xs">
                    {selected.diff_unified.split('\n').map((line, index) => (
                      <span
                        key={index}
                        className={`block min-w-max px-3 ${line.startsWith('+')
                          ? 'bg-green-50 text-green-900'
                          : line.startsWith('-')
                            ? 'bg-red-50 text-red-900'
                            : 'text-gray-800'}`}
                      >
                        {line || ' '}
                      </span>
                    ))}
                  </pre>
                ) : <p className="text-sm text-gray-700">No patch diff available.</p>}
              </section>

              <section aria-label="Patch verification">
                <h2 className="text-sm font-semibold text-gray-900 mb-1">Verification</h2>
                {experiments.length ? (
                  <ul className="space-y-1 text-sm text-gray-700">
                    {experiments.map((run, index) => (
                      <li key={index}>
                        {run.ran === false ? 'Tests did not run' : run.ok === true
                          ? 'Passed' : run.ok === false ? 'Failed' : 'Result unavailable'}
                        {run.backend && ` · ${run.backend}`}
                        {Array.isArray(run.commands) && run.commands.map((command: unknown, commandIndex: number) =>
                          typeof command === 'string' && (
                            <code className="block break-all text-xs" key={commandIndex}>{command}</code>
                          )
                        )}
                      </li>
                    ))}
                  </ul>
                ) : <p className="text-sm text-gray-700">No experiment results for this diff.</p>}
                {(patchCheck?.errors || []).map((error: {path: string; error: string}, index: number) => (
                  <p key={index} className="text-sm text-red-800">{error.path} — {error.error}</p>
                ))}
              </section>

              <div className="flex flex-wrap gap-2">
                <Button
                  variant="secondary"
                  onClick={approveSelected}
                  disabled={!canReview || actionPending || selected.approval_valid}
                >
                  {selected.approval_valid ? 'Approved' : 'Approve diff'}
                </Button>
                <Button variant="secondary" onClick={() => mergeSelected(true)} disabled={!canApply || actionPending}>
                  Check applicability
                </Button>
                <Button onClick={() => mergeSelected(false)} disabled={!canApply || !selected.approval_valid || actionPending}>
                  {selected.status === 'merged' ? 'Applied to Library' : 'Apply to Library'}
                </Button>
                <Button variant="secondary" onClick={downloadPatch} disabled={!selected.diff_unified}>
                  Download patch
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PatchPRsPage;
