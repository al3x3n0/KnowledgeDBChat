import React, { useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';

import apiClient from '../services/api';
import {
  PatchPR,
  PatchPRFromChainRequest,
  PatchPRListItem,
  PatchPRMergeResponse,
} from '../types';
import Button from '../components/common/Button';
import LoadingSpinner from '../components/common/LoadingSpinner';

const PatchPRsPage: React.FC = () => {
  const [items, setItems] = useState<PatchPRListItem[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);

  const [selected, setSelected] = useState<PatchPR | null>(null);
  const [selectedLoading, setSelectedLoading] = useState<boolean>(false);

  const [rootJobId, setRootJobId] = useState<string>('');
  const [proposalStrategy, setProposalStrategy] = useState<'best_passing' | 'latest'>('best_passing');

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
    setSelectedLoading(true);
    try {
      const pr = await apiClient.getPatchPR(id);
      setSelected(pr);
    } catch (e: any) {
      toast.error(e?.message || 'Failed to load Patch PR');
    } finally {
      setSelectedLoading(false);
    }
  };

  const createFromChain = async () => {
    if (!canCreate) return;
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
    }
  };

  const approveSelected = async () => {
    if (!selected) return;
    try {
      const pr = await apiClient.approvePatchPR(selected.id, {});
      setSelected(pr);
      toast.success('Approved');
      await refresh();
    } catch (e: any) {
      toast.error(e?.message || 'Approve failed');
    }
  };

  const mergeSelected = async (dryRun: boolean) => {
    if (!selected) return;
    try {
      const res: PatchPRMergeResponse = await apiClient.mergePatchPR(selected.id, {
        dry_run: dryRun,
        require_approved: true,
      });
      if (res.ok) {
        toast.success(dryRun ? 'Dry-run OK' : 'Merged');
      } else {
        toast.error(dryRun ? 'Dry-run has errors' : 'Merge has errors');
      }
      await loadDetail(selected.id);
      await refresh();
    } catch (e: any) {
      toast.error(e?.message || 'Merge failed');
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Patch PRs</h1>
          <p className="text-sm text-gray-700">PR-style flow for Code Patch Proposals</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={refresh} disabled={loading}>
            Refresh
          </Button>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 space-y-4">
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
              <Button onClick={createFromChain} disabled={!canCreate}>
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

        <div className="bg-gray-100/60 border border-gray-300 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-800">Details</div>
            {selectedLoading && <LoadingSpinner size="sm" />}
          </div>

          {!selected && <div className="mt-3 text-sm text-gray-700">Select a Patch PR to view.</div>}

          {selected && (
            <div className="mt-3 space-y-3">
              <div>
                <div className="text-sm text-gray-900 font-semibold">{selected.title}</div>
                <div className="text-xs text-gray-700">{selected.status}</div>
                <div className="mt-1 text-xs text-gray-600 break-all">{selected.id}</div>
              </div>

              <div className="flex flex-wrap gap-2">
                <Button variant="secondary" onClick={approveSelected}>
                  Approve
                </Button>
                <Button variant="secondary" onClick={() => mergeSelected(true)}>
                  Dry-run merge
                </Button>
                <Button onClick={() => mergeSelected(false)}>
                  Merge
                </Button>
              </div>

              <div>
                <div className="text-xs text-gray-700">selected_proposal_id</div>
                <div className="mt-1 text-xs text-gray-900 break-all">{selected.selected_proposal_id || '—'}</div>
              </div>

              <div>
                <div className="text-xs text-gray-700">proposal_ids</div>
                <div className="mt-1 text-xs text-gray-900 break-all">{(selected.proposal_ids || []).join(', ') || '—'}</div>
              </div>

              <details className="mt-2">
                <summary className="text-xs text-gray-800 cursor-pointer">raw JSON</summary>
                <pre className="mt-2 text-xs text-gray-900 whitespace-pre-wrap break-words">
                  {JSON.stringify(selected, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PatchPRsPage;
