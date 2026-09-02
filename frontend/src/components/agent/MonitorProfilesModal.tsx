/**
 * Saved monitor profiles: list, edit, save.
 *
 * Module scope, like its siblings — declared in the page's render body it was
 * remounted on every page render, resetting the form below.
 */

import { RefreshCw, XCircle } from 'lucide-react';
import React, { useEffect, useMemo, useState } from 'react';

import Button from '../common/Button';
import LoadingSpinner from '../common/LoadingSpinner';

interface MonitorProfilesModalProps {
  onClose: () => void;
  monitorProfiles: any[] | undefined;
  monitorProfilesLoading: boolean;
  refetchMonitorProfiles: () => void;
  upsertMonitorProfileMutation: any;
}

export const MonitorProfilesModal: React.FC<MonitorProfilesModalProps> = ({
  onClose,
  monitorProfiles,
  monitorProfilesLoading,
  refetchMonitorProfiles,
  upsertMonitorProfileMutation,
}) => {

  const profiles = useMemo(() => (monitorProfiles || []) as any[], [monitorProfiles]);
  const [selectedCustomer, setSelectedCustomer] = useState<string>('');
  const selected = useMemo(() => {
    const key = (selectedCustomer || '').trim();
    if (!key) {
      return profiles.find((p: any) => !p?.customer) || null;
    }
    return profiles.find((p: any) => String(p?.customer || '') === key) || null;
  }, [profiles, selectedCustomer]);

  const [mutedTokensText, setMutedTokensText] = useState<string>('');
  const [mutedPatternsText, setMutedPatternsText] = useState<string>('');
  const [notes, setNotes] = useState<string>('');

  useEffect(() => {
    const mt = Array.isArray(selected?.muted_tokens) ? selected.muted_tokens : [];
    const mp = Array.isArray(selected?.muted_patterns) ? selected.muted_patterns : [];
    setMutedTokensText(mt.join('\n'));
    setMutedPatternsText(mp.join('\n'));
    setNotes(String(selected?.notes || ''));
  }, [selected?.id, selected?.muted_tokens, selected?.muted_patterns, selected?.notes]);

  const tokenScores = (selected?.token_scores || {}) as Record<string, number>;
  const recommendationScores = (selected?.recommendation_scores || {}) as Record<string, number>;
  const sourceTypeScores = (selected?.source_type_scores || {}) as Record<string, number>;
  const outcomeCounters = (selected?.outcome_counters || {}) as Record<string, number>;
  const topPositive = Object.entries(tokenScores)
    .filter(([, v]) => typeof v === 'number' && v > 0)
    .sort((a, b) => (b[1] as number) - (a[1] as number))
    .slice(0, 8);
  const topNegative = Object.entries(tokenScores)
    .filter(([, v]) => typeof v === 'number' && v < 0)
    .sort((a, b) => (a[1] as number) - (b[1] as number))
    .slice(0, 8);
  const topRecommendations = Object.entries(recommendationScores)
    .filter(([, v]) => typeof v === 'number')
    .sort((a, b) => Math.abs(Number(b[1] || 0)) - Math.abs(Number(a[1] || 0)))
    .slice(0, 6);
  const topSourceTypes = Object.entries(sourceTypeScores)
    .filter(([, v]) => typeof v === 'number')
    .sort((a, b) => Math.abs(Number(b[1] || 0)) - Math.abs(Number(a[1] || 0)))
    .slice(0, 6);

  const handleSave = () => {
    const customer = (selectedCustomer || '').trim() || (selected?.customer ? String(selected.customer) : '');
    const muted_tokens = mutedTokensText
      .split('\n')
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean);
    const muted_patterns = mutedPatternsText
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean);

    upsertMonitorProfileMutation.mutate({
      customer: customer || undefined,
      muted_tokens,
      muted_patterns,
      notes: notes || undefined,
      merge_lists: false,
    } as any);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Monitor Profiles</h2>
            <p className="text-sm text-gray-500">Manage mutes and inspect learned tokens (per customer)</p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" onClick={() => refetchMonitorProfiles()}>
              <RefreshCw className="w-4 h-4" />
            </Button>
            <Button variant="ghost" onClick={() => onClose()}>
              <XCircle className="w-5 h-5" />
            </Button>
          </div>
        </div>

        <div className="flex flex-1 min-h-0">
          <div className="w-1/3 border-r border-gray-200 p-4 overflow-y-auto">
            <div className="text-sm font-medium text-gray-900 mb-2">Profiles</div>
            {monitorProfilesLoading ? (
              <div className="py-6 flex justify-center">
                <LoadingSpinner />
              </div>
            ) : (
              <div className="space-y-2">
                <button
                  className={`w-full text-left px-3 py-2 rounded border ${
                    !selectedCustomer ? 'border-primary-300 bg-primary-50' : 'border-gray-200 hover:bg-gray-50'
                  }`}
                  onClick={() => setSelectedCustomer('')}
                >
                  <div className="text-sm font-medium text-gray-900">Global</div>
                  <div className="text-xs text-gray-500">Applies when customer is empty</div>
                </button>
                {profiles
                  .filter((p: any) => !!p?.customer)
                  .map((p: any) => (
                    <button
                      key={p.id}
                      className={`w-full text-left px-3 py-2 rounded border ${
                        selectedCustomer === String(p.customer)
                          ? 'border-primary-300 bg-primary-50'
                          : 'border-gray-200 hover:bg-gray-50'
                      }`}
                      onClick={() => setSelectedCustomer(String(p.customer))}
                    >
                      <div className="text-sm font-medium text-gray-900 truncate">{String(p.customer)}</div>
                      <div className="text-xs text-gray-500">Updated: {new Date(p.updated_at).toLocaleString()}</div>
                    </button>
                  ))}
                <div className="mt-4">
                  <div className="text-xs text-gray-500 mb-1">Create / select customer</div>
                  <input
                    className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                    placeholder="Customer tag (e.g. Acme)"
                    value={selectedCustomer}
                    onChange={(e) => setSelectedCustomer(e.target.value)}
                  />
                </div>
              </div>
            )}
          </div>

          <div className="w-2/3 p-4 overflow-y-auto">
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="text-sm font-medium text-gray-900">
                  {selectedCustomer ? `Customer: ${selectedCustomer}` : 'Global profile'}
                </div>
                <div className="text-xs text-gray-500">
                  Learned tokens come from accept/reject; mutes are applied immediately to monitors.
                </div>
              </div>
              <Button
                variant="secondary"
                onClick={handleSave}
                disabled={upsertMonitorProfileMutation.isLoading}
              >
                {upsertMonitorProfileMutation.isLoading ? 'Saving…' : 'Save'}
              </Button>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-gray-50 border border-gray-200 rounded p-3">
                <div className="text-xs font-medium text-gray-700 mb-2">Top positive tokens</div>
                {topPositive.length === 0 ? (
                  <div className="text-xs text-gray-500">No learned positives yet.</div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {topPositive.map(([t, v]) => (
                      <span key={t} className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                        {t} (+{v})
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <div className="bg-gray-50 border border-gray-200 rounded p-3">
                <div className="text-xs font-medium text-gray-700 mb-2">Top negative tokens</div>
                {topNegative.length === 0 ? (
                  <div className="text-xs text-gray-500">No learned negatives yet.</div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {topNegative.map(([t, v]) => (
                      <span key={t} className="text-xs bg-red-100 text-red-800 px-2 py-1 rounded">
                        {t} ({v})
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="bg-gray-50 border border-gray-200 rounded p-3">
                <div className="text-xs font-medium text-gray-700 mb-2">Recommendation weights</div>
                {topRecommendations.length === 0 ? (
                  <div className="text-xs text-gray-500">No learned recommendation signals yet.</div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {topRecommendations.map(([k, v]) => (
                      <span key={k} className={`text-xs px-2 py-1 rounded ${Number(v) >= 0 ? 'bg-blue-100 text-blue-800' : 'bg-red-100 text-red-800'}`}>
                        {k} ({Number(v) >= 0 ? '+' : ''}{v})
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <div className="bg-gray-50 border border-gray-200 rounded p-3">
                <div className="text-xs font-medium text-gray-700 mb-2">Source type weights</div>
                {topSourceTypes.length === 0 ? (
                  <div className="text-xs text-gray-500">No learned source-type signals yet.</div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {topSourceTypes.map(([k, v]) => (
                      <span key={k} className={`text-xs px-2 py-1 rounded ${Number(v) >= 0 ? 'bg-teal-100 text-teal-800' : 'bg-rose-100 text-rose-800'}`}>
                        {k} ({Number(v) >= 0 ? '+' : ''}{v})
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <div className="bg-gray-50 border border-gray-200 rounded p-3">
                <div className="text-xs font-medium text-gray-700 mb-2">Outcome counters</div>
                {Object.keys(outcomeCounters).length === 0 ? (
                  <div className="text-xs text-gray-500">No follow-up outcomes yet.</div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(outcomeCounters).map(([k, v]) => (
                      <span key={k} className="text-xs bg-slate-100 text-slate-700 px-2 py-1 rounded">
                        {k} {v}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Muted tokens (one per line)</label>
                <textarea
                  className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                  rows={10}
                  value={mutedTokensText}
                  onChange={(e) => setMutedTokensText(e.target.value)}
                  placeholder="e.g.\nbenchmark\nnewsletter"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Muted phrases (substring match)</label>
                <textarea
                  className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                  rows={10}
                  value={mutedPatternsText}
                  onChange={(e) => setMutedPatternsText(e.target.value)}
                  placeholder="e.g.\nweekly roundup\ncall for papers"
                />
              </div>
            </div>
            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
              <textarea
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                rows={3}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MonitorProfilesModal;
