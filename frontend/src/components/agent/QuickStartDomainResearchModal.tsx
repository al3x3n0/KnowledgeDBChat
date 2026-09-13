/**
 * Quick start: a domain research run.
 *
 * Module scope, like its siblings — declared in the page's render body it was
 * remounted on every page render, resetting the form below.
 */

import React, { useMemo, useState } from 'react';
import toast from 'react-hot-toast';

import {
  buildDomainResearchQuickStartPayload,
  DOMAIN_SOURCE_SCOPE_OPTIONS,
  DOMAIN_TRACK_OPTIONS,
  splitUniqueLines,
} from '../../pages/autonomousAgentQuickStarts';
import Button from '../common/Button';

interface QuickStartDomainResearchModalProps {
  onClose: () => void;
  quickStartDomainResearchMutation: any;
  /** The github/gitlab sources this run can be scoped to. */
  codeSources: any[];
  /** What the user typed into the template recommender, as a default goal. */
  templateRecommendGoal: string;
}

export const QuickStartDomainResearchModal: React.FC<QuickStartDomainResearchModalProps> = ({
  onClose,
  quickStartDomainResearchMutation,
  codeSources,
  templateRecommendGoal,
}) => {

  const [name, setName] = useState(`Domain Research - ${new Date().toLocaleDateString()}`);
  const [domain, setDomain] = useState('');
  const [objective, setObjective] = useState(
    templateRecommendGoal.trim() || 'Identify evidence-backed compiler opportunities, risks, and next experiments for this codebase'
  );
  const [customerContextValue, setCustomerContextValue] = useState('');
  const [trackType, setTrackType] = useState<'compiler' | 'microarchitecture' | 'generic'>('compiler');
  const [sourceScope, setSourceScope] = useState<'kb_only' | 'arxiv_only' | 'kb_plus_arxiv' | 'kb_plus_arxiv_plus_repo'>(
    codeSources.length > 0 ? 'kb_plus_arxiv_plus_repo' : 'kb_plus_arxiv'
  );
  const [monitorQueriesText, setMonitorQueriesText] = useState('');
  const [benchmarkQueriesText, setBenchmarkQueriesText] = useState('');
  const [repoSelection, setRepoSelection] = useState<Record<string, boolean>>({});
  const [reportFormat, setReportFormat] = useState<'brief_only' | 'report_only' | 'brief_and_report'>('brief_and_report');
  const [persistArtifacts, setPersistArtifacts] = useState(true);
  const [autoLaunchFollowUp, setAutoLaunchFollowUp] = useState(true);

  const monitorQueriesPreview = useMemo(
    () => splitUniqueLines(monitorQueriesText, 12),
    [monitorQueriesText]
  );
  const benchmarkQueriesPreview = useMemo(
    () => splitUniqueLines(benchmarkQueriesText, 16),
    [benchmarkQueriesText]
  );
  const selectedRepoSourceIds = useMemo(
    () => Object.entries(repoSelection).filter(([, enabled]) => enabled).map(([id]) => id),
    [repoSelection]
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!domain.trim()) {
      toast.error('Domain is required');
      return;
    }
    if (!objective.trim()) {
      toast.error('Objective is required');
      return;
    }
    quickStartDomainResearchMutation.mutate(
      buildDomainResearchQuickStartPayload({
        name,
        domain,
        objective,
        customerContextValue,
        trackType,
        sourceScope,
        monitorQueriesText,
        benchmarkQueriesText,
        selectedRepoSourceIds,
        sandboxProfileId:
          trackType === 'compiler'
            ? 'scientific-compiler-sandbox'
            : trackType === 'microarchitecture'
              ? 'scientific-microarchitecture-sandbox'
              : 'scientific-generic-sandbox',
        reportFormat,
        persistArtifacts,
        autoLaunchFollowUp,
      })
    );
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-lg font-semibold mb-1">Quick Start Domain Research</h2>
          <p className="text-sm text-gray-500 mb-4">
            Research a technical domain, rank ideas, generate a brief/report, and persist outputs as Research Notes.
          </p>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Job Name</label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Domain</label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
                placeholder="Compiler optimization and code generation"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Objective</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={4}
                value={objective}
                onChange={(e) => setObjective(e.target.value)}
                placeholder="Rank compiler opportunities, explain the strongest evidence, and propose next experiments"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Customer or operating context (optional)</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={3}
                value={customerContextValue}
                onChange={(e) => setCustomerContextValue(e.target.value)}
                placeholder="Constraints, target market, product context, or deployment assumptions"
              />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Track</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={trackType}
                  onChange={(e) => setTrackType(e.target.value as any)}
                >
                  {DOMAIN_TRACK_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Evidence scope</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={sourceScope}
                  onChange={(e) => setSourceScope(e.target.value as any)}
                >
                  {DOMAIN_SOURCE_SCOPE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Output format</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={reportFormat}
                  onChange={(e) => setReportFormat(e.target.value as any)}
                >
                  <option value="brief_and_report">Brief + report</option>
                  <option value="brief_only">Brief only</option>
                  <option value="report_only">Report only</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Monitor queries (optional, one per line) {monitorQueriesPreview.length}/12
              </label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={3}
                value={monitorQueriesText}
                onChange={(e) => setMonitorQueriesText(e.target.value)}
                placeholder={'multimodal retrieval benchmarking\nretrieval-augmented generation latency'}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Benchmark queries (optional, one per line) {benchmarkQueriesPreview.length}/16
              </label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={3}
                value={benchmarkQueriesText}
                onChange={(e) => setBenchmarkQueriesText(e.target.value)}
                placeholder={'compile time regression\nbranch miss benchmark\ncache-sensitive kernel'}
              />
            </div>
            {codeSources.length > 0 ? (
              <div className="border border-gray-200 rounded-lg p-3 bg-gray-50">
                <div className="text-sm font-medium text-gray-800 mb-2">Repository evidence sources</div>
                <div className="space-y-2 max-h-40 overflow-auto">
                  {codeSources.map((source: any) => (
                    <label key={String(source.id)} className="flex items-start gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={Boolean(repoSelection[String(source.id)])}
                        onChange={(e) => setRepoSelection((prev) => ({ ...prev, [String(source.id)]: e.target.checked }))}
                      />
                      <span>
                        <span className="font-medium text-gray-900">{String(source.name || source.id)}</span>
                        <span className="block text-xs text-gray-500">{String(source.source_type || '').toLowerCase()}</span>
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            ) : null}
            <div className="rounded-lg border border-cyan-100 bg-cyan-50 px-3 py-2 text-xs text-cyan-800 space-y-1">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  className="rounded border-gray-300"
                  checked={persistArtifacts}
                  onChange={(e) => setPersistArtifacts(e.target.checked)}
                />
                Persist brief/report as Research Notes
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  className="rounded border-gray-300"
                  checked={autoLaunchFollowUp}
                  onChange={(e) => setAutoLaunchFollowUp(e.target.checked)}
                />
                Auto-launch deep-dive follow-up for the strongest idea when confidence passes
              </label>
            </div>
            <div className="flex justify-end gap-3 pt-4 border-t">
              <Button type="button" variant="secondary" onClick={() => onClose()}>
                Cancel
              </Button>
              <Button type="submit" disabled={quickStartDomainResearchMutation.isLoading}>
                {quickStartDomainResearchMutation.isLoading ? 'Starting...' : 'Start Research'}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default QuickStartDomainResearchModal;
