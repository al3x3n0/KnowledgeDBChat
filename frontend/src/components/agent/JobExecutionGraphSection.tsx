/**
 * The execution-graph and scope-observability section of a run's detail view.
 *
 * The first section lifted out of JobDetailPanel, and it only became possible
 * once the derivations moved: measured against the panel it needed fifteen
 * values, of which six were read by other sections too. With
 * executionGraphView owning those, it needs one prop — the job — and derives
 * the rest itself.
 */

import React, { useMemo } from 'react';

import type { AgentJob } from '../../types';
import {
  executionGraphView,
  executiveDigestOf,
  isLiveRuntimeJob,
} from '../../utils/agentJobDetail';

export const JobExecutionGraphSection: React.FC<{ job: AgentJob }> = ({ job }) => {
  const {
    executionGraph,
    scopeObservability,
    graphHealth,
    dagStats,
    graphHealthStatus,
    graphHealthBadgeClass,
    graphRecommendedActions,
    graphVerificationActions,
    graphSummarizationActions,
    scopeResolvedId,
    scopeSource,
    scopeEvents,
    recentScopeEvents,
    scopeGuardBlocks,
  } = useMemo(() => executionGraphView(job), [job]);

  const executiveDigest = useMemo(() => executiveDigestOf(job), [job]);
  const liveRuntimeJob = isLiveRuntimeJob(job);

  return (
    <>
      {/* Execution graph */}
      {(executionGraph || dagStats || graphHealth) && (
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-sm font-medium text-gray-700">Execution Graph</h3>
            {liveRuntimeJob && (
              <span className="px-2 py-0.5 rounded-full border border-violet-200 bg-violet-100 text-[11px] font-medium text-violet-700">
                Live runtime
              </span>
            )}
          </div>
          <div className="bg-violet-50 border border-violet-100 rounded-lg p-3 space-y-2">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className={`px-2 py-0.5 rounded-full border font-medium ${graphHealthBadgeClass}`}>
                Health: {graphHealthStatus || 'unknown'}
              </span>
              {graphHealth?.severity_score !== undefined && (
                <span className="text-violet-700">Severity: {Number(graphHealth.severity_score || 0)}</span>
              )}
              {graphHealth?.blocked_ratio !== undefined && (
                <span className="text-violet-700">Blocked ratio: {(Number(graphHealth.blocked_ratio || 0) * 100).toFixed(1)}%</span>
              )}
              {graphVerificationActions.length > 0 && (
                <span className="text-violet-700">Verification actions: {graphVerificationActions.length}</span>
              )}
              {graphSummarizationActions.length > 0 && (
                <span className="text-violet-700">Summaries: {graphSummarizationActions.length}</span>
              )}
            </div>

            {dagStats && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-violet-800">
                <div>Nodes: {Number((dagStats as any)?.total_nodes || 0)}</div>
                <div>Edges: {Number((dagStats as any)?.total_edges || 0)}</div>
                <div>Critical path: {Number((dagStats as any)?.critical_path_length || 0)}</div>
                <div>Blocked nodes: {Number((dagStats as any)?.blocked_nodes || 0)}</div>
                <div>Root nodes: {Number((dagStats as any)?.root_nodes || 0)}</div>
                <div>Leaf nodes: {Number((dagStats as any)?.leaf_nodes || 0)}</div>
                <div>Orphans: {Number((dagStats as any)?.orphan_nodes || 0)}</div>
                <div>Cycle: {(dagStats as any)?.has_cycle ? 'yes' : 'no'}</div>
              </div>
            )}

            {Array.isArray(graphHealth?.reasons) && (graphHealth?.reasons?.length || 0) > 0 && (
              <div>
                <div className="text-xs font-medium text-violet-900 mb-1">Signals</div>
                <ul className="text-xs text-violet-800 space-y-1">
                  {(graphHealth?.reasons || []).slice(0, 8).map((r: string, idx: number) => (
                    <li key={`${idx}-${r.slice(0, 24)}`}>- {r}</li>
                  ))}
                </ul>
              </div>
            )}

            {graphRecommendedActions.length > 0 && (
              <div>
                <div className="text-xs font-medium text-violet-900 mb-1">Recommended Actions</div>
                <ul className="text-xs text-violet-800 space-y-1">
                  {graphRecommendedActions.map((r: string, idx: number) => (
                    <li key={`${idx}-${r.slice(0, 24)}`}>- {r}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {scopeObservability && (
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-sm font-medium text-gray-700">Scope Observability</h3>
            {liveRuntimeJob && (
              <span className="px-2 py-0.5 rounded-full border border-sky-200 bg-sky-100 text-[11px] font-medium text-sky-700">
                Live runtime
              </span>
            )}
          </div>
          <div className="bg-sky-50 border border-sky-100 rounded-lg p-3 space-y-2">
            <div className="flex flex-wrap gap-3 text-xs text-sky-800">
              <span>Resolved scope: {scopeResolvedId || 'none'}</span>
              <span>Scope source: {scopeSource || 'none'}</span>
              <span>Scope events: {scopeEvents.length}</span>
              {scopeGuardBlocks > 0 && <span>Guard blocks: {scopeGuardBlocks}</span>}
            </div>

            {recentScopeEvents.length > 0 && (
              <div>
                <div className="text-xs font-medium text-sky-900 mb-1">Recent scope events</div>
                <ul className="text-xs text-sky-800 space-y-1">
                  {recentScopeEvents.map((event, idx) => {
                    const eventType = String(event?.type || 'event').trim() || 'event';
                    const eventScope = String(event?.source_id || event?.resolved_scope_id || '').trim();
                    const eventSource = String(event?.scope_source || '').trim();
                    return (
                      <li key={`${idx}-${eventType}-${eventScope}`}>
                        - {eventType}
                        {eventScope ? ` | scope ${eventScope}` : ''}
                        {eventSource ? ` | source ${eventSource}` : ''}
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Executive digest */}
      {executiveDigest && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Executive Digest</h3>
          <div className="bg-sky-50 border border-sky-100 rounded-lg p-3 space-y-2">
            {executiveDigest?.outcome && (
              <p className="text-sm text-sky-800">{String(executiveDigest.outcome)}</p>
            )}
            {executiveDigest?.metrics && typeof executiveDigest.metrics === 'object' && (
              <div className="flex flex-wrap gap-3 text-xs text-sky-700">
                <span>Progress: {Number((executiveDigest.metrics as any).goal_progress || 0)}%</span>
                <span>Iterations: {Number((executiveDigest.metrics as any).iterations || 0)}</span>
                <span>Findings: {Number((executiveDigest.metrics as any).findings_count || 0)}</span>
                <span>Artifacts: {Number((executiveDigest.metrics as any).artifacts_count || 0)}</span>
              </div>
            )}
            {Array.isArray(executiveDigest?.key_findings) && executiveDigest.key_findings.length > 0 && (
              <div>
                <div className="text-xs font-medium text-sky-800 mb-1">Key findings</div>
                <ul className="text-xs text-sky-700 space-y-1">
                  {executiveDigest.key_findings.slice(0, 5).map((f: string, idx: number) => (
                    <li key={`${idx}-${f.slice(0, 24)}`}>- {f}</li>
                  ))}
                </ul>
              </div>
            )}
            {Array.isArray(executiveDigest?.risks) && executiveDigest.risks.length > 0 && (
              <div>
                <div className="text-xs font-medium text-sky-800 mb-1">Risks</div>
                <ul className="text-xs text-sky-700 space-y-1">
                  {executiveDigest.risks.slice(0, 4).map((r: string, idx: number) => (
                    <li key={`${idx}-${r.slice(0, 24)}`}>- {r}</li>
                  ))}
                </ul>
              </div>
            )}
            {Array.isArray(executiveDigest?.next_actions) && executiveDigest.next_actions.length > 0 && (
              <div>
                <div className="text-xs font-medium text-sky-800 mb-1">Next actions</div>
                <ul className="text-xs text-sky-700 space-y-1">
                  {executiveDigest.next_actions.slice(0, 4).map((step: string, idx: number) => (
                    <li key={`${idx}-${step.slice(0, 24)}`}>- {step}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

    </>
  );
};

export default JobExecutionGraphSection;
