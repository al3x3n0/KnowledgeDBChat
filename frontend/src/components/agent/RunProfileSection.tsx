/**
 * Where a run's time went: each job of the run on one timeline, split into
 * queued, running and the wait after the job it follows, with the totals by
 * phase and by tool beneath. Adapted from ucb-bar/chia's profiler, which
 * renders the same two views (a timeline and an aggregated table) of a loop.
 *
 * Collapsed until opened: the profile reads every job's log in the run, and
 * most visits to a job do not need it.
 */

import React, { useState } from 'react';
import { useQuery } from 'react-query';

import type { components } from '../../api/schema';
import { apiClient } from '../../services/api';

type Profile = components['schemas']['AgentRunProfile'];
type ProfileJob = components['schemas']['AgentRunProfileJob'];

const PHASES = ['observe', 'think', 'act', 'evaluate'] as const;

export function formatSeconds(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
}

function offsetSeconds(job: ProfileJob, start: number): number {
  const created = job.created_at ? Date.parse(job.created_at) : start;
  return Math.max(0, (created - start) / 1000);
}

function JobBar({ job, start, wall }: { job: ProfileJob; start: number; wall: number }) {
  const scale = (s: number | null | undefined) => `${(100 * (s || 0)) / Math.max(wall, 1)}%`;
  const label = job.stage || job.name || job.id.slice(0, 8);
  const offset = offsetSeconds(job, start);
  return (
    <div className="flex items-center gap-2 text-xs">
      <div className="w-40 truncate text-gray-700" title={job.name || job.id}>
        {label}
      </div>
      <div className="relative flex-1 h-4 bg-gray-100 rounded">
        <span
          className="absolute top-0 h-4 bg-amber-300"
          style={{ left: scale(offset), width: scale(job.queued_seconds) }}
          title={`queued ${formatSeconds(job.queued_seconds)}`}
        />
        <span
          className="absolute top-0 h-4 rounded bg-sky-500"
          style={{ left: scale(offset + (job.queued_seconds || 0)), width: scale(job.ran_seconds) }}
          title={`ran ${formatSeconds(job.ran_seconds)}`}
        />
      </div>
      <div className="w-24 text-right text-gray-600">
        {formatSeconds(job.ran_seconds)}
        {job.paused_seconds ? (
          <span className="block text-[11px] text-amber-700" title="Stopped without completing">
            paused {formatSeconds(job.paused_seconds)}
          </span>
        ) : null}
      </div>
    </div>
  );
}

export function RunProfileView({ profile }: { profile: Profile }) {
  const starts = profile.jobs
    .map((j) => (j.created_at ? Date.parse(j.created_at) : NaN))
    .filter((t) => !Number.isNaN(t));
  const start = starts.length ? Math.min(...starts) : Date.now();
  const wall = profile.wall_seconds || 0;
  const phaseTotal = PHASES.reduce((sum, p) => sum + (profile.totals.phase_ms[p] || 0), 0);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-3 text-xs text-gray-700">
        <span>Wall: {formatSeconds(profile.wall_seconds)}</span>
        <span>Running: {formatSeconds(profile.totals.ran_seconds)}</span>
        <span>Queued: {formatSeconds(profile.totals.queued_seconds)}</span>
        <span title="From a job ending to the next one existing, including any approval waited for">
          Between jobs: {formatSeconds(profile.totals.after_parent_seconds)}
        </span>
      </div>
      <div className="space-y-1">
        {profile.jobs.map((job) => (
          <JobBar key={job.id} job={job} start={start} wall={wall} />
        ))}
      </div>
      {phaseTotal > 0 && (
        <div className="text-xs">
          <div className="font-medium text-gray-700 mb-1">By phase</div>
          <div className="flex h-3 rounded overflow-hidden">
            {PHASES.map((phase) => (
              <span
                key={phase}
                className={
                  {
                    observe: 'bg-gray-400',
                    think: 'bg-violet-500',
                    act: 'bg-sky-500',
                    evaluate: 'bg-emerald-500',
                  }[phase]
                }
                style={{ width: `${(100 * (profile.totals.phase_ms[phase] || 0)) / phaseTotal}%` }}
                title={`${phase} ${formatSeconds((profile.totals.phase_ms[phase] || 0) / 1000)}`}
              />
            ))}
          </div>
          <div className="flex gap-3 mt-1 text-gray-600">
            {PHASES.map((phase) => (
              <span key={phase}>
                {phase} {formatSeconds((profile.totals.phase_ms[phase] || 0) / 1000)}
              </span>
            ))}
          </div>
        </div>
      )}
      {profile.totals.tools.length > 0 && (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-gray-600">
              <th className="font-medium">Tool</th>
              <th className="font-medium text-right">Calls</th>
              <th className="font-medium text-right">Failed</th>
              <th className="font-medium text-right">Act time</th>
            </tr>
          </thead>
          <tbody>
            {profile.totals.tools.slice(0, 12).map((row) => (
              <tr key={row.tool} className="text-gray-700">
                <td className="truncate">{row.tool}</td>
                <td className="text-right">{row.calls}</td>
                <td className="text-right">{row.failures}</td>
                <td className="text-right">{formatSeconds(row.act_ms / 1000)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {profile.iterations_without_phase_times > 0 && (
        <p className="text-xs text-gray-500">
          {profile.iterations_without_phase_times} iteration(s) were logged before phases were
          timed; they count toward the timeline but not the phase or tool totals.
        </p>
      )}
      {profile.truncated && (
        <p className="text-xs text-gray-500">Only the first 50 jobs of this run are shown.</p>
      )}
    </div>
  );
}

export const RunProfileSection: React.FC<{ jobId: string }> = ({ jobId }) => {
  const [open, setOpen] = useState(false);
  const { data, isLoading, isError } = useQuery(
    ['agent-job-profile', jobId],
    () => apiClient.getAgentJobProfile(jobId),
    { enabled: open, staleTime: 30_000 }
  );
  return (
    <div className="mb-4">
      <button
        type="button"
        className="text-sm font-medium text-gray-700 mb-2"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        {open ? '▾' : '▸'} Where the time went
      </button>
      {open && (
        <div className="bg-sky-50 border border-sky-100 rounded-lg p-3">
          {isLoading && <p className="text-xs text-gray-500">Reading the run…</p>}
          {isError && <p className="text-xs text-red-600">The profile could not be read.</p>}
          {data && <RunProfileView profile={data} />}
        </div>
      )}
    </div>
  );
};

export default RunProfileSection;
