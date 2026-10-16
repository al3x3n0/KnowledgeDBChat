import React from 'react';
import { render, screen } from '@testing-library/react';

import { RunProfileView, formatSeconds } from '../RunProfileSection';

const profile = {
  root_job_id: 'root',
  wall_seconds: 3660,
  jobs: [
    {
      id: 'root-id',
      name: 'planner',
      stage: null,
      status: 'completed',
      created_at: '2026-10-14T12:00:00+00:00',
      queued_seconds: 30,
      ran_seconds: 570,
      after_parent_seconds: null,
      iterations: [],
      phase_ms: { observe: 0, think: 0, act: 0, evaluate: 0 },
      tools: [],
    },
    {
      id: 'stage-id',
      parent_id: 'root-id',
      name: 'pipeline: implement',
      stage: 'implement',
      status: 'completed',
      created_at: '2026-10-14T12:40:00+00:00',
      queued_seconds: 60,
      ran_seconds: 1200,
      after_parent_seconds: 1800,
      iterations: [],
      phase_ms: { observe: 4000, think: 500000, act: 540000, evaluate: 2000 },
      tools: [],
    },
  ],
  totals: {
    queued_seconds: 90,
    ran_seconds: 1770,
    after_parent_seconds: 1800,
    phase_ms: { observe: 4000, think: 500000, act: 540000, evaluate: 2000 },
    tools: [{ tool: 'check_implementation', calls: 2, failures: 1, act_ms: 540000 }],
  },
  iterations_without_phase_times: 2,
  truncated: false,
};

describe('RunProfileView', () => {
  it('shows each job of the run, named by its stage where it has one', () => {
    render(<RunProfileView profile={profile} />);

    expect(screen.getByText('planner')).toBeInTheDocument();
    expect(screen.getByText('implement')).toBeInTheDocument();
  });

  it('says how long the run waited between jobs', () => {
    render(<RunProfileView profile={profile} />);

    expect(screen.getByText('Between jobs: 30m 0s')).toBeInTheDocument();
    expect(screen.getByText('Wall: 1h 1m')).toBeInTheDocument();
  });

  it('ranks the tools by the time spent acting', () => {
    render(<RunProfileView profile={profile} />);

    const row = screen.getByText('check_implementation').closest('tr');
    expect(row).toHaveTextContent('check_implementation219m 0s');
  });

  it('says when older iterations were not timed by phase', () => {
    render(<RunProfileView profile={profile} />);

    expect(screen.getByText(/2 iteration\(s\) were logged before phases were timed/)).toBeInTheDocument();
  });

  it('says a job that stopped without completing is paused, apart from its running time', () => {
    const paused = {
      ...profile,
      jobs: [{ ...profile.jobs[1], status: 'paused', ran_seconds: 590, paused_seconds: 35400 }],
    };

    render(<RunProfileView profile={paused} />);

    expect(screen.getByText('paused 9h 50m')).toBeInTheDocument();
  });

  it('formats seconds at every scale', () => {
    expect(formatSeconds(null)).toBe('—');
    expect(formatSeconds(42)).toBe('42s');
    expect(formatSeconds(125)).toBe('2m 5s');
    expect(formatSeconds(7260)).toBe('2h 1m');
  });
});
