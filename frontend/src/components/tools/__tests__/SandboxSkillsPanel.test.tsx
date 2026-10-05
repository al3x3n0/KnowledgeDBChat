import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import toast from 'react-hot-toast';
import { SandboxSkillsPanel } from '../SandboxSkillsPanel';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
}));

jest.mock('../../../services/api', () => ({
  apiClient: {
    listSandboxSkills: jest.fn(),
    createSandboxSkill: jest.fn(),
    updateSandboxSkill: jest.fn(),
    deleteSandboxSkill: jest.fn(),
    dryRunSandboxSkill: jest.fn(),
    activateSandboxSkill: jest.fn(),
    disableSandboxSkill: jest.fn(),
    draftSandboxSkill: jest.fn(),
    getSandboxSkillDraft: jest.fn(),
    listSandboxSkillImages: jest.fn(),
    proposeSandboxSkillImage: jest.fn(),
    buildSandboxSkillImage: jest.fn(),
    rejectSandboxSkillImage: jest.fn(),
  },
}));

const apiClient = require('../../../services/api').apiClient;

const IMAGE = 'ghcr.io/al3x3n0/kdbc-compiler-research:latest';

const skill = {
  id: 'skill-1',
  slug: 'loop_trip',
  name: 'Loop trip counts',
  description: 'Count how many times each loop runs.',
  manifest: {
    id: 'loop_trip',
    name: 'Loop trip counts',
    description: 'Count how many times each loop runs.',
    image: IMAGE,
    procedure: 'Build the kernel, then run skill/report.sh.',
    files: {},
    result: { fields: { loops: 'number' } },
    control: { command: 'sh skill/report.sh' },
  },
  status: 'draft',
  origin: 'manual',
  origin_job_id: null,
  produces: 'skill_loop_trip',
  verified: false,
  last_dry_run: null,
  notes: [],
  created_at: '2026-10-01T10:00:00Z',
  updated_at: '2026-10-01T10:00:00Z',
};

const listing = (items: any[], overrides: Record<string, any> = {}) => ({
  items,
  images: [IMAGE],
  execution_enabled: true,
  authoring_enabled: true,
  image_build_enabled: false,
  ...overrides,
});

describe('SandboxSkillsPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.listSandboxSkills.mockResolvedValue(listing([skill]));
    apiClient.listSandboxSkillImages.mockResolvedValue({ items: [], build_enabled: false });
  });

  it('does not offer Activate for a skill whose control has not passed', async () => {
    // The API would refuse it, and an action that fails is worse than one
    // that is absent.
    render(<SandboxSkillsPanel />);

    expect(await screen.findByText('Loop trip counts')).toBeInTheDocument();
    expect(screen.getByText('skill_loop_trip')).toBeInTheDocument();
    expect(screen.getByText('Unverified')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Activate/ })).toBeNull();
  });

  it('offers Activate once the control has passed', async () => {
    apiClient.listSandboxSkills.mockResolvedValue(
      listing([{ ...skill, verified: true }])
    );
    apiClient.activateSandboxSkill.mockResolvedValue({ ...skill, status: 'active' });
    render(<SandboxSkillsPanel />);

    fireEvent.click(await screen.findByRole('button', { name: /Activate/ }));

    await waitFor(() =>
      expect(apiClient.activateSandboxSkill).toHaveBeenCalledWith('skill-1')
    );
  });

  it('runs the control and shows why it failed', async () => {
    apiClient.dryRunSandboxSkill.mockResolvedValue({
      skill,
      ok: false,
      ran: true,
      detail: 'The control command exited 127: sh: skill/report.sh: not found',
    });
    render(<SandboxSkillsPanel />);

    fireEvent.click(await screen.findByRole('button', { name: /Run control/ }));

    await waitFor(() =>
      expect(apiClient.dryRunSandboxSkill).toHaveBeenCalledWith('skill-1')
    );
    expect((toast.error as jest.Mock).mock.calls[0][0]).toMatch(/not found/);
  });

  it('marks a skill a run proposed, so it is read before it is activated', async () => {
    apiClient.listSandboxSkills.mockResolvedValue(
      listing([{ ...skill, origin: 'agent' }])
    );
    render(<SandboxSkillsPanel />);

    expect(await screen.findByText('proposed by a run')).toBeInTheDocument();
  });

  it('distinguishes a control that could not run from one that failed', async () => {
    apiClient.listSandboxSkills.mockResolvedValue(
      listing([
        {
          ...skill,
          last_dry_run: {
            ok: false,
            ran: false,
            detail: 'Sandbox execution is disabled on this server.',
          },
        },
      ])
    );
    render(<SandboxSkillsPanel />);

    fireEvent.click(await screen.findByText('Loop trip counts'));

    expect(await screen.findByText(/Control could not run:/)).toBeInTheDocument();
  });

  it('says so when no skill can be activated on this server', async () => {
    apiClient.listSandboxSkills.mockResolvedValue(
      listing([skill], { execution_enabled: false })
    );
    render(<SandboxSkillsPanel />);

    expect(
      await screen.findByText(/Sandbox execution is disabled on this server, so/)
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Run control/ })).toBeDisabled();
  });

  it('says where the skill is wrong rather than only that it is', async () => {
    render(<SandboxSkillsPanel />);

    await screen.findByText('Loop trip counts');
    fireEvent.click(screen.getByRole('button', { name: /New Skill/ }));
    fireEvent.change(screen.getByLabelText('Skill manifest'), {
      target: { value: '{ not json' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => expect(toast.error).toHaveBeenCalled());
    expect((toast.error as jest.Mock).mock.calls[0][0]).toMatch(
      /The skill is not valid JSON/
    );
    expect(apiClient.createSandboxSkill).not.toHaveBeenCalled();
  });

  it('revises what is in the editor rather than drafting from nothing', async () => {
    // A hand edit made between passes must reach the drafter, or it is
    // silently discarded by the next draft.
    apiClient.draftSandboxSkill.mockResolvedValue({ task_id: 't1', poll_url: '' });
    apiClient.getSandboxSkillDraft.mockResolvedValue({
      state: 'SUCCESS',
      stage: 'done',
      attempt: 1,
      attempts: 1,
      notes: [],
      manifest: skill.manifest,
      dry_run: { ok: true, ran: true, detail: 'passed' },
      pending: false,
    });
    render(<SandboxSkillsPanel />);

    await screen.findByText('Loop trip counts');
    fireEvent.click(screen.getByTitle('Edit'));
    fireEvent.change(screen.getByLabelText('Describe the skill'), {
      target: { value: 'add a judge' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Revise' }));

    await waitFor(() => expect(apiClient.draftSandboxSkill).toHaveBeenCalled());
    const [description, current] = apiClient.draftSandboxSkill.mock.calls[0];
    expect(description).toBe('add a judge');
    expect(current.id).toBe('loop_trip');
  });

  it('offers building an image only to an administrator, and only when enabled', async () => {
    const image = {
      id: 'img-1',
      slug: 'z3',
      dockerfile: `FROM ${IMAGE}\nRUN true\n`,
      base_image: IMAGE,
      tag: 'kdbc-skill/z3:abc',
      status: 'proposed',
      created_at: '',
      updated_at: '',
    };
    apiClient.listSandboxSkillImages.mockResolvedValue({
      items: [image],
      build_enabled: true,
    });

    const { unmount } = render(<SandboxSkillsPanel />);
    fireEvent.click(await screen.findByText('Skill images'));
    expect(await screen.findByText('kdbc-skill/z3:abc')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Approve/ })).toBeNull();
    unmount();

    render(<SandboxSkillsPanel isAdmin />);
    fireEvent.click(await screen.findByText('Skill images'));
    expect(await screen.findByRole('button', { name: /Approve/ })).toBeInTheDocument();
  });
});
