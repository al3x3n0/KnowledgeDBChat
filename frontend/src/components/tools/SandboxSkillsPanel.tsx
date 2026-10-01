/**
 * Sandbox skills: packaged procedures an agent run loads and carries out.
 *
 * Every row leads with the two things that decide whether a skill should be
 * trusted: whether its control has passed against the content it has *now*,
 * and who wrote it. A skill a run proposed and a skill a person wrote look the
 * same once active, so the difference is shown while it still matters.
 *
 * There is one path to "active" and the buttons are arranged along it: create
 * or draft, run the control, activate. Activate is not offered until the
 * control has passed, because the API would refuse it and an action that
 * fails is worse than one that is absent.
 */

import {
  AlertTriangle,
  Bot,
  Box,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  FlaskConical,
  Loader2,
  Pencil,
  Play,
  Plus,
  Power,
  Trash2,
  Wand2,
  XCircle,
} from 'lucide-react';
import React, { useCallback, useEffect, useState } from 'react';
import toast from 'react-hot-toast';

import { apiClient } from '../../services/api';
import type {
  SandboxSkill,
  SandboxSkillDryRun,
  SandboxSkillImage,
} from '../../types';

const exampleManifest = (image: string) =>
  JSON.stringify(
    {
      id: 'loop_trip',
      name: 'Loop trip counts',
      description:
        'Count how many times each loop in a C kernel runs, to find the hot one.',
      image,
      procedure:
        'Write the kernel to kernel.c with a main() that drives it.\n' +
        'Build it: clang -O1 -fprofile-instr-generate -fcoverage-mapping kernel.c -o prog\n' +
        'Run ./prog, then call the tool again with collect_result=true and\n' +
        'the command: sh skill/report.sh',
      files: {
        'report.sh':
          '#!/bin/sh\n# Writes result.json from the coverage report.\necho \'{"loops": 1, "hottest": "main"}\' > result.json\n',
      },
      result: { fields: { loops: 'number', hottest: 'string' } },
      control: { command: 'sh skill/report.sh' },
      timeout_seconds: 300,
    },
    null,
    2
  );

const STATUS_STYLE: Record<string, string> = {
  active: 'bg-primary-600 text-gray-50',
  draft: 'bg-gray-200 text-gray-700',
  disabled: 'bg-gray-200 text-gray-500',
};

const IMAGE_STATUS_STYLE: Record<string, string> = {
  built: 'bg-primary-600 text-gray-50',
  building: 'bg-amber-100 text-amber-800',
  proposed: 'bg-gray-200 text-gray-700',
  failed: 'bg-red-100 text-red-700',
  rejected: 'bg-gray-200 text-gray-500',
};

const ORIGIN_LABEL: Record<string, string> = {
  manual: 'written by hand',
  drafted: 'drafted from a description',
  agent: 'proposed by a run',
};

const DryRunVerdict: React.FC<{ run: SandboxSkillDryRun; verified: boolean }> = ({
  run,
  verified,
}) => (
  <div
    className={`rounded border px-2 py-1.5 text-[11px] ${
      run.ok
        ? 'border-gray-200 bg-white text-gray-700'
        : 'border-amber-300 bg-amber-50 text-amber-800'
    }`}
  >
    <p className="flex items-start gap-1.5">
      {run.ok ? (
        <CheckCircle className="mt-0.5 h-3 w-3 flex-shrink-0 text-primary-600" />
      ) : (
        <XCircle className="mt-0.5 h-3 w-3 flex-shrink-0" />
      )}
      <span>
        {/* Nothing could be tested is not the skill's fault, and saying
            "failed" would send its author to fix something that is fine. */}
        {!run.ran && !run.ok ? 'Control could not run: ' : 'Last control: '}
        {run.detail}
        {run.ok && !verified && ' The skill has been edited since.'}
      </span>
    </p>
    {!run.ok && (run.stderr || run.stdout) && (
      <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap font-mono text-[10px]">
        {run.stderr || run.stdout}
      </pre>
    )}
  </div>
);

export const SandboxSkillsPanel: React.FC<{ isAdmin?: boolean }> = ({
  isAdmin = false,
}) => {
  const [skills, setSkills] = useState<SandboxSkill[]>([]);
  const [images, setImages] = useState<string[]>([]);
  const [executionEnabled, setExecutionEnabled] = useState(true);
  const [authoringEnabled, setAuthoringEnabled] = useState(true);
  const [imageBuildEnabled, setImageBuildEnabled] = useState(false);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState<Record<string, boolean>>({});

  const [composing, setComposing] = useState(false);
  // The skill being edited, or null when composing a new one. The id decides
  // whether Save creates or replaces.
  const [editing, setEditing] = useState<SandboxSkill | null>(null);
  const [manifestText, setManifestText] = useState('');
  const [saving, setSaving] = useState(false);
  const [wish, setWish] = useState('');
  const [drafting, setDrafting] = useState(false);
  const [draftNotes, setDraftNotes] = useState<string[]>([]);
  const [draftStage, setDraftStage] = useState<string | null>(null);
  // True once the editor holds something other than the untouched example, so
  // the Draft box revises what is there instead of discarding it.
  const [hasDraft, setHasDraft] = useState(false);

  const [showImages, setShowImages] = useState(false);
  const [skillImages, setSkillImages] = useState<SandboxSkillImage[]>([]);
  const [imageSlug, setImageSlug] = useState('');
  const [imageDockerfile, setImageDockerfile] = useState('');
  const [proposing, setProposing] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiClient.listSandboxSkills();
      setSkills(data.items || []);
      setImages(data.images || []);
      setExecutionEnabled(Boolean(data.execution_enabled));
      setAuthoringEnabled(Boolean(data.authoring_enabled));
      setImageBuildEnabled(Boolean(data.image_build_enabled));
    } catch {
      // apiClient surfaces the error itself.
    } finally {
      setLoading(false);
    }
  }, []);

  const loadImages = useCallback(async () => {
    try {
      const data = await apiClient.listSandboxSkillImages();
      setSkillImages(data.items || []);
      setImageBuildEnabled(Boolean(data.build_enabled));
    } catch {
      // apiClient surfaces the error itself.
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (showImages) loadImages();
  }, [showImages, loadImages]);

  const openComposer = (skill: SandboxSkill | null) => {
    setEditing(skill);
    setManifestText(
      skill ? JSON.stringify(skill.manifest, null, 2) : exampleManifest(images[0] || '')
    );
    setHasDraft(Boolean(skill));
    setDraftNotes([]);
    setWish('');
    setComposing(true);
  };

  const parseManifest = (): Record<string, any> | null => {
    try {
      return JSON.parse(manifestText);
    } catch (err: any) {
      // Say where, not just that: a skill is long enough that "invalid JSON"
      // is not actionable.
      toast.error(`The skill is not valid JSON: ${err.message}`);
      return null;
    }
  };

  const save = async () => {
    const manifest = parseManifest();
    if (!manifest) return;
    setSaving(true);
    try {
      if (editing) {
        const updated = await apiClient.updateSandboxSkill(editing.id, manifest);
        toast.success(
          updated.status === 'draft' && editing.status === 'active'
            ? 'Saved — it is a draft again until its control passes'
            : 'Saved'
        );
      } else {
        await apiClient.createSandboxSkill(manifest);
        toast.success('Created as a draft — run its control next');
      }
      setComposing(false);
      setEditing(null);
      await load();
    } catch {
      // The API refuses with the reason and apiClient shows it.
    } finally {
      setSaving(false);
    }
  };

  const draft = async () => {
    const description = wish.trim();
    if (!description) {
      toast.error('Say what the skill should do');
      return;
    }
    // A revision reads from the editor, not from the model's last answer, so a
    // hand edit made between passes is kept.
    let current: Record<string, any> | null = null;
    if (hasDraft) {
      current = parseManifest();
      if (!current) return;
    }
    setDrafting(true);
    setDraftNotes([]);
    setDraftStage(null);
    try {
      const { task_id } = await apiClient.draftSandboxSkill(description, current);
      const started = Date.now();
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const status = await apiClient.getSandboxSkillDraft(task_id);
        setDraftNotes(status.notes || []);
        setDraftStage(
          status.attempt > 0
            ? `${status.stage === 'checking' ? 'Running its control' : 'Writing'} — attempt ${status.attempt}`
            : 'Queued'
        );
        if (!status.pending) {
          if (!status.manifest) {
            toast.error('Could not draft a skill from that — see the notes');
            return;
          }
          setManifestText(JSON.stringify(status.manifest, null, 2));
          setHasDraft(true);
          if (status.dry_run?.ok) {
            toast.success('Drafted, and its control passed — review it before creating');
          } else {
            toast.error('Drafted, but its control has not passed — see the notes');
          }
          return;
        }
        // The task's own limit is 11 minutes; do not spin past it.
        if (Date.now() - started > 12 * 60 * 1000) {
          toast.error('Drafting is taking longer than it should — give up on it');
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    } catch {
      // apiClient surfaces the error.
    } finally {
      setDrafting(false);
      setDraftStage(null);
    }
  };

  const withBusy = async (id: string, work: () => Promise<void>) => {
    setBusy((b) => ({ ...b, [id]: true }));
    try {
      await work();
    } catch {
      // apiClient surfaces the error.
    } finally {
      setBusy((b) => ({ ...b, [id]: false }));
    }
  };

  const runControl = (skill: SandboxSkill) =>
    withBusy(skill.id, async () => {
      const outcome = await apiClient.dryRunSandboxSkill(skill.id);
      if (outcome.ok) toast.success(`${skill.name}: control passed`);
      else toast.error(`${skill.name}: ${outcome.detail}`);
      setExpanded((e) => ({ ...e, [skill.id]: true }));
      await load();
    });

  const activate = (skill: SandboxSkill) =>
    withBusy(skill.id, async () => {
      await apiClient.activateSandboxSkill(skill.id);
      toast.success(`${skill.name} is now offered to your runs`);
      await load();
    });

  const disable = (skill: SandboxSkill) =>
    withBusy(skill.id, async () => {
      await apiClient.disableSandboxSkill(skill.id);
      await load();
    });

  const remove = (skill: SandboxSkill) =>
    withBusy(skill.id, async () => {
      await apiClient.deleteSandboxSkill(skill.id);
      toast.success(`${skill.name} deleted`);
      await load();
    });

  const proposeImage = async () => {
    if (!imageSlug.trim() || !imageDockerfile.trim()) {
      toast.error('An image needs a name and a Dockerfile');
      return;
    }
    setProposing(true);
    try {
      await apiClient.proposeSandboxSkillImage({
        slug: imageSlug.trim(),
        dockerfile: imageDockerfile,
      });
      toast.success('Proposed — an administrator has to build it');
      setImageSlug('');
      setImageDockerfile('');
      await loadImages();
    } catch {
      // The API refuses with the reason and apiClient shows it.
    } finally {
      setProposing(false);
    }
  };

  const buildImage = (image: SandboxSkillImage) =>
    withBusy(image.id, async () => {
      await apiClient.buildSandboxSkillImage(image.id);
      toast.success(`Building ${image.tag}`);
      await loadImages();
    });

  const rejectImage = (image: SandboxSkillImage) =>
    withBusy(image.id, async () => {
      await apiClient.rejectSandboxSkillImage(image.id);
      await loadImages();
    });

  const activeCount = skills.filter((s) => s.status === 'active').length;

  return (
    <div className="mb-6 rounded-lg border border-gray-300 bg-gray-100 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-5 w-5 text-primary-600" />
          <h2 className="text-base font-semibold text-gray-900">Sandbox Skills</h2>
          <span className="text-xs text-gray-500">
            {activeCount} active of {skills.length}
          </span>
        </div>
        {authoringEnabled && (
          <button
            onClick={() => (composing ? setComposing(false) : openComposer(null))}
            className="inline-flex items-center gap-1.5 rounded bg-primary-600 px-3 py-1.5 text-sm text-gray-50 hover:bg-primary-700"
          >
            <Plus className="h-4 w-4" />
            New Skill
          </button>
        )}
      </div>

      <p className="mt-1 text-xs text-gray-600">
        A skill packages one kind of sandboxed work: a procedure the agent
        follows, the image it runs in, and the result it must leave behind. The
        agent decides how to carry it out; the result is checked against what
        the skill declares. An active skill&apos;s result can be required by a
        pipeline stage as <code>skill_&lt;id&gt;</code>.
      </p>

      {!loading && !executionEnabled && (
        <p className="mt-2 flex items-start gap-1.5 rounded border border-amber-300 bg-amber-50 px-2 py-1.5 text-[11px] text-amber-800">
          <AlertTriangle className="mt-0.5 h-3 w-3 flex-shrink-0" />
          <span>
            Sandbox execution is disabled on this server, so no control can run
            and no skill can be activated. Skills can still be written.
          </span>
        </p>
      )}

      {composing && (
        <div className="mt-3 rounded border border-gray-300 bg-white p-3">
          <label className="block text-xs font-medium uppercase tracking-wide text-gray-500">
            {hasDraft ? 'Describe a change' : 'Describe it'}
          </label>
          <div className="mt-1 flex gap-2">
            <input
              value={wish}
              onChange={(e) => setWish(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !drafting) draft();
              }}
              aria-label="Describe the skill"
              placeholder={
                hasDraft
                  ? 'Add a judge that computes the result from the profile'
                  : 'Count instructions per opcode in the LLVM IR of a C kernel'
              }
              className="flex-1 rounded border border-gray-300 bg-white px-2 py-1.5 text-sm"
            />
            <button
              onClick={draft}
              disabled={drafting}
              className="inline-flex flex-shrink-0 items-center gap-1.5 rounded bg-gray-200 px-3 py-1.5 text-sm text-gray-800 hover:bg-gray-300 disabled:opacity-50"
            >
              {drafting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wand2 className="h-4 w-4" />
              )}
              {drafting ? draftStage || 'Drafting…' : hasDraft ? 'Revise' : 'Draft'}
            </button>
          </div>
          <p className="mt-1 text-xs text-gray-500">
            A draft is checked by the same validator that runs on create, and
            its control is run in the sandbox; when either refuses it is asked
            again. Nothing is stored until you press{' '}
            {editing ? 'Save' : 'Create'}.
          </p>

          {draftNotes.length > 0 && (
            <div className="mt-2 rounded border border-amber-300 bg-amber-50 px-2 py-1.5">
              <p className="text-[11px] font-medium text-amber-800">
                What it had to fix, or could not:
              </p>
              <ul className="mt-0.5 space-y-0.5">
                {draftNotes.map((note, idx) => (
                  <li key={idx} className="text-[11px] text-amber-800">
                    {note}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <label className="mt-3 block text-xs font-medium uppercase tracking-wide text-gray-500">
            Skill
          </label>
          <textarea
            value={manifestText}
            onChange={(e) => {
              setManifestText(e.target.value);
              setHasDraft(true);
            }}
            rows={18}
            spellCheck={false}
            aria-label="Skill manifest"
            className="mt-1 w-full rounded border border-gray-300 bg-white px-2 py-1.5 font-mono text-xs"
          />
          <p className="mt-1 text-[11px] text-gray-500">
            {images.length > 0
              ? `image must be one of: ${images.join(', ')}`
              : 'No sandbox image is allowed on this server, so no skill can be created.'}
          </p>
          <div className="mt-2 flex justify-end gap-2">
            <button
              onClick={() => {
                setComposing(false);
                setEditing(null);
              }}
              className="rounded px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-200"
            >
              Cancel
            </button>
            <button
              onClick={save}
              disabled={saving}
              className="rounded bg-primary-600 px-3 py-1.5 text-sm text-gray-50 hover:bg-primary-700 disabled:opacity-50"
            >
              {saving ? 'Saving…' : editing ? 'Save' : 'Create'}
            </button>
          </div>
        </div>
      )}

      {loading ? (
        <p className="mt-3 text-sm text-gray-500">Loading…</p>
      ) : skills.length === 0 ? (
        <p className="mt-3 text-sm text-gray-500">
          No skills yet. Write one, draft one from a description, or let a run
          propose one — each arrives as a draft.
        </p>
      ) : (
        <div className="mt-3 space-y-2">
          {skills.map((skill) => {
            const open = Boolean(expanded[skill.id]);
            const working = Boolean(busy[skill.id]);
            const fields = skill.manifest?.result?.fields || {};
            return (
              <div key={skill.id} className="rounded border border-gray-300 bg-white">
                <div className="flex items-center gap-2 px-3 py-2">
                  <button
                    onClick={() => setExpanded({ ...expanded, [skill.id]: !open })}
                    className="flex min-w-0 flex-1 items-center gap-2 text-left"
                  >
                    {open ? (
                      <ChevronDown className="h-4 w-4 flex-shrink-0 text-gray-500" />
                    ) : (
                      <ChevronRight className="h-4 w-4 flex-shrink-0 text-gray-500" />
                    )}
                    <span className="truncate text-sm font-medium text-gray-900">
                      {skill.name}
                    </span>
                    <span
                      className={`flex-shrink-0 rounded px-1.5 py-0.5 text-[11px] ${
                        STATUS_STYLE[skill.status] || STATUS_STYLE.draft
                      }`}
                    >
                      {skill.status}
                    </span>
                    {skill.origin === 'agent' && (
                      <span
                        className="flex flex-shrink-0 items-center gap-1 rounded bg-amber-100 px-1.5 py-0.5 text-[11px] text-amber-800"
                        title="A run proposed this. Read it before activating."
                      >
                        <Bot className="h-3 w-3" />
                        proposed by a run
                      </span>
                    )}
                    <code className="flex-shrink-0 text-[11px] text-gray-500">
                      {skill.produces}
                    </code>
                  </button>

                  <button
                    onClick={() => runControl(skill)}
                    disabled={working || !executionEnabled}
                    title={
                      executionEnabled
                        ? 'Run the control in the sandbox'
                        : 'Sandbox execution is disabled on this server'
                    }
                    className="flex items-center gap-1 rounded bg-gray-200 px-2 py-1 text-xs text-gray-700 hover:bg-gray-300 disabled:opacity-50"
                  >
                    {working ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <Play className="h-3 w-3" />
                    )}
                    Run control
                  </button>

                  {skill.status === 'active' ? (
                    <button
                      onClick={() => disable(skill)}
                      disabled={working}
                      title="Stop offering it to runs"
                      className="flex items-center gap-1 rounded bg-primary-600 px-2 py-1 text-xs text-gray-50 hover:bg-primary-700"
                    >
                      <Power className="h-3 w-3" />
                      Active
                    </button>
                  ) : skill.verified ? (
                    <button
                      onClick={() => activate(skill)}
                      disabled={working}
                      className="flex items-center gap-1 rounded bg-primary-600 px-2 py-1 text-xs text-gray-50 hover:bg-primary-700"
                    >
                      <Power className="h-3 w-3" />
                      Activate
                    </button>
                  ) : (
                    <span
                      className="rounded bg-gray-200 px-2 py-1 text-xs text-gray-500"
                      title="A skill can be activated once its control has passed against its current content"
                    >
                      Unverified
                    </span>
                  )}

                  {authoringEnabled && (
                    <button
                      onClick={() => openComposer(skill)}
                      title="Edit"
                      className="rounded p-1 text-gray-500 hover:bg-gray-200 hover:text-gray-900"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                  )}
                  <button
                    onClick={() => remove(skill)}
                    disabled={working}
                    title="Delete this skill"
                    className="rounded p-1 text-gray-500 hover:bg-gray-200 hover:text-red-600"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>

                {open && (
                  <div className="space-y-1.5 border-t border-gray-200 px-3 py-2">
                    <p className="text-xs text-gray-600">{skill.description}</p>
                    <p className="text-[11px] text-gray-500">
                      {ORIGIN_LABEL[skill.origin] || skill.origin} · runs in{' '}
                      <code>{skill.manifest?.image}</code> ·{' '}
                      {skill.manifest?.judge_command
                        ? 'result written by the skill’s judge'
                        : 'result written by the run’s own command'}
                    </p>
                    <p className="text-[11px] text-gray-500">
                      result fields:{' '}
                      {Object.entries(fields)
                        .map(([name, kind]) => `${name} (${kind})`)
                        .join(', ')}
                    </p>
                    <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded border border-gray-200 bg-white px-2 py-1.5 font-mono text-[11px] text-gray-700">
                      {skill.manifest?.procedure}
                    </pre>
                    {skill.last_dry_run ? (
                      <DryRunVerdict run={skill.last_dry_run} verified={skill.verified} />
                    ) : (
                      <p className="text-[11px] text-gray-500">
                        Its control has never been run.
                      </p>
                    )}
                    {skill.notes.length > 0 && (
                      <ul className="space-y-0.5">
                        {skill.notes.map((note, idx) => (
                          <li key={idx} className="text-[11px] text-gray-500">
                            {note}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <button
        onClick={() => setShowImages((v) => !v)}
        className="mt-3 flex items-center gap-1.5 text-xs text-gray-600 hover:text-gray-900"
      >
        {showImages ? (
          <ChevronDown className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
        <Box className="h-3.5 w-3.5" />
        Skill images
      </button>

      {showImages && (
        <div className="mt-2 rounded border border-gray-300 bg-white p-3">
          <p className="text-xs text-gray-600">
            For a toolchain none of the allowed images has. An image extends an
            allowed one, is proposed here, and is built only when an
            administrator approves it.
            {!imageBuildEnabled &&
              ' Building is disabled on this deployment, so a proposal will wait.'}
          </p>

          {skillImages.length > 0 && (
            <div className="mt-2 space-y-1.5">
              {skillImages.map((image) => (
                <div
                  key={image.id}
                  className="rounded border border-gray-200 px-2 py-1.5"
                >
                  <div className="flex items-center gap-2">
                    <code className="min-w-0 flex-1 truncate text-xs text-gray-900">
                      {image.tag}
                    </code>
                    <span
                      className={`rounded px-1.5 py-0.5 text-[11px] ${
                        IMAGE_STATUS_STYLE[image.status] || IMAGE_STATUS_STYLE.proposed
                      }`}
                    >
                      {image.status}
                    </span>
                    {isAdmin &&
                      imageBuildEnabled &&
                      (image.status === 'proposed' || image.status === 'failed') && (
                        <>
                          <button
                            onClick={() => buildImage(image)}
                            disabled={Boolean(busy[image.id])}
                            className="rounded bg-primary-600 px-2 py-1 text-xs text-gray-50 hover:bg-primary-700 disabled:opacity-50"
                          >
                            Approve &amp; build
                          </button>
                          <button
                            onClick={() => rejectImage(image)}
                            disabled={Boolean(busy[image.id])}
                            className="rounded bg-gray-200 px-2 py-1 text-xs text-gray-700 hover:bg-gray-300 disabled:opacity-50"
                          >
                            Reject
                          </button>
                        </>
                      )}
                  </div>
                  <p className="mt-0.5 text-[11px] text-gray-500">
                    extends <code>{image.base_image}</code>
                  </p>
                  {image.status === 'failed' && image.build_log && (
                    <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap font-mono text-[10px] text-red-700">
                      {image.build_log}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          )}

          {authoringEnabled && (
            <div className="mt-3">
              <input
                value={imageSlug}
                onChange={(e) => setImageSlug(e.target.value)}
                aria-label="Image name"
                placeholder="z3-solver"
                className="w-full rounded border border-gray-300 bg-white px-2 py-1.5 text-sm"
              />
              <textarea
                value={imageDockerfile}
                onChange={(e) => setImageDockerfile(e.target.value)}
                rows={5}
                spellCheck={false}
                aria-label="Dockerfile"
                placeholder={`FROM ${images[0] || '<an allowed image>'}\nUSER root\nRUN apt-get update && apt-get install -y z3`}
                className="mt-1 w-full rounded border border-gray-300 bg-white px-2 py-1.5 font-mono text-xs"
              />
              <div className="mt-1 flex justify-end">
                <button
                  onClick={proposeImage}
                  disabled={proposing}
                  className="rounded bg-gray-200 px-3 py-1.5 text-sm text-gray-800 hover:bg-gray-300 disabled:opacity-50"
                >
                  {proposing ? 'Proposing…' : 'Propose image'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SandboxSkillsPanel;
