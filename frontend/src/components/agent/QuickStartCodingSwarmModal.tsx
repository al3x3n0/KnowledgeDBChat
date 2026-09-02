/**
 * Quick start: a coding swarm, from one of the three presets.
 *
 * Module scope, like its siblings — declared in the page's render body it was
 * remounted on every page render, resetting the form below. It already took
 * props; the five it used to reach through the closure for are props now too.
 */

import React, { useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';

import {
  findUnsafeQuickStartCommands,
  parseQuickStartCommands,
  parseSafeRelativeFilePaths,
} from '../../pages/autonomousAgentQuickStarts';
import Button from '../common/Button';

export const QuickStartCodingSwarmModal: React.FC<{
  presetKey: 'bug_triage_swarm' | 'build_break_swarm' | 'frontend_regression_swarm';
  title: string;
  description: string;
  defaultName: string;
  defaultFailureSymptom: string;
  defaultGoal: string;
  defaultScope: 'auto' | 'backend' | 'frontend' | 'worker';
  accentClassName: string;
  initialProfileId?: string;
  initialSourceId?: string;
  onClose: () => void;
  submitLabel: string;
  submitMutation: { mutate: (payload: any) => void; isLoading: boolean };
  buildPayload: (payload: any) => any;
  /** The github/gitlab sources a swarm can be pointed at. */
  codeSources: any[];
  /** Saved swarm profiles, and the mutations that maintain them. */
  codingSwarmProfiles: any[];
  createCodingSwarmProfileMutation: any;
  updateCodingSwarmProfileMutation: any;
  deleteCodingSwarmProfileMutation: any;
  /** Who is looking: a saved profile is only editable by the user who owns it. */
  currentUserId: string;
}> = ({
  presetKey,
  title,
  description,
  defaultName,
  defaultFailureSymptom,
  defaultGoal,
  defaultScope,
  accentClassName,
  initialProfileId,
  initialSourceId,
  onClose,
  submitLabel,
  submitMutation,
  buildPayload,
  codeSources,
  codingSwarmProfiles,
  createCodingSwarmProfileMutation,
  updateCodingSwarmProfileMutation,
  deleteCodingSwarmProfileMutation,
  currentUserId,
}) => {
  const MAX_QS_COMMANDS = 6;
  const MAX_QS_FILE_PATHS = 32;
  const [name, setName] = useState(defaultName);
  const [failureSymptom, setFailureSymptom] = useState(defaultFailureSymptom);
  const [goal, setGoal] = useState(defaultGoal);
  const [selectedSourceId, setSelectedSourceId] = useState<string>('');
  const [scope, setScope] = useState<'auto' | 'backend' | 'frontend' | 'worker'>(defaultScope);
  const [searchQuery, setSearchQuery] = useState('');
  const [commandsText, setCommandsText] = useState('');
  const [filePathsText, setFilePathsText] = useState('');
  const [errorOutput, setErrorOutput] = useState('');
  const [maxAgents, setMaxAgents] = useState(4);
  const [selectedProfileId, setSelectedProfileId] = useState(initialProfileId || '');
  const [saveProfile, setSaveProfile] = useState(false);
  const [saveAsNewProfile, setSaveAsNewProfile] = useState(false);
  const [saveProfileTitle, setSaveProfileTitle] = useState('');

  const matchingProfiles = useMemo(
    () =>
      codingSwarmProfiles.filter((profile) => {
        if (String(profile.preset_key || '') !== presetKey) return false;
        if (!selectedSourceId) return true;
        return String(profile.source_id || '') === String(selectedSourceId);
      }),
    [codingSwarmProfiles, presetKey, selectedSourceId]
  );
  const selectedProfile = useMemo(
    () => matchingProfiles.find((profile) => String(profile.id) === String(selectedProfileId)) || null,
    [matchingProfiles, selectedProfileId]
  );
  const selectedProfileOwnedByCurrentUser = String(selectedProfile?.user_id || '') === currentUserId;

  useEffect(() => {
    if (!selectedSourceId && initialSourceId) {
      setSelectedSourceId(String(initialSourceId));
      return;
    }
    if (!selectedSourceId && codeSources.length > 0) {
      setSelectedSourceId(String((codeSources[0] as any)?.id || ''));
    }
  }, [codeSources, selectedSourceId, initialSourceId]);

  useEffect(() => {
    if (initialProfileId && matchingProfiles.some((profile) => String(profile.id) === String(initialProfileId)) && !selectedProfileId) {
      setSelectedProfileId(String(initialProfileId));
      return;
    }
    if (matchingProfiles.length > 0 && !selectedProfileId) {
      const preferred = matchingProfiles.find((profile) => profile.is_default) || matchingProfiles[0];
      setSelectedProfileId(String(preferred?.id || ''));
    }
    if (matchingProfiles.length === 0 && selectedProfileId) {
      setSelectedProfileId('');
    }
  }, [matchingProfiles, selectedProfileId, initialProfileId]);

  useEffect(() => {
    if (!selectedProfile) return;
    if (String(selectedProfile.source_id || '') !== String(selectedSourceId || '')) {
      setSelectedSourceId(String(selectedProfile.source_id || ''));
    }
    if (selectedProfile.scope_default) setScope(selectedProfile.scope_default as any);
    if (selectedProfile.saved_search_query) setSearchQuery(String(selectedProfile.saved_search_query));
    if (Array.isArray(selectedProfile.default_commands)) setCommandsText(selectedProfile.default_commands.join('\n'));
    if (Array.isArray(selectedProfile.default_file_paths)) setFilePathsText(selectedProfile.default_file_paths.join('\n'));
    if (selectedProfile.max_agents) setMaxAgents(Math.max(1, Math.min(Number(selectedProfile.max_agents || 4), 4)));
    if (!saveProfileTitle.trim()) {
      setSaveProfileTitle(String(selectedProfile.title || '').trim());
    }
  }, [selectedProfile, selectedSourceId, saveProfileTitle]);

  useEffect(() => {
    if (!selectedProfileId) {
      setSaveAsNewProfile(false);
    }
  }, [selectedProfileId]);
  useEffect(() => {
    if (selectedProfile && !selectedProfileOwnedByCurrentUser) {
      setSaveAsNewProfile(true);
    }
  }, [selectedProfile, selectedProfileOwnedByCurrentUser]);

  const commandsPreview = useMemo(
    () => parseQuickStartCommands(commandsText, MAX_QS_COMMANDS),
    [commandsText]
  );
  const unsafeCommandsPreview = useMemo(
    () => findUnsafeQuickStartCommands(commandsPreview),
    [commandsPreview]
  );
  const filePathsParsed = useMemo(
    () => parseSafeRelativeFilePaths(filePathsText, MAX_QS_FILE_PATHS),
    [filePathsText]
  );
  const filePathsPreview = filePathsParsed.items;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!failureSymptom.trim() && !goal.trim()) {
      toast.error('Failure symptom or goal is required');
      return;
    }
    if (!selectedSourceId) {
      toast.error('Target code source is required');
      return;
    }
    if (unsafeCommandsPreview.length > 0) {
      toast.error(`Blocked unsafe command(s): ${unsafeCommandsPreview.join(' | ')}`);
      return;
    }
    if (saveProfile && !saveProfileTitle.trim()) {
      toast.error('Profile title is required to save this preset');
      return;
    }
    let profileId = selectedProfileId || undefined;
    if (saveProfile) {
      const profilePayload = {
        title: saveProfileTitle.trim(),
        source_id: selectedSourceId,
        preset_key: presetKey,
        description: selectedProfile?.description || undefined,
        scope_default: scope,
        default_commands: commandsPreview.length ? commandsPreview : undefined,
        default_file_paths: filePathsPreview.length ? filePathsPreview : undefined,
        max_agents: maxAgents,
        saved_search_query: searchQuery.trim() || undefined,
        safe_command_policy: selectedProfile?.safe_command_policy || 'standard',
        visibility: selectedProfile?.visibility || 'private',
        shared_with_user_ids: selectedProfile?.shared_with_user_ids || [],
        is_default: selectedProfileId && !saveAsNewProfile
          ? Boolean(selectedProfile?.is_default)
          : matchingProfiles.length === 0,
      };
      if (selectedProfileId && !saveAsNewProfile && selectedProfileOwnedByCurrentUser) {
        await updateCodingSwarmProfileMutation.mutateAsync({
          profileId: String(selectedProfileId),
          data: {
            title: profilePayload.title,
            preset_key: profilePayload.preset_key,
            description: profilePayload.description,
            scope_default: profilePayload.scope_default,
            default_commands: profilePayload.default_commands,
            default_file_paths: profilePayload.default_file_paths,
            max_agents: profilePayload.max_agents,
            saved_search_query: profilePayload.saved_search_query,
            safe_command_policy: profilePayload.safe_command_policy,
            visibility: profilePayload.visibility,
            shared_with_user_ids: profilePayload.shared_with_user_ids,
            is_default: profilePayload.is_default,
          },
        });
        profileId = String(selectedProfileId);
      } else {
        const created = await createCodingSwarmProfileMutation.mutateAsync(profilePayload);
        profileId = String((created as any)?.id || '').trim() || undefined;
      }
    }
    submitMutation.mutate(
      buildPayload({
        name,
        goal,
        failureSymptom,
        selectedSourceId,
        scope,
        searchQuery,
        commandsText,
        filePathsText,
        errorOutput,
        maxAgents,
        maxCommands: MAX_QS_COMMANDS,
        maxFilePaths: MAX_QS_FILE_PATHS,
        profileId,
      })
    );
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-lg font-semibold mb-1">{title}</h2>
          <p className="text-sm text-gray-500 mb-4">{description}</p>
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
              <label className="block text-sm font-medium text-gray-700 mb-1">Failure symptom</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={4}
                value={failureSymptom}
                onChange={(e) => setFailureSymptom(e.target.value)}
                placeholder="Saving a document returns 500 and leaves the spinner running"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Desired outcome (optional)</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={3}
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                placeholder="Fix the regression without changing successful save flows"
              />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Target code source</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={selectedSourceId}
                  onChange={(e) => {
                    setSelectedSourceId(e.target.value);
                    setSelectedProfileId('');
                  }}
                >
                  <option value="">Select a git source…</option>
                  {codeSources.map((s: any) => (
                    <option key={String(s.id)} value={String(s.id)}>
                      {String(s.name || s.id)}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Scope profile</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={scope}
                  onChange={(e) => setScope(e.target.value as any)}
                >
                  <option value="auto">Auto</option>
                  <option value="backend">Backend</option>
                  <option value="frontend">Frontend</option>
                  <option value="worker">Worker</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Max agents</label>
                <input
                  type="number"
                  min={1}
                  max={4}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={maxAgents}
                  onChange={(e) => setMaxAgents(Math.max(1, Math.min(parseInt(e.target.value || '4', 10), 4)))}
                />
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-1">Saved repo profile (optional)</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={selectedProfileId}
                  onChange={(e) => setSelectedProfileId(e.target.value)}
                >
                  <option value="">No saved profile</option>
                  {matchingProfiles.map((profile) => (
                    <option key={String(profile.id)} value={String(profile.id)}>
                      {profile.title}{profile.is_default ? ' (default)' : ''}{String(profile.visibility || 'private').toLowerCase() === 'shared' ? ' (shared)' : ''}{String(profile.status || '').toLowerCase() !== 'active' ? ` (${profile.status})` : ''}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-end">
                <Button
                  type="button"
                  variant="ghost"
                  className="w-full"
                  disabled={!selectedProfileId || deleteCodingSwarmProfileMutation.isLoading || !selectedProfileOwnedByCurrentUser}
                  onClick={() => deleteCodingSwarmProfileMutation.mutate(String(selectedProfileId))}
                >
                  Delete profile
                </Button>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Search Query (optional)</label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Leave empty to derive from scope + symptom"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Commands (optional, one per line) {commandsPreview.length}/{MAX_QS_COMMANDS}
              </label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={4}
                value={commandsText}
                onChange={(e) => setCommandsText(e.target.value)}
                placeholder={'python -m pytest -q backend/tests\nCI=true npm --prefix frontend test -- --watchAll=false'}
              />
              {unsafeCommandsPreview.length > 0 && (
                <p className="mt-1 text-xs text-rose-700">
                  Unsafe command(s) blocked: {unsafeCommandsPreview.join(' | ')}
                </p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Focused file paths (optional, one per line) {filePathsPreview.length}/{MAX_QS_FILE_PATHS}
              </label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={3}
                value={filePathsText}
                onChange={(e) => setFilePathsText(e.target.value)}
                placeholder={'frontend/src/pages/DocumentsPage.tsx\nbackend/app/api/endpoints/documents.py'}
              />
              {filePathsParsed.droppedUnsafe > 0 && (
                <p className="mt-1 text-xs text-amber-700">
                  {filePathsParsed.droppedUnsafe} unsafe path row(s) ignored.
                </p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Error output (optional)</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={4}
                value={errorOutput}
                onChange={(e) => setErrorOutput(e.target.value)}
                placeholder="Paste stack trace, failing assertion, or representative logs"
              />
            </div>
            <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-3 text-xs text-gray-700 space-y-2">
              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  className="rounded border-gray-300"
                  checked={saveProfile}
                  onChange={(e) => setSaveProfile(Boolean(e.target.checked))}
                />
                Save this launch configuration as a repo profile
              </label>
              {saveProfile && (
                <div className="space-y-2">
                  <input
                    type="text"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={saveProfileTitle}
                    onChange={(e) => setSaveProfileTitle(e.target.value)}
                    placeholder="Profile title"
                  />
                  {selectedProfileId ? (
                    <label className="inline-flex items-center gap-2">
                      <input
                        type="checkbox"
                        className="rounded border-gray-300"
                        checked={saveAsNewProfile}
                        onChange={(e) => setSaveAsNewProfile(Boolean(e.target.checked))}
                        disabled={!selectedProfileOwnedByCurrentUser}
                      />
                      Save as new profile instead of updating the selected one
                    </label>
                  ) : null}
                  {selectedProfileId && !selectedProfileOwnedByCurrentUser ? (
                    <p className="text-[11px] text-gray-500">
                      Shared profiles are read-only here. Launching with save enabled will create a new profile for you.
                    </p>
                  ) : null}
                  {selectedProfileId && !saveAsNewProfile && selectedProfileOwnedByCurrentUser ? (
                    <p className="text-[11px] text-gray-500">
                      Saving will update the selected profile before launch.
                    </p>
                  ) : null}
                </div>
              )}
            </div>
            <div className={`rounded-lg border px-3 py-2 text-xs ${accentClassName}`}>
              The swarm can auto-promote the strongest slice into the existing repair loop, while unresolved paths can be sent to backlog with preserved lineage.
            </div>
            <div className="flex justify-end gap-3 pt-4 border-t">
              <Button type="button" variant="secondary" onClick={onClose}>
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={submitMutation.isLoading || createCodingSwarmProfileMutation.isLoading || codeSources.length === 0 || unsafeCommandsPreview.length > 0}
                title={
                  codeSources.length === 0
                    ? 'Add a GitHub/GitLab source first'
                    : unsafeCommandsPreview.length > 0
                      ? 'Remove unsafe commands before starting'
                      : undefined
                }
              >
                {submitMutation.isLoading ? 'Starting...' : submitLabel}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default QuickStartCodingSwarmModal;
