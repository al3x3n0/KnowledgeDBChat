import type {
  AnyMutation,
  QueryClient,
  SetActiveTab,
} from '../propTypes';
import Button from '../../../components/common/Button';
import type {
  User,
  CodingSwarmProfileUpdate,
  CollaborationSummary,
  CodingSwarmProfile,
} from '../../../types';
import CollaborationSummaryPanel from '../CollaborationSummaryPanel';
import {
  RefreshCw,
} from 'lucide-react';
import {
  parseQuickStartCommands,
  parseSafeRelativeFilePaths,
} from '../../../pages/autonomousAgentQuickStarts';
import React, { useCallback, useMemo } from 'react';

const codingSwarmPresetLabel = (presetKey?: string | null) => {
  const normalized = String(presetKey || '').trim().toLowerCase();
  if (normalized === 'build_break_swarm') return 'Build Break Swarm';
  if (normalized === 'frontend_regression_swarm') return 'Frontend Regression Swarm';
  return 'Bug Triage Swarm';
};

export interface SwarmProfilesTabProps {
  setActiveTab: SetActiveTab;
  setCodingSwarmLaunchSeed: React.Dispatch<React.SetStateAction<{ presetKey: string; profileId?: string; sourceId?: string } | null>>;
  setShowBugTriageSwarmQuickStartModal: React.Dispatch<React.SetStateAction<boolean>>;
  setShowBuildBreakSwarmQuickStartModal: React.Dispatch<React.SetStateAction<boolean>>;
  setShowFrontendRegressionSwarmQuickStartModal: React.Dispatch<React.SetStateAction<boolean>>;
  user: User | null;
  codeSources: any[];
  codingSwarmProfileDraft: CodingSwarmProfileUpdate & { source_id?: string; duplicate_mode?: boolean; title: string };
  setCodingSwarmProfileDraft: React.Dispatch<React.SetStateAction<CodingSwarmProfileUpdate & { source_id?: string; duplicate_mode?: boolean; title: string }>>;
  codingSwarmProfiles: CodingSwarmProfile[];
  collaborationUsers: User[];
  createCodingSwarmProfileMutation: AnyMutation;
  deleteCodingSwarmProfileMutation: AnyMutation;
  editingCodingSwarmProfileId: string;
  setEditingCodingSwarmProfileId: React.Dispatch<React.SetStateAction<string>>;
  profileDefaultOnly: boolean;
  setProfileDefaultOnly: React.Dispatch<React.SetStateAction<boolean>>;
  profileOwnerFilter: string;
  setProfileOwnerFilter: React.Dispatch<React.SetStateAction<string>>;
  profileOwnershipFilter: string;
  setProfileOwnershipFilter: React.Dispatch<React.SetStateAction<string>>;
  profilePresetFilter: string;
  setProfilePresetFilter: React.Dispatch<React.SetStateAction<string>>;
  profileSourceFilter: string;
  setProfileSourceFilter: React.Dispatch<React.SetStateAction<string>>;
  profileStatusFilter: string;
  setProfileStatusFilter: React.Dispatch<React.SetStateAction<string>>;
  profileVisibilityFilter: string;
  setProfileVisibilityFilter: React.Dispatch<React.SetStateAction<string>>;
  queryClient: QueryClient;
  updateCodingSwarmProfileMutation: AnyMutation;
  userLabelById: (candidateId?: string | null) => string;
}

export const SwarmProfilesTab: React.FC<SwarmProfilesTabProps> = ({
  setActiveTab,
  setCodingSwarmLaunchSeed,
  setShowBugTriageSwarmQuickStartModal,
  setShowBuildBreakSwarmQuickStartModal,
  setShowFrontendRegressionSwarmQuickStartModal,
  user,
  codeSources,
  codingSwarmProfileDraft,
  setCodingSwarmProfileDraft,
  codingSwarmProfiles,
  collaborationUsers,
  createCodingSwarmProfileMutation,
  deleteCodingSwarmProfileMutation,
  editingCodingSwarmProfileId,
  setEditingCodingSwarmProfileId,
  profileDefaultOnly,
  setProfileDefaultOnly,
  profileOwnerFilter,
  setProfileOwnerFilter,
  profileOwnershipFilter,
  setProfileOwnershipFilter,
  profilePresetFilter,
  setProfilePresetFilter,
  profileSourceFilter,
  setProfileSourceFilter,
  profileStatusFilter,
  setProfileStatusFilter,
  profileVisibilityFilter,
  setProfileVisibilityFilter,
  queryClient,
  updateCodingSwarmProfileMutation,
  userLabelById,
}) => {
  const codeSourceById = useMemo(
    () =>
      Object.fromEntries(
        codeSources.map((source: any) => [String(source.id), source] as const)
      ) as Record<string, any>,
    [codeSources]
  );

  const filteredCodingSwarmProfiles = useMemo(
    () =>
      codingSwarmProfiles.filter((profile) => {
        if (profilePresetFilter && String(profile.preset_key || '') !== profilePresetFilter) return false;
        if (profileSourceFilter && String(profile.source_id || '') !== profileSourceFilter) return false;
        if (profileStatusFilter && String(profile.status || '').toLowerCase() !== profileStatusFilter) return false;
        if (profileDefaultOnly && !profile.is_default) return false;
        if (profileVisibilityFilter && String(profile.visibility || 'private').toLowerCase() !== profileVisibilityFilter) return false;
        if (profileOwnershipFilter === 'mine' && String(profile.user_id || '') !== String(user?.id || '')) return false;
        if (profileOwnershipFilter === 'shared' && String(profile.user_id || '') === String(user?.id || '')) return false;
        if (profileOwnerFilter && String(profile.user_id || '') !== profileOwnerFilter) return false;
        return true;
      }),
    [codingSwarmProfiles, profilePresetFilter, profileSourceFilter, profileStatusFilter, profileDefaultOnly, profileVisibilityFilter, profileOwnershipFilter, profileOwnerFilter, user]
  );

  const openCodingSwarmProfileEditor = useCallback((profile?: CodingSwarmProfile | null, options?: { duplicate?: boolean }) => {
    const duplicate = Boolean(options?.duplicate);
    setEditingCodingSwarmProfileId(duplicate ? '' : String(profile?.id || ''));
    setCodingSwarmProfileDraft({
      title: duplicate
        ? `${String(profile?.title || 'Coding Swarm Profile').trim()} Copy`
        : String(profile?.title || '').trim(),
      source_id: String(profile?.source_id || codeSources[0]?.id || ''),
      preset_key: String(profile?.preset_key || 'bug_triage_swarm'),
      description: String(profile?.description || ''),
      scope_default: String(profile?.scope_default || 'auto'),
      default_commands: Array.isArray(profile?.default_commands) ? [...profile!.default_commands] : [],
      default_file_paths: Array.isArray(profile?.default_file_paths) ? [...profile!.default_file_paths] : [],
      max_agents: Math.max(1, Math.min(Number(profile?.max_agents || 4), 4)),
      safe_command_policy: String(profile?.safe_command_policy || 'standard'),
      saved_search_query: String(profile?.saved_search_query || ''),
      is_default: duplicate ? false : Boolean(profile?.is_default),
      status: String(profile?.status || 'active'),
      visibility: String(profile?.visibility || 'private'),
      shared_with_user_ids: Array.isArray(profile?.shared_with_user_ids) ? [...profile.shared_with_user_ids] : [],
      profile_metadata: (profile?.profile_metadata && typeof profile.profile_metadata === 'object') ? profile.profile_metadata : {},
      duplicate_mode: duplicate,
    });
    setActiveTab('profiles');
  }, [codeSources, setActiveTab, setCodingSwarmProfileDraft, setEditingCodingSwarmProfileId]);

  const closeCodingSwarmProfileEditor = useCallback(() => {
    setEditingCodingSwarmProfileId('');
    setCodingSwarmProfileDraft({
      title: '',
      source_id: '',
      preset_key: 'bug_triage_swarm',
      description: '',
      scope_default: 'auto',
      default_commands: [],
      default_file_paths: [],
      max_agents: 4,
      safe_command_policy: 'standard',
      saved_search_query: '',
      is_default: false,
      status: 'active',
      visibility: 'private',
      shared_with_user_ids: [],
      profile_metadata: {},
      duplicate_mode: false,
    });
  }, [setCodingSwarmProfileDraft, setEditingCodingSwarmProfileId]);

  return (
    <div className="w-full flex flex-col min-h-0 gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Coding Swarm Profiles</h2>
          <p className="text-sm text-gray-500">
            Save, edit, duplicate, and launch repo-scoped coding swarm presets for repeat triage work.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={() => queryClient.invalidateQueries(['coding-swarm-profiles'])}>
            <RefreshCw className="w-4 h-4 mr-1" />
            Refresh
          </Button>
          <Button
            size="sm"
            variant="primary"
            onClick={() => openCodingSwarmProfileEditor(null)}
          >
            New profile
          </Button>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-3">
        <div className="flex flex-wrap gap-3 items-center">
          <select
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
            value={profileOwnershipFilter}
            onChange={(e) => setProfileOwnershipFilter(e.target.value)}
          >
            <option value="">Mine + shared</option>
            <option value="mine">Mine</option>
            <option value="shared">Shared with me</option>
          </select>
          <select
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
            value={profilePresetFilter}
            onChange={(e) => setProfilePresetFilter(e.target.value)}
          >
            <option value="">All presets</option>
            <option value="bug_triage_swarm">Bug Triage</option>
            <option value="build_break_swarm">Build Break</option>
            <option value="frontend_regression_swarm">Frontend Regression</option>
          </select>
          <select
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
            value={profileSourceFilter}
            onChange={(e) => setProfileSourceFilter(e.target.value)}
          >
            <option value="">All repos</option>
            {codeSources.map((source: any) => (
              <option key={String(source.id)} value={String(source.id)}>
                {String(source.name || source.id)}
              </option>
            ))}
          </select>
          <select
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
            value={profileStatusFilter}
            onChange={(e) => setProfileStatusFilter(e.target.value)}
          >
            <option value="">Any status</option>
            <option value="active">Active</option>
            <option value="disabled">Disabled</option>
          </select>
          <select
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
            value={profileVisibilityFilter}
            onChange={(e) => setProfileVisibilityFilter(e.target.value)}
          >
            <option value="">Any visibility</option>
            <option value="private">Private</option>
            <option value="shared">Shared</option>
          </select>
          <select
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
            value={profileOwnerFilter}
            onChange={(e) => setProfileOwnerFilter(e.target.value)}
          >
            <option value="">Any owner</option>
            {collaborationUsers.map((candidate) => (
              <option key={String(candidate.id)} value={String(candidate.id)}>
                {userLabelById(String(candidate.id))}
              </option>
            ))}
          </select>
          <label className="inline-flex items-center gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              className="rounded border-gray-300"
              checked={profileDefaultOnly}
              onChange={(e) => setProfileDefaultOnly(e.target.checked)}
            />
            Default only
          </label>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 min-h-0">
        <div className="xl:col-span-2 bg-white border border-gray-200 rounded-lg p-4 min-h-0">
          <div className="flex items-center justify-between mb-3">
            <h3 className="section-heading">Saved Profiles</h3>
            <div className="text-xs text-gray-500">{filteredCodingSwarmProfiles.length} profiles</div>
          </div>
          <div className="space-y-3 max-h-[42rem] overflow-y-auto pr-1">
            {filteredCodingSwarmProfiles.length === 0 ? (
              <div className="text-sm text-gray-500">No coding swarm profiles match the current filters.</div>
            ) : (
              filteredCodingSwarmProfiles.map((profile) => {
                const sourceLabel = String(codeSourceById[String(profile.source_id || '')]?.name || profile.source_id || '').trim();
                const isOwner = String(profile.user_id || '') === String(user?.id || '');
                const profileCollaborationSummary = ((profile.collaboration_summary && typeof profile.collaboration_summary === 'object')
                  ? profile.collaboration_summary
                  : {}) as CollaborationSummary;
                return (
                  <div key={String(profile.id)} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <div className="font-medium text-gray-900">{profile.title}</div>
                          <span className="text-xs px-2 py-1 rounded bg-rose-50 text-rose-700 border border-rose-100">
                            {codingSwarmPresetLabel(profile.preset_key)}
                          </span>
                          <span className={`text-xs px-2 py-1 rounded ${String(profile.status || '').toLowerCase() === 'active' ? 'bg-emerald-50 text-emerald-700 border border-emerald-100' : 'bg-gray-200 text-gray-700 border border-gray-200'}`}>
                            {String(profile.status || 'active')}
                          </span>
                          <span className={`text-xs px-2 py-1 rounded ${String(profile.visibility || 'private').toLowerCase() === 'shared' ? 'bg-cyan-50 text-cyan-700 border border-cyan-100' : 'bg-gray-200 text-gray-700 border border-gray-200'}`}>
                            {String(profile.visibility || 'private')}
                          </span>
                          {profile.is_default ? (
                            <span className="text-xs px-2 py-1 rounded bg-amber-50 text-amber-700 border border-amber-100">Default</span>
                          ) : null}
                        </div>
                        {profile.description ? (
                          <div className="mt-1 text-sm text-gray-600">{String(profile.description)}</div>
                        ) : null}
                        <CollaborationSummaryPanel
                          summary={profileCollaborationSummary}
                          fallbackOwnerId={String(profile.user_id || '')}
                          fallbackVisibility={String(profile.visibility || 'private')}
                          fallbackSharedWithUserIds={Array.isArray(profile.shared_with_user_ids) ? profile.shared_with_user_ids : []}
                          userLabelById={userLabelById}
                        />
                        <div className="mt-2 text-xs text-gray-500 flex flex-wrap gap-3">
                          <span>Repo {sourceLabel}</span>
                          <span>Scope {String(profile.scope_default || 'auto')}</span>
                          <span>Agents {Number(profile.max_agents || 4)}</span>
                          <span>Policy {String(profile.safe_command_policy || 'standard')}</span>
                          {profile.saved_search_query ? <span>Query saved</span> : null}
                          <span>Updated {new Date(profile.updated_at).toLocaleDateString()}</span>
                        </div>
                        {(profile.default_commands?.length || profile.default_file_paths?.length) ? (
                          <div className="mt-2 text-xs text-gray-500">
                            {profile.default_commands?.length ? `Commands ${profile.default_commands.length}` : 'No commands'}
                            {' · '}
                            {profile.default_file_paths?.length ? `Files ${profile.default_file_paths.length}` : 'No files'}
                          </div>
                        ) : null}
                      </div>
                      <div className="flex flex-wrap gap-2 shrink-0">
                        <Button size="sm" variant="ghost" onClick={() => openCodingSwarmProfileEditor(profile)} disabled={!isOwner}>
                          Edit
                        </Button>
                        <Button size="sm" variant="ghost" onClick={() => openCodingSwarmProfileEditor(profile, { duplicate: true })}>
                          Duplicate
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() =>
                            updateCodingSwarmProfileMutation.mutate({
                              profileId: String(profile.id),
                              data: { is_default: true, status: 'active' },
                            })
                          }
                          disabled={updateCodingSwarmProfileMutation.isLoading || profile.is_default || !isOwner}
                        >
                          Set default
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() =>
                            updateCodingSwarmProfileMutation.mutate({
                              profileId: String(profile.id),
                              data: { status: String(profile.status || '').toLowerCase() === 'active' ? 'disabled' : 'active' },
                            })
                          }
                          disabled={updateCodingSwarmProfileMutation.isLoading || !isOwner}
                        >
                          {String(profile.status || '').toLowerCase() === 'active' ? 'Disable' : 'Enable'}
                        </Button>
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => {
                            setCodingSwarmLaunchSeed({
                              presetKey: String(profile.preset_key || ''),
                              profileId: String(profile.id),
                              sourceId: String(profile.source_id || ''),
                            });
                            if (String(profile.preset_key || '') === 'build_break_swarm') setShowBuildBreakSwarmQuickStartModal(true);
                            else if (String(profile.preset_key || '') === 'frontend_regression_swarm') setShowFrontendRegressionSwarmQuickStartModal(true);
                            else setShowBugTriageSwarmQuickStartModal(true);
                          }}
                        >
                          Launch
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => deleteCodingSwarmProfileMutation.mutate(String(profile.id))}
                          disabled={deleteCodingSwarmProfileMutation.isLoading || !isOwner}
                        >
                          Delete
                        </Button>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="section-heading">
              {editingCodingSwarmProfileId ? 'Edit Profile' : codingSwarmProfileDraft.duplicate_mode ? 'Duplicate Profile' : 'New Profile'}
            </h3>
            {(editingCodingSwarmProfileId || codingSwarmProfileDraft.title) ? (
              <Button size="sm" variant="ghost" onClick={closeCodingSwarmProfileEditor}>
                Clear
              </Button>
            ) : null}
          </div>
          <div className="space-y-3">
            <input
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              placeholder="Profile title"
              value={codingSwarmProfileDraft.title}
              onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, title: e.target.value }))}
            />
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              rows={3}
              placeholder="Description"
              value={String(codingSwarmProfileDraft.description || '')}
              onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, description: e.target.value }))}
            />
            <select
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              value={String(codingSwarmProfileDraft.source_id || '')}
              onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, source_id: e.target.value }))}
              disabled={Boolean(editingCodingSwarmProfileId && !codingSwarmProfileDraft.duplicate_mode)}
            >
              <option value="">Select repo source</option>
              {codeSources.map((source: any) => (
                <option key={String(source.id)} value={String(source.id)}>
                  {String(source.name || source.id)}
                </option>
              ))}
            </select>
            <div className="grid grid-cols-2 gap-3">
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={String(codingSwarmProfileDraft.preset_key || 'bug_triage_swarm')}
                onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, preset_key: e.target.value }))}
              >
                <option value="bug_triage_swarm">Bug Triage Swarm</option>
                <option value="build_break_swarm">Build Break Swarm</option>
                <option value="frontend_regression_swarm">Frontend Regression Swarm</option>
              </select>
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={String(codingSwarmProfileDraft.scope_default || 'auto')}
                onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, scope_default: e.target.value }))}
              >
                <option value="auto">Auto scope</option>
                <option value="backend">Backend</option>
                <option value="frontend">Frontend</option>
                <option value="worker">Worker</option>
              </select>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <input
                type="number"
                min={1}
                max={4}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={Number(codingSwarmProfileDraft.max_agents || 4)}
                onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, max_agents: Math.max(1, Math.min(parseInt(e.target.value || '4', 10), 4)) }))}
              />
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={String(codingSwarmProfileDraft.safe_command_policy || 'standard')}
                onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, safe_command_policy: e.target.value }))}
              >
                <option value="standard">Standard</option>
              </select>
            </div>
            <input
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              placeholder="Saved search query"
              value={String(codingSwarmProfileDraft.saved_search_query || '')}
              onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, saved_search_query: e.target.value }))}
            />
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
              rows={3}
              placeholder="Default commands, one per line"
              value={Array.isArray(codingSwarmProfileDraft.default_commands) ? codingSwarmProfileDraft.default_commands.join('\n') : ''}
              onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, default_commands: parseQuickStartCommands(e.target.value, 8) }))}
            />
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
              rows={3}
              placeholder="Default file paths, one per line"
              value={Array.isArray(codingSwarmProfileDraft.default_file_paths) ? codingSwarmProfileDraft.default_file_paths.join('\n') : ''}
              onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, default_file_paths: parseSafeRelativeFilePaths(e.target.value, 16).items }))}
            />
            <div className="flex items-center justify-between gap-3">
              <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  className="rounded border-gray-300"
                  checked={Boolean(codingSwarmProfileDraft.is_default)}
                  onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, is_default: e.target.checked }))}
                />
                Make default
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  className="rounded border-gray-300"
                  checked={String(codingSwarmProfileDraft.status || 'active').toLowerCase() === 'active'}
                  onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, status: e.target.checked ? 'active' : 'disabled' }))}
                />
                Active
              </label>
            </div>
            <div className="space-y-2 border border-gray-200 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-700">Sharing</div>
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={String(codingSwarmProfileDraft.visibility || 'private')}
                onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, visibility: e.target.value }))}
              >
                <option value="private">Private</option>
                <option value="shared">Shared</option>
              </select>
              {String(codingSwarmProfileDraft.visibility || 'private') === 'shared' ? (
                <div className="grid grid-cols-1 gap-1 max-h-40 overflow-y-auto">
                  {collaborationUsers
                    .filter((candidate) => String(candidate.id) !== String(user?.id || ''))
                    .map((candidate) => {
                      const candidateId = String(candidate.id);
                      const checked = Array.isArray(codingSwarmProfileDraft.shared_with_user_ids) && codingSwarmProfileDraft.shared_with_user_ids.includes(candidateId);
                      return (
                        <label key={candidateId} className="inline-flex items-center gap-2 text-sm text-gray-700">
                          <input
                            type="checkbox"
                            className="rounded border-gray-300"
                            checked={checked}
                            onChange={(e) =>
                              setCodingSwarmProfileDraft((prev) => {
                                const current = Array.isArray(prev.shared_with_user_ids) ? prev.shared_with_user_ids : [];
                                return {
                                  ...prev,
                                  shared_with_user_ids: e.target.checked
                                    ? Array.from(new Set([...current, candidateId]))
                                    : current.filter((value) => value !== candidateId),
                                };
                              })
                            }
                          />
                          {userLabelById(candidateId)}
                        </label>
                      );
                    })}
                </div>
              ) : null}
            </div>
            <div className="flex gap-2 pt-2">
              <Button
                variant="primary"
                disabled={
                  (!codingSwarmProfileDraft.title || !String(codingSwarmProfileDraft.source_id || '').trim()) ||
                  createCodingSwarmProfileMutation.isLoading ||
                  updateCodingSwarmProfileMutation.isLoading
                }
                onClick={async () => {
                  const payload = {
                    title: String(codingSwarmProfileDraft.title || '').trim(),
                    source_id: String(codingSwarmProfileDraft.source_id || '').trim(),
                    preset_key: String(codingSwarmProfileDraft.preset_key || 'bug_triage_swarm').trim(),
                    description: String(codingSwarmProfileDraft.description || '').trim() || undefined,
                    scope_default: String(codingSwarmProfileDraft.scope_default || 'auto').trim() || 'auto',
                    default_commands: Array.isArray(codingSwarmProfileDraft.default_commands) ? codingSwarmProfileDraft.default_commands : [],
                    default_file_paths: Array.isArray(codingSwarmProfileDraft.default_file_paths) ? codingSwarmProfileDraft.default_file_paths : [],
                    max_agents: Math.max(1, Math.min(Number(codingSwarmProfileDraft.max_agents || 4), 4)),
                    safe_command_policy: String(codingSwarmProfileDraft.safe_command_policy || 'standard').trim() || 'standard',
                    saved_search_query: String(codingSwarmProfileDraft.saved_search_query || '').trim() || undefined,
                    is_default: Boolean(codingSwarmProfileDraft.is_default),
                    status: String(codingSwarmProfileDraft.status || 'active').trim() || 'active',
                    visibility: String(codingSwarmProfileDraft.visibility || 'private').trim() || 'private',
                    shared_with_user_ids: Array.isArray(codingSwarmProfileDraft.shared_with_user_ids) ? codingSwarmProfileDraft.shared_with_user_ids : [],
                  };
                  if (editingCodingSwarmProfileId && !codingSwarmProfileDraft.duplicate_mode) {
                    await updateCodingSwarmProfileMutation.mutateAsync({
                      profileId: editingCodingSwarmProfileId,
                      data: {
                        title: payload.title,
                        description: payload.description,
                        preset_key: payload.preset_key,
                        scope_default: payload.scope_default,
                        default_commands: payload.default_commands,
                        default_file_paths: payload.default_file_paths,
                        max_agents: payload.max_agents,
                        safe_command_policy: payload.safe_command_policy,
                        saved_search_query: payload.saved_search_query,
                        is_default: payload.is_default,
                        status: payload.status,
                        visibility: payload.visibility,
                        shared_with_user_ids: payload.shared_with_user_ids,
                      },
                    });
                  } else {
                    await createCodingSwarmProfileMutation.mutateAsync(payload);
                  }
                  closeCodingSwarmProfileEditor();
                }}
              >
                {editingCodingSwarmProfileId && !codingSwarmProfileDraft.duplicate_mode ? 'Save profile' : 'Create profile'}
              </Button>
              <Button variant="secondary" onClick={closeCodingSwarmProfileEditor}>
                Cancel
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SwarmProfilesTab;
