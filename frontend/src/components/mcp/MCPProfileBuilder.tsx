import React, { useId, useState } from 'react';
import { useQuery } from 'react-query';
import { Pencil, Plus } from 'lucide-react';
import toast from 'react-hot-toast';

import { apiClient } from '../../services/api';
import type { DocumentSource, MCPCustomPrompt, MCPKeyConfigResponse, MCPNoteOption } from '../../types';
import Button from '../common/Button';
import MCPProfileCredentials from './MCPProfileCredentials';
import MCPProfileAccessPreview from './MCPProfileAccessPreview';
import MCPResearchNotePicker from './MCPResearchNotePicker';
import MCPProfilePromptEditor, { validCustomPrompts } from './MCPProfilePromptEditor';

const SOURCE_PROFILE_TOOLS = ['search', 'list_documents', 'get_document'];

type Props =
  | {
      onCreated: (id: string) => void;
      profile?: undefined;
      initialSelection?: {
        program?: string;
        note?: MCPNoteOption;
        source?: Pick<DocumentSource, 'id' | 'name'> | null;
      };
      onClose?: () => void;
    }
  | { profile: MCPKeyConfigResponse; onSaved: (profile: MCPKeyConfigResponse) => void };

export default function MCPProfileBuilder(props: Props) {
  const titleId = useId();
  const profile = props.profile;
  const selection = 'initialSelection' in props ? props.initialSelection : undefined;
  const sourceRequested = selection?.source !== undefined;
  const [open, setOpen] = useState(Boolean(selection?.program || selection?.note || sourceRequested));
  const [name, setName] = useState(
    (selection?.note?.title ?? selection?.program ?? selection?.source?.name ?? '').slice(0, 100)
  );
  const [programs, setPrograms] = useState<string[]>(selection?.program ? [selection.program] : []);
  const [notes, setNotes] = useState<MCPNoteOption[]>(selection?.note ? [selection.note] : []);
  const [prompts, setPrompts] = useState<MCPCustomPrompt[]>([]);
  const [tools, setTools] = useState<string[]>(selection?.source ? SOURCE_PROFILE_TOOLS : []);
  const [restrictSources, setRestrictSources] = useState(sourceRequested);
  const [sourceIds, setSourceIds] = useState<string[]>(selection?.source ? [selection.source.id] : []);
  const [created, setCreated] = useState<{ id: string; api_key: string }>();
  const [status, setStatus] = useState<'idle' | 'saving' | 'error'>('idle');
  const options = useQuery('mcpProfileOptions', () => apiClient.getMCPProfileOptions(), {
    enabled: open,
  });
  const sources = useQuery('documentSources', () => apiClient.getDocumentSources(), {
    enabled: open && restrictSources,
  });
  const activeSources = sources.data?.filter((source) => source.is_active) ?? [];
  const retainedSources = (profile?.source_access ?? []).filter(
    (access) => sourceIds.includes(access.source_id) && !activeSources.some((source) => source.id === access.source_id)
  );
  const selectableSources = [
    ...activeSources.map((source) => ({ id: source.id, name: source.name, active: true })),
    ...retainedSources.map((source) => ({ id: source.source_id, name: source.source_name, active: false })),
  ];
  const selectedSources = selectableSources.filter(({ id }) => sourceIds.includes(id));
  const unavailablePrograms = programs.filter(
    (name) => options.isSuccess && !options.data.programs.some((program) => program.name === name)
  );
  const canSave = Boolean(
    options.isSuccess && name.trim() && (programs.length || notes.length || tools.length || prompts.length) &&
    validCustomPrompts(prompts) &&
    (profile || unavailablePrograms.length === 0) &&
    (profile || !tools.length || !restrictSources || selectedSources.length)
  );
  const previewTools = [
    ...tools.map((name) => ({
      name,
      label: options.data?.tools.find((tool) => tool.name === name)?.display_name ?? name,
    })),
    ...(programs.length ? [{ name: 'get_research_programs', label: 'Research Program Context' }] : []),
    ...(notes.length ? [{ name: 'get_research_notes', label: 'Research Notes' }] : []),
  ].map((tool) => ({
    ...tool,
    enabled: profile?.tool_configs.find(({ tool_name }) => tool_name === tool.name)?.is_enabled ?? true,
  }));

  function start() {
    if (profile) {
      setName(profile.api_key_name);
      setPrograms(profile.mcp_programs ?? []);
      setNotes(profile.research_notes ?? []);
      setPrompts(profile.prompts ?? []);
      setTools((profile.allowed_tools ?? []).filter(
        (tool) => !['get_research_programs', 'get_research_notes'].includes(tool)
      ));
      setRestrictSources(profile.source_access_mode === 'restricted');
      setSourceIds(profile.source_access.map((source) => source.source_id));
    }
    setOpen(true);
  }

  async function saveProfile(event: React.FormEvent) {
    event.preventDefault();
    if (!canSave || status === 'saving') {
      return;
    }
    setStatus('saving');
    try {
      const sourceSelection = restrictSources && (profile || tools.length)
        ? selectedSources.map(({ id }) => id)
        : null;
      const definition = {
        name: name.trim(),
        programs,
        tools,
        ...(profile || notes.length ? { note_ids: notes.map(({ id }) => id) } : {}),
        ...(prompts.length || profile?.prompts?.length ? { prompts } : {}),
        ...(profile || sourceSelection !== null ? { source_ids: sourceSelection } : {}),
      };
      if (props.profile) {
        const updated = await apiClient.updateMCPProfile(props.profile.api_key_id, definition);
        close();
        props.onSaved(updated);
        toast.success('MCP profile updated');
        return;
      }
      const result = await apiClient.createMCPProfile(definition);
      setCreated(result);
      setStatus('idle');
      props.onCreated(result.id);
    } catch {
      setStatus('error');
    }
  }

  function close() {
    setOpen(false);
    setCreated(undefined);
    setName('');
    setPrograms([]);
    setNotes([]);
    setPrompts([]);
    setTools([]);
    setRestrictSources(false);
    setSourceIds([]);
    setStatus('idle');
    if ('onClose' in props) props.onClose?.();
  }

  function toggle(values: string[], value: string) {
    return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
  }

  return (
    <section className="mb-6 rounded-lg border bg-white p-5" aria-labelledby={titleId}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 id={titleId} className="font-semibold text-gray-900">{profile ? 'Profile scope' : 'Publish research context'}</h2>
          <p className="mt-1 text-sm text-gray-600">
            {profile ? 'Change what this client can access using its current API key.' : 'Choose research programs, notes, and knowledge tools for an external client.'}
          </p>
        </div>
        {!open && (
          <Button onClick={start} variant={profile ? 'secondary' : 'primary'}>
            {profile ? <Pencil className="mr-2 h-4 w-4" /> : <Plus className="mr-2 h-4 w-4" />}
            {profile ? 'Edit profile' : 'New MCP profile'}
          </Button>
        )}
      </div>
      {open && (created ? (
        <div className="mt-5 border-t pt-5">
          <MCPProfileCredentials profile={created} message={`${name.trim()} is ready`} onDone={close} />
        </div>
      ) : (
        <form className="mt-5 space-y-5 border-t pt-5" onSubmit={saveProfile}>
          <fieldset disabled={status === 'saving'} className="space-y-5">
            <label className="block text-sm font-medium text-gray-700">
              Name
              <input required maxLength={100} value={name} onChange={(event) => setName(event.target.value)}
                className="mt-1 block w-full max-w-lg rounded-md border px-3 py-2"
                placeholder="Architecture research" />
            </label>
            {!options.data && !options.isError && <p role="status">Loading options…</p>}
            {options.isError && (
              <p role="alert" className="text-sm text-red-700">
                Could not load MCP options.{' '}
                <button type="button" className="underline" onClick={() => void options.refetch()}>Retry</button>
              </p>
            )}
            {options.data && (
              <div className="grid gap-5 md:grid-cols-2">
                <fieldset>
                  <legend className="font-medium text-gray-900">Research programs</legend>
                  <p className="mb-2 text-sm text-gray-600">
                    Share program descriptions, versions, and tool schemas as MCP resources.
                  </p>
                  {options.data.programs.length === 0 && unavailablePrograms.length === 0 && <p className="text-sm text-gray-500">No research programs installed.</p>}
                  {unavailablePrograms.map((program) => (
                    <label key={program} className="flex items-center gap-2 p-2 text-sm">
                      <input type="checkbox" checked onChange={() => setPrograms(toggle(programs, program))} />
                      {program} (unavailable)
                    </label>
                  ))}
                  {!profile && unavailablePrograms.length > 0 && (
                    <p className="text-sm text-amber-800">Deselect the unavailable program to continue.</p>
                  )}
                  {options.data.programs.map((program) => (
                    <label key={program.name} className="flex items-start gap-2 rounded-md p-2 hover:bg-gray-50">
                      <input type="checkbox" className="mt-1" checked={programs.includes(program.name)}
                        onChange={() => setPrograms(toggle(programs, program.name))} />
                      <span className="text-sm">
                        <span className="block font-medium">{program.name} {program.version}</span>
                        <span className="text-gray-600">{program.description} {program.tools.length} tool schemas</span>
                      </span>
                    </label>
                  ))}
                </fieldset>
                <fieldset>
                  <legend className="font-medium text-gray-900">Knowledge tools</legend>
                  {options.data.tools.map((tool) => (
                    <label key={tool.name} className="flex items-start gap-2 rounded-md p-2 hover:bg-gray-50">
                      <input type="checkbox" className="mt-1" checked={tools.includes(tool.name)}
                        onChange={() => setTools(toggle(tools, tool.name))} />
                      <span className="text-sm">
                        <span className="block font-medium">{tool.display_name}</span>
                        <span className="text-gray-600">{tool.description}</span>
                      </span>
                    </label>
                  ))}
                </fieldset>
              </div>
            )}
            <MCPResearchNotePicker selected={notes} onChange={setNotes} />
            <MCPProfilePromptEditor prompts={prompts} onChange={setPrompts} />
            {(profile || tools.length > 0) && (
              <fieldset className="rounded-md border p-3">
                <legend className="px-1 font-medium text-gray-900">Document sources</legend>
                <label className="flex items-center gap-2 text-sm text-gray-700">
                  <input type="checkbox" checked={restrictSources}
                    onChange={(event) => setRestrictSources(event.target.checked)} />
                  Limit to selected sources
                </label>
                {restrictSources ? (
                  <div className="ml-5 mt-2 max-h-48 space-y-2 overflow-y-auto text-sm">
                    {sources.isLoading && <p role="status">Loading sources…</p>}
                    {sources.isError && (
                      <p role="alert">Could not load sources.{' '}
                        <button type="button" className="underline" onClick={() => void sources.refetch()}>Retry</button>
                      </p>
                    )}
                    {sources.isSuccess && selectableSources.length === 0 && <p>No active document sources.</p>}
                    {selectableSources.map((source) => (
                      <label key={source.id} className="flex items-center gap-2">
                        <input type="checkbox" checked={sourceIds.includes(source.id)}
                          onChange={() => setSourceIds(toggle(sourceIds, source.id))} />
                        {source.name}
                      </label>
                    ))}
                  </div>
                ) : <p className="ml-5 mt-1 text-sm text-gray-600">The selected tools can use the entire shared knowledge base.</p>}
              </fieldset>
            )}
          </fieldset>
          {(previewTools.length > 0 || prompts.length > 0) && (
            <MCPProfileAccessPreview
              tools={previewTools}
              programs={programs}
              notes={notes}
              prompts={prompts}
              unavailablePrograms={unavailablePrograms}
              sources={selectedSources}
              restricted={restrictSources}
              configuredSources={profile?.source_access ?? []}
              mcpEnabled={profile?.mcp_enabled ?? true}
            />
          )}
          {status === 'error' && <p role="alert" className="text-sm text-red-700">Could not save the MCP profile. Try again.</p>}
          <div className="flex gap-2">
            <Button type="submit" loading={status === 'saving'} disabled={!canSave}>
              {profile ? 'Save changes' : 'Create MCP profile'}
            </Button>
            <Button type="button" variant="ghost" onClick={close} disabled={status === 'saving'}>Cancel</Button>
          </div>
        </form>
      ))}
    </section>
  );
}
