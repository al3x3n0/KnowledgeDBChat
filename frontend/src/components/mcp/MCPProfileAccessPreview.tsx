import React from 'react';

import type { MCPCustomPrompt, MCPNoteOption, MCPSourceAccessResponse } from '../../types';

type Tool = { name: string; label: string; enabled: boolean };
type Source = { id: string; name: string; active: boolean };

const documentOperations = [
  { label: 'Read', tools: ['list_documents', 'get_document', 'list_sources'], permission: 'can_read' },
  { label: 'Search', tools: ['search'], permission: 'can_search' },
  { label: 'Chat', tools: ['chat'], permission: 'can_chat' },
] as const;

export default function MCPProfileAccessPreview({
  tools, programs, notes, prompts, unavailablePrograms, sources, restricted, configuredSources, mcpEnabled,
}: {
  tools: Tool[];
  programs: string[];
  notes: MCPNoteOption[];
  prompts: MCPCustomPrompt[];
  unavailablePrograms: string[];
  sources: Source[];
  restricted: boolean;
  configuredSources: MCPSourceAccessResponse[];
  mcpEnabled: boolean;
}) {
  const selectedTools = new Set(tools.map(({ name }) => name));
  const hasDocumentTools = documentOperations.some(({ tools: names }) =>
    names.some((name) => selectedTools.has(name))
  );
  const enabledTools = new Set(tools.filter(({ enabled }) => enabled).map(({ name }) => name));
  const selectedOperations = documentOperations.filter(({ tools: names }) =>
    names.some((name) => enabledTools.has(name))
  );
  const enabledPrograms = enabledTools.has('get_research_programs');
  const enabledNotes = enabledTools.has('get_research_notes');

  return (
    <details className="rounded-md border bg-gray-50 p-4">
      <summary className="cursor-pointer font-medium text-gray-900">
        Access preview
        {hasDocumentTools && (
          <span className={`ml-2 text-sm font-normal ${restricted ? 'text-gray-600' : 'text-amber-800'}`}>
            {restricted
              ? `${sources.length} selected document ${sources.length === 1 ? 'source' : 'sources'}`
              : 'All document sources'}
          </span>
        )}
      </summary>
      <div className="mt-3 space-y-3 text-sm text-gray-700">
        <p>Policies apply to client requests.</p>
        {!mcpEnabled && <p className="font-medium text-amber-800">MCP access is disabled for this profile.</p>}
        {tools.length > 0 && <div>
          <h3 className="font-medium text-gray-900">Tools</h3>
          <ul aria-label="Previewed tools" className="mt-1 list-inside list-disc">
            {tools.map(({ name, label, enabled }) => (
              <li key={name}>
                {label} <code className="text-xs text-gray-600">{name}</code>{!enabled && ' · Disabled'}
              </li>
            ))}
          </ul>
        </div>}
        {prompts.length > 0 && (
          <div>
            <h3 className="font-medium text-gray-900">Custom prompts</h3>
            <ul aria-label="Previewed prompts" className="mt-1 list-inside list-disc">
              {prompts.map((prompt, index) => (
                <li key={index}>{prompt.title || 'Untitled prompt'} <code>{prompt.name}</code></li>
              ))}
            </ul>
          </div>
        )}
        {(programs.length > 0 || notes.length > 0) && (
          <div>
            <h3 className="font-medium text-gray-900">Resources</h3>
            <ul aria-label="Previewed resources" className="mt-1 list-inside list-disc">
              {programs.map((program) => (
                <li key={`program:${program}`}>
                  {program} · Program{unavailablePrograms.includes(program) ? ' · Unavailable' : !enabledPrograms && ' · Disabled'}
                </li>
              ))}
              {notes.map(({ id, title }) => (
                <li key={`note:${id}`}>{title} · Note{!enabledNotes && ' · Disabled'}</li>
              ))}
            </ul>
          </div>
        )}
        {hasDocumentTools && (
          <div>
            <h3 className="font-medium text-gray-900">Document scope</h3>
            {!restricted ? (
              <p>Entire shared knowledge base</p>
            ) : sources.length ? (
              <ul aria-label="Previewed document sources" className="mt-1 list-inside list-disc">
                {sources.map(({ id, name, active }) => {
                  const access = configuredSources.find((source) => source.source_id === id);
                  const operations = selectedOperations.filter(
                    ({ permission }) => access?.[permission] ?? true
                  ).map(({ label }) => label);
                  return (
                    <li key={id}>
                      {name} · {active ? operations.join(', ') || 'No enabled operations' : 'Inactive or unavailable'}
                    </li>
                  );
                })}
              </ul>
            ) : <p>Access requires a selected source.</p>}
          </div>
        )}
      </div>
    </details>
  );
}
