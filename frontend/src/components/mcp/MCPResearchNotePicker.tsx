import React, { useState } from 'react';
import { useQuery } from 'react-query';

import { apiClient } from '../../services/api';
import type { MCPNoteOption } from '../../types';
import Button from '../common/Button';

const MAX_NOTES = 50;

export default function MCPResearchNotePicker({ selected, onChange }: {
  selected: MCPNoteOption[];
  onChange: (notes: MCPNoteOption[]) => void;
}) {
  const [input, setInput] = useState('');
  const [query, setQuery] = useState('');
  const notes = useQuery(['mcpProfileNotes', query], () => apiClient.listMCPProfileNotes(query));
  const matches = notes.data ?? [];
  const selectedIds = new Set(selected.map(({ id }) => id));
  const options = [...selected, ...matches.filter(({ id }) => !selectedIds.has(id))];

  return (
    <fieldset className="rounded-md border p-3">
      <legend className="px-1 font-medium text-gray-900">Research notes</legend>
      <p className="mb-2 text-sm text-gray-600">
        Share selected notes and their current Markdown as MCP resources and through the research notes tool.
      </p>
      <div className="flex flex-wrap items-end gap-2">
        <label className="min-w-0 flex-1 text-sm font-medium text-gray-700">
          Search by note title
          <input
            className="mt-1 block w-full rounded-md border px-3 py-2"
            maxLength={100}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                setQuery(input.trim());
              }
            }}
          />
        </label>
        <Button type="button" variant="secondary" onClick={() => setQuery(input.trim())}>Search notes</Button>
      </div>
      <p className="mt-2 text-sm text-gray-600">{selected.length} of {MAX_NOTES} notes selected</p>
      {notes.isLoading && <p className="mt-2 text-sm" role="status">Loading research notes…</p>}
      {notes.isError && (
        <p className="mt-2 text-sm text-red-700" role="alert">
          Could not load research notes.{' '}
          <button type="button" className="underline" onClick={() => void notes.refetch()}>Retry</button>
        </p>
      )}
      {notes.isSuccess && matches.length === 0 && <p className="mt-2 text-sm text-gray-600">No matching research notes.</p>}
      {options.length > 0 && (
        <div className="mt-2 max-h-40 space-y-2 overflow-y-auto">
          {options.map((note) => (
            <label key={note.id} className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={selectedIds.has(note.id)}
                disabled={!selectedIds.has(note.id) && selected.length >= MAX_NOTES}
                onChange={() => onChange(selectedIds.has(note.id)
                  ? selected.filter(({ id }) => id !== note.id)
                  : [...selected, note])}
              />
              {note.title}
            </label>
          ))}
        </div>
      )}
    </fieldset>
  );
}
