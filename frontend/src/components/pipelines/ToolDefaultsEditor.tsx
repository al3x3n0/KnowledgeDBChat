import React, { useEffect, useState } from 'react';

interface Props {
  name: string;
  defaults?: Record<string, unknown>;
  onChange: (values: Record<string, unknown> | undefined) => void;
}

export default function ToolDefaultsEditor({ name, defaults, onChange }: Props) {
  const saved = JSON.stringify(defaults || {}, null, 2);
  const [draft, setDraft] = useState(saved);
  const [error, setError] = useState('');
  useEffect(() => { setDraft(saved); setError(''); }, [saved]);

  const apply = () => {
    try {
      const values = JSON.parse(draft);
      if (values === null || typeof values !== 'object' || Array.isArray(values)) {
        throw new Error('Defaults must be a JSON object.');
      }
      onChange(Object.keys(values).length ? values : undefined);
      setDraft(JSON.stringify(values, null, 2));
      setError('');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Invalid JSON.');
    }
  };

  return (
    <div className="space-y-1 border-t border-gray-200 pt-2">
      <label className="block font-medium text-gray-800">
        Stage argument defaults
        <textarea
          aria-label={`Argument defaults for ${name}`}
          className="mt-1 w-full rounded border border-gray-300 bg-gray-50 p-2 font-mono text-xs text-gray-900"
          rows={5}
          value={draft}
          onChange={(event) => { setDraft(event.target.value); setError(''); }}
        />
      </label>
      <p>Set a JSON object of default arguments for this stage. Explicit call arguments override these values.</p>
      <p>{draft !== saved ? 'Changes not applied.' : Object.keys(defaults || {}).length ? 'Defaults saved in this stage.' : 'No stage defaults set.'}</p>
      {error && <p role="alert" className="text-red-600">{error}</p>}
      <div className="flex gap-2">
        <button type="button" className="underline disabled:opacity-50" disabled={draft === saved} onClick={apply}>Apply defaults</button>
        <button
          type="button" className="underline disabled:opacity-50"
          disabled={!Object.keys(defaults || {}).length && draft === saved}
          onClick={() => { onChange(undefined); setDraft('{}'); setError(''); }}
        >Clear defaults</button>
      </div>
    </div>
  );
}
