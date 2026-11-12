import React, { useEffect, useState } from 'react';

export interface ToolInputBinding { stage: string; path: string; }
interface Props {
  name: string;
  arguments: string[];
  sources: string[];
  bindings?: Record<string, ToolInputBinding>;
  onChange: (values: Record<string, ToolInputBinding> | undefined) => void;
}
type Row = ToolInputBinding & { argument: string };

export default function ToolBindingsEditor({ name, arguments: args, sources, bindings, onChange }: Props) {
  const saved = JSON.stringify(Object.fromEntries(Object.entries(bindings || {}).map(([argument, reference]) => [argument, {
    stage: typeof reference?.stage === 'string' ? reference.stage : '',
    path: typeof reference?.path === 'string' ? reference.path : '',
  }])));
  const [rows, setRows] = useState<Row[]>([]);
  useEffect(() => {
    setRows(Object.entries(JSON.parse(saved) as Record<string, ToolInputBinding>)
      .map(([argument, reference]) => ({
        argument,
        stage: typeof reference?.stage === 'string' ? reference.stage : '',
        path: typeof reference?.path === 'string' ? reference.path : '',
      })));
  }, [saved]);
  const values = Object.fromEntries(rows.map(({ argument, stage, path }) => [argument, { stage, path }]));
  const dirty = JSON.stringify(values) !== saved;
  const invalidPointer = (path: string) => (path !== '' && !path.startsWith('/')) || /~(?![01])/.test(path);
  const rowErrors = rows.map((row) => [
    ...(!args.includes(row.argument) ? [`Argument '${row.argument}' is no longer declared by this tool. Choose a declared argument or remove the binding.`] : []),
    ...(!sources.includes(row.stage) ? [`Source '${row.stage}' is not a dependency. Restore the dependency, choose another source, or remove the binding.`] : []),
    ...(invalidPointer(row.path) ? ['Use a JSON Pointer such as /structured_output/value.'] : []),
  ]);
  const invalid = rowErrors.some((errors) => errors.length > 0);
  const duplicate = new Set(rows.map((row) => row.argument)).size !== rows.length;
  const update = (index: number, patch: Partial<Row>) => {
    setRows(rows.map((row, i) => i === index ? { ...row, ...patch } : row));
  };
  const apply = () => {
    if (duplicate || invalid) return;
    onChange(rows.length ? values : undefined);
  };

  return <fieldset className="space-y-2 border-t border-gray-200 pt-2">
    <legend className="font-medium text-gray-800">Bind arguments to stage results</legend>
    <p>Bound values override defaults and call arguments. Missing results stop the tool call.</p>
    {!sources.length && <p>Add a stage dependency to bind its results.</p>}
    {!args.length && <p>This tool has no declared input fields.</p>}
    {rows.map((row, index) => <div key={index} className="space-y-1 rounded border border-gray-300 p-2">
      <label className="block">Argument
        <select aria-label={`Binding argument ${index + 1} for ${name}`} aria-invalid={!args.includes(row.argument)} value={row.argument} onChange={(e) => update(index, { argument: e.target.value })} className="block w-full rounded border border-gray-300 bg-gray-50 p-1">
          {Array.from(new Set([...args, row.argument])).map((arg) => <option key={arg} value={arg}>{args.includes(arg) ? arg : `${arg} (not declared)`}</option>)}
        </select>
      </label>
      <label className="block">Dependency stage
        <select aria-label={`Binding source ${index + 1} for ${name}`} aria-invalid={!sources.includes(row.stage)} value={row.stage} onChange={(e) => update(index, { stage: e.target.value })} className="block w-full rounded border border-gray-300 bg-gray-50 p-1">
          {Array.from(new Set([...sources, row.stage])).map((stage) => <option key={stage} value={stage}>{!stage ? 'Choose a dependency' : sources.includes(stage) ? stage : `${stage} (not a dependency)`}</option>)}
        </select>
      </label>
      <label className="block">Result path (JSON Pointer)
        <input aria-label={`Binding path ${index + 1} for ${name}`} aria-invalid={invalidPointer(row.path)} value={row.path} onChange={(e) => update(index, { path: e.target.value })} className="block w-full rounded border border-gray-300 bg-gray-50 p-1 font-mono" />
      </label>
      {rowErrors[index].map((message) => <p key={message} role="alert" className="text-red-600">{message}</p>)}
      <button type="button" className="underline" aria-label={`Remove binding ${index + 1} for ${name}`} onClick={() => setRows(rows.filter((_, i) => i !== index))}>Remove binding</button>
    </div>)}
    <p>Use /structured_output/value, /findings/0/value, or an empty path for the whole result.</p>
    {duplicate && <p role="alert" className="text-red-600">Each argument can have only one binding.</p>}
    {dirty && <p>Binding changes not applied.</p>}
    <div className="flex gap-2">
      <button type="button" className="underline disabled:opacity-50" disabled={!sources.length || !args.some((arg) => !rows.some((row) => row.argument === arg))} onClick={() => setRows([...rows, { argument: args.find((arg) => !rows.some((row) => row.argument === arg)) || '', stage: sources[0], path: '/structured_output/value' }])}>Add binding</button>
      <button type="button" className="underline disabled:opacity-50" disabled={!dirty || invalid || duplicate} onClick={apply}>Apply bindings</button>
    </div>
  </fieldset>;
}
