import React, { useState } from 'react';
import { Link } from 'react-router-dom';

import type { PipelineTool } from '../../types';
import ToolDetails from './ToolDetails';
import type { ToolInputBinding } from './ToolBindingsEditor';

interface Props {
  tools: PipelineTool[] | undefined;
  selected: string[] | null | undefined;
  jobType: string;
  onChange: (tools: string[] | undefined) => void;
  onRemoveTool: (name: string) => void;
  defaults?: Record<string, Record<string, unknown>>;
  onDefaultsChange?: (name: string, values: Record<string, unknown> | undefined) => void;
  bindings?: Record<string, Record<string, ToolInputBinding>>;
  sources?: string[];
  onBindingsChange?: (name: string, values: Record<string, ToolInputBinding> | undefined) => void;
}

export default function StageToolPicker({
  tools, selected, jobType, onChange, onRemoveTool, defaults, onDefaultsChange, bindings, sources, onBindingsChange,
}: Props) {
  const [filter, setFilter] = useState('');
  const limited = Array.isArray(selected);
  const names = selected || [];
  const configured = Array.from(new Set([...names, ...Object.keys(defaults || {}), ...Object.keys(bindings || {})]));
  const needle = filter.trim().toLowerCase();
  const compatible = (tool: PipelineTool) =>
    tool.job_types === null || tool.job_types.includes(jobType);
  const visible = (tools || [])
    .filter((tool) =>
      (compatible(tool) || configured.includes(tool.name)) &&
      `${tool.name} ${tool.description} ${tool.source} ${tool.execution} ${tool.produces.join(' ')} ${tool.image || ''}`
        .toLowerCase().includes(needle)
    )
    .sort((a, b) =>
      Number(configured.includes(b.name)) - Number(configured.includes(a.name)) ||
      a.name.localeCompare(b.name)
    );
  const missing = tools && configured.filter((name) =>
    !tools.some((tool) => tool.name === name)
  );

  return (
    <fieldset className="space-y-2 rounded-md border border-gray-300 p-2">
      <legend className="text-xs font-medium text-gray-700 px-1">Tool access</legend>
      <label className="block text-[11px] text-gray-500">
        Available to this stage
        <select
          aria-label="Stage tool access"
          className="mt-1 w-full px-2 py-1 text-xs rounded-md bg-gray-50 border border-gray-300 text-gray-900"
          value={limited ? 'selected' : 'all'}
          onChange={(event) => onChange(event.target.value === 'all' ? undefined : [])}
        >
          <option value="all">All permitted tools</option>
          <option value="selected">Evidence tools + selected extras</option>
        </select>
      </label>
      <p className="text-[11px] text-gray-500">
        Tools needed for required evidence, results, and declared backward edges stay available automatically.
        Select extras for reading files, processing data, calling services, or other work.
        Job permissions and approvals still apply.
      </p>
      <div className="space-y-2">
          <input
            aria-label="Filter stage tools"
            placeholder="Search tools, sources, or execution types…"
            className="w-full px-2 py-1 text-xs rounded-md bg-gray-50 border border-gray-300 text-gray-900"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          />
          <div className="max-h-52 overflow-y-auto scrollbar-thin space-y-1">
            {visible.map((tool) => {
              const chosen = names.includes(tool.name);
              return (
                <div key={tool.name} className="border-b border-gray-200 last:border-0">
                  <label className="flex items-start gap-2 p-1 text-xs text-gray-700">
                    <input
                      type="checkbox"
                      aria-label={`Allow tool ${tool.name}`}
                      className="mt-0.5"
                      checked={(!limited && compatible(tool)) || chosen}
                      disabled={!limited || (!compatible(tool) && !chosen)}
                      onChange={() => onChange(
                        chosen
                          ? names.filter((name) => name !== tool.name)
                          : [...names, tool.name]
                      )}
                    />
                    <span className="min-w-0">
                      <span className="block font-mono break-all">{tool.name}</span>
                      <span className="block text-[10px] text-gray-500">{tool.source} · {tool.execution}</span>
                      <span className="block text-[11px]">{tool.description}</span>
                      {defaults?.[tool.name] && <span className="block text-[10px] text-primary-700">Stage defaults configured</span>}
                      {bindings?.[tool.name] && <span className="block text-[10px] text-primary-700">Stage result bindings configured</span>}
                      {!compatible(tool) && (
                        <span className="block text-yellow-600">
                          Unavailable for {jobType}; remove or change job type.
                        </span>
                      )}
                    </span>
                  </label>
                  {!compatible(tool) && <button type="button"
                    aria-label={`Remove ${tool.name} and stage settings`}
                    className="ml-6 text-[11px] text-yellow-700 underline"
                    onClick={() => onRemoveTool(tool.name)}>Remove tool and settings</button>}
                  <ToolDetails
                    tool={tool}
                    defaults={defaults?.[tool.name]}
                    onDefaultsChange={onDefaultsChange ? (values) => onDefaultsChange(tool.name, values) : undefined}
                    bindings={bindings?.[tool.name]}
                    sources={sources}
                    onBindingsChange={onBindingsChange ? (values) => onBindingsChange(tool.name, values) : undefined}
                  />
                </div>
              );
            })}
            {visible.length === 0 && !missing?.length && (
              <p className="text-[11px] text-gray-500 p-1">
                {tools ? 'No tools match.' : 'Tool catalog unavailable. Refresh tools and skills to choose tools.'}
              </p>
            )}
            {missing && missing.map((name) => (
              <div key={name} className="rounded border border-yellow-600 p-2 space-y-1 text-[11px]">
                <p className="font-mono break-all text-yellow-700">{name} is unavailable</p>
                <p>Its saved settings remain in this stage. Restore the tool, or remove its extra selection, defaults, and bindings here.</p>
                {(defaults?.[name] || bindings?.[name]) && <details>
                  <summary className="cursor-pointer text-primary-700">Saved stage settings</summary>
                  <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all rounded bg-gray-50 p-2">
                    {JSON.stringify({ defaults: defaults?.[name], bindings: bindings?.[name] }, null, 2)}
                  </pre>
                </details>}
                <button type="button" aria-label={`${name} is unavailable — remove tool and settings`}
                  className="underline text-yellow-700" onClick={() => onRemoveTool(name)}>Remove tool and settings</button>
              </div>
            ))}
          </div>
      </div>
      <p className="text-[11px] text-gray-500">
        <Link to="/tools" className="underline">Manage tools and Sandbox Skills</Link>.
        Choose an active skill under “Must produce” to use its configured image and procedure.
      </p>
    </fieldset>
  );
}
