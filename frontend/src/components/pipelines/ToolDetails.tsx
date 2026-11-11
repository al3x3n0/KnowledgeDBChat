import React, { useState } from 'react';

import type { PipelineTool } from '../../types';
import ToolDefaultsEditor from './ToolDefaultsEditor';
import ToolBindingsEditor, { ToolInputBinding } from './ToolBindingsEditor';

function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function inputType(value: unknown): string {
  const schema = object(value);
  if (typeof schema.type === 'string') return schema.type;
  if (Array.isArray(schema.type)) return schema.type.join(' | ');
  const alternatives = schema.anyOf || schema.oneOf;
  if (Array.isArray(alternatives)) return alternatives.map(inputType).join(' | ');
  return typeof schema.$ref === 'string' ? schema.$ref.split('/').pop() || 'value' : 'value';
}

export default function ToolDetails({ tool, defaults, onDefaultsChange, bindings, sources = [], onBindingsChange }: {
  tool: PipelineTool;
  defaults?: Record<string, unknown>;
  onDefaultsChange?: (values: Record<string, unknown> | undefined) => void;
  bindings?: Record<string, ToolInputBinding>;
  sources?: string[];
  onBindingsChange?: (values: Record<string, ToolInputBinding> | undefined) => void;
}) {
  const [open, setOpen] = useState(false);
  const schema = tool.parameters;
  const properties = object(schema?.properties);
  const required = Array.isArray(schema?.required) ? schema.required : [];

  return (
    <details
      className="ml-6 px-1 pb-2 text-[11px] text-gray-600"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer text-primary-700" aria-label={`Inspect tool ${tool.name}`}>
        Inputs, results, and execution
      </summary>
      {open && <div className="mt-2 space-y-2">
        <p>The agent supplies arguments when it calls this tool. Stage defaults fill omitted arguments.</p>
        <div>
          <h4 className="font-medium text-gray-800">Inputs</h4>
          {Object.keys(properties).length > 0 ? (
            <dl className="space-y-1 mt-1">
              {Object.entries(properties).map(([name, value]) => {
                const description = object(value).description;
                return (
                  <div key={name}>
                    <dt className="font-mono break-all">
                      {name} · {inputType(value)} · {required.includes(name) ? 'required' : 'optional'}
                    </dt>
                    {typeof description === 'string' && <dd>{description}</dd>}
                  </div>
                );
              })}
            </dl>
          ) : <p>{schema ? 'No named input fields declared.' : 'Input schema unavailable.'}</p>}
          {schema && (
            <details className="mt-1">
              <summary className="cursor-pointer">Full input schema</summary>
              <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-all rounded bg-gray-50 p-2">
                {JSON.stringify(schema, null, 2)}
              </pre>
            </details>
          )}
          {tool.consumes && <p className="mt-1">Consumes: {tool.consumes}</p>}
        </div>
        <div>
          <h4 className="font-medium text-gray-800">Evidence produced</h4>
          <p className="break-words">{tool.produces.length
            ? tool.produces.join(', ')
            : 'No evidence type declared. The tool returns its own result.'}</p>
        </div>
        {Boolean(tool.requires?.length) && <p>Prerequisite tools: {tool.requires?.join(', ')}</p>}
        <div>
          <h4 className="font-medium text-gray-800">Declared execution</h4>
          <p>Effects: {tool.effects || 'unknown'} · Network: {tool.network || 'unknown'} · Cost: {tool.cost_tier || 'unknown'}</p>
          {tool.image && <p className="break-all">Configured image: {tool.image}</p>}
          {Boolean(tool.program_images?.length) && (
            <div>
              <p>Program images; the tool handler chooses which to use:</p>
              <ul className="list-disc pl-4 break-all">
                {tool.program_images?.map((image) => <li key={image}>{image}</li>)}
              </ul>
            </div>
          )}
        </div>
        {onDefaultsChange && <ToolDefaultsEditor name={tool.name} defaults={defaults} onChange={onDefaultsChange} />}
        {onBindingsChange && <ToolBindingsEditor name={tool.name} arguments={Object.keys(properties)} sources={sources} bindings={bindings} onChange={onBindingsChange} />}
      </div>}
    </details>
  );
}
