import React from 'react';

import type { MCPCustomPrompt } from '../../types';
import Button from '../common/Button';

const MAX_PROMPTS = 10;
const MAX_PROMPT_INPUTS = 10;
const reservedNames = new Set(['research_brief', 'compare_research_notes']);
const inputClass = 'mt-1 block w-full rounded-md border bg-white px-3 py-2 text-sm';

function promptNameError(name: string, prompts: MCPCustomPrompt[]): string {
  const trimmed = name.trim();
  if (!/^[a-z][a-z0-9_-]{0,47}$/.test(trimmed)) {
    return 'Start with a lowercase letter. Use lowercase letters, digits, hyphens, or underscores.';
  }
  if (reservedNames.has(trimmed) || prompts.filter((prompt) => prompt.name.trim() === trimmed).length > 1) {
    return 'Choose a unique name distinct from the built-in prompts.';
  }
  return '';
}

function promptInputNames(instructions: string): string[] {
  const matches = Array.from(instructions.matchAll(/\{\{[ \t]*([a-z][a-z0-9_]{0,47})[ \t]*\}\}/g));
  const names = matches.filter((match) => {
    const start = match.index ?? 0;
    return instructions[start - 1] !== '{' && instructions[start + match[0].length] !== '}';
  }).map(([, name]) => name);
  return Array.from(new Set(names));
}

function inputsAreValid(inputs: string[]): boolean {
  return inputs.length >= 1 && inputs.length <= MAX_PROMPT_INPUTS;
}

export function validCustomPrompts(prompts: MCPCustomPrompt[]): boolean {
  return prompts.every(({ name, title, instructions, use_inputs }) =>
    !promptNameError(name, prompts) && Boolean(title.trim() && instructions.trim()) &&
    (!use_inputs || inputsAreValid(promptInputNames(instructions)))
  );
}

export default function MCPProfilePromptEditor({ prompts, onChange }: {
  prompts: MCPCustomPrompt[];
  onChange: (prompts: MCPCustomPrompt[]) => void;
}) {
  function updatePrompt(index: number, values: Partial<MCPCustomPrompt>) {
    onChange(prompts.map((prompt, position) => position === index ? { ...prompt, ...values } : prompt));
  }

  return (
    <fieldset className="space-y-3 rounded-md border p-3">
      <legend className="px-1 font-medium text-gray-900">Custom prompts</legend>
      <p className="text-sm text-gray-600">Write instructions clients can select and reuse.</p>
      {prompts.map((prompt, index) => {
        const nameError = prompt.name && promptNameError(prompt.name, prompts);
        const inputs = prompt.use_inputs ? promptInputNames(prompt.instructions) : [];
        return (
          <fieldset key={index} className="space-y-2 rounded-md border bg-gray-50 p-3">
            <legend className="px-1 text-sm font-medium text-gray-900">{prompt.title || 'Untitled prompt'}</legend>
            <div className="grid gap-2 sm:grid-cols-2">
              <label className="block text-sm font-medium text-gray-700">
                Prompt title
                <input maxLength={100} className={inputClass} value={prompt.title}
                  onChange={(event) => updatePrompt(index, { title: event.target.value })} />
              </label>
              <label className="block text-sm font-medium text-gray-700">
                Prompt name
                <input maxLength={48} className={inputClass} value={prompt.name} aria-invalid={Boolean(nameError)}
                  placeholder="summarize_findings" onChange={(event) => updatePrompt(index, { name: event.target.value })} />
              </label>
            </div>
            {nameError && <p className="text-sm text-red-700" role="alert">{nameError}</p>}
            <label className="block text-sm font-medium text-gray-700">
              Instructions
              <textarea maxLength={8000} rows={4} className={inputClass} value={prompt.instructions}
                onChange={(event) => updatePrompt(index, { instructions: event.target.value })} />
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input type="checkbox" checked={prompt.use_inputs}
                onChange={(event) => updatePrompt(index, { use_inputs: event.target.checked })} />
              Enable prompt inputs
            </label>
            {prompt.use_inputs && (
              <div className="space-y-1 text-sm text-gray-600">
                <p>Use placeholders like <code>{'{{topic}}'}</code> in the instructions. Clients ask for a value for each input.</p>
                {inputsAreValid(inputs) ? (
                  <p>Client inputs <code>{inputs.join(', ')}</code></p>
                ) : (
                  <p role="alert" className="text-red-700">Use between one and ten named inputs.</p>
                )}
              </div>
            )}
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input type="checkbox" checked={prompt.include_resources}
                onChange={(event) => updatePrompt(index, { include_resources: event.target.checked })} />
              Reference available profile resources
            </label>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={() => onChange(prompts.filter((_prompt, position) => position !== index))}
            >
              Remove prompt
            </Button>
          </fieldset>
        );
      })}
      <Button
        type="button"
        variant="secondary"
        size="sm"
        disabled={prompts.length >= MAX_PROMPTS}
        onClick={() => onChange([
          ...prompts,
          { name: '', title: '', instructions: '', include_resources: true, use_inputs: false },
        ])}
      >
        Add prompt
      </Button>
      {prompts.length >= MAX_PROMPTS && <p className="text-sm text-gray-600">This profile has reached the limit of ten custom prompts.</p>}
    </fieldset>
  );
}
