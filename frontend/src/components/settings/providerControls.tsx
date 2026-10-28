/**
 * The pieces every provider choice is made with: the settings page's
 * default and per-task pickers, and the chat header's pick for one chat.
 * Kept in one place so a saved provider is offered, named and listed the
 * same way wherever a provider is chosen.
 */

import React from 'react';
import { useQuery } from 'react-query';

import Button from '../common/Button';
import { apiClient } from '../../services/api';
import type { LLMProviderCatalog, SavedLLMProvider } from '../../services/api';

export const CUSTOM_PREFIX = 'custom:';

export const inputClass =
  'w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm';

export function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/** A provider choice as a person reads it. */
export function describeRef(
  ref: string,
  catalog: LLMProviderCatalog | undefined,
  saved: SavedLLMProvider[],
): string {
  if (!ref) return 'Default';
  if (ref.startsWith(CUSTOM_PREFIX)) {
    const p = saved.find((s) => `${CUSTOM_PREFIX}${s.id}` === ref);
    return p ? p.name : `${ref} (deleted)`;
  }
  return catalog?.builtin.find((b) => b.id === ref)?.label || ref;
}

/** Model names a provider is known to serve, asked of the backend. */
export function useProviderModels(ref: string) {
  return useQuery(
    ['myLLMModels', ref],
    () => apiClient.listMyLLMModels({ provider: ref || undefined }),
    { refetchOnWindowFocus: false, staleTime: 60_000 },
  );
}

export const ModelField: React.FC<{
  id: string;
  providerRef: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}> = ({ id, providerRef, value, onChange, placeholder }) => {
  const { data, isFetching, refetch } = useProviderModels(providerRef);
  const listId = `models-${id}`;
  const models = data?.models || [];
  return (
    <div>
      <div className="flex gap-2">
        <input
          type="text"
          list={listId}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={
            placeholder ||
            (data?.default_model ? `Provider default: ${data.default_model}` : 'Provider default')
          }
          className={inputClass}
          aria-label="Model"
        />
        <datalist id={listId}>
          {models.map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
        <Button type="button" variant="ghost" size="sm" onClick={() => refetch()} loading={isFetching}>
          Refresh
        </Button>
      </div>
      <p className="text-xs text-gray-500 mt-1">
        {data?.error
          ? `Could not list models: ${data.error}`
          : models.length
            ? `${models.length} model${models.length === 1 ? '' : 's'} known; type to pick one or enter any name.`
            : 'Type a model name, or leave empty for the provider default.'}
      </p>
    </div>
  );
};

/** One `<select>` for every place a provider is chosen. */
export const ProviderSelect: React.FC<{
  value: string;
  onChange: (v: string) => void;
  catalog: LLMProviderCatalog | undefined;
  saved: SavedLLMProvider[];
  emptyLabel: string;
}> = ({ value, onChange, catalog, saved, emptyLabel }) => (
  <select value={value} onChange={(e) => onChange(e.target.value)} className={inputClass} aria-label="Provider">
    <option value="">{emptyLabel}</option>
    <optgroup label="Built-in">
      {(catalog?.builtin || []).map((b) => (
        <option key={b.id} value={b.id}>
          {b.label}
          {b.requires_api_key && !b.system_configured ? ' (needs your key)' : ''}
        </option>
      ))}
    </optgroup>
    {saved.length > 0 && (
      <optgroup label="Your providers">
        {saved.map((p) => (
          <option key={p.id} value={`${CUSTOM_PREFIX}${p.id}`}>
            {p.name}
          </option>
        ))}
      </optgroup>
    )}
    {value && !value.startsWith(CUSTOM_PREFIX) && !catalog?.builtin.some((b) => b.id === value) && (
      <option value={value}>{value} (unknown)</option>
    )}
  </select>
);
