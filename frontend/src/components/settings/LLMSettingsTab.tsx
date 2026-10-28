/**
 * Choosing where your model calls go.
 *
 * Three kinds of choice, one picker: the deployment's default, a built-in
 * provider the deployment knows by name, or a provider you saved yourself
 * (a vLLM on your machine, a LiteLLM gateway, OpenRouter). A saved provider
 * is named `custom:<id>` everywhere a provider is named, so a per-task
 * override can send title generation to a small local model and chat to a
 * hosted one, each with its own endpoint and key.
 *
 * What can be chosen comes from `GET /users/me/llm-providers` rather than
 * being listed here, so this page cannot offer a provider the backend does
 * not route or forget one it does. Saved providers are written as soon as
 * they are saved in the editor; the rest of the form is written by Save.
 */

import React, { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { CheckCircle2, Pencil, Plus, Server, Trash2, XCircle } from 'lucide-react';
import toast from 'react-hot-toast';

import Button from '../common/Button';
import { apiClient } from '../../services/api';
import type {
  LLMProviderCatalog,
  LLMSettings,
  ProviderTemplate,
  SavedLLMProvider,
  SavedLLMProviderInput,
} from '../../services/api';

const CUSTOM_PREFIX = 'custom:';

export const LLM_TASK_TYPES = [
  { id: 'chat', label: 'Chat', description: 'Main conversation model' },
  { id: 'title_generation', label: 'Title Generation', description: 'Chat session titles' },
  { id: 'summarization', label: 'Summarization', description: 'Document summaries' },
  { id: 'query_expansion', label: 'Query Expansion', description: 'Search query variations' },
  { id: 'memory_extraction', label: 'Memory Extraction', description: 'Conversation memories' },
  { id: 'workflow_synthesis', label: 'Workflow Synthesis', description: 'Generate workflows from descriptions' },
];

const KIND_LABELS: Record<string, string> = {
  openai_compatible: 'OpenAI-compatible',
  anthropic: 'Anthropic-compatible',
  ollama: 'Ollama',
};

const inputClass =
  'w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm';

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/** A provider choice as a person reads it. */
function describeRef(
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
function useProviderModels(ref: string) {
  return useQuery(
    ['myLLMModels', ref],
    () => apiClient.listMyLLMModels({ provider: ref || undefined }),
    { refetchOnWindowFocus: false, staleTime: 60_000 },
  );
}

const ModelField: React.FC<{
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
const ProviderSelect: React.FC<{
  value: string;
  onChange: (v: string) => void;
  catalog: LLMProviderCatalog | undefined;
  saved: SavedLLMProvider[];
  emptyLabel: string;
}> = ({ value, onChange, catalog, saved, emptyLabel }) => (
  <select value={value} onChange={(e) => onChange(e.target.value)} className={inputClass}>
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

interface EditorDraft extends SavedLLMProviderInput {
  api_key_set?: boolean;
}

const emptyDraft = (): EditorDraft => ({
  name: '',
  kind: 'openai_compatible',
  api_url: '',
  default_model: '',
  models: [],
  api_key: '',
});

/** Add or edit one saved provider, starting from a template if you like. */
const ProviderEditor: React.FC<{
  initial: EditorDraft;
  templates: ProviderTemplate[];
  kinds: string[];
  saving: boolean;
  onSave: (draft: EditorDraft) => void;
  onCancel: () => void;
}> = ({ initial, templates, kinds, saving, onSave, onCancel }) => {
  const [draft, setDraft] = useState<EditorDraft>(initial);
  const [test, setTest] = useState<{ ok: boolean; models: string[]; error?: string | null } | null>(null);
  const [testing, setTesting] = useState(false);
  const isNew = !initial.id;
  const set = (patch: Partial<EditorDraft>) => setDraft((d) => ({ ...d, ...patch }));

  const applyTemplate = (t: ProviderTemplate) => {
    set({
      kind: t.kind,
      api_url: t.api_url,
      name: draft.name && draft.name !== '' ? draft.name : t.label,
    });
    setTest(null);
  };

  const runTest = async () => {
    setTesting(true);
    try {
      const result = await apiClient.testLLMProvider({
        kind: draft.kind || 'openai_compatible',
        api_url: draft.api_url,
        api_key: draft.api_key || undefined,
        id: draft.id || undefined,
      });
      const models = result.models || [];
      setTest({ ok: result.ok, models, error: result.error });
      if (result.ok && models.length) {
        set({
          models,
          default_model: draft.default_model || models[0],
        });
      }
    } catch {
      setTest({ ok: false, models: [], error: 'The test request failed' });
    } finally {
      setTesting(false);
    }
  };

  const canSave = draft.name.trim() && draft.api_url.trim();
  const modelOptions = test?.models?.length ? test.models : draft.models || [];

  return (
    <div className="border border-primary-500 rounded-lg p-4 space-y-4 bg-gray-50">
      <div className="flex items-center justify-between">
        <h4 className="font-medium text-gray-900">{isNew ? 'Add a provider' : `Edit ${initial.name}`}</h4>
      </div>

      {isNew && (
        <div>
          <span className="block text-xs font-medium text-gray-600 mb-1">Start from</span>
          <div className="flex flex-wrap gap-2">
            {templates.map((t) => (
              <button
                key={t.id}
                type="button"
                title={t.description}
                onClick={() => applyTemplate(t)}
                className={`px-2.5 py-1 text-xs rounded-full border transition-colors ${
                  draft.api_url === t.api_url && draft.kind === t.kind && t.api_url
                    ? 'border-primary-500 bg-primary-100 text-primary-800'
                    : 'border-gray-300 text-gray-700 hover:border-gray-400'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <label className="block">
          <span className="block text-xs font-medium text-gray-600 mb-1">Name</span>
          <input
            className={inputClass}
            value={draft.name}
            onChange={(e) => set({ name: e.target.value })}
            placeholder="e.g. GPU box vLLM"
          />
        </label>
        <label className="block">
          <span className="block text-xs font-medium text-gray-600 mb-1">Protocol</span>
          <select className={inputClass} value={draft.kind || 'openai_compatible'} onChange={(e) => set({ kind: e.target.value })}>
            {kinds.map((k) => (
              <option key={k} value={k}>
                {KIND_LABELS[k] || k}
              </option>
            ))}
          </select>
        </label>
      </div>

      <label className="block">
        <span className="block text-xs font-medium text-gray-600 mb-1">Base URL</span>
        <input
          className={`${inputClass} font-mono`}
          value={draft.api_url}
          onChange={(e) => {
            set({ api_url: e.target.value });
            setTest(null);
          }}
          placeholder={draft.kind === 'ollama' ? 'http://host:11434' : 'http://host:8000/v1'}
        />
        <span className="block text-xs text-gray-500 mt-1">
          {draft.kind === 'ollama'
            ? 'The Ollama server root, without /api.'
            : 'Used exactly as entered: include /v1 if the server expects it. Requests go to <base>/chat/completions.'}
        </span>
      </label>

      <label className="block">
        <span className="block text-xs font-medium text-gray-600 mb-1">
          API key {draft.api_key_set && !draft.clear_api_key && <span className="text-green-600">(saved)</span>}
        </span>
        <input
          type="password"
          autoComplete="off"
          className={inputClass}
          value={draft.api_key || ''}
          onChange={(e) => set({ api_key: e.target.value, clear_api_key: false })}
          placeholder={draft.api_key_set ? 'Leave empty to keep the saved key' : 'Optional for most local servers'}
        />
        {draft.api_key_set && (
          <label className="mt-1 flex items-center gap-2 text-xs text-gray-600">
            <input
              type="checkbox"
              checked={Boolean(draft.clear_api_key)}
              onChange={(e) => set({ clear_api_key: e.target.checked, api_key: '' })}
            />
            Remove the saved key
          </label>
        )}
      </label>

      <div>
        <span className="block text-xs font-medium text-gray-600 mb-1">Default model</span>
        <div className="flex gap-2">
          <input
            className={inputClass}
            list="editor-models"
            value={draft.default_model || ''}
            onChange={(e) => set({ default_model: e.target.value })}
            placeholder="Model name the server serves"
          />
          <datalist id="editor-models">
            {modelOptions.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
          <Button type="button" variant="secondary" size="sm" onClick={runTest} loading={testing} disabled={!draft.api_url.trim()}>
            Test &amp; fetch models
          </Button>
        </div>
        {test && (
          <p className={`text-xs mt-1 flex items-center gap-1 ${test.ok ? 'text-green-600' : 'text-red-600'}`}>
            {test.ok ? <CheckCircle2 className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
            {test.ok
              ? test.models.length
                ? `Connected: ${test.models.length} model${test.models.length === 1 ? '' : 's'} available.`
                : 'Connected, but it listed no models. Enter the model name by hand.'
              : test.error}
          </p>
        )}
      </div>

      <div className="flex justify-end gap-2">
        <Button type="button" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="button" onClick={() => onSave(draft)} disabled={!canSave} loading={saving}>
          {isNew ? 'Add provider' : 'Save provider'}
        </Button>
      </div>
    </div>
  );
};

/** A saved provider as the API takes it back: the key only when changed. */
function toInput(p: SavedLLMProvider): SavedLLMProviderInput {
  return {
    id: p.id,
    name: p.name,
    kind: p.kind,
    api_url: p.api_url,
    default_model: p.default_model || null,
    models: p.models || [],
  };
}

const LLMSettingsTab: React.FC = () => {
  const queryClient = useQueryClient();
  const [provider, setProvider] = useState('');
  const [model, setModel] = useState('');
  const [apiUrl, setApiUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [temperature, setTemperature] = useState('');
  const [maxTokens, setMaxTokens] = useState('');
  const [taskModels, setTaskModels] = useState<Record<string, string>>({});
  const [taskProviders, setTaskProviders] = useState<Record<string, string>>({});
  const [showOwnKey, setShowOwnKey] = useState(false);
  const [showTasks, setShowTasks] = useState(false);
  const [editor, setEditor] = useState<EditorDraft | null>(null);
  // When the editor was opened to convert a legacy single URL.
  const [convertingLegacy, setConvertingLegacy] = useState(false);

  const { data: catalog } = useQuery('llmProviderCatalog', () => apiClient.getLLMProviderCatalog(), {
    staleTime: 5 * 60_000,
    refetchOnWindowFocus: false,
  });

  const loadForm = (data: LLMSettings) => {
    setProvider(data.llm_provider || '');
    setModel(data.llm_model || '');
    setApiUrl(data.llm_api_url || '');
    setTemperature(data.llm_temperature != null ? String(data.llm_temperature) : '');
    setMaxTokens(data.llm_max_tokens != null ? String(data.llm_max_tokens) : '');
    setTaskModels(data.llm_task_models || {});
    setTaskProviders(data.llm_task_providers || {});
    setShowOwnKey(Boolean(data.llm_api_url || data.llm_api_key_set));
    if (
      Object.keys(data.llm_task_models || {}).length ||
      Object.keys(data.llm_task_providers || {}).length
    ) {
      setShowTasks(true);
    }
  };

  const { data: settings, isLoading } = useQuery('userLLMSettings', () => apiClient.getUserLLMSettings(), {
    onSuccess: loadForm,
    refetchOnWindowFocus: false,
  });
  const saved: SavedLLMProvider[] = useMemo(() => settings?.llm_providers || [], [settings]);

  const afterWrite = (data: LLMSettings) => {
    queryClient.setQueryData('userLLMSettings', data);
    queryClient.invalidateQueries('myLLMModels');
  };

  const saveMutation = useMutation(
    (payload: Parameters<typeof apiClient.updateUserLLMSettings>[0]) => apiClient.updateUserLLMSettings(payload),
    {
      onSuccess: (data) => {
        afterWrite(data);
        loadForm(data);
        setApiKey('');
        toast.success('LLM settings saved');
      },
    },
  );

  // Saved providers are written on their own, keeping the rest of the form
  // as it is on screen (unsaved edits included).
  const providersMutation = useMutation(
    (payload: Parameters<typeof apiClient.updateUserLLMSettings>[0]) => apiClient.updateUserLLMSettings(payload),
    { onSuccess: afterWrite },
  );

  const clearMutation = useMutation(() => apiClient.clearUserLLMSettings(), {
    onSuccess: async () => {
      setApiKey('');
      const fresh = await apiClient.getUserLLMSettings();
      afterWrite(fresh);
      loadForm(fresh);
      toast.success('LLM settings reset: using the deployment defaults');
    },
    onError: () => {
      toast.error('Failed to reset LLM settings');
    },
  });

  const selectedBuiltin = catalog?.builtin.find((b) => b.id === provider);
  const selectedSaved = provider.startsWith(CUSTOM_PREFIX)
    ? saved.find((p) => `${CUSTOM_PREFIX}${p.id}` === provider)
    : undefined;
  const legacyUrlOnDefault = !provider && Boolean(apiUrl);

  const saveProvider = (draft: EditorDraft) => {
    const input: SavedLLMProviderInput = {
      id: draft.id,
      name: draft.name.trim(),
      kind: draft.kind,
      api_url: draft.api_url.trim(),
      default_model: draft.default_model?.trim() || null,
      models: draft.models || [],
      ...(draft.api_key ? { api_key: draft.api_key } : {}),
      ...(draft.clear_api_key ? { clear_api_key: true } : {}),
    };
    const list = draft.id
      ? saved.map((p) => (p.id === draft.id ? input : toInput(p)))
      : [...saved.map(toInput), input];
    const before = new Set(saved.map((p) => p.id));
    const payload: Parameters<typeof apiClient.updateUserLLMSettings>[0] = { llm_providers: list };
    providersMutation.mutate(payload, {
      onSuccess: (data) => {
        const added = (data.llm_providers || []).find((p) => !before.has(p.id));
        setEditor(null);
        toast.success(draft.id ? `Saved ${input.name}` : `Added ${input.name}`);
        // A new provider becomes the default when nothing else is chosen,
        // or when it replaces a legacy single URL.
        if (added && (convertingLegacy || !provider)) {
          const ref = `${CUSTOM_PREFIX}${added.id}`;
          setProvider(ref);
          if (convertingLegacy) {
            setApiUrl('');
            setModel('');
            saveMutation.mutate({ llm_provider: ref, llm_api_url: null, llm_api_key: null, llm_model: null });
          } else {
            toast('Selected as your default provider; press Save to keep it.', { icon: 'ℹ️' });
          }
        }
        setConvertingLegacy(false);
      },
    });
  };

  const deleteProvider = (p: SavedLLMProvider) => {
    const ref = `${CUSTOM_PREFIX}${p.id}`;
    const usedByTasks = Object.entries(taskProviders).filter(([, v]) => v === ref).map(([k]) => k);
    const usedByDefault = settings?.llm_provider === ref || provider === ref;
    const uses = [usedByDefault ? 'your default provider' : '', usedByTasks.length ? `${usedByTasks.length} task override(s)` : '']
      .filter(Boolean)
      .join(' and ');
    if (!window.confirm(`Delete "${p.name}"?${uses ? ` It is ${uses}; those will fall back to the default.` : ''}`)) {
      return;
    }
    const nextTasks = Object.fromEntries(Object.entries(settings?.llm_task_providers || {}).filter(([, v]) => v !== ref));
    const payload: Parameters<typeof apiClient.updateUserLLMSettings>[0] = {
      llm_providers: saved.filter((s) => s.id !== p.id).map(toInput),
      llm_task_providers: Object.keys(nextTasks).length ? nextTasks : null,
    };
    if (settings?.llm_provider === ref) payload.llm_provider = null;
    providersMutation.mutate(payload, {
      onSuccess: () => {
        if (provider === ref) setProvider('');
        setTaskProviders((prev) => Object.fromEntries(Object.entries(prev).filter(([, v]) => v !== ref)));
        toast.success(`Deleted ${p.name}`);
      },
    });
  };

  const handleSave = () => {
    const clean = (m: Record<string, string>) => Object.fromEntries(Object.entries(m).filter(([, v]) => v && v.trim()));
    const tm = clean(taskModels);
    const tp = clean(taskProviders);
    const isSaved = provider.startsWith(CUSTOM_PREFIX);
    saveMutation.mutate({
      llm_provider: provider || null,
      llm_model: model.trim() || null,
      // A saved provider carries its own endpoint and key.
      llm_api_url: isSaved ? null : apiUrl.trim() || null,
      ...(isSaved ? { llm_api_key: null } : apiKey ? { llm_api_key: apiKey } : {}),
      llm_temperature: temperature ? parseFloat(temperature) : null,
      llm_max_tokens: maxTokens ? parseInt(maxTokens, 10) : null,
      llm_task_models: Object.keys(tm).length ? tm : null,
      llm_task_providers: Object.keys(tp).length ? tp : null,
    });
  };

  const choose = (ref: string) => {
    if (ref !== provider) setModel('');
    setProvider(ref);
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 rounded w-1/4"></div>
          <div className="h-10 bg-gray-200 rounded"></div>
          <div className="h-10 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  const systemLabel = catalog
    ? `${describeRef(catalog.system_provider, catalog, saved)}${catalog.system_model ? ` · ${catalog.system_model}` : ''}`
    : '…';

  const card = (selected: boolean) =>
    `p-3 text-left border rounded-lg transition-all hover:shadow-md ${
      selected ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-500' : 'border-gray-200 hover:border-gray-300'
    }`;

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-2">LLM Settings</h2>
      <p className="text-sm text-gray-500 mb-6">
        Choose where your model calls go. Anything left empty uses the deployment default.
      </p>

      <div className="space-y-8">
        {/* Default provider */}
        <section>
          <h3 className="text-sm font-medium text-gray-700 mb-2">Default provider</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            <button type="button" className={card(provider === '')} onClick={() => choose('')}>
              <div className="font-medium text-sm text-gray-900 mb-1">Deployment default</div>
              <p className="text-xs text-gray-500 font-mono">{systemLabel}</p>
            </button>
            {(catalog?.builtin || []).map((b) => {
              const ready = !b.requires_api_key || b.system_configured;
              return (
                <button key={b.id} type="button" className={card(provider === b.id)} onClick={() => choose(b.id)}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-sm text-gray-900">{b.label}</span>
                    {b.id === 'ollama' ? (
                      <span className="text-xs px-1.5 py-0.5 bg-gray-200 text-gray-700 rounded">Local</span>
                    ) : ready ? (
                      <span className="text-xs px-1.5 py-0.5 bg-green-100 text-green-700 rounded">Ready</span>
                    ) : (
                      <span className="text-xs px-1.5 py-0.5 bg-amber-100 text-amber-700 rounded">Needs your key</span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 font-mono truncate">{b.default_model || hostOf(b.api_url || '')}</p>
                </button>
              );
            })}
            {saved.map((p) => {
              const ref = `${CUSTOM_PREFIX}${p.id}`;
              return (
                <button key={p.id} type="button" className={card(provider === ref)} onClick={() => choose(ref)}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-sm text-gray-900 truncate">{p.name}</span>
                    <span className="text-xs px-1.5 py-0.5 bg-primary-100 text-primary-800 rounded">Yours</span>
                  </div>
                  <p className="text-xs text-gray-500 font-mono truncate">
                    {hostOf(p.api_url)}
                    {p.default_model ? ` · ${p.default_model}` : ''}
                  </p>
                </button>
              );
            })}
            <button
              type="button"
              className="p-3 text-left border border-dashed border-gray-300 rounded-lg hover:border-primary-500 text-gray-600 flex items-center gap-2"
              onClick={() => {
                setConvertingLegacy(false);
                setEditor(emptyDraft());
              }}
            >
              <Plus className="w-4 h-4" />
              <span className="text-sm">Add a provider (vLLM, LM Studio, OpenRouter…)</span>
            </button>
          </div>
        </section>

        {editor && (
          <ProviderEditor
            key={editor.id || 'new'}
            initial={editor}
            templates={catalog?.templates || []}
            kinds={catalog?.kinds || Object.keys(KIND_LABELS)}
            saving={providersMutation.isLoading}
            onSave={saveProvider}
            onCancel={() => {
              setEditor(null);
              setConvertingLegacy(false);
            }}
          />
        )}

        {/* What the chosen provider needs */}
        <section className="space-y-4">
          {legacyUrlOnDefault && (
            <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800 flex items-start justify-between gap-3">
              <span>
                Your calls go to <span className="font-mono">{apiUrl}</span>, set as a single API URL. Save it as a
                provider to give it a name and use it per task (the key has to be entered again).
              </span>
              <Button
                type="button"
                size="sm"
                variant="secondary"
                onClick={() => {
                  setConvertingLegacy(true);
                  setEditor({ ...emptyDraft(), name: hostOf(apiUrl), api_url: apiUrl, default_model: model });
                }}
              >
                Save as provider
              </Button>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model{provider ? ` for ${describeRef(provider, catalog, saved)}` : ''}
            </label>
            <ModelField id="default" providerRef={provider || catalog?.system_provider || ''} value={model} onChange={setModel} />
          </div>

          {selectedSaved && (
            <div className="flex items-center justify-between rounded-lg border border-gray-200 p-3 text-sm">
              <div className="flex items-center gap-2 text-gray-700 min-w-0">
                <Server className="w-4 h-4 shrink-0" />
                <span className="font-mono truncate">{selectedSaved.api_url}</span>
                <span className="text-xs text-gray-500">
                  {KIND_LABELS[selectedSaved.kind] || selectedSaved.kind}
                  {selectedSaved.api_key_set ? ' · key saved' : ' · no key'}
                </span>
              </div>
              <Button type="button" size="sm" variant="ghost" icon={<Pencil className="w-3.5 h-3.5" />} onClick={() => setEditor({ ...toInput(selectedSaved), api_key: '', api_key_set: selectedSaved.api_key_set })}>
                Edit
              </Button>
            </div>
          )}

          {(selectedBuiltin || legacyUrlOnDefault) && (
            <div className="border rounded-lg">
              <button
                type="button"
                onClick={() => setShowOwnKey(!showOwnKey)}
                className="w-full px-4 py-2.5 text-left text-sm hover:bg-gray-50 rounded-lg flex justify-between"
              >
                <span className="text-gray-900">Use my own key or endpoint for {selectedBuiltin?.label || 'the default'}</span>
                <span className="text-xs text-gray-500">
                  {settings?.llm_api_key_set ? 'key set' : selectedBuiltin?.requires_api_key && !selectedBuiltin.system_configured ? 'required' : 'optional'}
                </span>
              </button>
              {showOwnKey && (
                <div className="px-4 pb-4 pt-2 border-t grid grid-cols-1 md:grid-cols-2 gap-3">
                  <label className="block">
                    <span className="block text-xs font-medium text-gray-600 mb-1">
                      API key {settings?.llm_api_key_set && <span className="text-green-600">(set)</span>}
                    </span>
                    <input
                      type="password"
                      autoComplete="off"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={settings?.llm_api_key_set ? 'Leave empty to keep it' : 'Your key'}
                      className={inputClass}
                    />
                  </label>
                  <label className="block">
                    <span className="block text-xs font-medium text-gray-600 mb-1">Endpoint override</span>
                    <input
                      type="url"
                      value={apiUrl}
                      onChange={(e) => setApiUrl(e.target.value)}
                      placeholder={selectedBuiltin?.api_url || 'https://…/v1'}
                      className={`${inputClass} font-mono`}
                    />
                  </label>
                </div>
              )}
            </div>
          )}
        </section>

        {/* Saved providers */}
        {saved.length > 0 && (
          <section>
            <h3 className="text-sm font-medium text-gray-700 mb-2">Your providers</h3>
            <ul className="divide-y divide-gray-200 border border-gray-200 rounded-lg">
              {saved.map((p) => (
                <li key={p.id} className="flex items-center justify-between gap-3 px-3 py-2 text-sm">
                  <div className="min-w-0">
                    <div className="text-gray-900 truncate">{p.name}</div>
                    <div className="text-xs text-gray-500 font-mono truncate">
                      {p.api_url}
                      {p.default_model ? ` · ${p.default_model}` : ''}
                      {p.api_key_set ? ' · key saved' : ''}
                    </div>
                  </div>
                  <div className="flex gap-1 shrink-0">
                    <Button type="button" size="sm" variant="ghost" aria-label={`Edit ${p.name}`} icon={<Pencil className="w-3.5 h-3.5" />} onClick={() => setEditor({ ...toInput(p), api_key: '', api_key_set: p.api_key_set })} />
                    <Button type="button" size="sm" variant="ghost" aria-label={`Delete ${p.name}`} icon={<Trash2 className="w-3.5 h-3.5" />} onClick={() => deleteProvider(p)} />
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* Generation */}
        <section className="grid grid-cols-2 gap-4">
          <label className="block">
            <span className="block text-sm font-medium text-gray-700 mb-1">Temperature</span>
            <input type="number" step="0.1" min="0" max="2" value={temperature} onChange={(e) => setTemperature(e.target.value)} placeholder="0.7" className={inputClass} />
            <span className="text-xs text-gray-500 mt-1 block">0.0 - 2.0</span>
          </label>
          <label className="block">
            <span className="block text-sm font-medium text-gray-700 mb-1">Max Tokens</span>
            <input type="number" min="1" max="32000" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} placeholder="Provider default" className={inputClass} />
            <span className="text-xs text-gray-500 mt-1 block">Max response length</span>
          </label>
        </section>

        {/* Per task */}
        <section className="border rounded-lg">
          <button
            type="button"
            onClick={() => setShowTasks(!showTasks)}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-gray-50 rounded-lg"
          >
            <div>
              <span className="font-medium text-gray-900">Per-task providers and models</span>
              <p className="text-xs text-gray-500 mt-0.5">Send a task somewhere other than your default, e.g. titles to a small local model.</p>
            </div>
            <span className="text-xs text-gray-500">
              {Object.values(taskProviders).filter(Boolean).length + Object.values(taskModels).filter(Boolean).length || 'none'} set
            </span>
          </button>
          {showTasks && (
            <div className="px-4 pb-4 border-t space-y-4 pt-3">
              {LLM_TASK_TYPES.map((task) => {
                const ref = taskProviders[task.id] || '';
                return (
                  <div key={task.id}>
                    <div className="text-sm font-medium text-gray-700 mb-1">
                      {task.label} <span className="text-xs font-normal text-gray-500">{task.description}</span>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <ProviderSelect
                        value={ref}
                        onChange={(v) => setTaskProviders((prev) => ({ ...prev, [task.id]: v }))}
                        catalog={catalog}
                        saved={saved}
                        emptyLabel={`Your default (${describeRef(provider, catalog, saved)})`}
                      />
                      <ModelField
                        id={task.id}
                        providerRef={ref || provider || catalog?.system_provider || ''}
                        value={taskModels[task.id] || ''}
                        onChange={(v) => setTaskModels((prev) => ({ ...prev, [task.id]: v }))}
                        placeholder={ref && ref !== provider ? 'That provider\'s default' : model ? `Your default: ${model}` : undefined}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <div className="flex justify-between pt-4 border-t">
          <Button variant="ghost" onClick={() => clearMutation.mutate()} disabled={clearMutation.isLoading}>
            Reset to Defaults
          </Button>
          <Button onClick={handleSave} loading={saveMutation.isLoading}>
            Save Settings
          </Button>
        </div>
      </div>
    </div>
  );
};

export default LLMSettingsTab;
