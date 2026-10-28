/**
 * Which provider and model this one chat uses.
 *
 * A chat could be pinned only by AI Hub's "use in chat", and then only to
 * Ollama; the chat page showed the pin and could clear it, nothing more. This
 * lets any chat be pinned to any provider a person may choose -- built-in or
 * one they saved -- through the session's own `llm_task_providers.chat` and
 * `llm_task_models.chat`, which the chat service already lays over the
 * person's settings for that session alone.
 */

import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { Cpu, X } from 'lucide-react';

import Button from '../common/Button';
import { apiClient } from '../../services/api';
import { describeRef, ModelField, ProviderSelect } from '../settings/providerControls';

type SessionMeta = Record<string, any>;

/** The session's metadata with this chat's pin replaced (or removed). */
export function withChatPin(meta: SessionMeta, provider: string, model: string): SessionMeta {
  const next: SessionMeta = { ...(meta || {}) };
  const providers = { ...(next.llm_task_providers || {}) };
  const models = { ...(next.llm_task_models || {}) };
  if (provider) providers.chat = provider;
  else delete providers.chat;
  if (model) models.chat = model;
  else delete models.chat;
  if (Object.keys(providers).length) next.llm_task_providers = providers;
  else delete next.llm_task_providers;
  if (Object.keys(models).length) next.llm_task_models = models;
  else delete next.llm_task_models;
  // An AI Hub adapter is a pin to one Ollama model; any other choice ends it.
  delete next.ai_hub;
  return next;
}

const ChatModelPicker: React.FC<{
  meta: SessionMeta;
  saving: boolean;
  onApply: (nextMeta: SessionMeta) => void;
}> = ({ meta, saving, onApply }) => {
  const pinnedProvider: string = meta?.llm_task_providers?.chat || '';
  const pinnedModel: string = meta?.llm_task_models?.chat || '';
  const adapterId: string | undefined = meta?.ai_hub?.adapter_id;
  const isPinned = Boolean(pinnedProvider || pinnedModel || adapterId);

  const [open, setOpen] = useState(false);
  const [provider, setProvider] = useState(pinnedProvider);
  const [model, setModel] = useState(pinnedModel);

  // Only fetched once the picker is used: most chats are never pinned.
  const { data: catalog } = useQuery('llmProviderCatalog', () => apiClient.getLLMProviderCatalog(), {
    enabled: open || isPinned,
    staleTime: 5 * 60_000,
    refetchOnWindowFocus: false,
  });
  const { data: settings } = useQuery('userLLMSettings', () => apiClient.getUserLLMSettings(), {
    enabled: open || isPinned,
    refetchOnWindowFocus: false,
  });
  const saved = settings?.llm_providers || [];

  const label = adapterId
    ? `AI Hub · ${pinnedModel || String(adapterId).slice(0, 8)}`
    : [pinnedProvider ? describeRef(pinnedProvider, catalog, saved) : '', pinnedModel].filter(Boolean).join(' · ');

  const startEditing = () => {
    setProvider(pinnedProvider);
    setModel(pinnedModel);
    setOpen(true);
  };

  const apply = (nextProvider: string, nextModel: string) => {
    onApply(withChatPin(meta, nextProvider, nextModel.trim()));
    setOpen(false);
  };

  return (
    <div className="mt-2">
      <div className="inline-flex items-center gap-2 text-xs">
        <button
          type="button"
          onClick={() => (open ? setOpen(false) : startEditing())}
          className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 ${
            isPinned
              ? 'bg-indigo-50 text-indigo-700 border-indigo-100'
              : 'border-gray-300 text-gray-600 hover:border-gray-400'
          }`}
          aria-expanded={open}
          title="Choose the provider and model for this chat"
        >
          <Cpu className="w-3 h-3" />
          <span className="font-medium">{isPinned ? label || 'Pinned' : 'Your default model'}</span>
        </button>
        {isPinned && (
          <button
            type="button"
            className="text-indigo-600 hover:text-indigo-800"
            onClick={() => apply('', '')}
            disabled={saving}
            title="Use your default model in this chat"
            aria-label="Clear pinned model"
          >
            <X className="w-3 h-3" />
          </button>
        )}
      </div>

      {open && (
        <div className="mt-2 w-full max-w-xl rounded-lg border border-gray-200 bg-gray-50 p-3 space-y-2">
          <div className="text-xs text-gray-600">This chat only. Your settings stay as they are.</div>
          <ProviderSelect
            value={provider}
            onChange={(v) => {
              if (v !== provider) setModel('');
              setProvider(v);
            }}
            catalog={catalog}
            saved={saved}
            emptyLabel="Your default provider"
          />
          <ModelField
            id="chat-pin"
            providerRef={provider || settings?.llm_task_providers?.chat || settings?.llm_provider || catalog?.system_provider || ''}
            value={model}
            onChange={setModel}
          />
          <div className="flex justify-end gap-2">
            <Button type="button" size="sm" variant="ghost" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button type="button" size="sm" onClick={() => apply(provider, model)} loading={saving}>
              Use in this chat
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatModelPicker;
