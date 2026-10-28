import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';

import ChatModelPicker, { withChatPin } from '../ChatModelPicker';

const mockApi = {
  getLLMProviderCatalog: jest.fn(),
  getUserLLMSettings: jest.fn(),
  listMyLLMModels: jest.fn(),
};

jest.mock('../../../services/api', () => ({
  apiClient: new Proxy({}, { get: (_t, name: string) => (mockApi as any)[name] }),
}));

beforeEach(() => {
  jest.clearAllMocks();
  mockApi.getLLMProviderCatalog.mockResolvedValue({
    system_provider: 'deepseek',
    system_model: 'deepseek-v4-pro',
    builtin: [
      { id: 'deepseek', label: 'DeepSeek', api_url: '', default_model: 'deepseek-v4-pro', models: [], system_configured: true, requires_api_key: true },
    ],
    templates: [],
    kinds: ['openai_compatible'],
  });
  mockApi.getUserLLMSettings.mockResolvedValue({
    llm_provider: null,
    llm_model: null,
    llm_api_url: null,
    llm_api_key_set: false,
    llm_temperature: null,
    llm_max_tokens: null,
    llm_task_models: null,
    llm_task_providers: null,
    llm_providers: [
      { id: 'gpu', name: 'GPU box', kind: 'openai_compatible', api_url: 'http://gpu:8000/v1', default_model: 'qwen3', models: [], api_key_set: false },
    ],
  });
  mockApi.listMyLLMModels.mockResolvedValue({ provider: 'custom:gpu', models: ['qwen3'], default_model: 'qwen3' });
});

function renderPicker(meta: Record<string, any>, onApply = jest.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <ChatModelPicker meta={meta} saving={false} onApply={onApply} />
    </QueryClientProvider>,
  );
  return onApply;
}

test('withChatPin sets, replaces and clears only the chat pin', () => {
  const meta = {
    other: 1,
    ai_hub: { adapter_id: 'a1' },
    llm_task_providers: { chat: 'ollama', summarization: 'openai' },
    llm_task_models: { chat: 'adapter-model' },
  };
  expect(withChatPin(meta, 'custom:gpu', 'qwen3')).toEqual({
    other: 1,
    llm_task_providers: { chat: 'custom:gpu', summarization: 'openai' },
    llm_task_models: { chat: 'qwen3' },
  });
  expect(withChatPin(meta, '', '')).toEqual({
    other: 1,
    llm_task_providers: { summarization: 'openai' },
  });
});

test('a chat can be pinned to a saved provider', async () => {
  const onApply = renderPicker({});
  fireEvent.click(screen.getByRole('button', { name: /Your default model/ }));
  const select = await screen.findByRole('combobox', { name: 'Provider' });
  await screen.findByRole('option', { name: 'GPU box' });
  fireEvent.change(select, { target: { value: 'custom:gpu' } });
  fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'qwen3' } });
  fireEvent.click(screen.getByRole('button', { name: /Use in this chat/ }));
  expect(onApply).toHaveBeenCalledWith({
    llm_task_providers: { chat: 'custom:gpu' },
    llm_task_models: { chat: 'qwen3' },
  });
});

test('a pinned chat names its saved provider and can be cleared', async () => {
  const onApply = renderPicker({ llm_task_providers: { chat: 'custom:gpu' }, llm_task_models: { chat: 'qwen3' } });
  expect(await screen.findByText('GPU box · qwen3')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Clear pinned model' }));
  expect(onApply).toHaveBeenCalledWith({});
});
