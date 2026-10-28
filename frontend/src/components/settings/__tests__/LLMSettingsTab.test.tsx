import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';

import LLMSettingsTab from '../LLMSettingsTab';

const mockApi = {
  getLLMProviderCatalog: jest.fn(),
  getUserLLMSettings: jest.fn(),
  updateUserLLMSettings: jest.fn(),
  clearUserLLMSettings: jest.fn(),
  listMyLLMModels: jest.fn(),
  testLLMProvider: jest.fn(),
};

jest.mock('../../../services/api', () => ({
  apiClient: new Proxy({}, { get: (_t, name: string) => (mockApi as any)[name] }),
}));

jest.mock('react-hot-toast', () => {
  const toast = Object.assign(jest.fn(), { success: jest.fn(), error: jest.fn() });
  return { __esModule: true, default: toast };
});

const catalog = {
  system_provider: 'deepseek',
  system_model: 'deepseek-v4-pro',
  builtin: [
    { id: 'deepseek', label: 'DeepSeek', api_url: 'https://api.deepseek.com/v1', default_model: 'deepseek-v4-pro', models: ['deepseek-v4-pro'], system_configured: true, requires_api_key: true },
    { id: 'openai', label: 'OpenAI', api_url: 'https://api.openai.com/v1', default_model: 'gpt-4o', models: ['gpt-4o'], system_configured: false, requires_api_key: true },
    { id: 'ollama', label: 'Ollama', api_url: 'http://localhost:11434', default_model: 'llama2', models: [], system_configured: false, requires_api_key: false },
  ],
  templates: [
    { id: 'vllm', label: 'vLLM', kind: 'openai_compatible', api_url: 'http://localhost:8000/v1', requires_api_key: false, description: '' },
  ],
  kinds: ['openai_compatible', 'anthropic', 'ollama'],
};

const emptySettings = {
  llm_provider: null,
  llm_model: null,
  llm_api_url: null,
  llm_api_key_set: false,
  llm_temperature: null,
  llm_max_tokens: null,
  llm_task_models: null,
  llm_task_providers: null,
  llm_providers: [],
};

function renderTab() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <LLMSettingsTab />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  mockApi.getLLMProviderCatalog.mockResolvedValue(catalog);
  mockApi.getUserLLMSettings.mockResolvedValue(emptySettings);
  mockApi.listMyLLMModels.mockResolvedValue({ provider: 'deepseek', models: [], default_model: null });
});

test('shows every provider the backend offers, with whether it is ready', async () => {
  renderTab();
  expect(await screen.findByText('OpenAI')).toBeInTheDocument();
  expect(screen.getByText('Deployment default')).toBeInTheDocument();
  expect(screen.getByText('deepseek-v4-pro', { selector: 'p' })).toBeInTheDocument();
  expect(screen.getByText('Needs your key')).toBeInTheDocument();
  expect(screen.getByText('Ready')).toBeInTheDocument();
});

test('a provider added from a template is saved with its key and becomes selectable', async () => {
  mockApi.testLLMProvider.mockResolvedValue({ ok: true, models: ['qwen3-32b', 'tiny'], error: null });
  mockApi.updateUserLLMSettings.mockImplementation(async (payload: any) => ({
    ...emptySettings,
    llm_providers: payload.llm_providers.map((p: any) => ({
      id: p.id || 'abcd1234',
      name: p.name,
      kind: p.kind,
      api_url: p.api_url,
      default_model: p.default_model,
      models: p.models,
      api_key_set: Boolean(p.api_key),
    })),
  }));

  renderTab();
  fireEvent.click(await screen.findByText(/Add a provider/));
  fireEvent.click(screen.getByRole('button', { name: 'vLLM' }));
  expect(screen.getByDisplayValue('http://localhost:8000/v1')).toBeInTheDocument();

  fireEvent.change(screen.getByPlaceholderText('Optional for most local servers'), { target: { value: 'sk-local' } });
  fireEvent.click(screen.getByRole('button', { name: /Test & fetch models/ }));
  expect(await screen.findByText(/Connected: 2 models available/)).toBeInTheDocument();
  expect(screen.getByDisplayValue('qwen3-32b')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Add provider' }));
  await waitFor(() => expect(mockApi.updateUserLLMSettings).toHaveBeenCalled());
  const payload = mockApi.updateUserLLMSettings.mock.calls[0][0];
  expect(payload.llm_providers).toEqual([
    expect.objectContaining({
      name: 'vLLM',
      kind: 'openai_compatible',
      api_url: 'http://localhost:8000/v1',
      default_model: 'qwen3-32b',
      api_key: 'sk-local',
    }),
  ]);
  // Only the providers were written; the rest of the form waits for Save.
  expect(payload).not.toHaveProperty('llm_temperature');
  expect(await screen.findByText('Yours')).toBeInTheDocument();
});

test('saving with a saved provider chosen sends its reference and no endpoint', async () => {
  mockApi.getUserLLMSettings.mockResolvedValue({
    ...emptySettings,
    llm_api_url: 'http://old:9000/v1',
    llm_providers: [
      { id: 'gpu', name: 'GPU box', kind: 'openai_compatible', api_url: 'http://gpu:8000/v1', default_model: 'qwen3', models: [], api_key_set: true },
    ],
  });
  mockApi.updateUserLLMSettings.mockImplementation(async () => emptySettings);

  renderTab();
  fireEvent.click(await screen.findByText('GPU box', { selector: 'span' }));
  fireEvent.click(screen.getByRole('button', { name: /Save Settings/ }));
  await waitFor(() => expect(mockApi.updateUserLLMSettings).toHaveBeenCalled());
  const payload = mockApi.updateUserLLMSettings.mock.calls[0][0];
  expect(payload.llm_provider).toBe('custom:gpu');
  expect(payload.llm_api_url).toBeNull();
  expect(payload.llm_api_key).toBeNull();
});
