import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';

import RoutingObservabilityPage from '../RoutingObservabilityPage';

jest.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({
    user: { id: 'user-1' },
  }),
}));

jest.mock('../../services/api', () => ({
  apiClient: {
    getLLMRoutingSummary: jest.fn(),
    listLLMUsageEvents: jest.fn(),
    getLLMRoutingExperimentRecommendation: jest.fn(),
  },
}));

const apiClient = require('../../services/api').apiClient;

const renderPage = (initialEntry = '/usage/routing') => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
    },
  });

  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <QueryClientProvider client={queryClient}>
        <RoutingObservabilityPage />
      </QueryClientProvider>
    </MemoryRouter>
  );
};

describe('RoutingObservabilityPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.getLLMRoutingSummary.mockResolvedValue({
      items: [
        {
          provider: 'openai',
          model: 'gpt-5.4',
          task_type: 'planner',
          routing_tier: 'balanced',
          routing_requested_tier: 'balanced',
          routing_attempt: 1,
          routing_experiment_id: 'exp-1',
          routing_experiment_variant_id: 'var-1',
          request_count: 10,
          success_count: 9,
          error_count: 1,
          success_rate: 0.9,
          total_tokens: 1000,
          avg_latency_ms: 800,
          p50_latency_ms: 700,
          p95_latency_ms: 1500,
        },
        {
          provider: 'anthropic',
          model: 'claude',
          task_type: 'planner',
          routing_tier: 'premium',
          routing_requested_tier: 'premium',
          routing_attempt: 1,
          routing_experiment_id: null,
          routing_experiment_variant_id: null,
          request_count: 8,
          success_count: 8,
          error_count: 0,
          success_rate: 1,
          total_tokens: 800,
          avg_latency_ms: 500,
          p50_latency_ms: 450,
          p95_latency_ms: 900,
        },
      ],
      scanned_events: 2,
      truncated: false,
    });
    apiClient.listLLMUsageEvents.mockResolvedValue({ items: [] });
    apiClient.getLLMRoutingExperimentRecommendation.mockResolvedValue({});
  });

  test('prefilters and auto-selects the routing row from query params', async () => {
    renderPage('/usage/routing?provider=openai&model=gpt-5.4&routing_tier=balanced&experiment_id=exp-1&variant_id=var-1');

    expect(await screen.findByText(/Selected route/i)).toBeInTheDocument();
    expect(screen.getByText(/openai · gpt-5\.4 · planner · tier: balanced · attempt: 1/i)).toBeInTheDocument();
    expect(screen.queryByText(/anthropic · claude/i)).not.toBeInTheDocument();

    await waitFor(() =>
      expect(apiClient.getLLMRoutingExperimentRecommendation).toHaveBeenCalledWith(
        expect.objectContaining({ experiment_id: 'exp-1' })
      )
    );
  });
});
