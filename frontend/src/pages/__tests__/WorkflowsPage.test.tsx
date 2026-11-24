import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';

import WorkflowExecutionHistoryPage from '../WorkflowExecutionHistoryPage';
import WorkflowEditorPage from '../WorkflowEditorPage';
import WorkflowsPage from '../WorkflowsPage';

jest.mock('../../components/workflows/WorkflowEditor', () => () => null);

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

jest.mock('../../services/api', () => ({
  __esModule: true,
  default: {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
  },
  apiClient: { listWorkflowExecutions: jest.fn() },
}));

const api = require('../../services/api').default;
const { apiClient } = require('../../services/api');

const renderPage = (initialEntry = '/workflows') =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { cacheTime: 0 } } })}>
      <MemoryRouter
        initialEntries={[initialEntry]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/workflows" element={<WorkflowsPage />} />
          <Route path="/workflows/:workflowId/executions" element={<WorkflowExecutionHistoryPage />} />
          <Route path="/workflows/:id/edit" element={<WorkflowEditorPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );

describe('WorkflowsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    api.post.mockResolvedValue({ data: { execution_id: 'exec-queued', status: 'pending' } });
    apiClient.listWorkflowExecutions.mockImplementation(async (_workflowId: string, { offset }: { offset: number }) => ({
      executions: [{
        id: offset === 0 ? 'exec-1' : 'exec-2',
        workflow_name: 'Compiler workflow',
        trigger_type: 'manual',
        status: offset === 0 ? 'completed' : 'failed',
        progress: offset === 0 ? 100 : 35,
        error: offset === 0 ? null : 'Could not collect benchmark results',
        created_at: '2026-04-21T12:01:00Z',
      }],
      total: 26,
    }));
    api.get.mockImplementation(async (path: string) => {
      if (path === '/workflows') {
        return {
          data: {
            workflows: [
              {
                id: 'workflow-1',
                name: 'Compiler workflow',
                description: 'Track regressions',
                is_active: true,
                trigger_config: { type: 'manual' },
                node_count: 3,
                execution_count: 4,
                created_at: '2026-04-21T12:00:00Z',
                updated_at: '2026-04-21T12:30:00Z',
              },
            ],
          },
        };
      }
      if (path === '/workflows/workflow-1') {
        return {
          data: { id: 'workflow-1', name: 'Compiler workflow', is_active: true, nodes: [], edges: [] },
        };
      }
      if (path === '/workflows/executions/exec-1' || path === '/workflows/executions/exec-queued') {
        const queued = path.endsWith('/exec-queued');
        return {
          data: {
            id: queued ? 'exec-queued' : 'exec-1',
            workflow_id: 'workflow-1',
            trigger_type: 'manual',
            status: queued ? 'pending' : 'completed',
            progress: queued ? 0 : 100,
            current_node_id: queued ? null : 'finalize',
            error: null,
            created_at: '2026-04-21T12:01:00Z',
            started_at: '2026-04-21T12:02:00Z',
            completed_at: '2026-04-21T12:05:00Z',
            context: {},
          },
        };
      }
      throw new Error(`Unexpected GET ${path}`);
    });
  });

  test('loads and highlights a workflow execution from executionId query param', async () => {
    renderPage('/workflows?executionId=exec-1');

    expect(await screen.findByText(/Execution Drilldown/i)).toBeInTheDocument();
    expect(screen.getByText(/Compiler workflow · exec-1/i)).toBeInTheDocument();
    expect(screen.getByText(/Current node: finalize/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Open executions/i }));
    expect(await screen.findByRole('heading', { name: 'Execution history' })).toBeInTheDocument();
    const execution = await screen.findByRole('link', { name: 'Open execution' });
    expect(execution).toHaveAttribute('href', '/workflows?executionId=exec-1');
    expect(apiClient.listWorkflowExecutions).toHaveBeenCalledWith('workflow-1', { limit: 25, offset: 0 });
    fireEvent.click(execution);
    expect(await screen.findByText(/Current node: finalize/i)).toBeInTheDocument();

    await waitFor(() => expect(api.get).toHaveBeenCalledWith('/workflows/executions/exec-1', expect.anything()));
  });

  test('shows the exact queued execution after running from the list', async () => {
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: 'Run workflow' }));

    const execution = await screen.findByRole('link', { name: 'View execution' });
    expect(screen.getByText('Queued')).toBeInTheDocument();
    expect(api.post).toHaveBeenCalledWith('/workflows/workflow-1/execute/async', {
      trigger_type: 'manual', trigger_data: {}, inputs: {},
    });
    expect(execution).toHaveAttribute('href', '/workflows?executionId=exec-queued');

    fireEvent.click(execution);
    expect(await screen.findByRole('heading', { name: /Compiler workflow · exec-queued/ })).toBeInTheDocument();
    expect(api.get).toHaveBeenCalledWith('/workflows/executions/exec-queued', expect.anything());
  });

  test('opens the queued execution from the editor without leaving or saving the draft', async () => {
    renderPage('/workflows/workflow-1/edit');
    const name = await screen.findByDisplayValue('Compiler workflow');
    fireEvent.change(name, { target: { value: 'Draft workflow name' } });

    fireEvent.click(screen.getByRole('button', { name: 'Run saved version' }));

    const execution = await screen.findByRole('link', { name: /View execution/ });
    expect(execution).toHaveAttribute('href', '/workflows?executionId=exec-queued');
    expect(execution).toHaveAttribute('target', '_blank');
    fireEvent.click(execution);

    expect(screen.getByDisplayValue('Draft workflow name')).toBeInTheDocument();
    expect(screen.getByText('Unsaved changes')).toBeInTheDocument();
    expect(api.put).not.toHaveBeenCalled();
    expect(api.post).toHaveBeenCalledTimes(1);
  });

  test('refreshes a queued execution and stops polling when it completes', async () => {
    jest.useFakeTimers();
    try {
      const defaultGet = api.get.getMockImplementation();
      let executionReads = 0;
      api.get.mockImplementation(async (path: string) => {
        const response = await defaultGet(path);
        if (path === '/workflows/executions/exec-queued' && ++executionReads > 1) {
          return { data: { ...response.data, status: 'completed', progress: 100 } };
        }
        return response;
      });

      renderPage('/workflows?executionId=exec-queued');
      expect(await screen.findByText('Queued')).toBeInTheDocument();

      await act(async () => { jest.advanceTimersByTime(5000); });
      expect(await screen.findByText('completed')).toBeInTheDocument();

      await act(async () => { jest.advanceTimersByTime(10000); });
      expect(executionReads).toBe(2);
    } finally {
      jest.useRealTimers();
    }
  });

  test('can browse older executions on a workflow history page', async () => {
    renderPage('/workflows/workflow-1/executions');
    fireEvent.click(await screen.findByRole('button', { name: 'Next' }));

    expect(await screen.findByText('Could not collect benchmark results')).toBeInTheDocument();
    expect(apiClient.listWorkflowExecutions).toHaveBeenLastCalledWith('workflow-1', { limit: 25, offset: 25 });
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
  });
});
