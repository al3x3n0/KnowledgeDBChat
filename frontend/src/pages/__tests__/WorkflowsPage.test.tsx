import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';

import WorkflowsPage from '../WorkflowsPage';

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
}));

const api = require('../../services/api').default;

const renderPage = (initialEntry = '/workflows') =>
  render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/workflows" element={<WorkflowsPage />} />
        <Route path="/workflows/:workflowId/executions" element={<div>Executions page</div>} />
      </Routes>
    </MemoryRouter>
  );

describe('WorkflowsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
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
      if (path === '/workflows/executions/exec-1') {
        return {
          data: {
            id: 'exec-1',
            workflow_id: 'workflow-1',
            trigger_type: 'manual',
            status: 'completed',
            progress: 100,
            current_node_id: 'finalize',
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
    expect(await screen.findByText('Executions page')).toBeInTheDocument();

    await waitFor(() => expect(api.get).toHaveBeenCalledWith('/workflows/executions/exec-1'));
  });
});
