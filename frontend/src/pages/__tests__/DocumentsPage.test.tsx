/**
 * Tests for DocumentsPage component
 */

import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import DocumentsPage from '../DocumentsPage';
import { AuthProvider } from '../../contexts/AuthContext';
import { apiClient } from '../../services/api';

// Mock the auth context
jest.mock('../../contexts/AuthContext', () => ({
  ...jest.requireActual('../../contexts/AuthContext'),
  useAuth: () => ({
    user: { id: '1', username: 'testuser', role: 'user' },
    loading: false,
  }),
}));

// Mock the API client
jest.mock('../../services/api', () => ({
  apiClient: {
    getDocuments: jest.fn().mockResolvedValue([]),
    // The folder tree loads alongside the list. Unmocked it rejects, which
    // react-query swallows into a console error rather than a failure — the
    // test would pass while the page logged noise on every run.
    getDocumentFolderTree: jest.fn().mockResolvedValue({ system: [], folders: [] }),
    createDocumentFolder: jest.fn(),
    updateDocumentFolder: jest.fn(),
    deleteDocumentFolder: jest.fn(),
    addDocumentsToFolder: jest.fn(),
    getDocumentSources: jest.fn().mockResolvedValue([]),
    getActiveGitSources: jest.fn().mockResolvedValue([]),
    getGitComparisonJobs: jest.fn().mockResolvedValue([]),
    listPersonas: jest.fn().mockResolvedValue({ items: [], total: 0 }),
    deleteDocument: jest.fn().mockResolvedValue({}),
    reprocessDocument: jest.fn().mockResolvedValue({}),
  },
}));

const renderWithProviders = (component: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return render(
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          {component}
        </AuthProvider>
      </QueryClientProvider>
    </BrowserRouter>
  );
};

describe('DocumentsPage', () => {
  beforeEach(() => {
    (apiClient.getDocuments as jest.Mock).mockResolvedValue([]);
    (apiClient.getDocumentSources as jest.Mock).mockResolvedValue([]);
    (apiClient.getActiveGitSources as jest.Mock).mockResolvedValue([]);
    (apiClient.getGitComparisonJobs as jest.Mock).mockResolvedValue([]);
    (apiClient.listPersonas as jest.Mock).mockResolvedValue({ items: [], total: 0 });
  });

  it('renders documents page', async () => {
    renderWithProviders(<DocumentsPage />);
    
    expect(await screen.findByRole('button', { name: /documents/i })).toBeInTheDocument();
  });

  it('displays document list', async () => {
    renderWithProviders(<DocumentsPage />);
    
    // Should show document list or empty state
    expect(await screen.findByRole('button', { name: /upload/i })).toBeInTheDocument();
  });

  it('links only the active source selected in the source filter to the MCP builder', async () => {
    jest.mocked(apiClient.getDocumentSources).mockResolvedValue([
      { id: 'source-1', name: 'Lab corpus', is_active: true, source_type: 'file', config: {}, created_at: '', updated_at: '' },
      { id: 'source-2', name: 'Archived corpus', is_active: false, source_type: 'file', config: {}, created_at: '', updated_at: '' },
    ]);
    renderWithProviders(<DocumentsPage />);
    await screen.findByRole('option', { name: 'Archived corpus' });
    const sourceFilter = screen.getByRole('combobox', { name: 'Document source' });
    expect(screen.queryByRole('link', { name: 'Create MCP from source' })).toBeNull();
    await act(async () => {
      fireEvent.change(sourceFilter, { target: { value: 'source-2' } });
    });
    expect(screen.queryByRole('link', { name: 'Create MCP from source' })).toBeNull();
    await act(async () => {
      fireEvent.change(sourceFilter, { target: { value: 'source-1' } });
    });
    expect(screen.getByRole('link', { name: 'Create MCP from source' })).toHaveAttribute(
      'href', '/mcp-config?source=source-1'
    );
  });
});
