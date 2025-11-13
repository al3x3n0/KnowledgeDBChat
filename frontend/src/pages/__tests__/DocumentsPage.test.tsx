/**
 * Tests for DocumentsPage component
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import DocumentsPage from '../DocumentsPage';
import { AuthProvider } from '../../contexts/AuthContext';

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
    getDocumentSources: jest.fn().mockResolvedValue([]),
    deleteDocument: jest.fn().mockResolvedValue({}),
    reprocessDocument: jest.fn().mockResolvedValue({}),
  },
}));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
  },
});

const renderWithProviders = (component: React.ReactElement) => {
  return render(
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          {component}
        </AuthProvider>
      </QueryClientProvider>
    </BrowserRouter>
  );
};

describe('DocumentsPage', () => {
  it('renders documents page', async () => {
    renderWithProviders(<DocumentsPage />);
    
    await waitFor(() => {
      expect(screen.getByText(/documents/i) || screen.getByText(/upload/i)).toBeInTheDocument();
    });
  });

  it('displays document list', async () => {
    renderWithProviders(<DocumentsPage />);
    
    // Should show document list or empty state
    await waitFor(() => {
      expect(screen.getByText(/no documents/i) || screen.getByText(/upload/i)).toBeInTheDocument();
    });
  });
});

