/**
 * Tests for ChatPage component
 */

import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import ChatPage from '../ChatPage';
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
    getChatSessions: jest.fn().mockResolvedValue([]),
    getChatSession: jest.fn().mockResolvedValue(null),
    createChatSession: jest.fn().mockResolvedValue({ id: '1', title: 'New Session' }),
    updateChatSession: jest.fn().mockResolvedValue({ id: '1', title: 'New Session' }),
    deleteChatSession: jest.fn().mockResolvedValue({}),
    sendMessage: jest.fn().mockResolvedValue({ id: '1', content: 'Response', role: 'assistant' }),
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

describe('ChatPage', () => {
  it('renders chat interface', async () => {
    renderWithProviders(<ChatPage />);

    const apiClient = require('../../services/api').apiClient;
    const newChatButton = await screen.findByRole('button', { name: /^new chat$/i });
    fireEvent.click(newChatButton);

    await waitFor(() => {
      expect(apiClient.createChatSession).toHaveBeenCalled();
    });
  });

  it('displays session list', async () => {
    renderWithProviders(<ChatPage />);
    
    // Should show session list or empty state
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /start new chat/i })).toBeInTheDocument();
    });
  });
});
