/**
 * Tests for ChatPage component
 */

import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import ChatPage from '../ChatPage';
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
    getChatSessions: jest.fn().mockResolvedValue([]),
    getChatSession: jest.fn().mockResolvedValue(null),
    createChatSession: jest.fn().mockResolvedValue({ id: '1', title: 'New Session' }),
    updateChatSession: jest.fn().mockResolvedValue({ id: '1', title: 'New Session' }),
    deleteChatSession: jest.fn().mockResolvedValue({}),
    sendMessage: jest.fn().mockResolvedValue({ id: '1', content: 'Response', role: 'assistant' }),
  },
}));

const renderWithProviders = (component: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
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

describe('ChatPage', () => {
  beforeEach(() => {
    window.history.replaceState({}, '', '/chat');
    (apiClient.getChatSessions as jest.Mock).mockResolvedValue([]);
    (apiClient.getChatSession as jest.Mock).mockResolvedValue(null);
    (apiClient.createChatSession as jest.Mock).mockResolvedValue({ id: '1', title: 'New Session' });
    (apiClient.updateChatSession as jest.Mock).mockResolvedValue({ id: '1', title: 'New Session' });
    (apiClient.deleteChatSession as jest.Mock).mockResolvedValue({});
    (apiClient.sendMessage as jest.Mock).mockResolvedValue({ id: '1', content: 'Response', role: 'assistant' });
  });

  it('renders chat interface', async () => {
    renderWithProviders(<ChatPage />);

    const newChatButton = await screen.findByRole('button', { name: /^new chat$/i });
    fireEvent.click(newChatButton);

    await waitFor(() => {
      expect(apiClient.createChatSession).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(window.location.pathname).toBe('/chat/1');
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
