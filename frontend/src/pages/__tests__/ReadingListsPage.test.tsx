import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';

import ReadingListsPage from '../ReadingListsPage';

const mockNavigate = jest.fn();

jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

jest.mock('../../services/api', () => ({
  apiClient: {
    listReadingLists: jest.fn(),
    createReadingList: jest.fn(),
  },
}));

const apiClient = require('../../services/api').apiClient;

describe('ReadingListsPage', () => {
  beforeEach(() => {
    apiClient.listReadingLists.mockResolvedValue({
      items: [],
      total: 0,
      limit: 100,
      offset: 0,
    });
    apiClient.createReadingList.mockResolvedValue({ id: 'list-1' });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('creates a reading list from the inline name field', async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, cacheTime: 0 } },
    });

    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <QueryClientProvider client={queryClient}>
          <ReadingListsPage />
        </QueryClientProvider>
      </MemoryRouter>
    );

    fireEvent.change(screen.getByPlaceholderText('Reading list name'), {
      target: { value: 'Operator Notes' },
    });
    fireEvent.click(screen.getByText('New List'));

    await waitFor(() => {
      expect(apiClient.createReadingList).toHaveBeenCalledWith({
        name: 'Operator Notes',
        auto_populate_from_source: false,
      });
    });
    expect(mockNavigate).toHaveBeenCalledWith('/reading-lists/list-1');
  });
});
