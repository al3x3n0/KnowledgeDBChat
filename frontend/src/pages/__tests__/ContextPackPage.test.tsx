import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';

import { apiClient } from '../../services/api';
import ContextPackPage from '../ContextPackPage';

jest.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({ user: { id: 'owner', role: 'user' } }),
}));
jest.mock('../../services/api', () => ({
  apiClient: { getRetrievalTrace: jest.fn(), saveContextPackAsNote: jest.fn() },
}));

function NoteLocation() {
  return <p>{useLocation().search}</p>;
}

test('saves a context pack and opens its research note', async () => {
  jest.mocked(apiClient.getRetrievalTrace).mockResolvedValue({
    id: 'trace-1',
    user_id: 'owner',
    trace_type: 'chat',
    query: 'Which result changed?',
    trace: { kg_context_pack: { kg_context: 'Context with evidence' } },
    created_at: '',
  });
  jest.mocked(apiClient.saveContextPackAsNote).mockResolvedValue({
    id: 'note-1', user_id: 'owner', title: 'Context pack', content_markdown: 'Cited evidence',
  });
  render(
    <QueryClientProvider client={new QueryClient()}>
      <MemoryRouter initialEntries={['/context-packs/trace-1']}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Routes>
          <Route path="/context-packs/:traceId" element={<ContextPackPage />} />
          <Route path="/research-notes" element={<NoteLocation />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );

  fireEvent.click(await screen.findByRole('button', { name: 'Save as Research Note' }));
  expect(await screen.findByText('?note=note-1')).toBeInTheDocument();
  expect(apiClient.saveContextPackAsNote).toHaveBeenCalledWith('trace-1');
});
