/**
 * Reading lists page.
 */

import React, { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from 'react-query';
import { Plus, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';

import { apiClient } from '../services/api';

interface ReadingListItem {
  id: string;
  name: string;
  description?: string | null;
  source_id?: string | null;
  updated_at?: string | null;
}

const ReadingListsPage: React.FC = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data, isLoading, isFetching } = useQuery(['readingLists'], () => apiClient.listReadingLists({ limit: 100, offset: 0 }), {
    staleTime: 5000,
  });

  const lists: ReadingListItem[] = useMemo(() => (data?.items || []) as ReadingListItem[], [data?.items]);

  const createList = async () => {
    const name = window.prompt('Reading list name', 'New Reading List');
    if (!name || !name.trim()) return;
    try {
      const created = await apiClient.createReadingList({ name: name.trim(), auto_populate_from_source: false });
      toast.success('Reading list created');
      queryClient.invalidateQueries('readingLists');
      if (created?.id) navigate(`/reading-lists/${created.id}`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Failed to create reading list');
    }
  };

  return (
    <div className="max-w-5xl mx-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Reading Lists</h1>
          <p className="text-gray-600">Collections of papers/documents with status and notes</p>
        </div>
        <button
          onClick={createList}
          className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          <Plus className="w-4 h-4" />
          New List
        </button>
      </div>

      <div className="bg-white border rounded-lg shadow-sm">
        <div className="px-4 py-3 border-b flex items-center justify-between">
          <div className="font-semibold text-gray-900">Your Lists</div>
          <div className="text-xs text-gray-500">{isFetching ? 'Refreshing…' : ''}</div>
        </div>
        <div className="p-4 space-y-3">
          {isLoading ? (
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading…
            </div>
          ) : lists.length === 0 ? (
            <div className="text-sm text-gray-600">No reading lists yet. Create one or make one from a Papers import.</div>
          ) : (
            lists.map((rl) => (
              <button
                key={rl.id}
                type="button"
                onClick={() => navigate(`/reading-lists/${rl.id}`)}
                className="w-full text-left border rounded-lg p-3 hover:bg-gray-50"
              >
                <div className="font-medium text-gray-900">{rl.name}</div>
                {rl.description ? <div className="text-sm text-gray-600 mt-1">{rl.description}</div> : null}
                {rl.updated_at ? (
                  <div className="text-xs text-gray-500 mt-1">Updated {rl.updated_at.slice(0, 19).replace('T', ' ')}</div>
                ) : null}
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default ReadingListsPage;

