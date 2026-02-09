/**
 * Reading list detail page.
 */

import React, { useCallback, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useQuery, useQueryClient } from 'react-query';
import { Loader2, ArrowLeft, ExternalLink } from 'lucide-react';
import toast from 'react-hot-toast';

import { apiClient } from '../services/api';

type ReadingStatus = 'to-read' | 'reading' | 'done';

interface ReadingListItemRow {
  id: string;
  document_id: string;
  document_title?: string | null;
  status: ReadingStatus;
  priority: number;
  position: number;
  notes?: string | null;
}

const ReadingListDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [savingIds, setSavingIds] = useState<Record<string, boolean>>({});

  const { data, isLoading } = useQuery(['readingList', id], () => apiClient.getReadingList(id as string), {
    enabled: Boolean(id),
    staleTime: 3000,
  });

  const listName = data?.name || 'Reading List';
  const items: ReadingListItemRow[] = useMemo(() => (data?.items || []) as ReadingListItemRow[], [data?.items]);

  const updateItem = useCallback(
    async (itemId: string, patch: Partial<Pick<ReadingListItemRow, 'status' | 'priority' | 'position' | 'notes'>>) => {
      if (!id) return;
      setSavingIds(prev => ({ ...prev, [itemId]: true }));
      try {
        await apiClient.updateReadingListItem(id, itemId, patch as any);
        queryClient.invalidateQueries(['readingList', id]);
      } catch (e: any) {
        toast.error(e?.response?.data?.detail || 'Failed to update item');
      } finally {
        setSavingIds(prev => ({ ...prev, [itemId]: false }));
      }
    },
    [id, queryClient]
  );

  if (isLoading) {
    return (
      <div className="max-w-5xl mx-auto p-6 flex items-center gap-2 text-sm text-gray-600">
        <Loader2 className="w-4 h-4 animate-spin" /> Loading…
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto p-6">
      <div className="flex items-center gap-3 mb-6">
        <button
          type="button"
          onClick={() => navigate('/reading-lists')}
          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <div className="min-w-0">
          <h1 className="text-2xl font-bold text-gray-900 truncate">{listName}</h1>
          <p className="text-gray-600">Status, priority, and notes</p>
        </div>
      </div>

      <div className="bg-white border rounded-lg shadow-sm">
        <div className="px-4 py-3 border-b flex items-center justify-between">
          <div className="font-semibold text-gray-900">Items</div>
          <div className="text-xs text-gray-500">{items.length} total</div>
        </div>

        <div className="p-4 space-y-4">
          {items.length === 0 ? (
            <div className="text-sm text-gray-600">No items yet.</div>
          ) : (
            items.map((it) => (
              <div key={it.id} className="border rounded-lg p-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="font-medium text-gray-900 truncate">{it.document_title || it.document_id}</div>
                    <button
                      type="button"
                      className="mt-1 inline-flex items-center gap-1 text-xs text-primary-700 hover:text-primary-800"
                      onClick={() => navigate('/documents', { state: { openDocId: it.document_id } })}
                    >
                      Open in Documents <ExternalLink className="w-3 h-3" />
                    </button>
                  </div>
                  <div className="flex items-center gap-2">
                    <select
                      value={it.status}
                      onChange={(e) => updateItem(it.id, { status: e.target.value as ReadingStatus })}
                      className="border border-gray-300 rounded-lg px-2 py-1 text-sm focus:ring-2 focus:ring-primary-500"
                    >
                      <option value="to-read">To read</option>
                      <option value="reading">Reading</option>
                      <option value="done">Done</option>
                    </select>
                    <input
                      type="number"
                      value={it.priority}
                      onChange={(e) => updateItem(it.id, { priority: parseInt(e.target.value || '0', 10) })}
                      className="w-20 border border-gray-300 rounded-lg px-2 py-1 text-sm focus:ring-2 focus:ring-primary-500"
                      title="Priority"
                    />
                    {savingIds[it.id] ? <Loader2 className="w-4 h-4 animate-spin text-gray-400" /> : null}
                  </div>
                </div>
                <div className="mt-3">
                  <textarea
                    defaultValue={it.notes || ''}
                    placeholder="Notes…"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500"
                    rows={3}
                    onBlur={(e) => updateItem(it.id, { notes: e.target.value })}
                  />
                  <div className="text-xs text-gray-500 mt-1">Notes save on blur.</div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default ReadingListDetailPage;

