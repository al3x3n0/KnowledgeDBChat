import React from 'react';
import { useQuery } from 'react-query';
import toast from 'react-hot-toast';
import apiClient from '../services/api';

const KGAdminPage: React.FC = () => {
  const [q, setQ] = React.useState('');
  const [debouncedQ, setDebouncedQ] = React.useState('');
  React.useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(q), 300);
    return () => clearTimeout(t);
  }, [q]);

  const { data: stats, refetch: refetchStats } = useQuery(['kg-stats'], () => apiClient.getKGStats());
  const [limit, setLimit] = React.useState(50);
  const [offset, setOffset] = React.useState(0);
  const { data: page, refetch } = useQuery(['kg-entities', debouncedQ, limit, offset], () => apiClient.searchKGEntities(debouncedQ || undefined, limit, offset));
  const results = page?.items || [];
  const total = page?.total || 0;

  const [sourceId, setSourceId] = React.useState<string>('');
  const [targetId, setTargetId] = React.useState<string>('');
  const [busy, setBusy] = React.useState(false);
  const [editId, setEditId] = React.useState<string>('');
  const [editData, setEditData] = React.useState<{ canonical_name: string; entity_type: string; description?: string; properties?: string }>({ canonical_name: '', entity_type: 'other' });
  const [propsValid, setPropsValid] = React.useState<boolean | null>(null);
  const [confirmDelete, setConfirmDelete] = React.useState('');

  const loadForEdit = async (id: string) => {
    try {
      const ent = await apiClient.getKGEntity(id);
      setEditId(ent.id);
      setEditData({
        canonical_name: ent.canonical_name,
        entity_type: ent.entity_type,
        description: ent.description || '',
        properties: ent.properties || '',
      });
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to load entity');
    }
  };

  const onMerge = async () => {
    if (!sourceId || !targetId) {
      toast.error('Select source and target entities');
      return;
    }
    if (sourceId === targetId) {
      toast.error('Source and target must be different');
      return;
    }
    const source = (results || []).find(e => e.id === sourceId);
    const target = (results || []).find(e => e.id === targetId);
    if (!window.confirm(`Merge\n\n${source?.canonical_name} (${source?.entity_type})\n→\n${target?.canonical_name} (${target?.entity_type})\n\nThis cannot be undone.`)) {
      return;
    }
    try {
      setBusy(true);
      await apiClient.mergeKGEntities(sourceId, targetId);
      toast.success('Entities merged');
      setSourceId('');
      setTargetId('');
      refetch();
      refetchStats();
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Merge failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-gray-900">Knowledge Graph Admin</h1>
          <div className="text-sm text-gray-600">
            {stats ? (
              <span>
                {stats.entities} entities · {stats.relationships} relations · {stats.mentions} mentions
              </span>
            ) : '—'}
          </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-3">
        <div className="flex items-center gap-2">
          <input
            className="flex-1 border border-gray-300 rounded px-3 py-2 text-sm"
            placeholder="Search entities by name/type"
            value={q}
            onChange={e => setQ(e.target.value)}
          />
          <button className="px-3 py-2 text-sm rounded bg-gray-100 hover:bg-gray-200" onClick={() => { setDebouncedQ(q); refetch(); }}>Search</button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="text-sm font-medium text-gray-800 mb-1">Results</div>
            <div className="border rounded divide-y max-h-80 overflow-auto">
              {(results || []).map(e => (
                <div key={e.id} className="p-2 flex items-center justify-between">
                  <div className="min-w-0">
                    <div className="text-sm text-gray-900 truncate">{e.canonical_name}</div>
                    <div className="text-xs text-gray-500">{e.entity_type} · {e.id}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button className={`px-2 py-1 text-xs rounded ${sourceId === e.id ? 'bg-primary-600 text-white' : 'bg-gray-100 hover:bg-gray-200'}`} onClick={() => setSourceId(e.id)}>
                      {sourceId === e.id ? 'Source' : 'Set Source'}
                    </button>
                    <button className={`px-2 py-1 text-xs rounded ${targetId === e.id ? 'bg-primary-600 text-white' : 'bg-gray-100 hover:bg-gray-200'}`} onClick={() => setTargetId(e.id)}>
                      {targetId === e.id ? 'Target' : 'Set Target'}
                    </button>
                    <button className={`px-2 py-1 text-xs rounded ${editId === e.id ? 'bg-primary-600 text-white' : 'bg-gray-100 hover:bg-gray-200'}`} onClick={() => loadForEdit(e.id)}>
                      Edit
                    </button>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-2 flex items-center gap-2">
              <button className="px-2 py-1 text-xs rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50" onClick={() => setOffset(Math.max(0, offset - limit))} disabled={offset === 0}>Prev</button>
              <button className="px-2 py-1 text-xs rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50" onClick={() => setOffset(offset + limit)} disabled={offset + limit >= total}>Next</button>
              <select className="ml-auto border border-gray-300 rounded px-2 py-1 text-xs" value={limit} onChange={e => { setLimit(parseInt(e.target.value, 10)); setOffset(0); }}>
                {[25, 50, 100, 200].map(n => <option key={n} value={n}>{n}/page</option>)}
              </select>
              <span className="text-xs text-gray-500">{offset + 1}-{Math.min(offset + limit, total)} of {total}</span>
            </div>
          </div>
          <div>
            <div className="text-sm font-medium text-gray-800 mb-1">Merge</div>
            <div className="p-3 border rounded bg-gray-50 space-y-2 text-sm">
              <div>
                <span className="text-gray-500">Source:</span>
                <span className="ml-2 font-medium">{(results || []).find(e => e.id === sourceId)?.canonical_name || '—'}</span>
              </div>
              <div>
                <span className="text-gray-500">Target:</span>
                <span className="ml-2 font-medium">{(results || []).find(e => e.id === targetId)?.canonical_name || '—'}</span>
              </div>
              <button
                className="mt-2 px-3 py-2 text-sm rounded bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50"
                onClick={onMerge}
                disabled={!sourceId || !targetId || busy}
              >
                Merge Entities
              </button>
              <div className="text-xs text-gray-500">Merges mentions and relationships; source entity is deleted.</div>
            </div>
          </div>
        </div>

        <div className="mt-6">
          <div className="text-sm font-medium text-gray-800 mb-2">Edit Entity</div>
          {editId ? (
            <div className="grid grid-cols-1 gap-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-gray-600">Name</label>
                  <input className="w-full border rounded px-2 py-1 text-sm" value={editData.canonical_name} onChange={e => setEditData({ ...editData, canonical_name: e.target.value })} />
                </div>
                <div>
                  <label className="text-xs text-gray-600">Type</label>
                  <select className="w-full border rounded px-2 py-1 text-sm" value={editData.entity_type} onChange={e => setEditData({ ...editData, entity_type: e.target.value })}>
                    {['person','org','location','product','email','url','other'].map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                </div>
              </div>
              <div>
                <label className="text-xs text-gray-600">Description</label>
                <textarea className="w-full border rounded px-2 py-1 text-sm" rows={3} value={editData.description || ''} onChange={e => setEditData({ ...editData, description: e.target.value })} />
              </div>
              <div>
                <label className="text-xs text-gray-600">Properties (JSON or text)</label>
                <textarea className={`w-full border rounded px-2 py-1 text-sm font-mono ${propsValid === false ? 'border-red-400' : ''}`} rows={3} value={editData.properties || ''} onChange={e => { setEditData({ ...editData, properties: e.target.value }); setPropsValid(null); }} />
                <div className="mt-1 flex items-center gap-2">
                  <button
                    className="px-2 py-1 text-xs rounded bg-gray-100 hover:bg-gray-200"
                    onClick={() => {
                      try {
                        if (!editData.properties || !editData.properties.trim()) { setPropsValid(null); return; }
                        const parsed = JSON.parse(editData.properties);
                        setEditData({ ...editData, properties: JSON.stringify(parsed, null, 2) });
                        setPropsValid(true);
                        toast.success('JSON is valid');
                      } catch {
                        setPropsValid(false);
                        toast.error('Invalid JSON');
                      }
                    }}
                  >
                    Validate/Format JSON
                  </button>
                  {propsValid === true && <span className="text-xs text-green-600">Valid JSON</span>}
                  {propsValid === false && <span className="text-xs text-red-600">Invalid JSON</span>}
                </div>
                {(() => {
                  if (!editData.properties || !editData.properties.trim()) return null;
                  try {
                    const parsed = JSON.parse(editData.properties);
                    return (
                      <div className="mt-2 p-2 bg-gray-50 border rounded">
                        <div className="text-xs text-gray-600 mb-1">Preview</div>
                        <div className="max-h-48 overflow-auto">
                          {/* lightweight nested viewer */}
                          <pre className="text-xs text-gray-800 whitespace-pre-wrap">{JSON.stringify(parsed, null, 2)}</pre>
                        </div>
                      </div>
                    );
                  } catch {
                    return null;
                  }
                })()}
              </div>
              <div className="flex items-center gap-2">
                <button
                  className="px-3 py-2 text-sm rounded bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50"
                  onClick={async () => {
                    try {
                      setBusy(true);
                      await apiClient.updateKGEntity(editId, editData);
                      toast.success('Entity updated');
                      refetch();
                      refetchStats();
                    } catch (e: any) {
                      toast.error(e?.response?.data?.detail || e?.message || 'Update failed');
                    } finally {
                      setBusy(false);
                    }
                  }}
                  disabled={busy}
                >
                  Save Changes
                </button>
                <button className="px-3 py-2 text-sm rounded bg-gray-100 hover:bg-gray-200" onClick={() => setEditId('')}>Cancel</button>
              </div>
              <div className="mt-4 p-3 border rounded bg-red-50">
                <div className="text-sm font-medium text-red-700 mb-1">Danger Zone</div>
                <div className="text-xs text-red-700 mb-2">Deleting an entity removes its mentions and related edges. This cannot be undone.</div>
                <div className="flex items-center gap-2">
                  <input
                    className="border rounded px-2 py-1 text-sm"
                    placeholder="Type the entity name to confirm"
                    value={confirmDelete}
                    onChange={e => setConfirmDelete(e.target.value)}
                  />
                  <button
                    className="px-3 py-2 text-sm rounded bg-red-600 text-white hover:bg-red-700 disabled:opacity-50"
                    onClick={async () => {
                      if (!editId) return;
                      try {
                        setBusy(true);
                        await apiClient.deleteKGEntity(editId, confirmDelete);
                        toast.success('Entity deleted');
                        setEditId('');
                        setConfirmDelete('');
                        refetch();
                        refetchStats();
                      } catch (e: any) {
                        toast.error(e?.response?.data?.detail || e?.message || 'Delete failed');
                      } finally {
                        setBusy(false);
                      }
                    }}
                    disabled={busy || !editId || confirmDelete !== editData.canonical_name}
                  >
                    Delete Entity
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-sm text-gray-500">Select an entity from results and click Edit.</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default KGAdminPage;
