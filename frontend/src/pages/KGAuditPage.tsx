import React from 'react';
import { useQuery } from 'react-query';
import apiClient from '../services/api';
import JsonViewer from '../components/common/JsonViewer';

const KGAuditPage: React.FC = () => {
  const [action, setAction] = React.useState('');
  const [userId, setUserId] = React.useState('');
  const [userSearch, setUserSearch] = React.useState('');
  const [debouncedUserSearch, setDebouncedUserSearch] = React.useState('');
  const [userDropdownOpen, setUserDropdownOpen] = React.useState(false);
  const userBoxRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const t = setTimeout(() => setDebouncedUserSearch(userSearch), 250);
    return () => clearTimeout(t);
  }, [userSearch]);
  const { data: userPage } = useQuery(['users', debouncedUserSearch], () => apiClient.searchUsers(debouncedUserSearch, 1, 10), { enabled: !!debouncedUserSearch && userDropdownOpen });
  const userItems = userPage?.items || [];

  React.useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (!userBoxRef.current) return;
      if (!userBoxRef.current.contains(e.target as Node)) {
        setUserDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);
  const [dateFrom, setDateFrom] = React.useState('');
  const [dateTo, setDateTo] = React.useState('');
  const [limit, setLimit] = React.useState(50);
  const [offset, setOffset] = React.useState(0);

  const { data, refetch, isFetching } = useQuery(
    ['kg-audit', action, userId, dateFrom, dateTo, limit, offset],
    () => apiClient.getKGAuditLogs({
      action: action || undefined,
      user_id: userId || undefined,
      date_from: dateFrom || undefined,
      date_to: dateTo || undefined,
      limit,
      offset,
    })
  );

  const items = data?.items || [];
  const total = data?.total || 0;

  const [pretty, setPretty] = React.useState(true);

  const toCSV = () => {
    const rows = [
      ['created_at', 'action', 'user_name', 'user_id', 'details'],
      ...items.map((it: any) => [it.created_at, it.action, it.user_name || '', it.user_id, it.details || ''])
    ];
    const esc = (v: any) => {
      const s = String(v ?? '');
      if (s.includes('"') || s.includes(',') || s.includes('\n')) {
        return '"' + s.replace(/"/g, '""') + '"';
      }
      return s;
    };
    const csv = rows.map(r => r.map(esc).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `kg_audit_${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-gray-900">KG Audit Log</h1>
        <div className="flex items-center gap-3">
          <label className="text-sm text-gray-600 inline-flex items-center gap-1">
            <input type="checkbox" checked={pretty} onChange={e => setPretty(e.target.checked)} /> Pretty JSON
          </label>
          <button className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200" onClick={toCSV} disabled={!items.length}>Export CSV</button>
          <div className="text-sm text-gray-600">{isFetching ? 'Refreshing…' : ''}</div>
        </div>
      </div>

      <div className="bg-white border rounded p-4">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          <input className="border rounded px-2 py-1 text-sm" placeholder="Action (merge_entities, update_entity, delete_entity)" value={action} onChange={e => setAction(e.target.value)} />
          <div className="relative" ref={userBoxRef}>
            <input
              className="w-full border rounded px-2 py-1 text-sm"
              placeholder="User (type to search)"
              value={userSearch}
              onChange={e => { setUserSearch(e.target.value); setUserDropdownOpen(true); }}
              onFocus={() => setUserDropdownOpen(true)}
            />
            {(userId || userSearch) && (
              <button
                type="button"
                className="absolute right-1 top-1 text-xs text-gray-500 hover:text-gray-800 px-2 py-0.5"
                onClick={() => { setUserId(''); setUserSearch(''); setUserDropdownOpen(false); setOffset(0); refetch(); }}
                aria-label="Clear user filter"
              >
                Clear
              </button>
            )}
            {userDropdownOpen && (
              <div className="absolute z-10 mt-1 w-full bg-white border rounded shadow max-h-56 overflow-auto" onMouseDown={e => e.preventDefault()}>
                {userItems.map(u => (
                  <button
                    key={u.id as any}
                    className="w-full text-left px-2 py-1 hover:bg-gray-100 text-sm"
                    onClick={() => { setUserId(String(u.id)); setUserSearch(u.full_name || u.username); setUserDropdownOpen(false); }}
                  >
                    <div className="font-medium text-gray-900">{u.full_name || u.username}</div>
                    <div className="text-xs text-gray-600">{u.email}</div>
                  </button>
                ))}
                {(!userItems.length) && (
                  <div className="px-2 py-2 text-sm text-gray-500">No users</div>
                )}
              </div>
            )}
          </div>
          <input className="border rounded px-2 py-1 text-sm" placeholder="From (YYYY-MM-DD)" value={dateFrom} onChange={e => setDateFrom(e.target.value)} />
          <input className="border rounded px-2 py-1 text-sm" placeholder="To (YYYY-MM-DD)" value={dateTo} onChange={e => setDateTo(e.target.value)} />
          <button className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200" onClick={() => { setOffset(0); refetch(); }}>Apply</button>
        </div>
      </div>

      <div className="bg-white border rounded">
        <div className="overflow-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left px-3 py-2">Time</th>
                <th className="text-left px-3 py-2">Action</th>
                <th className="text-left px-3 py-2">User</th>
                <th className="text-left px-3 py-2">Details</th>
              </tr>
            </thead>
            <tbody>
              {items.map((it: any) => (
                <tr key={it.id} className="border-t">
                  <td className="px-3 py-2 whitespace-nowrap text-gray-600">{it.created_at}</td>
                  <td className="px-3 py-2 font-medium text-gray-900">{it.action}</td>
                  <td className="px-3 py-2 text-gray-700">{it.user_name || it.user_id}</td>
                  <td className="px-3 py-2">
                    {(() => {
                      const raw = it.details || '';
                      if (!pretty) return <pre className="text-xs text-gray-700 whitespace-pre-wrap break-words">{raw}</pre>;
                      try {
                        const parsed = JSON.parse(raw);
                        return <JsonViewer json={parsed} />;
                      } catch {
                        return <pre className="text-xs text-gray-700 whitespace-pre-wrap break-words">{raw}</pre>;
                      }
                    })()}
                  </td>
                </tr>
              ))}
              {items.length === 0 && (
                <tr>
                  <td className="px-3 py-4 text-gray-500" colSpan={4}>No results</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="p-2 border-t flex items-center gap-2">
          <button className="px-2 py-1 text-xs rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50" onClick={() => setOffset(Math.max(0, offset - limit))} disabled={offset === 0}>Prev</button>
          <button className="px-2 py-1 text-xs rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50" onClick={() => setOffset(offset + limit)} disabled={offset + limit >= total}>Next</button>
          <select className="ml-auto border border-gray-300 rounded px-2 py-1 text-xs" value={limit} onChange={e => { setLimit(parseInt(e.target.value, 10)); setOffset(0); }}>
            {[25, 50, 100, 200].map(n => <option key={n} value={n}>{n}/page</option>)}
          </select>
          <span className="text-xs text-gray-500">{offset + 1}-{Math.min(offset + limit, total)} of {total}</span>
        </div>
      </div>
    </div>
  );
};

export default KGAuditPage;
