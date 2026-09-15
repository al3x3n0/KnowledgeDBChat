/**
 * Editing your own navigation.
 *
 * Shows every destination you are permitted, grouped by door, with the state
 * each is in — shown, hidden, pinned, renamed — rather than only the ones
 * currently visible. A settings page that listed only what you can see would
 * give you no way to un-hide anything.
 *
 * Saves the whole document on each change rather than batching behind a Save
 * button. The changes here are individually tiny and instantly visible in the
 * sidebar beside the form, so a pending-state to reason about would cost more
 * than it buys.
 */

import {
  Eye,
  EyeOff,
  Home,
  Pin,
  PinOff,
  RotateCcw,
  Undo2,
} from 'lucide-react';
import React, { useState } from 'react';
import toast from 'react-hot-toast';

import type { NavDoor, NavItem } from './catalog';
import type { NavPreferences } from './preferences';
import { useNavigation } from './useNavigation';

const NavigationSettings: React.FC = () => {
  const { catalog, nav, landing, save, isSaving, isLoading } = useNavigation();
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [draftLabel, setDraftLabel] = useState('');

  const hidden = new Set(nav.hidden || []);
  const pinned = new Set(nav.pinned || []);
  const renamed = nav.renamed || {};

  const commit = async (next: NavPreferences, message?: string) => {
    try {
      await save(next);
      if (message) toast.success(message);
    } catch {
      // apiClient surfaces the error; the sidebar still shows the old state,
      // which is the truthful thing to show when the save did not land.
    }
  };

  const toggleHidden = (item: NavItem) => {
    const next = new Set(hidden);
    next.has(item.key) ? next.delete(item.key) : next.add(item.key);
    commit({ ...nav, hidden: Array.from(next) });
  };

  const togglePinned = (item: NavItem) => {
    const next = new Set(pinned);
    next.has(item.key) ? next.delete(item.key) : next.add(item.key);
    commit({ ...nav, pinned: Array.from(next) });
  };

  const setLanding = (item: NavItem) => {
    commit({ ...nav, landing: item.key }, `Opening on ${item.name} from now on`);
  };

  const applyRename = (item: NavItem) => {
    const label = draftLabel.trim();
    const next = { ...renamed };
    // Clearing the box restores the original name rather than blanking it: a
    // nav entry with no label is unreachable by anything but position.
    if (!label || label === item.name) delete next[item.key];
    else next[item.key] = label;
    setEditingKey(null);
    commit({ ...nav, renamed: next });
  };

  const moveDoor = (doorId: string, delta: number) => {
    const order = (nav.doorOrder && nav.doorOrder.length
      ? nav.doorOrder
      : catalog.filter((d) => !d.utility).map((d) => d.id)
    ).slice();
    const at = order.indexOf(doorId);
    const to = at + delta;
    if (at < 0 || to < 0 || to >= order.length) return;
    [order[at], order[to]] = [order[to], order[at]];
    commit({ ...nav, doorOrder: order });
  };

  const resetAll = () => commit({}, 'Navigation reset to defaults');

  const customized =
    Boolean(nav.hidden?.length) ||
    Boolean(nav.pinned?.length) ||
    Boolean(nav.doorOrder?.length) ||
    Boolean(nav.landing) ||
    Object.keys(renamed).length > 0;

  if (isLoading) {
    return <p className="text-sm text-gray-500">Loading…</p>;
  }

  const workDoors = catalog.filter((d) => !d.utility);

  const renderDoor = (door: NavDoor, reorderable: boolean) => (
    <div key={door.id} className="rounded-lg border border-gray-300 bg-white">
      <div className="flex items-center gap-2 border-b border-gray-200 px-3 py-2">
        <door.icon className="h-4 w-4 text-gray-500" />
        <span className="flex-1 text-sm font-medium text-gray-900">
          {door.name}
        </span>
        {reorderable ? (
          <>
            <button
              onClick={() => moveDoor(door.id, -1)}
              disabled={isSaving}
              className="rounded px-1.5 py-0.5 text-xs text-gray-600 hover:bg-gray-200 disabled:opacity-40"
              aria-label={`Move ${door.name} up`}
            >
              ↑
            </button>
            <button
              onClick={() => moveDoor(door.id, 1)}
              disabled={isSaving}
              className="rounded px-1.5 py-0.5 text-xs text-gray-600 hover:bg-gray-200 disabled:opacity-40"
              aria-label={`Move ${door.name} down`}
            >
              ↓
            </button>
          </>
        ) : (
          <span className="text-[11px] text-gray-500">always last</span>
        )}
      </div>

      <div className="divide-y divide-gray-200">
        {door.sections.map((item) => {
          const isHidden = hidden.has(item.key);
          const isPinned = pinned.has(item.key);
          const isLanding = landing === item.key;
          const label = renamed[item.key] || item.name;

          return (
            <div
              key={item.key}
              className={`flex items-center gap-2 px-3 py-1.5 ${
                isHidden ? 'opacity-50' : ''
              }`}
            >
              <item.icon className="h-3.5 w-3.5 flex-shrink-0 text-gray-500" />

              {editingKey === item.key ? (
                <input
                  autoFocus
                  value={draftLabel}
                  onChange={(e) => setDraftLabel(e.target.value)}
                  onBlur={() => applyRename(item)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') applyRename(item);
                    if (e.key === 'Escape') setEditingKey(null);
                  }}
                  aria-label={`Rename ${item.name}`}
                  className="flex-1 rounded border border-gray-300 bg-white px-1.5 py-0.5 text-sm"
                />
              ) : (
                <button
                  onClick={() => {
                    setEditingKey(item.key);
                    setDraftLabel(label);
                  }}
                  title="Rename"
                  className="flex-1 truncate text-left text-sm text-gray-900 hover:underline"
                >
                  {label}
                  {renamed[item.key] && (
                    <span className="ml-1.5 text-[11px] text-gray-500">
                      ({item.name})
                    </span>
                  )}
                </button>
              )}

              {renamed[item.key] && (
                <button
                  onClick={() => {
                    const next = { ...renamed };
                    delete next[item.key];
                    commit({ ...nav, renamed: next });
                  }}
                  title={`Restore the name ${item.name}`}
                  className="rounded p-1 text-gray-500 hover:bg-gray-200"
                >
                  <Undo2 className="h-3.5 w-3.5" />
                </button>
              )}

              <button
                onClick={() => setLanding(item)}
                disabled={isSaving || isHidden}
                title={
                  isHidden
                    ? `${label} is hidden, so it cannot be your landing page`
                    : `Open on ${label} when you sign in`
                }
                className={`rounded p-1 disabled:opacity-30 ${
                  isLanding
                    ? 'text-primary-600'
                    : 'text-gray-500 hover:bg-gray-200'
                }`}
              >
                <Home className="h-3.5 w-3.5" />
              </button>

              <button
                onClick={() => togglePinned(item)}
                disabled={isSaving}
                title={
                  isPinned
                    ? `Unpin ${label}`
                    : `Pin ${label} to the top of ${door.name}`
                }
                className={`rounded p-1 disabled:opacity-40 ${
                  isPinned ? 'text-primary-600' : 'text-gray-500 hover:bg-gray-200'
                }`}
              >
                {isPinned ? (
                  <Pin className="h-3.5 w-3.5" />
                ) : (
                  <PinOff className="h-3.5 w-3.5" />
                )}
              </button>

              <button
                onClick={() => toggleHidden(item)}
                disabled={isSaving}
                title={isHidden ? `Show ${label}` : `Hide ${label}`}
                className="rounded p-1 text-gray-500 hover:bg-gray-200 disabled:opacity-40"
              >
                {isHidden ? (
                  <EyeOff className="h-3.5 w-3.5" />
                ) : (
                  <Eye className="h-3.5 w-3.5" />
                )}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );

  return (
    <div className="p-6">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Navigation</h2>
          <p className="mt-1 text-sm text-gray-600">
            Hide what you never open, rename what you call something else, pin
            what you use daily, and choose where signing in takes you. This
            follows your account rather than this browser.
          </p>
        </div>
        {customized && (
          <button
            onClick={resetAll}
            disabled={isSaving}
            className="inline-flex flex-shrink-0 items-center gap-1.5 rounded border border-gray-300 px-2.5 py-1.5 text-xs text-gray-700 hover:bg-gray-200 disabled:opacity-40"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset to defaults
          </button>
        )}
      </div>

      <div className="space-y-3">
        {workDoors.map((door) => renderDoor(door, true))}
        {catalog.filter((d) => d.utility).map((door) => renderDoor(door, false))}
      </div>

      <p className="mt-4 text-xs text-gray-500">
        A hidden page is still reachable by link or URL, and reappears in the
        sidebar while you are on it — a sidebar that cannot show where you are
        is worse than one showing a page you asked to remove.
      </p>
    </div>
  );
};

export default NavigationSettings;
