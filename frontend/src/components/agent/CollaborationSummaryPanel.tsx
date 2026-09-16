/**
 * Who owns a run, who it is shared with, and how it is visible.
 *
 * Declared inside `AutonomousAgentsPage` and used by four of its tabs, which
 * made it a shared component living in a page file -- so no tab could be
 * lifted out without either dragging it along or duplicating it.
 */

import React from 'react';

import Button from '../common/Button';
import { humanizeDecisionTraceValue } from '../../utils/agentJobDetail';

import type { CollaborationSummary, User } from '../../types';

const CollaborationSummaryPanel: React.FC<{
  summary?: CollaborationSummary | null;
  fallbackOwnerId?: string | null;
  fallbackVisibility?: string | null;
  fallbackSharedWithUserIds?: string[];
  userLabelById: (userId: string) => string;
  assigneeUsers?: User[];
  showAssigneeSelect?: boolean;
  assigneeValue?: string;
  onAssigneeChange?: (value: string) => void;
  onClearAssignee?: () => void;
  noteValue?: string;
  onNoteChange?: (value: string) => void;
  onNoteSave?: () => void;
  noteSaveLabel?: string;
  notePlaceholder?: string;
}> = ({
  summary,
  fallbackOwnerId,
  fallbackVisibility,
  fallbackSharedWithUserIds = [],
  userLabelById,
  assigneeUsers = [],
  showAssigneeSelect = false,
  assigneeValue,
  onAssigneeChange,
  onClearAssignee,
  noteValue,
  onNoteChange,
  onNoteSave,
  noteSaveLabel = 'Save note',
  notePlaceholder = 'Add a note',
}) => {
  const ownerId = String(summary?.owner_user_id || fallbackOwnerId || '').trim();
  const assigneeId = String(summary?.assigned_user_id || assigneeValue || '').trim();
  const assignedById = String(summary?.assigned_by_user_id || '').trim();
  const sharedWithUserIds = Array.isArray(summary?.shared_with_user_ids)
    ? summary.shared_with_user_ids.map((value) => String(value || '').trim()).filter(Boolean)
    : fallbackSharedWithUserIds.map((value) => String(value || '').trim()).filter(Boolean);
  const visibilityScope = String(summary?.visibility_scope || fallbackVisibility || (sharedWithUserIds.length > 0 ? 'shared' : 'private')).trim() || 'private';
  const ownerLabel = String(summary?.owner_label || (ownerId ? userLabelById(ownerId) : '') || ownerId || 'n/a').trim();
  const assigneeLabel = String(summary?.assignee_label || (assigneeId ? userLabelById(assigneeId) : '') || assigneeId || '').trim();
  const assignedByLabel = String(assignedById ? userLabelById(assignedById) : '').trim();
  const noteText = String(noteValue ?? summary?.note ?? '').trim();
  const assigneeList = assigneeUsers.length > 0 ? assigneeUsers : [];

  return (
    <div className="mt-3 rounded-lg border border-gray-200 bg-gray-100 p-3">
      <div className="flex flex-wrap gap-2 text-xs text-gray-700">
        <span>Owner {ownerLabel}</span>
        {assigneeLabel ? <span>Assignee {assigneeLabel}</span> : null}
        {assignedByLabel ? <span>Assigned by {assignedByLabel}</span> : null}
        <span>Visibility {humanizeDecisionTraceValue(visibilityScope)}</span>
        {sharedWithUserIds.length > 0 ? <span>Shared with {sharedWithUserIds.length}</span> : null}
      </div>
      {showAssigneeSelect && onAssigneeChange ? (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <select
            className="border border-gray-300 rounded-lg px-2 py-1 text-xs"
            value={assigneeValue ?? assigneeId}
            onChange={(e) => onAssigneeChange(String(e.target.value || '').trim())}
          >
            <option value="">Unassigned</option>
            {assigneeList.map((candidate) => (
              <option key={String(candidate.id)} value={String(candidate.id)}>
                {userLabelById(String(candidate.id))}
              </option>
            ))}
          </select>
          {onClearAssignee ? (
            <Button size="sm" variant="ghost" onClick={onClearAssignee} disabled={!assigneeId}>
              Clear assignment
            </Button>
          ) : null}
        </div>
      ) : null}
      {onNoteChange ? (
        <div className="mt-2 space-y-2">
          <textarea
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
            rows={2}
            placeholder={notePlaceholder}
            value={noteText}
            onChange={(e) => onNoteChange(e.target.value)}
          />
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs text-gray-600">Note {noteText ? 'saved locally until you click save' : 'optional'}</div>
            {onNoteSave ? (
              <Button size="sm" variant="ghost" onClick={onNoteSave}>
                {noteSaveLabel}
              </Button>
            ) : null}
          </div>
        </div>
      ) : noteText ? (
        <div className="mt-2 text-xs text-gray-600">Note: {noteText}</div>
      ) : null}
    </div>
  );
};

export default CollaborationSummaryPanel;
