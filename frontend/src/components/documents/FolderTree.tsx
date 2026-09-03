/**
 * The document folder tree.
 *
 * Two halves that look alike and behave differently. The system half is
 * computed by the server from the documents themselves — read-only, always
 * current, and the reason a newly synced source appears here without anyone
 * filing anything. The user half is yours: create, rename, nest, delete.
 *
 * Selection is by `key`, never by id, because system nodes have no id. The key
 * is also exactly what `getDocuments({ folder })` takes, so what the tree says
 * a folder contains and what the list shows cannot drift apart.
 *
 * Expansion state lives here and in localStorage rather than on the server: it
 * is a per-browser convenience, not something worth a round trip or a column.
 */

import clsx from 'clsx';
import {
  ChevronDown,
  ChevronRight,
  Clock,
  Database,
  FileType2,
  Folder,
  FolderOpen,
  FolderPlus,
  Inbox,
  Layers,
  MoreHorizontal,
  Pencil,
  Trash2,
} from 'lucide-react';
import React, { useCallback, useEffect, useMemo, useState } from 'react';

import type { DocumentFolderNode } from '../../types';

const EXPANDED_KEY = 'document_folder_tree_expanded_v1';

interface FolderTreeProps {
  system: DocumentFolderNode[];
  folders: DocumentFolderNode[];
  selectedKey: string;
  onSelect: (key: string, node: DocumentFolderNode) => void;
  onCreate: (parentId: string | null) => void;
  onRename: (node: DocumentFolderNode) => void;
  onDelete: (node: DocumentFolderNode) => void;
  /** Called when a document is dropped onto a user folder. */
  onDropDocuments?: (folderId: string, documentIds: string[]) => void;
  loading?: boolean;
}

/** System groups get an icon that says what kind of grouping it is. */
const groupIcon = (key: string) => {
  if (key === 'group:source') return Database;
  if (key === 'group:type') return FileType2;
  if (key === 'group:recent') return Clock;
  if (key === 'unfiled') return Inbox;
  if (key === 'all') return Layers;
  return Folder;
};

const readExpanded = (): Set<string> => {
  // A per-viewer convenience: if storage is unavailable (private window,
  // blocked site data) the tree just opens at its defaults.
  try {
    const raw = window.localStorage.getItem(EXPANDED_KEY);
    return new Set(raw ? (JSON.parse(raw) as string[]) : ['group:source']);
  } catch {
    return new Set(['group:source']);
  }
};

const FolderTree: React.FC<FolderTreeProps> = ({
  system,
  folders,
  selectedKey,
  onSelect,
  onCreate,
  onRename,
  onDelete,
  onDropDocuments,
  loading = false,
}) => {
  const [expanded, setExpanded] = useState<Set<string>>(readExpanded);
  const [menuFor, setMenuFor] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<string | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(EXPANDED_KEY, JSON.stringify(Array.from(expanded)));
    } catch {
      // Not worth telling the user about; the tree still works.
    }
  }, [expanded]);

  const toggle = useCallback((key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const totalUserFolders = useMemo(() => {
    const count = (nodes: DocumentFolderNode[]): number =>
      nodes.reduce((n, node) => n + 1 + count(node.children), 0);
    return count(folders);
  }, [folders]);

  const renderNode = (node: DocumentFolderNode, depth: number): React.ReactNode => {
    const hasChildren = node.children.length > 0;
    const isOpen = expanded.has(node.key);
    const isSelected = selectedKey === node.key;
    // A group heading is scaffolding: it holds nodes but selects nothing, so
    // clicking it opens it rather than filtering to an empty result.
    const selectable = node.kind !== 'group';
    const isUser = node.kind === 'user';
    const Icon = isUser ? (isOpen && hasChildren ? FolderOpen : Folder) : groupIcon(node.key);
    const count = hasChildren ? node.subtree_count : node.document_count;

    return (
      <div key={node.key} role="none">
        <div
          role="treeitem"
          aria-label={node.name}
          aria-level={depth + 1}
          aria-expanded={hasChildren ? isOpen : undefined}
          aria-selected={selectable ? isSelected : undefined}
          className={clsx(
            'group flex items-center gap-1 pr-1 rounded-md text-sm',
            'transition-all duration-fast ease-ui',
            isSelected
              ? 'bg-primary-500/10 text-primary-700 font-medium shadow-[inset_2px_0_0_0_theme(colors.primary.600)]'
              : 'text-gray-600 hover:bg-gray-200 hover:text-gray-900',
            dropTarget === node.key && 'ring-1 ring-primary-500 bg-primary-500/10'
          )}
          style={{ paddingLeft: `${depth * 12 + 4}px` }}
          onDragOver={
            isUser && node.id
              ? (event) => {
                  // Only user folders accept documents: a computed folder's
                  // membership is a query, so there is nothing to add to.
                  event.preventDefault();
                  setDropTarget(node.key);
                }
              : undefined
          }
          onDragLeave={isUser ? () => setDropTarget(null) : undefined}
          onDrop={
            isUser && node.id
              ? (event) => {
                  event.preventDefault();
                  setDropTarget(null);
                  const raw = event.dataTransfer.getData('application/x-document-ids');
                  if (!raw || !onDropDocuments) return;
                  try {
                    const ids = JSON.parse(raw) as string[];
                    if (Array.isArray(ids) && ids.length) onDropDocuments(node.id!, ids);
                  } catch {
                    // A drag from somewhere else in the page: ignore it.
                  }
                }
              : undefined
          }
        >
          {/* The twisty is its own button, so opening a folder and selecting
              it are separate actions rather than one guessing the other. */}
          {hasChildren ? (
            <button
              type="button"
              aria-label={isOpen ? `Collapse ${node.name}` : `Expand ${node.name}`}
              className="p-0.5 rounded hover:bg-gray-300 shrink-0"
              onClick={(event) => {
                event.stopPropagation();
                toggle(node.key);
              }}
            >
              {isOpen ? (
                <ChevronDown className="w-3.5 h-3.5" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5" />
              )}
            </button>
          ) : (
            <span className="w-[19px] shrink-0" aria-hidden="true" />
          )}

          <button
            type="button"
            className={clsx(
              'flex-1 min-w-0 flex items-center gap-2 py-1.5 text-left',
              !selectable && 'cursor-default'
            )}
            onClick={() => (selectable ? onSelect(node.key, node) : toggle(node.key))}
            aria-current={isSelected ? 'true' : undefined}
          >
            <Icon
              className={clsx(
                'w-4 h-4 shrink-0',
                isSelected ? 'text-primary-700' : 'text-gray-500'
              )}
              style={isUser && node.color ? { color: node.color } : undefined}
            />
            <span className="truncate">{node.name}</span>
            {count > 0 && (
              <span
                className={clsx(
                  'ml-auto shrink-0 text-xs font-mono',
                  isSelected ? 'text-primary-700' : 'text-gray-500'
                )}
              >
                {count}
              </span>
            )}
          </button>

          {isUser && node.id && (
            <div className="relative shrink-0">
              <button
                type="button"
                aria-label={`Actions for ${node.name}`}
                className="p-1 rounded opacity-0 group-hover:opacity-100 focus:opacity-100 hover:bg-gray-300 transition-opacity duration-fast"
                onClick={(event) => {
                  event.stopPropagation();
                  setMenuFor((current) => (current === node.key ? null : node.key));
                }}
              >
                <MoreHorizontal className="w-3.5 h-3.5" />
              </button>
              {menuFor === node.key && (
                <>
                  {/* A click anywhere else closes the menu; without this it
                      would stay open behind whatever you did next. */}
                  <div
                    className="fixed inset-0 z-40"
                    onClick={() => setMenuFor(null)}
                    aria-hidden="true"
                  />
                  <div className="absolute right-0 top-full z-50 mt-1 w-40 surface-2 rounded-lg py-1 animate-scale-in">
                    <button
                      type="button"
                      className="w-full px-3 py-1.5 text-left text-xs text-gray-700 hover:bg-gray-300 flex items-center gap-2"
                      onClick={() => {
                        setMenuFor(null);
                        onCreate(node.id!);
                      }}
                    >
                      <FolderPlus className="w-3.5 h-3.5" /> New subfolder
                    </button>
                    <button
                      type="button"
                      className="w-full px-3 py-1.5 text-left text-xs text-gray-700 hover:bg-gray-300 flex items-center gap-2"
                      onClick={() => {
                        setMenuFor(null);
                        onRename(node);
                      }}
                    >
                      <Pencil className="w-3.5 h-3.5" /> Rename
                    </button>
                    <button
                      type="button"
                      className="w-full px-3 py-1.5 text-left text-xs text-red-400 hover:bg-red-500/10 flex items-center gap-2"
                      onClick={() => {
                        setMenuFor(null);
                        onDelete(node);
                      }}
                    >
                      <Trash2 className="w-3.5 h-3.5" /> Delete
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {isOpen && hasChildren && (
          <div role="group">{node.children.map((child) => renderNode(child, depth + 1))}</div>
        )}
      </div>
    );
  };

  return (
    <nav className="text-sm" aria-label="Document folders">
      {loading && (
        <div className="space-y-1.5 px-1" role="status" aria-live="polite" aria-busy="true">
          <span className="sr-only">Loading folders</span>
          {[0, 1, 2, 3, 4].map((i) => (
            <div
              key={i}
              aria-hidden="true"
              className="skeleton h-6 rounded"
              style={{ animationDelay: `${i * 90}ms`, width: `${90 - i * 8}%` }}
            />
          ))}
        </div>
      )}

      {!loading && (
        <>
          <div className="px-2 pb-1 text-[10px] font-semibold tracking-wide uppercase text-gray-500">
            Library
          </div>
          <div role="tree" aria-label="Library folders">
            {system.map((node) => renderNode(node, 0))}
          </div>

          <div className="mt-4 flex items-center justify-between px-2 pb-1">
            <span className="text-[10px] font-semibold tracking-wide uppercase text-gray-500">
              My folders
            </span>
            <button
              type="button"
              aria-label="New folder"
              title="New folder"
              className="p-1 rounded text-gray-500 hover:bg-gray-200 hover:text-gray-900 transition-colors duration-fast"
              onClick={() => onCreate(null)}
            >
              <FolderPlus className="w-3.5 h-3.5" />
            </button>
          </div>

          {totalUserFolders === 0 ? (
            <p className="px-2 py-1 text-xs text-gray-500">
              No folders yet. Everything is still reachable above.
            </p>
          ) : (
            <div role="tree" aria-label="My folders">
              {folders.map((node) => renderNode(node, 0))}
            </div>
          )}
        </>
      )}
    </nav>
  );
};

export default FolderTree;
