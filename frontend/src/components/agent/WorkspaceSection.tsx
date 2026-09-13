/**
 * The environment a run worked in.
 *
 * Not a repository browser. It is reached through the run and it is read-only,
 * because the question it answers is "how did this stage produce this result",
 * not "let me edit some code". Writing from here would put a person and an
 * agent on the same files with nothing arbitrating, and the run's own account
 * of what it did would stop being true.
 *
 * What it leads with is what the run CHANGED, because that is the one thing
 * looking at the directory cannot tell you — a directory shows what it ends
 * with. The answer exists only because the hashes of the starting files were
 * kept beside it.
 *
 * Three states are worth telling apart, and each means something different to
 * whoever is reading:
 *
 *   - **no workspace** — this run never made one. Nothing to show, and nothing
 *     wrong; most runs are not coding runs. Renders nothing at all.
 *   - **retained** — the files are there and can be read.
 *   - **discarded** — the run made one and its files were swept after the
 *     retention window. Said out loud rather than hidden, because "the
 *     evidence is gone" is exactly what a reader needs to know when a
 *     measurement cannot be re-derived.
 */

import clsx from 'clsx';
import {
  FileCode,
  FolderTree,
  GitBranch,
  Trash2,
  X,
} from 'lucide-react';
import React, { useState } from 'react';
import { useQuery } from 'react-query';

import { apiClient } from '../../services/api';
import type { AgentJob, AgentJobWorkspace, AgentJobWorkspaceFile } from '../../types';

export interface WorkspaceSectionProps {
  job: AgentJob;
}

const STATUS_CLASS: Record<string, string> = {
  active: 'border-primary-500/60 bg-primary-500/10 text-primary-700',
  retained: 'border-gray-300 bg-gray-100 text-gray-600',
  discarded: 'border-amber-300 bg-amber-50 text-amber-700',
};

function bytes(size: number): string {
  if (!size) return '';
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export const WorkspaceSection: React.FC<WorkspaceSectionProps> = ({ job }) => {
  const [openFile, setOpenFile] = useState<string | null>(null);

  const { data, error } = useQuery<AgentJobWorkspace>(
    ['job-workspace', job.id],
    () => apiClient.getJobWorkspace(String(job.id)),
    {
      enabled: Boolean(job.id),
      // Rendered for every run, and most runs never made a workspace. Retrying
      // that 404 is noise.
      retry: false,
      staleTime: 30_000,
    }
  );

  const { data: file } = useQuery<AgentJobWorkspaceFile>(
    ['job-workspace-file', job.id, openFile],
    () => apiClient.getJobWorkspaceFile(String(job.id), String(openFile)),
    { enabled: Boolean(openFile), retry: false }
  );

  const gone = (error as any)?.response?.status === 410;

  if (gone) {
    // The run had a workspace and its files were swept. Worth a line: it is
    // the reason a measurement cannot be re-derived, and silence here would
    // read as "this run did no coding".
    return (
      <div className="mb-4" data-testid="workspace-section">
        <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1.5">
          <FolderTree className="w-4 h-4" />
          Workspace
        </h3>
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-3 flex items-start gap-1.5">
          <Trash2 className="w-3.5 h-3.5 mt-0.5 flex-none" />
          The files this run worked in were removed after the retention window.
          Its findings still reference the workspace, but the environment itself
          is gone.
        </p>
      </div>
    );
  }

  if (error || !data) return null;

  const statusClass =
    STATUS_CLASS[data.status] || 'border-gray-300 bg-gray-100 text-gray-600';
  const touched = data.modified.length + data.added.length + data.deleted.length;

  return (
    <div className="mb-4" data-testid="workspace-section">
      <div className="flex items-center gap-2 mb-2">
        <h3 className="text-sm font-medium text-gray-700 flex items-center gap-1.5">
          <FolderTree className="w-4 h-4" />
          Workspace
        </h3>
        <span
          className={clsx(
            'text-[11px] px-2 py-0.5 rounded-full border font-medium',
            statusClass
          )}
        >
          {data.status}
        </span>
        {data.repo_url && (
          <span className="text-xs text-gray-500 inline-flex items-center gap-1 truncate max-w-[20rem]">
            <GitBranch className="w-3 h-3" />
            {data.repo_url}
            {data.branch ? ` @ ${data.branch}` : ''}
          </span>
        )}
      </div>

      <div className="bg-gray-100 border border-gray-300 rounded-lg p-3 space-y-3">
        {/* What the run changed, first: the one thing the directory cannot
            say on its own. */}
        <div className="flex flex-wrap gap-3 text-xs">
          <span className="text-gray-600">
            {touched === 0
              ? 'The run changed nothing here'
              : `${touched} file${touched === 1 ? '' : 's'} touched`}
          </span>
          {data.modified.length > 0 && (
            <span className="text-amber-700">{data.modified.length} modified</span>
          )}
          {data.added.length > 0 && (
            <span className="text-primary-700">{data.added.length} added</span>
          )}
          {data.deleted.length > 0 && (
            <span className="text-red-600">{data.deleted.length} deleted</span>
          )}
        </div>

        {data.deleted.length > 0 && (
          // Deleted files cannot appear in the tree below, so they would be
          // invisible if not named here.
          <div className="text-[11px] text-gray-500">
            Deleted: <span className="font-mono">{data.deleted.join(', ')}</span>
          </div>
        )}

        <ul className="divide-y divide-gray-200 rounded-md border border-gray-300 overflow-hidden max-h-64 overflow-y-auto scrollbar-thin">
          {data.entries.map((entry) => (
            <li key={entry.path}>
              <button
                type="button"
                disabled={entry.is_dir}
                onClick={() => setOpenFile(entry.path)}
                className={clsx(
                  'w-full text-left px-2.5 py-1.5 flex items-center gap-2 text-xs transition-colors duration-fast',
                  entry.is_dir
                    ? 'text-gray-500 cursor-default'
                    : 'text-gray-800 hover:bg-gray-200',
                  openFile === entry.path && 'bg-primary-500/10'
                )}
              >
                {entry.is_dir ? (
                  <FolderTree className="w-3 h-3 flex-none" />
                ) : (
                  <FileCode className="w-3 h-3 flex-none" />
                )}
                <span className="font-mono truncate">{entry.path}</span>
                {entry.changed && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded-full border border-amber-300 bg-amber-50 text-amber-700 flex-none">
                    changed
                  </span>
                )}
                <span className="ml-auto text-[10px] text-gray-500 flex-none">
                  {entry.is_dir ? '' : bytes(entry.size)}
                </span>
              </button>
            </li>
          ))}
        </ul>

        {data.truncated && (
          <p className="text-[10px] text-gray-500">
            Showing the first {data.entries.length} entries.
          </p>
        )}

        {openFile && (
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[11px] font-mono text-gray-700 truncate">
                {openFile}
              </span>
              {file?.changed && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full border border-amber-300 bg-amber-50 text-amber-700">
                  changed by this run
                </span>
              )}
              <button
                type="button"
                aria-label="Close the file"
                className="ml-auto p-0.5 rounded text-gray-500 hover:text-gray-900 hover:bg-gray-200"
                onClick={() => setOpenFile(null)}
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
            {/* The change itself where git can answer, and the file as it
                stands otherwise. Shown as a patch rather than a rendered
                side-by-side because a patch is what the run would have
                produced, and inventing a prettier representation of it would
                be a second opinion about what changed. */}
            {file && file.diff ? (
              <pre className="text-[11px] font-mono bg-gray-50 border border-gray-300 rounded-md p-2 overflow-auto max-h-72 whitespace-pre">
                {file.diff.split('\n').map((line, index) => (
                  <div
                    key={index}
                    className={clsx(
                      line.startsWith('+') && !line.startsWith('+++')
                        ? 'text-primary-700'
                        : line.startsWith('-') && !line.startsWith('---')
                          ? 'text-red-600'
                          : line.startsWith('@@')
                            ? 'text-gray-500'
                            : 'text-gray-700'
                    )}
                  >
                    {line || ' '}
                  </div>
                ))}
              </pre>
            ) : (
              <pre className="text-[11px] font-mono text-gray-800 bg-gray-50 border border-gray-300 rounded-md p-2 overflow-auto max-h-72 whitespace-pre">
                {file ? file.content : 'Loading…'}
              </pre>
            )}

            {/* Three different things to say, and they are not the same:
                git showed the change; git is there and says nothing changed;
                or nothing here can tell you either way. */}
            {file && file.changed && !file.diff_available && (
              <p className="text-[10px] text-gray-500 mt-1">
                This is the file as it stands now. This workspace has no
                repository and its starting version was not kept, so the change
                itself cannot be shown.
              </p>
            )}
            {file && file.diff_available && !file.diff && (
              <p className="text-[10px] text-gray-500 mt-1">
                Tracked by git and unchanged since the run started.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default WorkspaceSection;
