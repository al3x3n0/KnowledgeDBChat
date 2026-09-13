/**
 * Quick start: a repo bug-triage run.
 *
 * Module scope, like its siblings — declared in the page's render body it was
 * remounted on every page render, resetting the form below.
 */

import React, { useMemo, useState } from 'react';
import toast from 'react-hot-toast';

import {
  buildRepoBugTriageQuickStartPayload,
  findUnsafeQuickStartCommands,
  parseQuickStartCommands,
  parseSafeRelativeFilePaths,
} from '../../pages/autonomousAgentQuickStarts';
import Button from '../common/Button';

interface QuickStartRepoBugTriageModalProps {
  onClose: () => void;
  quickStartRepoBugTriageMutation: any;
  codeSources: any[];
  templateRecommendGoal: string;
}

export const QuickStartRepoBugTriageModal: React.FC<QuickStartRepoBugTriageModalProps> = ({
  onClose,
  quickStartRepoBugTriageMutation,
  codeSources,
  templateRecommendGoal,
}) => {

  const MAX_QS_COMMANDS = 6;
  const MAX_QS_FILE_PATHS = 32;
  const [name, setName] = useState(`Repo Bug Triage - ${new Date().toLocaleDateString()}`);
  const [failureSymptom, setFailureSymptom] = useState(templateRecommendGoal.trim() || 'Describe the observed bug or failing behavior');
  const [goal, setGoal] = useState('Identify the minimal fix, verify it, and return a reviewable patch proposal');
  const [selectedSourceId, setSelectedSourceId] = useState<string>('');
  const [scope, setScope] = useState<'auto' | 'backend' | 'frontend' | 'worker'>('auto');
  const [searchQuery, setSearchQuery] = useState('');
  const [commandsText, setCommandsText] = useState('');
  const [filePathsText, setFilePathsText] = useState('');
  const [errorOutput, setErrorOutput] = useState('');

  const commandsPreview = useMemo(
    () => parseQuickStartCommands(commandsText, MAX_QS_COMMANDS),
    [commandsText]
  );
  const unsafeCommandsPreview = useMemo(
    () => findUnsafeQuickStartCommands(commandsPreview),
    [commandsPreview]
  );
  const filePathsParsed = useMemo(
    () => parseSafeRelativeFilePaths(filePathsText, MAX_QS_FILE_PATHS),
    [filePathsText]
  );
  const filePathsPreview = filePathsParsed.items;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!failureSymptom.trim() && !goal.trim()) {
      toast.error('Failure symptom or goal is required');
      return;
    }
    if (!selectedSourceId) {
      toast.error('Target code source is required');
      return;
    }
    if (unsafeCommandsPreview.length > 0) {
      toast.error(`Blocked unsafe command(s): ${unsafeCommandsPreview.join(' | ')}`);
      return;
    }
    quickStartRepoBugTriageMutation.mutate(
      buildRepoBugTriageQuickStartPayload({
        name,
        goal,
        failureSymptom,
        selectedSourceId,
        scope,
        searchQuery,
        commandsText,
        filePathsText,
        errorOutput,
        maxCommands: MAX_QS_COMMANDS,
        maxFilePaths: MAX_QS_FILE_PATHS,
      })
    );
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-lg font-semibold mb-1">Quick Start Repo Bug Triage</h2>
          <p className="text-sm text-gray-500 mb-4">
            Launch a symptom-driven patch and verification loop. By default it stops at a reviewable patch proposal.
          </p>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Job Name</label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Failure symptom</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={4}
                value={failureSymptom}
                onChange={(e) => setFailureSymptom(e.target.value)}
                placeholder="Saving a document returns 500 and leaves the spinner running"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Desired outcome (optional)</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={3}
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                placeholder="Fix the regression without changing successful save flows"
              />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Target code source</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={selectedSourceId}
                  onChange={(e) => setSelectedSourceId(e.target.value)}
                >
                  <option value="">Select a git source…</option>
                  {codeSources.map((s: any) => (
                    <option key={String(s.id)} value={String(s.id)}>
                      {String(s.name || s.id)}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Scope profile</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={scope}
                  onChange={(e) => setScope(e.target.value as any)}
                >
                  <option value="auto">Auto</option>
                  <option value="backend">Backend</option>
                  <option value="frontend">Frontend</option>
                  <option value="worker">Worker</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Search Query (optional)</label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Leave empty to derive from scope + symptom"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Commands (optional, one per line) {commandsPreview.length}/{MAX_QS_COMMANDS}
              </label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={4}
                value={commandsText}
                onChange={(e) => setCommandsText(e.target.value)}
                placeholder={'python -m pytest -q backend/tests\nCI=true npm --prefix frontend test -- --watchAll=false'}
              />
              <p className="mt-1 text-xs text-gray-500">
                Leave empty to auto-infer a bounded verification set from the repo profile after launch.
              </p>
              {unsafeCommandsPreview.length > 0 && (
                <p className="mt-1 text-xs text-rose-700">
                  Unsafe command(s) blocked: {unsafeCommandsPreview.join(' | ')}
                </p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Focused file paths (optional, one per line) {filePathsPreview.length}/{MAX_QS_FILE_PATHS}
              </label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={3}
                value={filePathsText}
                onChange={(e) => setFilePathsText(e.target.value)}
                placeholder={'frontend/src/pages/DocumentsPage.tsx\nbackend/app/api/endpoints/documents.py'}
              />
              {filePathsParsed.droppedUnsafe > 0 && (
                <p className="mt-1 text-xs text-amber-700">
                  {filePathsParsed.droppedUnsafe} unsafe path row(s) ignored.
                </p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Error output (optional)</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={4}
                value={errorOutput}
                onChange={(e) => setErrorOutput(e.target.value)}
                placeholder="Paste stack trace, failing assertion, or representative logs"
              />
            </div>
            <div className="rounded-lg border border-amber-100 bg-amber-50 px-3 py-2 text-xs text-amber-800">
              Default mode is patch-only. The agent can iterate on patch and verification, but it will not apply KB writes unless separately enabled.
            </div>
            <div className="flex justify-end gap-3 pt-4 border-t">
              <Button type="button" variant="secondary" onClick={() => onClose()}>
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={quickStartRepoBugTriageMutation.isLoading || codeSources.length === 0 || unsafeCommandsPreview.length > 0}
                title={
                  codeSources.length === 0
                    ? 'Add a GitHub/GitLab source first'
                    : unsafeCommandsPreview.length > 0
                      ? 'Remove unsafe commands before starting'
                      : undefined
                }
              >
                {quickStartRepoBugTriageMutation.isLoading ? 'Starting...' : 'Start Triage'}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default QuickStartRepoBugTriageModal;
