/**
 * What a run remembered, and the button that makes it remember.
 *
 * A finished run extracts memories that later runs recall. This section shows
 * them and can re-run the extraction by hand.
 *
 * Lifted out of the job detail panel with its own loading state and its own
 * mounted/request guards — the panel's guards were shared across three
 * unrelated loaders, so a slow memories request could only be told apart from
 * a slow log request by a counter. One loader, one ref.
 *
 * The one thing it does not own is the extraction summary: the memory
 * persistence panel above shows the same numbers, preferring a manual run's
 * over the job's own. That is the panel's state, and arrives here as
 * `onExtracted`.
 */

import { BookOpen, Brain, Layers, Lightbulb, Loader2, Search, Sparkles } from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import toast from 'react-hot-toast';

import { apiClient } from '../../services/api';
import type { AgentJob, AgentJobMemoryListResponse } from '../../types';
import type { JobMemoryExtractionSummary } from '../../utils/agentMemoryExtraction';
import { normalizeManualExtractionResult } from '../../utils/agentMemoryExtraction';
import Button from '../common/Button';
import LoadingSpinner from '../common/LoadingSpinner';

interface JobMemoriesSectionProps {
  job: AgentJob;
  /** A manual extraction's numbers, for the persistence panel that shows them. */
  onExtracted: (summary: JobMemoryExtractionSummary) => void;
}

const memoryIcon = (type: string) => {
  switch (type) {
    case 'finding':
      return <Search className="w-3 h-3" />;
    case 'insight':
      return <Lightbulb className="w-3 h-3" />;
    case 'pattern':
      return <Layers className="w-3 h-3" />;
    case 'lesson':
      return <BookOpen className="w-3 h-3" />;
    default:
      return <Brain className="w-3 h-3" />;
  }
};

const memoryColor = (type: string) => {
  switch (type) {
    case 'finding':
      return 'text-blue-600 bg-blue-100';
    case 'insight':
      return 'text-purple-600 bg-purple-100';
    case 'pattern':
      return 'text-orange-600 bg-orange-100';
    case 'lesson':
      return 'text-green-600 bg-green-100';
    default:
      return 'text-gray-600 bg-gray-100';
  }
};

export const JobMemoriesSection: React.FC<JobMemoriesSectionProps> = ({ job, onExtracted }) => {
  const [memoriesData, setMemoriesData] = useState<AgentJobMemoryListResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [showMemories, setShowMemories] = useState(false);
  const [extracting, setExtracting] = useState(false);

  const mountedRef = useRef(true);
  const requestIdRef = useRef(0);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const loadMemories = useCallback(async () => {
    const requestId = ++requestIdRef.current;
    if (mountedRef.current) {
      setLoading(true);
    }
    try {
      const data = await apiClient.getJobMemories(job.id);
      if (!mountedRef.current || requestIdRef.current !== requestId) return;
      setMemoriesData(data);
    } catch (error) {
      if (!mountedRef.current || requestIdRef.current !== requestId) return;
      console.error('Failed to load memories:', error);
    } finally {
      if (mountedRef.current && requestIdRef.current === requestId) {
        setLoading(false);
      }
    }
  }, [job.id]);

  useEffect(() => {
    if (!showMemories || loading || memoriesData) return;
    loadMemories();
  }, [showMemories, loading, memoriesData, loadMemories]);

  const handleExtractMemories = async () => {
    setExtracting(true);
    try {
      const result = await apiClient.extractJobMemories(job.id);
      const summary = normalizeManualExtractionResult(result);
      onExtracted(summary);
      const skippedDuplicates = Number(summary.skipped_duplicates || 0);
      const createdCount = Number(summary.created_count || 0);
      const duplicateSuffix =
        skippedDuplicates > 0 ? ` (${skippedDuplicates} duplicates skipped)` : '';
      toast.success(`Extracted ${createdCount} memories${duplicateSuffix}`);
      await loadMemories();
    } catch (error: any) {
      console.error('Failed to extract memories:', error);
      toast.error(error.message || 'Failed to extract memories');
    }
    setExtracting(false);
  };

  const canExtract = ['completed', 'failed'].includes(job.status);

  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-700 flex items-center gap-1">
          <Brain className="w-4 h-4" />
          Memories
          {memoriesData && memoriesData.total > 0 && (
            <span className="ml-1 text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full">
              {memoriesData.total}
            </span>
          )}
        </h3>
        <div className="flex items-center gap-2">
          {canExtract && (
            <Button
              size="sm"
              variant="ghost"
              onClick={handleExtractMemories}
              disabled={extracting}
              title="Extract memories from job results"
            >
              {extracting ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <Sparkles className="w-3 h-3" />
              )}
            </Button>
          )}
          <Button size="sm" variant="ghost" onClick={() => setShowMemories(!showMemories)}>
            {showMemories ? 'Hide Memories' : 'Show Memories'}
          </Button>
        </div>
      </div>

      {showMemories && (
        <div className="border border-purple-200 rounded-lg p-3 bg-purple-50">
          {loading ? (
            <div className="flex justify-center py-4">
              <LoadingSpinner size="sm" />
            </div>
          ) : memoriesData && memoriesData.memories.length > 0 ? (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {memoriesData.memories.map((memory) => (
                <div key={memory.id} className="bg-white rounded-lg p-2 border border-purple-100">
                  <div className="flex items-start gap-2">
                    <div className={`p-1 rounded ${memoryColor(memory.type)}`}>
                      {memoryIcon(memory.type)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-medium text-purple-700 uppercase">
                          {memory.type}
                        </span>
                        <span className="text-xs text-gray-400">
                          {(memory.importance_score * 100).toFixed(0)}% importance
                        </span>
                      </div>
                      <p className="text-xs text-gray-700">{memory.content}</p>
                      {memory.tags && memory.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {memory.tags.slice(0, 4).map((tag, idx) => (
                            <span
                              key={idx}
                              className="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-4">
              <Brain className="w-8 h-8 text-purple-300 mx-auto mb-2" />
              <p className="text-sm text-purple-600">No memories extracted yet</p>
              {canExtract && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="mt-2 text-purple-600"
                  onClick={handleExtractMemories}
                  disabled={extracting}
                >
                  <Sparkles className="w-3 h-3 mr-1" />
                  Extract Memories
                </Button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default JobMemoriesSection;
