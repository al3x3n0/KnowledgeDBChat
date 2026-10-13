/**
 * Where an `answer_question` result came from.
 *
 * The widget showed this tool's output as raw JSON. An answer written from
 * the knowledge base is only as good as the passages behind it, so the
 * sources are listed, and the retrieval that chose them can be opened: the
 * question as it was searched for and each selected passage with its score.
 * The chat page has had this for its own answers; the tool records the same
 * trace now and returns its id.
 */

import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Loader2 } from 'lucide-react';

import { apiClient } from '../../services/api';
import { RetrievalTrace } from '../../types';

export interface AnswerSource {
  document_id?: string;
  title?: string;
  relevance_score?: number;
  excerpt?: string;
}

export interface AnswerOutput {
  answer?: string;
  sources?: AnswerSource[];
  confidence?: string;
  retrieval_trace_id?: string;
}

/** A score in 0..1 as a whole percentage; anything else is not shown. */
export function scorePercent(score: unknown): string | null {
  if (typeof score !== 'number' || !Number.isFinite(score)) return null;
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`;
}

const AnswerSources: React.FC<{ output: AnswerOutput }> = ({ output }) => {
  const [open, setOpen] = useState(false);
  const [trace, setTrace] = useState<RetrievalTrace | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);

  const sources = Array.isArray(output.sources) ? output.sources : [];
  const traceId = output.retrieval_trace_id;

  const toggle = async () => {
    const next = !open;
    setOpen(next);
    if (!next || !traceId || trace || loading) return;
    try {
      setFailed(false);
      setLoading(true);
      setTrace(await apiClient.getRetrievalTrace(traceId));
    } catch {
      setFailed(true);
    } finally {
      setLoading(false);
    }
  };

  const selected: any[] = Array.isArray(trace?.trace?.selected_results)
    ? trace!.trace.selected_results
    : [];

  return (
    <div className="space-y-1">
      {output.confidence && (
        <div className="text-[10px] text-gray-600">Confidence: {output.confidence}</div>
      )}
      {sources.length === 0 ? (
        <div className="text-[10px] text-gray-600">No sources: nothing relevant was found.</div>
      ) : (
        sources.map((source, idx) => (
          <div key={source.document_id || idx} className="bg-white rounded px-2 py-1 border">
            <div className="font-medium truncate">{source.title || 'Untitled'}</div>
            {scorePercent(source.relevance_score) && (
              <div className="text-gray-500 text-[10px]">
                Relevance: {scorePercent(source.relevance_score)}
              </div>
            )}
            {source.excerpt && (
              <div className="text-gray-700 text-[11px] mt-1 line-clamp-2">{source.excerpt}</div>
            )}
          </div>
        ))
      )}

      {traceId && (
        <div>
          <button
            type="button"
            onClick={toggle}
            aria-expanded={open}
            className="inline-flex items-center gap-1 text-[10px] text-primary-600 hover:underline"
          >
            {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            How this was retrieved
          </button>
          {open && (
            <div className="mt-1 rounded border bg-white px-2 py-1 text-[10px] text-gray-700 space-y-1">
              {loading && (
                <div className="flex items-center gap-1">
                  <Loader2 className="w-3 h-3 animate-spin" /> Loading…
                </div>
              )}
              {failed && <div className="text-red-700">The retrieval record could not be loaded.</div>}
              {trace && (
                <>
                  <div>
                    Searched for: <span className="font-medium">{trace.processed_query || trace.query}</span>
                  </div>
                  {selected.length === 0 ? (
                    <div>No passage was selected.</div>
                  ) : (
                    <ol className="list-decimal pl-4">
                      {selected.map((row, idx) => (
                        <li key={row?.id || idx}>
                          {row?.title || 'Untitled'}
                          {scorePercent(row?.score) && ` (${scorePercent(row?.score)})`}
                        </li>
                      ))}
                    </ol>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AnswerSources;
