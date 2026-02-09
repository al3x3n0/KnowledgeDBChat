/**
 * Global Search Page with multiple search modes and result ranking
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery } from 'react-query';
import { Search, Loader2, FileText, ChevronLeft, ChevronRight } from 'lucide-react';

import { apiClient } from '../services/api';
import { SearchMode, SearchSortBy, SearchSortOrder, SearchResult } from '../types';
import SearchResultCard from '../components/search/SearchResultCard';

const SearchPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  // Initialize state from URL params
  const [query, setQuery] = useState(searchParams.get('q') || '');
  const [submittedQuery, setSubmittedQuery] = useState(searchParams.get('q') || '');
  const [mode, setMode] = useState<SearchMode>((searchParams.get('mode') as SearchMode) || 'smart');
  const [sortBy, setSortBy] = useState<SearchSortBy>((searchParams.get('sort_by') as SearchSortBy) || 'relevance');
  const [sortOrder, setSortOrder] = useState<SearchSortOrder>((searchParams.get('sort_order') as SearchSortOrder) || 'desc');
  const [page, setPage] = useState(parseInt(searchParams.get('page') || '1', 10));
  const pageSize = 10;

  const handleSubmit = useCallback(() => {
    const next = query.trim();
    setSubmittedQuery(next);
    setPage(1);
  }, [query]);

  // Update URL when search params change
  useEffect(() => {
    const params: Record<string, string> = {};
    if (submittedQuery) params.q = submittedQuery;
    if (mode !== 'smart') params.mode = mode;
    if (sortBy !== 'relevance') params.sort_by = sortBy;
    if (sortOrder !== 'desc') params.sort_order = sortOrder;
    if (page > 1) params.page = page.toString();
    setSearchParams(params, { replace: true });
  }, [submittedQuery, mode, sortBy, sortOrder, page, setSearchParams]);

  // Search query
  const { data, isLoading, isFetching, error } = useQuery(
    ['search', submittedQuery, mode, sortBy, sortOrder, page],
    () => apiClient.searchDocuments({
      q: submittedQuery,
      mode,
      sort_by: sortBy,
      sort_order: sortOrder,
      page,
      page_size: pageSize,
    }),
    {
      enabled: submittedQuery.trim().length >= 2,
      keepPreviousData: true,
      staleTime: 30000,
    }
  );

  const totalPages = data ? Math.ceil(data.total / pageSize) : 0;

  const handleViewDocument = useCallback((result: SearchResult) => {
    navigate('/documents', {
      state: {
        openDocId: result.id,
        highlightChunkId: result.chunk_id,
      },
    });
  }, [navigate]);

  return (
    <div className="p-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Search Documents</h1>
        <p className="text-gray-600">Find information across all your knowledge base documents</p>
      </div>

      {/* Search Input */}
      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleSubmit();
              }
            }}
            placeholder="Search for documents, topics, or questions..."
            className="w-full pl-12 pr-4 py-3 text-lg border border-gray-300 rounded-lg shadow-sm bg-white text-gray-900 placeholder-gray-400 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            autoFocus
          />
          {isFetching && (
            <Loader2 className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 animate-spin" />
          )}
        </div>
        <div className="mt-3 flex justify-end">
          <button
            onClick={handleSubmit}
            className="px-4 py-2 rounded-lg bg-primary-600 text-white text-sm font-medium hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={query.trim().length < 2 || isFetching}
          >
            Search
          </button>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6 pb-4 border-b border-gray-200">
        {/* Search Mode */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Mode:</span>
          <div className="flex rounded-lg border border-gray-300 overflow-hidden">
            {[
              { value: 'smart', label: 'Smart', desc: 'AI-powered semantic search' },
              { value: 'keyword', label: 'Keyword', desc: 'Traditional keyword matching' },
              { value: 'exact', label: 'Exact', desc: 'Exact phrase matching' },
            ].map((opt) => (
              <button
                key={opt.value}
                onClick={() => setMode(opt.value as SearchMode)}
                className={`px-3 py-1.5 text-sm transition-colors ${
                  mode === opt.value
                    ? 'bg-primary-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
                title={opt.desc}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Sort Controls */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Sort by:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SearchSortBy)}
            className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:ring-2 focus:ring-primary-500"
          >
            <option value="relevance">Relevance</option>
            <option value="date">Date</option>
            <option value="title">Title</option>
          </select>
          <button
            onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
            className="px-2 py-1.5 border border-gray-300 rounded-lg text-sm hover:bg-gray-50"
            title={sortOrder === 'desc' ? 'Descending' : 'Ascending'}
          >
            {sortOrder === 'desc' ? '↓' : '↑'}
          </button>
        </div>
      </div>

      {/* Results Info */}
      {data && submittedQuery.trim().length >= 2 && (
        <div className="flex items-center justify-between mb-4">
          <p className="text-sm text-gray-600">
            Showing {((page - 1) * pageSize) + 1}-{Math.min(page * pageSize, data.total)} of {data.total} results
            <span className="text-gray-400 ml-2">({data.took_ms}ms)</span>
          </p>
        </div>
      )}

      {/* Results */}
      <div className="space-y-4">
        {/* Empty state - no query */}
        {submittedQuery.trim().length < 2 ? (
          <div className="text-center py-16">
            <FileText className="w-16 h-16 mx-auto text-gray-300 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Start searching</h3>
            <p className="text-gray-600">Enter at least 2 characters and click Search</p>
          </div>
        ) : isLoading ? (
          /* Loading state */
          <div className="text-center py-16">
            <Loader2 className="w-8 h-8 mx-auto text-primary-600 animate-spin mb-4" />
            <p className="text-gray-600">Searching...</p>
          </div>
        ) : error ? (
          /* Error state */
          <div className="text-center py-16">
            <p className="text-red-600">Failed to search. Please try again.</p>
          </div>
        ) : data && data.results.length === 0 ? (
          /* No results */
          <div className="text-center py-16">
            <FileText className="w-16 h-16 mx-auto text-gray-300 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
            <p className="text-gray-600">Try different keywords or switch search mode</p>
          </div>
        ) : data && data.results.length > 0 ? (
          /* Results list */
          <>
            {data.results.map((result) => (
              <SearchResultCard
                key={result.id + (result.chunk_id || '')}
                result={result}
                query={submittedQuery}
                onView={() => handleViewDocument(result)}
              />
            ))}
          </>
        ) : null}
      </div>

      {/* Pagination */}
      {data && totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-8 pt-6 border-t border-gray-200">
          <button
            onClick={() => setPage(Math.max(1, page - 1))}
            disabled={page === 1}
            className="flex items-center gap-1 px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="w-4 h-4" />
            Previous
          </button>
          <div className="flex items-center gap-1">
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              let pageNum: number;
              if (totalPages <= 5) {
                pageNum = i + 1;
              } else if (page <= 3) {
                pageNum = i + 1;
              } else if (page >= totalPages - 2) {
                pageNum = totalPages - 4 + i;
              } else {
                pageNum = page - 2 + i;
              }
              return (
                <button
                  key={pageNum}
                  onClick={() => setPage(pageNum)}
                  className={`w-10 h-10 text-sm rounded-lg ${
                    page === pageNum
                      ? 'bg-primary-600 text-white'
                      : 'border border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  {pageNum}
                </button>
              );
            })}
          </div>
          <button
            onClick={() => setPage(Math.min(totalPages, page + 1))}
            disabled={page === totalPages}
            className="flex items-center gap-1 px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
};

export default SearchPage;
