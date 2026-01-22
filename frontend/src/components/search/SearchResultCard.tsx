/**
 * Search result card with highlighted snippet and actions
 */

import React, { useState } from 'react';
import { Eye, ExternalLink, Download, FileText, Code, Video, Loader2 } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { SearchResult } from '../../types';
import { apiClient } from '../../services/api';
import toast from 'react-hot-toast';

interface SearchResultCardProps {
  result: SearchResult;
  query: string;
  onView: () => void;
}

// Get icon based on file type
const getFileIcon = (fileType?: string) => {
  if (!fileType) return FileText;
  const type = fileType.toLowerCase();
  if (type.includes('video') || type.includes('mp4') || type.includes('webm')) return Video;
  if (type.includes('code') || type.includes('py') || type.includes('js') || type.includes('ts')) return Code;
  return FileText;
};

// Get badge color based on source type
const getSourceColor = (sourceType: string) => {
  const colors: Record<string, string> = {
    gitlab: 'bg-orange-100 text-orange-800',
    github: 'bg-gray-100 text-gray-800',
    confluence: 'bg-blue-100 text-blue-800',
    web: 'bg-green-100 text-green-800',
    arxiv: 'bg-red-100 text-red-800',
    file: 'bg-purple-100 text-purple-800',
  };
  return colors[sourceType.toLowerCase()] || 'bg-gray-100 text-gray-800';
};

// Highlight query terms in text
const highlightText = (text: string, query: string): React.ReactNode => {
  if (!query.trim()) return text;

  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
  if (terms.length === 0) return text;

  // Create regex pattern for all terms
  const pattern = new RegExp(`(${terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  const parts = text.split(pattern);

  return (
    <>
      {parts.map((part, i) => {
        const isMatch = terms.some(t => part.toLowerCase() === t.toLowerCase());
        if (isMatch) {
          return (
            <mark key={i} className="bg-yellow-200 text-gray-900 px-0.5 rounded">
              {part}
            </mark>
          );
        }
        return <React.Fragment key={i}>{part}</React.Fragment>;
      })}
    </>
  );
};

const SearchResultCard: React.FC<SearchResultCardProps> = ({ result, query, onView }) => {
  const [downloading, setDownloading] = useState(false);
  const FileIcon = getFileIcon(result.file_type);

  const handleDownload = async () => {
    if (!result.id) return;
    try {
      setDownloading(true);
      const { blob, filename } = await apiClient.downloadDocument(result.id, true);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      toast.success('Download started');
    } catch (error) {
      toast.error('Failed to download document');
    } finally {
      setDownloading(false);
    }
  };

  const relevancePercent = Math.round(result.relevance_score * 100);

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3 min-w-0">
          <div className="flex-shrink-0 w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
            <FileIcon className="w-5 h-5 text-gray-600" />
          </div>
          <div className="min-w-0">
            <h3 className="font-medium text-gray-900 truncate">{result.title}</h3>
            <div className="flex items-center gap-2 mt-1 flex-wrap">
              <span className={`text-xs px-2 py-0.5 rounded-full ${getSourceColor(result.source_type)}`}>
                {result.source_type}
              </span>
              <span className="text-xs text-gray-500 truncate">{result.source}</span>
              {result.author && (
                <span className="text-xs text-gray-500">by {result.author}</span>
              )}
            </div>
          </div>
        </div>
        <div className="flex-shrink-0 text-right">
          <div className={`text-sm font-medium ${
            relevancePercent >= 80 ? 'text-green-600' :
            relevancePercent >= 60 ? 'text-yellow-600' :
            'text-gray-600'
          }`}>
            {relevancePercent}%
          </div>
          <div className="text-xs text-gray-500">relevance</div>
        </div>
      </div>

      {/* Snippet */}
      {result.snippet && (
        <div className="mt-3 text-sm text-gray-600 line-clamp-3">
          "{highlightText(result.snippet, query)}"
        </div>
      )}

      {/* Footer */}
      <div className="mt-4 flex items-center justify-between">
        <div className="text-xs text-gray-500">
          Updated {formatDistanceToNow(new Date(result.updated_at))} ago
          {result.file_type && (
            <span className="ml-2 text-gray-400">• {result.file_type}</span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={onView}
            className="flex items-center gap-1 px-3 py-1.5 text-sm text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
            title="View in document"
          >
            <Eye className="w-4 h-4" />
            View
          </button>
          {result.url && (
            <button
              onClick={() => window.open(result.url, '_blank', 'noopener,noreferrer')}
              className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title="Open source"
            >
              <ExternalLink className="w-4 h-4" />
            </button>
          )}
          {result.id && (
            <button
              onClick={handleDownload}
              disabled={downloading}
              className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
              title="Download"
            >
              {downloading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Download className="w-4 h-4" />
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default SearchResultCard;
