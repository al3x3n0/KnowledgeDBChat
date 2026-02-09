/**
 * Floating Agent Widget for document operations.
 *
 * Provides a chat interface for interacting with the document agent.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Bot,
  X,
  Minus,
  Send,
  Trash2,
  Upload,
  Search,
  Globe,
  FileText,
  List,
  Play,
  Loader2,
  CheckCircle,
  XCircle,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  FilePlus,
  Files,
  Tags,
  BarChart3,
  Copy,
  Hash,
  GitCompare,
  FileType,
  Clock,
  Download,
  History,
  Plus,
  MessageSquare,
  GripVertical,
  Network,
  Brain,
  Users,
  Workflow,
  Info,
} from 'lucide-react';
import MermaidDiagram from '../common/MermaidDiagram';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';
import { useAgentWebSocket, AgentMessage, AgentToolCall, StreamingState, ConnectionStatus, TemplateUploadContext, TemplateJobProgress, AgentInfo, AgentRoutingInfo } from './useAgentWebSocket';

// ============================================================================
// AgentToggleButton Component
// ============================================================================

interface AgentToggleButtonProps {
  isOpen: boolean;
  onClick: () => void;
}

const AgentToggleButton: React.FC<AgentToggleButtonProps> = ({ isOpen, onClick }) => {
  return (
    <button
      onClick={onClick}
      className={`fixed bottom-6 right-6 w-14 h-14 rounded-full shadow-lg flex items-center justify-center transition-all duration-300 z-50 ${
        isOpen
          ? 'bg-gray-600 hover:bg-gray-700'
          : 'bg-primary-600 hover:bg-primary-700'
      }`}
      title={isOpen ? 'Close agent' : 'Open document assistant'}
    >
      {isOpen ? (
        <X className="w-6 h-6 text-white" />
      ) : (
        <Bot className="w-6 h-6 text-white" />
      )}
    </button>
  );
};

// ============================================================================
// AgentToolResult Component
// ============================================================================

interface AgentToolResultProps {
  toolCall: AgentToolCall;
}

const AgentToolResult: React.FC<AgentToolResultProps> = ({ toolCall }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getIcon = () => {
    switch (toolCall.tool_name) {
      case 'search_documents': return <Search className="w-3.5 h-3.5" />;
      case 'search_arxiv': return <Search className="w-3.5 h-3.5" />;
      case 'web_scrape': return <Globe className="w-3.5 h-3.5" />;
      case 'ingest_url': return <Download className="w-3.5 h-3.5" />;
      case 'ingest_arxiv_papers': return <Download className="w-3.5 h-3.5" />;
      case 'literature_review_arxiv': return <FileText className="w-3.5 h-3.5" />;
      case 'summarize_document': return <FileText className="w-3.5 h-3.5" />;
      case 'delete_document': return <Trash2 className="w-3.5 h-3.5" />;
      case 'request_file_upload': return <Upload className="w-3.5 h-3.5" />;
      case 'list_recent_documents': return <List className="w-3.5 h-3.5" />;
      case 'get_document_details': return <FileText className="w-3.5 h-3.5" />;
      case 'create_document_from_text': return <FilePlus className="w-3.5 h-3.5" />;
      case 'find_similar_documents': return <Files className="w-3.5 h-3.5" />;
      case 'update_document_tags': return <Tags className="w-3.5 h-3.5" />;
      case 'get_knowledge_base_stats': return <BarChart3 className="w-3.5 h-3.5" />;
      case 'batch_delete_documents': return <Trash2 className="w-3.5 h-3.5" />;
      case 'batch_summarize_documents': return <Copy className="w-3.5 h-3.5" />;
      case 'search_by_tags': return <Hash className="w-3.5 h-3.5" />;
      case 'list_all_tags': return <Tags className="w-3.5 h-3.5" />;
      case 'compare_documents': return <GitCompare className="w-3.5 h-3.5" />;
      case 'start_template_fill': return <FileType className="w-3.5 h-3.5" />;
      case 'list_template_jobs': return <Clock className="w-3.5 h-3.5" />;
      case 'get_template_job_status': return <FileType className="w-3.5 h-3.5" />;
      case 'generate_diagram': return <Network className="w-3.5 h-3.5" />;
      case 'propose_workflow_from_description': return <Workflow className="w-3.5 h-3.5" />;
      case 'create_workflow_from_description': return <Workflow className="w-3.5 h-3.5" />;
      case 'run_workflow': return <Play className="w-3.5 h-3.5" />;
      default: return <Bot className="w-3.5 h-3.5" />;
    }
  };

  const getStatusIcon = () => {
    switch (toolCall.status) {
      case 'completed': return <CheckCircle className="w-3.5 h-3.5 text-green-500" />;
      case 'failed': return <XCircle className="w-3.5 h-3.5 text-red-500" />;
      case 'running': return <Loader2 className="w-3.5 h-3.5 text-blue-500 animate-spin" />;
      default: return <AlertCircle className="w-3.5 h-3.5 text-gray-400" />;
    }
  };

  const getStatusBgColor = () => {
    switch (toolCall.status) {
      case 'completed': return 'bg-green-50 border-green-200';
      case 'failed': return 'bg-red-50 border-red-200';
      case 'running': return 'bg-blue-50 border-blue-200';
      default: return 'bg-gray-50 border-gray-200';
    }
  };

  const formatToolName = (name: string) => {
    return name.split('_').map(word =>
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  const hasOutput = toolCall.tool_output &&
    (Array.isArray(toolCall.tool_output) ? toolCall.tool_output.length > 0 : Object.keys(toolCall.tool_output).length > 0);

  return (
    <div className={`rounded-md border text-xs ${getStatusBgColor()}`}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-2 py-1.5 flex items-center justify-between"
      >
        <div className="flex items-center space-x-1.5">
          {getIcon()}
          <span className="font-medium">{formatToolName(toolCall.tool_name)}</span>
          {getStatusIcon()}
          {toolCall.execution_time_ms && (
            <span className="text-gray-400">({toolCall.execution_time_ms}ms)</span>
          )}
        </div>
        {hasOutput && (
          isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />
        )}
      </button>

      {isExpanded && hasOutput && (
        <div className="px-2 pb-2 border-t border-gray-200 mt-1 pt-1">
          <ToolOutputDisplay output={toolCall.tool_output} toolName={toolCall.tool_name} />
        </div>
      )}

      {toolCall.error && (
        <div className="px-2 pb-2 text-red-600">
          Error: {toolCall.error}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// ToolOutputDisplay Component
// ============================================================================

interface ToolOutputDisplayProps {
  output: any;
  toolName: string;
}

const ToolOutputDisplay: React.FC<ToolOutputDisplayProps> = ({ output, toolName }) => {
  // Ingest URL
  if (toolName === 'ingest_url' && output && (Array.isArray(output.created) || Array.isArray(output.updated))) {
    const created = Array.isArray(output.created) ? output.created : [];
    const updated = Array.isArray(output.updated) ? output.updated : [];
    const skipped = Array.isArray(output.skipped) ? output.skipped : [];
    const errors = Array.isArray(output.errors) ? output.errors : [];

    return (
      <div className="space-y-2">
        <div className="text-[10px] text-gray-600">
          Created: {created.length} · Updated: {updated.length} · Skipped: {skipped.length} · Errors: {errors.length}
        </div>
        {[...created, ...updated].slice(0, 5).map((item: any, idx: number) => (
          <div key={idx} className="bg-white rounded px-2 py-1 border">
            <div className="font-medium truncate">{item.title || 'Document'}</div>
            <div className="text-gray-500 truncate text-[10px]">{item.url}</div>
          </div>
        ))}
      </div>
    );
  }

  // Web scrape
  if (toolName === 'web_scrape' && output && Array.isArray(output.pages)) {
    return (
      <div className="space-y-1">
        {output.pages.slice(0, 5).map((page: any, idx: number) => (
          <div key={idx} className="bg-white rounded px-2 py-1 border">
            <div className="font-medium truncate">{page.title || 'Untitled page'}</div>
            <div className="text-gray-500 truncate text-[10px]">{page.url}</div>
            {page.content && (
              <div className="text-gray-700 text-[11px] mt-1 line-clamp-3 whitespace-pre-wrap">
                {String(page.content).slice(0, 250)}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  }

  // Search results
  if (toolName === 'search_documents' && Array.isArray(output)) {
    return (
      <div className="space-y-1">
        {output.map((doc: any, idx: number) => (
          <div key={idx} className="bg-white rounded px-2 py-1 border">
            <div className="font-medium truncate">{doc.title}</div>
            <div className="text-gray-500 truncate text-[10px]">
              Score: {(doc.score * 100).toFixed(0)}% | {doc.source_type}
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Recent documents list
  if (toolName === 'list_recent_documents' && Array.isArray(output)) {
    return (
      <div className="space-y-1">
        {output.map((doc: any, idx: number) => (
          <div key={idx} className="bg-white rounded px-2 py-1 border flex justify-between">
            <span className="truncate">{doc.title}</span>
            <span className="text-gray-400">{doc.file_type}</span>
          </div>
        ))}
      </div>
    );
  }

  // Similar documents
  if (toolName === 'find_similar_documents' && output.similar_documents) {
    return (
      <div className="space-y-1">
        <div className="text-gray-500 text-[10px] mb-1">
          Similar to: {output.reference_document?.title}
        </div>
        {output.similar_documents.map((doc: any, idx: number) => (
          <div key={idx} className="bg-white rounded px-2 py-1 border">
            <div className="font-medium truncate">{doc.title}</div>
            <div className="text-gray-500 text-[10px]">
              Similarity: {(doc.similarity_score * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </div>
    );
  }

  // arXiv paper search
  if (toolName === 'search_arxiv' && output?.items && Array.isArray(output.items)) {
    return (
      <div className="space-y-1">
        <div className="text-gray-500 text-[10px] mb-1">
          Total: {output.total_results ?? output.items.length} • Showing: {output.items.length}
        </div>
        {output.items.slice(0, 10).map((paper: any) => (
          <div key={paper.entry_url || paper.id} className="bg-white rounded px-2 py-1 border space-y-0.5">
            <div className="font-medium truncate">{paper.title}</div>
            <div className="text-gray-500 truncate text-[10px]">
              {(paper.authors || []).slice(0, 3).join(', ')}
              {(paper.authors || []).length > 3 ? ' et al.' : ''}
              {paper.primary_category ? ` • ${paper.primary_category}` : ''}
              {paper.published ? ` • ${String(paper.published).slice(0, 10)}` : ''}
            </div>
            <div className="flex items-center gap-2 text-[10px]">
              {paper.pdf_url && (
                <a
                  href={paper.pdf_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary-600 hover:text-primary-700 underline"
                >
                  PDF
                </a>
              )}
              {paper.entry_url && (
                <a
                  href={paper.entry_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary-600 hover:text-primary-700 underline"
                >
                  arXiv
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Ingest arXiv papers
  if (toolName === 'ingest_arxiv_papers' && output?.source_id) {
    return (
      <div className="bg-white rounded px-2 py-2 border space-y-1">
        <div className="font-medium text-[10px] truncate">{output.source_name || 'ArXiv import'}</div>
        <div className="text-[10px] text-gray-600">
          Source: <span className="font-mono">{output.source_id}</span>
        </div>
        <div className="text-[10px] text-gray-600">
          Papers: {output.paper_ids_count ?? 0} • Queries: {output.search_queries_count ?? 0} • Categories: {output.categories_count ?? 0}
        </div>
        <div className="text-[10px] text-gray-500">
          {output.queued ? 'Queued for ingestion.' : 'Created source.'}
        </div>
      </div>
    );
  }

  // Literature review (arXiv)
  if (toolName === 'literature_review_arxiv' && output?.papers && Array.isArray(output.papers)) {
    return (
      <div className="bg-white rounded px-2 py-2 border space-y-2">
        <div className="text-[10px] text-gray-600">
          Topic: <span className="font-medium text-gray-900">{output.topic}</span>
        </div>
        <div className="text-[10px] text-gray-500 font-mono truncate" title={output.query}>
          {output.query}
        </div>
        {output.ingest?.source_id && (
          <div className="text-[10px] text-gray-600">
            Import source: <span className="font-mono">{output.ingest.source_id}</span>
          </div>
        )}
        <div className="space-y-1">
          {output.papers.slice(0, 5).map((paper: any) => (
            <div key={paper.entry_url || paper.id} className="border rounded px-2 py-1">
              <div className="font-medium truncate">{paper.title}</div>
              <div className="text-gray-500 text-[10px] truncate">
                {(paper.authors || []).slice(0, 3).join(', ')}
                {(paper.authors || []).length > 3 ? ' et al.' : ''}
                {paper.primary_category ? ` • ${paper.primary_category}` : ''}
              </div>
            </div>
          ))}
        </div>
        {Array.isArray(output.next_steps) && output.next_steps.length > 0 && (
          <div className="text-[10px] text-gray-500">
            Next: {output.next_steps[0]}
          </div>
        )}
      </div>
    );
  }

  // Knowledge base stats
  if (toolName === 'get_knowledge_base_stats' && output.total_documents !== undefined) {
    return (
      <div className="bg-white rounded px-2 py-1 border space-y-1">
        <div className="grid grid-cols-2 gap-x-2 text-[10px]">
          <span className="text-gray-500">Total Docs:</span>
          <span className="font-medium">{output.total_documents}</span>
          <span className="text-gray-500">Processed:</span>
          <span className="font-medium">{output.processed_documents}</span>
          <span className="text-gray-500">Summarized:</span>
          <span className="font-medium">{output.summarized_documents}</span>
          <span className="text-gray-500">Storage:</span>
          <span className="font-medium">{output.total_storage_mb} MB</span>
          <span className="text-gray-500">Last 7 days:</span>
          <span className="font-medium">{output.documents_last_7_days}</span>
        </div>
      </div>
    );
  }

  // Workflow draft proposal
  if (toolName === 'propose_workflow_from_description' && output?.workflow) {
    const workflow = output.workflow;
    const warnings: string[] = output.warnings || [];

    const openDraft = () => {
      try {
        localStorage.setItem('workflow_draft_pending', JSON.stringify(workflow));
        window.location.href = '/workflows/new/edit';
      } catch {
        // fallback: best effort
        window.location.href = '/workflows/new/edit';
      }
    };

    return (
      <div className="bg-white rounded px-2 py-2 border space-y-2">
        <div className="flex items-start justify-between">
          <div className="min-w-0">
            <div className="font-medium truncate">{workflow.name || 'Workflow Draft'}</div>
            <div className="text-[10px] text-gray-500">
              Nodes: {workflow.nodes?.length ?? 0} • Edges: {workflow.edges?.length ?? 0} • Trigger: {workflow.trigger_config?.type || 'manual'}
            </div>
          </div>
          <button
            onClick={openDraft}
            className="text-[10px] px-2 py-1 rounded bg-primary-600 text-white hover:bg-primary-700"
          >
            Review
          </button>
        </div>
        {workflow.description && (
          <div className="text-[10px] text-gray-600 line-clamp-3">{workflow.description}</div>
        )}
        {warnings.length > 0 && (
          <div className="text-[10px] text-amber-700 bg-amber-50 border border-amber-200 rounded p-1">
            {warnings.length} warning{warnings.length === 1 ? '' : 's'} in draft
          </div>
        )}
        <div className="text-[10px] text-gray-500">
          Review in the editor, then click Save to approve and make it runnable.
        </div>
      </div>
    );
  }

  // Workflow created
  if (toolName === 'create_workflow_from_description' && output?.workflow_id) {
    const openWorkflow = () => {
      window.location.href = `/workflows/${output.workflow_id}/edit`;
    };

    return (
      <div className="bg-white rounded px-2 py-2 border space-y-2">
        <div className="flex items-start justify-between">
          <div className="min-w-0">
            <div className="font-medium truncate">{output.workflow_name || 'Workflow Created'}</div>
            <div className="text-[10px] text-gray-500">
              Nodes: {output.node_count ?? 0} • Edges: {output.edge_count ?? 0}
            </div>
          </div>
          <button
            onClick={openWorkflow}
            className="text-[10px] px-2 py-1 rounded bg-primary-600 text-white hover:bg-primary-700"
          >
            Open
          </button>
        </div>
        {Array.isArray(output.warnings) && output.warnings.length > 0 && (
          <div className="text-[10px] text-amber-700 bg-amber-50 border border-amber-200 rounded p-1">
            {output.warnings.length} warning{output.warnings.length === 1 ? '' : 's'}
          </div>
        )}
      </div>
    );
  }

  // Tag search results
  if (toolName === 'search_by_tags' && output.documents) {
    return (
      <div className="space-y-1">
        <div className="text-gray-500 text-[10px] mb-1">
          Tags: {output.search_tags?.join(', ')} ({output.match_type})
        </div>
        {output.documents.map((doc: any, idx: number) => (
          <div key={idx} className="bg-white rounded px-2 py-1 border">
            <div className="font-medium truncate">{doc.title}</div>
            <div className="text-gray-500 text-[10px] truncate">
              {doc.tags?.join(', ')}
            </div>
          </div>
        ))}
      </div>
    );
  }

  // All tags list
  if (toolName === 'list_all_tags' && output.tags) {
    return (
      <div className="bg-white rounded px-2 py-1 border">
        <div className="text-[10px] text-gray-500 mb-1">
          {output.total_unique_tags} unique tags
        </div>
        <div className="flex flex-wrap gap-1">
          {output.tags.slice(0, 15).map((item: any, idx: number) => (
            <span key={idx} className="bg-gray-100 px-1.5 py-0.5 rounded text-[10px]">
              {item.tag} ({item.count})
            </span>
          ))}
          {output.tags.length > 15 && (
            <span className="text-[10px] text-gray-400">+{output.tags.length - 15} more</span>
          )}
        </div>
      </div>
    );
  }

  // Tag update result
  if (toolName === 'update_document_tags' && output.current_tags) {
    return (
      <div className="bg-white rounded px-2 py-1 border">
        <div className="font-medium text-[10px] mb-1">{output.title}</div>
        <div className="flex flex-wrap gap-1">
          {output.current_tags.map((tag: string, idx: number) => (
            <span key={idx} className="bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded text-[10px]">
              {tag}
            </span>
          ))}
        </div>
      </div>
    );
  }

  // Batch operations
  if ((toolName === 'batch_delete_documents' || toolName === 'batch_summarize_documents') && output.message) {
    return (
      <div className="bg-white rounded px-2 py-1 border">
        <div className="text-[10px]">{output.message}</div>
        {output.deleted_count !== undefined && (
          <div className="text-[10px] text-green-600">Deleted: {output.deleted_count}</div>
        )}
        {output.queued_count !== undefined && (
          <div className="text-[10px] text-blue-600">Queued: {output.queued_count}</div>
        )}
        {output.failed_count > 0 && (
          <div className="text-[10px] text-red-600">Failed: {output.failed_count}</div>
        )}
      </div>
    );
  }

  // Document created
  if (toolName === 'create_document_from_text' && output.action === 'created') {
    return (
      <div className="bg-white rounded px-2 py-1 border">
        <div className="font-medium text-[10px]">{output.title}</div>
        <div className="text-gray-500 text-[10px] truncate">{output.content_preview}</div>
        {output.tags?.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1">
            {output.tags.map((tag: string, idx: number) => (
              <span key={idx} className="bg-gray-100 px-1 py-0.5 rounded text-[9px]">
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Document comparison
  if (toolName === 'compare_documents' && output.document_1) {
    return (
      <div className="bg-white rounded px-2 py-1 border space-y-2">
        {/* Documents being compared */}
        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <div className="bg-blue-50 p-1 rounded">
            <div className="font-medium truncate">{output.document_1.title}</div>
            <div className="text-gray-500">{output.document_1.word_count} words</div>
          </div>
          <div className="bg-green-50 p-1 rounded">
            <div className="font-medium truncate">{output.document_2.title}</div>
            <div className="text-gray-500">{output.document_2.word_count} words</div>
          </div>
        </div>

        {/* Similarity scores */}
        {output.keyword_analysis && (
          <div className="text-[10px]">
            <span className="text-gray-500">Keyword similarity: </span>
            <span className="font-medium">{(output.keyword_analysis.similarity_score * 100).toFixed(0)}%</span>
            <span className="text-gray-400 ml-1">({output.keyword_analysis.common_word_count} common words)</span>
          </div>
        )}
        {output.semantic_analysis && !output.semantic_analysis.error && (
          <div className="text-[10px]">
            <span className="text-gray-500">Semantic similarity: </span>
            <span className="font-medium">{(output.semantic_analysis.similarity_score * 100).toFixed(0)}%</span>
            <span className="text-gray-400 ml-1">({output.semantic_analysis.interpretation})</span>
          </div>
        )}

        {/* Summary */}
        {output.comparison_summary && (
          <div className="text-[10px] text-gray-600 border-t pt-1">
            {output.comparison_summary}
          </div>
        )}
      </div>
    );
  }

  // Template fill request
  if (toolName === 'start_template_fill' && output.action === 'template_upload_required') {
    return (
      <div className="bg-white rounded px-2 py-1 border">
        <div className="text-[10px] font-medium text-amber-700 mb-1">Template Upload Required</div>
        <div className="text-[10px] text-gray-600 mb-1">{output.message}</div>
        <div className="text-[10px] text-gray-500">
          Source documents: {output.source_documents?.length || 0}
        </div>
        {output.source_documents?.slice(0, 3).map((doc: any, idx: number) => (
          <div key={idx} className="text-[9px] text-gray-400 truncate">• {doc.title}</div>
        ))}
      </div>
    );
  }

  // Template jobs list
  if (toolName === 'list_template_jobs' && output.jobs) {
    return (
      <div className="space-y-1">
        <div className="text-[10px] text-gray-500 mb-1">
          {output.count} template job(s)
        </div>
        {output.jobs.map((job: any, idx: number) => (
          <div key={idx} className="bg-white rounded px-2 py-1 border">
            <div className="flex justify-between items-center">
              <span className="font-medium text-[10px] truncate">{job.template_filename}</span>
              <span className={`text-[9px] px-1 rounded ${
                job.status === 'completed' ? 'bg-green-100 text-green-700' :
                job.status === 'failed' ? 'bg-red-100 text-red-700' :
                'bg-yellow-100 text-yellow-700'
              }`}>
                {job.status}
              </span>
            </div>
            {job.status !== 'completed' && job.status !== 'failed' && (
              <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
                <div
                  className="bg-blue-500 h-1 rounded-full"
                  style={{ width: `${job.progress}%` }}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    );
  }

  // Template job status
  if (toolName === 'get_template_job_status' && output.status) {
    const statusColors: Record<string, string> = {
      pending: 'bg-gray-100 text-gray-700',
      analyzing: 'bg-blue-100 text-blue-700',
      extracting: 'bg-purple-100 text-purple-700',
      filling: 'bg-amber-100 text-amber-700',
      completed: 'bg-green-100 text-green-700',
      failed: 'bg-red-100 text-red-700',
    };

    return (
      <div className="bg-white rounded px-2 py-1 border">
        <div className="flex justify-between items-center mb-1">
          <span className="font-medium text-[10px] truncate">{output.template_filename}</span>
          <span className={`text-[9px] px-1 rounded ${statusColors[output.status] || 'bg-gray-100'}`}>
            {output.status}
          </span>
        </div>

        {output.progress !== undefined && output.status !== 'completed' && (
          <div className="mb-1">
            <div className="w-full bg-gray-200 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full transition-all"
                style={{ width: `${output.progress}%` }}
              />
            </div>
            <div className="text-[9px] text-gray-500 mt-0.5">
              {output.progress}% - {output.current_section || 'Processing...'}
            </div>
          </div>
        )}

        {output.total_sections && (
          <div className="text-[9px] text-gray-500">
            {output.total_sections} sections
          </div>
        )}

        {output.download_available && (
          <a
            href={output.download_url}
            className="inline-flex items-center space-x-1 text-[10px] text-blue-600 hover:underline mt-1"
          >
            <Download className="w-3 h-3" />
            <span>Download filled document</span>
          </a>
        )}

        {output.error_message && (
          <div className="text-[9px] text-red-600 mt-1">
            Error: {output.error_message}
          </div>
        )}
      </div>
    );
  }

  // Generate diagram output
  if (toolName === 'generate_diagram' && output.mermaid_code) {
    return (
      <div className="space-y-2">
        {/* Diagram title */}
        <div className="text-[10px] text-gray-500">
          {output.diagram_type?.charAt(0).toUpperCase() + output.diagram_type?.slice(1)} diagram
          {output.source_documents?.length > 0 && (
            <span> • {output.source_documents.length} source document(s)</span>
          )}
        </div>

        {/* Mermaid diagram */}
        {output.can_render !== false ? (
          <MermaidDiagram
            code={output.mermaid_code}
            title={output.diagram_type ? `${output.diagram_type.charAt(0).toUpperCase() + output.diagram_type.slice(1)} Diagram` : 'Generated Diagram'}
            className="text-xs"
          />
        ) : (
          <div className="bg-white rounded px-2 py-1 border">
            <div className="text-amber-600 text-[10px] mb-1">Diagram code (cannot render in widget)</div>
            <pre className="text-[10px] overflow-x-auto whitespace-pre-wrap">{output.mermaid_code}</pre>
          </div>
        )}

        {/* Source documents list */}
        {output.source_documents?.length > 0 && (
          <details className="text-[10px]">
            <summary className="cursor-pointer text-gray-500 hover:text-gray-700">
              View source documents ({output.source_documents.length})
            </summary>
            <div className="mt-1 space-y-0.5 pl-2">
              {output.source_documents.map((doc: any, idx: number) => (
                <div key={idx} className="text-gray-600 truncate">• {doc.title}</div>
              ))}
            </div>
          </details>
        )}
      </div>
    );
  }

  // Summary
  if (output.summary) {
    return (
      <div className="bg-white rounded px-2 py-1 border">
        <div className="font-medium mb-1">{output.title}</div>
        <div className="text-gray-600 line-clamp-3">{output.summary}</div>
      </div>
    );
  }

  // Content preview
  if (output.content_preview) {
    return (
      <div className="bg-white rounded px-2 py-1 border">
        <div className="font-medium mb-1">{output.title}</div>
        <div className="text-gray-600 line-clamp-2">{output.content_preview}</div>
      </div>
    );
  }

  // Generic JSON display
  return (
    <pre className="bg-white rounded px-2 py-1 border overflow-x-auto text-[10px]">
      {JSON.stringify(output, null, 2)}
    </pre>
  );
};

// ============================================================================
// AgentMessage Component
// ============================================================================

interface AgentMessageComponentProps {
  message: AgentMessage;
}

const AgentMessageComponent: React.FC<AgentMessageComponentProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] rounded-lg px-3 py-2 ${
        isUser
          ? 'bg-primary-600 text-white'
          : 'bg-gray-100 text-gray-900'
      }`}>
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="text-sm prose prose-sm max-w-none">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}

        {/* Tool results */}
        {message.tool_calls && message.tool_calls.length > 0 && (
          <div className="mt-2 space-y-1.5">
            {message.tool_calls.map((tool) => (
              <AgentToolResult key={tool.id} toolCall={tool} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// FileUploadArea Component
// ============================================================================

interface FileUploadAreaProps {
  onUpload: (file: File) => void;
  onCancel: () => void;
  isLoading: boolean;
  suggestedTitle?: string;
  suggestedTags?: string[];
}

const FileUploadArea: React.FC<FileUploadAreaProps> = ({
  onUpload,
  onCancel,
  isLoading,
  suggestedTitle,
  suggestedTags
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  return (
    <div className="p-3 border-t bg-blue-50">
      <div
        className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors ${
          dragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {isLoading ? (
          <div className="flex items-center justify-center space-x-2">
            <Loader2 className="w-5 h-5 animate-spin text-primary-600" />
            <span className="text-sm text-gray-600">Uploading...</span>
          </div>
        ) : (
          <>
            <Upload className="w-8 h-8 mx-auto text-gray-400 mb-2" />
            <p className="text-sm text-gray-600 mb-2">
              Drag & drop a file here, or{' '}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-primary-600 hover:underline"
              >
                browse
              </button>
            </p>
            {suggestedTitle && (
              <p className="text-xs text-gray-500">Suggested title: {suggestedTitle}</p>
            )}
            {suggestedTags && suggestedTags.length > 0 && (
              <p className="text-xs text-gray-500">Tags: {suggestedTags.join(', ')}</p>
            )}
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileSelect}
              className="hidden"
              accept=".pdf,.doc,.docx,.txt,.md,.json,.csv"
            />
          </>
        )}
      </div>
      <button
        onClick={onCancel}
        className="mt-2 text-xs text-gray-500 hover:text-gray-700"
      >
        Cancel upload
      </button>
    </div>
  );
};

// ============================================================================
// TemplateUploadArea Component
// ============================================================================

interface TemplateUploadAreaProps {
  onUpload: (file: File) => void;
  onCancel: () => void;
  isLoading: boolean;
  sourceDocuments?: { id: string; title: string }[];
}

const TemplateUploadArea: React.FC<TemplateUploadAreaProps> = ({
  onUpload,
  onCancel,
  isLoading,
  sourceDocuments
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.docx')) {
        onUpload(file);
      } else {
        toast.error('Please upload a .docx file');
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.name.endsWith('.docx')) {
        onUpload(file);
      } else {
        toast.error('Please upload a .docx file');
      }
    }
  };

  return (
    <div className="p-3 border-t bg-amber-50">
      <div
        className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors ${
          dragActive ? 'border-amber-500 bg-amber-100' : 'border-amber-300'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {isLoading ? (
          <div className="flex items-center justify-center space-x-2">
            <Loader2 className="w-5 h-5 animate-spin text-amber-600" />
            <span className="text-sm text-gray-600">Uploading template...</span>
          </div>
        ) : (
          <>
            <FileType className="w-8 h-8 mx-auto text-amber-500 mb-2" />
            <p className="text-sm text-gray-600 mb-2">
              Upload your DOCX template file, or{' '}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-amber-600 hover:underline font-medium"
              >
                browse
              </button>
            </p>
            <p className="text-xs text-gray-500 mb-2">
              Only .docx files are supported
            </p>
            {sourceDocuments && sourceDocuments.length > 0 && (
              <div className="text-xs text-gray-500 border-t pt-2 mt-2">
                <p className="font-medium mb-1">Source documents:</p>
                <div className="max-h-20 overflow-y-auto">
                  {sourceDocuments.slice(0, 5).map((doc, idx) => (
                    <div key={idx} className="truncate text-left px-2">• {doc.title}</div>
                  ))}
                  {sourceDocuments.length > 5 && (
                    <div className="text-gray-400 px-2">+{sourceDocuments.length - 5} more</div>
                  )}
                </div>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileSelect}
              className="hidden"
              accept=".docx"
            />
          </>
        )}
      </div>
      <button
        onClick={onCancel}
        className="mt-2 text-xs text-gray-500 hover:text-gray-700"
      >
        Cancel template upload
      </button>
    </div>
  );
};

// ============================================================================
// TemplateJobProgress Component
// ============================================================================

interface TemplateJobProgressProps {
  job: {
    jobId: string;
    templateFilename: string;
    status: string;
    progress: number;
    currentSection?: string;
    downloadUrl?: string;
    error?: string;
  };
}

const TemplateJobProgressComponent: React.FC<TemplateJobProgressProps> = ({ job }) => {
  const statusColors: Record<string, string> = {
    pending: 'bg-gray-100 text-gray-700',
    analyzing: 'bg-blue-100 text-blue-700',
    extracting: 'bg-purple-100 text-purple-700',
    filling: 'bg-amber-100 text-amber-700',
    completed: 'bg-green-100 text-green-700',
    failed: 'bg-red-100 text-red-700',
  };

  return (
    <div className="p-3 border-t bg-blue-50">
      <div className="flex items-start space-x-2">
        <FileType className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <div className="flex justify-between items-center mb-1">
            <p className="text-sm font-medium text-gray-800 truncate">{job.templateFilename}</p>
            <span className={`text-xs px-1.5 py-0.5 rounded ${statusColors[job.status] || 'bg-gray-100'}`}>
              {job.status}
            </span>
          </div>

          {job.status !== 'completed' && job.status !== 'failed' && (
            <>
              <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${job.progress}%` }}
                />
              </div>
              <p className="text-xs text-gray-500">
                {job.progress}% - {job.currentSection || 'Processing...'}
              </p>
            </>
          )}

          {job.status === 'completed' && job.downloadUrl && (
            <a
              href={job.downloadUrl}
              className="inline-flex items-center space-x-1 text-sm text-blue-600 hover:underline mt-1"
            >
              <Download className="w-4 h-4" />
              <span>Download filled document</span>
            </a>
          )}

          {job.error && (
            <p className="text-xs text-red-600 mt-1">Error: {job.error}</p>
          )}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// DeleteConfirmation Component
// ============================================================================

interface DeleteConfirmationProps {
  documentTitle: string;
  onConfirm: () => void;
  onCancel: () => void;
  isLoading: boolean;
}

const DeleteConfirmation: React.FC<DeleteConfirmationProps> = ({
  documentTitle,
  onConfirm,
  onCancel,
  isLoading,
}) => {
  return (
    <div className="p-3 border-t bg-red-50">
      <div className="flex items-start space-x-2">
        <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="text-sm font-medium text-red-800">Confirm Deletion</p>
          <p className="text-xs text-red-600 mt-1">
            Are you sure you want to delete "{documentTitle}"? This cannot be undone.
          </p>
          <div className="flex space-x-2 mt-2">
            <button
              onClick={onConfirm}
              disabled={isLoading}
              className="px-3 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700 disabled:opacity-50"
            >
              {isLoading ? 'Deleting...' : 'Delete'}
            </button>
            <button
              onClick={onCancel}
              disabled={isLoading}
              className="px-3 py-1 bg-gray-200 text-gray-700 text-xs rounded hover:bg-gray-300"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// StreamingIndicator Component
// ============================================================================

interface StreamingIndicatorProps {
  streamingState: StreamingState;
}

const StreamingIndicator: React.FC<StreamingIndicatorProps> = ({ streamingState }) => {
  if (streamingState.phase === 'idle') return null;

  const getPhaseColor = () => {
    switch (streamingState.phase) {
      case 'thinking': return 'text-blue-600 bg-blue-50';
      case 'planning': return 'text-purple-600 bg-purple-50';
      case 'executing': return 'text-amber-600 bg-amber-50';
      case 'generating': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getPhaseIcon = () => {
    switch (streamingState.phase) {
      case 'thinking':
      case 'planning':
      case 'generating':
        return <Loader2 className="w-3.5 h-3.5 animate-spin" />;
      case 'executing':
        return streamingState.currentTool ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
        ) : (
          <CheckCircle className="w-3.5 h-3.5" />
        );
      default:
        return <Loader2 className="w-3.5 h-3.5 animate-spin" />;
    }
  };

  return (
    <div className={`rounded-lg px-3 py-2 text-sm ${getPhaseColor()}`}>
      <div className="flex items-center space-x-2">
        {getPhaseIcon()}
        <span className="font-medium">{streamingState.message}</span>
      </div>

      {/* Show tool progress */}
      {streamingState.phase === 'executing' && streamingState.toolCount && (
        <div className="mt-2 text-xs">
          <div className="flex items-center space-x-2 mb-1">
            <span>Progress: {streamingState.completedTools.length}/{streamingState.toolCount}</span>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-gray-200 rounded-full h-1.5">
            <div
              className="bg-amber-500 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${(streamingState.completedTools.length / streamingState.toolCount) * 100}%` }}
            />
          </div>

          {/* Current tool */}
          {streamingState.currentTool && (
            <div className="mt-2 flex items-center space-x-1.5 text-amber-700">
              <Loader2 className="w-3 h-3 animate-spin" />
              <span>Running: {formatToolName(streamingState.currentTool.tool_name)}</span>
            </div>
          )}

          {/* Completed tools */}
          {streamingState.completedTools.length > 0 && (
            <div className="mt-1 space-y-0.5">
              {streamingState.completedTools.slice(-3).map((tool, idx) => (
                <div key={idx} className="flex items-center space-x-1.5 text-green-600">
                  <CheckCircle className="w-3 h-3" />
                  <span>{formatToolName(tool.tool_name)}</span>
                  {tool.execution_time_ms && (
                    <span className="text-gray-400">({tool.execution_time_ms}ms)</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

function formatToolName(name: string): string {
  return name.split('_').map(word =>
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ');
}

// ============================================================================
// ConnectionStatusBadge Component
// ============================================================================

interface ConnectionStatusBadgeProps {
  status: ConnectionStatus;
  onReconnect: () => void;
}

const ConnectionStatusBadge: React.FC<ConnectionStatusBadgeProps> = ({ status, onReconnect }) => {
  if (status === 'connected') return null;

  const getStatusConfig = () => {
    switch (status) {
      case 'connecting':
        return { color: 'bg-yellow-100 text-yellow-700', label: 'Connecting...' };
      case 'disconnected':
        return { color: 'bg-gray-100 text-gray-700', label: 'Disconnected' };
      case 'error':
        return { color: 'bg-red-100 text-red-700', label: 'Connection error' };
      default:
        return { color: 'bg-gray-100 text-gray-700', label: status };
    }
  };

  const config = getStatusConfig();

  return (
    <div className={`px-2 py-1 rounded text-xs flex items-center space-x-2 ${config.color}`}>
      <span>{config.label}</span>
      {(status === 'disconnected' || status === 'error') && (
        <button
          onClick={onReconnect}
          className="underline hover:no-underline"
        >
          Reconnect
        </button>
      )}
    </div>
  );
};

// ============================================================================
// AgentChatWindow Component
// ============================================================================

// Default and constraint values for widget size
const DEFAULT_WIDTH = 384; // w-96 = 24rem = 384px
const DEFAULT_HEIGHT = 500;
const MIN_WIDTH = 320;
const MAX_WIDTH = 800;
const MIN_HEIGHT = 400;
const MAX_HEIGHT = 900;

// Load saved size from localStorage
const loadSavedSize = () => {
  try {
    const saved = localStorage.getItem('agent_widget_size');
    if (saved) {
      const { width, height } = JSON.parse(saved);
      return {
        width: Math.min(Math.max(width, MIN_WIDTH), MAX_WIDTH),
        height: Math.min(Math.max(height, MIN_HEIGHT), MAX_HEIGHT),
      };
    }
  } catch (e) {
    // Ignore parse errors
  }
  return { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT };
};

// Save size to localStorage
const saveSizeToStorage = (width: number, height: number) => {
  try {
    localStorage.setItem('agent_widget_size', JSON.stringify({ width, height }));
  } catch (e) {
    // Ignore storage errors
  }
};

interface AgentChatWindowProps {
  isMinimized: boolean;
  onMinimize: () => void;
  onClose: () => void;
}

const AgentChatWindow: React.FC<AgentChatWindowProps> = ({
  isMinimized,
  onMinimize,
  onClose,
}) => {
  // Resize state
  const [size, setSize] = useState(loadSavedSize);
  const [isResizing, setIsResizing] = useState(false);
  const [resizeDirection, setResizeDirection] = useState<string | null>(null);
  const resizeStartRef = useRef({ x: 0, y: 0, width: 0, height: 0 });

  // Handle resize start
  const handleResizeStart = useCallback((e: React.MouseEvent, direction: string) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResizing(true);
    setResizeDirection(direction);
    resizeStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      width: size.width,
      height: size.height,
    };
  }, [size]);

  // Handle resize move
  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      const deltaX = e.clientX - resizeStartRef.current.x;
      const deltaY = e.clientY - resizeStartRef.current.y;

      let newWidth = resizeStartRef.current.width;
      let newHeight = resizeStartRef.current.height;

      if (resizeDirection?.includes('w')) {
        // Resizing from left edge (width increases when moving left)
        newWidth = Math.min(Math.max(resizeStartRef.current.width - deltaX, MIN_WIDTH), MAX_WIDTH);
      }
      if (resizeDirection?.includes('e')) {
        // Resizing from right edge
        newWidth = Math.min(Math.max(resizeStartRef.current.width + deltaX, MIN_WIDTH), MAX_WIDTH);
      }
      if (resizeDirection?.includes('n')) {
        // Resizing from top edge (height increases when moving up)
        newHeight = Math.min(Math.max(resizeStartRef.current.height - deltaY, MIN_HEIGHT), MAX_HEIGHT);
      }
      if (resizeDirection?.includes('s')) {
        // Resizing from bottom edge
        newHeight = Math.min(Math.max(resizeStartRef.current.height + deltaY, MIN_HEIGHT), MAX_HEIGHT);
      }

      setSize({ width: newWidth, height: newHeight });
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      setResizeDirection(null);
      saveSizeToStorage(size.width, size.height);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, resizeDirection, size]);

  // Save size when it changes (debounced via mouseup)
  const {
    messages,
    isLoading,
    connectionStatus,
    streamingState,
    sendMessage,
    pendingUpload,
    uploadContext,
    handleFileUpload,
    clearPendingUpload,
    clearMessages,
    confirmDelete,
    pendingDelete,
    clearPendingDelete,
    pendingTemplateUpload,
    templateUploadContext,
    handleTemplateUpload,
    clearPendingTemplateUpload,
    activeTemplateJob,
    connect,
    // Conversation memory
    conversationId,
    conversationHistory,
    isLoadingConversation,
    startNewConversation,
    loadConversation,
    refreshConversationHistory,
    // Multi-agent support
    currentAgent,
    injectedMemoriesCount,
    lastRoutingReason,
  } = useAgentWebSocket();

  const [input, setInput] = useState('');
  const [showHistory, setShowHistory] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on open
  useEffect(() => {
    if (!isMinimized) {
      inputRef.current?.focus();
    }
  }, [isMinimized]);

  const handleSend = () => {
    if (!input.trim() || isLoading) return;
    const message = input;
    setInput('');
    sendMessage(message);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (isMinimized) {
    return (
      <div
        className="fixed bottom-24 right-6 bg-white rounded-lg shadow-lg px-4 py-2 cursor-pointer hover:bg-gray-50 z-40"
        onClick={onMinimize}
      >
        <div className="flex items-center space-x-2">
          <Bot className="w-4 h-4 text-primary-600" />
          <span className="text-sm font-medium">Document Assistant</span>
        </div>
      </div>
    );
  }

  const handleSelectConversation = async (id: string) => {
    await loadConversation(id);
    setShowHistory(false);
  };

  const handleStartNew = async () => {
    await startNewConversation();
    setShowHistory(false);
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (days === 1) {
      return 'Yesterday';
    } else if (days < 7) {
      return date.toLocaleDateString([], { weekday: 'short' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };

  return (
    <div
      className={`fixed bottom-24 right-6 bg-white rounded-lg shadow-2xl flex flex-col z-40 border ${isResizing ? 'select-none' : ''}`}
      style={{ width: size.width, height: size.height }}
    >
      {/* Resize handles */}
      {/* Top edge */}
      <div
        className="absolute top-0 left-4 right-4 h-1.5 cursor-n-resize hover:bg-primary-300/50 z-50"
        onMouseDown={(e) => handleResizeStart(e, 'n')}
      />
      {/* Left edge */}
      <div
        className="absolute left-0 top-4 bottom-4 w-1.5 cursor-w-resize hover:bg-primary-300/50 z-50"
        onMouseDown={(e) => handleResizeStart(e, 'w')}
      />
      {/* Top-left corner with grip indicator */}
      <div
        className="absolute top-0 left-0 w-6 h-6 cursor-nw-resize z-50 flex items-center justify-center group"
        onMouseDown={(e) => handleResizeStart(e, 'nw')}
      >
        <div className="w-4 h-4 rounded-tl-lg bg-gray-200 group-hover:bg-primary-300 flex items-center justify-center transition-colors">
          <GripVertical className="w-3 h-3 text-gray-400 group-hover:text-primary-600 -rotate-45" />
        </div>
      </div>

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b bg-primary-600 text-white rounded-t-lg">
        <div className="flex items-center space-x-2">
          <Bot className="w-5 h-5" />
          <div className="flex flex-col">
            <span className="font-medium">
              {currentAgent?.displayName || 'Document Assistant'}
            </span>
            {/* Agent indicator with memory info */}
            <div className="flex items-center space-x-2 text-xs text-primary-200">
              {currentAgent && (
                <span
                  className="flex items-center space-x-1"
                  title={lastRoutingReason || 'Active agent'}
                >
                  <Users className="w-3 h-3" />
                  <span>{currentAgent.name}</span>
                </span>
              )}
              {injectedMemoriesCount > 0 && (
                <span
                  className="flex items-center space-x-1"
                  title={`Using ${injectedMemoriesCount} memories from previous conversations`}
                >
                  <Brain className="w-3 h-3" />
                  <span>{injectedMemoriesCount}</span>
                </span>
              )}
            </div>
          </div>
          {connectionStatus === 'connected' && (
            <span className="w-2 h-2 bg-green-400 rounded-full" title="Connected" />
          )}
        </div>
        <div className="flex items-center space-x-1">
          <button
            onClick={handleStartNew}
            className="p-1 hover:bg-primary-700 rounded"
            title="New conversation"
            disabled={isLoadingConversation}
          >
            <Plus className="w-4 h-4" />
          </button>
          <button
            onClick={() => {
              refreshConversationHistory();
              setShowHistory(!showHistory);
            }}
            className={`p-1 hover:bg-primary-700 rounded ${showHistory ? 'bg-primary-700' : ''}`}
            title="Conversation history"
          >
            <History className="w-4 h-4" />
          </button>
          <button
            onClick={onMinimize}
            className="p-1 hover:bg-primary-700 rounded"
            title="Minimize"
          >
            <Minus className="w-4 h-4" />
          </button>
          <button
            onClick={onClose}
            className="p-1 hover:bg-primary-700 rounded"
            title="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Conversation history dropdown */}
      {showHistory && (
        <div className="absolute top-14 right-0 left-0 bg-white border-b shadow-lg max-h-64 overflow-y-auto z-50">
          <div className="p-2">
            <div className="text-xs font-medium text-gray-500 mb-2 px-2">Recent Conversations</div>
            {isLoadingConversation ? (
              <div className="flex justify-center py-4">
                <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
              </div>
            ) : conversationHistory.length === 0 ? (
              <div className="text-sm text-gray-400 text-center py-4">No conversations yet</div>
            ) : (
              <div className="space-y-1">
                {conversationHistory.map((conv) => (
                  <button
                    key={conv.id}
                    onClick={() => handleSelectConversation(conv.id)}
                    className={`w-full text-left px-2 py-2 rounded hover:bg-gray-100 transition-colors ${
                      conv.id === conversationId ? 'bg-primary-50 border-l-2 border-primary-500' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2 flex-1 min-w-0">
                        <MessageSquare className="w-4 h-4 text-gray-400 flex-shrink-0" />
                        <span className="text-sm truncate">
                          {conv.title || `Conversation ${conv.message_count} messages`}
                        </span>
                      </div>
                      <span className="text-xs text-gray-400 flex-shrink-0 ml-2">
                        {formatDate(conv.last_message_at)}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2 mt-0.5 ml-6 text-xs text-gray-400">
                      <span>{conv.message_count} msgs</span>
                      {conv.tool_calls_count > 0 && (
                        <span>• {conv.tool_calls_count} tools</span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Connection status */}
      {connectionStatus !== 'connected' && (
        <div className="px-3 py-2 border-b">
          <ConnectionStatusBadge status={connectionStatus} onReconnect={connect} />
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.map((msg) => (
          <AgentMessageComponent key={msg.id} message={msg} />
        ))}

        {/* Streaming indicator */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-[85%]">
              <StreamingIndicator streamingState={streamingState} />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* File upload area */}
      {pendingUpload && (
        <FileUploadArea
          onUpload={handleFileUpload}
          onCancel={clearPendingUpload}
          isLoading={isLoading}
          suggestedTitle={uploadContext?.suggestedTitle}
          suggestedTags={uploadContext?.suggestedTags}
        />
      )}

      {/* Delete confirmation */}
      {pendingDelete && (
        <DeleteConfirmation
          documentTitle={pendingDelete.title}
          onConfirm={() => confirmDelete(pendingDelete.documentId)}
          onCancel={clearPendingDelete}
          isLoading={isLoading}
        />
      )}

      {/* Template upload area */}
      {pendingTemplateUpload && (
        <TemplateUploadArea
          onUpload={handleTemplateUpload}
          onCancel={clearPendingTemplateUpload}
          isLoading={isLoading}
          sourceDocuments={templateUploadContext?.sourceDocuments}
        />
      )}

      {/* Active template job progress */}
      {activeTemplateJob && !pendingTemplateUpload && (
        <TemplateJobProgressComponent job={activeTemplateJob} />
      )}

      {/* Input area */}
      <div className="p-3 border-t">
        <div className="flex space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about documents..."
            disabled={isLoading || pendingUpload || pendingDelete !== null || pendingTemplateUpload}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:bg-gray-100"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim() || pendingUpload || pendingDelete !== null || pendingTemplateUpload}
            className="px-3 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Main AgentWidget Component
// ============================================================================

interface AgentWidgetProps {
  defaultOpen?: boolean;
}

const AgentWidget: React.FC<AgentWidgetProps> = ({ defaultOpen = false }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const [isMinimized, setIsMinimized] = useState(false);

  // Keyboard shortcut to toggle widget (Ctrl+Shift+A)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'A') {
        e.preventDefault();
        setIsOpen(prev => !prev);
        setIsMinimized(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <>
      {/* Toggle button - always visible */}
      <AgentToggleButton
        isOpen={isOpen}
        onClick={() => {
          setIsOpen(!isOpen);
          setIsMinimized(false);
        }}
      />

      {/* Chat window - conditionally rendered */}
      {isOpen && (
        <AgentChatWindow
          isMinimized={isMinimized}
          onMinimize={() => setIsMinimized(!isMinimized)}
          onClose={() => setIsOpen(false)}
        />
      )}
    </>
  );
};

export default AgentWidget;
