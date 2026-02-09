/**
 * Documents management page
 */

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from 'react-query';
// Use built distribution to avoid ESM fully specified resolution issues
import videojs from 'video.js';
import 'video.js/dist/video-js.css';
import {
  Upload,
  Search,
  Download,
  Trash2,
  RefreshCw,
  FileText,
  CheckCircle,
  XCircle,
  Clock,
  Eye,
  MoreVertical,
  Plus,
  X,
  Video,
  FileVideo,
  BookOpen,
  Loader2,
  UserCircle2,
  MessageSquare,
  Edit,
  Copy,
  Link2,
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

import { apiClient, GitRepoRequestPayload, ArxivRequestPayload } from '../services/api';
import type {
  Document as KnowledgeDocument,
  DocumentSource,
  ActiveGitSource,
  GitBranch,
  GitCompareJob,
  Persona,
  DocumentPersonaDetection,
} from '../types';
import { useAuth } from '../contexts/AuthContext';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ConfirmationModal from '../components/common/ConfirmationModal';
import ProgressBar from '../components/common/ProgressBar';
import { DocxEditorModal } from '../components/docx';
import { formatFileSize } from '../utils/formatting';
import toast from 'react-hot-toast';

// Helpers for naming
const getBaseName = (document: KnowledgeDocument): string => {
  const original = document.extra_metadata?.original_filename || document.title || '';
  return original.replace(/\.[^.]+$/, '');
};

const getDisplayTitle = (document: KnowledgeDocument): string => {
  // Display without extension while converting/after upload
  return getBaseName(document) || document.title || '';
};

const getDownloadFilename = (document: KnowledgeDocument): string => {
  const base = getBaseName(document) || `document_${document.id}`;
  // Choose extension based on current file_type or original
  if (document.file_type === 'video/mp4') return `${base}.mp4`;
  const original = document.extra_metadata?.original_filename || document.title || '';
  const m = original.match(/\.([^.]+)$/);
  return m ? `${base}.${m[1]}` : base;
};

type GitRepoFormState = {
  provider: 'github' | 'gitlab';
  name: string;
  token: string;
  repos: string;
  gitlabUrl: string;
  includeFiles: boolean;
  includeIssues: boolean;
  includePRs: boolean;
  includeWiki: boolean;
  incrementalFiles: boolean;
  useGitignore: boolean;
  autoSync: boolean;
};

const initialGitRepoForm: GitRepoFormState = {
  provider: 'github',
  name: '',
  token: '',
  repos: '',
  gitlabUrl: '',
  includeFiles: true,
  includeIssues: true,
  includePRs: false,
  includeWiki: false,
  incrementalFiles: true,
  useGitignore: true,
  autoSync: true,
};

type ArxivFormState = {
  name: string;
  queries: string;
  categories: string;
  paperIds: string;
  maxResults: number;
  sortBy: 'relevance' | 'lastUpdatedDate' | 'submittedDate';
  sortOrder: 'ascending' | 'descending';
  autoSync: boolean;
};

const initialArxivForm: ArxivFormState = {
  name: '',
  queries: '',
  categories: '',
  paperIds: '',
  maxResults: 50,
  sortBy: 'submittedDate',
  sortOrder: 'descending',
  autoSync: true,
};

// Type for location state passed from navigation
interface LocationState {
  highlightChunkId?: string;
  selectedSourceId?: string;
  selectedSourceTab?: 'documents' | 'videos' | 'repos' | 'arxiv';
  selectedDocumentId?: string;
  initialSeekSeconds?: number;
  openDocId?: string;
}

const DocumentsPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const canManagePersona = user?.role === 'admin';
  const canRequestPersonaEdit = !!user && !canManagePersona;
  const openPersonaManager = useCallback(
    (personaId: string) => {
      if (!personaId) return;
      navigate(`/admin?tab=personas&personaId=${personaId}`);
    },
    [navigate]
  );
  const [personaEditRequest, setPersonaEditRequest] = useState<{ persona: Persona; document?: KnowledgeDocument | null } | null>(null);
  
  const [activeTab, setActiveTab] = useState<'documents' | 'videos' | 'repos' | 'arxiv'>('documents');
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearchQuery, setDebouncedSearchQuery] = useState('');
  const [selectedSource, setSelectedSource] = useState<string>('');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showIngestUrlModal, setShowIngestUrlModal] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<KnowledgeDocument | null>(null);
  const [initialSeekSeconds, setInitialSeekSeconds] = useState<number | null>(null);
  const [previousTranscriptId, setPreviousTranscriptId] = useState<string | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState<string | null>(null);
  const [docxEditorOpen, setDocxEditorOpen] = useState(false);
  const [documentToEdit, setDocumentToEdit] = useState<KnowledgeDocument | null>(null);
  const [transcriptionProgress, setTranscriptionProgress] = useState<Record<string, { progress: number; message: string; stage: string; remaining_seconds?: number; remaining_formatted?: string; duration?: number; elapsed?: number }>>({});
  const transcriptionWebSockets = React.useRef<Record<string, WebSocket>>({});
  const [summarizationProgress, setSummarizationProgress] = useState<Record<string, { progress: number; stage?: string }>>({});
  const summarizationWebSockets = React.useRef<Record<string, WebSocket>>({});
  const [uploadProgress, setUploadProgress] = useState<Record<string, { progress: number; status: string }>>({});
  const [uploadStatus, setUploadStatus] = useState<Record<string, string>>({});
  const [streamingSegments, setStreamingSegments] = useState<Record<string, Array<{ start: number; text: string; speaker?: string }>>>({});
  const [gitRepoForm, setGitRepoForm] = useState<GitRepoFormState>(initialGitRepoForm);
  const [arxivForm, setArxivForm] = useState<ArxivFormState>(initialArxivForm);
  const [pendingSourceIds, setPendingSourceIds] = useState<string[]>([]);
  const ingestionWebSockets = React.useRef<Record<string, WebSocket>>({});
  const [sourceProgress, setSourceProgress] = useState<Record<string, { progress?: number; current?: number; total?: number; status?: string; stage?: string; remaining_formatted?: string; canceled?: boolean }>>({});
  const [expandedRepos, setExpandedRepos] = useState<Record<string, boolean>>({});
  const [highlightIngestionSourceId, setHighlightIngestionSourceId] = useState<string>('');
  const ingestionCardRefs = React.useRef<Record<string, HTMLDivElement | null>>({});
  const [highlightRepoGroupSourceId, setHighlightRepoGroupSourceId] = useState<string>('');
  const repoGroupRefs = React.useRef<Record<string, HTMLDivElement | null>>({});
  const [compareSourceId, setCompareSourceId] = useState<string>('');
  const [compareRepository, setCompareRepository] = useState<string>('');
  const [compareBaseBranch, setCompareBaseBranch] = useState<string>('');
  const [compareTargetBranch, setCompareTargetBranch] = useState<string>('');
  const [branchExplain, setBranchExplain] = useState<boolean>(true);
  const [compareIncludeFiles, setCompareIncludeFiles] = useState<boolean>(true);
  const [branchList, setBranchList] = useState<GitBranch[]>([]);
  const [loadingBranches, setLoadingBranches] = useState<boolean>(false);
  const summarizeMutation = useMutation(
    async (documentId: string) => {
      return apiClient.summarizeDocument(documentId, false);
    },
    {
      onMutate: async (documentId: string) => {
        // optimistically mark as summarizing to trigger WS connection
        setDocSumStatus(prev => ({ ...prev, [documentId]: { ...(prev[documentId] || {}), is_summarizing: true } }));
      },
      onSuccess: (_data, documentId) => {
        toast.success('Summarization started');
        // keep optimistic flag until status comes from WS/backend
        queryClient.invalidateQueries('documents');
      },
      onError: (e: any, documentId) => {
        toast.error(e?.response?.data?.detail || e?.message || 'Failed to start summarization');
        // revert optimistic flag on error
        setDocSumStatus(prev => {
          const next = { ...prev } as any;
          delete next[documentId as string];
          return next;
        });
      },
    }
  );

  useEffect(() => {
    const state = (location.state as LocationState) || {};
    const sid = String(state.selectedSourceId || '').trim();
    if (sid) {
      setSelectedSource(sid);
      const tab = state.selectedSourceTab;
      if (tab === 'documents' || tab === 'videos' || tab === 'repos' || tab === 'arxiv') setActiveTab(tab);
      else setActiveTab('documents');
      if (tab === 'repos') {
        setHighlightIngestionSourceId(sid);
      }
    }
  }, [location.state]);

  useEffect(() => {
    if (activeTab !== 'repos') return;
    const sid = String(highlightIngestionSourceId || '').trim();
    if (!sid) return;
    const t = window.setTimeout(() => {
      const el = ingestionCardRefs.current[sid];
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 350);
    return () => window.clearTimeout(t);
  }, [activeTab, highlightIngestionSourceId]);

  useEffect(() => {
    if (activeTab !== 'documents') return;
    const sid = String(highlightRepoGroupSourceId || '').trim();
    if (!sid) return;
    const t = window.setTimeout(() => {
      const el = repoGroupRefs.current[sid];
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 350);
    return () => window.clearTimeout(t);
  }, [activeTab, highlightRepoGroupSourceId]);
  const personaEditRequestMutation = useMutation(
    ({ personaId, message, documentId }: { personaId: string; message: string; documentId?: string }) =>
      apiClient.requestPersonaEdit(personaId, {
        message,
        document_id: documentId || undefined,
      }),
    {
      onSuccess: () => {
        toast.success('Persona edit request sent to admins');
        setPersonaEditRequest(null);
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to submit request';
        toast.error(message);
      },
    }
  );
  // Live transcript preview toggle per document
  const [livePreviewOpen, setLivePreviewOpen] = useState<Record<string, boolean>>({});
  // Live document status overrides from WebSocket to avoid refetch loops
  const [docStatus, setDocStatus] = useState<Record<string, { is_transcoding?: boolean; is_transcribing?: boolean; is_transcribed?: boolean; failed?: boolean; error?: string }>>({});
  const [docSumStatus, setDocSumStatus] = useState<Record<string, { is_summarizing?: boolean; failed?: boolean; error?: string }>>({});
  const [ownerPersonaFilter, setOwnerPersonaFilter] = useState<string>('');
  const [speakerPersonaFilter, setSpeakerPersonaFilter] = useState<string>('');

  const getDocFlags = (doc: KnowledgeDocument) => {
    const override = docStatus[doc.id] || {};
    const sumOverride = docSumStatus[doc.id] || {};
    const isTranscoding = override.is_transcoding ?? (doc.extra_metadata?.is_transcoding === true);
    const isTranscribing = override.is_transcribing ?? (doc.extra_metadata?.is_transcribing === true);
    const isTranscribed = override.is_transcribed ?? (doc.extra_metadata?.is_transcribed === true);
    const isSummarizing = sumOverride.is_summarizing ?? (doc.extra_metadata?.is_summarizing === true);
    return { isTranscoding, isTranscribing, isTranscribed, isSummarizing };
  };

  // Helper function to check if document is video/audio
  const isVideoAudio = (doc: KnowledgeDocument): boolean => {
    if (doc.file_type) {
      return doc.file_type.startsWith('video/') || doc.file_type.startsWith('audio/');
    }
    const ext = doc.title?.toLowerCase().split('.').pop() || '';
    return ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
      .some(e => ext === e.substring(1));
  };

  // If navigation passes a specific document to open, fetch it directly and select it.
  useEffect(() => {
    const state = (location.state as LocationState) || {};
    const docId = String(state.selectedDocumentId || '').trim();
    if (!docId) return;

    let cancelled = false;
    apiClient
      .getDocument(docId)
      .then((doc) => {
        if (cancelled) return;
        setSelectedDocument(doc as any);
        setActiveTab(isVideoAudio(doc as any) ? 'videos' : 'documents');
        if (typeof state.initialSeekSeconds === 'number') {
          setInitialSeekSeconds(state.initialSeekSeconds);
        }
        if (state.highlightChunkId) {
          setTimeout(() => {
            const el = document.getElementById(`chunk-card-${state.highlightChunkId}`);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }, 350);
        }
      })
      .catch(() => {
        // ignore
      });
    return () => {
      cancelled = true;
    };
  }, [location.state]);

  const { data: personaResponse, isLoading: personasLoading } = useQuery(
    ['personas', 'filters'],
    () => apiClient.listPersonas({ page_size: 100 }),
    { staleTime: 60_000 }
  );
  const personaOptions = useMemo(() => {
    const source = personaResponse?.items || [];
    return [...source].sort((a, b) => a.name.localeCompare(b.name));
  }, [personaResponse?.items]);

  // Debounce document search input to avoid spamming requests on every keystroke
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchQuery(searchQuery.trim());
    }, 350);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Fetch documents
  const { data: allDocuments, isLoading: documentsLoading, refetch: refetchDocuments } = useQuery<KnowledgeDocument[]>(
    ['documents', debouncedSearchQuery, selectedSource, ownerPersonaFilter, speakerPersonaFilter],
    () => apiClient.getDocuments({
      search: debouncedSearchQuery || undefined,
      source_id: selectedSource || undefined,
      limit: 100,
      owner_persona_id: ownerPersonaFilter || undefined,
      persona_id: speakerPersonaFilter || undefined,
      persona_role: speakerPersonaFilter ? 'speaker' : undefined,
    }),
    {
      refetchOnWindowFocus: false,
      // Poll only when something is actively processing; avoid polling while user is searching.
      refetchInterval: (data) => {
        if (debouncedSearchQuery) return false;
        const docs = data || [];
        const hasInFlightWork = docs.some((doc) => {
          const flags = getDocFlags(doc);
          return flags.isTranscribing || flags.isTranscoding || flags.isSummarizing;
        });
        return hasInFlightWork ? 10_000 : false;
      },
    }
  );

  // Filter documents based on active tab
  const documents: KnowledgeDocument[] = allDocuments?.filter(doc => {
    if (activeTab === 'videos') {
      return isVideoAudio(doc);
    } else {
      return !isVideoAudio(doc);
    }
  }) || [];

  const handlePersonaFilter = (personaId: string, role: 'owner' | 'speaker' = 'owner') => {
    if (!personaId) return;
    if (role === 'owner') {
      setOwnerPersonaFilter(personaId);
    } else {
      setSpeakerPersonaFilter(personaId);
    }
    setActiveTab(prev => (prev === 'repos' || prev === 'arxiv' ? 'documents' : prev));
    setSelectedDocument(null);
    setPreviousTranscriptId(null);
    setInitialSeekSeconds(null);
  };

  const handlePersonaEditRequest = (persona: Persona, document?: KnowledgeDocument) => {
    if (!persona) return;
    setPersonaEditRequest({ persona, document });
  };


  // Connect WebSocket for transcription progress on video/audio documents
  useEffect(() => {
    // Connect to WebSocket for each video/audio document that is being transcribed
    documents.forEach((doc) => {
      const flags = getDocFlags(doc);
      const wantsProgress = flags.isTranscribing || flags.isTranscoding;
      if (isVideoAudio(doc) && wantsProgress && !transcriptionWebSockets.current[doc.id]) {
        try {
          const ws = apiClient.createTranscriptionProgressWebSocket(doc.id);
          
          ws.onopen = () => {
            console.log(`Transcription progress WebSocket connected for document ${doc.id}`);
          };
          
          ws.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              
              if (data.type === 'transcription_progress') {
                const progress = data.progress || {};
                setTranscriptionProgress(prev => ({
                  ...prev,
                  [doc.id]: {
                    progress: progress.progress || 0,
                    message: progress.message || 'Processing...',
                    stage: progress.stage || 'unknown',
                    remaining_seconds: progress.remaining_seconds,
                    remaining_formatted: progress.remaining_formatted,
                    duration: progress.duration,
                    elapsed: progress.elapsed,
                  }
                }));
              } else if (data.type === 'document_status') {
                // Update local override for flags; avoid refetching to prevent WS churn
                const status = data.status || {};
                setDocStatus(prev => ({
                  ...prev,
                  [doc.id]: { ...(prev[doc.id] || {}), ...status }
                }));
              } else if (data.type === 'transcription_segment') {
                const seg = data.segment || {};
                if (typeof seg.start === 'number' && typeof seg.text === 'string') {
                  setStreamingSegments(prev => {
                    const cur = prev[doc.id] || [];
                    const last = cur[cur.length - 1];
                    // Avoid duplicate adjacent same text
                    if (last && last.text === seg.text && (last.speaker || '') === (seg.speaker || '')) {
                      return prev;
                    }
                    const nextItem: any = { start: seg.start, text: seg.text };
                    if (seg.speaker) nextItem.speaker = seg.speaker;
                    return { ...prev, [doc.id]: [...cur, nextItem] };
                  });
                }
              } else if (data.type === 'transcription_complete') {
                // Transcription complete, refetch documents
                queryClient.invalidateQueries('documents');
                // Close WebSocket
                if (transcriptionWebSockets.current[doc.id]) {
                  transcriptionWebSockets.current[doc.id].close();
                  delete transcriptionWebSockets.current[doc.id];
                }
                // Clear progress
                setTranscriptionProgress(prev => {
                  const newProgress = { ...prev };
                  delete newProgress[doc.id];
                  return newProgress;
                });
                // Clear status override
                setDocStatus(prev => {
                  const next = { ...prev } as any;
                  delete next[doc.id];
                  return next;
                });
                setStreamingSegments(prev => {
                  const next = { ...prev } as any;
                  delete next[doc.id];
                  return next;
                });
              } else if (data.type === 'transcription_error') {
                toast.error(`Transcription error: ${data.error || 'Unknown error'}`);
                // Mark failed locally so status shows immediately
                setDocStatus(prev => ({
                  ...prev,
                  [doc.id]: { ...(prev[doc.id] || {}), failed: true, error: data.error || 'Unknown error' }
                }));
                // Close WebSocket
                if (transcriptionWebSockets.current[doc.id]) {
                  transcriptionWebSockets.current[doc.id].close();
                  delete transcriptionWebSockets.current[doc.id];
                }
                // Clear progress
                setTranscriptionProgress(prev => {
                  const newProgress = { ...prev };
                  delete newProgress[doc.id];
                  return newProgress;
                });
                // Clear status override
                setDocStatus(prev => {
                  const next = { ...prev } as any;
                  // keep failed flag until next fetch updates it
                  return next;
                });
                setStreamingSegments(prev => {
                  const next = { ...prev } as any;
                  delete next[doc.id];
                  return next;
                });
              }
            } catch (error) {
              console.error('Error parsing transcription progress message:', error);
            }
          };
          
          ws.onclose = () => {
            console.log(`Transcription progress WebSocket closed for document ${doc.id}`);
            delete transcriptionWebSockets.current[doc.id];
          };
          
          ws.onerror = (error) => {
            console.error(`Transcription progress WebSocket error for document ${doc.id}:`, error);
          };
          
          transcriptionWebSockets.current[doc.id] = ws;
        } catch (error) {
          console.error(`Failed to create transcription progress WebSocket for document ${doc.id}:`, error);
        }
      }
    });
    
    // Cleanup: close WebSockets for documents that are no longer being transcribed
    Object.keys(transcriptionWebSockets.current).forEach((docId) => {
      const doc = documents.find(d => d.id === docId);
      const flags = doc ? getDocFlags(doc) : { isTranscoding: false, isTranscribing: false };
      const wantsProgress = flags.isTranscribing || flags.isTranscoding;
      if (!doc || !isVideoAudio(doc) || !wantsProgress) {
        transcriptionWebSockets.current[docId].close();
        delete transcriptionWebSockets.current[docId];
        setTranscriptionProgress(prev => {
          const newProgress = { ...prev };
          delete newProgress[docId];
          return newProgress;
        });
        setDocStatus(prev => {
          const next = { ...prev } as any;
          delete next[docId];
          return next;
        });
        setStreamingSegments(prev => {
          const next = { ...prev } as any;
          delete next[docId];
          return next;
        });
      }
    });
    
    // Cleanup on unmount
    return () => {
      Object.values(transcriptionWebSockets.current).forEach(ws => ws.close());
      transcriptionWebSockets.current = {};
    };
  }, [documents, queryClient]);

  // Connect WebSocket for summarization progress on documents
  useEffect(() => {
    documents.forEach((doc) => {
      const flags = getDocFlags(doc);
      const wantsProgress = flags.isSummarizing === true;
      if (!isVideoAudio(doc) && wantsProgress && !summarizationWebSockets.current[doc.id]) {
        try {
          const ws = apiClient.createSummarizationProgressWebSocket(doc.id);
          ws.onopen = () => {
            console.log(`Summarization progress WebSocket connected for document ${doc.id}`);
          };
          ws.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              if (data.type === 'summarization_progress') {
                const progress = data.progress || {};
                setSummarizationProgress(prev => ({
                  ...prev,
                  [doc.id]: {
                    progress: progress.progress || 0,
                    stage: progress.stage,
                  }
                }));
              } else if (data.type === 'summarization_status') {
                const status = data.status || {};
                setDocSumStatus(prev => ({
                  ...prev,
                  [doc.id]: { ...(prev[doc.id] || {}), ...status }
                }));
              } else if (data.type === 'summarization_complete') {
                // Clear status/progress, close WS, refetch
                if (summarizationWebSockets.current[doc.id]) {
                  summarizationWebSockets.current[doc.id].close();
                  delete summarizationWebSockets.current[doc.id];
                }
                setSummarizationProgress(prev => {
                  const next = { ...prev } as any;
                  delete next[doc.id];
                  return next;
                });
                setDocSumStatus(prev => {
                  const next = { ...prev } as any;
                  delete next[doc.id];
                  return next;
                });
                queryClient.invalidateQueries('documents');
              } else if (data.type === 'summarization_error') {
                toast.error(`Summarization error: ${data.error || 'Unknown error'}`);
                if (summarizationWebSockets.current[doc.id]) {
                  summarizationWebSockets.current[doc.id].close();
                  delete summarizationWebSockets.current[doc.id];
                }
                setSummarizationProgress(prev => {
                  const next = { ...prev } as any;
                  delete next[doc.id];
                  return next;
                });
                setDocSumStatus(prev => ({
                  ...prev,
                  [doc.id]: { ...(prev[doc.id] || {}), failed: true, error: data.error || 'Unknown error', is_summarizing: false }
                }));
              }
            } catch (err) {
              console.error('Error parsing summarization progress message:', err);
            }
          };
          ws.onclose = () => {
            console.log(`Summarization progress WebSocket closed for document ${doc.id}`);
            delete summarizationWebSockets.current[doc.id];
          };
          ws.onerror = (error) => {
            console.error(`Summarization progress WebSocket error for document ${doc.id}:`, error);
          };
          summarizationWebSockets.current[doc.id] = ws;
        } catch (error) {
          console.error(`Failed to create summarization progress WebSocket for document ${doc.id}:`, error);
        }
      }
    });

    // Cleanup: close websockets for docs no longer summarizing
    Object.keys(summarizationWebSockets.current).forEach((docId) => {
      const doc = documents.find(d => d.id === docId);
      const flags = doc ? getDocFlags(doc) : { isSummarizing: false } as any;
      if (!doc || isVideoAudio(doc) || !flags.isSummarizing) {
        try { summarizationWebSockets.current[docId].close(); } catch {}
        delete summarizationWebSockets.current[docId];
        setSummarizationProgress(prev => {
          const next = { ...prev } as any;
          delete next[docId];
          return next;
        });
        setDocSumStatus(prev => {
          const next = { ...prev } as any;
          delete next[docId];
          return next;
        });
      }
    });

    return () => {
      Object.values(summarizationWebSockets.current).forEach(ws => ws.close());
      summarizationWebSockets.current = {};
    };
  }, [documents, queryClient, docSumStatus]);

  // Open document modal based on navigation state
  useEffect(() => {
    const st = location?.state as { openDocId?: string; highlightChunkId?: string } | undefined;
    const openDocId = st?.openDocId;
    if (!openDocId || !allDocuments || !allDocuments.length) return;
    const doc = allDocuments.find(d => d.id === openDocId);
    if (doc) {
      setSelectedDocument({ ...doc, chunks: doc.chunks });
      // Scroll to chunk after modal mounts
      setTimeout(() => {
        const el = document.getElementById(`chunk-card-${st?.highlightChunkId}`);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 300);
    }
  }, [location?.state, allDocuments]);

  // Fetch document sources
  const { data: sources } = useQuery<DocumentSource[]>(
    'documentSources',
    () => apiClient.getDocumentSources(),
    {
      refetchOnWindowFocus: false,
    }
  );
  const { data: activeGitStatuses, refetch: refetchActiveGitStatuses } = useQuery<ActiveGitSource[]>(
    'activeGitSources',
    () => apiClient.getActiveGitSources(),
    {
      refetchOnWindowFocus: false,
      refetchInterval: 10000,
    }
  );
  const { data: gitCompareJobs = [], refetch: refetchGitCompareJobs } = useQuery<GitCompareJob[]>(
    'gitCompareJobs',
    () => apiClient.getGitComparisonJobs(),
    {
      refetchOnWindowFocus: false,
      refetchInterval: 10000,
    }
  );

  const repoSourceMap = useMemo(() => {
    const map: Record<string, DocumentSource> = {};
    (sources || []).forEach((src) => {
      map[src.id] = src;
    });
    return map;
  }, [sources]);
  const gitSourcesForUser = useMemo(() => {
    return (sources || []).filter((source) => {
      const isGit = source.source_type === 'github' || source.source_type === 'gitlab';
      if (!isGit) return false;
      if (user?.role === 'admin') return true;
      const cfg = (source.config || {}) as any;
      const requestedBy = cfg?.requested_by || cfg?.requestedBy;
      return requestedBy && requestedBy === user?.username;
    });
  }, [sources, user]);
  const repoOptionsBySource = useMemo(() => {
    const map: Record<string, string[]> = {};
    gitSourcesForUser.forEach((source) => {
      const cfg = (source.config || {}) as any;
      if (source.source_type === 'github') {
        const repos = cfg?.repos || [];
        map[source.id] = repos.map((entry: any) => {
          if (typeof entry === 'string') return entry;
          if (entry?.owner && entry?.repo) {
            return `${entry.owner}/${entry.repo}`;
          }
          return '';
        }).filter(Boolean);
      } else {
        const projects = cfg?.projects || [];
        map[source.id] = projects
          .map((project: any) => String(project?.id || project?.path || project?.name || ''))
          .filter(Boolean);
      }
    });
    return map;
  }, [gitSourcesForUser]);
  const serverPendingSourceIds = useMemo(
    () => (activeGitStatuses || []).filter((entry) => entry.pending).map((entry) => entry.source.id),
    [activeGitStatuses]
  );
  const combinedPendingSourceIds = useMemo(
    () => Array.from(new Set([...pendingSourceIds, ...serverPendingSourceIds])),
    [pendingSourceIds, serverPendingSourceIds]
  );

  const shouldGroupRepoDocs = activeTab === 'documents';
  const isRepoDocument = (doc: KnowledgeDocument) => {
    const type = doc.source?.source_type;
    return type === 'github' || type === 'gitlab';
  };

  const repoDocuments = useMemo(
    () => (shouldGroupRepoDocs ? documents.filter(isRepoDocument) : []),
    [documents, shouldGroupRepoDocs]
  );

  const regularDocuments = useMemo(
    () => (shouldGroupRepoDocs ? documents.filter((doc) => !isRepoDocument(doc)) : documents),
    [documents, shouldGroupRepoDocs]
  );

  const repoDocumentsBySource = useMemo(() => {
    if (!shouldGroupRepoDocs) {
      return {} as Record<string, KnowledgeDocument[]>;
    }
    return repoDocuments.reduce((acc, doc) => {
      const srcId = doc.source?.id;
      if (!srcId) return acc;
      if (!acc[srcId]) acc[srcId] = [];
      acc[srcId].push(doc);
      return acc;
    }, {} as Record<string, KnowledgeDocument[]>);
  }, [repoDocuments, shouldGroupRepoDocs]);

  const repoGroupEntries = Object.entries(repoDocumentsBySource);

  useEffect(() => {
    return () => {
      Object.values(ingestionWebSockets.current).forEach((ws) => {
        try {
          ws.close();
        } catch {}
      });
      ingestionWebSockets.current = {};
    };
  }, []);

  useEffect(() => {
    if (!sources || !user) return;

    const canTrackSource = (source: DocumentSource) => {
      const isGit = source.source_type === 'github' || source.source_type === 'gitlab';
      if (!isGit) return false;
      if (user.role === 'admin') return true;
      const cfg = (source.config || {}) as any;
      const requestedBy = cfg?.requested_by || cfg?.requestedBy;
      return requestedBy && requestedBy === user.username;
    };

    Object.keys(ingestionWebSockets.current).forEach((sourceId) => {
      const stillValid = sources.some(
        (src) =>
          src.id === sourceId &&
          canTrackSource(src) &&
          ((src as any)?.is_syncing === true || combinedPendingSourceIds.includes(sourceId))
      );
      if (!stillValid) {
        try {
          ingestionWebSockets.current[sourceId].close();
        } catch {}
        delete ingestionWebSockets.current[sourceId];
        setSourceProgress((prev) => {
          if (!(sourceId in prev)) return prev;
          const next = { ...prev } as any;
          delete next[sourceId];
          return next;
        });
        setPendingSourceIds((prev) => prev.filter((id) => id !== sourceId));
      }
    });

    sources.forEach((source) => {
      const pending = combinedPendingSourceIds.includes(source.id);
      if (!canTrackSource(source) || (!pending && (source as any)?.is_syncing !== true)) {
        return;
      }
      if (ingestionWebSockets.current[source.id]) {
        return;
      }
      try {
        const ws = apiClient.createIngestionProgressWebSocket(source.id, { admin: user.role === 'admin' });
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'ingestion_progress') {
              const progress = data.progress || {};
              setSourceProgress((prev) => ({
                ...prev,
                [source.id]: {
                  progress: typeof progress.progress === 'number' ? progress.progress : undefined,
                  current: progress.current,
                  total: progress.total,
                  status: progress.status || progress.stage,
                  stage: progress.stage,
                  remaining_formatted: progress.remaining_formatted,
                },
              }));
            } else if (data.type === 'ingestion_complete') {
              setSourceProgress((prev) => {
                const next = { ...prev } as any;
                delete next[source.id];
                return next;
              });
              try {
                ingestionWebSockets.current[source.id]?.close();
              } catch {}
              delete ingestionWebSockets.current[source.id];
              toast.success(`Finished processing ${source.name}`);
              queryClient.invalidateQueries('documentSources');
              queryClient.invalidateQueries('documents');
              setPendingSourceIds((prev) => prev.filter((id) => id !== source.id));
              refetchActiveGitStatuses();
            } else if (data.type === 'ingestion_error') {
              toast.error(`Ingestion error: ${data.error || 'Unknown error'}`);
              setSourceProgress((prev) => {
                const next = { ...prev } as any;
                delete next[source.id];
                return next;
              });
              try {
                ingestionWebSockets.current[source.id]?.close();
              } catch {}
              delete ingestionWebSockets.current[source.id];
              setPendingSourceIds((prev) => prev.filter((id) => id !== source.id));
              refetchActiveGitStatuses();
            } else if (data.type === 'ingestion_status') {
              const status = data.status || {};
              if (status.canceled) {
                toast('Ingestion canceled', { icon: '⏹️' });
                setSourceProgress((prev) => ({
                  ...prev,
                  [source.id]: {
                    ...(prev[source.id] || {}),
                    canceled: true,
                    status: 'Canceled',
                  },
                }));
                setPendingSourceIds((prev) => prev.filter((id) => id !== source.id));
                refetchActiveGitStatuses();
              }
            }
          } catch (err) {
            console.error('Error parsing ingestion progress', err);
          }
        };
        ws.onclose = () => {
          delete ingestionWebSockets.current[source.id];
        };
        ws.onerror = (err) => console.error('Ingestion WS error', err);
        ingestionWebSockets.current[source.id] = ws;
      } catch (e) {
        console.error('Failed to open ingestion WebSocket', e);
      }
    });
  }, [sources, user, queryClient, combinedPendingSourceIds, refetchActiveGitStatuses]);

  useEffect(() => {
    if (!compareSourceId && gitSourcesForUser.length > 0) {
      setCompareSourceId(gitSourcesForUser[0].id);
    }
  }, [gitSourcesForUser, compareSourceId]);

  useEffect(() => {
    if (!compareSourceId) {
      setCompareRepository('');
      return;
    }
    const repos = repoOptionsBySource[compareSourceId] || [];
    setCompareRepository((prev) => (prev && repos.includes(prev) ? prev : (repos[0] || '')));
  }, [compareSourceId, repoOptionsBySource]);

  useEffect(() => {
    setCompareBaseBranch('');
    setCompareTargetBranch('');
    setBranchList([]);
  }, [compareRepository]);

  useEffect(() => {
    if (!compareSourceId || !compareRepository) {
      setBranchList([]);
      return;
    }
    let active = true;
    setLoadingBranches(true);
    apiClient
      .getGitBranches(compareSourceId, compareRepository)
      .then((branches) => {
        if (!active) return;
        setBranchList(branches);
        if (branches.length > 0) {
          if (!compareBaseBranch) {
            setCompareBaseBranch(branches[0].name);
          }
          if (!compareTargetBranch && branches.length > 1) {
            setCompareTargetBranch(branches[1].name);
          }
        }
      })
      .catch((error) => {
        if (!active) return;
        const message = error?.response?.data?.detail || error?.message || 'Failed to load branches';
        toast.error(message);
        setBranchList([]);
      })
      .finally(() => {
        if (active) setLoadingBranches(false);
      });
    return () => {
      active = false;
    };
  }, [compareSourceId, compareRepository]);

  // Delete document mutation
  const deleteDocumentMutation = useMutation(
    (documentId: string) => apiClient.deleteDocument(documentId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('documents');
        toast.success('Document deleted successfully');
      },
      onError: (error: any) => {
        console.error('Delete document error:', error);
        const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to delete document';
        toast.error(errorMessage);
      },
    }
  );

  // Reprocess document mutation
  const reprocessDocumentMutation = useMutation(
    (documentId: string) => apiClient.reprocessDocument(documentId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('documents');
        toast.success('Document reprocessing started');
      },
      onError: () => {
        toast.error('Failed to reprocess document');
      },
    }
  );

  const addGitRepoMutation = useMutation<DocumentSource, any, GitRepoRequestPayload>(
    (payload: GitRepoRequestPayload) => apiClient.requestGitRepository(payload),
    {
      onSuccess: (source) => {
        toast.success('Repository submitted for processing');
        setGitRepoForm(prev => ({ ...initialGitRepoForm, provider: prev.provider }));
        queryClient.invalidateQueries('documentSources');
        if (source?.id) {
          setPendingSourceIds((prev) => (prev.includes(source.id) ? prev : [...prev, source.id]));
        }
        refetchActiveGitStatuses();
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to submit repository';
        toast.error(message);
      },
    }
  );
  const addArxivSourceMutation = useMutation(
    (payload: ArxivRequestPayload) => apiClient.requestArxivSource(payload),
    {
      onSuccess: () => {
        toast.success('ArXiv request submitted');
        setArxivForm(initialArxivForm);
        queryClient.invalidateQueries('documentSources');
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to submit ArXiv request';
        toast.error(message);
      },
    }
  );
  const cancelCompareMutation = useMutation(
    (jobId: string) => apiClient.cancelGitComparisonJob(jobId),
    {
      onSuccess: () => {
        toast.success('Cancellation requested');
        refetchGitCompareJobs();
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to cancel comparison';
        toast.error(message);
      },
    }
  );
  const deleteRepoDocumentsMutation = useMutation(
    (sourceId: string) => apiClient.deleteRepoDocuments(sourceId),
    {
      onSuccess: (data, sourceId) => {
        toast.success(data?.message || 'Deleted repository documents');
        setExpandedRepos((prev) => ({ ...prev, [sourceId]: false }));
        setSourceProgress((prev) => {
          const next = { ...prev } as any;
          delete next[sourceId];
          return next;
        });
        setPendingSourceIds((prev) => prev.filter((id) => id !== sourceId));
        queryClient.invalidateQueries('documents');
        queryClient.invalidateQueries('documentSources');
        refetchActiveGitStatuses();
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to delete documents';
        toast.error(message);
      },
    }
  );
  const deletingRepoSourceId = deleteRepoDocumentsMutation.variables as string | undefined;
  const cancelingCompareJobId = cancelCompareMutation.variables as string | undefined;
  const cancelSourceMutation = useMutation(
    (sourceId: string) => apiClient.cancelUserSource(sourceId),
    {
      onSuccess: (data, sourceId) => {
        toast.success(data?.message || 'Cancellation requested');
        setPendingSourceIds((prev) => prev.filter((id) => id !== sourceId));
        refetchActiveGitStatuses();
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to cancel ingestion';
        toast.error(message);
      },
    }
  );
  const cancelingSourceId = cancelSourceMutation.variables as string | undefined;

  const handleDeleteDocument = (documentId: string) => {
    setDocumentToDelete(documentId);
    setDeleteConfirmOpen(true);
  };

  const confirmDeleteDocument = () => {
    if (documentToDelete) {
      deleteDocumentMutation.mutate(documentToDelete);
      setDeleteConfirmOpen(false);
      setDocumentToDelete(null);
    }
  };

  const handleReprocessDocument = (documentId: string) => {
    reprocessDocumentMutation.mutate(documentId);
  };

  const handleEditDocument = (document: KnowledgeDocument) => {
    setDocumentToEdit(document);
    setDocxEditorOpen(true);
  };

  const isDocxFile = (document: KnowledgeDocument): boolean => {
    const fileType = document.file_type?.toLowerCase() || '';
    const filePath = document.file_path?.toLowerCase() || '';
    return (
      fileType.includes('wordprocessingml') ||
      fileType.includes('msword') ||
      filePath.endsWith('.docx') ||
      filePath.endsWith('.doc')
    );
  };

  const repoGroupComponents: JSX.Element[] = [];
  if (shouldGroupRepoDocs) {
    repoGroupEntries.forEach(([sourceId, repoDocs]) => {
      if (repoDocs.length === 0) return;
      const repoSource = repoSourceMap[sourceId];
      const firstDoc = repoDocs[0];
      const meta = (firstDoc?.extra_metadata || {}) as Record<string, any>;
      const repoLabel =
        (meta.owner && meta.repo ? `${meta.owner}/${meta.repo}` : undefined) ||
        repoSource?.name ||
        firstDoc?.source?.name ||
        'Repository';
      const fileCount = repoDocs.length;
      const progress = sourceProgress[sourceId];
      const pending = combinedPendingSourceIds.includes(sourceId);
      const isExpanded = expandedRepos[sourceId] ?? false;
      const isHighlightedRepoGroup = String(highlightRepoGroupSourceId || '') === String(sourceId);
      const statusText =
        progress?.status ||
        (pending ? 'Queued for ingestion...' : `${fileCount} ${fileCount === 1 ? 'file' : 'files'} ingested`);
      const percent =
        typeof progress?.progress === 'number'
          ? Math.min(100, Math.max(0, progress.progress))
          : undefined;
      repoGroupComponents.push(
        <div
          key={`repo-group-${sourceId}`}
          ref={(el) => {
            repoGroupRefs.current[sourceId] = el;
          }}
          className={`bg-white border rounded-lg p-4 ${
            isHighlightedRepoGroup ? 'border-amber-400 ring-2 ring-amber-200' : 'border-gray-200'
          }`}
        >
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-primary-600" />
                <h3 className="text-lg font-semibold text-gray-900">{repoLabel}</h3>
              </div>
              <p className="text-sm text-gray-600 mt-1">
                {fileCount} {fileCount === 1 ? 'file' : 'files'} from {repoSource?.name || firstDoc?.source?.name}
              </p>
              <p className="text-xs text-gray-500">{statusText}</p>
            </div>
            <div className="flex items-center gap-3">
              <Button
                variant="ghost"
                size="sm"
                onClick={() =>
                  setExpandedRepos((prev) => ({
                    ...prev,
                    [sourceId]: !isExpanded,
                  }))
                }
              >
                {isExpanded ? 'Hide files' : 'Browse files'}
              </Button>
              <Button
                variant="danger"
                size="sm"
                disabled={fileCount === 0 || deleteRepoDocumentsMutation.isLoading}
                loading={deleteRepoDocumentsMutation.isLoading && deletingRepoSourceId === sourceId}
                onClick={() => {
                  if (fileCount === 0) return;
                  if (!window.confirm(`Delete all ${fileCount} files from ${repoLabel}? This cannot be undone.`)) {
                    return;
                  }
                  deleteRepoDocumentsMutation.mutate(sourceId);
                }}
              >
                Delete files
              </Button>
            </div>
          </div>
          {(percent !== undefined || progress?.remaining_formatted) && (
            <div className="mt-3">
              {percent !== undefined && (
                <ProgressBar value={percent} showLabel={false} size="sm" variant="primary" />
              )}
              {progress?.remaining_formatted && (
                <p className="text-xs text-gray-500 mt-1">ETA {progress.remaining_formatted}</p>
              )}
            </div>
          )}
          {pending && !progress && (
            <p className="text-xs text-amber-600 mt-2">Waiting for ingestion to start…</p>
          )}
          {isExpanded && (
            <div className="mt-4 space-y-4">
              {repoDocs.slice(0, 200).map((document) => (
                <DocumentCard
                  key={document.id}
                  document={document}
                  onFilterPersona={handlePersonaFilter}
                  canManagePersona={canManagePersona}
                  onManagePersona={openPersonaManager}
                  canRequestPersonaEdit={canRequestPersonaEdit}
                  onRequestPersonaEdit={handlePersonaEditRequest}
                />
              ))}
              {repoDocs.length > 200 && (
                <div className="text-xs text-gray-500">
                  Showing first 200 files. Use search or filters to narrow results.
                </div>
              )}
            </div>
          )}
        </div>
      );
    });
  }

  const handleGitRepoSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const repoList = gitRepoForm.repos
      .split(/[\n,]+/)
      .map(r => r.trim())
      .filter(Boolean);
    if (repoList.length === 0) {
      toast.error('Please enter at least one repository');
      return;
    }
    const requiresToken = gitRepoForm.provider === 'gitlab';
    const trimmedToken = gitRepoForm.token.trim();
    if (requiresToken && !trimmedToken) {
      toast.error('Access token is required for GitLab repositories');
      return;
    }
    if (gitRepoForm.provider === 'gitlab' && !gitRepoForm.gitlabUrl.trim()) {
      toast.error('GitLab URL is required for GitLab repositories');
      return;
    }

    const payload: GitRepoRequestPayload = {
      provider: gitRepoForm.provider,
      repositories: repoList,
      include_files: gitRepoForm.includeFiles,
      include_issues: gitRepoForm.includeIssues,
      include_pull_requests: gitRepoForm.includePRs,
      include_wiki: gitRepoForm.includeWiki,
      incremental_files: gitRepoForm.incrementalFiles,
      use_gitignore: gitRepoForm.useGitignore,
      auto_sync: gitRepoForm.autoSync,
    };
    if (trimmedToken) {
      payload.token = trimmedToken;
    }
    if (gitRepoForm.name.trim()) {
      payload.name = gitRepoForm.name.trim();
    }
    if (gitRepoForm.provider === 'gitlab') {
      payload.gitlab_url = gitRepoForm.gitlabUrl.trim();
    }
    addGitRepoMutation.mutate(payload);
  };

  const handleArxivSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const parseList = (value: string) =>
      value
        .split(/[\n,]+/)
        .map((item) => item.trim())
        .filter(Boolean);

    const queries = parseList(arxivForm.queries);
    const categories = parseList(arxivForm.categories);
    const paperIds = parseList(arxivForm.paperIds);

    if (queries.length === 0 && paperIds.length === 0 && categories.length === 0) {
      toast.error('Enter at least one query, category, or arXiv ID');
      return;
    }

    const payload: ArxivRequestPayload = {
      search_queries: queries.length > 0 ? queries : undefined,
      paper_ids: paperIds.length > 0 ? paperIds : undefined,
      categories: categories.length > 0 ? categories : undefined,
      max_results: Number(arxivForm.maxResults),
      sort_by: arxivForm.sortBy,
      sort_order: arxivForm.sortOrder,
      auto_sync: arxivForm.autoSync,
    };

    if (arxivForm.name.trim()) {
      payload.name = arxivForm.name.trim();
    }

    addArxivSourceMutation.mutate(payload);
  };

  const handleGitCompareSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!compareSourceId) {
      toast.error('Please select a Git source');
      return;
    }
    if (!compareRepository) {
      toast.error('Please select a repository');
      return;
    }
    if (!compareBaseBranch || !compareTargetBranch) {
      toast.error('Select both base and comparison branches');
      return;
    }
    if (compareBaseBranch === compareTargetBranch) {
      toast.error('Choose two distinct branches');
      return;
    }
    try {
      await apiClient.startGitComparison(compareSourceId, {
        repository: compareRepository,
        base_branch: compareBaseBranch,
        compare_branch: compareTargetBranch,
        include_files: compareIncludeFiles,
        explain: branchExplain,
      });
      toast.success('Comparison started');
      refetchGitCompareJobs();
    } catch (error: any) {
      const message = error?.response?.data?.detail || error?.message || 'Failed to start comparison';
      toast.error(message);
    }
  };

  const getStatusIcon = (document: KnowledgeDocument) => {
    if (document.is_processed) {
      return <CheckCircle className="w-4 h-4 text-green-500" />;
    } else if (document.processing_error) {
      return <XCircle className="w-4 h-4 text-red-500" />;
    } else {
      return <Clock className="w-4 h-4 text-yellow-500" />;
    }
  };

  const getStatusText = (document: KnowledgeDocument) => {
    const isVideoAudioDoc = isVideoAudio(document);
    const { isTranscoding, isTranscribing, isTranscribed, isSummarizing } = getDocFlags(document);
    const failed = docStatus[document.id]?.failed || !!document.processing_error || !!document.extra_metadata?.transcription_error;
    const sumFailed = docSumStatus[document.id]?.failed;
    
    // Error has priority
    if (failed || sumFailed) {
      return 'Failed';
    }

    // Check if we have real-time progress
    const progress = transcriptionProgress[document.id];
    if (progress && (isTranscribing || isTranscoding)) {
      return progress.message || (isTranscribing ? 'Transcribing...' : 'Transcoding to MP4...');
    }
    if (isSummarizing) {
      const sp = summarizationProgress[document.id];
      if (sp) return sp.stage ? `Summarizing (${sp.stage})...` : 'Summarizing...';
      return 'Summarizing...';
    }
    
    if (isTranscoding && isVideoAudioDoc) {
      return 'Transcoding to MP4...';
    }

    if (isTranscribing) {
      return 'Transcribing...';
    } else if (isTranscribed && isVideoAudioDoc) {
      return 'Transcribed';
    } else if (document.is_processed) {
      return 'Processed';
    } else if (document.processing_error) {
      return 'Failed';
    } else {
      return 'Processing';
    }
  };

  const getProgressPercentage = (document: KnowledgeDocument): number | null => {
    const progress = transcriptionProgress[document.id];
    const sp = summarizationProgress[document.id];
    const { isTranscribing, isTranscoding, isSummarizing } = getDocFlags(document);
    if (progress && (isTranscribing || isTranscoding)) {
      return progress.progress;
    }
    if (isSummarizing && sp) {
      return sp.progress ?? null;
    }
    return null;
  };

  function DocumentCard({
    document,
    onFilterPersona = () => {},
    canManagePersona,
    onManagePersona,
    canRequestPersonaEdit,
    onRequestPersonaEdit,
  }: {
    document: KnowledgeDocument;
    onFilterPersona?: (personaId: string, role: 'owner' | 'speaker') => void;
    canManagePersona?: boolean;
    onManagePersona?: (personaId: string) => void;
    canRequestPersonaEdit?: boolean;
    onRequestPersonaEdit?: (persona: Persona, document: KnowledgeDocument) => void;
  }) {
    const ownerPersona = document.owner_persona;
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow duration-200">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            {/* Title and status */}
            <div className="flex items-center space-x-2 mb-2">
              {isVideoAudio(document) && (
                <Video className="w-5 h-5 text-primary-600 flex-shrink-0" />
              )}
              <h3 className="text-lg font-medium text-gray-900 truncate">
                {getDisplayTitle(document)}
              </h3>
              <div className="flex items-center space-x-1 ml-auto">
                {getStatusIcon(document)}
                <span className="text-sm text-gray-500">
                  {getStatusText(document)}
                </span>
              </div>
            </div>

            {/* Progress bar for transcription/transcoding */}
            {(document.extra_metadata?.is_transcribing || document.extra_metadata?.is_transcoding) &&
              getProgressPercentage(document) !== null && (
                <div className="mt-2">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700">
                      {transcriptionProgress[document.id]?.stage
                        ? transcriptionProgress[document.id].stage.charAt(0).toUpperCase() +
                          transcriptionProgress[document.id].stage.slice(1).replace(/_/g, ' ')
                        : 'Processing'}
                    </span>
                    <span className="text-sm text-gray-600">
                      {Math.round(getProgressPercentage(document) || 0)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${getProgressPercentage(document)}%` }}
                    />
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {(() => {
                      const p = transcriptionProgress[document.id];
                      if (!p) return 'Processing...';
                      const parts: string[] = [];
                      const pct = Math.round(getProgressPercentage(document) || 0);
                      if (!isNaN(pct)) parts.push(`${pct}%`);
                      const rem = (p as any).remaining_formatted || null;
                      if (rem) parts.push(`ETA ${rem}`);
                      if (p.message) parts.unshift(p.message);
                      return parts.join(' • ');
                    })()}
                  </div>
                  {/* Live transcript preview toggle - only for transcribing */}
                  {document.extra_metadata?.is_transcribing && (
                    <div className="mt-2">
                      <button
                        className="text-xs text-primary-700 hover:text-primary-800"
                        onClick={() => {
                          setLivePreviewOpen((prev) => ({
                            ...prev,
                            [document.id]: !prev[document.id],
                          }));
                        }}
                      >
                        {livePreviewOpen[document.id] ? 'Hide live transcript' : 'Show live transcript'}
                      </button>
                      {livePreviewOpen[document.id] && (
                        <div className="mt-2 p-2 bg-white border border-gray-200 rounded shadow max-h-40 overflow-auto">
                          {(() => {
                            const segs = streamingSegments[document.id] || [];
                            if (!segs.length) return <div className="text-xs text-gray-500">Waiting for transcript...</div>;
                            const start = Math.max(0, segs.length - 12);
                            const slice = segs.slice(start);
                            const fmt = (s: number) => {
                              const h = Math.floor(s / 3600);
                              const m = Math.floor((s % 3600) / 60);
                              const sec = Math.floor(s % 60);
                              return h > 0
                                ? `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
                                : `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
                            };
                            return (
                              <div className="space-y-1">
                                {slice.map((s, idx) => (
                                  <div key={idx} className="text-xs text-gray-700">
                                    <span className="font-mono text-gray-600 mr-2">{fmt(s.start || 0)}</span>
                                    {s.speaker && <span className="text-gray-600 mr-2">[{s.speaker}]</span>}
                                    <span>{s.text}</span>
                                  </div>
                                ))}
                              </div>
                            );
                          })()}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

            {/* Metadata */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-600 mb-3">
              <div>
                <span className="font-medium">Source:</span>
                <span className="ml-1">{document.source.name}</span>
              </div>
              <div>
                <span className="font-medium">Type:</span>
                <span className="ml-1">{document.file_type || 'Unknown'}</span>
              </div>
              <div>
                <span className="font-medium">Size:</span>
                <span className="ml-1">{formatFileSize(document.file_size)}</span>
              </div>
              <div>
                <span className="font-medium">Updated:</span>
                <span className="ml-1">{formatDistanceToNow(new Date(document.updated_at))} ago</span>
              </div>
            </div>

            {ownerPersona && (
              <div className="flex flex-col gap-1 text-sm text-gray-700 mb-2">
                <div className="flex flex-wrap items-center gap-2">
                  <UserCircle2 className="w-4 h-4 text-primary-600" />
                  <span className="font-medium">Owner persona:</span>
                  <button
                    type="button"
                    className="text-primary-700 hover:text-primary-900 underline-offset-2 hover:underline"
                    onClick={() => onFilterPersona(ownerPersona.id, 'owner')}
                  >
                    {ownerPersona.name}
                  </button>
                  {canManagePersona && onManagePersona && (
                    <button
                      type="button"
                      className="text-xs text-primary-700 hover:text-primary-900 underline-offset-2 hover:underline"
                      onClick={() => onManagePersona(ownerPersona.id)}
                    >
                      Manage persona
                    </button>
                  )}
                  {!canManagePersona && canRequestPersonaEdit && onRequestPersonaEdit && (
                    <button
                      type="button"
                      className="text-xs text-primary-700 hover:text-primary-900 underline-offset-2 hover:underline flex items-center gap-1"
                      onClick={() => onRequestPersonaEdit(ownerPersona, document)}
                    >
                      <MessageSquare className="w-3 h-3" />
                      Suggest change
                    </button>
                  )}
                </div>
                <PersonaMetadata persona={ownerPersona} />
              </div>
            )}

            {/* Tags */}
            {document.tags && document.tags.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-3">
                {document.tags.map((tag, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            )}

            {/* Author */}
            {document.author && (
              <div className="text-sm text-gray-600">
                <span className="font-medium">Author:</span>
                <span className="ml-1">{document.author}</span>
              </div>
            )}

            {/* Error message */}
            {document.processing_error && (
              <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                <strong>Processing Error:</strong> {document.processing_error}
              </div>
            )}

            {/* Summarization progress */}
            {(() => {
              const flags = getDocFlags(document);
              const sp = summarizationProgress[document.id];
              if (!isVideoAudio(document) && flags.isSummarizing && sp) {
                return (
                  <div className="mt-2">
                    <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
                      <span>{sp.stage ? `Summarizing (${sp.stage})...` : 'Summarizing...'}</span>
                      <span className="font-medium">{sp.progress ?? 0}%</span>
                    </div>
                    <ProgressBar value={sp.progress ?? 0} showLabel={false} size="md" />
                  </div>
                );
              }
              return null;
            })()}
          </div>

          {/* Actions */}
          <div className="flex items-center space-x-2 ml-4">
            <Button
              variant="ghost"
              size="sm"
              icon={<Eye className="w-4 h-4" />}
              onClick={() => {
                if (document.extra_metadata?.is_transcoding) {
                  toast.error('Video is converting to MP4. Please wait.');
                  return;
                }
                setInitialSeekSeconds(null);
                setSelectedDocument(document);
              }}
              disabled={document.extra_metadata?.is_transcoding === true}
            >
              View
            </Button>
            <Button
              variant="ghost"
              size="sm"
              icon={<Eye className="w-4 h-4" />}
              onClick={() => navigate(`/documents/${document.id}/graph`)}
              disabled={document.extra_metadata?.is_transcoding === true}
            >
              Graph
            </Button>
            {isVideoAudio(document) && document.extra_metadata?.transcript_document_id && (
              <Button
                variant="ghost"
                size="sm"
                icon={<FileText className="w-4 h-4" />}
                onClick={() => {
                  const tId = document.extra_metadata?.transcript_document_id as string;
                  const tDoc = (allDocuments || []).find((d) => d.id === tId);
                  if (tDoc) {
                    setPreviousTranscriptId(null);
                    setSelectedDocument(tDoc);
                  } else {
                    toast.error('Transcript not found yet');
                  }
                }}
              >
                Open Transcript
              </Button>
            )}
            {document.extra_metadata?.doc_type === 'transcript' &&
              document.extra_metadata?.parent_document_id && (
                <Button
                  variant="ghost"
                  size="sm"
                  icon={<FileVideo className="w-4 h-4" />}
                  onClick={() => {
                    const vId = document.extra_metadata?.parent_document_id as string;
                    const vDoc = (allDocuments || []).find((d) => d.id === vId);
                    if (vDoc) {
                      setInitialSeekSeconds(null);
                      setPreviousTranscriptId(document.id);
                      setSelectedDocument(vDoc);
                    } else {
                      toast.error('Parent video not found');
                    }
                  }}
                >
                  Open Video
                </Button>
              )}
            {isVideoAudio(document) &&
              (docStatus[document.id]?.failed || !!document.extra_metadata?.transcription_error) &&
              !document.extra_metadata?.is_transcribing &&
              !document.extra_metadata?.is_transcoding && (
                <Button
                  variant="ghost"
                  size="sm"
                  icon={<RefreshCw className="w-4 h-4" />}
                  onClick={async () => {
                    try {
                      const res = await apiClient.transcribeDocument(document.id);
                      toast.success(res.message || 'Transcription scheduled');
                      queryClient.invalidateQueries('documents');
                    } catch (e: any) {
                      toast.error(
                        e?.response?.data?.detail || e?.message || 'Failed to schedule transcription'
                      );
                    }
                  }}
                >
                  Retry Transcription
                </Button>
              )}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => summarizeMutation.mutate(document.id)}
              disabled={document.extra_metadata?.is_transcoding === true || getDocFlags(document).isSummarizing}
            >
              Summarize
            </Button>
            {document.url && (
              <Button
                variant="ghost"
                size="sm"
                icon={<Eye className="w-4 h-4" />}
                onClick={() => window.open(document.url, '_blank')}
                disabled={document.extra_metadata?.is_transcoding === true}
              >
                Open
              </Button>
            )}
            {document.file_path && (
              <Button
                variant="ghost"
                size="sm"
                icon={<Download className="w-4 h-4" />}
                onClick={async () => {
                  if (document.extra_metadata?.is_transcoding) {
                    toast.error('Video is converting to MP4. Please wait.');
                    return;
                  }
                  try {
                    const downloadUrl = await apiClient.getDocumentDownloadUrl(document.id, true);
                    const response = await fetch(downloadUrl, {
                      method: 'GET',
                      headers: {
                        Authorization: `Bearer ${localStorage.getItem('access_token') || ''}`,
                      },
                      credentials: 'include',
                    });

                    if (!response.ok) {
                      throw new Error(`Download failed: ${response.statusText}`);
                    }

                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const link = window.document.createElement('a');
                    link.href = url;
                    link.download = getDownloadFilename(document) || `document_${document.id}`;
                    window.document.body.appendChild(link);
                    link.click();
                    window.document.body.removeChild(link);
                    window.URL.revokeObjectURL(url);
                  } catch (error) {
                    toast.error('Failed to generate download URL');
                  }
                }}
                disabled={document.extra_metadata?.is_transcoding === true}
              >
                Download
              </Button>
            )}
            {isDocxFile(document) && (
              <Button
                variant="ghost"
                size="sm"
                icon={<Edit className="w-4 h-4" />}
                onClick={() => handleEditDocument(document)}
                disabled={document.extra_metadata?.is_transcoding === true}
              >
                Edit
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              icon={<Trash2 className="w-4 h-4" />}
              onClick={() => handleDeleteDocument(document.id)}
              disabled={document.extra_metadata?.is_transcoding === true}
              loading={deleteDocumentMutation.isLoading}
            >
              Delete
            </Button>
            {user?.role === 'admin' && (
              <Button
                variant="ghost"
                size="sm"
                icon={<RefreshCw className="w-4 h-4" />}
                onClick={() => handleReprocessDocument(document.id)}
                loading={reprocessDocumentMutation.isLoading}
              >
                Reprocess
              </Button>
            )}
          </div>
        </div>
      </div>
    );
  }

  const tabs = [
    { id: 'documents' as const, name: 'Documents', icon: FileText },
    { id: 'videos' as const, name: 'Videos & Audio', icon: Video },
    { id: 'repos' as const, name: 'Git Repositories', icon: Plus },
    { id: 'arxiv' as const, name: 'ArXiv', icon: BookOpen },
  ];
  const tabTitle =
    activeTab === 'videos'
      ? 'Videos & Audio'
      : activeTab === 'repos'
        ? 'Git Repositories'
        : activeTab === 'arxiv'
          ? 'ArXiv'
          : 'Documents';

  const activeGitRequests = activeGitStatuses || [];
  const activeGitSources = activeGitRequests.map((entry) => entry.source);

  // If navigated to Repos for a specific source but it's not active anymore, fall back to Documents tab
  // and expand the repo files (so the user still lands in a relevant view).
  useEffect(() => {
    if (activeTab !== 'repos') return;
    const sid = String(highlightIngestionSourceId || '').trim();
    if (!sid) return;
    if (!activeGitStatuses) return; // wait for first fetch

    const isActiveRequest = activeGitRequests.some((r) => String(r?.source?.id) === sid);
    const isPending = combinedPendingSourceIds.includes(sid);
    const hasProgress = Boolean(sourceProgress[sid]);

    if (isActiveRequest || isPending || hasProgress) return;

    const t = window.setTimeout(() => {
      setActiveTab('documents');
      setExpandedRepos((prev) => ({ ...prev, [sid]: true }));
      setHighlightRepoGroupSourceId(sid);
    }, 450);
    return () => window.clearTimeout(t);
  }, [activeTab, highlightIngestionSourceId, activeGitStatuses, activeGitRequests, combinedPendingSourceIds, sourceProgress]);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold text-gray-900">
            {tabTitle}
          </h1>
          {(activeTab === 'documents' || activeTab === 'videos') && (
            <div className="flex items-center space-x-3">
              <Button
                onClick={() => refetchDocuments()}
                variant="ghost"
                icon={<RefreshCw className="w-4 h-4" />}
              >
                Refresh
              </Button>
              {activeTab === 'documents' && (
                <Button
                  onClick={() => setShowIngestUrlModal(true)}
                  variant="ghost"
                  icon={<Link2 className="w-4 h-4" />}
                >
                  Ingest URL
                </Button>
              )}
              <Button
                onClick={() => setShowUploadModal(true)}
                icon={<Upload className="w-4 h-4" />}
              >
                Upload {activeTab === 'videos' ? 'Video/Audio' : 'Document'}
              </Button>
            </div>
          )}
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 border-b border-gray-200 mb-4">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-700'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="w-4 h-4 mr-2" />
                {tab.name}
            {tab.id === 'videos' && (
              <span className="ml-2 px-2 py-0.5 text-xs font-medium bg-gray-100 text-gray-700 rounded-full">
                {allDocuments?.filter(isVideoAudio).length || 0}
              </span>
            )}
            {tab.id === 'documents' && (
                  <span className="ml-2 px-2 py-0.5 text-xs font-medium bg-gray-100 text-gray-700 rounded-full">
                    {allDocuments?.filter(d => !isVideoAudio(d)).length || 0}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {(activeTab === 'documents' || activeTab === 'videos') && (
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex-1 min-w-[220px]">
              <Input
                placeholder="Search documents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                leftIcon={<Search className="w-4 h-4" />}
              />
            </div>
            <div className="w-48 min-w-[180px]">
              <select
                value={selectedSource}
                onChange={(e) => setSelectedSource(e.target.value)}
                className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
              >
                <option value="">All Sources</option>
                {sources?.map((source) => (
                  <option key={source.id} value={source.id}>
                    {source.name}
                  </option>
                ))}
              </select>
            </div>
            <PersonaFilterAutosuggest
              label="Owner persona"
              placeholder="All owners"
              value={ownerPersonaFilter}
              onChange={setOwnerPersonaFilter}
              options={personaOptions}
              loading={personasLoading}
            />
            <PersonaFilterAutosuggest
              label="Speaker persona"
              placeholder="All speakers"
              value={speakerPersonaFilter}
              onChange={setSpeakerPersonaFilter}
              options={personaOptions}
              loading={personasLoading}
            />
            {(ownerPersonaFilter || speakerPersonaFilter) && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setOwnerPersonaFilter('');
                  setSpeakerPersonaFilter('');
                }}
              >
                Clear persona filters
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === 'repos' ? (
          <div className="p-6">
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-medium text-gray-900">Process a Git Repository</h3>
                  <p className="text-sm text-gray-600">Provide repository details to ingest documentation, code, issues, or wiki content.</p>
                </div>
              </div>
              {activeGitSources.length > 0 && (
                <div className="mt-4 mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Active requests</h4>
                    <span className="text-xs text-gray-500">{activeGitSources.length} running</span>
                  </div>
                  <div className="space-y-3">
                    {activeGitRequests.map(({ source, pending }) => {
                      const progress = sourceProgress[source.id];
                      const syncing = (source as any)?.is_syncing === true;
                      const pendingState = pending || combinedPendingSourceIds.includes(source.id);
                      const isCanceled = progress?.canceled === true;
                      const isHighlighted = String(highlightIngestionSourceId || '') === String(source.id);
                      let percent = typeof progress?.progress === 'number' ? progress.progress : undefined;
                      if (percent === undefined && progress?.current && progress?.total) {
                        percent = Math.round((progress.current / Math.max(1, progress.total)) * 100);
                      }
                      if (percent === undefined) {
                        percent = syncing || pendingState ? 5 : 100;
                      }
                      const statusText =
                        isCanceled
                          ? 'Canceled'
                          : progress?.status ||
                            progress?.stage ||
                            (pendingState ? 'Queued for ingestion...' : syncing ? 'Preparing repository...' : 'Waiting for updates...');
                      return (
                        <div
                          key={source.id}
                          ref={(el) => {
                            ingestionCardRefs.current[source.id] = el;
                          }}
                          className={`p-3 border rounded-lg bg-primary-50/40 ${
                            isHighlighted ? 'border-amber-400 ring-2 ring-amber-200' : 'border-primary-100'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div>
                              <p className="text-sm font-medium text-gray-900">{source.name}</p>
                              <p className="text-xs text-gray-600">{statusText}</p>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-semibold text-primary-700">
                                {Math.min(100, Math.max(0, percent || 0)).toFixed(0)}%
                              </span>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => cancelSourceMutation.mutate(source.id)}
                                loading={cancelSourceMutation.isLoading && cancelingSourceId === source.id}
                              >
                                Cancel
                              </Button>
                            </div>
                          </div>
                          <ProgressBar value={Math.min(100, Math.max(0, percent || 0))} showLabel={false} size="sm" />
                          {progress?.remaining_formatted && (
                            <p className="text-xs text-gray-500 mt-1">ETA {progress.remaining_formatted}</p>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
              <form className="mt-4 space-y-4" onSubmit={handleGitRepoSubmit}>
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Source Name (optional)</label>
                    <input
                      type="text"
                      value={gitRepoForm.name}
                      onChange={(e) => setGitRepoForm(prev => ({ ...prev, name: e.target.value }))}
                      className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                      placeholder="e.g. Platform Docs Repos"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Provider</label>
                    <select
                      value={gitRepoForm.provider}
                      onChange={(e) => setGitRepoForm(prev => ({ ...prev, provider: e.target.value as 'github' | 'gitlab' }))}
                      className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                    >
                      <option value="github">GitHub</option>
                      <option value="gitlab">GitLab</option>
                    </select>
                  </div>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Access Token {gitRepoForm.provider === 'github' ? '(optional for public repos)' : '(required)'}
                    </label>
                    <input
                      type="password"
                      value={gitRepoForm.token}
                      onChange={(e) => setGitRepoForm(prev => ({ ...prev, token: e.target.value }))}
                      className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                      placeholder="Personal access token"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      {gitRepoForm.provider === 'github'
                        ? 'Optional for public repositories. Provide a PAT for private repos or higher rate limits.'
                        : 'Required: GitLab PAT with read_api scope.'}
                    </p>
                  </div>
                  {gitRepoForm.provider === 'gitlab' && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">GitLab URL</label>
                      <input
                        type="text"
                        value={gitRepoForm.gitlabUrl}
                        onChange={(e) => setGitRepoForm(prev => ({ ...prev, gitlabUrl: e.target.value }))}
                        className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                        placeholder="https://gitlab.example.com"
                      />
                    </div>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Repositories</label>
                  <textarea
                    value={gitRepoForm.repos}
                    onChange={(e) => setGitRepoForm(prev => ({ ...prev, repos: e.target.value }))}
                    rows={3}
                    className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                    placeholder="owner/repo-one&#10;owner/repo-two"
                  />
                  <p className="text-xs text-gray-500 mt-1">Separate multiple repositories with commas or new lines.</p>
                </div>
                <div className="flex flex-wrap gap-4 text-sm text-gray-700">
                  <label className="inline-flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={gitRepoForm.includeFiles}
                      onChange={(e) => setGitRepoForm(prev => ({ ...prev, includeFiles: e.target.checked }))}
                    />
                    Include repository files
                  </label>
                  <label className="inline-flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={gitRepoForm.includeIssues}
                      onChange={(e) => setGitRepoForm(prev => ({ ...prev, includeIssues: e.target.checked }))}
                    />
                    Include issues
                  </label>
                  <label className="inline-flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={gitRepoForm.includePRs}
                      onChange={(e) => setGitRepoForm(prev => ({ ...prev, includePRs: e.target.checked }))}
                    />
                    Include PRs / MRs
                  </label>
                  <label className="inline-flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={gitRepoForm.includeWiki}
                      onChange={(e) => setGitRepoForm(prev => ({ ...prev, includeWiki: e.target.checked }))}
                    />
                    Include wiki pages
                  </label>
                  <label className="inline-flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={gitRepoForm.autoSync}
                      onChange={(e) => setGitRepoForm(prev => ({ ...prev, autoSync: e.target.checked }))}
                    />
                    Start processing immediately
                  </label>
                  <label className="inline-flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={gitRepoForm.useGitignore}
                      onChange={(e) => setGitRepoForm(prev => ({ ...prev, useGitignore: e.target.checked }))}
                    />
                    Merge repository .gitignore
                  </label>
                </div>
                <div className="flex justify-end">
                  <Button
                    type="submit"
                    loading={addGitRepoMutation.isLoading}
                    disabled={(gitRepoForm.provider === 'gitlab' && !gitRepoForm.token.trim()) || !gitRepoForm.repos.trim()}
                  >
                    Add Repository
                  </Button>
                </div>
              </form>
              <div className="mt-8 border-t border-gray-100 pt-6">
                <h3 className="text-lg font-medium text-gray-900">Compare branches with LLM explanation</h3>
                <p className="text-sm text-gray-600">
                  Select one of your Git sources, choose two branches, and generate an automated summary of the differences.
                </p>
                <form className="mt-4 space-y-4" onSubmit={handleGitCompareSubmit}>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Git source</label>
                      <select
                        value={compareSourceId}
                        onChange={(e) => setCompareSourceId(e.target.value)}
                        className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                      >
                        {gitSourcesForUser.length === 0 && <option value="">No Git sources available</option>}
                        {gitSourcesForUser.map((source) => (
                          <option key={source.id} value={source.id}>
                            {source.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Repository</label>
                      <select
                        value={compareRepository}
                        onChange={(e) => setCompareRepository(e.target.value)}
                        disabled={!compareSourceId || !(repoOptionsBySource[compareSourceId] || []).length}
                        className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500 disabled:bg-gray-50"
                      >
                        {(repoOptionsBySource[compareSourceId] || []).map((repo) => (
                          <option key={repo} value={repo}>
                            {repo}
                          </option>
                        ))}
                        {(!repoOptionsBySource[compareSourceId] || repoOptionsBySource[compareSourceId].length === 0) && (
                          <option value="">Add repositories to this source first</option>
                        )}
                      </select>
                    </div>
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Base branch</label>
                      <select
                        value={compareBaseBranch}
                        onChange={(e) => setCompareBaseBranch(e.target.value)}
                        disabled={!branchList.length}
                        className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500 disabled:bg-gray-50"
                      >
                        {branchList.map((branch) => (
                          <option key={branch.name} value={branch.name}>
                            {branch.name}
                          </option>
                        ))}
                        {!branchList.length && <option value="">{loadingBranches ? 'Loading branches...' : 'No branches found'}</option>}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Compare branch</label>
                      <select
                        value={compareTargetBranch}
                        onChange={(e) => setCompareTargetBranch(e.target.value)}
                        disabled={!branchList.length}
                        className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500 disabled:bg-gray-50"
                      >
                        {branchList.map((branch) => (
                          <option key={branch.name} value={branch.name}>
                            {branch.name}
                          </option>
                        ))}
                        {!branchList.length && <option value="">{loadingBranches ? 'Loading branches...' : 'No branches found'}</option>}
                      </select>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-4 text-sm text-gray-700">
                    <label className="inline-flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={compareIncludeFiles}
                        onChange={(e) => setCompareIncludeFiles(e.target.checked)}
                      />
                      Include file statistics
                    </label>
                    <label className="inline-flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={branchExplain}
                        onChange={(e) => setBranchExplain(e.target.checked)}
                      />
                      Generate LLM explanation
                    </label>
                  </div>
                  <div className="flex items-center gap-3">
                    <Button type="submit" disabled={!compareSourceId || !compareRepository || loadingBranches}>
                      {loadingBranches ? 'Loading branches...' : 'Compare branches'}
                    </Button>
                    {loadingBranches && <span className="text-xs text-gray-500">Fetching branches…</span>}
                  </div>
                </form>
                <div className="mt-6">
                  <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Comparison jobs</h4>
                  {gitCompareJobs.length === 0 ? (
                    <p className="text-xs text-gray-500 mt-2">No branch comparison jobs yet.</p>
                  ) : (
                    <div className="mt-3 space-y-3">
                      {gitCompareJobs.map((job) => {
                        const stats = job.diff_summary?.stats || {};
                        return (
                          <div key={job.id} className="border border-gray-200 rounded-lg p-3 bg-white">
                            <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-2">
                              <div>
                                <p className="text-sm font-medium text-gray-900">{job.repository}</p>
                                <p className="text-xs text-gray-600">
                                  {job.base_branch} → {job.compare_branch}
                                </p>
                                <p className="text-xs text-gray-500 capitalize">Status: {job.status.replace('_', ' ')}</p>
                              </div>
                              {['queued', 'running', 'cancel_requested'].includes(job.status) && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => cancelCompareMutation.mutate(job.id)}
                                  loading={cancelCompareMutation.isLoading && cancelingCompareJobId === job.id}
                                >
                                  Cancel
                                </Button>
                              )}
                            </div>
                            {job.llm_summary && (
                              <div className="mt-2 text-sm text-gray-800 bg-primary-50 border border-primary-100 rounded p-2 whitespace-pre-wrap">
                                {job.llm_summary}
                              </div>
                            )}
                            {job.diff_summary?.files && (
                              <div className="mt-2 text-xs text-gray-600">
                                <p>
                                  Files changed: {stats.total_files ?? job.diff_summary.files.length} | Commits:{' '}
                                  {stats.total_commits ?? job.diff_summary.raw?.commit_messages?.length ?? 0} | Ahead:{' '}
                                  {stats.ahead_by ?? 0} | Behind: {stats.behind_by ?? 0}
                                </p>
                                <div className="mt-1 max-h-28 overflow-auto space-y-1">
                                  {job.diff_summary.files.slice(0, 5).map((file: any) => (
                                    <div key={`${job.id}-${file.filename}`} className="flex items-center text-gray-700">
                                      <span className="font-mono text-[11px] w-20">{file.status || 'modified'}</span>
                                      <span className="flex-1 truncate">{file.filename}</span>
                                      <span className="ml-2 text-xs text-gray-500">
                                        +{file.additions || 0}/-{file.deletions || 0}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                            {job.error && (
                              <p className="mt-2 text-xs text-red-600">
                                <strong>Error:</strong> {job.error}
                              </p>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : activeTab === 'arxiv' ? (
          <div className="p-6">
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-medium text-gray-900">Ingest Papers from ArXiv</h3>
                  <p className="text-sm text-gray-600">Search the ArXiv API or provide explicit IDs to keep research papers in sync.</p>
                </div>
              </div>
              <form className="mt-4 space-y-4" onSubmit={handleArxivSubmit}>
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Source Name (optional)</label>
                    <input
                      type="text"
                      value={arxivForm.name}
                      onChange={(e) => setArxivForm(prev => ({ ...prev, name: e.target.value }))}
                      className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                      placeholder="e.g. ArXiv - Latest LLM papers"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Max Results per Query</label>
                    <input
                      type="number"
                      min={1}
                      max={200}
                      value={arxivForm.maxResults}
                      onChange={(e) => setArxivForm(prev => ({ ...prev, maxResults: Number(e.target.value) }))}
                      className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">ArXiv allows up to 200 results per request.</p>
                  </div>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Search Queries</label>
                    <textarea
                      value={arxivForm.queries}
                      onChange={(e) => setArxivForm(prev => ({ ...prev, queries: e.target.value }))}
                      rows={4}
                      className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                      placeholder={`cat:cs.CL AND ti:transformer\nall:"retrieval augmented"`}
                    />
                    <p className="text-xs text-gray-500 mt-1">Use ArXiv search syntax. Separate multiple queries with commas or new lines.</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Specific ArXiv IDs (optional)</label>
                    <textarea
                      value={arxivForm.paperIds}
                      onChange={(e) => setArxivForm(prev => ({ ...prev, paperIds: e.target.value }))}
                      rows={4}
                      className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                      placeholder={'2401.01234\narXiv:2305.06789'}
                    />
                    <p className="text-xs text-gray-500 mt-1">Enter full IDs (with or without the arXiv: prefix) separated by commas or new lines.</p>
                  </div>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Categories</label>
                    <input
                      type="text"
                      value={arxivForm.categories}
                      onChange={(e) => setArxivForm(prev => ({ ...prev, categories: e.target.value }))}
                      className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                      placeholder="cs.CL, cs.AI"
                    />
                    <p className="text-xs text-gray-500 mt-1">Optional. Codes are ANDed with each query.</p>
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Sort By</label>
                      <select
                        value={arxivForm.sortBy}
                        onChange={(e) => setArxivForm(prev => ({ ...prev, sortBy: e.target.value as ArxivFormState['sortBy'] }))}
                        className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                      >
                        <option value="submittedDate">Submitted Date</option>
                        <option value="lastUpdatedDate">Last Updated</option>
                        <option value="relevance">Relevance</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Sort Order</label>
                      <select
                        value={arxivForm.sortOrder}
                        onChange={(e) => setArxivForm(prev => ({ ...prev, sortOrder: e.target.value as ArxivFormState['sortOrder'] }))}
                        className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-primary-500 focus:border-primary-500"
                      >
                        <option value="descending">Descending</option>
                        <option value="ascending">Ascending</option>
                      </select>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                    <input
                      type="checkbox"
                      checked={arxivForm.autoSync}
                      onChange={(e) => setArxivForm(prev => ({ ...prev, autoSync: e.target.checked }))}
                    />
                    Start ingestion immediately
                  </label>
                </div>
                <div className="flex justify-end">
                  <Button
                    type="submit"
                    loading={addArxivSourceMutation.isLoading}
                    disabled={
                      addArxivSourceMutation.isLoading ||
                      (!arxivForm.queries.trim() && !arxivForm.paperIds.trim() && !arxivForm.categories.trim())
                    }
                  >
                    Submit ArXiv Request
                  </Button>
                </div>
              </form>
            </div>
          </div>
        ) : documentsLoading ? (
          <LoadingSpinner className="h-64" text={`Loading ${activeTab === 'videos' ? 'videos' : 'documents'}...`} />
        ) : documents.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              {activeTab === 'videos' ? (
                <Video className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              ) : (
                <FileText className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              )}
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                No {activeTab === 'videos' ? 'videos or audio files' : 'documents'} found
              </h3>
              <p className="text-gray-500 mb-6">
                {searchQuery || selectedSource 
                  ? 'Try adjusting your search or filter criteria'
                  : activeTab === 'videos'
                    ? 'Upload video or audio files to get started'
                    : 'Upload documents or configure data sources to get started'
                }
              </p>
              <Button
                onClick={() => setShowUploadModal(true)}
                icon={<Upload className="w-4 h-4" />}
              >
                Upload {activeTab === 'videos' ? 'Video/Audio' : 'Document'}
              </Button>
            </div>
          </div>
        ) : (
          <div className="p-6">
            <div className="grid gap-4">
              {repoGroupComponents}
              {regularDocuments.map((document) => (
                <DocumentCard
                  key={document.id}
                  document={document}
                  onFilterPersona={handlePersonaFilter}
                  canManagePersona={canManagePersona}
                  onManagePersona={openPersonaManager}
                  canRequestPersonaEdit={canRequestPersonaEdit}
                  onRequestPersonaEdit={handlePersonaEditRequest}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <UploadModal
          onClose={() => setShowUploadModal(false)}
          onSuccess={() => {
            setShowUploadModal(false);
            queryClient.invalidateQueries('documents');
          }}
        />
      )}

      {showIngestUrlModal && (
        <IngestUrlModal
          isAdmin={user?.role === 'admin'}
          onClose={() => setShowIngestUrlModal(false)}
          onSuccess={() => {
            setShowIngestUrlModal(false);
            queryClient.invalidateQueries('documents');
          }}
        />
      )}

      {/* Document Details Modal */}
      {selectedDocument && (
        <DocumentDetailsModal
          document={selectedDocument}
          onClose={() => setSelectedDocument(null)}
          liveSegments={streamingSegments[selectedDocument.id]}
          initialSeekSeconds={initialSeekSeconds ?? undefined}
          previousTranscriptId={previousTranscriptId ?? undefined}
          onOpenParentVideo={(parentId: string, seconds?: number) => {
            const vDoc = (allDocuments || []).find(d => d.id === parentId);
            if (vDoc) {
              setInitialSeekSeconds(typeof seconds === 'number' ? seconds : null);
              // If we are currently on a transcript, remember it
              if (selectedDocument && (selectedDocument as any)?.extra_metadata?.doc_type === 'transcript') {
                setPreviousTranscriptId(selectedDocument.id);
              }
              setSelectedDocument(vDoc);
            } else {
              toast.error('Parent video not found');
            }
          }}
          onOpenTranscript={(transcriptId: string) => {
            const tDoc = (allDocuments || []).find(d => d.id === transcriptId);
            if (tDoc) {
              setInitialSeekSeconds(null);
              setSelectedDocument(tDoc);
            } else {
              toast.error('Transcript not found');
            }
          }}
          highlightChunkId={(location.state as LocationState)?.highlightChunkId}
          onFilterPersona={handlePersonaFilter}
          canManagePersona={canManagePersona}
          onManagePersona={openPersonaManager}
          canRequestPersonaEdit={canRequestPersonaEdit}
          onRequestPersonaEdit={handlePersonaEditRequest}
        />
      )}

      {personaEditRequest && (
        <PersonaEditRequestModal
          persona={personaEditRequest.persona}
          document={personaEditRequest.document || undefined}
          onClose={() => setPersonaEditRequest(null)}
          onSubmit={(message) =>
            personaEditRequestMutation.mutate({
              personaId: personaEditRequest.persona.id,
              message,
              documentId: personaEditRequest.document?.id,
            })
          }
          isSubmitting={personaEditRequestMutation.isLoading}
        />
      )}

      {/* DOCX Editor Modal */}
      {docxEditorOpen && documentToEdit && (
        <DocxEditorModal
          documentId={documentToEdit.id}
          documentTitle={documentToEdit.title || 'Untitled'}
          isOpen={docxEditorOpen}
          onClose={() => {
            setDocxEditorOpen(false);
            setDocumentToEdit(null);
          }}
          onSaved={() => {
            queryClient.invalidateQueries('documents');
          }}
        />
      )}

      {/* Delete Confirmation Modal */}
      <ConfirmationModal
        isOpen={deleteConfirmOpen}
        onClose={() => {
          setDeleteConfirmOpen(false);
          setDocumentToDelete(null);
        }}
        onConfirm={confirmDeleteDocument}
        title="Delete Document"
        message="Are you sure you want to delete this document? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        variant="danger"
        isLoading={deleteDocumentMutation.isLoading}
      />
    </div>
  );
};

// Upload Modal Component
interface UploadModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

const UploadModal: React.FC<UploadModalProps> = ({ onClose, onSuccess }) => {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState('');
  const [tags, setTags] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [uploadedBytes, setUploadedBytes] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  const uploadMutation = useMutation(
    async ({ file, title, tags }: { file: File; title?: string; tags?: string[] }) => {
      setIsUploading(true);
      setUploadProgress(0);
      setUploadStatus('Preparing upload...');
      
      try {
        const result = await apiClient.uploadDocumentWithProgress(
          file,
          title,
          tags,
          (progress) => {
            setUploadProgress(progress);
          },
          (status) => {
            setUploadStatus(status);
          },
          (uploaded, total) => {
            setUploadedBytes(uploaded);
          }
        );
        return result;
      } finally {
        setIsUploading(false);
      }
    },
    {
      onSuccess: () => {
        toast.success('Document uploaded successfully');
        setUploadProgress(0);
        setUploadStatus('');
        onSuccess();
      },
      onError: (error: any) => {
        const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to upload document';
        toast.error(errorMessage);
        setUploadProgress(0);
        setUploadStatus('');
        setIsUploading(false);
      },
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    const tagArray = tags
      .split(',')
      .map(tag => tag.trim())
      .filter(tag => tag.length > 0);

    uploadMutation.mutate({
      file,
      title: title || undefined,
      tags: tagArray.length > 0 ? tagArray : undefined,
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-md">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload Document</h2>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              File
            </label>
            <input
              type="file"
              accept=".pdf,.doc,.docx,.ppt,.pptx,.txt,.html,.htm,.md,.markdown,.mp4,.avi,.mkv,.mov,.webm,.flv,.wmv,.mp3,.wav,.m4a,.flac,.ogg,.aac,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/vnd.ms-powerpoint,text/plain,text/html,text/markdown,video/mp4,video/x-msvideo,video/x-matroska,video/quicktime,video/webm,video/x-flv,video/x-ms-wmv,audio/mpeg,audio/wav,audio/x-m4a,audio/flac,audio/ogg,audio/aac"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
              required
            />
          </div>

          <Input
            label="Title (Optional)"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Enter document title"
          />

          <Input
            label="Tags (Optional)"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="Enter tags separated by commas"
            helpText="e.g. documentation, guide, reference"
          />

          {/* Upload Progress */}
          {isUploading && (
            <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">{uploadStatus || 'Uploading...'}</span>
            <span className="text-gray-600 font-medium">{uploadProgress}%</span>
          </div>
          <ProgressBar
            value={uploadProgress}
            showLabel={false}
            size="md"
            variant="primary"
          />
          {file && (
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span>{formatFileSize(uploadedBytes)} / {formatFileSize(file.size)}</span>
            </div>
          )}
          {file && (
            <p className="text-xs text-gray-500">
              {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          )}
            </div>
          )}

          <div className="flex justify-end space-x-3 pt-4">
            <Button
              type="button"
              variant="ghost"
              onClick={onClose}
              disabled={isUploading}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              loading={uploadMutation.isLoading}
              disabled={!file || isUploading}
            >
              {isUploading ? 'Uploading...' : 'Upload'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

interface IngestUrlModalProps {
  isAdmin: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

const IngestUrlModal: React.FC<IngestUrlModalProps> = ({ isAdmin, onClose, onSuccess }) => {
  const [url, setUrl] = useState('');
  const [tags, setTags] = useState('');
  const [runInBackground, setRunInBackground] = useState(true);
  const [followLinks, setFollowLinks] = useState(false);
  const [onePerPage, setOnePerPage] = useState(false);
  const [maxPages, setMaxPages] = useState(3);
  const [maxDepth, setMaxDepth] = useState(1);
  const [sameDomainOnly, setSameDomainOnly] = useState(true);
  const [allowPrivateNetworks, setAllowPrivateNetworks] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState<{ progress?: number; stage?: string; status?: string; current?: number; total?: number } | null>(null);
  const progressWsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    return () => {
      try {
        progressWsRef.current?.close();
      } catch {}
      progressWsRef.current = null;
    };
  }, []);

  const ingestMutation = useMutation(
    async () => {
      const tagList = tags
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean);
      const payload = {
        url: url.trim(),
        tags: tagList.length ? tagList : undefined,
        follow_links: followLinks,
        one_document_per_page: onePerPage,
        max_pages: followLinks ? maxPages : 1,
        max_depth: followLinks ? maxDepth : 0,
        same_domain_only: sameDomainOnly,
        allow_private_networks: allowPrivateNetworks,
      };
      if (runInBackground) {
        return apiClient.ingestUrlAsync(payload as any);
      }
      return apiClient.ingestUrl(payload as any);
    },
    {
      onSuccess: (res) => {
        if (runInBackground) {
          const id = (res as any)?.job_id;
          if (!id) {
            toast.error('Failed to start background ingest job');
            return;
          }
          setJobId(id);
          setJobProgress({ progress: 1, stage: 'queued', status: 'Job queued…' });
          toast.success('URL ingest started');

          try {
            const ws = apiClient.createUrlIngestProgressWebSocket(id);
            progressWsRef.current = ws;
            ws.onmessage = (event) => {
              try {
                const data = JSON.parse(event.data);
                if (data.type === 'ingestion_progress') {
                  const p = data.progress || {};
                  setJobProgress({
                    progress: typeof p.progress === 'number' ? p.progress : undefined,
                    stage: p.stage,
                    status: p.status,
                    current: p.current,
                    total: p.total,
                  });
                } else if (data.type === 'ingestion_status') {
                  const s = data.status || {};
                  setJobProgress((prev) => ({
                    ...(prev || {}),
                    progress: typeof s.progress === 'number' ? s.progress : prev?.progress,
                    stage: s.stage || prev?.stage,
                    status: s.status || prev?.status,
                  }));
                } else if (data.type === 'ingestion_complete') {
                  const r = data.result || {};
                  const created = r?.created?.length || 0;
                  const updated = r?.updated?.length || 0;
                  const skipped = r?.skipped?.length || 0;
                  const errors = r?.errors?.length || 0;
                  toast.success(`Ingested URL. Created ${created}, updated ${updated}, skipped ${skipped}, errors ${errors}.`);
                  try {
                    progressWsRef.current?.close();
                  } catch {}
                  progressWsRef.current = null;
                  onSuccess();
                } else if (data.type === 'ingestion_error') {
                  toast.error(data.error || 'URL ingest failed');
                  try {
                    progressWsRef.current?.close();
                  } catch {}
                  progressWsRef.current = null;
                }
              } catch {
                // ignore invalid payloads
              }
            };
          } catch (e: any) {
            toast.error(e?.message || 'Failed to open progress WebSocket');
          }
        } else {
          const created = (res as any)?.created?.length || 0;
          const updated = (res as any)?.updated?.length || 0;
          const skipped = (res as any)?.skipped?.length || 0;
          const errors = (res as any)?.errors?.length || 0;
          toast.success(`Ingested URL. Created ${created}, updated ${updated}, skipped ${skipped}, errors ${errors}.`);
          onSuccess();
        }
      },
      onError: (err: any) => {
        const message = err?.response?.data?.detail || err?.message || 'Failed to ingest URL';
        toast.error(message);
      },
    }
  );

  const canSubmit = url.trim().length > 0 && !ingestMutation.isLoading;
  const isRunning = ingestMutation.isLoading || !!jobId;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-lg w-full p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Ingest URL</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4">
          <Input
            label="URL"
            placeholder="https://…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={isRunning}
            fullWidth
          />

          <Input
            label="Tags (comma-separated)"
            placeholder="wiki, portal, onboarding"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            disabled={isRunning}
            fullWidth
          />

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <label className="flex items-center space-x-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={runInBackground}
                onChange={(e) => setRunInBackground(e.target.checked)}
                disabled={isRunning}
              />
              <span>Run in background (progress)</span>
            </label>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <label className="flex items-center space-x-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={followLinks}
                onChange={(e) => setFollowLinks(e.target.checked)}
                disabled={isRunning}
              />
              <span>Follow links</span>
            </label>
            <label className="flex items-center space-x-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={onePerPage}
                onChange={(e) => setOnePerPage(e.target.checked)}
                disabled={isRunning || !followLinks}
              />
              <span>One doc per page</span>
            </label>
            <label className="flex items-center space-x-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={sameDomainOnly}
                onChange={(e) => setSameDomainOnly(e.target.checked)}
                disabled={isRunning || !followLinks}
              />
              <span>Same domain only</span>
            </label>
            <label className="flex items-center space-x-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={allowPrivateNetworks}
                onChange={(e) => setAllowPrivateNetworks(e.target.checked)}
                disabled={isRunning || !isAdmin}
              />
              <span>Allow private networks {isAdmin ? '' : '(admin)'}</span>
            </label>
          </div>

          {followLinks && (
            <div className="grid grid-cols-2 gap-3">
              <Input
                label="Max pages"
                type="number"
                min={1}
                max={25}
                value={String(maxPages)}
                onChange={(e) => setMaxPages(parseInt(e.target.value || '1', 10))}
                disabled={isRunning}
                fullWidth
              />
              <Input
                label="Max depth"
                type="number"
                min={0}
                max={5}
                value={String(maxDepth)}
                onChange={(e) => setMaxDepth(parseInt(e.target.value || '0', 10))}
                disabled={isRunning}
                fullWidth
              />
            </div>
          )}

          {jobId && (
            <div className="space-y-2">
              <div className="text-sm text-gray-700">
                {jobProgress?.status || 'Working…'}
                {typeof jobProgress?.current === 'number' && typeof jobProgress?.total === 'number'
                  ? ` (${jobProgress.current}/${jobProgress.total})`
                  : ''}
              </div>
              <ProgressBar
                value={typeof jobProgress?.progress === 'number' ? jobProgress.progress : 0}
                max={100}
                indeterminate={typeof jobProgress?.progress !== 'number'}
                showLabel
              />
            </div>
          )}
        </div>

        <div className="mt-6 flex justify-end space-x-3">
          <Button variant="ghost" onClick={onClose} disabled={ingestMutation.isLoading}>
            Close
          </Button>
          {jobId ? (
            <Button
              variant="danger"
              onClick={async () => {
                try {
                  await apiClient.cancelUrlIngest(jobId);
                  toast.success('Cancel requested');
                  try {
                    progressWsRef.current?.close();
                  } catch {}
                  progressWsRef.current = null;
                  setJobId(null);
                  setJobProgress(null);
                } catch (err: any) {
                  toast.error(err?.response?.data?.detail || err?.message || 'Failed to cancel job');
                }
              }}
            >
              Stop
            </Button>
          ) : (
            <Button onClick={() => ingestMutation.mutate()} disabled={!canSubmit} icon={<Link2 className="w-4 h-4" />}>
              {ingestMutation.isLoading ? 'Ingesting…' : runInBackground ? 'Start' : 'Ingest'}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};

// Document Details Modal Component
interface DocumentDetailsModalProps {
  document: KnowledgeDocument;
  onClose: () => void;
  liveSegments?: Array<{ start: number; text: string; speaker?: string }>;
  initialSeekSeconds?: number;
  onOpenParentVideo?: (parentId: string, seconds?: number) => void;
  previousTranscriptId?: string;
  onOpenTranscript?: (transcriptId: string) => void;
  highlightChunkId?: string;
  onFilterPersona?: (personaId: string, role?: 'owner' | 'speaker') => void;
  canManagePersona?: boolean;
  onManagePersona?: (personaId: string) => void;
  canRequestPersonaEdit?: boolean;
  onRequestPersonaEdit?: (persona: Persona, document?: KnowledgeDocument) => void;
}

const DocumentDetailsModal: React.FC<DocumentDetailsModalProps> = ({
  document,
  onClose,
  liveSegments,
  initialSeekSeconds,
  onOpenParentVideo,
  previousTranscriptId,
  onOpenTranscript,
  highlightChunkId,
  onFilterPersona,
  canManagePersona,
  onManagePersona,
  canRequestPersonaEdit,
  onRequestPersonaEdit,
}) => {
  const mainPlayerRef = useRef<any>(null);
  const secondaryPlayerRef = useRef<any>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [activeSegmentIndex, setActiveSegmentIndex] = useState<number | null>(null);
  const [summarizing, setSummarizing] = useState(false);
  const [docState, setDocState] = useState(document);
  const currentDocument = docState || document;
  const audioInputRef = useRef<HTMLInputElement | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const [presentationAudio, setPresentationAudio] = useState<{ audio_url: string; alignment: any[]; duration?: number } | null>(null);
  const [presentationAudioLoading, setPresentationAudioLoading] = useState(false);
  const [audioUploadLoading, setAudioUploadLoading] = useState(false);
  const [audioCurrentTime, setAudioCurrentTime] = useState(0);
  const [activeAudioSlide, setActiveAudioSlide] = useState<number | null>(null);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [audioDurationState, setAudioDurationState] = useState<number | null>(null);
  const queryClient = useQueryClient();
  const isArxivPaper =
    (currentDocument as any)?.source?.source_type === 'arxiv' ||
    Boolean((currentDocument as any)?.extra_metadata?.paper_metadata?.arxiv_id) ||
    Boolean((currentDocument as any)?.extra_metadata?.doi) ||
    Boolean((currentDocument as any)?.extra_metadata?.primary_category);
  const paperMeta = (currentDocument as any)?.extra_metadata?.paper_metadata as any | undefined;
  const arxivId = paperMeta?.arxiv_id || (currentDocument as any)?.extra_metadata?.arxiv_id || undefined;
  const doi = paperMeta?.doi || (currentDocument as any)?.extra_metadata?.doi || undefined;
  const venue = paperMeta?.venue || undefined;
  const publisher = paperMeta?.publisher || undefined;
  const year = paperMeta?.year || undefined;
  const keywords = Array.isArray(paperMeta?.keywords) ? (paperMeta?.keywords as string[]) : [];
  const bibtex = typeof paperMeta?.bibtex === 'string' ? paperMeta.bibtex : undefined;
  const authorAffiliations = Array.isArray(paperMeta?.author_affiliations) ? (paperMeta?.author_affiliations as any[]) : [];

  const { data: relatedDocsData, isLoading: relatedDocsLoading } = useQuery(
    ['relatedDocs', currentDocument.id],
    () => apiClient.getRelatedDocuments(currentDocument.id, 8),
    { enabled: Boolean(currentDocument?.id) && isArxivPaper, staleTime: 30000 }
  );
  const relatedDocs = (relatedDocsData?.items || []) as Array<{
    document_id: string;
    title?: string | null;
    score?: number;
    kg_overlap?: number;
    vector_score?: number;
    common_entities?: string[];
    best_chunk_id?: string | null;
  }>;

  // Poll for summary updates while modal is open if no summary yet or summarizing
  React.useEffect(() => {
    let timer: any;
    let cancelled = false;
    const poll = async () => {
      try {
        const updated = await apiClient.getDocument(document.id);
        if (!cancelled) {
          setDocState(updated);
          // stop polling if summary is available
          if (!updated.summary) {
            timer = setTimeout(poll, 5000);
          }
        }
      } catch {
        if (!cancelled) timer = setTimeout(poll, 8000);
      }
    };
    if (!document.summary) {
      poll();
    }
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [document.id, document.summary]);

  // Scroll to highlighted chunk when provided.
  React.useEffect(() => {
    if (!highlightChunkId) return;
    const timer = setTimeout(() => {
      const el = window.document.getElementById(`chunk-card-${highlightChunkId}`);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 250);
    return () => clearTimeout(timer);
  }, [highlightChunkId, currentDocument?.id, (currentDocument as any)?.chunks?.length]);
  // Track only basic state; player handles loading internally
  
  // Check if document is video/audio
  const isVideoAudio = (doc: KnowledgeDocument): boolean => {
    if (doc.file_type) {
      return doc.file_type.startsWith('video/') || doc.file_type.startsWith('audio/');
    }
    const ext = doc.title?.toLowerCase().split('.').pop() || '';
    return ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
      .some(e => ext === e.substring(1));
  };
  
  const isMediaFile = isVideoAudio(document);
  const presentationMeta = (currentDocument.extra_metadata?.presentation || (currentDocument.extra_metadata as any)?.ppt) as any;
  const audioAlignments = React.useMemo(() => {
    if (presentationAudio?.alignment) return presentationAudio.alignment;
    if (presentationMeta?.audio_track?.alignment) return presentationMeta.audio_track.alignment;
    return [];
  }, [presentationAudio?.alignment, presentationMeta?.audio_track?.alignment]);
  const alignmentMap = React.useMemo(() => {
    const map = new Map<number, any>();
    (audioAlignments || []).forEach((entry: any) => {
      const rawIdx = entry.slide_index ?? entry.index;
      const idx = typeof rawIdx === 'number' ? rawIdx : Number(rawIdx);
      if (!Number.isNaN(idx)) {
        map.set(idx, entry);
      }
    });
    return map;
  }, [audioAlignments]);
  const hasPresentationSlides = Array.isArray(presentationMeta?.slides) && presentationMeta.slides.length > 0;
  const canSeekSlides = !!(presentationAudio?.audio_url && audioAlignments.length);
  const narrationTranscriptId = presentationMeta?.audio_track?.transcript_document_id as string | undefined;
  const resolvedAudioDuration = audioDurationState ?? presentationAudio?.duration ?? presentationMeta?.audio_track?.duration ?? null;
  const ownerPersona = currentDocument.owner_persona;
  const personaDetections = (currentDocument.persona_detections || []) as DocumentPersonaDetection[];
  const speakerSummary = useMemo(() => {
    const summaryMap = new Map<string, { persona: Persona; count: number }>();
    personaDetections
      .filter(det => det.role === 'speaker' && det.persona)
      .forEach((det) => {
        const existing = summaryMap.get(det.persona.id);
        if (existing) {
          existing.count += 1;
        } else {
          summaryMap.set(det.persona.id, { persona: det.persona, count: 1 });
        }
      });
    return Array.from(summaryMap.values());
  }, [personaDetections]);
  const diarizedPersonaByName = useMemo(() => {
    const map = new Map<string, Persona>();
    personaDetections
      .filter(det => det.role === 'speaker' && det.persona?.name)
      .forEach(det => {
        map.set(det.persona.name, det.persona);
      });
    return map;
  }, [personaDetections]);

  const handleAttachAudio = async (file: File) => {
    if (!file) return;
    try {
      setAudioUploadLoading(true);
      const response = await apiClient.attachPresentationAudio(currentDocument.id, file);
      setPresentationAudio({
        audio_url: response.audio_url,
        alignment: response.alignment || [],
        duration: response.duration,
      });
      setDocState((prev) => {
        const base = prev || currentDocument;
        const updatedPresentation = {
          ...(base.extra_metadata?.presentation || {}),
          audio_track: response.audio_track,
        };
        return {
          ...base,
          extra_metadata: {
            ...(base.extra_metadata || {}),
            presentation: updatedPresentation,
          },
        };
      });
      queryClient.invalidateQueries('documents');
      toast.success('Audio narration synced');
    } catch (err: any) {
      const message = err?.response?.data?.detail || err?.message || 'Failed to attach audio narration';
      toast.error(message);
    } finally {
      setAudioUploadLoading(false);
    }
  };

  const handleAudioInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleAttachAudio(file);
    }
    event.target.value = '';
  };

  const formatClock = (value?: number) => {
    if (value === undefined || value === null || isNaN(value)) {
      return '00:00';
    }
    const totalSeconds = Math.max(0, Math.floor(value));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    if (hours > 0) {
      return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${seconds
        .toString()
        .padStart(2, '0')}`;
    }
    return `${mins.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  useEffect(() => {
    let active = true;
    if (presentationMeta?.audio_track) {
      setPresentationAudioLoading(true);
      apiClient
        .getPresentationAudio(currentDocument.id)
        .then((data) => {
          if (!active) return;
          setPresentationAudio({
            audio_url: data.audio_url,
            alignment: data.alignment || [],
            duration: data.duration,
          });
        })
        .catch(() => {
          if (active) {
            setPresentationAudio(null);
          }
        })
        .finally(() => {
          if (active) setPresentationAudioLoading(false);
        });
    } else {
      setPresentationAudio(null);
    }
    return () => {
      active = false;
    };
  }, [currentDocument.id, presentationMeta?.audio_track?.object_path]);
  useEffect(() => {
    if (presentationAudio?.duration) {
      setAudioDurationState(presentationAudio.duration);
    } else if (presentationMeta?.audio_track?.duration) {
      setAudioDurationState(presentationMeta.audio_track.duration);
    } else {
      setAudioDurationState(null);
    }
  }, [presentationAudio?.duration, presentationMeta?.audio_track?.duration]);
  useEffect(() => {
    const audioEl = audioPlayerRef.current;
    if (!audioEl) return;
    const handleTimeUpdate = () => setAudioCurrentTime(audioEl.currentTime || 0);
    const handlePlay = () => setIsAudioPlaying(true);
    const handlePause = () => setIsAudioPlaying(false);
    const handleLoaded = () => {
      if (!Number.isNaN(audioEl.duration)) {
        setAudioDurationState(audioEl.duration);
      }
    };
    const handleEnded = () => setIsAudioPlaying(false);
    audioEl.addEventListener('timeupdate', handleTimeUpdate);
    audioEl.addEventListener('play', handlePlay);
    audioEl.addEventListener('pause', handlePause);
    audioEl.addEventListener('loadedmetadata', handleLoaded);
    audioEl.addEventListener('ended', handleEnded);
    return () => {
      audioEl.removeEventListener('timeupdate', handleTimeUpdate);
      audioEl.removeEventListener('play', handlePlay);
      audioEl.removeEventListener('pause', handlePause);
      audioEl.removeEventListener('loadedmetadata', handleLoaded);
      audioEl.removeEventListener('ended', handleEnded);
    };
  }, [presentationAudio?.audio_url, currentDocument.id]);
  useEffect(() => {
    setAudioCurrentTime(0);
    setActiveAudioSlide(null);
  }, [currentDocument.id, presentationAudio?.audio_url]);
  useEffect(() => {
    if (!audioAlignments || audioAlignments.length === 0) {
      setActiveAudioSlide(null);
      return;
    }
    const time = audioCurrentTime;
    if (typeof time !== 'number' || Number.isNaN(time)) {
      setActiveAudioSlide(null);
      return;
    }
    const normalized = Math.max(0, time);
    const matched = audioAlignments.find((alignment: any, index: number) => {
      const rawStart = alignment.start ?? 0;
      const rawEnd = alignment.end ?? rawStart;
      const start = typeof rawStart === 'number' ? rawStart : Number(rawStart) || 0;
      let end = typeof rawEnd === 'number' ? rawEnd : Number(rawEnd);
      if (!Number.isFinite(end) || end < start) {
        const nextRaw = audioAlignments[index + 1]?.start;
        const nextStart = typeof nextRaw === 'number' ? nextRaw : Number(nextRaw);
        if (Number.isFinite(nextStart)) {
          end = Math.max(start, nextStart);
        } else {
          end = start;
        }
      }
      return normalized >= start && normalized < end + 0.35;
    });
    const idxRaw = matched?.slide_index ?? matched?.index;
    const idx = typeof idxRaw === 'number' ? idxRaw : Number(idxRaw);
    if (matched && !Number.isNaN(idx)) {
      setActiveAudioSlide(idx);
    } else {
      setActiveAudioSlide(null);
    }
  }, [audioAlignments, audioCurrentTime]);

  const handleSlideSeek = (slideKey: number) => {
    if (!canSeekSlides) return;
    const alignment = alignmentMap.get(slideKey);
    if (!alignment || !audioPlayerRef.current) return;
    const rawStart = alignment.start ?? 0;
    const start = typeof rawStart === 'number' ? rawStart : Number(rawStart);
    if (Number.isNaN(start)) return;
    try {
      audioPlayerRef.current.currentTime = Math.max(0, start);
      const playPromise = audioPlayerRef.current.play?.();
      if (playPromise && typeof (playPromise as Promise<void>).then === 'function') {
        (playPromise as Promise<void>).catch(() => {});
      }
    } catch {
      // ignore playback errors (e.g., autoplay restrictions)
    }
  };
  
  // Load video URL when modal opens - use streaming URL for better performance
  useEffect(() => {
    if (isMediaFile && !videoUrl && !document.extra_metadata?.is_transcoding) {
      setIsVideoLoading(true);
      // Use streaming URL - native player supports HTTP range requests for streaming
      // This allows the video to start playing while still downloading
      const token = localStorage.getItem('access_token');
      
      if (!token) {
        console.error('No authentication token found');
        toast.error('Please log in to view videos');
        setIsVideoLoading(false);
        return;
      }
      
      // Use video streamer for video/audio files
      apiClient.getDocumentDownloadUrl(document.id, true, true)
        .then(url => {
          // Add token as query parameter for authentication
          const separator = url.includes('?') ? '&' : '?';
          const streamingUrl = `${url}${separator}token=${encodeURIComponent(token)}`;
          console.log('Video URL with token:', streamingUrl);
          console.log('Token length:', token.length);
          setVideoUrl(streamingUrl);
          setIsVideoLoading(false);
        })
        .catch(error => {
          console.error('Failed to get video URL:', error);
          toast.error(`Failed to load video: ${error.message || 'Unknown error'}`);
          setIsVideoLoading(false);
        });
    }
  }, [isMediaFile, document.id, videoUrl, document.extra_metadata?.is_transcoding]);

  // Initialize Video.js players when URL is ready
  useEffect(() => {
    if (!videoUrl || document.extra_metadata?.is_transcoding) return;
    
    const players: any[] = [];
    
    // Wait for DOM elements to be available
    const timer = setTimeout(() => {
      const mainEl = window.document.getElementById('vjs-player-main') as HTMLVideoElement | null;
      const secEl = window.document.getElementById('vjs-player-secondary') as HTMLVideoElement | null;
      const type = (document.file_type?.startsWith('audio/') ? document.file_type : (document.file_type || 'video/mp4')) as string;
      
      const setup = (el: HTMLVideoElement | null) => {
        if (!el) return;
        try {
          // Check if player already exists
          const existingPlayer = (videojs as any).getPlayer(el);
          if (existingPlayer) {
            existingPlayer.dispose();
          }
          
          // Ensure the URL includes the token query parameter
          const urlWithToken = videoUrl; // videoUrl already has token from useEffect
          console.log('Initializing Video.js with URL:', urlWithToken);
          
          const player = videojs(el, {
            controls: true,
            preload: 'metadata',
            autoplay: false,
            fluid: true,
            responsive: true,
            sources: [{ src: urlWithToken, type }],
          });
          
          // Intercept video element's src changes to ensure token is always present
          player.ready(() => {
            const videoEl = player.el().querySelector('video') as HTMLVideoElement;
            if (videoEl) {
              // Ensure src has token
              const currentSrc = videoEl.src || videoEl.currentSrc;
              if (currentSrc && !currentSrc.includes('token=')) {
                const token = localStorage.getItem('access_token');
                if (token) {
                  const separator = currentSrc.includes('?') ? '&' : '?';
                  const newSrc = `${currentSrc}${separator}token=${encodeURIComponent(token)}`;
                  videoEl.src = newSrc;
                  console.log('Added token to video element src:', newSrc);
                }
              }
              
              // Monitor for src changes (e.g., during range requests)
              const observer = new MutationObserver(() => {
                const src = videoEl.src || videoEl.currentSrc;
                if (src && !src.includes('token=')) {
                  const token = localStorage.getItem('access_token');
                  if (token) {
                    const separator = src.includes('?') ? '&' : '?';
                    const newSrc = `${src}${separator}token=${encodeURIComponent(token)}`;
                    if (videoEl.src !== newSrc) {
                      videoEl.src = newSrc;
                      console.log('Added token to video element src (after change):', newSrc);
                    }
                  }
                }
              });
              
              observer.observe(videoEl, {
                attributes: true,
                attributeFilter: ['src'],
              });
              
              // Also listen to loadstart to catch range requests
              videoEl.addEventListener('loadstart', () => {
                const src = videoEl.src || videoEl.currentSrc;
                if (src && !src.includes('token=')) {
                  const token = localStorage.getItem('access_token');
                  if (token) {
                    const separator = src.includes('?') ? '&' : '?';
                    const newSrc = `${src}${separator}token=${encodeURIComponent(token)}`;
                    if (videoEl.src !== newSrc) {
                      videoEl.src = newSrc;
                      console.log('Added token to video element src (loadstart):', newSrc);
                    }
                  }
                }
              });
            }
          });
          
          player.on('timeupdate', () => {
            try { 
              const time = player.currentTime() || 0;
              setCurrentTime(time);
              handleProgress({ playedSeconds: time });
            } catch {}
          });
          
          player.on('loadedmetadata', () => {
            try {
              // Seek to preview or requested time
              const target = (typeof initialSeekSeconds === 'number' && initialSeekSeconds >= 0) ? initialSeekSeconds : 0.1;
              player.currentTime(target);
              if (typeof initialSeekSeconds === 'number' && initialSeekSeconds >= 0) {
                if (typeof player.play === 'function') player.play();
              } else {
                player.pause();
              }
            } catch {}
          });
          
          player.on('error', (error: any) => {
            console.error('Video.js player error:', error);
            const playerError = player.error();
            if (playerError) {
              console.error('Player error code:', playerError.code);
              console.error('Player error message:', playerError.message);
            }
          });
          
          players.push(player);
          if (el.id === 'vjs-player-main') mainPlayerRef.current = player;
          if (el.id === 'vjs-player-secondary') secondaryPlayerRef.current = player;
        } catch (e) {
          console.warn('Video.js init error', e);
        }
      };
      
      setup(mainEl);
      setup(secEl);
    }, 100);
    
    return () => {
      clearTimeout(timer);
      players.forEach(p => { 
        try { 
          if (p && typeof p.dispose === 'function') {
            p.dispose(); 
          }
        } catch {} 
      });
    };
  }, [videoUrl, document.extra_metadata?.is_transcoding, initialSeekSeconds]);
  
  const handleDownload = async () => {
    try {
      const downloadUrl = await apiClient.getDocumentDownloadUrl(document.id, true);
      // Use fetch to download with authentication
      const response = await fetch(downloadUrl, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token') || ''}`,
        },
        credentials: 'include',
      });
      
      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }
      
      // Get filename from Content-Disposition header
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = getDownloadFilename(document) || `document_${document.id}`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = window.document.createElement('a');
      link.href = url;
      link.download = filename;
      window.document.body.appendChild(link);
      link.click();
      window.document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      toast.error('Failed to generate download URL');
    }
  };

  const jumpToTimestamp = (seconds: number) => {
    try {
      const player = (mainPlayerRef.current || secondaryPlayerRef.current) as any;
      if (player && typeof player.currentTime === 'function') {
        player.currentTime(seconds);
        if (typeof player.play === 'function') {
          player.play();
        }
      }
    } catch {}
  };
  
  // Get segments from metadata (prefer diarized sentence segments)
  const meta = document.extra_metadata?.transcription_metadata || {};
  const diarizedSentences = meta.sentence_segments || [];
  const segments = meta.segments || [];
  const liveSegs = liveSegments || [];
  const playbackSegments: Array<any> = (diarizedSentences.length > 0 ? diarizedSentences : segments);
  
  // Update active transcript item based on current playback time
  useEffect(() => {
    const items = (playbackSegments && playbackSegments.length > 0) ? playbackSegments : liveSegs;
    if (!items || items.length === 0) return;

    let foundIndex = -1;
    for (let i = 0; i < items.length; i++) {
      const s = items[i];
      const start = s.start || 0;
      const end = (s.end != null) ? s.end : ((i + 1 < items.length && items[i + 1].start != null) ? items[i + 1].start : start + 1);
      if (currentTime >= start && currentTime < end) {
        foundIndex = i;
        break;
      }
    }

    if (foundIndex !== -1 && foundIndex !== activeSegmentIndex) {
      setActiveSegmentIndex(foundIndex);
      if (transcriptRef.current) {
        const segmentElement = transcriptRef.current.children[foundIndex] as HTMLElement;
        if (segmentElement) segmentElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    } else if (foundIndex === -1) {
      setActiveSegmentIndex(null);
    }
  }, [currentTime, playbackSegments, liveSegs, activeSegmentIndex]);
  
  const handleProgress = (state: any) => {
    if (state && typeof state === 'object' && 'playedSeconds' in state) {
      setCurrentTime(state.playedSeconds);
    }
  };
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-6xl max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">{getDisplayTitle(document)}</h2>
          <div className="flex items-center gap-2">
            {isMediaFile && document.extra_metadata?.transcript_document_id && (
              <Button
                variant="ghost"
                size="sm"
                icon={<FileText className="w-4 h-4" />}
                onClick={() => {
                  const tId = document.extra_metadata?.transcript_document_id as string;
                  if (onOpenTranscript) onOpenTranscript(tId);
                }}
              >
                View Transcript
              </Button>
            )}
            {isMediaFile && previousTranscriptId && (
              <Button
                variant="ghost"
                size="sm"
                icon={<FileText className="w-4 h-4" />}
                onClick={() => {
                  if (onOpenTranscript && previousTranscriptId) onOpenTranscript(previousTranscriptId);
                }}
              >
                Back to Transcript
              </Button>
            )}
            {document.file_path && (
              <Button
                variant="ghost"
                size="sm"
                icon={<Download className="w-4 h-4" />}
                onClick={handleDownload}
                disabled={document.extra_metadata?.is_transcoding === true}
              >
                Download
              </Button>
            )}
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        <div className="space-y-4">
          {isArxivPaper && (
            <div className="bg-white border rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium text-gray-900">Related papers</div>
                  <div className="text-xs text-gray-500">KG overlap + embedding similarity</div>
                </div>
                <div className="text-xs text-gray-500">{relatedDocsLoading ? 'Loading…' : ''}</div>
              </div>
              <div className="mt-3 space-y-2">
                {relatedDocs.length === 0 && !relatedDocsLoading ? (
                  <div className="text-sm text-gray-600">No related papers found yet.</div>
                ) : (
                  relatedDocs.map((item) => (
                    <button
                      key={item.document_id}
                      type="button"
                      onClick={() =>
                        navigate('/documents', {
                          state: {
                            openDocId: item.document_id,
                            highlightChunkId: item.best_chunk_id || undefined,
                          },
                        })
                      }
                      className="w-full text-left border rounded-lg p-3 hover:bg-gray-50"
                      title="Open"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0">
                          <div className="font-medium text-gray-900 truncate">{item.title || item.document_id}</div>
                          {item.common_entities?.length ? (
                            <div className="text-xs text-gray-600 mt-1 truncate">
                              Shared: {item.common_entities.slice(0, 6).join(', ')}
                              {item.common_entities.length > 6 ? '…' : ''}
                            </div>
                          ) : null}
                        </div>
                        <div className="text-xs text-gray-500 shrink-0">
                          {typeof item.score === 'number' ? `${Math.round(item.score * 100)}%` : ''}
                          {typeof item.kg_overlap === 'number' ? ` • KG ${item.kg_overlap}` : ''}
                        </div>
                      </div>
                    </button>
                  ))
                )}
              </div>
            </div>
          )}
          {isArxivPaper && (
            <div className="bg-white border rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium text-gray-900">Paper metadata</div>
                  <div className="text-xs text-gray-500">DOI/venue/keywords (from enrichment)</div>
                </div>
              </div>
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                <div className="border rounded-lg p-3">
                  <div className="text-xs text-gray-500">arXiv</div>
                  {arxivId ? (
                    <a
                      className="text-primary-700 hover:text-primary-800 break-all"
                      href={`https://arxiv.org/abs/${encodeURIComponent(arxivId)}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {arxivId}
                    </a>
                  ) : (
                    <div className="text-gray-600">—</div>
                  )}
                </div>
                <div className="border rounded-lg p-3">
                  <div className="text-xs text-gray-500">DOI</div>
                  {doi ? (
                    <a
                      className="text-primary-700 hover:text-primary-800 break-all"
                      href={`https://doi.org/${encodeURIComponent(doi)}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {doi}
                    </a>
                  ) : (
                    <div className="text-gray-600">—</div>
                  )}
                </div>
                <div className="border rounded-lg p-3">
                  <div className="text-xs text-gray-500">Venue</div>
                  <div className="text-gray-900">{venue || '—'}</div>
                  {(publisher || year) && (
                    <div className="text-xs text-gray-500 mt-1">
                      {publisher ? publisher : ''}
                      {publisher && year ? ' • ' : ''}
                      {year ? String(year) : ''}
                    </div>
                  )}
                </div>
                <div className="border rounded-lg p-3">
                  <div className="text-xs text-gray-500">Keywords</div>
                  {keywords.length ? (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {keywords.slice(0, 12).map((k) => (
                        <span key={k} className="text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-700">
                          {k}
                        </span>
                      ))}
                      {keywords.length > 12 ? <span className="text-xs text-gray-500">+{keywords.length - 12}</span> : null}
                    </div>
                  ) : (
                    <div className="text-gray-600">—</div>
                  )}
                </div>
                <div className="border rounded-lg p-3 md:col-span-2">
                  <div className="text-xs text-gray-500">Author affiliations</div>
                  {authorAffiliations.length ? (
                    <div className="mt-1 space-y-2">
                      {authorAffiliations.slice(0, 8).map((a, idx) => (
                        <div key={`${a?.name || 'author'}-${idx}`} className="text-sm">
                          <div className="font-medium text-gray-900">{a?.name || 'Unknown author'}</div>
                          {Array.isArray(a?.affiliations) && a.affiliations.length ? (
                            <div className="text-xs text-gray-600">
                              {a.affiliations.slice(0, 3).join(' • ')}
                              {a.affiliations.length > 3 ? '…' : ''}
                            </div>
                          ) : (
                            <div className="text-xs text-gray-500">—</div>
                          )}
                        </div>
                      ))}
                      {authorAffiliations.length > 8 ? (
                        <div className="text-xs text-gray-500">+{authorAffiliations.length - 8} more</div>
                      ) : null}
                    </div>
                  ) : (
                    <div className="text-gray-600">—</div>
                  )}
                </div>
              </div>

              {bibtex ? (
                <details className="mt-3">
                  <summary className="cursor-pointer text-sm text-primary-700 hover:text-primary-800">
                    BibTeX
                  </summary>
                  <div className="mt-2 border rounded-lg p-3 bg-gray-50">
                    <div className="flex items-center justify-end mb-2">
                      <button
                        type="button"
                        className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded border border-gray-300 hover:bg-white"
                        onClick={async () => {
                          try {
                            await navigator.clipboard.writeText(bibtex);
                            toast.success('BibTeX copied');
                          } catch {
                            toast.error('Failed to copy');
                          }
                        }}
                      >
                        <Copy className="w-3 h-3" /> Copy
                      </button>
                    </div>
                    <pre className="text-xs whitespace-pre-wrap break-words">{bibtex}</pre>
                  </div>
                </details>
              ) : null}
            </div>
          )}
          {/* Video/Audio Player with YouTube-style Layout */}
          {document.extra_metadata?.is_transcoding ? (
            <div className="p-6 bg-yellow-50 border border-yellow-200 rounded">
              <h3 className="font-medium text-yellow-900 mb-2">Converting to MP4</h3>
              <p className="text-yellow-800">This video is being converted to MP4 for playback. Viewing and downloading are temporarily disabled. Please close this dialog and check back shortly.</p>
            </div>
          ) : isMediaFile && segments.length > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Video Player - Takes 2/3 width on large screens */}
              <div className="lg:col-span-2">
                <h3 className="font-medium text-gray-900 mb-2">
                  {document.file_type?.startsWith('video/') ? 'Video Player' : 'Audio Player'}
                </h3>
                <div className="bg-black rounded-lg overflow-hidden">
                  {isVideoLoading ? (
                    <div className="aspect-video flex items-center justify-center bg-black">
                      <div className="text-gray-800">Loading player...</div>
                    </div>
                  ) : videoUrl ? (
                    <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9' }}>
                      <video id="vjs-player-main" className="video-js vjs-default-skin w-full h-full" playsInline preload="metadata" />
                    </div>
                  ) : (
                    <div className="aspect-video flex items-center justify-center bg-black">
                      <div className="text-gray-800">Failed to load media</div>
                    </div>
                  )}
                </div>
              </div>
              
          {/* Transcript Sidebar - Takes 1/3 width on large screens */}
          <div className="lg:col-span-1">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium text-gray-900">
                Transcript ({(playbackSegments.length > 0 ? playbackSegments.length : liveSegs.length)} segments)
              </h3>
              {document.extra_metadata?.transcript_document_id && (
                <button
                  className="text-sm text-primary-700 hover:text-primary-800"
                  onClick={() => {
                    const tId = document.extra_metadata?.transcript_document_id as string;
                    if (tId && onOpenTranscript) onOpenTranscript(tId);
                  }}
                >
                  View full transcript
                </button>
              )}
            </div>
            <div 
              ref={transcriptRef}
              className="bg-gray-50 rounded-lg p-4 max-h-[600px] overflow-y-auto"
              style={{ scrollBehavior: 'smooth' }}
            >
              <div className="space-y-2">
                {(playbackSegments.length > 0 ? playbackSegments : liveSegs).map((segment: any, index: number) => {
                  const formatTime = (seconds: number) => {
                    const hours = Math.floor(seconds / 3600);
                    const minutes = Math.floor((seconds % 3600) / 60);
                    const secs = Math.floor(seconds % 60);
                    if (hours > 0) {
                      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                    }
                    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                  };
                  
                  const startTime = segment.start || 0;
                  const endTime = (segment.end != null) ? segment.end : undefined;
                  const text = segment.text || '';
                  const speaker = segment.speaker;
                  const personaForSegment = speaker ? diarizedPersonaByName.get(speaker) : undefined;
                  const isActive = activeSegmentIndex === index;
                  
                  return (
                    <div 
                      key={segment.id || index} 
                          className={`p-3 rounded-lg border transition-all cursor-pointer ${
                            isActive 
                              ? 'bg-primary-50 border-primary-300 shadow-md' 
                              : 'bg-white border-gray-200 hover:border-primary-300 hover:shadow-sm'
                          }`}
                          onClick={() => {
                            jumpToTimestamp(startTime);
                          }}
                        >
                          <div className="flex items-center space-x-2 mb-1">
                            <span className={`text-xs font-mono font-medium px-2 py-1 rounded ${
                              isActive 
                                ? 'text-primary-700 bg-primary-100' 
                                : 'text-primary-600 bg-primary-50'
                            }`}>
                            {formatTime(startTime)}{endTime != null ? ` - ${formatTime(endTime)}` : ''}
                            </span>
                            {personaForSegment ? (
                              <button
                                type="button"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  onFilterPersona?.(personaForSegment.id, 'speaker');
                                }}
                                className="text-xs font-medium text-primary-700 bg-primary-50 px-2 py-1 rounded hover:bg-primary-100"
                              >
                                {personaForSegment.name}
                              </button>
                            ) : speaker && (
                              <span className="text-xs font-medium text-gray-600 bg-gray-100 px-2 py-1 rounded">
                                {speaker}
                              </span>
                            )}
                          </div>
                          <p className={`text-sm mt-1 ${
                            isActive ? 'text-gray-900 font-medium' : 'text-gray-700'
                          }`}>
                            {text}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          ) : isMediaFile ? (
            <div>
              <h3 className="font-medium text-gray-900 mb-2">
                {document.file_type?.startsWith('video/') ? 'Video Player' : 'Audio Player'}
              </h3>
              <div className="bg-black rounded-lg overflow-hidden">
                  {isVideoLoading ? (
                    <div className="aspect-video flex items-center justify-center bg-black">
                      <div className="text-gray-800">Loading player...</div>
                    </div>
                  ) : videoUrl ? (
                  <video id="vjs-player-secondary" className="video-js vjs-default-skin w-full" playsInline preload="metadata" />
                  ) : (
                    <div className="aspect-video flex items-center justify-center bg-black">
                      <div className="text-gray-800">Failed to load media</div>
                    </div>
                  )}
                </div>
            </div>
          ) : null}

          {/* Metadata */}
          <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
            <div>
              <span className="font-medium text-gray-700">Source:</span>
              <span className="ml-2">{document.source.name}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Type:</span>
              <span className="ml-2">{document.file_type || 'Unknown'}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Size:</span>
              <span className="ml-2">{formatFileSize(document.file_size)}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Status:</span>
              <span className="ml-2">{document.is_processed ? 'Processed' : 'Processing'}</span>
            </div>
            {ownerPersona && (
              <div className="col-span-2 flex flex-col gap-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium text-gray-700 flex items-center gap-1">
                    <UserCircle2 className="w-4 h-4 text-primary-600" />
                    Owner persona:
                  </span>
                  <span className="text-gray-900 font-medium">{ownerPersona.name}</span>
                  <button
                    type="button"
                    className="text-xs text-primary-700 hover:text-primary-900 underline-offset-2 hover:underline"
                    onClick={() => onFilterPersona?.(ownerPersona.id, 'owner')}
                  >
                    Filter by owner
                  </button>
                  {canManagePersona && onManagePersona && (
                    <button
                      type="button"
                      className="text-xs text-primary-700 hover:text-primary-900 underline-offset-2 hover:underline"
                      onClick={() => onManagePersona(ownerPersona.id)}
                    >
                      Manage persona
                    </button>
                  )}
                  {!canManagePersona && canRequestPersonaEdit && onRequestPersonaEdit && (
                    <button
                      type="button"
                      className="text-xs text-primary-700 hover:text-primary-900 underline-offset-2 hover:underline flex items-center gap-1"
                      onClick={() => onRequestPersonaEdit(ownerPersona, document)}
                    >
                      <MessageSquare className="w-3 h-3" />
                      Suggest change
                    </button>
                  )}
                </div>
                <PersonaMetadata persona={ownerPersona} />
              </div>
            )}
          </div>

          {speakerSummary.length > 0 && (
            <div className="border border-gray-200 rounded-lg p-4 bg-white">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-medium text-gray-900">Detected speakers</h3>
                <span className="text-xs text-gray-500">
                  {speakerSummary.reduce((acc, item) => acc + item.count, 0)} segments
                </span>
              </div>
              <div className="flex flex-wrap gap-3">
                {speakerSummary.map(({ persona, count }) => (
                  <div key={persona.id} className="flex flex-col gap-1">
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => onFilterPersona?.(persona.id, 'speaker')}
                        className="inline-flex items-center gap-1 rounded-full border border-primary-200 bg-primary-50 px-3 py-1 text-sm text-primary-800 hover:bg-primary-100"
                      >
                        <UserCircle2 className="w-4 h-4" />
                        <span>{persona.name}</span>
                        <span className="text-xs text-primary-600">({count})</span>
                      </button>
                      {canManagePersona && onManagePersona && (
                        <button
                          type="button"
                          className="text-xs text-primary-700 hover:text-primary-900 underline-offset-2 hover:underline"
                          onClick={() => onManagePersona(persona.id)}
                        >
                          Manage persona
                        </button>
                      )}
                      {!canManagePersona && canRequestPersonaEdit && onRequestPersonaEdit && (
                        <button
                          type="button"
                          className="text-xs text-primary-700 hover:text-primary-900 underline-offset-2 hover:underline flex items-center gap-1"
                          onClick={() => onRequestPersonaEdit(persona, document)}
                        >
                          <MessageSquare className="w-3 h-3" />
                          Suggest change
                        </button>
                      )}
                    </div>
                    <PersonaMetadata persona={persona} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {hasPresentationSlides && (
            <div className="space-y-4">
              <div className="border border-gray-200 rounded-lg p-4 bg-white">
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div>
                    <h3 className="font-medium text-gray-900">Narration Audio</h3>
                    <p className="text-sm text-gray-600">
                      Attach a narration track to sync with slides. Slides highlight automatically while audio plays.
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <input
                      ref={audioInputRef}
                      type="file"
                      accept=".mp3,.wav,.m4a,.flac,.ogg,.aac,audio/*"
                      className="hidden"
                      onChange={handleAudioInputChange}
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      icon={<Upload className="w-4 h-4" />}
                      onClick={() => audioInputRef.current?.click()}
                      loading={audioUploadLoading}
                      disabled={audioUploadLoading}
                    >
                      {presentationMeta?.audio_track ? 'Replace narration' : 'Upload narration'}
                    </Button>
                    {narrationTranscriptId && onOpenTranscript && (
                      <Button
                        variant="ghost"
                        size="sm"
                        icon={<FileText className="w-4 h-4" />}
                        onClick={() => onOpenTranscript(narrationTranscriptId)}
                      >
                        View transcript
                      </Button>
                    )}
                  </div>
                </div>
                {presentationAudioLoading ? (
                  <div className="mt-4 flex items-center space-x-2 text-sm text-gray-600">
                    <Loader2 className="w-4 h-4 animate-spin text-primary-600" />
                    <span>Loading audio track…</span>
                  </div>
                ) : presentationAudio?.audio_url ? (
                  <div className="mt-4 space-y-2">
                    <audio
                      ref={audioPlayerRef}
                      src={presentationAudio.audio_url}
                      controls
                      preload="metadata"
                      className="w-full"
                    />
                    <div className="flex flex-wrap items-center gap-3 text-xs text-gray-600 justify-between">
                      <span>
                        {formatClock(audioCurrentTime)} / {formatClock(resolvedAudioDuration || undefined)}
                      </span>
                      <span>Status: {isAudioPlaying ? 'Playing' : 'Paused'}</span>
                      {presentationMeta?.audio_track?.language && (
                        <span className="uppercase">
                          Language: {presentationMeta.audio_track.language}
                        </span>
                      )}
                      {presentationMeta?.audio_track?.file_name && (
                        <span className="truncate">
                          File: {presentationMeta.audio_track.file_name}
                        </span>
                      )}
                      {presentationMeta?.audio_track?.created_at && (
                        <span>
                          Uploaded {formatDistanceToNow(new Date(presentationMeta.audio_track.created_at))} ago
                        </span>
                      )}
                    </div>
                  </div>
                ) : (
                  <p className="mt-4 text-sm text-gray-600">
                    No narration attached yet. Upload an audio file to sync it with these slides.
                  </p>
                )}
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-gray-900">Slides & Comments</h3>
                  <span className="text-sm text-gray-500">
                    {presentationMeta.slide_count || presentationMeta.slides.length} slides
                  </span>
                </div>
                <div className="space-y-3 max-h-96 overflow-auto">
                  {presentationMeta.slides.map((slide: any, index: number) => {
                    const rawIdx = slide.index ?? slide.slide_index ?? slide.number;
                    const numericIdx = typeof rawIdx === 'number' ? rawIdx : Number(rawIdx);
                    const slideKey = !Number.isNaN(numericIdx) ? numericIdx : index + 1;
                    const alignment = alignmentMap.get(slideKey);
                    const slideIsActive = canSeekSlides && activeAudioSlide === slideKey;
                    const showJump = Boolean(alignment) && canSeekSlides;
                    const alignStartRaw = alignment?.start ?? 0;
                    const alignStart = typeof alignStartRaw === 'number' ? alignStartRaw : Number(alignStartRaw) || 0;
                    const alignEndRaw = alignment?.end ?? alignStart;
                    const alignEnd = typeof alignEndRaw === 'number' ? alignEndRaw : Number(alignEndRaw);
                    const safeAlignEnd = Number.isNaN(alignEnd) ? alignStart : alignEnd;
                    return (
                      <div
                        key={`${slideKey}-${index}`}
                        className={`border rounded-lg p-3 bg-white transition-colors ${
                          slideIsActive ? 'border-primary-400 bg-primary-50 shadow-sm' : 'border-gray-200'
                        } ${showJump ? 'cursor-pointer hover:border-primary-300' : ''}`}
                        onClick={() => {
                          if (showJump) handleSlideSeek(slideKey);
                        }}
                      >
                        <div className="flex items-center justify-between text-sm text-gray-600">
                          <span className="font-medium text-gray-900 flex items-center">
                            Slide {slideKey}
                            {slideIsActive && (
                              <span className="ml-2 text-xs font-semibold text-primary-700 bg-primary-100 px-2 py-0.5 rounded-full">
                                Now playing
                              </span>
                            )}
                          </span>
                          {slide.title && (
                            <span className="ml-2 text-gray-500 truncate">{slide.title}</span>
                          )}
                        </div>
                        {slide.text && (
                          <p className="mt-2 text-sm text-gray-700 whitespace-pre-wrap">
                            {slide.text}
                          </p>
                        )}
                        {slide.notes && (
                          <div className="mt-3 text-xs text-gray-600">
                            <span className="font-semibold text-gray-700">Notes:</span>
                            <p className="text-gray-600 whitespace-pre-wrap">{slide.notes}</p>
                          </div>
                        )}
                        {slide.comments && slide.comments.length > 0 && (
                          <div className="mt-3">
                            <div className="text-xs font-semibold text-gray-700">Comments</div>
                            <div className="mt-1 space-y-2">
                              {slide.comments.map((comment: any, commentIndex: number) => (
                                <div key={commentIndex} className="text-xs text-gray-700 bg-gray-50 rounded px-2 py-1">
                                  <div className="flex items-center justify-between">
                                    <span className="font-medium text-primary-700">
                                      {comment.author || comment.author_initials || 'Comment'}
                                    </span>
                                    {comment.created_at && (
                                      <span className="text-gray-400 ml-2">
                                        {new Date(comment.created_at).toLocaleString()}
                                      </span>
                                    )}
                                  </div>
                                  <p className="text-gray-700 whitespace-pre-wrap">{comment.text}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        {alignment && (
                          <div className="mt-3 flex items-center justify-between text-xs text-gray-600">
                            <span>
                              Narration {formatClock(alignStart)} – {formatClock(safeAlignEnd)}
                            </span>
                            {showJump && (
                              <button
                                className="text-primary-600 hover:text-primary-700 font-medium"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  handleSlideSeek(slideKey);
                                }}
                              >
                                Play section
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Content (hide for transcript docs) */}
          {document.content && document.extra_metadata?.doc_type !== 'transcript' && (
            <div>
              <h3 className="font-medium text-gray-900 mb-2">Content Preview</h3>
              <div className="p-4 bg-gray-50 rounded-lg max-h-96 overflow-auto">
                <pre className="whitespace-pre-wrap text-sm text-gray-700">
                  {document.content.substring(0, 2000)}
                  {document.content.length > 2000 && '...'}
                </pre>
              </div>
            </div>
          )}

          {/* Transcription Segments (only show if not in YouTube-style layout) */}
          {!isMediaFile && (
            (document.extra_metadata?.transcription_metadata?.sentence_segments && document.extra_metadata.transcription_metadata.sentence_segments.length > 0) ||
            (document.extra_metadata?.transcription_metadata?.segments && document.extra_metadata.transcription_metadata.segments.length > 0)
           ) && (
            <div>
              <h3 className="font-medium text-gray-900 mb-2">
                Transcript with Time Codes ({(document.extra_metadata.transcription_metadata.sentence_segments || document.extra_metadata.transcription_metadata.segments).length} segments)
              </h3>
              <div className="p-4 bg-gray-50 rounded-lg max-h-96 overflow-auto">
                <div className="space-y-3">
                  {(document.extra_metadata.transcription_metadata.sentence_segments || document.extra_metadata.transcription_metadata.segments).map((segment: any, index: number) => {
                    const formatTime = (seconds: number) => {
                      const hours = Math.floor(seconds / 3600);
                      const minutes = Math.floor((seconds % 3600) / 60);
                      const secs = Math.floor(seconds % 60);
                      if (hours > 0) {
                        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                      }
                      return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                    };
                    
                  const startTime = segment.start || 0;
                  const endTime = segment.end || 0;
                  const text = segment.text || '';
                  const speaker = segment.speaker;
                  const personaForSegment = speaker ? diarizedPersonaByName.get(speaker) : undefined;
                    
                    return (
                      <div 
                        key={segment.id || index} 
                        className="p-3 bg-white rounded border border-gray-200 hover:border-primary-300 hover:shadow-sm transition-all cursor-pointer"
                        onClick={() => {
                          const parentId = document.extra_metadata?.parent_document_id as string | undefined;
                          if (parentId && onOpenParentVideo) {
                            onOpenParentVideo(parentId, startTime);
                          }
                        }}
                      >
                        <div className="flex items-start justify-between mb-1">
                          <div className="flex items-center space-x-2">
                            <span className="text-xs font-mono font-medium text-primary-600 bg-primary-50 px-2 py-1 rounded">
                              {formatTime(startTime)}{endTime ? ` - ${formatTime(endTime)}` : ''}
                            </span>
                            {personaForSegment ? (
                              <button
                                type="button"
                                className="text-xs font-medium text-primary-700 bg-primary-50 px-2 py-1 rounded hover:bg-primary-100"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  onFilterPersona?.(personaForSegment.id, 'speaker');
                                }}
                              >
                                {personaForSegment.name}
                              </button>
                            ) : speaker && (
                              <span className="text-xs font-medium text-gray-600 bg-gray-100 px-2 py-1 rounded">
                                {speaker}
                              </span>
                            )}
                          </div>
                        </div>
                        <p className="text-sm text-gray-700 mt-1">{text}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Chunks */}
          {document.chunks && document.chunks.length > 0 && (
            <div>
              <h3 className="font-medium text-gray-900 mb-2">
                Processed Chunks ({document.chunks.length})
              </h3>
              <div className="space-y-2 max-h-64 overflow-auto">
                {document.chunks.map((chunk) => (
                  <div
                    key={chunk.id}
                    id={`chunk-card-${chunk.id}`}
                    className={`p-3 rounded border ${highlightChunkId === chunk.id ? 'bg-yellow-50 border-yellow-300' : 'bg-gray-50'}`}
                  >
                    <div className="text-xs text-gray-500 mb-1">
                      Chunk {chunk.chunk_index + 1}
                    </div>
                    <p className="text-sm text-gray-700">
                      {chunk.content.substring(0, 200)}
                      {chunk.content.length > 200 && '...'}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Summary */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium text-gray-900">Summary</h3>
              <button
                className="text-sm text-primary-700 hover:text-primary-800"
                onClick={async () => {
                  try {
                    setSummarizing(true);
                    await apiClient.summarizeDocument(document.id, true);
                    toast.success('Summarization started');
                  } catch (e: any) {
                    toast.error(e?.response?.data?.detail || e?.message || 'Failed to start summarization');
                  } finally {
                    setSummarizing(false);
                  }
                }}
              >
                {summarizing ? 'Starting…' : 'Regenerate'}
              </button>
            </div>
            {docState.summary ? (
              <div className="prose prose-sm max-w-none">
                <pre className="whitespace-pre-wrap text-sm text-gray-800">{docState.summary}</pre>
                <div className="mt-1 text-xs text-gray-500">{docState.summary_generated_at ? `Generated: ${new Date(docState.summary_generated_at).toLocaleString()}` : ''} {docState.summary_model ? `· Model: ${docState.summary_model}` : ''}</div>
              </div>
            ) : (
              <div className="text-sm text-gray-500">No summary yet.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

interface PersonaEditRequestModalProps {
  persona: Persona;
  document?: KnowledgeDocument;
  onSubmit: (message: string) => void;
  onClose: () => void;
  isSubmitting: boolean;
}

const PersonaEditRequestModal: React.FC<PersonaEditRequestModalProps> = ({
  persona,
  document,
  onSubmit,
  onClose,
  isSubmitting,
}) => {
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (document) {
      setMessage(`Persona "${persona.name}" looks incorrect for document "${getDisplayTitle(document)}". Please update...`);
    } else {
      setMessage('');
    }
  }, [persona.id, document?.id]);

  const isDisabled = message.trim().length < 5 || isSubmitting;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-white rounded-lg shadow-lg w-full max-w-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Suggest persona change</h3>
            <p className="text-sm text-gray-600">
              Your request will be routed to administrators for review. Please include what should change.
            </p>
          </div>
          <button className="text-gray-400 hover:text-gray-600" onClick={onClose} disabled={isSubmitting}>
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="space-y-4">
          <div>
            <div className="text-sm text-gray-600">
              <span className="font-medium text-gray-800">{persona.name}</span>
              {document && (
                <>
                  {' '}
                  referenced in{' '}
                  <span className="font-medium text-gray-800">{getDisplayTitle(document)}</span>
                </>
              )}
            </div>
            {persona.platform_id && (
              <div className="text-xs text-gray-500 mt-1">
                Platform ID: <span className="font-mono">{persona.platform_id}</span>
              </div>
            )}
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Suggested updates</label>
            <textarea
              className="w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 text-sm"
              rows={5}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Describe what needs to change or what seems incorrect..."
            />
            <p className="text-xs text-gray-500 mt-1">Minimum 5 characters.</p>
          </div>
        </div>
        <div className="flex justify-end gap-3 mt-6">
          <Button variant="ghost" onClick={onClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button onClick={() => onSubmit(message.trim())} disabled={isDisabled} loading={isSubmitting} icon={<MessageSquare className="w-4 h-4" />}>
            Send request
          </Button>
        </div>
      </div>
    </div>
  );
};

interface PersonaFilterAutosuggestProps {
  label: string;
  placeholder: string;
  value: string;
  onChange: (personaId: string) => void;
  options: Persona[];
  loading?: boolean;
}

const PersonaFilterAutosuggest: React.FC<PersonaFilterAutosuggestProps> = ({
  label,
  placeholder,
  value,
  onChange,
  options,
  loading = false,
}) => {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const selected = options.find((option) => option.id === value);
    if (selected) {
      setQuery(selected.name);
    } else {
      setQuery('');
    }
  }, [value, options]);

  useEffect(() => {
    const handler = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const filtered = useMemo(() => {
    if (!query) return options.slice(0, 10);
    const q = query.toLowerCase();
    return options
      .filter(
        (persona) =>
          persona.name.toLowerCase().includes(q) ||
          (persona.platform_id?.toLowerCase().includes(q) ?? false)
      )
      .slice(0, 10);
  }, [query, options]);

  const handleSelect = (persona: Persona) => {
    setQuery(persona.name);
    onChange(persona.id);
    setOpen(false);
  };

  const handleClear = () => {
    setQuery('');
    onChange('');
    setOpen(false);
  };

  return (
    <div className="flex-1 min-w-[220px]" ref={containerRef}>
      <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      <div className="relative">
        <input
          type="text"
          className="w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 text-sm pr-8"
          placeholder={loading ? 'Loading personas...' : placeholder}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && filtered.length > 0) {
              e.preventDefault();
              handleSelect(filtered[0]);
            }
            if (e.key === 'Escape') {
              setOpen(false);
            }
          }}
          disabled={loading}
        />
        {(value || query) && (
          <button
            type="button"
            className="absolute inset-y-0 right-2 text-gray-400 hover:text-gray-600 text-xs"
            onClick={handleClear}
          >
            ×
          </button>
        )}
        {open && (
          <div className="absolute z-20 mt-1 w-full max-h-56 overflow-auto rounded-lg border border-gray-200 bg-white shadow-lg">
            {loading ? (
              <div className="p-3 text-sm text-gray-500">Loading personas...</div>
            ) : filtered.length === 0 ? (
              <div className="p-3 text-sm text-gray-500">No personas match “{query}”.</div>
            ) : (
              <ul className="divide-y divide-gray-100">
                {filtered.map((persona) => (
                  <li key={persona.id}>
                    <button
                      type="button"
                      className="w-full text-left px-3 py-2 text-sm hover:bg-primary-50"
                      onClick={() => handleSelect(persona)}
                    >
                      <div className="font-medium text-gray-900">{persona.name}</div>
                      <div className="text-xs text-gray-500 flex items-center gap-2">
                        {persona.platform_id && (
                          <span className="font-mono">{persona.platform_id}</span>
                        )}
                        {persona.user_id && <span>User: {persona.user_id}</span>}
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const PersonaMetadata: React.FC<{ persona: Persona }> = ({ persona }) => {
  const details: Array<{ label: string; value?: string }> = [];
  if (persona.platform_id) {
    details.push({ label: 'Platform ID', value: persona.platform_id });
  }
  if (persona.user_id) {
    details.push({ label: 'Linked user', value: persona.user_id });
  }
  if (persona.updated_at) {
    details.push({
      label: `Updated ${formatDistanceToNow(new Date(persona.updated_at))} ago`,
    });
  }
  if (details.length === 0) {
    return null;
  }
  return (
    <div className="text-xs text-gray-500 flex flex-wrap gap-x-4 gap-y-1">
      {details.map((item, index) => (
        <span key={`${item.label}-${index}`}>
          {item.value ? (
            <>
              {item.label}: <span className="font-mono text-gray-700">{item.value}</span>
            </>
          ) : (
            item.label
          )}
        </span>
      ))}
    </div>
  );
};

export default DocumentsPage;
