/**
 * LaTeX Studio Page
 *
 * Single-file LaTeX editor with server-side compile (PDF) and KB-assisted copilot hooks.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Download, Loader2, Play, Wand2, Trash2, Upload, Quote, Search } from 'lucide-react';
import toast from 'react-hot-toast';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { useNavigate, useSearchParams } from 'react-router-dom';

import Button from '../components/common/Button';
import { useAuth } from '../contexts/AuthContext';
import { apiClient } from '../services/api';
import type { DocumentsLocationState } from '../types/navigation';
import type {
  LatexCompileResponse,
  LatexCompileJobResponse,
  LatexCopilotFixResponse,
  LatexMathCopilotResponse,
  LatexCitationsResponse,
  LatexApplyUnifiedDiffResponse,
  SearchResult,
  LatexProjectFileListResponse,
  LatexProjectCompileResponse,
  LatexProjectListResponse,
  LatexProjectResponse,
  LatexProjectPublishResponse,
  LatexStatusResponse,
  ActiveGitSource,
  AgentJob,
  AgentJobChainDefinition,
  AgentJobTemplate,
  UnsafeExecStatusResponse,
} from '../types';

const DEFAULT_TEX = [
  '% LaTeX Studio (single-file)',
  '\\documentclass[11pt]{article}',
  '\\usepackage[utf8]{inputenc}',
  '\\usepackage{amsmath, amssymb}',
  '\\usepackage{hyperref}',
  '',
  '\\title{Untitled Paper}',
  '\\author{Author}',
  '\\date{\\today}',
  '',
  '\\begin{document}',
  '\\maketitle',
  '',
  '\\begin{abstract}',
  'Write your abstract here.',
  '\\end{abstract}',
  '',
  '\\section{Introduction}',
  'Start writing here.',
  '',
  '\\end{document}',
  '',
].join('\n');

function base64ToBlobUrl(base64: string, mimeType: string): string {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: mimeType });
  return URL.createObjectURL(blob);
}

const LatexStudioPage: React.FC = () => {
  const { user } = useAuth();
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const projectId = (searchParams.get('project') || '').trim() || null;

  const [projectTitle, setProjectTitle] = useState<string>('Untitled LaTeX Project');
  const [texSource, setTexSource] = useState(() => localStorage.getItem('latex_studio_tex') || DEFAULT_TEX);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [compileLog, setCompileLog] = useState('');
  const [safeMode, setSafeMode] = useState(true);
  const [activeCompileJobId, setActiveCompileJobId] = useState<string | null>(null);
  const lastCompileJobStatusRef = useRef<string | null>(null);
  const [copilotPrompt, setCopilotPrompt] = useState('');
  const [copilotSearchQuery, setCopilotSearchQuery] = useState('');
  const [copilotCitationMode, setCopilotCitationMode] = useState<'thebibliography' | 'bibtex'>('thebibliography');
  const [bibFilename, setBibFilename] = useState<string>('refs.bib');
  const [copilotResult, setCopilotResult] = useState<{ snippet: string; bibtex: string } | null>(null);
  const [publishResult, setPublishResult] = useState<LatexProjectPublishResponse | null>(null);
  const [publishTags, setPublishTags] = useState<string>('latex,paper');
  const [publishIncludePdf, setPublishIncludePdf] = useState(true);
  const [publishIncludeTex, setPublishIncludeTex] = useState(true);
  const [fixNotes, setFixNotes] = useState<string>('');

  const [showMathCopilot, setShowMathCopilot] = useState(false);
  const [mathGoal, setMathGoal] = useState<string>('Standardize math notation and fix equation references.');
  const [mathEnforceSiunitx, setMathEnforceSiunitx] = useState(true);
  const [mathEnforceShapes, setMathEnforceShapes] = useState(true);
  const [mathEnforceBoldItalic, setMathEnforceBoldItalic] = useState(true);
  const [mathEnforceEquationLabels, setMathEnforceEquationLabels] = useState(true);
  const [mathSelection, setMathSelection] = useState<string>('');
  const [mathCursorContext, setMathCursorContext] = useState<string>('');
  const [mathResult, setMathResult] = useState<LatexMathCopilotResponse | null>(null);
  const [mathLastMode, setMathLastMode] = useState<'analyze' | 'autocomplete'>('analyze');

  const [showResearchEngineer, setShowResearchEngineer] = useState(false);
  const [researchEngineerGoal, setResearchEngineerGoal] = useState<string>('');
  const [researchEngineerSearchQuery, setResearchEngineerSearchQuery] = useState<string>('');
  const [researchEngineerTargetSourceId, setResearchEngineerTargetSourceId] = useState<string>('');
  const [researchEngineerWorkflow, setResearchEngineerWorkflow] = useState<'loop' | 'simple'>('loop');
  const [researchEngineerEnableExperiments, setResearchEngineerEnableExperiments] = useState<boolean>(true);
  const [researchEngineerCommands, setResearchEngineerCommands] = useState<string>('');
  const [researchEngineerApplyPatchToKb, setResearchEngineerApplyPatchToKb] = useState<boolean>(false);
  const [researchEngineerApplyPatchToKbConfirm, setResearchEngineerApplyPatchToKbConfirm] = useState<boolean>(false);
  const [researchEngineerRequireExperimentsOk, setResearchEngineerRequireExperimentsOk] = useState<boolean>(true);
  const [researchEngineerProposalStrategy, setResearchEngineerProposalStrategy] = useState<'best_passing' | 'latest'>('best_passing');

  const [showCitationSync, setShowCitationSync] = useState(false);
  const [citationSyncMode, setCitationSyncMode] = useState<'bibtex' | 'thebibliography'>('bibtex');
  const [citationSyncBibFilename, setCitationSyncBibFilename] = useState<string>('refs.bib');
  const [citationSyncJobId, setCitationSyncJobId] = useState<string | null>(null);

  const [showReviewer, setShowReviewer] = useState(false);
  const [reviewFocus, setReviewFocus] = useState<string>('');
  const [reviewJobId, setReviewJobId] = useState<string | null>(null);

  const [showPaperPipeline, setShowPaperPipeline] = useState(false);
  const [paperPipelineGoal, setPaperPipelineGoal] = useState<string>('');
  const [paperPipelineSearchQuery, setPaperPipelineSearchQuery] = useState<string>('');
  const [paperPipelineTargetSourceId, setPaperPipelineTargetSourceId] = useState<string>('');
  const [paperPipelineCommands, setPaperPipelineCommands] = useState<string>('');
  const [paperPipelineApplyReviewDiff, setPaperPipelineApplyReviewDiff] = useState<boolean>(false);
  const [paperPipelineEnableExperiments, setPaperPipelineEnableExperiments] = useState<boolean>(true);
  const [paperPipelineEnableCitationSync, setPaperPipelineEnableCitationSync] = useState<boolean>(true);
  const [paperPipelineEnableReviewer, setPaperPipelineEnableReviewer] = useState<boolean>(true);
  const [paperPipelineEnableCompile, setPaperPipelineEnableCompile] = useState<boolean>(true);
  const [paperPipelineEnablePublish, setPaperPipelineEnablePublish] = useState<boolean>(true);
  const [paperPipelineCompileSafeMode, setPaperPipelineCompileSafeMode] = useState<boolean>(true);
  const [paperPipelinePublishIncludeTex, setPaperPipelinePublishIncludeTex] = useState<boolean>(true);
  const [paperPipelinePublishIncludePdf, setPaperPipelinePublishIncludePdf] = useState<boolean>(true);

  const [citeQuery, setCiteQuery] = useState('');
  const [selectedCiteDocIds, setSelectedCiteDocIds] = useState<Record<string, boolean>>({});
  const [citeMode, setCiteMode] = useState<'bibtex' | 'thebibliography'>('bibtex');
  const [citeAutoInsert, setCiteAutoInsert] = useState(true);
  const [citeAutoUpdateBib, setCiteAutoUpdateBib] = useState(true);
  const [citeAutoInsertBibliography, setCiteAutoInsertBibliography] = useState(true);
  const [citeAutoInsertThebibliography, setCiteAutoInsertThebibliography] = useState(true);
  const [lastCitations, setLastCitations] = useState<LatexCitationsResponse | null>(null);

  const { data: latexStatus } = useQuery<LatexStatusResponse>(
    ['latex-status'],
    () => apiClient.getLatexStatus(),
    { staleTime: 30000 }
  );

  const compilerAvailable = useMemo(() => {
    if (!latexStatus) return false;
    const anyEngine = Object.values(latexStatus.available_engines || {}).some(Boolean);
    return latexStatus.enabled && anyEngine;
  }, [latexStatus]);

  const useWorkerCompile = useMemo(() => {
    return !!latexStatus?.use_celery_worker;
  }, [latexStatus?.use_celery_worker]);

  const { data: unsafeExecStatus } = useQuery<UnsafeExecStatusResponse>(
    ['unsafe-exec-status'],
    () => apiClient.getUnsafeExecAvailability(),
    { staleTime: 30000 }
  );

  useEffect(() => {
    if (unsafeExecStatus && !unsafeExecStatus.enabled) {
      setPaperPipelineEnableExperiments(false);
      setResearchEngineerEnableExperiments(false);
    }
  }, [unsafeExecStatus?.enabled]);

  useEffect(() => {
    if (!researchEngineerApplyPatchToKbConfirm) return;
    if (!researchEngineerApplyPatchToKb) setResearchEngineerApplyPatchToKb(true);
  }, [researchEngineerApplyPatchToKbConfirm, researchEngineerApplyPatchToKb]);

  useEffect(() => {
    if (researchEngineerApplyPatchToKb) return;
    if (researchEngineerApplyPatchToKbConfirm) setResearchEngineerApplyPatchToKbConfirm(false);
  }, [researchEngineerApplyPatchToKb, researchEngineerApplyPatchToKbConfirm]);

  const { data: activeGitSources } = useQuery<ActiveGitSource[]>(
    ['active-git-sources'],
    () => apiClient.getActiveGitSources(),
    { staleTime: 30000 }
  );

  const { data: chainDefinitionsData } = useQuery<{ chains: AgentJobChainDefinition[] }>(
    ['agent-job-chains'],
    async () => {
      const res = await apiClient.listChainDefinitions();
      return { chains: res.chains || [] };
    },
    { staleTime: 30000 }
  );

  const researchEngineerChain = useMemo(() => {
    const chains = chainDefinitionsData?.chains || [];
    return chains.find((c) => (c as any)?.name === 'research_engineer_chain') || null;
  }, [chainDefinitionsData?.chains]);

  const researchEngineerLoopChain = useMemo(() => {
    const chains = chainDefinitionsData?.chains || [];
    return chains.find((c) => (c as any)?.name === 'research_engineer_loop_chain') || null;
  }, [chainDefinitionsData?.chains]);

  useEffect(() => {
    if (researchEngineerWorkflow === 'loop' && !researchEngineerLoopChain && researchEngineerChain) {
      setResearchEngineerWorkflow('simple');
    }
  }, [researchEngineerWorkflow, researchEngineerLoopChain, researchEngineerChain]);

  const selectedResearchEngineerChain = useMemo(() => {
    if (researchEngineerWorkflow === 'loop') return researchEngineerLoopChain || researchEngineerChain;
    return researchEngineerChain || researchEngineerLoopChain;
  }, [researchEngineerWorkflow, researchEngineerLoopChain, researchEngineerChain]);

  const paperPipelineChain = useMemo(() => {
    const chains = chainDefinitionsData?.chains || [];
    return chains.find((c) => (c as any)?.name === 'paper_pipeline_chain') || null;
  }, [chainDefinitionsData?.chains]);

  const { data: latexAgentTemplates } = useQuery<{ templates: AgentJobTemplate[] }>(
    ['agent-job-templates', 'latex'],
    async () => {
      const res = await apiClient.listAgentJobTemplates('latex');
      return { templates: res.templates || [] };
    },
    { staleTime: 30000 }
  );

  const citationSyncTemplate = useMemo(() => {
    const templates = latexAgentTemplates?.templates || [];
    return templates.find((t) => (t as any)?.name === 'latex_citation_sync') || null;
  }, [latexAgentTemplates?.templates]);

  const reviewerTemplate = useMemo(() => {
    const templates = latexAgentTemplates?.templates || [];
    return templates.find((t) => (t as any)?.name === 'latex_reviewer_critic') || null;
  }, [latexAgentTemplates?.templates]);

  const { data: citationSyncJob } = useQuery<AgentJob>(
    ['agent-job-detail', citationSyncJobId],
    () => apiClient.getAgentJob(citationSyncJobId as string),
    { enabled: !!citationSyncJobId, refetchInterval: 2000 }
  );

  const { data: reviewerJob } = useQuery<AgentJob>(
    ['agent-job-detail', reviewJobId],
    () => apiClient.getAgentJob(reviewJobId as string),
    { enabled: !!reviewJobId, refetchInterval: 2000 }
  );

  useEffect(() => {
    if (!projectId) localStorage.setItem('latex_studio_tex', texSource);
  }, [texSource, projectId]);

  useEffect(() => {
    return () => {
      if (pdfUrl && pdfUrl.startsWith('blob:')) URL.revokeObjectURL(pdfUrl);
    };
  }, [pdfUrl]);

  const compileMutation = useMutation(
    (payload: { tex_source: string; safe_mode: boolean }) => apiClient.compileLatex(payload),
    {
      onSuccess: (res: LatexCompileResponse) => {
        setCompileLog(res.log || '');
        if (!res.success || !res.pdf_base64) {
          if (res.violations && res.violations.length > 0) {
            toast.error(`Blocked by safe mode: ${res.violations[0]}`);
          } else {
            toast.error('Compilation failed');
          }
          return;
        }
        const nextUrl = base64ToBlobUrl(res.pdf_base64, 'application/pdf');
        setPdfUrl((prev) => {
          if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev);
          return nextUrl;
        });
        toast.success(`Compiled (${res.engine || 'unknown'})`);
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Compilation failed';
        toast.error(msg);
      },
    }
  );

  const { data: projectsData } = useQuery<LatexProjectListResponse>(
    ['latex-projects'],
    () => apiClient.listLatexProjects({ limit: 200, offset: 0 }),
    { staleTime: 10000 }
  );

  const { data: currentProject, refetch: refetchCurrentProject } = useQuery<LatexProjectResponse>(
    ['latex-project', projectId],
    () => apiClient.getLatexProject(projectId as string),
    { enabled: !!projectId, staleTime: 0 }
  );

  useEffect(() => {
    const compileAllowed = !!compilerAvailable && !(latexStatus?.admin_only && user?.role !== 'admin');
    if (compileAllowed) return;
    if (paperPipelineEnableCompile) setPaperPipelineEnableCompile(false);
    const hasExistingPdf = !!(currentProject?.pdf_file_path || currentProject?.pdf_download_url);
    if (!hasExistingPdf && paperPipelinePublishIncludePdf) setPaperPipelinePublishIncludePdf(false);
  }, [
    compilerAvailable,
    latexStatus?.admin_only,
    user?.role,
    currentProject?.pdf_file_path,
    currentProject?.pdf_download_url,
    paperPipelineEnableCompile,
    paperPipelinePublishIncludePdf,
  ]);

  const { data: projectFilesData } = useQuery<LatexProjectFileListResponse>(
    ['latex-project-files', projectId],
    () => apiClient.listLatexProjectFiles(projectId as string),
    { enabled: !!projectId, staleTime: 0 }
  );

  useEffect(() => {
    if (copilotCitationMode !== 'bibtex') return;
    const bibs = (projectFilesData?.items || []).filter((f) => (f.filename || '').toLowerCase().endsWith('.bib'));
    if (bibs.length === 0) return;
    const current = (bibFilename || '').trim();
    if (!current || current === 'refs.bib') {
      setBibFilename(bibs[0].filename);
    }
  }, [copilotCitationMode, projectFilesData?.items, bibFilename]);

  useEffect(() => {
    setCiteMode(copilotCitationMode === 'bibtex' ? 'bibtex' : 'thebibliography');
  }, [copilotCitationMode]);

  useEffect(() => {
    if (!projectId) return;
    if (!currentProject) return;
    setProjectTitle(currentProject.title || 'Untitled LaTeX Project');
    setTexSource(currentProject.tex_source || DEFAULT_TEX);
    if (currentProject.pdf_download_url) {
      setPdfUrl((prev) => {
        if (prev && prev.startsWith('blob:') && prev !== currentProject.pdf_download_url) URL.revokeObjectURL(prev);
        return currentProject.pdf_download_url as string;
      });
    }
    if (currentProject.last_compile_log) setCompileLog(currentProject.last_compile_log);
  }, [projectId, currentProject]);

  const saveProjectMutation = useMutation(
    async () => {
      if (projectId) {
        return apiClient.updateLatexProject(projectId, { title: projectTitle, tex_source: texSource });
      }
      return apiClient.createLatexProject({ title: projectTitle, tex_source: texSource });
    },
    {
      onSuccess: (res: LatexProjectResponse) => {
        toast.success('Saved');
        queryClient.invalidateQueries(['latex-projects']);
        if (!projectId) {
          const next = new URLSearchParams(searchParams);
          next.set('project', res.id);
          setSearchParams(next, { replace: true });
        } else {
          refetchCurrentProject();
        }
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Save failed';
        toast.error(msg);
      },
    }
  );

  const deleteProjectMutation = useMutation(
    async () => {
      if (!projectId) return;
      await apiClient.deleteLatexProject(projectId);
    },
    {
      onSuccess: () => {
        toast.success('Deleted');
        queryClient.invalidateQueries(['latex-projects']);
        const next = new URLSearchParams(searchParams);
        next.delete('project');
        setSearchParams(next, { replace: true });
        setProjectTitle('Untitled LaTeX Project');
        setTexSource(DEFAULT_TEX);
        setPdfUrl(null);
        setCompileLog('');
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Delete failed';
        toast.error(msg);
      },
    }
  );

  const compileProjectMutation = useMutation(
    async () => {
      if (!projectId) throw new Error('No project selected');
      return apiClient.compileLatexProject(projectId, { safe_mode: safeMode });
    },
    {
      onSuccess: (res: LatexProjectCompileResponse) => {
        setCompileLog(res.log || '');
        if (!res.success) {
          if (res.violations && res.violations.length > 0) toast.error(res.violations[0]);
          else toast.error('Compilation failed');
          return;
        }
        if (res.pdf_download_url) setPdfUrl(res.pdf_download_url);
        toast.success(`Compiled (${res.engine || 'unknown'})`);
        refetchCurrentProject();
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Compilation failed';
        toast.error(msg);
      },
    }
  );

  const createCompileJobMutation = useMutation(
    async () => {
      if (!projectId) throw new Error('No project selected');
      return apiClient.createLatexProjectCompileJob(projectId, { safe_mode: safeMode });
    },
    {
      onSuccess: (res: LatexCompileJobResponse) => {
        lastCompileJobStatusRef.current = null;
        setActiveCompileJobId(res.id);
        setCompileLog((res.log || '').trim() || 'Queued compile job…');
        toast.success('Queued compile job');
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Failed to enqueue compile job';
        toast.error(msg);
      },
    }
  );

  useQuery<LatexCompileJobResponse>(
    ['latex-compile-job', activeCompileJobId],
    () => apiClient.getLatexCompileJob(activeCompileJobId as string),
    {
      enabled: !!activeCompileJobId,
      staleTime: 0,
      refetchInterval: (data) => {
        const status = (data as any)?.status;
        if (!status) return 1000;
        if (status === 'queued' || status === 'running') return 1000;
        return false;
      },
      onSuccess: (job) => {
        if (!job) return;
        if (job.log != null) setCompileLog(job.log);

        const prev = lastCompileJobStatusRef.current;
        if (job.status && job.status !== prev) {
          lastCompileJobStatusRef.current = job.status;

          if (job.status === 'succeeded') {
            if (job.pdf_download_url) setPdfUrl(job.pdf_download_url);
            toast.success(`Compiled (${job.engine || 'unknown'})`);
            setActiveCompileJobId(null);
            refetchCurrentProject();
          } else if (job.status === 'failed') {
            if (job.violations && job.violations.length > 0) toast.error(job.violations[0]);
            else toast.error('Compilation failed');
            setActiveCompileJobId(null);
            refetchCurrentProject();
          }
        }
      },
    }
  );

  const publishMutation = useMutation(
    async () => {
      if (!projectId) throw new Error('Save the project before publishing');
      const tags = publishTags
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean);
      return apiClient.publishLatexProject(projectId, {
        include_tex: publishIncludeTex,
        include_pdf: publishIncludePdf,
        safe_mode: safeMode,
        tags: tags.length > 0 ? tags : undefined,
      });
    },
    {
      onSuccess: (res) => {
        setPublishResult(res);
        queryClient.invalidateQueries(['documents']);
        toast.success('Published to Knowledge DB');
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Publish failed';
        toast.error(msg);
      },
    }
  );

  const startResearchEngineerMutation = useMutation(
    async () => {
      if (!projectId) throw new Error('Save the project first');
      if (!selectedResearchEngineerChain?.id) throw new Error('ResearchEngineer chain not available');
      const goal = (researchEngineerGoal || '').trim();
      if (!goal) throw new Error('Goal is required');
      const sourceId = (researchEngineerTargetSourceId || '').trim();
      if (!sourceId) throw new Error('Select a git source');

      const overrides: Record<string, any> = {
        latex_project_id: projectId,
        target_source_id: sourceId,
      };
      const q = (researchEngineerSearchQuery || '').trim();
      if (q) overrides.search_query = q;

      if (researchEngineerWorkflow === 'loop') {
        overrides.enable_experiments = researchEngineerEnableExperiments;
        const commands = (researchEngineerCommands || '')
          .split('\n')
          .map((c) => c.trim())
          .filter(Boolean)
          .slice(0, 6);
        overrides.commands = commands;
        overrides.apply_patch_to_kb = researchEngineerApplyPatchToKb;
        overrides.apply_patch_to_kb_confirm = researchEngineerApplyPatchToKbConfirm;
        overrides.require_experiments_ok = researchEngineerRequireExperimentsOk;
        overrides.proposal_strategy = researchEngineerProposalStrategy;
      }

      return apiClient.createJobFromChain({
        chain_definition_id: selectedResearchEngineerChain.id,
        name_prefix: `ResearchEngineer — ${projectTitle || 'LaTeX Project'} — ${new Date().toLocaleDateString()}`,
        variables: { goal },
        config_overrides: overrides,
        start_immediately: true,
      });
    },
    {
      onSuccess: (job: any) => {
        toast.success('ResearchEngineer started');
        setShowResearchEngineer(false);
        const id = String(job?.id || '').trim();
        if (id) navigate(`/autonomous-agents?job=${encodeURIComponent(id)}`);
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Failed to start ResearchEngineer';
        toast.error(String(msg));
      },
    }
  );

  const startPaperPipelineMutation = useMutation(
    async () => {
      if (!projectId) throw new Error('Save the project first');
      if (!paperPipelineChain?.id) throw new Error('PaperPipeline chain not available');
      const goal = (paperPipelineGoal || '').trim();
      if (!goal) throw new Error('Goal is required');
      const sourceId = (paperPipelineTargetSourceId || '').trim();
      if (!sourceId) throw new Error('Select a git source');

      const overrides: Record<string, any> = {
        latex_project_id: projectId,
        target_source_id: sourceId,
        mode: citeMode,
        bib_filename: bibFilename,
        publish_tags: publishTags,
        safe_mode: paperPipelineCompileSafeMode,
        include_tex: paperPipelinePublishIncludeTex,
        include_pdf: paperPipelinePublishIncludePdf,
        enable_experiments: paperPipelineEnableExperiments,
        enable_citation_sync: paperPipelineEnableCitationSync,
        enable_reviewer: paperPipelineEnableReviewer,
        enable_compile: paperPipelineEnableCompile,
        enable_publish: paperPipelineEnablePublish,
        apply_review_diff: paperPipelineApplyReviewDiff,
      };
      const q = (paperPipelineSearchQuery || '').trim();
      if (q) overrides.search_query = q;

      const commands = (paperPipelineCommands || '')
        .split('\n')
        .map((x) => x.trim())
        .filter(Boolean);
      if (commands.length > 0) overrides.commands = commands.slice(0, 6);

      return apiClient.createJobFromChain({
        chain_definition_id: paperPipelineChain.id,
        name_prefix: `PaperPipeline — ${projectTitle || 'LaTeX Project'} — ${new Date().toLocaleDateString()}`,
        variables: { goal },
        config_overrides: overrides,
        start_immediately: true,
      });
    },
    {
      onSuccess: (job: any) => {
        toast.success('PaperPipeline started');
        setShowPaperPipeline(false);
        const id = String(job?.id || '').trim();
        if (id) navigate(`/autonomous-agents?job=${encodeURIComponent(id)}`);
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Failed to start PaperPipeline';
        toast.error(String(msg));
      },
    }
  );

  const startCitationSyncMutation = useMutation(
    async () => {
      if (!projectId) throw new Error('Save the project first');
      if (!citationSyncTemplate?.id) throw new Error('CitationSync template not available');

      const job = await apiClient.createAgentJobFromTemplate({
        template_id: String(citationSyncTemplate.id),
        name: `CitationSync — ${projectTitle || 'LaTeX Project'} — ${new Date().toLocaleDateString()}`,
        goal: 'Synchronize citations for this LaTeX project.',
        config: {
          latex_project_id: projectId,
          mode: citationSyncMode,
          bib_filename: citationSyncBibFilename,
        },
        start_immediately: true,
      });
      return job;
    },
    {
      onSuccess: (job: AgentJob) => {
        setCitationSyncJobId(job.id);
        toast.success('CitationSync started');
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Failed to start CitationSync';
        toast.error(String(msg));
      },
    }
  );

  const startReviewerMutation = useMutation(
    async () => {
      if (!projectId) throw new Error('Save the project first');
      if (!reviewerTemplate?.id) throw new Error('Reviewer template not available');

      const job = await apiClient.createAgentJobFromTemplate({
        template_id: String(reviewerTemplate.id),
        name: `Reviewer — ${projectTitle || 'LaTeX Project'} — ${new Date().toLocaleDateString()}`,
        goal: 'Review this LaTeX paper for missing citations, clarity, and notation consistency.',
        config: {
          latex_project_id: projectId,
          focus: reviewFocus,
        },
        start_immediately: true,
      });
      return job;
    },
    {
      onSuccess: (job: AgentJob) => {
        setReviewJobId(job.id);
        toast.success('Reviewer started');
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Failed to start Reviewer';
        toast.error(String(msg));
      },
    }
  );

  const applyReviewerDiffMutation = useMutation(
    async (diffUnified: string) => {
      if (!projectId) throw new Error('Save the project first');
      return apiClient.applyLatexProjectUnifiedDiff(projectId, { diff_unified: diffUnified });
    },
    {
      onSuccess: async (res: LatexApplyUnifiedDiffResponse) => {
        setTexSource(res.tex_source);
        await refetchCurrentProject();
        if (res.applied) {
          toast.success(res.warnings?.length ? `Applied diff (warnings: ${res.warnings.length})` : 'Applied diff');
        } else {
          toast.success('No changes to apply');
        }
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Failed to apply diff';
        toast.error(String(msg));
      },
    }
  );

  const uploadFileMutation = useMutation(
    async (f: File) => {
      if (!projectId) throw new Error('Save the project first');
      return apiClient.uploadLatexProjectFile(projectId, f, true);
    },
    {
      onSuccess: () => {
        toast.success('File uploaded');
        queryClient.invalidateQueries(['latex-project-files', projectId]);
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Upload failed';
        toast.error(msg);
      },
    }
  );

  const deleteFileMutation = useMutation(
    async (fileId: string) => {
      if (!projectId) return;
      await apiClient.deleteLatexProjectFile(projectId, fileId);
    },
    {
      onSuccess: () => {
        toast.success('File deleted');
        queryClient.invalidateQueries(['latex-project-files', projectId]);
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Delete failed';
        toast.error(msg);
      },
    }
  );

  const insertAtCursor = useCallback((text: string) => {
    const el = textareaRef.current;
    if (!el) {
      setTexSource((prev) => prev + text);
      return;
    }
    const start = el.selectionStart ?? el.value.length;
    const end = el.selectionEnd ?? el.value.length;
    const next = el.value.slice(0, start) + text + el.value.slice(end);
    setTexSource(next);
    requestAnimationFrame(() => {
      el.focus();
      const pos = start + text.length;
      el.setSelectionRange(pos, pos);
    });
  }, []);

  const downloadTex = useCallback(() => {
    const blob = new Blob([texSource], { type: 'text/x-tex;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'paper.tex';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [texSource]);

  const downloadPdf = useCallback(() => {
    if (!pdfUrl) {
      toast.error('Compile a PDF first');
      return;
    }
    const a = document.createElement('a');
    a.href = pdfUrl;
    a.download = 'paper.pdf';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [pdfUrl]);

  const exportZip = useCallback(async () => {
    if (!projectId) {
      toast.error('Save the project first');
      return;
    }
    try {
      await apiClient.downloadLatexProjectZip(projectId, projectTitle);
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || 'Export failed';
      toast.error(String(msg));
    }
  }, [projectId, projectTitle]);

  const copilotMutation = useMutation(
    (payload: { prompt: string; search_query?: string; citation_mode?: 'thebibliography' | 'bibtex' }) => apiClient.latexCopilotSection(payload),
    {
      onSuccess: (
        res: { tex_snippet: string; bibtex: string; references_tex?: string; bibtex_entries?: string },
        vars: { citation_mode?: 'thebibliography' | 'bibtex' }
      ) => {
        const mode = vars?.citation_mode || 'thebibliography';
        const refs = (res.references_tex ?? res.bibtex ?? '').trim();
        const entries = (res.bibtex_entries || '').trim();
        setCopilotResult({ snippet: res.tex_snippet || '', bibtex: mode === 'bibtex' ? entries : refs });
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Copilot failed';
        toast.error(msg);
      },
    }
  );

  const mathCopilotMutation = useMutation(
    async (vars: { mode: 'analyze' | 'autocomplete' }) => {
      const mode = vars?.mode || 'analyze';
      const goal = (mathGoal || '').trim() || 'Standardize math notation and fix equation references.';
      setMathLastMode(mode);
      return apiClient.latexMathCopilot({
        tex_source: texSource,
        mode,
        goal,
        selection: mathSelection || undefined,
        cursor_context: mathCursorContext || undefined,
        enforce_siunitx: mathEnforceSiunitx,
        enforce_shapes: mathEnforceShapes,
        enforce_bold_italic_conventions: mathEnforceBoldItalic,
        enforce_equation_labels: mathEnforceEquationLabels,
        return_patched_source: true,
      });
    },
    {
      onSuccess: (res) => {
        setMathResult(res);
        if (res.diff_unified?.trim()) {
          toast.success(res.diff_applies ? 'Math copilot ready (diff applies)' : 'Math copilot ready (diff may not apply)');
        } else {
          toast.success('Math copilot ready');
        }
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Math copilot failed';
        toast.error(String(msg));
      },
    }
  );

  const applyMathToEditorMutation = useMutation(
    async () => {
      if (!mathResult?.tex_source_patched) throw new Error('No patched source returned');
      setTexSource(mathResult.tex_source_patched);
      toast.success('Applied to editor');
    },
    {
      onError: (e: any) => {
        toast.error(String(e?.message || 'Failed to apply'));
      },
    }
  );

  const applyMathAndSaveMutation = useMutation(
    async () => {
      if (!projectId) throw new Error('Save the project first');
      if (!mathResult?.tex_source_patched) throw new Error('No patched source returned');
      const res = await apiClient.updateLatexProject(projectId, { title: projectTitle, tex_source: mathResult.tex_source_patched });
      setTexSource(res.tex_source || mathResult.tex_source_patched);
      await refetchCurrentProject();
      return res;
    },
    {
      onSuccess: () => {
        toast.success('Applied and saved');
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Failed to apply & save';
        toast.error(String(msg));
      },
    }
  );

  const insertBeforeEndDocument = useCallback((source: string, addition: string) => {
    const marker = '\\end{document}';
    const idx = source.lastIndexOf(marker);
    if (idx === -1) return source + (source.endsWith('\n') ? '' : '\n') + addition + '\n';
    const before = source.slice(0, idx).trimEnd();
    const after = source.slice(idx);
    return `${before}\n\n${addition.trim()}\n\n${after}`;
  }, []);

  const refreshMathContextFromEditor = useCallback(() => {
    const el = textareaRef.current;
    const text = (el?.value ?? texSource) || '';
    const start = typeof el?.selectionStart === 'number' ? el.selectionStart : 0;
    const end = typeof el?.selectionEnd === 'number' ? el.selectionEnd : start;
    const selection = start !== end ? text.slice(start, end) : '';
    const cursor = end;
    const windowChars = 800;
    const ctx = text.slice(Math.max(0, cursor - windowChars), Math.min(text.length, cursor + windowChars));
    setMathSelection(selection.trim());
    setMathCursorContext(ctx.trim());
  }, [texSource]);

  const ensureBibliographyScaffold = useCallback((source: string, bib: string) => {
    const stem = (bib || 'refs.bib').trim().replace(/\.bib$/i, '');
    const marker = '\\bibliography{';
    if (source.includes(marker)) return source;
    const scaffold = `\\bibliographystyle{plain}\n\\bibliography{${stem}}`;
    return insertBeforeEndDocument(source, scaffold);
  }, [insertBeforeEndDocument]);

  const appendOrCreateProjectBib = useCallback(async (bibName: string, newEntries: string) => {
    if (!projectId) throw new Error('No project selected');
    const name = (bibName || 'refs.bib').trim();
    if (!name || name.includes('/') || name.includes('\\')) throw new Error('Invalid bib filename');
    const filename = name.toLowerCase().endsWith('.bib') ? name : `${name}.bib`;

    const existing = (projectFilesData?.items || []).find((f) => (f.filename || '').toLowerCase() === filename.toLowerCase());
    let existingText = '';
    if (existing?.download_url) {
      try {
        const resp = await fetch(existing.download_url);
        if (resp.ok) existingText = await resp.text();
      } catch {
        // ignore
      }
    }

    const keysInExisting = new Set<string>();
    const re = /@\w+\s*\{\s*([^,\s]+)\s*,/g;
    let m: RegExpExecArray | null = null;
    while ((m = re.exec(existingText))) {
      keysInExisting.add(m[1]);
    }

    const filtered = (newEntries || '')
      .split(/\n(?=@\w+\s*\{)/g)
      .map((x) => x.trim())
      .filter(Boolean)
      .filter((block) => {
        const mm = /@\w+\s*\{\s*([^,\s]+)\s*,/.exec(block);
        if (!mm) return true;
        return !keysInExisting.has(mm[1]);
      })
      .join('\n\n');

    const merged = (existingText || '').trim() + (filtered ? `\n\n${filtered.trim()}\n` : '\n');
    const file = new File([merged], filename, { type: 'text/x-bibtex' });
    uploadFileMutation.mutate(file);
  }, [projectId, projectFilesData?.items, uploadFileMutation]);

  const copilotEnabled = useMemo(() => true, []);

  const fixMutation = useMutation(
    async () => {
      const log = (compileLog || '').trim();
      if (!log) throw new Error('No compile log to fix');
      return apiClient.latexCopilotFix({ tex_source: texSource, compile_log: log, safe_mode: safeMode });
    },
    {
      onSuccess: (res: LatexCopilotFixResponse) => {
        setTexSource(res.tex_source_fixed || texSource);
        setFixNotes(res.notes || '');
        if (res.unsafe_warnings && res.unsafe_warnings.length > 0) {
          toast.error(`Unsafe warning: ${res.unsafe_warnings[0]}`);
        } else {
          toast.success('Applied suggested fix');
        }
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Fix failed';
        toast.error(msg);
      },
    }
  );

  const { data: citeSearchData, isFetching: isCiteSearching } = useQuery(
    ['latex-cite-search', citeQuery],
    () =>
      apiClient.searchDocuments({
        q: citeQuery,
        mode: 'smart',
        sort_by: 'relevance',
        sort_order: 'desc',
        page: 1,
        page_size: 8,
      }),
    { enabled: citeQuery.trim().length >= 2, staleTime: 15000 }
  );

  const citeResults: SearchResult[] = (citeSearchData as any)?.results || [];

  const citeMutation = useMutation(
    async (docIds: string[]) => {
      return apiClient.generateLatexCitationsFromDocuments({
        document_ids: docIds,
        mode: citeMode,
        bib_filename: bibFilename,
      });
    },
    {
      onSuccess: async (res) => {
        setLastCitations(res);
        if (citeAutoInsert && res.cite_command) {
          insertAtCursor(` ${res.cite_command} `);
        }
        if (res.mode === 'thebibliography' && citeAutoInsertThebibliography && res.references_tex) {
          setTexSource((prev) => insertBeforeEndDocument(prev, res.references_tex as string));
        }
        if (res.mode === 'bibtex') {
          if (citeAutoInsertBibliography) {
            setTexSource((prev) => ensureBibliographyScaffold(prev, bibFilename));
          }
          if (citeAutoUpdateBib && projectId && res.bibtex_entries) {
            try {
              await appendOrCreateProjectBib(bibFilename, res.bibtex_entries as string);
            } catch (e: any) {
              toast.error(e?.message || 'Failed to update .bib');
            }
          }
        }
        toast.success('Citations generated');
      },
      onError: (e: any) => {
        const msg = e?.response?.data?.detail || e?.message || 'Citation generation failed';
        toast.error(msg);
      },
    }
  );

  return (
    <div className="p-6 h-full flex flex-col">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">LaTeX Studio</h1>
          <p className="text-gray-500">Write LaTeX with AI assistance, then compile to PDF (server-side)</p>
          {latexStatus && !latexStatus.enabled && (
            <div className="mt-2 text-xs text-yellow-700 bg-yellow-50 border border-yellow-200 rounded px-2 py-1 inline-block">
              Compiler disabled on server. Set <span className="font-mono">LATEX_COMPILER_ENABLED=true</span> to enable.
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            onClick={() => {
              const next = new URLSearchParams(searchParams);
              next.delete('project');
              setSearchParams(next, { replace: true });
              setProjectTitle('Untitled LaTeX Project');
              setTexSource(DEFAULT_TEX);
              setPdfUrl(null);
              setCompileLog('');
            }}
          >
            New
          </Button>
          <Button onClick={() => saveProjectMutation.mutate()} disabled={saveProjectMutation.isLoading}>
            {saveProjectMutation.isLoading ? 'Saving…' : 'Save'}
          </Button>
          <Button
            variant="secondary"
            onClick={() => publishMutation.mutate()}
            disabled={!projectId || publishMutation.isLoading}
            title={!projectId ? 'Save the project first' : 'Publish project files into Knowledge DB'}
          >
            {publishMutation.isLoading ? 'Publishing…' : 'Publish'}
          </Button>
          <Button
            variant="secondary"
            onClick={() => {
              if (!projectId) {
                toast.error('Save the project first');
                return;
              }
              setResearchEngineerGoal('');
              setResearchEngineerSearchQuery('');
              setResearchEngineerTargetSourceId('');
              setResearchEngineerWorkflow(researchEngineerLoopChain ? 'loop' : 'simple');
              setResearchEngineerEnableExperiments(true);
              setResearchEngineerCommands('');
              setResearchEngineerApplyPatchToKb(false);
              setResearchEngineerApplyPatchToKbConfirm(false);
              setResearchEngineerRequireExperimentsOk(true);
              setResearchEngineerProposalStrategy('best_passing');
              setShowResearchEngineer(true);
            }}
            disabled={!projectId}
            title={!projectId ? 'Save the project first' : 'Start AI Scientist ↔ Code Agent chain'}
          >
            ResearchEngineer
          </Button>
          <Button
            variant="secondary"
            onClick={() => {
              if (!projectId) {
                toast.error('Save the project first');
                return;
              }
              setPaperPipelineGoal('');
              setPaperPipelineSearchQuery('');
              setPaperPipelineTargetSourceId('');
              setPaperPipelineCommands('');
              setPaperPipelineApplyReviewDiff(false);
              setPaperPipelineEnableExperiments(true);
              setPaperPipelineEnableCitationSync(true);
              setPaperPipelineEnableReviewer(true);
              setPaperPipelineEnableCompile(true);
              setPaperPipelineEnablePublish(true);
              setPaperPipelineCompileSafeMode(true);
              setPaperPipelinePublishIncludeTex(true);
              setPaperPipelinePublishIncludePdf(true);
              setShowPaperPipeline(true);
            }}
            disabled={!projectId}
            title={!projectId ? 'Save the project first' : 'Run the full Plan → Patch → Run → Cite → Review → Compile → Publish chain'}
          >
            PaperPipeline
          </Button>
          <Button
            variant="secondary"
            onClick={() => {
              if (!projectId) {
                toast.error('Save the project first');
                return;
              }
              setCitationSyncMode('bibtex');
              setCitationSyncBibFilename(bibFilename || 'refs.bib');
              setCitationSyncJobId(null);
              setShowCitationSync(true);
            }}
            disabled={!projectId}
            title={!projectId ? 'Save the project first' : 'Sync \\cite{KDB:<uuid>} into refs.bib or thebibliography'}
          >
            CitationSync
          </Button>
          <Button
            variant="secondary"
            onClick={() => {
              setMathResult(null);
              setMathLastMode('analyze');
              refreshMathContextFromEditor();
              setShowMathCopilot(true);
            }}
            title="Math-aware copilot: notation, shapes, units, equation refs"
          >
            MathCopilot
          </Button>
          <Button
            variant="secondary"
            onClick={() => {
              if (!projectId) {
                toast.error('Save the project first');
                return;
              }
              setReviewFocus('');
              setReviewJobId(null);
              setShowReviewer(true);
            }}
            disabled={!projectId}
            title={!projectId ? 'Save the project first' : 'Run Reviewer/Critic and apply suggested diff'}
          >
            Review
          </Button>
          <Button
            variant="secondary"
            onClick={() => {
              if (!projectId) return;
              if (!window.confirm('Delete this LaTeX project?')) return;
              deleteProjectMutation.mutate();
            }}
            disabled={!projectId || deleteProjectMutation.isLoading}
          >
            Delete
          </Button>
          <Button variant="secondary" onClick={downloadTex}>
            <Download className="w-4 h-4 mr-2" />
            Download .tex
          </Button>
          <Button variant="secondary" onClick={downloadPdf} disabled={!pdfUrl}>
            <Download className="w-4 h-4 mr-2" />
            Download PDF
          </Button>
          <Button variant="secondary" onClick={exportZip} disabled={!projectId}>
            <Download className="w-4 h-4 mr-2" />
            Export ZIP
          </Button>
          <Button
            onClick={() => {
              if (projectId) {
                if (activeCompileJobId) return;
                if (useWorkerCompile) createCompileJobMutation.mutate();
                else compileProjectMutation.mutate();
              }
              else compileMutation.mutate({ tex_source: texSource, safe_mode: safeMode });
            }}
            disabled={
              compileMutation.isLoading ||
              compileProjectMutation.isLoading ||
              createCompileJobMutation.isLoading ||
              !!activeCompileJobId ||
              !compilerAvailable
            }
            title={!compilerAvailable ? 'Compiler unavailable/disabled' : 'Compile to PDF'}
          >
            {compileMutation.isLoading || compileProjectMutation.isLoading || createCompileJobMutation.isLoading || !!activeCompileJobId ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Compiling…
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Compile
              </>
            )}
          </Button>
        </div>
      </div>

      {showResearchEngineer && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-xl">
            <div className="p-6 border-b border-gray-200 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Start ResearchEngineer</h2>
                <p className="text-sm text-gray-500">
                  {researchEngineerWorkflow === 'loop'
                    ? 'Plan → Patch → Run → Patch → Run → Paper'
                    : 'AI Scientist → Code Patch → Paper Update'}
                </p>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setShowResearchEngineer(false)}>
                Close
              </Button>
            </div>

            <div className="p-6 space-y-4">
              {!selectedResearchEngineerChain && (
                <div className="text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded px-3 py-2">
                  ResearchEngineer chain not found. Ensure the backend has the built-in chain enabled.
                </div>
              )}

              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-700">Workflow</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm text-gray-700">
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="researchEngineerWorkflow"
                      value="loop"
                      checked={researchEngineerWorkflow === 'loop'}
                      onChange={() => setResearchEngineerWorkflow('loop')}
                      disabled={!researchEngineerLoopChain}
                    />
                    Loop
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="researchEngineerWorkflow"
                      value="simple"
                      checked={researchEngineerWorkflow === 'simple'}
                      onChange={() => setResearchEngineerWorkflow('simple')}
                      disabled={!researchEngineerChain}
                    />
                    Simple
                  </label>
                </div>
                <div className="text-xs text-gray-500">
                  Loop = runs tests/commands between patch rounds. Simple = faster, no experiments.
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Goal</label>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={3}
                  value={researchEngineerGoal}
                  onChange={(e) => setResearchEngineerGoal(e.target.value)}
                  placeholder="What are we trying to prove/build? What change should the patch implement?"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Knowledge DB search query (optional)</label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={researchEngineerSearchQuery}
                  onChange={(e) => setResearchEngineerSearchQuery(e.target.value)}
                  placeholder="Used to ground the Scientist step (defaults to Goal)"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Target git source</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={researchEngineerTargetSourceId}
                  onChange={(e) => setResearchEngineerTargetSourceId(e.target.value)}
                >
                  <option value="">Select a source…</option>
                  {(activeGitSources || [])
                    .map((x) => x.source)
                    .filter(Boolean)
                    .map((s) => (
                      <option key={String(s.id)} value={String(s.id)}>
                        {s.name} ({s.source_type})
                      </option>
                    ))}
                </select>
                <div className="mt-1 text-xs text-gray-500">
                  Uses the Code Patch Proposer against this DocumentSource.
                </div>
              </div>

              {researchEngineerWorkflow === 'loop' ? (
                <div className="space-y-2">
                  <div className="text-sm font-medium text-gray-700">Experiments</div>
                  <label className="flex items-center gap-2 text-sm text-gray-700">
                    <input
                      type="checkbox"
                      checked={researchEngineerEnableExperiments}
                      onChange={(e) => setResearchEngineerEnableExperiments(e.target.checked)}
                      disabled={unsafeExecStatus ? !unsafeExecStatus.enabled : false}
                    />
                    Enable experiments (unsafe execution)
                  </label>
                  <div className="text-xs text-gray-500">
                    If blank, the runner uses the patch’s suggested tests_to_run. One command per line.
                    {unsafeExecStatus && !unsafeExecStatus.enabled ? <span> (currently disabled)</span> : null}
                  </div>
                  <textarea
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                    rows={4}
                    value={researchEngineerCommands}
                    onChange={(e) => setResearchEngineerCommands(e.target.value)}
                    placeholder={'e.g.\npython -m pytest -q\nnpm test'}
                    disabled={!researchEngineerEnableExperiments}
                  />
                </div>
              ) : null}

              {researchEngineerWorkflow === 'loop' ? (
                <div className="space-y-2">
                  <div className="text-sm font-medium text-gray-700">Apply to Knowledge DB (optional)</div>
                  <label className="flex items-center gap-2 text-sm text-gray-700">
                    <input
                      type="checkbox"
                      checked={researchEngineerApplyPatchToKb}
                      onChange={(e) => setResearchEngineerApplyPatchToKb(e.target.checked)}
                    />
                    Run KB apply dry-run
                  </label>
                  <label className="flex items-center gap-2 text-sm text-gray-700">
                    <input
                      type="checkbox"
                      checked={researchEngineerApplyPatchToKbConfirm}
                      onChange={(e) => setResearchEngineerApplyPatchToKbConfirm(e.target.checked)}
                      disabled={!researchEngineerApplyPatchToKb}
                    />
                    Write changes to Knowledge DB
                  </label>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm text-gray-700">
                    <label className="flex items-center gap-2">
                      <span className="text-gray-500">Strategy</span>
                      <select
                        className="border border-gray-300 rounded px-2 py-1 text-sm"
                        value={researchEngineerProposalStrategy}
                        onChange={(e) => setResearchEngineerProposalStrategy(e.target.value as any)}
                        disabled={!researchEngineerApplyPatchToKb}
                      >
                        <option value="best_passing">best_passing</option>
                        <option value="latest">latest</option>
                      </select>
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={researchEngineerRequireExperimentsOk}
                        onChange={(e) => setResearchEngineerRequireExperimentsOk(e.target.checked)}
                        disabled={!researchEngineerApplyPatchToKbConfirm}
                      />
                      Require experiments pass
                    </label>
                  </div>
                  {researchEngineerApplyPatchToKbConfirm ? (
                    <div className="text-xs text-yellow-700 bg-yellow-50 border border-yellow-200 rounded px-3 py-2">
                      Writes update stored code documents in the Knowledge DB. A dry-run is executed first.
                    </div>
                  ) : researchEngineerApplyPatchToKb ? (
                    <div className="text-xs text-gray-500">Dry-run only (no writes).</div>
                  ) : null}
                </div>
              ) : null}
            </div>

            <div className="p-6 border-t border-gray-200 flex justify-end gap-3">
              <Button variant="secondary" onClick={() => setShowResearchEngineer(false)}>
                Cancel
              </Button>
              <Button
                onClick={() => startResearchEngineerMutation.mutate()}
                disabled={startResearchEngineerMutation.isLoading || !projectId || !selectedResearchEngineerChain}
              >
                {startResearchEngineerMutation.isLoading ? 'Starting…' : 'Start'}
              </Button>
            </div>
          </div>
        </div>
      )}

      {showPaperPipeline && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-xl">
            <div className="p-6 border-b border-gray-200 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Start PaperPipeline</h2>
                <p className="text-sm text-gray-500">Plan → Patch → Run → Cite → Review → Compile → Publish</p>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setShowPaperPipeline(false)}>
                Close
              </Button>
            </div>

            <div className="p-6 space-y-4">
              {!paperPipelineChain && (
                <div className="text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded px-3 py-2">
                  PaperPipeline chain not found. Ensure the backend has the built-in chain enabled.
                </div>
              )}

              {latexStatus && (latexStatus.admin_only || !latexStatus.enabled) && (
                <div className="text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded px-3 py-2">
                  {latexStatus.admin_only ? (
                    <div>Compile may be skipped unless you are an admin (server is configured as admin-only).</div>
                  ) : null}
                  {!latexStatus.enabled ? (
                    <div>Server-side compilation is disabled; compile/publish-to-PDF steps will be skipped.</div>
                  ) : null}
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Goal</label>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={3}
                  value={paperPipelineGoal}
                  onChange={(e) => setPaperPipelineGoal(e.target.value)}
                  placeholder="What are we trying to prove/build? What change should the patch implement?"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Knowledge DB search query (optional)</label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={paperPipelineSearchQuery}
                  onChange={(e) => setPaperPipelineSearchQuery(e.target.value)}
                  placeholder="Used to ground the Scientist step (defaults to Goal)"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Target git source</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={paperPipelineTargetSourceId}
                  onChange={(e) => setPaperPipelineTargetSourceId(e.target.value)}
                >
                  <option value="">Select a source…</option>
                  {(activeGitSources || [])
                    .map((x) => x.source)
                    .filter(Boolean)
                    .map((s) => (
                      <option key={String(s.id)} value={String(s.id)}>
                        {s.name} ({s.source_type})
                      </option>
                    ))}
                </select>
                <div className="mt-1 text-xs text-gray-500">
                  Used for the code patch proposer and experiment runner.
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Commands (optional, one per line)</label>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                  rows={3}
                  value={paperPipelineCommands}
                  onChange={(e) => setPaperPipelineCommands(e.target.value)}
                  placeholder={'e.g.\\npython -m pytest -q\\nnpm test'}
                />
                <div className="mt-1 text-xs text-gray-500">
                  If blank, the pipeline uses inherited tests_to_run from the patch step (when available).
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-700">Pipeline steps</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
                  <label className="flex items-center gap-2 text-gray-700">
                    <input
                      type="checkbox"
                      checked={paperPipelineEnableExperiments}
                      onChange={(e) => setPaperPipelineEnableExperiments(e.target.checked)}
                      disabled={unsafeExecStatus ? !unsafeExecStatus.enabled : false}
                    />
                    Experiments
                  </label>
                  <label className="flex items-center gap-2 text-gray-700">
                    <input
                      type="checkbox"
                      checked={paperPipelineEnableCitationSync}
                      onChange={(e) => setPaperPipelineEnableCitationSync(e.target.checked)}
                    />
                    CitationSync
                  </label>
                  <label className="flex items-center gap-2 text-gray-700">
                    <input
                      type="checkbox"
                      checked={paperPipelineEnableReviewer}
                      onChange={(e) => {
                        const next = e.target.checked;
                        setPaperPipelineEnableReviewer(next);
                        if (!next) setPaperPipelineApplyReviewDiff(false);
                      }}
                    />
                    Reviewer
                  </label>
                  <label className="flex items-center gap-2 text-gray-700">
                    <input
                      type="checkbox"
                      checked={paperPipelineEnableCompile}
                      onChange={(e) => setPaperPipelineEnableCompile(e.target.checked)}
                      disabled={!compilerAvailable || (latexStatus?.admin_only && user?.role !== 'admin')}
                    />
                    Compile
                  </label>
                  <label className="flex items-center gap-2 text-gray-700">
                    <input
                      type="checkbox"
                      checked={paperPipelineEnablePublish}
                      onChange={(e) => setPaperPipelineEnablePublish(e.target.checked)}
                    />
                    Publish
                  </label>
                </div>
                <div className="text-xs text-gray-500">
                  Note: Experiments require unsafe code execution to be enabled on the server.
                  {unsafeExecStatus && !unsafeExecStatus.enabled ? (
                    <span> (currently disabled)</span>
                  ) : null}
                  {unsafeExecStatus ? (
                    <span className="ml-1">
                      — backend: <span className="font-mono">{String(unsafeExecStatus.backend || 'unknown')}</span>
                      {unsafeExecStatus.limits?.timeout_seconds != null ? (
                        <>
                          {' '}
                          • timeout: <span className="font-mono">{String(unsafeExecStatus.limits.timeout_seconds)}s</span>
                        </>
                      ) : null}
                    </span>
                  ) : null}
                </div>
                {unsafeExecStatus && !unsafeExecStatus.enabled && user?.role === 'admin' ? (
                  <div className="pt-1">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => {
                        navigate('/settings?tab=admin');
                        setShowPaperPipeline(false);
                      }}
                    >
                      Open admin settings
                    </Button>
                  </div>
                ) : null}

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-2 text-sm">
                  <label className="flex items-center gap-2 text-gray-700">
                    <input
                      type="checkbox"
                      checked={paperPipelineCompileSafeMode}
                      onChange={(e) => setPaperPipelineCompileSafeMode(e.target.checked)}
                      disabled={!paperPipelineEnableCompile}
                    />
                    Compile safe mode
                  </label>
                  <div />
                  <label className="flex items-center gap-2 text-gray-700">
                    <input
                      type="checkbox"
                      checked={paperPipelinePublishIncludeTex}
                      onChange={(e) => setPaperPipelinePublishIncludeTex(e.target.checked)}
                      disabled={!paperPipelineEnablePublish}
                    />
                    Publish .tex
                  </label>
                  <label className="flex items-center gap-2 text-gray-700">
                    <input
                      type="checkbox"
                      checked={paperPipelinePublishIncludePdf}
                      onChange={(e) => setPaperPipelinePublishIncludePdf(e.target.checked)}
                      disabled={
                        !paperPipelineEnablePublish ||
                        (
                          !(currentProject?.pdf_file_path || currentProject?.pdf_download_url) &&
                          (!paperPipelineEnableCompile || !compilerAvailable || (latexStatus?.admin_only && user?.role !== 'admin'))
                        )
                      }
                    />
                    Publish PDF
                  </label>
                </div>
              </div>

              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={paperPipelineApplyReviewDiff}
                  onChange={(e) => setPaperPipelineApplyReviewDiff(e.target.checked)}
                  disabled={!paperPipelineEnableReviewer}
                />
                Auto-apply reviewer diff (optional)
              </label>

              {!paperPipelinePublishIncludeTex && !paperPipelinePublishIncludePdf && paperPipelineEnablePublish ? (
                <div className="text-xs text-yellow-700 bg-yellow-50 border border-yellow-200 rounded px-3 py-2">
                  Publish step is enabled but both outputs are unchecked. Nothing will be published.
                </div>
              ) : null}

              {paperPipelineEnablePublish &&
              paperPipelinePublishIncludePdf &&
              !(currentProject?.pdf_file_path || currentProject?.pdf_download_url) &&
              (!paperPipelineEnableCompile || !compilerAvailable || (latexStatus?.admin_only && user?.role !== 'admin')) ? (
                <div className="text-xs text-yellow-700 bg-yellow-50 border border-yellow-200 rounded px-3 py-2">
                  PDF publishing is selected, but no existing PDF is available and compile is disabled/unavailable. The publish step will skip PDF.
                </div>
              ) : null}
            </div>

            <div className="p-6 border-t border-gray-200 flex justify-end gap-3">
              <Button variant="secondary" onClick={() => setShowPaperPipeline(false)}>
                Cancel
              </Button>
              <Button
                onClick={() => startPaperPipelineMutation.mutate()}
                disabled={startPaperPipelineMutation.isLoading || !projectId || !paperPipelineChain}
              >
                {startPaperPipelineMutation.isLoading ? 'Starting…' : 'Start'}
              </Button>
            </div>
          </div>
        </div>
      )}

      {showCitationSync && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-xl">
            <div className="p-6 border-b border-gray-200 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Run CitationSync</h2>
                <p className="text-sm text-gray-500">
                  Sync \\cite{'{'}KDB:&lt;uuid&gt;{'}'} into refs.bib or thebibliography
                </p>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setShowCitationSync(false)}>
                Close
              </Button>
            </div>

            <div className="p-6 space-y-4">
              {!citationSyncTemplate && (
                <div className="text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded px-3 py-2">
                  CitationSync template not found. Ensure the backend has the built-in LaTeX templates enabled.
                </div>
              )}

              <div className="flex items-center gap-3 text-sm">
                <label className="flex items-center gap-2 text-gray-700">
                  <input type="radio" checked={citationSyncMode === 'bibtex'} onChange={() => setCitationSyncMode('bibtex')} />
                  BibTeX
                </label>
                <label className="flex items-center gap-2 text-gray-700">
                  <input
                    type="radio"
                    checked={citationSyncMode === 'thebibliography'}
                    onChange={() => setCitationSyncMode('thebibliography')}
                  />
                  thebibliography
                </label>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Bib filename</label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                  value={citationSyncBibFilename}
                  onChange={(e) => setCitationSyncBibFilename(e.target.value)}
                  disabled={citationSyncMode !== 'bibtex'}
                />
                <div className="mt-1 text-xs text-gray-500">Used only in BibTeX mode.</div>
              </div>

              {citationSyncJobId && (
                <div className="text-sm border rounded-lg p-3 bg-gray-50">
                  <div className="flex items-center justify-between">
                    <div className="font-medium text-gray-800">Job</div>
                    <div className="text-xs text-gray-600 font-mono">{citationSyncJobId}</div>
                  </div>
                  <div className="mt-1 text-xs text-gray-600">
                    Status: <span className="font-mono">{(citationSyncJob as any)?.status || '...'}</span>
                  </div>
                  {(citationSyncJob as any)?.results?.citation_sync && (
                    <div className="mt-2 text-xs text-gray-700 whitespace-pre-wrap">
                      Resolved: {(citationSyncJob as any)?.results?.citation_sync?.resolved_count ?? 0}
                      {' • '}
                      Updated .tex: {String(!!(citationSyncJob as any)?.results?.citation_sync?.updated_tex)}
                      {' • '}
                      Updated .bib: {String(!!(citationSyncJob as any)?.results?.citation_sync?.updated_bib)}
                    </div>
                  )}
                  <div className="mt-2 flex gap-2">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => navigate(`/autonomous-agents?job=${encodeURIComponent(citationSyncJobId)}`)}
                    >
                      Open job
                    </Button>
                  </div>
                </div>
              )}
            </div>

            <div className="p-6 border-t border-gray-200 flex justify-end gap-3">
              <Button variant="secondary" onClick={() => setShowCitationSync(false)}>
                Cancel
              </Button>
              <Button
                onClick={() => startCitationSyncMutation.mutate()}
                disabled={startCitationSyncMutation.isLoading || !projectId || !citationSyncTemplate}
              >
                {startCitationSyncMutation.isLoading ? 'Starting…' : 'Start'}
              </Button>
            </div>
          </div>
        </div>
      )}

      {showMathCopilot && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl">
            <div className="p-6 border-b border-gray-200 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">MathCopilot</h2>
                <p className="text-sm text-gray-500">Notation, units (siunitx), tensor shapes, and equation cross-refs</p>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setShowMathCopilot(false)}>
                Close
              </Button>
            </div>

            <div className="p-6 space-y-4 max-h-[70vh] overflow-auto">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Goal</label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={mathGoal}
                  onChange={(e) => setMathGoal(e.target.value)}
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm text-gray-700">
                <label className="flex items-center gap-2">
                  <input type="checkbox" checked={mathEnforceSiunitx} onChange={(e) => setMathEnforceSiunitx(e.target.checked)} />
                  Use siunitx for units
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={mathEnforceEquationLabels}
                    onChange={(e) => setMathEnforceEquationLabels(e.target.checked)}
                  />
                  Fix equation labels/refs
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" checked={mathEnforceShapes} onChange={(e) => setMathEnforceShapes(e.target.checked)} />
                  Enforce tensor shapes
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={mathEnforceBoldItalic}
                    onChange={(e) => setMathEnforceBoldItalic(e.target.checked)}
                  />
                  Enforce bold/italic conventions
                </label>
              </div>

              <div className="border rounded-lg p-3 bg-gray-50">
                <div className="flex items-center justify-between gap-2">
                  <div className="text-sm font-medium text-gray-800">Selection/context (optional)</div>
                  <Button variant="secondary" size="sm" onClick={refreshMathContextFromEditor}>
                    Refresh from editor
                  </Button>
                </div>
                <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2">
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Selection</div>
                    <pre className="text-xs bg-white border rounded p-2 overflow-auto max-h-40 whitespace-pre-wrap">
                      {(mathSelection || '').trim() ? mathSelection : '(none)'}
                    </pre>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Cursor context</div>
                    <pre className="text-xs bg-white border rounded p-2 overflow-auto max-h-40 whitespace-pre-wrap">
                      {(mathCursorContext || '').trim() ? mathCursorContext : '(none)'}
                    </pre>
                  </div>
                </div>
              </div>

              {mathResult ? (
                <div className="space-y-3">
                  {mathResult.notes?.trim() ? (
                    <div className="text-sm text-gray-700 whitespace-pre-wrap border rounded-lg p-3">{mathResult.notes}</div>
                  ) : null}

                  {mathResult.conventions && Object.keys(mathResult.conventions).length > 0 ? (
                    <div className="border rounded-lg p-3">
                      <div className="font-medium text-gray-900 mb-2">Conventions</div>
                      <ul className="list-disc pl-5 text-sm text-gray-700 space-y-1">
                        {Object.entries(mathResult.conventions)
                          .slice(0, 20)
                          .map(([k, v]) => (
                            <li key={k}>
                              <span className="font-mono">{k}</span>: <span className="font-mono">{v}</span>
                            </li>
                          ))}
                      </ul>
                    </div>
                  ) : null}

                  {Array.isArray(mathResult.suggestions) && mathResult.suggestions.length > 0 ? (
                    <div className="border rounded-lg p-3">
                      <div className="font-medium text-gray-900 mb-2">Suggestions</div>
                      <ul className="list-disc pl-5 text-sm text-gray-700 space-y-1">
                        {mathResult.suggestions.slice(0, 12).map((s, idx) => (
                          <li key={idx}>
                            <span className="font-medium">{String((s as any)?.title || (s as any)?.category || 'Suggestion')}</span>
                            {(s as any)?.text ? (
                              <div className="text-gray-700 whitespace-pre-wrap mt-1">{String((s as any)?.text || '')}</div>
                            ) : null}
                            {(s as any)?.insert_text ? (
                              <div className="mt-2">
                                <Button
                                  size="sm"
                                  variant="secondary"
                                  onClick={() => insertAtCursor(String((s as any)?.insert_text || ''))}
                                >
                                  Insert
                                </Button>
                              </div>
                            ) : null}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null}

                  {mathResult.diff_unified?.trim() ? (
                    <div className="border rounded-lg p-3">
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <div className="font-medium text-gray-900">Suggested diff (paper.tex)</div>
                        <div className="text-xs text-gray-600 font-mono">
                          {mathResult.diff_applies ? 'applies' : 'not validated'}
                        </div>
                      </div>
                      {Array.isArray(mathResult.diff_warnings) && mathResult.diff_warnings.length > 0 ? (
                        <div className="text-xs text-yellow-800 bg-yellow-50 border border-yellow-200 rounded px-2 py-1 mb-2 whitespace-pre-wrap">
                          {mathResult.diff_warnings.slice(0, 6).join('\n')}
                        </div>
                      ) : null}
                      <pre className="text-xs bg-gray-50 border rounded p-2 overflow-auto max-h-64 whitespace-pre-wrap">
                        {mathResult.diff_unified}
                      </pre>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>

            <div className="p-6 border-t border-gray-200 flex justify-end gap-3">
              <Button variant="secondary" onClick={() => setShowMathCopilot(false)}>
                Close
              </Button>
              <Button
                variant="secondary"
                onClick={() => mathCopilotMutation.mutate({ mode: 'autocomplete' })}
                disabled={mathCopilotMutation.isLoading}
              >
                {mathCopilotMutation.isLoading ? 'Working…' : 'Autocomplete'}
              </Button>
              <Button
                variant="secondary"
                onClick={() => mathCopilotMutation.mutate({ mode: 'analyze' })}
                disabled={mathCopilotMutation.isLoading}
              >
                {mathCopilotMutation.isLoading ? 'Working…' : 'Analyze'}
              </Button>
              <Button
                variant="secondary"
                onClick={() => applyMathToEditorMutation.mutate()}
                disabled={applyMathToEditorMutation.isLoading || !mathResult?.tex_source_patched || mathLastMode !== 'analyze'}
                title={mathLastMode !== 'analyze' ? 'Run Analyze first' : !mathResult?.tex_source_patched ? 'No patched source returned' : 'Apply to editor (local)'}
              >
                Apply to editor
              </Button>
              <Button
                onClick={() => applyMathAndSaveMutation.mutate()}
                disabled={applyMathAndSaveMutation.isLoading || !projectId || !mathResult?.tex_source_patched || mathLastMode !== 'analyze'}
                title={!projectId ? 'Save the project first' : 'Apply and save to project'}
              >
                {applyMathAndSaveMutation.isLoading ? 'Applying…' : 'Apply & Save'}
              </Button>
            </div>
          </div>
        </div>
      )}

      {showReviewer && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl">
            <div className="p-6 border-b border-gray-200 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Run Reviewer/Critic</h2>
                <p className="text-sm text-gray-500">Get a minimal unified diff suggestion for paper.tex</p>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setShowReviewer(false)}>
                Close
              </Button>
            </div>

            <div className="p-6 space-y-4 max-h-[70vh] overflow-auto">
              {!reviewerTemplate && (
                <div className="text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded px-3 py-2">
                  Reviewer template not found. Ensure the backend has the built-in LaTeX templates enabled.
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Focus (optional)</label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={reviewFocus}
                  onChange={(e) => setReviewFocus(e.target.value)}
                  placeholder="e.g. notation consistency; missing citations; abstract clarity"
                />
              </div>

              {reviewJobId && (
                <div className="text-sm border rounded-lg p-3 bg-gray-50">
                  <div className="flex items-center justify-between">
                    <div className="font-medium text-gray-800">Job</div>
                    <div className="text-xs text-gray-600 font-mono">{reviewJobId}</div>
                  </div>
                  <div className="mt-1 text-xs text-gray-600">
                    Status: <span className="font-mono">{(reviewerJob as any)?.status || '...'}</span>
                  </div>
                  <div className="mt-2 flex gap-2">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => navigate(`/autonomous-agents?job=${encodeURIComponent(reviewJobId)}`)}
                    >
                      Open job
                    </Button>
                  </div>
                </div>
              )}

              {(reviewerJob as any)?.results?.latex_review && (
                <div className="space-y-3">
                  <div className="border rounded-lg p-3">
                    <div className="font-medium text-gray-900 mb-2">Issues</div>
                    {Array.isArray((reviewerJob as any)?.results?.latex_review?.issues) &&
                    ((reviewerJob as any)?.results?.latex_review?.issues as any[]).length > 0 ? (
                      <ul className="list-disc pl-5 text-sm text-gray-700 space-y-1">
                        {((reviewerJob as any)?.results?.latex_review?.issues as any[]).slice(0, 12).map((it, idx) => (
                          <li key={idx}>
                            <span className="font-medium">{String(it?.category || 'Issue')}</span>
                            {it?.severity ? ` (${String(it.severity)})` : ''}: {String(it?.message || '')}
                            {it?.location_hint ? ` — ${String(it.location_hint)}` : ''}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div className="text-sm text-gray-600">No issues returned.</div>
                    )}
                  </div>

                  <div className="border rounded-lg p-3">
                    <div className="font-medium text-gray-900 mb-2">Suggested diff (paper.tex)</div>
                    <pre className="text-xs bg-gray-50 border rounded p-2 overflow-auto whitespace-pre-wrap font-mono max-h-64">
                      {String((reviewerJob as any)?.results?.latex_review?.diff_unified || '')}
                    </pre>
                    <div className="mt-2 flex justify-end gap-2">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => {
                          const diff = String((reviewerJob as any)?.results?.latex_review?.diff_unified || '').trim();
                          navigator.clipboard
                            .writeText(diff)
                            .then(() => toast.success('Copied diff'))
                            .catch(() => toast.error('Copy failed'));
                        }}
                      >
                        Copy diff
                      </Button>
                      <Button
                        size="sm"
                        onClick={() => {
                          const diff = String((reviewerJob as any)?.results?.latex_review?.diff_unified || '').trim();
                          if (!diff) {
                            toast.error('No diff to apply');
                            return;
                          }
                          applyReviewerDiffMutation.mutate(diff);
                        }}
                        disabled={applyReviewerDiffMutation.isLoading}
                      >
                        {applyReviewerDiffMutation.isLoading ? 'Applying…' : 'Apply diff'}
                      </Button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="p-6 border-t border-gray-200 flex justify-end gap-3">
              <Button variant="secondary" onClick={() => setShowReviewer(false)}>
                Cancel
              </Button>
              <Button
                onClick={() => startReviewerMutation.mutate()}
                disabled={startReviewerMutation.isLoading || !projectId || !reviewerTemplate}
              >
                {startReviewerMutation.isLoading ? 'Starting…' : 'Start'}
              </Button>
            </div>
          </div>
        </div>
      )}

      <div className="mb-3 flex items-center gap-3 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-gray-700">Project</span>
          <select
            className="border border-gray-300 rounded-lg px-2 py-1 text-sm bg-white"
            value={projectId || ''}
            onChange={(e) => {
              const nextId = (e.target.value || '').trim();
              const next = new URLSearchParams(searchParams);
              if (nextId) next.set('project', nextId);
              else next.delete('project');
              setSearchParams(next, { replace: true });
            }}
          >
            <option value="">(unsaved)</option>
            {(projectsData?.items || []).map((p) => (
              <option key={p.id} value={p.id}>
                {p.title}
              </option>
            ))}
          </select>
          <input
            className="border border-gray-300 rounded-lg px-2 py-1 text-sm w-64"
            value={projectTitle}
            onChange={(e) => setProjectTitle(e.target.value)}
            placeholder="Project title"
          />
        </div>
        <label className="flex items-center gap-2 text-gray-700">
          <input
            type="checkbox"
            checked={safeMode}
            onChange={(e) => setSafeMode(e.target.checked)}
          />
          Safe mode
        </label>
        <div className="flex items-center gap-2">
          <span className="text-gray-700">Publish</span>
          <label className="flex items-center gap-2 text-gray-700">
            <input type="checkbox" checked={publishIncludeTex} onChange={(e) => setPublishIncludeTex(e.target.checked)} />
            .tex
          </label>
          <label className="flex items-center gap-2 text-gray-700">
            <input type="checkbox" checked={publishIncludePdf} onChange={(e) => setPublishIncludePdf(e.target.checked)} />
            PDF
          </label>
          <input
            className="border border-gray-300 rounded-lg px-2 py-1 text-sm w-56"
            value={publishTags}
            onChange={(e) => setPublishTags(e.target.value)}
            placeholder="tags (comma-separated)"
            title="Tags applied to published Knowledge DB documents"
          />
        </div>
        {latexStatus?.admin_only && (
          <div className="text-xs text-gray-500">
            Compile is admin-only
          </div>
        )}
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0">
        <div className="bg-white border rounded-lg overflow-hidden flex flex-col min-h-0">
          <div className="px-4 py-3 border-b flex items-center justify-between">
            <div className="font-medium text-gray-900">Editor</div>
            <div className="flex items-center gap-3">
              {projectId && (
                <label className="text-xs px-2 py-1 rounded-lg border border-gray-300 hover:bg-gray-50 cursor-pointer flex items-center gap-2">
                  <Upload className="w-3 h-3" />
                  Add file
                  <input
                    type="file"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files && e.target.files[0];
                      e.target.value = '';
                      if (!f) return;
                      uploadFileMutation.mutate(f);
                    }}
                  />
                </label>
              )}
              <div className="text-xs text-gray-500">{texSource.length.toLocaleString()} chars</div>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <textarea
              ref={textareaRef}
              className="w-full h-full resize-none p-4 font-mono text-sm outline-none"
              value={texSource}
              onChange={(e) => setTexSource(e.target.value)}
              spellCheck={false}
            />
          </div>
        </div>

        <div className="bg-white border rounded-lg overflow-hidden flex flex-col min-h-0">
          <div className="px-4 py-3 border-b flex items-center justify-between">
            <div className="font-medium text-gray-900">Preview</div>
            <div className="text-xs text-gray-500">{pdfUrl ? 'PDF ready' : 'No PDF yet'}</div>
          </div>
          <div className="flex-1 min-h-0 bg-gray-50">
            {pdfUrl ? (
              <iframe title="pdf-preview" src={pdfUrl} className="w-full h-full" />
            ) : (
              <div className="h-full flex items-center justify-center text-sm text-gray-500">
                Compile to see PDF preview
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="font-medium text-gray-900">Copilot</div>
            <Button
              variant="secondary"
              size="sm"
                  onClick={() =>
                    copilotMutation.mutate({
                      prompt: copilotPrompt.trim(),
                      search_query: copilotSearchQuery.trim() || undefined,
                      citation_mode: copilotCitationMode,
                    })
                  }
                  disabled={!copilotEnabled || copilotMutation.isLoading || !copilotPrompt.trim()}
                >
              {copilotMutation.isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Thinking…
                </>
              ) : (
                <>
                  <Wand2 className="w-4 h-4 mr-2" />
                  Generate LaTeX
                </>
              )}
            </Button>
          </div>

          <div className="space-y-3">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">What should I write?</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={3}
                value={copilotPrompt}
                onChange={(e) => setCopilotPrompt(e.target.value)}
                placeholder="Write an Introduction about … include citations."
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Optional: knowledge base search query</label>
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={copilotSearchQuery}
                onChange={(e) => setCopilotSearchQuery(e.target.value)}
                placeholder="e.g. RAG reranking cross-encoder"
              />
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <label className="block text-xs font-medium text-gray-600">Citations</label>
                <select
                  className="border border-gray-300 rounded-lg px-2 py-1 text-sm bg-white"
                  value={copilotCitationMode}
                  onChange={(e) => setCopilotCitationMode(e.target.value as any)}
                >
                  <option value="thebibliography">thebibliography</option>
                  <option value="bibtex">BibTeX (.bib)</option>
                </select>
              </div>
              {copilotCitationMode === 'bibtex' && (
                <div className="flex items-center gap-2">
                  <label className="block text-xs font-medium text-gray-600">Bib file</label>
                  <input
                    className="border border-gray-300 rounded-lg px-2 py-1 text-sm w-40"
                    value={bibFilename}
                    onChange={(e) => setBibFilename(e.target.value)}
                    placeholder="refs.bib"
                  />
                </div>
              )}
              {latexStatus?.available_tools?.bibtex === false && copilotCitationMode === 'bibtex' && (
                <div className="text-xs text-yellow-700">Server has no `bibtex`; compilation may fail.</div>
              )}
            </div>

            {copilotResult && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    onClick={() => insertAtCursor(`\n\n% --- Copilot snippet ---\n${copilotResult.snippet}\n`)}
                  >
                    Insert snippet
                  </Button>
                  {copilotCitationMode === 'thebibliography' && copilotResult.bibtex?.trim() && (
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => setTexSource((prev) => insertBeforeEndDocument(prev, copilotResult.bibtex))}
                      title="Insert references block before \\end{document}"
                    >
                      Insert references
                    </Button>
                  )}
                  {copilotCitationMode === 'thebibliography' && copilotResult.bibtex?.trim() && copilotResult.snippet?.trim() && (
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => {
                        const combined = `${copilotResult.snippet}\n\n${copilotResult.bibtex}`;
                        setTexSource((prev) => insertBeforeEndDocument(prev, combined));
                      }}
                      title="Insert snippet + references before \\end{document}"
                    >
                      Insert both
                    </Button>
                  )}
                  {copilotCitationMode === 'thebibliography' && copilotResult.bibtex?.trim() && (
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => {
                        navigator.clipboard
                          .writeText(copilotResult.bibtex)
                          .then(() => toast.success('References copied'))
                          .catch(() => toast.error('Copy failed'));
                      }}
                    >
                      Copy references
                    </Button>
                  )}
                  {copilotCitationMode === 'bibtex' && copilotResult.bibtex?.trim() && projectId && (
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={async () => {
                        const name = (bibFilename || 'refs.bib').trim();
                        if (!name || name.includes('/') || name.includes('\\')) {
                          toast.error('Invalid bib filename');
                          return;
                        }
                        const file = new File([copilotResult.bibtex], name, { type: 'text/x-bibtex' });
                        uploadFileMutation.mutate(file);
                      }}
                      title="Create/replace the project .bib file with these entries"
                    >
                      Update .bib
                    </Button>
                  )}
                  {copilotCitationMode === 'bibtex' && (
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => {
                        const base = (bibFilename || 'refs.bib').trim();
                        const stem = base.endsWith('.bib') ? base.slice(0, -4) : base;
                        const scaffold = `\\bibliographystyle{plain}\n\\bibliography{${stem}}`;
                        setTexSource((prev) => insertBeforeEndDocument(prev, scaffold));
                      }}
                      title="Insert \\bibliography{...} before \\end{document}"
                    >
                      Insert \\bibliography
                    </Button>
                  )}
                </div>
                <div className="text-xs text-gray-500">Snippet preview</div>
                <pre className="text-xs bg-gray-50 border rounded p-2 overflow-auto max-h-40 whitespace-pre-wrap">{copilotResult.snippet}</pre>
                {copilotResult.bibtex?.trim() && copilotCitationMode === 'thebibliography' && (
                  <>
                    <div className="text-xs text-gray-500">References snippet (paste near the end of the document)</div>
                    <pre className="text-xs bg-gray-50 border rounded p-2 overflow-auto max-h-40 whitespace-pre-wrap">{copilotResult.bibtex}</pre>
                  </>
                )}
                {copilotResult.bibtex?.trim() && copilotCitationMode === 'bibtex' && (
                  <>
                    <div className="text-xs text-gray-500">BibTeX entries (write into your .bib file)</div>
                    <pre className="text-xs bg-gray-50 border rounded p-2 overflow-auto max-h-40 whitespace-pre-wrap">{copilotResult.bibtex}</pre>
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="bg-white border rounded-lg p-4">
          <div className="font-medium text-gray-900 mb-2">Compile log</div>
          <div className="flex items-center justify-between mb-2">
            <div className="text-xs text-gray-500">Use “Fix errors” to let the copilot propose a minimal patch.</div>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => fixMutation.mutate()}
              disabled={fixMutation.isLoading || !(compileLog || '').trim()}
              title={!(compileLog || '').trim() ? 'Compile (or paste a log) first' : 'Suggest a fix'}
            >
              {fixMutation.isLoading ? 'Fixing…' : 'Fix errors'}
            </Button>
          </div>
          <pre className="text-xs bg-gray-50 border rounded p-2 overflow-auto max-h-60 whitespace-pre-wrap">
            {compileLog || 'No log yet.'}
          </pre>
          {fixNotes && (
            <pre className="mt-2 text-xs bg-gray-50 border rounded p-2 overflow-auto max-h-40 whitespace-pre-wrap">
              {fixNotes}
            </pre>
          )}

          {publishResult && (
            <div className="mt-3">
              <div className="font-medium text-gray-900 mb-2">Publish result</div>
              <div className="text-xs text-gray-600 mb-2">
                Published: {publishResult.published.length} • Skipped: {publishResult.skipped.length}
              </div>
              <div className="space-y-2">
                {publishResult.published.map((it) => (
                  <div key={it.document_id} className="text-xs flex items-center justify-between gap-3 border rounded px-2 py-1">
                    <div className="truncate">
                      <span className="font-mono mr-2">{it.kind}</span>
                      <span className="text-gray-800">{it.title}</span>
                      <span className="text-gray-500 ml-2 font-mono">{it.document_id}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => {
                          const state: DocumentsLocationState = { openDocId: it.document_id };
                          navigate('/documents', { state });
                        }}
                      >
                        Open
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => {
                          navigator.clipboard
                            .writeText(it.document_id)
                            .then(() => toast.success('Document ID copied'))
                            .catch(() => toast.error('Copy failed'));
                        }}
                      >
                        Copy ID
                      </Button>
                    </div>
                  </div>
                ))}
                {publishResult.skipped.map((s, idx) => (
                  <div key={`${s.kind}-${idx}`} className="text-xs text-gray-600">
                    Skipped {s.kind}: {s.reason}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {projectId && (
        <div className="mt-4 bg-white border rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="font-medium text-gray-900">Project files</div>
            <div className="text-xs text-gray-500">
              Use in LaTeX via <span className="font-mono">\\includegraphics</span>, <span className="font-mono">\\input</span>, etc.
            </div>
          </div>
          <div className="space-y-2">
            {(projectFilesData?.items || []).length === 0 ? (
              <div className="text-sm text-gray-600">No files uploaded yet.</div>
            ) : (
              (projectFilesData?.items || []).map((f) => (
                <div key={f.id} className="flex items-center justify-between gap-3 border rounded px-3 py-2">
                  <div className="min-w-0">
                    <div className="text-sm text-gray-900 truncate font-mono">{f.filename}</div>
                    <div className="text-xs text-gray-500 truncate">
                      {(f.content_type || 'unknown')} • {(f.file_size || 0).toLocaleString()} bytes
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {f.download_url && (
                      <Button size="sm" variant="secondary" onClick={() => window.open(f.download_url as string, '_blank')}>
                        Download
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => {
                        if (!window.confirm(`Delete ${f.filename}?`)) return;
                        deleteFileMutation.mutate(f.id);
                      }}
                      disabled={deleteFileMutation.isLoading}
                      title="Delete file"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      <div className="mt-4 bg-white border rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="font-medium text-gray-900 flex items-center gap-2">
            <Quote className="w-4 h-4" />
            Cite from Knowledge DB
          </div>
          <div className="text-xs text-gray-500">Search KB → select docs → generate \\cite + bibliography</div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div>
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                className="w-full border border-gray-300 rounded-lg pl-9 pr-3 py-2 text-sm"
                value={citeQuery}
                onChange={(e) => setCiteQuery(e.target.value)}
                placeholder="Search documents to cite…"
              />
              {isCiteSearching && <Loader2 className="w-4 h-4 absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 animate-spin" />}
            </div>

            <div className="mt-2 space-y-2">
              {citeResults.length === 0 ? (
                <div className="text-sm text-gray-600">Enter a query (min 2 chars) to search.</div>
              ) : (
                citeResults.map((r) => {
                  const checked = !!selectedCiteDocIds[r.id];
                  return (
                    <label key={r.id} className="block border rounded-lg px-3 py-2 hover:bg-gray-50 cursor-pointer">
                      <div className="flex items-start gap-2">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={(e) => setSelectedCiteDocIds((prev) => ({ ...prev, [r.id]: e.target.checked }))}
                        />
                        <div className="min-w-0">
                          <div className="text-sm text-gray-900 truncate">{r.title}</div>
                          <div className="text-xs text-gray-500 truncate">{r.source} {r.url ? `• ${r.url}` : ''}</div>
                          <div className="text-xs text-gray-600 line-clamp-2">{r.snippet}</div>
                        </div>
                      </div>
                    </label>
                  );
                })
              )}
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input type="radio" checked={citeMode === 'bibtex'} onChange={() => setCiteMode('bibtex')} />
                BibTeX
              </label>
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input type="radio" checked={citeMode === 'thebibliography'} onChange={() => setCiteMode('thebibliography')} />
                thebibliography
              </label>
              <div className="text-xs text-gray-500">
                Key file: <span className="font-mono">{bibFilename}</span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
              <label className="flex items-center gap-2 text-gray-700">
                <input type="checkbox" checked={citeAutoInsert} onChange={(e) => setCiteAutoInsert(e.target.checked)} />
                Insert \\cite at cursor
              </label>
              <label className="flex items-center gap-2 text-gray-700">
                <input
                  type="checkbox"
                  checked={citeAutoInsertBibliography}
                  onChange={(e) => setCiteAutoInsertBibliography(e.target.checked)}
                  disabled={citeMode !== 'bibtex'}
                />
                Insert \\bibliography scaffold
              </label>
              <label className="flex items-center gap-2 text-gray-700">
                <input
                  type="checkbox"
                  checked={citeAutoUpdateBib}
                  onChange={(e) => setCiteAutoUpdateBib(e.target.checked)}
                  disabled={citeMode !== 'bibtex' || !projectId}
                />
                Update project .bib
              </label>
              <label className="flex items-center gap-2 text-gray-700">
                <input
                  type="checkbox"
                  checked={citeAutoInsertThebibliography}
                  onChange={(e) => setCiteAutoInsertThebibliography(e.target.checked)}
                  disabled={citeMode !== 'thebibliography'}
                />
                Insert thebibliography block
              </label>
            </div>

            <div className="flex items-center gap-2">
              <Button
                onClick={() => {
                  const ids = Object.entries(selectedCiteDocIds)
                    .filter(([, v]) => v)
                    .map(([k]) => k);
                  if (ids.length === 0) {
                    toast.error('Select at least one document');
                    return;
                  }
                  citeMutation.mutate(ids);
                }}
                disabled={citeMutation.isLoading}
              >
                {citeMutation.isLoading ? 'Generating…' : 'Generate citations'}
              </Button>
              {lastCitations?.cite_command && (
                <Button
                  variant="secondary"
                  onClick={() => {
                    navigator.clipboard
                      .writeText(lastCitations.cite_command)
                      .then(() => toast.success('Copied \\cite'))
                      .catch(() => toast.error('Copy failed'));
                  }}
                >
                  Copy \\cite
                </Button>
              )}
            </div>

            {lastCitations && (
              <div className="text-xs bg-gray-50 border rounded p-2 overflow-auto max-h-56 whitespace-pre-wrap">
                <div className="text-gray-700 mb-1">Cite:</div>
                <div className="font-mono mb-2">{lastCitations.cite_command}</div>
                {lastCitations.mode === 'bibtex' && lastCitations.bibtex_entries && (
                  <>
                    <div className="text-gray-700 mb-1">BibTeX entries:</div>
                    <pre className="font-mono whitespace-pre-wrap">{lastCitations.bibtex_entries}</pre>
                  </>
                )}
                {lastCitations.mode === 'thebibliography' && lastCitations.references_tex && (
                  <>
                    <div className="text-gray-700 mb-1">References block:</div>
                    <pre className="font-mono whitespace-pre-wrap">{lastCitations.references_tex}</pre>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LatexStudioPage;
