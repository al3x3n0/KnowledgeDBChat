/**
 * Coding Backlog -- work the swarm proposed, and what a person decided about it.
 *
 * This was a tab on the Runs page while Patch PRs was already its own
 * destination, so coding work was split across a page and a tab on an unrelated
 * one. It now sits beside Patch PRs, which is the rest of the same pipeline: an
 * item here becomes a patch there.
 *
 * Runs keeps its own `coding-backlog-items` query. That is not duplication --
 * the swarm and outcomes tabs derive `backlogBySwarmJobId` from it to show what
 * a swarm produced, and they need every item regardless of the filters someone
 * set here. Two readers with genuinely different questions, one endpoint, and
 * react-query keys that differ by scope so neither disturbs the other.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import toast from 'react-hot-toast';

import CodingBacklogTab from '../components/agent/tabs/CodingBacklogTab';
import useSwarmOutcomes from '../components/agent/useSwarmOutcomes';
import { useAuth } from '../contexts/AuthContext';
import { apiClient } from '../services/api';
import type { CodingBacklogItem, CodingBacklogItemCreate, User } from '../types';

const CodingBacklogPage: React.FC = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { user } = useAuth();

  const [backlogVisibilityScope, setBacklogVisibilityScope] = useState<'mine' | 'shared' | 'all'>('mine');
  const [backlogAssignmentFilter, setBacklogAssignmentFilter] = useState<string>('');
  const [backlogQueueStateFilter, setBacklogQueueStateFilter] = useState<string>('');
  const [backlogTitle, setBacklogTitle] = useState('');
  const [backlogGoal, setBacklogGoal] = useState('');
  const [backlogSourceId, setBacklogSourceId] = useState('');
  const [backlogFailureSymptom, setBacklogFailureSymptom] = useState('');
  const [backlogCommandsText, setBacklogCommandsText] = useState('');
  const [backlogFilePathsText, setBacklogFilePathsText] = useState('');
  const [backlogNoteDrafts, setBacklogNoteDrafts] = useState<Record<string, string>>({});
  const [backlogCloseReasonDrafts, setBacklogCloseReasonDrafts] = useState<Record<string, string>>({});

  const { data: documentSources } = useQuery(
    ['document-sources', 'all'],
    () => apiClient.getDocumentSources(),
    { staleTime: 30000 }
  );

  const codeSources = useMemo(() => {
    const items = (documentSources || []) as any[];
    return items.filter((s) => ['github', 'gitlab'].includes(String(s?.source_type || s?.sourceType || '').toLowerCase()));
  }, [documentSources]);

  const { data: collaborationUsersData } = useQuery(
    ['collaboration-users'],
    () =>
      typeof (apiClient as any).listCollaborationUsers === 'function'
        ? apiClient.listCollaborationUsers('', 1, 100)
        : Promise.resolve({ items: [], total: 0, page: 1, page_size: 100 }),
    { staleTime: 30000 }
  );

  const collaborationUsers = useMemo(
    () => (((collaborationUsersData as any)?.items || []) as User[]),
    [collaborationUsersData]
  );

  const collaborationUserById = useMemo(
    () => Object.fromEntries(collaborationUsers.map((candidate) => [String(candidate.id), candidate] as const)) as Record<string, User>,
    [collaborationUsers]
  );

  const userLabelById = useCallback(
    (candidateId?: string | null) => {
      const normalized = String(candidateId || '').trim();
      if (!normalized) return '';
      if (normalized === String(user?.id || '')) return 'You';
      const candidate = collaborationUserById[normalized];
      return String(candidate?.full_name || candidate?.username || candidate?.email || normalized).trim();
    },
    [collaborationUserById, user]
  );

  const createCodingBacklogMutation = useMutation(
    (data: CodingBacklogItemCreate) => apiClient.createCodingBacklogItem(data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['coding-backlog-items']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Coding backlog item created');
        setBacklogTitle('');
        setBacklogGoal('');
        setBacklogFailureSymptom('');
        setBacklogCommandsText('');
        setBacklogFilePathsText('');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to create coding backlog item');
      },
    }
  );

  /** Links back to Runs: a backlog item's job lives there. */
  const buildAutonomousAgentsUrl = useCallback(
    (jobId?: string, extras?: Record<string, string | null | undefined>) => {
      const next = new URLSearchParams();
      if (jobId && String(jobId).trim()) next.set('job', String(jobId).trim());
      Object.entries(extras || {}).forEach(([key, value]) => {
        const text = String(value ?? '').trim();
        if (text) next.set(key, text);
      });
      const query = next.toString();
      return query ? `/autonomous-agents?${query}` : '/autonomous-agents';
    },
    []
  );

  // The swarm outcome each backlog item came from. Passing 'outcomes' asks the
  // hook for the same data the Runs page's outcomes tab uses.
  const { swarmOutcomeBySwarmJobId } = useSwarmOutcomes('outcomes');

  const { data: codingBacklogData, isLoading: codingBacklogLoading, refetch: refetchCodingBacklog } = useQuery(
    ['coding-backlog-items', backlogVisibilityScope, backlogAssignmentFilter],
    () => apiClient.listCodingBacklogItems({
      limit: 100,
      offset: 0,
      visibility_scope: backlogVisibilityScope,
      assigned_user_id: backlogAssignmentFilter || undefined,
    }),
    { refetchInterval: 15000 }
  );

  const backlogItems = useMemo(
    () => (((codingBacklogData as any)?.items || []) as CodingBacklogItem[]),
    [codingBacklogData]
  );

  // Default the repository to the first one available rather than making
  // everyone pick it every time.
  useEffect(() => {
    if (!backlogSourceId && codeSources.length > 0) {
      setBacklogSourceId(String((codeSources[0] as any)?.id || ''));
    }
  }, [backlogSourceId, codeSources]);

  return (
    <CodingBacklogTab
      backlogAssignmentFilter={backlogAssignmentFilter}
      backlogCloseReasonDrafts={backlogCloseReasonDrafts}
      backlogCommandsText={backlogCommandsText}
      backlogFailureSymptom={backlogFailureSymptom}
      backlogFilePathsText={backlogFilePathsText}
      backlogGoal={backlogGoal}
      backlogItems={backlogItems}
      backlogNoteDrafts={backlogNoteDrafts}
      backlogQueueStateFilter={backlogQueueStateFilter}
      backlogSourceId={backlogSourceId}
      backlogTitle={backlogTitle}
      backlogVisibilityScope={backlogVisibilityScope}
      buildAutonomousAgentsUrl={buildAutonomousAgentsUrl}
      codeSources={codeSources}
      codingBacklogData={codingBacklogData}
      codingBacklogLoading={codingBacklogLoading}
      collaborationUsers={collaborationUsers}
      createCodingBacklogMutation={createCodingBacklogMutation}
      navigate={navigate}
      queryClient={queryClient}
      refetchCodingBacklog={refetchCodingBacklog}
      setActiveTab={() => { /* no tabs here; links go to Runs */ }}
      setBacklogAssignmentFilter={setBacklogAssignmentFilter}
      setBacklogCloseReasonDrafts={setBacklogCloseReasonDrafts}
      setBacklogCommandsText={setBacklogCommandsText}
      setBacklogFailureSymptom={setBacklogFailureSymptom}
      setBacklogFilePathsText={setBacklogFilePathsText}
      setBacklogGoal={setBacklogGoal}
      setBacklogNoteDrafts={setBacklogNoteDrafts}
      setBacklogQueueStateFilter={setBacklogQueueStateFilter}
      setBacklogSourceId={setBacklogSourceId}
      setBacklogTitle={setBacklogTitle}
      setBacklogVisibilityScope={setBacklogVisibilityScope}
      setSelectedJob={() => { /* the job opens on Runs, via buildAutonomousAgentsUrl */ }}
      swarmOutcomeBySwarmJobId={swarmOutcomeBySwarmJobId}
      user={user}
      userLabelById={userLabelById}
    />
  );
};

export default CodingBacklogPage;
