/**
 * Everything the swarm-outcomes view needs, in one place.
 *
 * These five filters, the query they key, and the three maps derived from its
 * response were spread across four regions of a 16.5k-line page -- declared at
 * line 1198, queried at 2638, reduced at 7130, rendered at 12734. Nothing else
 * on the page touched the filters, so they had no business being page state.
 *
 * The maps are the reason this is a hook rather than a component: the panel is
 * not their only consumer. `swarmOutcomeBySwarmJobId` is read by the backlog
 * view and both maps are handed to `JobDetailPanel`, so moving the query into
 * the panel would have quietly emptied them. Owning the cluster here keeps one
 * source of truth and still leaves the page able to read it.
 *
 * `activeTab` is passed in only to gate fetching, exactly as before.
 */

import { useMemo, useState } from 'react';
import { useQuery } from 'react-query';

import { apiClient } from '../../services/api';
import type { AgentJobSwarmOutcomeCase } from '../../types';

export interface SwarmOutcomesState {
  swarmOutcomePresetFilter: string;
  setSwarmOutcomePresetFilter: (value: string) => void;
  swarmOutcomeTerminalFilter: string;
  setSwarmOutcomeTerminalFilter: (value: string) => void;
  swarmOutcomePromotionFilter: string;
  setSwarmOutcomePromotionFilter: (value: string) => void;
  swarmOutcomeDateRange: string;
  setSwarmOutcomeDateRange: (value: string) => void;
  swarmOutcomeVisibilityScope: 'mine' | 'shared' | 'all';
  setSwarmOutcomeVisibilityScope: (value: 'mine' | 'shared' | 'all') => void;
  swarmOutcomeAnalyticsData: unknown;
  swarmOutcomeAnalyticsLoading: boolean;
  refetchSwarmOutcomeAnalytics: () => void;
  swarmOutcomeCases: AgentJobSwarmOutcomeCase[];
  swarmOutcomeBySwarmJobId: Record<string, AgentJobSwarmOutcomeCase>;
  swarmOutcomeByRepairJobId: Record<string, AgentJobSwarmOutcomeCase>;
}

export function useSwarmOutcomes(activeTab: string): SwarmOutcomesState {
  const [swarmOutcomePresetFilter, setSwarmOutcomePresetFilter] = useState<string>('');
  const [swarmOutcomeTerminalFilter, setSwarmOutcomeTerminalFilter] = useState<string>('');
  const [swarmOutcomePromotionFilter, setSwarmOutcomePromotionFilter] = useState<string>('');
  const [swarmOutcomeDateRange, setSwarmOutcomeDateRange] = useState<string>('all');
  const [swarmOutcomeVisibilityScope, setSwarmOutcomeVisibilityScope] = useState<'mine' | 'shared' | 'all'>('mine');
  const swarmOutcomeDateFrom = useMemo(() => {
    if (swarmOutcomeDateRange === '7d') {
      return new Date(Date.now() - (7 * 24 * 60 * 60 * 1000)).toISOString();
    }
    if (swarmOutcomeDateRange === '30d') {
      return new Date(Date.now() - (30 * 24 * 60 * 60 * 1000)).toISOString();
    }
    return undefined;
  }, [swarmOutcomeDateRange]);
  const { data: swarmOutcomeAnalyticsData, isLoading: swarmOutcomeAnalyticsLoading, refetch: refetchSwarmOutcomeAnalytics } = useQuery(
    ['agent-job-swarm-outcomes', swarmOutcomePresetFilter, swarmOutcomeTerminalFilter, swarmOutcomePromotionFilter, swarmOutcomeDateFrom, swarmOutcomeVisibilityScope],
    () =>
      apiClient.getAgentJobSwarmOutcomeAnalytics({
        preset_key: swarmOutcomePresetFilter || undefined,
        terminal_outcome: swarmOutcomeTerminalFilter || undefined,
        promotion_mode: swarmOutcomePromotionFilter || undefined,
        visibility_scope: swarmOutcomeVisibilityScope,
        date_from: swarmOutcomeDateFrom,
      }),
    {
      enabled: ['outcomes', 'jobs', 'swarm', 'backlog'].includes(activeTab),
      refetchInterval: activeTab === 'outcomes' ? 15000 : false,
    }
  );
  const swarmOutcomeCases = useMemo(
    () => ((((swarmOutcomeAnalyticsData as any)?.cases || []) as AgentJobSwarmOutcomeCase[])),
    [swarmOutcomeAnalyticsData]
  );
  const swarmOutcomeBySwarmJobId = useMemo(() => {
    const out: Record<string, AgentJobSwarmOutcomeCase> = {};
    for (const item of swarmOutcomeCases) {
      const key = String(item?.swarm_job_id || '').trim();
      if (key) out[key] = item;
    }
    return out;
  }, [swarmOutcomeCases]);
  const swarmOutcomeByRepairJobId = useMemo(() => {
    const out: Record<string, AgentJobSwarmOutcomeCase> = {};
    for (const item of swarmOutcomeCases) {
      const key = String(item?.repair_job_id || '').trim();
      if (key) out[key] = item;
    }
    return out;
  }, [swarmOutcomeCases]);

  return {
    swarmOutcomePresetFilter,
    setSwarmOutcomePresetFilter,
    swarmOutcomeTerminalFilter,
    setSwarmOutcomeTerminalFilter,
    swarmOutcomePromotionFilter,
    setSwarmOutcomePromotionFilter,
    swarmOutcomeDateRange,
    setSwarmOutcomeDateRange,
    swarmOutcomeVisibilityScope,
    setSwarmOutcomeVisibilityScope,
    swarmOutcomeAnalyticsData,
    swarmOutcomeAnalyticsLoading,
    refetchSwarmOutcomeAnalytics,
    swarmOutcomeCases,
    swarmOutcomeBySwarmJobId,
    swarmOutcomeByRepairJobId,
  };
}

export default useSwarmOutcomes;
