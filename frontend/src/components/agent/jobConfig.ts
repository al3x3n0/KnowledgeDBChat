/**
 * How a job's type and status are drawn.
 *
 * Both maps were module constants inside AutonomousAgentsPage. They are here
 * so that components lifted out of that page can keep rendering the same badge
 * without either importing from a 24,000-line page module or being handed the
 * map as a prop at every call site.
 *
 * Static data, no React state: importing this pulls in the icon set and
 * nothing else.
 */

import {
  Activity,
  AlertCircle,
  BarChart3,
  BookOpen,
  CheckCircle2,
  Clock,
  Layers,
  Loader2,
  Pause,
  Settings,
  XCircle,
  Zap,
} from 'lucide-react';
import React from 'react';

import type { AgentJobStatus, AgentJobType } from '../../types';

export interface JobTypeConfig {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  color: string;
}

export interface JobStatusConfig {
  color: string;
  bgColor: string;
  icon: React.ComponentType<{ className?: string }>;
}

export const JOB_TYPE_CONFIG: Record<AgentJobType, JobTypeConfig> = {
  research: { icon: BookOpen, label: 'Research', color: 'text-blue-600 bg-blue-100' },
  monitor: { icon: Activity, label: 'Monitor', color: 'text-green-600 bg-green-100' },
  analysis: { icon: BarChart3, label: 'Analysis', color: 'text-purple-600 bg-purple-100' },
  synthesis: { icon: Layers, label: 'Synthesis', color: 'text-orange-600 bg-orange-100' },
  knowledge_expansion: {
    icon: Zap,
    label: 'Knowledge Expansion',
    color: 'text-yellow-600 bg-yellow-100',
  },
  data_analysis: {
    icon: BarChart3,
    label: 'Data Analysis',
    color: 'text-indigo-600 bg-indigo-100',
  },
  custom: { icon: Settings, label: 'Custom', color: 'text-gray-600 bg-gray-100' },
};

export const STATUS_CONFIG: Record<AgentJobStatus, JobStatusConfig> = {
  pending: { color: 'text-yellow-700', bgColor: 'bg-yellow-100', icon: Clock },
  running: { color: 'text-blue-700', bgColor: 'bg-blue-100', icon: Loader2 },
  paused: { color: 'text-orange-700', bgColor: 'bg-orange-100', icon: Pause },
  completed: { color: 'text-green-700', bgColor: 'bg-green-100', icon: CheckCircle2 },
  failed: { color: 'text-red-700', bgColor: 'bg-red-100', icon: AlertCircle },
  cancelled: { color: 'text-gray-700', bgColor: 'bg-gray-100', icon: XCircle },
};

/** The badge for a job type, falling back the way every call site did by hand. */
export const jobTypeConfig = (jobType: string): JobTypeConfig =>
  JOB_TYPE_CONFIG[jobType as AgentJobType] || JOB_TYPE_CONFIG.custom;

/** The badge for a job status, with the same fallback. */
export const jobStatusConfig = (status: string): JobStatusConfig =>
  STATUS_CONFIG[status as AgentJobStatus] || STATUS_CONFIG.pending;
