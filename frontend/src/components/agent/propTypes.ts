/**
 * Prop types shared by the agent tab components.
 *
 * These tabs were lifted out of one very large page, and the lift typed most of
 * their props `any` — which is worse than it looks: an `any` prop disables
 * inference *inside* the component, so a `.filter((row) => ...)` several calls
 * away silently loses its types too. Three real bugs in this area came from
 * exactly that (a badge typed as a ReactNode when it is an object, and two
 * invented response shapes that dropped fields the code reads).
 *
 * The names below recur across every tab, so they are defined once here.
 */

import type { QueryClient, UseMutationResult, QueryObserverResult } from 'react-query';
import type { NavigateFunction } from 'react-router-dom';
import type { AgentJobsTab } from './jobConfig';
import type { AgentJob } from '../../types';

/**
 * A mutation handle whose payload types are not constrained.
 *
 * This is a deliberate half-measure: it types the *handle* — `.mutate`,
 * `.isLoading`, `.reset` are all checked — while leaving the variables and
 * result loose. That is worth having on its own, and narrowing a given
 * mutation's payloads later is a local change that does not touch call sites.
 */
export type AnyMutation = UseMutationResult<any, any, any>;

/**
 * The same, for a mutation that genuinely takes no variables — `mutate()` with
 * no argument. Distinct because `AnyMutation` would require one.
 */
export type AnyVoidMutation = UseMutationResult<any, any, void>;

/** react-query's refetch, with the result deliberately unconstrained. */
export type Refetch = () => Promise<QueryObserverResult<any, any>> | void;

/** Build a link into the Runs page: a job, plus any extra query parameters. */
export type BuildRunsUrl = (
  jobId?: string,
  extras?: Record<string, string | null | undefined>
) => string;

export type SetActiveTab = (tab: AgentJobsTab) => void;
export type SetSelectedJob = (job: AgentJob | null) => void;

/** The router location, narrowed to what these components actually read. */
export interface RouterLocation {
  pathname: string;
  search: string;
}

export type { QueryClient, NavigateFunction };
