/**
 * Job chains: multi-step workflows where a job triggers the next on completion.
 *
 * First of the thirteen tabs lifted out of `AutonomousAgentsPage`, which was
 * 16,270 lines and 788K of JavaScript in one component holding 169 `useState`.
 * Opening "My Jobs" downloaded every line that renders Coding Backlog.
 *
 * The shape follows `SwarmOutcomesPanel`, the tab extracted before this one:
 * data and callbacks in as props, no page state reached through. That is what
 * makes a tab testable without mounting the page, and what lets the router
 * above decide what a click means.
 */

import { GitBranch, Play } from 'lucide-react';
import React from 'react';

import type { AgentJobChainDefinition } from '../../../types';
import Button from '../../common/Button';

export interface JobChainsTabProps {
  chains: AgentJobChainDefinition[];
  onStartChain: (chain: AgentJobChainDefinition) => void;
}

/** A chain saved from a recovery, rather than one somebody designed. */
function isRecoveryPlaybook(chain: AgentJobChainDefinition): boolean {
  const name = String(chain.name || '').toLowerCase();
  const displayName = String(chain.display_name || '').toLowerCase();
  const description = String(chain.description || '').toLowerCase();
  return (
    name.startsWith('playbook_recovery_') ||
    displayName.includes('recovery playbook') ||
    description.includes('saved as a recovery playbook')
  );
}

export const JobChainsTab: React.FC<JobChainsTabProps> = ({
  chains,
  onStartChain,
}) => (
  <div className="w-full">
    <p className="text-sm text-gray-500 mb-4">
      Job chains allow you to create multi-step workflows where jobs
      automatically trigger subsequent jobs on completion
    </p>
    {chains.length === 0 ? (
      <div className="flex flex-col items-center justify-center py-12 text-gray-500">
        <GitBranch className="w-12 h-12 mb-3 text-gray-400" />
        <p className="text-lg font-medium">No chain definitions yet</p>
        <p className="text-sm">
          Chain definitions allow you to create multi-step workflows
        </p>
      </div>
    ) : (
      <div className="grid grid-cols-3 gap-4">
        {chains.map((chain) => (
          <div
            key={chain.id}
            className="bg-white border border-gray-200 rounded-lg p-4 transition-all duration-fast ease-ui hover:shadow-level-2 hover:-translate-y-px hover:border-gray-400"
          >
            <div className="flex items-start gap-3 mb-3">
              <div className="p-2 rounded-lg bg-purple-100 text-purple-600">
                <GitBranch className="w-5 h-5" />
              </div>
              <div className="flex-1">
                <h3 className="section-heading">{chain.display_name}</h3>
                <p className="text-sm text-gray-500">
                  {chain.chain_steps.length} steps
                </p>
              </div>
              {isRecoveryPlaybook(chain) ? (
                <span className="text-xs bg-amber-100 text-amber-800 px-2 py-1 rounded">
                  Recovery
                </span>
              ) : null}
              {chain.is_system && (
                <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                  System
                </span>
              )}
            </div>
            {chain.description && (
              <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                {chain.description}
              </p>
            )}
            <div className="flex flex-wrap gap-2 mb-3">
              {chain.chain_steps.slice(0, 3).map((step, idx) => (
                <span
                  key={idx}
                  className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded"
                >
                  {step.step_name}
                </span>
              ))}
              {chain.chain_steps.length > 3 && (
                <span className="text-xs text-gray-500">
                  +{chain.chain_steps.length - 3} more
                </span>
              )}
            </div>
            <Button
              size="sm"
              variant="secondary"
              className="w-full"
              onClick={() => onStartChain(chain)}
            >
              <Play className="w-3 h-3 mr-1" />
              Start Chain
            </Button>
          </div>
        ))}
      </div>
    )}
  </div>
);

export default JobChainsTab;
