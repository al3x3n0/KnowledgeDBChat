/**
 * What several agents independently found — and where they disagreed.
 *
 * This sits beside the evidence rather than in a tab of its own, because a
 * swarm's output IS evidence about evidence: whether a number one agent
 * produced was corroborated by another looking separately. Putting it on a
 * separate screen would mean the finding and the judgement of that finding
 * live in different places, and only one of them is where you are looking.
 *
 * **Contested comes first.** A disagreement is the single thing a swarm exists
 * to surface — four agents agreeing is mildly reassuring, two agents measuring
 * 12ms and 48ms for the same kernel is the reason you ran four agents. Sorting
 * agreements to the top would bury the one output worth the cost.
 *
 * Four verdicts, deliberately not two:
 *
 *   - **contested** — two roles measured the same thing and materially
 *     disagreed. Shown with both values, because "the roles disagreed" tells
 *     you nothing about which to distrust.
 *   - **inconclusive** — they measured the same thing through an instrument
 *     too imprecise to say whether the answers match. Sits directly under
 *     contested because it is nearly as actionable: the run established
 *     nothing, and a rerun on a quiet machine may fix it. Measured case: two
 *     roles reporting 130% and 142% of their own variation, whose 60ms and
 *     45ms were being scored as 100% agreement.
 *   - **corroborated** — two or more distinct roles found it, and any numbers
 *     agreed within a spread narrow enough for "agreed" to mean something.
 *   - **uncorroborated** — only one role spoke to it. Not a fault and not a
 *     disagreement; it simply was not checked, and calling it either would be
 *     a claim nobody made.
 */

import clsx from 'clsx';
import { AlertTriangle, CheckCircle2, CircleDashed, HelpCircle, Users } from 'lucide-react';
import React, { useMemo, useState } from 'react';

import type { AgentJob } from '../../types';

export interface SwarmConsensusSectionProps {
  job: AgentJob;
}

interface ConsensusGroup {
  finding_type: string;
  subject: string;
  verdict: string;
  detail: string;
  roles: string[];
}

const VERDICT_STYLE: Record<string, { border: string; icon: React.ReactNode }> = {
  contested: {
    border: 'border-amber-300 bg-amber-50/50',
    icon: <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />,
  },
  inconclusive: {
    border: 'border-sky-300 bg-sky-50/50',
    icon: <HelpCircle className="w-3.5 h-3.5 text-sky-600" />,
  },
  corroborated: {
    border: 'border-primary-500/40 bg-primary-500/5',
    icon: <CheckCircle2 className="w-3.5 h-3.5 text-primary-600" />,
  },
  uncorroborated: {
    border: 'border-gray-300 bg-gray-50',
    icon: <CircleDashed className="w-3.5 h-3.5 text-gray-400" />,
  },
};

const GroupRow: React.FC<{ group: ConsensusGroup }> = ({ group }) => {
  const style = VERDICT_STYLE[group.verdict] || VERDICT_STYLE.uncorroborated;
  return (
    <div className={clsx('rounded-md border px-2.5 py-2', style.border)}>
      <div className="flex flex-wrap items-center gap-1.5 mb-0.5">
        {style.icon}
        <span className="text-[11px] font-mono text-gray-800">
          {group.finding_type}
        </span>
        {group.subject && (
          <span className="text-[11px] text-gray-600">· {group.subject}</span>
        )}
        <span className="ml-auto text-[10px] text-gray-500">
          {group.roles.join(', ')}
        </span>
      </div>
      {/* The values, not just the verdict: which role measured what is what
          tells a reader which of the two to distrust. */}
      {group.detail && (
        <p className="text-[11px] text-gray-700 pl-5">{group.detail}</p>
      )}
    </div>
  );
};

export const SwarmConsensusSection: React.FC<SwarmConsensusSectionProps> = ({
  job,
}) => {
  const [showUncorroborated, setShowUncorroborated] = useState(false);

  const fanIn = useMemo(() => {
    const results = (job.results || {}) as Record<string, any>;
    const raw = results.swarm_fan_in;
    return raw && typeof raw === 'object' ? raw : null;
  }, [job.results]);

  if (!fanIn) return null;

  const contested: ConsensusGroup[] = Array.isArray(fanIn.contested)
    ? fanIn.contested
    : [];
  const corroborated: ConsensusGroup[] = Array.isArray(fanIn.corroborated)
    ? fanIn.corroborated
    : [];
  const inconclusive: ConsensusGroup[] = Array.isArray(fanIn.inconclusive)
    ? fanIn.inconclusive
    : [];
  const uncorroborated: ConsensusGroup[] = Array.isArray(fanIn.uncorroborated)
    ? fanIn.uncorroborated
    : [];

  if (
    !contested.length &&
    !inconclusive.length &&
    !corroborated.length &&
    !uncorroborated.length
  ) {
    return null;
  }

  const agreement = fanIn.typed_agreement;

  return (
    <div className="mb-4" data-testid="swarm-consensus-section">
      <div className="flex items-center gap-2 mb-2">
        <h3 className="text-sm font-medium text-gray-700 flex items-center gap-1.5">
          <Users className="w-4 h-4" />
          Swarm consensus
        </h3>
        {typeof agreement === 'number' ? (
          <span
            className={clsx(
              'text-[11px] px-2 py-0.5 rounded-full border font-medium',
              agreement >= 0.999
                ? 'border-primary-500/60 bg-primary-500/10 text-primary-700'
                : 'border-amber-300 bg-amber-50 text-amber-700'
            )}
            title="Of the claims more than one role spoke to, the share they agreed on."
          >
            {Math.round(agreement * 100)}% agreement
          </span>
        ) : inconclusive.length > 0 ? (
          // Agreement is null here for a different reason, and saying "nothing
          // cross-checked" would be untrue: roles DID measure the same things,
          // and the instrument could not resolve the answer.
          <span
            className="text-[11px] px-2 py-0.5 rounded-full border border-sky-300 bg-sky-50 text-sky-700 font-medium"
            title="Roles measured the same things, but reported too much variation of their own for agreement to be decided."
          >
            cross-checked, unresolved
          </span>
        ) : (
          // Not 0%. Nothing was checked, which is a different statement from
          // everything checked having disagreed.
          <span className="text-[11px] px-2 py-0.5 rounded-full border border-gray-300 text-gray-600">
            nothing cross-checked
          </span>
        )}
      </div>

      <div className="bg-gray-100 border border-gray-300 rounded-lg p-3 space-y-3">
        {contested.length > 0 && (
          <div>
            {/* First, always: this is what the swarm was for. */}
            <div className="text-[11px] font-medium text-amber-800 mb-1.5">
              {contested.length} contested — roles measured the same thing and
              disagreed
            </div>
            <div className="space-y-1.5">
              {contested.map((g) => (
                <GroupRow key={`${g.finding_type}:${g.subject}`} group={g} />
              ))}
            </div>
          </div>
        )}

        {inconclusive.length > 0 && (
          <div>
            {/* Second, always: not a disagreement, but not a result either. */}
            <div className="text-[11px] font-medium text-sky-800 mb-1.5">
              {inconclusive.length} unresolved — measured too noisily to tell
              agreement from coincidence
            </div>
            <div className="space-y-1.5">
              {inconclusive.map((g) => (
                <GroupRow key={`${g.finding_type}:${g.subject}`} group={g} />
              ))}
            </div>
          </div>
        )}

        {corroborated.length > 0 && (
          <div>
            <div className="text-[11px] font-medium text-gray-700 mb-1.5">
              {corroborated.length} corroborated by two or more roles
            </div>
            <div className="space-y-1.5">
              {corroborated.map((g) => (
                <GroupRow key={`${g.finding_type}:${g.subject}`} group={g} />
              ))}
            </div>
          </div>
        )}

        {uncorroborated.length > 0 && (
          <div>
            <button
              type="button"
              className="text-[11px] text-gray-500 hover:text-gray-800"
              onClick={() => setShowUncorroborated((open) => !open)}
            >
              {showUncorroborated ? 'Hide' : 'Show'} {uncorroborated.length} that
              only one role spoke to
            </button>
            {showUncorroborated && (
              <div className="space-y-1.5 mt-1.5">
                {uncorroborated.map((g) => (
                  <GroupRow key={`${g.finding_type}:${g.subject}`} group={g} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default SwarmConsensusSection;
