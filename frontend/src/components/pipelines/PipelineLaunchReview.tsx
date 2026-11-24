import React from 'react';

import type { PipelinePlan } from '../../types';
import ConfirmationModal from '../common/ConfirmationModal';
import type { PipelineSpec } from './PipelineGraph';

export function formatPipelineTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  if (!minutes) return `${remainder}s`;
  return remainder ? `${minutes} min ${remainder}s` : `${minutes} min`;
}

function names(value: unknown): string[] {
  const entries = Array.isArray(value) ? value : typeof value === 'string' ? value.split(',') : [];
  return entries.map((entry) => String(entry).trim()).filter(Boolean);
}

function requiredTypes(value: unknown): string[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return names(value);
  return Object.entries(value).map(([name, count]) => Number(count) > 1 ? `${name} ×${count}` : name);
}

interface PipelineLaunchReviewProps {
  spec: PipelineSpec;
  plan: PipelinePlan;
  budgetSeconds?: number;
  launching: boolean;
  onClose: () => void;
  onLaunch: () => void;
}

const PipelineLaunchReview: React.FC<PipelineLaunchReviewProps> = ({
  spec, plan, budgetSeconds, launching, onClose, onLaunch,
}) => {
  const stagesById = new Map((spec.stages || []).flatMap((stage, index) => (
    stage && typeof stage === 'object' ? [[String(stage.id || `stage_${index + 1}`).trim(), stage] as const] : []
  )));
  const unpriced = Array.from(new Set(plan.stages.flatMap((stage) => stage.unpriced || [])));

  return (
    <ConfirmationModal
      isOpen
      title="Review pipeline launch"
      confirmText="Start pipeline"
      cancelText="Keep editing"
      variant="info"
      isLoading={launching}
      onClose={onClose}
      onConfirm={onLaunch}
    >
      <div className="space-y-4 text-sm text-gray-700">
        <div>
          <p className="break-words font-medium text-gray-900">{String(spec.name || 'Untitled pipeline')}</p>
          <dl className="mt-2 grid grid-cols-2 gap-2">
            <div>
              <dt className="text-xs text-gray-600">Estimated work</dt>
              <dd className="font-medium">{formatPipelineTime(plan.total_seconds)}</dd>
            </div>
            <div>
              <dt className="text-xs text-gray-600">Planning budget</dt>
              <dd className="font-medium">{budgetSeconds === undefined ? 'Unset' : formatPipelineTime(budgetSeconds)}</dd>
            </div>
          </dl>
          {unpriced.length > 0 && (
            <p className="mt-2 break-words text-xs text-amber-700 dark:text-amber-300">
              The estimate excludes tools with unknown costs ({unpriced.join(', ')}).
            </p>
          )}
        </div>
        {plan.checkpoints.length > 0 && (
          <div>
            <p className="font-medium text-gray-900">Approval checkpoints</p>
            <p className="break-words">{plan.checkpoints.join(', ')}</p>
          </div>
        )}
        <div>
          <p className="font-medium text-gray-900">Stage inputs and evidence</p>
          <ol className="mt-1 divide-y divide-gray-200">
            {plan.order.map((id) => {
              const stage = stagesById.get(id);
              const evidence = [
                ...requiredTypes(stage?.contract?.required_finding_type_counts ?? stage?.contract?.required_finding_types),
                ...requiredTypes(stage?.contract?.required_artifact_type_counts ?? stage?.contract?.required_artifact_types),
                ...names(stage?.contract?.required_result_keys),
              ];
              const inputs = names(stage?.assumes);
              const bindings = Object.entries(stage?.tool_bindings || {}).flatMap(([tool, values]) => (
                Object.entries(values || {}).flatMap(([arg, binding]) => binding?.stage ? [`${tool}.${arg} from ${binding.stage}`] : [])
              ));
              return (
                <li key={id} className="break-words py-2">
                  <p className="font-medium text-gray-900">{id}</p>
                  {inputs.length > 0 && <p className="text-xs">Upstream evidence · {inputs.join(', ')}</p>}
                  {bindings.length > 0 && <p className="text-xs">Bound inputs · {bindings.join(', ')}</p>}
                  <p className="text-xs">{evidence.length > 0 ? `Required evidence · ${evidence.join(', ')}` : 'Evidence requirements are open-ended.'}</p>
                </li>
              );
            })}
          </ol>
        </div>
      </div>
    </ConfirmationModal>
  );
};

export default PipelineLaunchReview;
