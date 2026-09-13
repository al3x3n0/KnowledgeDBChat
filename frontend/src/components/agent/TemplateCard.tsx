/**
 * One job template, as a card.
 *
 * Lifted out of AutonomousAgentsPage's render closure. Eighteen components are
 * declared inside that component, which means a new function identity on every
 * render — and a new identity is a new component type to React, so it unmounts
 * the old subtree and mounts a fresh one rather than reconciling. In a
 * component holding 328 hooks, that is every keystroke anywhere on the page
 * remounting every card on it.
 *
 * Out here the identity is stable, and what the card needs is a props
 * interface rather than whatever happened to be in scope.
 */

import React from 'react';

import type { AgentJobTemplate, AgentJobType } from '../../types';

export interface TemplateCardProps {
  template: AgentJobTemplate;
  /** The type badge's icon and colour, resolved by the caller that owns the map. */
  typeConfig: { icon: React.ComponentType<{ className?: string }>; color: string };
  onSelect: (template: AgentJobTemplate) => void;
}

export const TemplateCard: React.FC<TemplateCardProps> = ({
  template,
  typeConfig,
  onSelect,
}) => {
  const TypeIcon = typeConfig.icon;

  return (
    <div
      className="bg-white border border-gray-200 rounded-lg p-4 cursor-pointer
        transition-all duration-fast ease-ui
        hover:shadow-level-2 hover:-translate-y-px hover:border-gray-400
        active:translate-y-0 active:shadow-level-1"
      onClick={() => onSelect(template)}
    >
      <div className="flex items-start gap-3 mb-3">
        <div className={`p-2 rounded-lg ${typeConfig.color}`}>
          <TypeIcon className="w-5 h-5" />
        </div>
        <div className="flex-1">
          <h3 className="font-medium text-gray-900">{template.display_name}</h3>
          <p className="text-sm text-gray-500">{template.category}</p>
        </div>
        {template.recommended && (
          <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-1 rounded">
            Recommended
            {typeof template.recommendation_score === 'number'
              ? ` (${template.recommendation_score})`
              : ''}
          </span>
        )}
        {template.is_system && (
          <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">System</span>
        )}
      </div>
      <p className="text-sm text-gray-600 mb-3 line-clamp-2">{template.description}</p>
      <div className="flex items-center gap-4 text-xs text-gray-500">
        <span>Max {template.default_max_iterations} iterations</span>
        <span>{template.default_max_runtime_minutes} min runtime</span>
        {template.recommended && template.recommendation_reasons?.length ? (
          <span className="truncate">
            why: {template.recommendation_reasons.slice(0, 2).join(', ')}
          </span>
        ) : null}
      </div>
    </div>
  );
};

export default TemplateCard;

/** Re-exported so a caller can type its own config map against the same shape. */
export type TemplateCardJobType = AgentJobType;
