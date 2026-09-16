/**
 * Templates: pre-configured jobs, and the quick starts that skip choosing one.
 *
 * The quick starts were seven near-identical JSX blocks -- set a scope, fill a
 * default goal if the box is empty, open a modal -- differing only in three
 * values. They are a list here, and the tab takes one `quickStarts` prop
 * instead of seven setters. The page keeps owning the modals, because that is
 * where they render; this only stops the tab knowing about them.
 *
 * That is the difference between 20 things reached through and 7 props.
 */

import { FileText } from 'lucide-react';
import React from 'react';

import type { AgentJobTemplate, AgentJobType } from '../../../types';
import Button from '../../common/Button';
import { JOB_TYPE_CONFIG } from '../jobConfig';
import TemplateCard from '../TemplateCard';

export interface QuickStart {
  /** Button text. */
  label: string;
  onStart: () => void;
}

export interface JobTemplatesTabProps {
  templates: AgentJobTemplate[];
  quickStarts: QuickStart[];
  /** Said when the Claude backend template is missing, because a quick start
   *  that silently does nothing is worse than one that says why. */
  claudeBackendAvailable: boolean;
  scope: string;
  onScopeChange: (scope: string) => void;
  goal: string;
  onGoalChange: (goal: string) => void;
  onSelectTemplate: (template: AgentJobTemplate) => void;
}

export const JobTemplatesTab: React.FC<JobTemplatesTabProps> = ({
  templates,
  quickStarts,
  claudeBackendAvailable,
  scope,
  onScopeChange,
  goal,
  onGoalChange,
  onSelectTemplate,
}) => (
  <div className="w-full">
    <p className="text-sm text-gray-500 mb-4">
      Choose a template to quickly create a pre-configured autonomous job
    </p>

    <div className="mb-3 flex items-center justify-between">
      <div className="flex items-center gap-2">
        {quickStarts.map((quickStart) => (
          <Button
            key={quickStart.label}
            variant="secondary"
            onClick={quickStart.onStart}
          >
            {quickStart.label}
          </Button>
        ))}
      </div>
      {!claudeBackendAvailable && (
        <span className="text-xs text-gray-500">
          Claude backend template not available
        </span>
      )}
    </div>

    <div className="mb-4 grid grid-cols-3 gap-3">
      <div>
        <label
          htmlFor="template-recommend-scope"
          className="block text-xs font-medium text-gray-600 mb-1"
        >
          Recommendation scope
        </label>
        <select
          id="template-recommend-scope"
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
          value={scope}
          onChange={(e) => onScopeChange(e.target.value)}
        >
          <option value="">Auto</option>
          <option value="backend">Backend</option>
          <option value="frontend">Frontend</option>
          <option value="latex">LaTeX</option>
          <option value="research">Research</option>
        </select>
      </div>
      <div className="col-span-2">
        <label
          htmlFor="template-recommend-goal"
          className="block text-xs font-medium text-gray-600 mb-1"
        >
          Goal hint (optional)
        </label>
        <input
          id="template-recommend-goal"
          type="text"
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
          value={goal}
          onChange={(e) => onGoalChange(e.target.value)}
          placeholder="e.g. Fix backend API tests for source ingestion"
        />
      </div>
    </div>

    {templates.length === 0 ? (
      <div className="flex flex-col items-center justify-center py-12 text-gray-500">
        <FileText className="w-12 h-12 mb-3 text-gray-400" />
        <p className="text-lg font-medium">No templates available</p>
      </div>
    ) : (
      <div className="grid grid-cols-3 gap-4">
        {templates.map((template) => (
          <TemplateCard
            key={template.id}
            template={template}
            typeConfig={
              JOB_TYPE_CONFIG[template.job_type as AgentJobType] ||
              JOB_TYPE_CONFIG.custom
            }
            onSelect={onSelectTemplate}
          />
        ))}
      </div>
    )}
  </div>
);

export default JobTemplatesTab;
