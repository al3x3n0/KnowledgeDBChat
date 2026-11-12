import React from 'react';
import { Link } from 'react-router-dom';

import type { PipelineEvidenceType } from '../../types';

export default function StageSkillDetails({ required, evidence, loading }: {
  required: string[];
  evidence: Record<string, PipelineEvidenceType>;
  loading: boolean;
}) {
  const selected = required.filter((name) => name.startsWith('skill_'));
  if (!selected.length) return null;

  return (
    <section aria-label="Stage Sandbox Skills" className="space-y-2 text-[11px]">
      <h4 className="font-medium text-gray-800">Sandbox Skills used by this stage</h4>
      {selected.map((name) => {
        const entry = evidence[name];
        const skill = entry?.sandbox_skill;
        return (
          <div key={name} className="rounded border border-gray-300 p-2 space-y-1">
            <p className="font-mono break-all">{name}</p>
            {skill ? <>
              <p className="font-medium">{skill.name}</p>
              <p>{skill.description}</p>
              <p className="break-all">Configured image: {skill.image}</p>
              <p>Command timeout: {skill.timeout_seconds}s · Network: disabled</p>
              <p>{entry.perishable ? 'Evidence goes stale when working files change.' : 'Evidence is retained when working files change.'}</p>
              <details>
                <summary className="cursor-pointer text-primary-700">Required result.json fields</summary>
                <dl className="mt-1 font-mono break-all">
                  {Object.entries(skill.result_fields).map(([field, type]) => (
                    <div key={field} className="flex gap-2"><dt>{field}</dt><dd>{type}</dd></div>
                  ))}
                </dl>
              </details>
            </> : <p className="text-yellow-700">
              {loading ? 'Loading skill details…' : 'No active skill details in the current catalog. Check the skill in Tools and refresh the catalog before launch.'}
            </p>}
          </div>
        );
      })}
      <p className="text-gray-500">
        The agent loads each skill’s procedure and runs commands in its configured image.
        The runner records evidence when collect_result is true and result.json passes the declared checks.
        {' '}<Link className="underline" to="/tools">Edit and validate skills in Tools</Link>.
      </p>
    </section>
  );
}
