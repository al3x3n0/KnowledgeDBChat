import { ArrowUpRight } from 'lucide-react';
import React from 'react';
import { Link } from 'react-router-dom';

const AgentRunSourceLinks: React.FC<{ sources: unknown }> = ({ sources }) => {
  if (!Array.isArray(sources)) return null;
  const runs = sources.filter((run): run is { id: string; name: string } =>
    typeof run?.id === 'string' && typeof run?.name === 'string' && Boolean(run.id && run.name)
  );
  if (!runs.length) return null;

  return (
    <div className="space-y-1">
      <p className="text-xs font-medium text-gray-700">Source runs</p>
      <ul className="space-y-1">
        {runs.map(({ id, name }) => (
          <li key={id} className="min-w-0">
            <Link className="inline-flex max-w-full items-start gap-1 text-sm text-primary-700 hover:underline"
              to={`/autonomous-agents?job=${encodeURIComponent(id)}`}>
              <span className="min-w-0 break-words">{name}</span>
              <ArrowUpRight className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default AgentRunSourceLinks;
