import React from 'react';
import { useQuery } from 'react-query';
import { RefreshCw } from 'lucide-react';

import { apiClient } from '../../services/api';
import Button from '../common/Button';
import LoadingSpinner from '../common/LoadingSpinner';

const ResearchProgramsPanel: React.FC = () => {
  const { data, isLoading, isFetching, error, refetch } = useQuery(
    ['admin', 'research-programs'],
    () => apiClient.getResearchPrograms(),
    { refetchOnWindowFocus: false }
  );

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Research Programs</h2>
          <p className="text-sm text-gray-600">
            Domain tools and sandbox images available to research agents.
          </p>
        </div>
        <Button
          variant="secondary"
          icon={<RefreshCw className="w-4 h-4" />}
          loading={isFetching}
          onClick={() => refetch()}
        >
          Refresh
        </Button>
      </div>

      {isLoading ? (
        <LoadingSpinner text="Loading research programs" />
      ) : (
        <>
          {!!error && (
            <div role="alert" className="rounded-lg border border-red-300 p-4 text-red-700">
              Research program status could not be loaded. Try refreshing.
              {data && ' The information below is from the last successful request.'}
            </div>
          )}
          {data && (
            <>
              <section aria-label="Loaded programs" className="space-y-4">
                <h3 className="font-medium text-gray-900">Loaded ({data.loaded.length})</h3>
                {data.loaded.length === 0 && (
                  <p className="text-sm text-gray-600">
                    No research programs are loaded. Built-in research tools remain available.
                  </p>
                )}
                {data.loaded.map((program) => (
                  <article key={program.name} className="bg-white rounded-lg shadow p-6 space-y-4">
                    <div>
                      <h4 className="font-semibold text-gray-900">
                        {program.name}{program.version && ` · ${program.version}`}
                      </h4>
                      {program.description && <p className="text-sm text-gray-600">{program.description}</p>}
                    </div>
                    <details>
                      <summary className="cursor-pointer text-sm font-medium text-gray-900">
                        Tools ({program.tools.length})
                      </summary>
                      {program.tools.length > 0 ? (
                        <ul className="mt-2 space-y-1 text-sm text-gray-700">
                          {program.tools.map((tool) => <li key={tool} className="break-all"><code>{tool}</code></li>)}
                        </ul>
                      ) : <p className="mt-2 text-sm text-gray-600">No tools declared.</p>}
                    </details>
                    <details>
                      <summary className="cursor-pointer text-sm font-medium text-gray-900">
                        Sandbox images ({program.sandbox_images.length})
                      </summary>
                      {program.sandbox_images.length > 0 ? (
                        <ul className="mt-2 space-y-1 text-sm text-gray-700">
                          {program.sandbox_images.map((image) => <li key={image} className="break-all"><code>{image}</code></li>)}
                        </ul>
                      ) : <p className="mt-2 text-sm text-gray-600">No sandbox images declared.</p>}
                    </details>
                  </article>
                ))}
              </section>

              {data.failed.length > 0 && (
                <section aria-label="Failed programs" className="space-y-3">
                  <h3 className="font-medium text-red-700">Failed ({data.failed.length})</h3>
                  {data.failed.map((program, index) => (
                    <div key={`${program.name}-${index}`} className="rounded-lg border border-red-300 p-4">
                      <h4 className="font-medium text-gray-900">{program.name}</h4>
                      <p className="text-sm text-red-700 whitespace-pre-wrap break-words">{program.reason}</p>
                    </div>
                  ))}
                </section>
              )}

              {data.disabled.length > 0 && (
                <section aria-label="Disabled programs" className="space-y-3">
                  <h3 className="font-medium text-gray-900">Disabled ({data.disabled.length})</h3>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {data.disabled.map((name, index) => <li key={`${name}-${index}`}>{name}</li>)}
                  </ul>
                </section>
              )}
              <p className="text-sm text-gray-600">
                This status describes the API process. Workers must have the same programs installed.
                Installation and enablement changes take effect after restarting the affected services.
                Refresh retrieves status; it does not reload programs.
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
};

export default ResearchProgramsPanel;
