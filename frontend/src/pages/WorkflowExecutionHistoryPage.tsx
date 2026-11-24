import { ArrowLeft, RefreshCw } from 'lucide-react';
import React from 'react';
import { useQuery } from 'react-query';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import Button from '../components/common/Button';
import { apiClient } from '../services/api';

const PAGE_SIZE = 25;

const WorkflowExecutionHistoryPage: React.FC = () => {
  const { workflowId = '' } = useParams<'workflowId'>();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedPage = Number(searchParams.get('page') || 1);
  const page = Number.isSafeInteger(requestedPage) && requestedPage > 0 && requestedPage < Number.MAX_SAFE_INTEGER / PAGE_SIZE
    ? requestedPage
    : 1;
  const offset = (page - 1) * PAGE_SIZE;
  const history = useQuery(
    ['workflow-execution-history', workflowId, offset],
    () => apiClient.listWorkflowExecutions(workflowId, { limit: PAGE_SIZE, offset }),
    { retry: false }
  );
  const executions = history.data?.executions || [];
  const total = history.data?.total ?? 0;
  const workflowName = executions[0]?.workflow_name;

  const goToPage = (next: number) => setSearchParams(next === 1 ? {} : { page: String(next) });

  return (
    <div className="w-full max-w-5xl space-y-4 p-4 lg:p-6">
      <Link to="/workflows" className="inline-flex items-center gap-2 text-sm text-primary-700 hover:underline">
        <ArrowLeft className="h-4 w-4" aria-hidden="true" /> Workflows
      </Link>
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold text-gray-900">Execution history</h1>
          {workflowName && <p className="mt-0.5 break-words text-sm text-gray-600">{workflowName}</p>}
        </div>
        <Button type="button" variant="secondary" size="sm" onClick={() => history.refetch()}
          loading={history.isFetching && !history.isLoading} icon={<RefreshCw className="h-4 w-4" />}>
          Refresh
        </Button>
      </header>

      {history.isLoading && <p role="status" className="text-sm text-gray-600">Loading executions…</p>}
      {history.isError && <p role="alert" className="text-sm text-red-300">Could not load workflow executions. Refresh to try again.</p>}
      {history.isSuccess && executions.length === 0 && (
        <p className="rounded-lg border border-gray-300 px-4 py-6 text-sm text-gray-600">
          {total === 0 ? 'This workflow has no executions yet.' : 'No executions on this page.'}
        </p>
      )}

      {executions.length > 0 && (
        <ul className="divide-y divide-gray-200 rounded-lg border border-gray-300 bg-gray-100">
          {executions.map((execution) => {
            const timestamp = execution.started_at || execution.created_at;
            return (
              <li key={execution.id} className="flex flex-wrap items-start justify-between gap-3 px-4 py-3">
                <div className="min-w-0 flex-1 basis-48">
                  <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
                    <span className="font-medium capitalize text-gray-900">{execution.status.replace(/_/g, ' ')}</span>
                    <time dateTime={timestamp} className="text-gray-600">{new Date(timestamp).toLocaleString()}</time>
                  </p>
                  <p className="mt-1 text-xs text-gray-600">{execution.trigger_type} · {execution.progress}% complete</p>
                  {execution.error && <p className="mt-1 break-words text-sm text-red-300">{execution.error}</p>}
                </div>
                <Link to={`/workflows?executionId=${encodeURIComponent(execution.id)}`}
                  className="shrink-0 text-sm text-primary-700 hover:underline">
                  Open execution
                </Link>
              </li>
            );
          })}
        </ul>
      )}

      {history.isSuccess && (page > 1 || total > PAGE_SIZE) && (
        <nav aria-label="Execution pages" className="flex flex-wrap items-center gap-3">
          <Button type="button" variant="secondary" size="sm" disabled={page <= 1} onClick={() => goToPage(page - 1)}>Previous</Button>
          <span className="text-sm text-gray-600">Page {page}</span>
          <Button type="button" variant="secondary" size="sm" disabled={offset + PAGE_SIZE >= total} onClick={() => goToPage(page + 1)}>Next</Button>
        </nav>
      )}
    </div>
  );
};

export default WorkflowExecutionHistoryPage;
