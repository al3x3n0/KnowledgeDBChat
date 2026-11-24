import { Loader2, Play } from 'lucide-react';
import React from 'react';
import toast from 'react-hot-toast';
import { useMutation } from 'react-query';
import { Link } from 'react-router-dom';
import api from '../../services/api';

interface WorkflowRunButtonProps {
  workflowId: string;
  inEditor?: boolean;
  hasUnsavedChanges?: boolean;
}

const WorkflowRunButton: React.FC<WorkflowRunButtonProps> = ({
  workflowId,
  inEditor = false,
  hasUnsavedChanges = false,
}) => {
  const launch = useMutation(
    async () => {
      const response = await api.post<{ execution_id?: string }>(
        `/workflows/${encodeURIComponent(workflowId)}/execute/async`,
        { trigger_type: 'manual', trigger_data: {}, inputs: {} }
      );
      return response.data?.execution_id;
    },
    {
      onError: () => {
        toast.error('Failed to queue workflow');
      },
    }
  );

  const label = hasUnsavedChanges ? 'Run saved version' : 'Run';
  const Icon = launch.isLoading ? Loader2 : Play;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={() => launch.mutate()}
        disabled={launch.isLoading}
        title={inEditor ? undefined : 'Run workflow'}
        aria-label={inEditor ? undefined : 'Run workflow'}
        className={inEditor
          ? 'inline-flex items-center gap-1 rounded bg-green-600 px-3 py-1.5 text-white hover:bg-green-700 disabled:opacity-50'
          : 'rounded p-2 text-green-600 hover:bg-green-50 disabled:opacity-50'}
      >
        <Icon aria-hidden="true" className={`${inEditor ? 'h-4 w-4' : 'h-5 w-5'} ${launch.isLoading ? 'animate-spin' : ''}`} />
        {inEditor && <span>{launch.isLoading ? 'Queuing…' : label}</span>}
      </button>
      {launch.isSuccess && (
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span role="status" className="text-gray-600">
            {launch.data ? 'Queued' : 'Queued. Execution link unavailable.'}
          </span>
          {launch.data && (
            <Link
              to={`/workflows?executionId=${encodeURIComponent(launch.data)}`}
              target={inEditor ? '_blank' : undefined}
              rel={inEditor ? 'noopener' : undefined}
              className="font-medium text-primary-700 underline underline-offset-2"
            >
              View execution{inEditor && <span className="sr-only"> (opens in a new tab)</span>}
            </Link>
          )}
        </div>
      )}
    </div>
  );
};

export default WorkflowRunButton;
