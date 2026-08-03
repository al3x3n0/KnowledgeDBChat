import React, { useEffect, useMemo, useState } from 'react';
import { BarChart3, ExternalLink, Loader2, ShieldCheck } from 'lucide-react';
import toast from 'react-hot-toast';
import { useMutation, useQuery } from 'react-query';
import { apiClient } from '../../services/api';

type ImportKind =
  | 'mlflow.runs.get'
  | 'mlflow.runs.search'
  | 'mlflow.artifacts.list'
  | 'mlflow.experiments.get'
  | 'mlflow.registered_models.get'
  | 'mlflow.model_versions.get';

const OPTIONS: Array<{ capability: ImportKind; label: string }> = [
  { capability: 'mlflow.runs.get', label: 'Run metrics and parameters' },
  { capability: 'mlflow.runs.search', label: 'Bounded run search' },
  { capability: 'mlflow.artifacts.list', label: 'Run artifact inventory' },
  { capability: 'mlflow.experiments.get', label: 'Experiment metadata' },
  { capability: 'mlflow.registered_models.get', label: 'Registered model' },
  { capability: 'mlflow.model_versions.get', label: 'Model version' },
];

interface Props {
  jobId: string;
  onImported?: () => void | Promise<unknown>;
}

export const MLflowEvidenceImportPanel: React.FC<Props> = ({
  jobId,
  onImported,
}) => {
  const [connectionId, setConnectionId] = useState('');
  const [capability, setCapability] = useState<ImportKind>('mlflow.runs.get');
  const [runId, setRunId] = useState('');
  const [experimentIds, setExperimentIds] = useState('');
  const [experimentId, setExperimentId] = useState('');
  const [artifactPath, setArtifactPath] = useState('');
  const [filter, setFilter] = useState('');
  const [modelName, setModelName] = useState('');
  const [modelVersion, setModelVersion] = useState('');
  const [pendingAuditId, setPendingAuditId] = useState('');

  const connectionsQuery = useQuery(
    ['external-agent-connections', 'mlflow-evidence-import'],
    () => apiClient.listExternalAgentConnections(),
    {
      staleTime: 30000,
      refetchOnWindowFocus: false,
      retry: false,
    }
  );
  const connections = useMemo(
    () =>
      (connectionsQuery.data?.agents || []).filter(
        (connection) => connection.provider_type === 'mlflow' && connection.is_enabled
      ),
    [connectionsQuery.data]
  );
  const selectedConnection = connections.find(
    (connection) => connection.id === connectionId
  );
  const availableOptions = useMemo(
    () =>
      OPTIONS.filter((option) =>
        selectedConnection
          ? selectedConnection.capabilities.includes(option.capability)
          : connections.some((connection) =>
              connection.capabilities.includes(option.capability)
            )
      ),
    [connections, selectedConnection]
  );

  useEffect(() => {
    if (!connectionId && connections.length > 0) {
      setConnectionId(connections[0].id);
    }
  }, [connectionId, connections]);

  useEffect(() => {
    if (
      selectedConnection
      && !selectedConnection.capabilities.includes(capability)
      && availableOptions.length > 0
    ) {
      setCapability(availableOptions[0].capability);
    }
  }, [availableOptions, capability, selectedConnection]);

  const importMutation = useMutation(
    async () => {
      if (!connectionId) throw new Error('Select an MLflow connection');
      const payload: Record<string, unknown> = {};
      if (capability === 'mlflow.runs.get') {
        if (!runId.trim()) throw new Error('Run ID is required');
        payload.run_id = runId.trim();
      } else if (capability === 'mlflow.runs.search') {
        const ids = experimentIds
          .split(',')
          .map((item) => item.trim())
          .filter(Boolean);
        if (!ids.length) throw new Error('Enter at least one experiment ID');
        payload.experiment_ids = ids;
        payload.max_results = 50;
        if (filter.trim()) payload.filter = filter.trim();
      } else if (capability === 'mlflow.artifacts.list') {
        if (!runId.trim()) throw new Error('Run ID is required');
        payload.run_id = runId.trim();
        if (artifactPath.trim()) payload.path = artifactPath.trim();
      } else if (capability === 'mlflow.experiments.get') {
        if (!experimentId.trim()) throw new Error('Experiment ID is required');
        payload.experiment_id = experimentId.trim();
      } else {
        if (!modelName.trim()) throw new Error('Model name is required');
        payload.name = modelName.trim();
        if (capability === 'mlflow.model_versions.get') {
          if (!modelVersion.trim()) throw new Error('Model version is required');
          payload.version = modelVersion.trim();
        }
      }
      return apiClient.invokeExternalAgentConnection(connectionId, {
        capability,
        payload,
        request_id: `knowledgeops-mlflow-evidence-${Date.now()}`,
        agent_job_id: jobId,
      });
    },
    {
      onSuccess: async (result) => {
        if (result.status === 'requires_approval') {
          setPendingAuditId(result.audit_id);
          toast.success('MLflow evidence import is waiting for policy approval');
          return;
        }
        if (result.status !== 'completed' || !result.evidence_linked) {
          toast.error(result.error || 'MLflow response was not linked to this job');
          return;
        }
        setPendingAuditId('');
        toast.success('MLflow provenance added as unverified evidence');
        await onImported?.();
      },
      onError: (error: any) => {
        toast.error(
          error?.response?.data?.detail
          || error?.message
          || 'Failed to import MLflow evidence'
        );
      },
    }
  );

  if (connectionsQuery.isLoading) {
    return (
      <div className="mb-4 rounded-lg border border-indigo-100 bg-indigo-50 p-3 text-xs text-indigo-700">
        <Loader2 className="mr-2 inline h-3.5 w-3.5 animate-spin" />
        Loading MLflow evidence sources…
      </div>
    );
  }

  if (connections.length === 0) {
    return (
      <section
        aria-label="MLflow evidence import"
        className="mb-4 rounded-lg border border-indigo-100 bg-indigo-50 p-4"
      >
        <div className="flex items-center gap-2 font-medium text-indigo-900">
          <BarChart3 className="h-4 w-4" />
          Import MLflow evidence
        </div>
        <p className="mt-1 text-xs text-indigo-800">
          Register an MLflow tracking server in Tools to attach audited run,
          artifact, and model provenance.
        </p>
        <a
          href="/tools"
          className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-indigo-700 hover:underline"
        >
          Configure MLflow in Tools
          <ExternalLink className="h-3 w-3" />
        </a>
      </section>
    );
  }

  return (
    <section
      aria-label="MLflow evidence import"
      className="mb-4 rounded-lg border border-indigo-100 bg-indigo-50 p-4"
    >
      <div className="flex items-start gap-2">
        <BarChart3 className="mt-0.5 h-4 w-4 text-indigo-700" />
        <div>
          <h3 className="text-sm font-semibold text-indigo-950">
            Import MLflow evidence
          </h3>
          <p className="mt-0.5 text-xs text-indigo-800">
            Only provenance enters the research trajectory. Raw MLflow responses
            remain in the audited tool record.
          </p>
        </div>
      </div>

      <div className="mt-3 grid gap-3 md:grid-cols-2">
        <label className="text-xs text-indigo-950">
          MLflow connection
          <select
            value={connectionId}
            onChange={(event) => setConnectionId(event.target.value)}
            className="mt-1 w-full rounded border border-indigo-200 bg-white px-2 py-2"
          >
            {connections.map((connection) => (
              <option key={connection.id} value={connection.id}>
                {connection.name}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-indigo-950">
          Evidence type
          <select
            value={capability}
            onChange={(event) => setCapability(event.target.value as ImportKind)}
            className="mt-1 w-full rounded border border-indigo-200 bg-white px-2 py-2"
          >
            {availableOptions.map((option) => (
              <option key={option.capability} value={option.capability}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        {(capability === 'mlflow.runs.get'
          || capability === 'mlflow.artifacts.list') && (
          <label className="text-xs text-indigo-950">
            Run ID
            <input
              value={runId}
              onChange={(event) => setRunId(event.target.value)}
              className="mt-1 w-full rounded border border-indigo-200 bg-white px-2 py-2"
            />
          </label>
        )}
        {capability === 'mlflow.artifacts.list' && (
          <label className="text-xs text-indigo-950">
            Artifact path (optional)
            <input
              value={artifactPath}
              onChange={(event) => setArtifactPath(event.target.value)}
              className="mt-1 w-full rounded border border-indigo-200 bg-white px-2 py-2"
            />
          </label>
        )}
        {capability === 'mlflow.runs.search' && (
          <>
            <label className="text-xs text-indigo-950">
              Experiment IDs (comma separated)
              <input
                value={experimentIds}
                onChange={(event) => setExperimentIds(event.target.value)}
                className="mt-1 w-full rounded border border-indigo-200 bg-white px-2 py-2"
              />
            </label>
            <label className="text-xs text-indigo-950">
              MLflow filter (optional)
              <input
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
                placeholder="metrics.cycles &lt; 1000"
                className="mt-1 w-full rounded border border-indigo-200 bg-white px-2 py-2"
              />
            </label>
          </>
        )}
        {capability === 'mlflow.experiments.get' && (
          <label className="text-xs text-indigo-950">
            Experiment ID
            <input
              value={experimentId}
              onChange={(event) => setExperimentId(event.target.value)}
              className="mt-1 w-full rounded border border-indigo-200 bg-white px-2 py-2"
            />
          </label>
        )}
        {(capability === 'mlflow.registered_models.get'
          || capability === 'mlflow.model_versions.get') && (
          <label className="text-xs text-indigo-950">
            Model name
            <input
              value={modelName}
              onChange={(event) => setModelName(event.target.value)}
              className="mt-1 w-full rounded border border-indigo-200 bg-white px-2 py-2"
            />
          </label>
        )}
        {capability === 'mlflow.model_versions.get' && (
          <label className="text-xs text-indigo-950">
            Model version
            <input
              value={modelVersion}
              onChange={(event) => setModelVersion(event.target.value)}
              className="mt-1 w-full rounded border border-indigo-200 bg-white px-2 py-2"
            />
          </label>
        )}
      </div>

      <button
        type="button"
        disabled={importMutation.isLoading}
        onClick={() => importMutation.mutate()}
        className="mt-3 inline-flex items-center gap-2 rounded bg-indigo-700 px-3 py-2 text-xs font-medium text-white disabled:opacity-50"
      >
        {importMutation.isLoading ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <ShieldCheck className="h-3.5 w-3.5" />
        )}
        Import audited provenance
      </button>
      {pendingAuditId && (
        <p className="mt-2 text-xs text-amber-800">
          Approval audit: <span className="font-mono">{pendingAuditId}</span>
        </p>
      )}
    </section>
  );
};

export default MLflowEvidenceImportPanel;
