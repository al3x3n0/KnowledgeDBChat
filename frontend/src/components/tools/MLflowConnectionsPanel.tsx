import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  CheckCircle2,
  KeyRound,
  Loader2,
  Plus,
  Server,
  X,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiClient } from '../../services/api';
import type { ExternalAgentConnection, SecretSummary } from '../../types';

const CAPABILITIES = [
  { id: 'mlflow.experiments.search', label: 'Discover experiments' },
  { id: 'mlflow.experiments.get', label: 'Read experiment metadata' },
  { id: 'mlflow.runs.search', label: 'Search runs and metrics' },
  { id: 'mlflow.runs.get', label: 'Read run evidence' },
  { id: 'mlflow.artifacts.list', label: 'List run artifacts' },
  { id: 'mlflow.registered_models.get', label: 'Read registered models' },
  { id: 'mlflow.model_versions.get', label: 'Read model versions' },
] as const;

const DEFAULT_CAPABILITIES = CAPABILITIES.map((item) => item.id);

type AuthType = 'none' | 'bearer' | 'basic' | 'api_key';

interface CheckResult {
  tone: 'success' | 'pending' | 'error';
  message: string;
  auditId?: string;
}

interface Props {
  onConnectionsChanged?: () => void;
}

export const MLflowConnectionsPanel: React.FC<Props> = ({
  onConnectionsChanged,
}) => {
  const [connections, setConnections] = useState<ExternalAgentConnection[]>([]);
  const [secrets, setSecrets] = useState<SecretSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [checkingId, setCheckingId] = useState<string | null>(null);
  const [checks, setChecks] = useState<Record<string, CheckResult>>({});
  const [name, setName] = useState('MLflow Research Tracking');
  const [endpointUrl, setEndpointUrl] = useState('');
  const [authType, setAuthType] = useState<AuthType>('bearer');
  const [secretId, setSecretId] = useState('');
  const [apiKeyHeader, setApiKeyHeader] = useState('X-API-Key');
  const [selectedCapabilities, setSelectedCapabilities] =
    useState<string[]>(DEFAULT_CAPABILITIES);
  const [newSecretName, setNewSecretName] = useState('mlflow-tracking-credential');
  const [newSecretValue, setNewSecretValue] = useState('');
  const [storingSecret, setStoringSecret] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const [connectionResponse, secretResponse] = await Promise.all([
        apiClient.listExternalAgentConnections(),
        apiClient.listSecrets(),
      ]);
      setConnections(
        connectionResponse.agents.filter(
          (connection) => connection.provider_type === 'mlflow'
        )
      );
      setSecrets(secretResponse);
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || 'Failed to load MLflow connections');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const selectedSecret = useMemo(
    () => secrets.find((secret) => secret.id === secretId),
    [secretId, secrets]
  );

  const storeCredential = async () => {
    if (!newSecretName.trim() || !newSecretValue) {
      toast.error('Secret name and credential are required');
      return;
    }
    if (authType === 'basic' && !newSecretValue.includes(':')) {
      toast.error('Basic authentication must use username:password');
      return;
    }
    setStoringSecret(true);
    try {
      const secret = await apiClient.storeSecret(
        newSecretName.trim(),
        newSecretValue
      );
      setSecrets((current) => [
        ...current.filter((item) => item.id !== secret.id),
        secret,
      ]);
      setSecretId(secret.id);
      setNewSecretValue('');
      toast.success('MLflow credential stored in the encrypted vault');
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || 'Failed to store MLflow credential');
    } finally {
      setStoringSecret(false);
    }
  };

  const register = async () => {
    if (!name.trim() || !endpointUrl.trim()) {
      toast.error('Name and HTTPS tracking-server URL are required');
      return;
    }
    if (authType !== 'none' && !secretId) {
      toast.error('Select or store a vault credential');
      return;
    }
    if (!selectedCapabilities.length) {
      toast.error('Select at least one MLflow capability');
      return;
    }
    setSaving(true);
    try {
      await apiClient.createExternalAgentConnection({
        name: name.trim(),
        description:
          'Read-only MLflow tracking and model-registry evidence connection.',
        provider_type: 'mlflow',
        endpoint_url: endpointUrl.trim(),
        capabilities: selectedCapabilities,
        auth_type: authType,
        secret_id: authType === 'none' ? null : secretId,
        auth_header_name: authType === 'api_key' ? apiKeyHeader.trim() : undefined,
        timeout_seconds: 60,
        is_enabled: true,
      });
      setShowForm(false);
      setEndpointUrl('');
      toast.success('MLflow connection registered');
      await load();
      onConnectionsChanged?.();
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || 'Failed to register MLflow');
    } finally {
      setSaving(false);
    }
  };

  const testConnection = async (connection: ExternalAgentConnection) => {
    setCheckingId(connection.id);
    try {
      const result = await apiClient.invokeExternalAgentConnection(connection.id, {
        capability: 'mlflow.experiments.search',
        payload: { max_results: 1 },
        request_id: `mlflow-ui-check-${Date.now()}`,
      });
      if (result.status === 'requires_approval') {
        setChecks((current) => ({
          ...current,
          [connection.id]: {
            tone: 'pending',
            message: 'Check is waiting for policy approval',
            auditId: result.audit_id,
          },
        }));
        return;
      }
      if (result.status !== 'completed') {
        throw new Error(result.error || 'MLflow check failed');
      }
      const experiments = result.output?.output?.experiments
        || result.output?.experiments;
      setChecks((current) => ({
        ...current,
        [connection.id]: {
          tone: 'success',
          message: Array.isArray(experiments)
            ? `Tracking API reached; ${experiments.length} experiment sampled`
            : 'MLflow Tracking API reached',
          auditId: result.audit_id,
        },
      }));
      toast.success('MLflow connection check passed');
    } catch (error: any) {
      const message =
        error?.response?.data?.detail || error?.message || 'MLflow check failed';
      setChecks((current) => ({
        ...current,
        [connection.id]: { tone: 'error', message },
      }));
      toast.error(message);
    } finally {
      setCheckingId(null);
    }
  };

  return (
    <section
      aria-label="MLflow connections"
      className="mb-6 overflow-hidden rounded-xl border border-indigo-200 bg-gradient-to-br from-indigo-50 to-white"
    >
      <div className="flex items-start justify-between gap-4 border-b border-indigo-100 px-5 py-4">
        <div className="flex gap-3">
          <div className="rounded-lg bg-indigo-100 p-2 text-indigo-700">
            <BarChart3 className="h-5 w-5" />
          </div>
          <div>
            <h2 className="font-semibold text-gray-900">MLflow research tracking</h2>
            <p className="mt-0.5 max-w-3xl text-sm text-gray-600">
              Import audited run metrics, parameters, artifact inventories, and
              model-version provenance from a remote tracking server.
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setShowForm(true)}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-lg bg-indigo-700 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-800"
        >
          <Plus className="h-4 w-4" />
          Connect MLflow
        </button>
      </div>

      <div className="space-y-3 p-5">
        {loading && (
          <div className="text-sm text-gray-500">
            <Loader2 className="mr-2 inline h-4 w-4 animate-spin" />
            Loading MLflow connections…
          </div>
        )}
        {!loading && connections.length === 0 && (
          <p className="text-sm text-gray-600">
            No MLflow tracking server is connected yet.
          </p>
        )}
        {connections.map((connection) => {
          const check = checks[connection.id];
          return (
            <article
              key={connection.id}
              className="rounded-lg border border-indigo-100 bg-white p-4"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <Server className="h-4 w-4 text-indigo-600" />
                    <h3 className="font-medium text-gray-900">{connection.name}</h3>
                    <span className="rounded bg-green-50 px-2 py-0.5 text-xs text-green-700">
                      read only
                    </span>
                  </div>
                  <p className="mt-1 break-all text-xs text-gray-500">
                    {connection.endpoint_url}
                  </p>
                  <p className="mt-2 text-xs text-gray-600">
                    {connection.capabilities.length} typed capabilities ·{' '}
                    {connection.auth_type} authentication
                  </p>
                </div>
                <button
                  type="button"
                  disabled={
                    checkingId === connection.id
                    || !connection.capabilities.includes('mlflow.experiments.search')
                  }
                  onClick={() => testConnection(connection)}
                  className="inline-flex items-center gap-1.5 rounded border border-indigo-200 px-3 py-1.5 text-xs font-medium text-indigo-700 hover:bg-indigo-50 disabled:opacity-50"
                >
                  {checkingId === connection.id ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Activity className="h-3.5 w-3.5" />
                  )}
                  Test tracking API
                </button>
              </div>
              {check && (
                <div
                  className={`mt-3 flex items-center gap-2 rounded px-3 py-2 text-xs ${
                    check.tone === 'success'
                      ? 'bg-green-50 text-green-800'
                      : check.tone === 'pending'
                        ? 'bg-amber-50 text-amber-800'
                        : 'bg-red-50 text-red-800'
                  }`}
                >
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  {check.message}
                </div>
              )}
            </article>
          );
        })}
      </div>

      {showForm && (
        <div className="border-t border-indigo-100 bg-white p-5">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="font-medium text-gray-900">Register tracking server</h3>
            <button type="button" onClick={() => setShowForm(false)}>
              <X className="h-4 w-4 text-gray-500" />
            </button>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <label className="text-sm text-gray-700">
              Connection name
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                className="mt-1 w-full rounded border border-gray-300 px-3 py-2"
              />
            </label>
            <label className="text-sm text-gray-700">
              HTTPS tracking-server URL
              <input
                value={endpointUrl}
                onChange={(event) => setEndpointUrl(event.target.value)}
                placeholder="https://mlflow.example.com"
                className="mt-1 w-full rounded border border-gray-300 px-3 py-2"
              />
            </label>
            <label className="text-sm text-gray-700">
              Authentication
              <select
                value={authType}
                onChange={(event) => setAuthType(event.target.value as AuthType)}
                className="mt-1 w-full rounded border border-gray-300 px-3 py-2"
              >
                <option value="bearer">Bearer token</option>
                <option value="basic">Username and password</option>
                <option value="api_key">API key header</option>
                <option value="none">None</option>
              </select>
            </label>
            {authType === 'api_key' && (
              <label className="text-sm text-gray-700">
                API-key header
                <input
                  value={apiKeyHeader}
                  onChange={(event) => setApiKeyHeader(event.target.value)}
                  className="mt-1 w-full rounded border border-gray-300 px-3 py-2"
                />
              </label>
            )}
            {authType !== 'none' && (
              <label className="text-sm text-gray-700">
                Vault credential
                <select
                  value={secretId}
                  onChange={(event) => setSecretId(event.target.value)}
                  className="mt-1 w-full rounded border border-gray-300 px-3 py-2"
                >
                  <option value="">Select credential</option>
                  {secrets.map((secret) => (
                    <option key={secret.id} value={secret.id}>
                      {secret.name}
                    </option>
                  ))}
                </select>
                {selectedSecret && (
                  <span className="mt-1 block text-xs text-gray-500">
                    Selected: {selectedSecret.name}
                  </span>
                )}
              </label>
            )}
          </div>

          {authType !== 'none' && (
            <div className="mt-4 rounded-lg border border-indigo-100 bg-indigo-50 p-3">
              <div className="mb-2 flex items-center gap-2 text-sm font-medium text-indigo-900">
                <KeyRound className="h-4 w-4" />
                Store a new credential
              </div>
              <div className="grid gap-2 md:grid-cols-[1fr_2fr_auto]">
                <input
                  aria-label="MLflow secret name"
                  value={newSecretName}
                  onChange={(event) => setNewSecretName(event.target.value)}
                  className="rounded border border-indigo-200 px-3 py-2 text-sm"
                />
                <input
                  aria-label="MLflow secret value"
                  type="password"
                  value={newSecretValue}
                  onChange={(event) => setNewSecretValue(event.target.value)}
                  placeholder={
                    authType === 'basic' ? 'username:password' : 'Token value'
                  }
                  className="rounded border border-indigo-200 px-3 py-2 text-sm"
                />
                <button
                  type="button"
                  disabled={storingSecret}
                  onClick={storeCredential}
                  className="rounded bg-indigo-100 px-3 py-2 text-sm font-medium text-indigo-800 disabled:opacity-50"
                >
                  {storingSecret ? 'Storing…' : 'Store'}
                </button>
              </div>
            </div>
          )}

          <fieldset className="mt-4">
            <legend className="text-sm font-medium text-gray-800">
              Read capabilities
            </legend>
            <div className="mt-2 grid gap-2 md:grid-cols-2">
              {CAPABILITIES.map((capability) => (
                <label
                  key={capability.id}
                  className="flex items-center gap-2 rounded border border-gray-200 px-3 py-2 text-sm"
                >
                  <input
                    type="checkbox"
                    checked={selectedCapabilities.includes(capability.id)}
                    onChange={(event) =>
                      setSelectedCapabilities((current) =>
                        event.target.checked
                          ? [...current, capability.id]
                          : current.filter((item) => item !== capability.id)
                      )
                    }
                  />
                  {capability.label}
                </label>
              ))}
            </div>
          </fieldset>
          <button
            type="button"
            disabled={saving}
            onClick={register}
            className="mt-4 inline-flex items-center gap-2 rounded bg-indigo-700 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {saving && <Loader2 className="h-4 w-4 animate-spin" />}
            Register MLflow
          </button>
        </div>
      )}
    </section>
  );
};
