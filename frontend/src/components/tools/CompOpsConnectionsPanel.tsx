import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  CheckCircle2,
  Cpu,
  KeyRound,
  Loader2,
  Plus,
  Server,
  ShieldCheck,
  X,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiClient } from '../../services/api';
import type { ExternalAgentConnection, SecretSummary } from '../../types';

const CAPABILITIES = [
  { id: 'compops.health', label: 'Health', effect: 'read' },
  { id: 'compops.operators.list', label: 'Discover operators', effect: 'read' },
  { id: 'compops.runs.get', label: 'Read runs', effect: 'read' },
  { id: 'compops.artifacts.get', label: 'Read artifact metadata', effect: 'read' },
  { id: 'compops.artifacts.lineage', label: 'Read artifact lineage', effect: 'read' },
  { id: 'compops.studies.get', label: 'Read studies', effect: 'read' },
  { id: 'compops.studies.report', label: 'Read study reports', effect: 'read' },
  {
    id: 'compops.studies.gates.evaluate',
    label: 'Evaluate study gates',
    effect: 'read',
  },
  { id: 'compops.runs.submit', label: 'Submit runs', effect: 'write' },
  { id: 'compops.batches.create', label: 'Create batches', effect: 'write' },
  { id: 'compops.actions.get', label: 'Read review actions', effect: 'read' },
  { id: 'compops.actions.approve', label: 'Approve actions', effect: 'write' },
  { id: 'compops.actions.reject', label: 'Reject actions', effect: 'write' },
] as const;

const DEFAULT_CAPABILITIES = CAPABILITIES.filter(
  (capability) => capability.effect === 'read'
).map((capability) => capability.id);

interface CheckResult {
  tone: 'success' | 'pending' | 'error';
  message: string;
  auditId?: string;
}

interface CompOpsConnectionsPanelProps {
  onConnectionsChanged?: () => void;
}

export const CompOpsConnectionsPanel: React.FC<CompOpsConnectionsPanelProps> = ({
  onConnectionsChanged,
}) => {
  const [connections, setConnections] = useState<ExternalAgentConnection[]>([]);
  const [secrets, setSecrets] = useState<SecretSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [checkingId, setCheckingId] = useState<string | null>(null);
  const [checks, setChecks] = useState<Record<string, CheckResult>>({});
  const [name, setName] = useState('CompOps Compiler Research');
  const [endpointUrl, setEndpointUrl] = useState('');
  const [secretId, setSecretId] = useState('');
  const [selectedCapabilities, setSelectedCapabilities] =
    useState<string[]>(DEFAULT_CAPABILITIES);
  const [newSecretName, setNewSecretName] = useState('compops-researcher-token');
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
          (connection) => connection.provider_type === 'compops'
        )
      );
      setSecrets(secretResponse);
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || 'Failed to load CompOps connections');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const writeCapabilityCount = useMemo(
    () =>
      selectedCapabilities.filter(
        (id) => CAPABILITIES.find((capability) => capability.id === id)?.effect === 'write'
      ).length,
    [selectedCapabilities]
  );

  const storeCredential = async () => {
    if (!newSecretName.trim() || !newSecretValue) {
      toast.error('Secret name and token are required');
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
      toast.success('CompOps token stored in the encrypted vault');
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || 'Failed to store CompOps token');
    } finally {
      setStoringSecret(false);
    }
  };

  const register = async () => {
    if (!name.trim() || !endpointUrl.trim() || !secretId) {
      toast.error('Name, HTTPS endpoint, and vault credential are required');
      return;
    }
    if (!selectedCapabilities.length) {
      toast.error('Select at least one CompOps capability');
      return;
    }
    setSaving(true);
    try {
      await apiClient.createExternalAgentConnection({
        name: name.trim(),
        description:
          'Typed CompOps compiler-research control-plane connection.',
        provider_type: 'compops',
        endpoint_url: endpointUrl.trim(),
        capabilities: selectedCapabilities,
        auth_type: 'bearer',
        secret_id: secretId,
        timeout_seconds: 60,
        is_enabled: true,
      });
      setShowForm(false);
      setEndpointUrl('');
      setSelectedCapabilities(DEFAULT_CAPABILITIES);
      toast.success('CompOps connection registered');
      await load();
      onConnectionsChanged?.();
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || 'Failed to register CompOps');
    } finally {
      setSaving(false);
    }
  };

  const invokeCheck = async (
    connection: ExternalAgentConnection,
    capability: 'compops.health' | 'compops.operators.list'
  ) => {
    setCheckingId(connection.id);
    try {
      const result = await apiClient.invokeExternalAgentConnection(connection.id, {
        capability,
        payload: {},
        request_id: `compops-ui-${capability.split('.').pop()}-${Date.now()}`,
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
        toast.success('CompOps check submitted for approval');
      } else if (result.status === 'completed') {
        const operatorPayload = result.output?.result ?? result.output;
        const count = Array.isArray(operatorPayload) ? operatorPayload.length : null;
        setChecks((current) => ({
          ...current,
          [connection.id]: {
            tone: 'success',
            message:
              capability === 'compops.health'
                ? 'CompOps API is healthy'
                : `Discovered ${count ?? 'available'} compiler operators`,
            auditId: result.audit_id,
          },
        }));
        toast.success(
          capability === 'compops.health'
            ? 'CompOps health check passed'
            : 'CompOps operator discovery completed'
        );
      } else {
        throw new Error(result.error || 'CompOps check failed');
      }
    } catch (error: any) {
      const message =
        error?.response?.data?.detail || error?.message || 'CompOps check failed';
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
      aria-label="CompOps connections"
      className="mb-6 overflow-hidden rounded-xl border border-cyan-200 bg-gradient-to-br from-cyan-50 to-white"
    >
      <div className="flex items-start justify-between gap-4 border-b border-cyan-100 px-5 py-4">
        <div className="flex gap-3">
          <div className="rounded-lg bg-cyan-100 p-2 text-cyan-700">
            <Cpu className="h-5 w-5" />
          </div>
          <div>
            <h2 className="font-semibold text-gray-900">CompOps research systems</h2>
            <p className="mt-0.5 max-w-3xl text-sm text-gray-600">
              Connect KnowledgeOps hypotheses to typed LLVM workflows, immutable
              artifacts, measurements, and noise-aware gates.
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setShowForm(true)}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-lg bg-cyan-700 px-3 py-2 text-sm font-medium text-white hover:bg-cyan-800"
        >
          <Plus className="h-4 w-4" />
          Add CompOps
        </button>
      </div>

      <div className="px-5 py-4">
        <div className="mb-4 grid gap-2 text-xs text-gray-600 sm:grid-cols-3">
          <div className="flex items-center gap-2 rounded-lg bg-white p-2.5">
            <ShieldCheck className="h-4 w-4 text-emerald-600" />
            Fixed REST capability map
          </div>
          <div className="flex items-center gap-2 rounded-lg bg-white p-2.5">
            <KeyRound className="h-4 w-4 text-amber-600" />
            Scoped token from encrypted vault
          </div>
          <div className="flex items-center gap-2 rounded-lg bg-white p-2.5">
            <Server className="h-4 w-4 text-cyan-700" />
            Audited external-system evidence
          </div>
        </div>

        {loading ? (
          <div className="flex justify-center py-6">
            <Loader2 className="h-5 w-5 animate-spin text-cyan-700" />
          </div>
        ) : connections.length === 0 ? (
          <div className="rounded-lg border border-dashed border-cyan-200 bg-white/70 p-4 text-sm text-gray-600">
            No CompOps control plane is registered yet.
          </div>
        ) : (
          <div className="space-y-3">
            {connections.map((connection) => {
              const check = checks[connection.id];
              return (
                <article
                  key={connection.id}
                  className="rounded-lg border border-cyan-100 bg-white p-4"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="font-medium text-gray-900">{connection.name}</h3>
                        <span className="rounded-full bg-cyan-50 px-2 py-0.5 text-[11px] font-medium text-cyan-700">
                          CompOps
                        </span>
                      </div>
                      <div className="mt-1 font-mono text-xs text-gray-500">
                        {connection.endpoint_url}
                      </div>
                      <div className="mt-2 text-xs text-gray-500">
                        {connection.capabilities.length} capabilities · bearer token ·
                        timeout {connection.timeout_seconds}s
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        disabled={
                          checkingId === connection.id ||
                          !connection.capabilities.includes('compops.health')
                        }
                        onClick={() => invokeCheck(connection, 'compops.health')}
                        className="inline-flex items-center gap-1.5 rounded-md border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
                      >
                        <Activity className="h-3.5 w-3.5" />
                        Health check
                      </button>
                      <button
                        type="button"
                        disabled={
                          checkingId === connection.id ||
                          !connection.capabilities.includes(
                            'compops.operators.list'
                          )
                        }
                        onClick={() =>
                          invokeCheck(connection, 'compops.operators.list')
                        }
                        className="inline-flex items-center gap-1.5 rounded-md bg-cyan-50 px-3 py-1.5 text-xs font-medium text-cyan-800 hover:bg-cyan-100 disabled:opacity-50"
                      >
                        {checkingId === connection.id ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <Cpu className="h-3.5 w-3.5" />
                        )}
                        Discover operators
                      </button>
                    </div>
                  </div>
                  {check && (
                    <div
                      className={`mt-3 flex items-center gap-2 rounded-md px-3 py-2 text-xs ${
                        check.tone === 'success'
                          ? 'bg-emerald-50 text-emerald-700'
                          : check.tone === 'pending'
                            ? 'bg-amber-50 text-amber-700'
                            : 'bg-red-50 text-red-700'
                      }`}
                    >
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      {check.message}
                      {check.auditId && (
                        <span className="ml-auto font-mono text-[10px] opacity-70">
                          audit {check.auditId}
                        </span>
                      )}
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </div>

      {showForm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Register CompOps connection"
            className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-xl bg-white shadow-xl"
          >
            <div className="flex items-center justify-between border-b px-5 py-4">
              <div>
                <h2 className="font-semibold text-gray-900">Register CompOps</h2>
                <p className="text-xs text-gray-500">
                  Credentials are referenced from the encrypted vault.
                </p>
              </div>
              <button
                type="button"
                aria-label="Close CompOps form"
                onClick={() => setShowForm(false)}
                className="rounded p-1 text-gray-500 hover:bg-gray-100"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-5 overflow-auto p-5">
              <div className="grid gap-4 sm:grid-cols-2">
                <label className="text-sm text-gray-700">
                  Connection name
                  <input
                    aria-label="Connection name"
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    className="mt-1 w-full rounded-lg border px-3 py-2"
                  />
                </label>
                <label className="text-sm text-gray-700">
                  CompOps HTTPS base URL
                  <input
                    aria-label="CompOps HTTPS base URL"
                    value={endpointUrl}
                    onChange={(event) => setEndpointUrl(event.target.value)}
                    placeholder="https://compops.example.com"
                    className="mt-1 w-full rounded-lg border px-3 py-2 font-mono text-sm"
                  />
                </label>
              </div>

              <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
                <div className="mb-3 flex items-center gap-2 text-sm font-medium text-amber-900">
                  <KeyRound className="h-4 w-4" />
                  Credential vault
                </div>
                <div className="grid gap-3 sm:grid-cols-3">
                  <input
                    aria-label="New secret name"
                    value={newSecretName}
                    onChange={(event) => setNewSecretName(event.target.value)}
                    placeholder="Secret name"
                    className="rounded-lg border bg-white px-3 py-2 text-sm"
                  />
                  <input
                    aria-label="New CompOps token"
                    type="password"
                    value={newSecretValue}
                    onChange={(event) => setNewSecretValue(event.target.value)}
                    placeholder="Project-scoped researcher token"
                    className="rounded-lg border bg-white px-3 py-2 text-sm"
                  />
                  <button
                    type="button"
                    disabled={storingSecret}
                    onClick={storeCredential}
                    className="rounded-lg border border-amber-300 bg-white px-3 py-2 text-sm font-medium text-amber-900 disabled:opacity-50"
                  >
                    {storingSecret ? 'Storing…' : 'Store token'}
                  </button>
                </div>
                <label className="mt-3 block text-xs text-amber-900">
                  Vault credential
                  <select
                    aria-label="Vault credential"
                    value={secretId}
                    onChange={(event) => setSecretId(event.target.value)}
                    className="mt-1 w-full rounded-lg border bg-white px-3 py-2 text-sm"
                  >
                    <option value="">Select a stored secret</option>
                    {secrets.map((secret) => (
                      <option key={secret.id} value={secret.id}>
                        {secret.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <fieldset>
                <legend className="text-sm font-medium text-gray-900">
                  Allowed CompOps capabilities
                </legend>
                <p className="mt-1 text-xs text-gray-500">
                  Write capabilities should be paired with explicit tool-policy
                  approval. Selected writes: {writeCapabilityCount}.
                </p>
                <div className="mt-3 grid gap-2 sm:grid-cols-2">
                  {CAPABILITIES.map((capability) => (
                    <label
                      key={capability.id}
                      className="flex items-start gap-2 rounded-lg border p-2.5 text-sm"
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
                        className="mt-0.5"
                      />
                      <span>
                        <span className="text-gray-800">{capability.label}</span>
                        <span
                          className={`ml-2 text-[10px] font-medium uppercase ${
                            capability.effect === 'write'
                              ? 'text-amber-700'
                              : 'text-emerald-700'
                          }`}
                        >
                          {capability.effect}
                        </span>
                        <span className="block font-mono text-[10px] text-gray-400">
                          {capability.id}
                        </span>
                      </span>
                    </label>
                  ))}
                </div>
              </fieldset>
            </div>

            <div className="flex justify-end gap-2 border-t bg-gray-50 px-5 py-4">
              <button
                type="button"
                onClick={() => setShowForm(false)}
                className="rounded-lg px-4 py-2 text-sm text-gray-700 hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={saving}
                onClick={register}
                className="rounded-lg bg-cyan-700 px-4 py-2 text-sm font-medium text-white hover:bg-cyan-800 disabled:opacity-50"
              >
                {saving ? 'Registering…' : 'Register connection'}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
