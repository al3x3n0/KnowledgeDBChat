import React, { useEffect, useMemo, useState } from 'react';
import {
  Database,
  ExternalLink,
  Loader2,
  PauseCircle,
  PlayCircle,
  RefreshCw,
  ShieldCheck,
  Webhook,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { useMutation, useQuery } from 'react-query';
import { apiClient } from '../../services/api';

interface Props {
  jobId: string;
  onImported?: () => void | Promise<unknown>;
}

type ImportKind =
  | 'compops.studies.report'
  | 'compops.studies.gates.evaluate'
  | 'compops.runs.get'
  | 'compops.artifacts.get'
  | 'compops.artifacts.lineage';

const IMPORT_OPTIONS: Array<{
  capability: ImportKind;
  label: string;
  idField: 'study_id' | 'run_id' | 'artifact_id';
  idLabel: string;
}> = [
  {
    capability: 'compops.studies.report',
    label: 'Study report',
    idField: 'study_id',
    idLabel: 'Study ID',
  },
  {
    capability: 'compops.studies.gates.evaluate',
    label: 'Study gate evaluation',
    idField: 'study_id',
    idLabel: 'Study ID',
  },
  {
    capability: 'compops.runs.get',
    label: 'Run result',
    idField: 'run_id',
    idLabel: 'Run ID',
  },
  {
    capability: 'compops.artifacts.get',
    label: 'Artifact metadata',
    idField: 'artifact_id',
    idLabel: 'Artifact ID',
  },
  {
    capability: 'compops.artifacts.lineage',
    label: 'Artifact lineage',
    idField: 'artifact_id',
    idLabel: 'Artifact ID',
  },
];

export const CompOpsEvidenceImportPanel: React.FC<Props> = ({
  jobId,
  onImported,
}) => {
  const [connectionId, setConnectionId] = useState('');
  const [capability, setCapability] = useState<ImportKind>(
    'compops.studies.report'
  );
  const [remoteId, setRemoteId] = useState('');
  const [metric, setMetric] = useState('');
  const [lineageDirection, setLineageDirection] = useState('both');
  const [lineageDepth, setLineageDepth] = useState('3');
  const [pendingAuditId, setPendingAuditId] = useState('');
  const [keepSynchronized, setKeepSynchronized] = useState(false);
  const [intervalMinutes, setIntervalMinutes] = useState('15');
  const [webhookSetup, setWebhookSetup] = useState<{
    subscriptionId: string;
    callbackUrl: string;
    signingSecret: string;
    signingFormat: string;
  } | null>(null);

  const connectionsQuery = useQuery(
    ['external-agent-connections', 'compops-evidence-import'],
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
        (connection) => connection.provider_type === 'compops' && connection.is_enabled
      ),
    [connectionsQuery.data]
  );
  const subscriptionsQuery = useQuery(
    ['compops-evidence-subscriptions', jobId],
    () => apiClient.listCompOpsEvidenceSubscriptions(jobId),
    {
      enabled: Boolean(jobId),
      refetchInterval: 60000,
      refetchOnWindowFocus: false,
      retry: false,
    }
  );
  const subscriptions = subscriptionsQuery.data?.subscriptions || [];
  const selectedConnection = connections.find(
    (connection) => connection.id === connectionId
  );
  const availableOptions = useMemo(
    () =>
      IMPORT_OPTIONS.filter((option) =>
        selectedConnection
          ? selectedConnection.capabilities.includes(option.capability)
          : connections.some((connection) =>
              connection.capabilities.includes(option.capability)
            )
      ),
    [connections, selectedConnection]
  );
  const selectedOption =
    IMPORT_OPTIONS.find((option) => option.capability === capability)
    || IMPORT_OPTIONS[0];

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
      if (!connectionId || !remoteId.trim()) {
        throw new Error('Select a CompOps connection and enter a remote ID');
      }
      const payload: Record<string, unknown> = {
        [selectedOption.idField]: remoteId.trim(),
      };
      if (capability === 'compops.studies.report' && metric.trim()) {
        payload.metric = metric.trim();
      }
      if (capability === 'compops.artifacts.lineage') {
        payload.direction = lineageDirection;
        payload.depth = Number(lineageDepth);
      }
      if (keepSynchronized) {
        return {
          kind: 'subscription' as const,
          value: await apiClient.createCompOpsEvidenceSubscription(jobId, {
            tool_id: connectionId,
            capability,
            payload,
            interval_minutes: Number(intervalMinutes),
            sync_immediately: true,
          }),
        };
      }
      return {
        kind: 'one-time' as const,
        value: await apiClient.invokeExternalAgentConnection(connectionId, {
          capability,
          payload,
          request_id: `knowledgeops-evidence-${Date.now()}`,
          agent_job_id: jobId,
        }),
      };
    },
    {
      onSuccess: async (result) => {
        if (result.kind === 'subscription') {
          setPendingAuditId('');
          setRemoteId('');
          toast.success('CompOps evidence synchronization started');
          await subscriptionsQuery.refetch();
          await onImported?.();
          return;
        }
        if (result.value.status === 'requires_approval') {
          setPendingAuditId(result.value.audit_id);
          toast.success('CompOps evidence import is waiting for policy approval');
          return;
        }
        if (
          result.value.status !== 'completed'
          || !result.value.evidence_linked
        ) {
          toast.error(
            result.value.error || 'CompOps response was not linked to this job'
          );
          return;
        }
        setPendingAuditId('');
        setRemoteId('');
        toast.success('CompOps provenance added as unverified evidence');
        await onImported?.();
      },
      onError: (error: any) => {
        toast.error(
          error?.response?.data?.detail
          || error?.message
          || 'Failed to import CompOps evidence'
        );
      },
    }
  );
  const subscriptionMutation = useMutation(
    async ({
      subscriptionId,
      action,
      enabled,
    }: {
      subscriptionId: string;
      action: 'sync' | 'toggle';
      enabled?: boolean;
    }) => {
      if (action === 'sync') {
        return apiClient.syncCompOpsEvidenceSubscription(jobId, subscriptionId);
      }
      const subscription = await apiClient.updateCompOpsEvidenceSubscription(
        jobId,
        subscriptionId,
        { is_enabled: enabled }
      );
      return { subscription, evidence_changed: false };
    },
    {
      onSuccess: async (result, variables) => {
        toast.success(
          variables.action === 'sync'
            ? result.evidence_changed
              ? 'CompOps evidence updated'
              : 'CompOps evidence is unchanged'
            : variables.enabled
              ? 'CompOps synchronization resumed'
              : 'CompOps synchronization paused'
        );
        await subscriptionsQuery.refetch();
        if (variables.action === 'sync' && result.evidence_changed) {
          await onImported?.();
        }
      },
      onError: (error: any) => {
        toast.error(
          error?.response?.data?.detail
          || error?.message
          || 'Failed to update CompOps synchronization'
        );
      },
    }
  );
  const webhookMutation = useMutation(
    async ({
      subscriptionId,
      action,
    }: {
      subscriptionId: string;
      action: 'enable' | 'disable';
    }) => {
      if (action === 'enable') {
        const setup = await apiClient.enableCompOpsSubscriptionWebhook(
          jobId,
          subscriptionId
        );
        return { action, subscriptionId, setup };
      }
      const subscription = await apiClient.disableCompOpsSubscriptionWebhook(
        jobId,
        subscriptionId
      );
      return { action, subscriptionId, subscription };
    },
    {
      onSuccess: async (result) => {
        if (result.action === 'enable' && 'setup' in result) {
          setWebhookSetup({
            subscriptionId: result.subscriptionId,
            callbackUrl: `${window.location.origin}${result.setup.callback_path}`,
            signingSecret: result.setup.signing_secret,
            signingFormat: result.setup.signing_format,
          });
          toast.success('Signed CompOps push events enabled');
        } else {
          setWebhookSetup((current) =>
            current?.subscriptionId === result.subscriptionId ? null : current
          );
          toast.success('CompOps push events disabled');
        }
        await subscriptionsQuery.refetch();
      },
      onError: (error: any) => {
        toast.error(
          error?.response?.data?.detail
          || error?.message
          || 'Failed to configure CompOps push events'
        );
      },
    }
  );

  if (connectionsQuery.isLoading) {
    return (
      <div className="mb-4 rounded-lg border border-cyan-100 bg-cyan-50 p-3 text-xs text-cyan-700">
        <Loader2 className="mr-2 inline h-3.5 w-3.5 animate-spin" />
        Loading CompOps evidence sources…
      </div>
    );
  }

  if (connections.length === 0) {
    return (
      <section
        className="mb-4 rounded-lg border border-dashed border-cyan-200 bg-cyan-50 p-3"
        aria-label="CompOps evidence import"
      >
        <h3 className="flex items-center gap-1.5 text-sm font-semibold text-gray-900">
          <Database className="h-4 w-4 text-cyan-700" />
          Import CompOps evidence
        </h3>
        <p className="mt-1 text-xs text-gray-600">
          Register an enabled CompOps connection to attach audited study, run, or
          artifact provenance to this R&amp;D job.
        </p>
        <a
          href="/tools"
          className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-cyan-700 hover:text-cyan-900"
        >
          Configure CompOps in Tools
          <ExternalLink className="h-3 w-3" />
        </a>
      </section>
    );
  }

  return (
    <section
      className="mb-4 rounded-lg border border-cyan-200 bg-gradient-to-br from-cyan-50 to-white p-3"
      aria-label="CompOps evidence import"
    >
      <div className="flex items-start gap-2">
        <Database className="mt-0.5 h-4 w-4 text-cyan-700" />
        <div>
          <h3 className="text-sm font-semibold text-gray-900">
            Import CompOps evidence
          </h3>
          <p className="mt-0.5 text-xs text-gray-600">
            Store only audited provenance, response digest, and remote IDs. The
            full compiler response remains in the tool audit.
          </p>
        </div>
      </div>

      <div className="mt-3 grid gap-2 md:grid-cols-3">
        <label className="text-xs text-gray-700">
          CompOps connection
          <select
            className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5"
            value={connectionId}
            onChange={(event) => setConnectionId(event.target.value)}
          >
            {connections.map((connection) => (
              <option key={connection.id} value={connection.id}>
                {connection.name}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-gray-700">
          Evidence source
          <select
            className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5"
            value={capability}
            onChange={(event) => setCapability(event.target.value as ImportKind)}
          >
            {availableOptions.map((option) => (
              <option key={option.capability} value={option.capability}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-gray-700">
          {selectedOption.idLabel}
          <input
            className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5 font-mono"
            value={remoteId}
            onChange={(event) => setRemoteId(event.target.value)}
            placeholder={selectedOption.idField}
          />
        </label>
      </div>

      {capability === 'compops.studies.report' && (
        <label className="mt-2 block text-xs text-gray-700">
          Metric (optional)
          <input
            className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5"
            value={metric}
            onChange={(event) => setMetric(event.target.value)}
            placeholder="cycles"
          />
        </label>
      )}
      {capability === 'compops.artifacts.lineage' && (
        <div className="mt-2 grid grid-cols-2 gap-2">
          <label className="text-xs text-gray-700">
            Direction
            <select
              className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5"
              value={lineageDirection}
              onChange={(event) => setLineageDirection(event.target.value)}
            >
              <option value="both">Both</option>
              <option value="upstream">Upstream</option>
              <option value="downstream">Downstream</option>
            </select>
          </label>
          <label className="text-xs text-gray-700">
            Depth
            <input
              type="number"
              min={1}
              max={20}
              className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5"
              value={lineageDepth}
              onChange={(event) => setLineageDepth(event.target.value)}
            />
          </label>
        </div>
      )}

      {pendingAuditId && (
        <div className="mt-3 rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800">
          Policy approval pending · audit <span className="font-mono">{pendingAuditId}</span>.
          Once approved, the same audited call will attach its provenance to this job.
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-3 rounded border border-cyan-100 bg-white/70 p-2">
        <label className="flex items-center gap-2 text-xs text-gray-700">
          <input
            type="checkbox"
            checked={keepSynchronized}
            onChange={(event) => setKeepSynchronized(event.target.checked)}
          />
          Keep this evidence synchronized
        </label>
        {keepSynchronized && (
          <label className="flex items-center gap-2 text-xs text-gray-700">
            Refresh every
            <select
              aria-label="CompOps synchronization interval"
              className="rounded border border-gray-300 px-2 py-1"
              value={intervalMinutes}
              onChange={(event) => setIntervalMinutes(event.target.value)}
            >
              <option value="5">5 minutes</option>
              <option value="15">15 minutes</option>
              <option value="30">30 minutes</option>
              <option value="60">1 hour</option>
              <option value="360">6 hours</option>
              <option value="1440">24 hours</option>
            </select>
          </label>
        )}
      </div>

      <button
        type="button"
        className="mt-3 inline-flex items-center gap-1.5 rounded bg-cyan-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-60"
        onClick={() => importMutation.mutate()}
        disabled={
          importMutation.isLoading
          || !connectionId
          || !remoteId.trim()
          || availableOptions.length === 0
        }
      >
        {importMutation.isLoading
          ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
          : <ShieldCheck className="h-3.5 w-3.5" />}
        {keepSynchronized
          ? 'Import and keep synchronized'
          : 'Import as unverified evidence'}
      </button>

      {subscriptions.length > 0 && (
        <div className="mt-4 border-t border-cyan-100 pt-3">
          <div className="flex items-center justify-between gap-2">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-cyan-900">
              Active evidence monitors ({subscriptions.length})
            </h4>
            <button
              type="button"
              className="rounded p-1 text-cyan-700 hover:bg-cyan-100"
              aria-label="Refresh CompOps subscriptions"
              onClick={() => subscriptionsQuery.refetch()}
            >
              <RefreshCw
                className={`h-3.5 w-3.5 ${
                  subscriptionsQuery.isFetching ? 'animate-spin' : ''
                }`}
              />
            </button>
          </div>
          <div className="mt-2 space-y-2">
            {subscriptions.map((subscription) => (
              <article
                key={subscription.id}
                className="rounded border border-cyan-100 bg-white p-2 text-xs"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="font-medium text-gray-800">
                      {IMPORT_OPTIONS.find(
                        (option) => option.capability === subscription.capability
                      )?.label || subscription.capability}
                      {' · '}
                      <span className="font-mono">{subscription.remote_id}</span>
                    </div>
                    <div className="mt-0.5 text-gray-500">
                      {subscription.status.replace(/_/g, ' ')}
                      {' · every '}
                      {subscription.interval_minutes} minutes
                      {subscription.last_success_at
                        ? ` · last ${new Date(
                            subscription.last_success_at
                          ).toLocaleString()}`
                        : ''}
                      {subscription.last_webhook_at
                        ? ` · push ${new Date(
                            subscription.last_webhook_at
                          ).toLocaleString()}`
                        : ''}
                    </div>
                    {subscription.last_error && (
                      <div className="mt-1 text-rose-700">
                        {subscription.last_error}
                      </div>
                    )}
                  </div>
                  <div className="flex shrink-0 gap-1">
                    <button
                      type="button"
                      className="rounded p-1 text-cyan-700 hover:bg-cyan-50 disabled:opacity-50"
                      aria-label={`${
                        subscription.webhook_enabled
                          ? 'Rotate webhook secret for'
                          : 'Enable push for'
                      } ${subscription.remote_id}`}
                      disabled={webhookMutation.isLoading}
                      onClick={() =>
                        webhookMutation.mutate({
                          subscriptionId: subscription.id,
                          action: 'enable',
                        })
                      }
                    >
                      <Webhook className="h-3.5 w-3.5" />
                    </button>
                    {subscription.webhook_enabled && (
                      <button
                        type="button"
                        className="rounded px-1.5 py-1 text-[10px] font-medium text-gray-600 hover:bg-gray-100 disabled:opacity-50"
                        aria-label={`Disable push for ${subscription.remote_id}`}
                        disabled={webhookMutation.isLoading}
                        onClick={() =>
                          webhookMutation.mutate({
                            subscriptionId: subscription.id,
                            action: 'disable',
                          })
                        }
                      >
                        Push off
                      </button>
                    )}
                    <button
                      type="button"
                      className="rounded p-1 text-cyan-700 hover:bg-cyan-50 disabled:opacity-50"
                      aria-label={`Sync ${subscription.remote_id} now`}
                      disabled={
                        subscriptionMutation.isLoading
                        || !subscription.is_enabled
                      }
                      onClick={() =>
                        subscriptionMutation.mutate({
                          subscriptionId: subscription.id,
                          action: 'sync',
                        })
                      }
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      className="rounded p-1 text-gray-600 hover:bg-gray-100 disabled:opacity-50"
                      aria-label={`${
                        subscription.is_enabled ? 'Pause' : 'Resume'
                      } ${subscription.remote_id}`}
                      disabled={subscriptionMutation.isLoading}
                      onClick={() =>
                        subscriptionMutation.mutate({
                          subscriptionId: subscription.id,
                          action: 'toggle',
                          enabled: !subscription.is_enabled,
                        })
                      }
                    >
                      {subscription.is_enabled
                        ? <PauseCircle className="h-3.5 w-3.5" />
                        : <PlayCircle className="h-3.5 w-3.5" />}
                    </button>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </div>
      )}
      {webhookSetup && (
        <div className="mt-3 rounded border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
          <div className="font-semibold">Copy this signing secret now</div>
          <p className="mt-1">
            Configure CompOps to POST event signals to this callback. The secret
            is shown only for this rotation.
          </p>
          <label className="mt-2 block">
            Callback URL
            <input
              readOnly
              aria-label="CompOps webhook callback URL"
              className="mt-1 w-full rounded border border-amber-300 bg-white px-2 py-1.5 font-mono text-[11px]"
              value={webhookSetup.callbackUrl}
            />
          </label>
          <label className="mt-2 block">
            Signing secret
            <input
              readOnly
              aria-label="CompOps webhook signing secret"
              className="mt-1 w-full rounded border border-amber-300 bg-white px-2 py-1.5 font-mono text-[11px]"
              value={webhookSetup.signingSecret}
            />
          </label>
          <div className="mt-2 break-all font-mono text-[10px] text-amber-800">
            {webhookSetup.signingFormat}
          </div>
          <button
            type="button"
            className="mt-2 text-xs font-medium text-amber-900 underline"
            onClick={() => setWebhookSetup(null)}
          >
            I stored the secret
          </button>
        </div>
      )}
    </section>
  );
};

export default CompOpsEvidenceImportPanel;
