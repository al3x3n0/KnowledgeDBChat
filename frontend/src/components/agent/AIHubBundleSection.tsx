/**
 * The AI Hub bundle a run proposed, and the reviewer's verdict on it.
 *
 * A research run can end by proposing a set of dataset presets and eval
 * templates — a bundle — plus new plugins it thinks should exist. This section
 * is where a human accepts, rejects or creates them.
 *
 * It was 265 lines inside the job detail panel, and the state behind it
 * (which plugin is mid-creation, whether to enable on create) had been lifted
 * two levels up to AutonomousAgentsPage as four props, for want of anywhere
 * closer to put it. Nothing outside this section ever read them. They live
 * here now, and the panel is four props lighter.
 */

import { Sparkles } from 'lucide-react';
import React, { useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import { useQuery, useQueryClient } from 'react-query';
import { useNavigate } from 'react-router-dom';

import Button from '../common/Button';
import { apiClient } from '../../services/api';
import type { AgentJob } from '../../types';
import { copyText } from '../../utils/clipboard';

interface AIHubBundleSectionProps {
  job: AgentJob;
}

/** Renders nothing unless the run actually proposed a bundle. */
export const AIHubBundleSection: React.FC<AIHubBundleSectionProps> = ({ job }) => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const aiHubBundle = (job.results as any)?.ai_hub_bundle;

  const [feedbackReasons, setFeedbackReasons] = useState<Record<string, string>>({});
  const [bulkReason, setBulkReason] = useState('');
  const [bulkSubmitting, setBulkSubmitting] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState<Record<string, boolean>>({});
  const [creatingPluginId, setCreatingPluginId] = useState<string | null>(null);
  const [enableAfterCreate, setEnableAfterCreate] = useState(true);

  const { data: feedbackData } = useQuery(
    ['agent-job', job.id, 'ai-hub', 'recommendation-feedback'],
    () => apiClient.listAIHubRecommendationFeedback(String(job.id)),
    { enabled: !!aiHubBundle, staleTime: 15000 },
  );

  const feedbackIndex = useMemo(() => {
    const idx: Record<string, any> = {};
    const items = (feedbackData as any)?.items || [];
    for (const it of items) {
      const key = `${it.workflow}:${it.item_type}:${it.item_id}`;
      idx[key] = it;
    }
    return idx;
  }, [feedbackData]);

  const applyAIHubBundle = async () => {
    const evalIds: string[] = aiHubBundle?.enabled_eval_templates || [];
    const presetIds: string[] = aiHubBundle?.enabled_dataset_presets || [];
    try {
      await apiClient.setEnabledAIHubEvalTemplates({ enabled: evalIds });
      await apiClient.setEnabledAIHubDatasetPresets({ enabled: presetIds });
      toast.success('AI Hub bundle applied');
      navigate('/ai-hub?tab=datasets');
    } catch (e: any) {
      toast.error(e?.message || 'Failed to apply bundle (admin required)');
    }
  };

  const envText = aiHubBundle?.env
    ? [
        `AI_HUB_DATASET_ENABLED_PRESET_IDS=${aiHubBundle.env.AI_HUB_DATASET_ENABLED_PRESET_IDS || ''}`,
        `AI_HUB_EVAL_ENABLED_TEMPLATE_IDS=${aiHubBundle.env.AI_HUB_EVAL_ENABLED_TEMPLATE_IDS || ''}`,
      ].join('\n')
    : '';

  const createPlugin = async (pluginType: 'dataset_preset' | 'eval_template', plugin: any) => {
    if (!plugin?.id) {
      toast.error('Plugin is missing id');
      return;
    }
    setCreatingPluginId(String(plugin.id));
    try {
      const res = await apiClient.createAIHubPlugin({
        plugin_type: pluginType,
        plugin,
        overwrite: false,
      });
      toast.success(`Created ${pluginType}: ${res.plugin_id}`);
      if (res.warnings && res.warnings.length > 0) {
        toast(res.warnings.join(' '), { duration: 6000 });
      }
      queryClient.invalidateQueries(['admin', 'ai-hub', 'eval-templates', 'all']);
      queryClient.invalidateQueries(['admin', 'ai-hub', 'dataset-presets', 'all']);

      if (enableAfterCreate) {
        if (pluginType === 'dataset_preset') {
          const current = await apiClient.getEnabledAIHubDatasetPresets();
          const enabled = (current as any)?.enabled || [];
          if (Array.isArray(enabled) && enabled.length > 0) {
            if (!enabled.includes(res.plugin_id)) {
              await apiClient.setEnabledAIHubDatasetPresets({
                enabled: [...enabled, res.plugin_id],
              });
              toast.success('Preset enabled');
              queryClient.invalidateQueries(['admin', 'ai-hub', 'dataset-presets', 'enabled']);
              queryClient.invalidateQueries(['ai-hub', 'dataset-presets', 'enabled']);
            }
          } else {
            toast('Preset created (all presets currently enabled)', {
              duration: 4000,
            });
          }
        } else {
          const current = await apiClient.getEnabledAIHubEvalTemplates();
          const enabled = (current as any)?.enabled || [];
          if (Array.isArray(enabled) && enabled.length > 0) {
            if (!enabled.includes(res.plugin_id)) {
              await apiClient.setEnabledAIHubEvalTemplates({
                enabled: [...enabled, res.plugin_id],
              });
              toast.success('Eval template enabled');
              queryClient.invalidateQueries(['admin', 'ai-hub', 'eval-templates', 'enabled']);
              queryClient.invalidateQueries(['training-eval-templates']);
            }
          } else {
            toast('Eval created (all eval templates currently enabled)', {
              duration: 4000,
            });
          }
        }
      }
    } catch (e: any) {
      const msg =
        e?.response?.data?.detail || e?.message || 'Failed to create plugin (admin required)';
      toast.error(msg);
    } finally {
      setCreatingPluginId(null);
    }
  };

  const submitFeedback = async (payload: {
    workflow: 'triage' | 'extraction' | 'literature';
    item_type: 'dataset_preset' | 'eval_template';
    item_id: string;
    decision: 'accept' | 'reject';
  }) => {
    const reasonKey = `${payload.workflow}:${payload.item_type}:${payload.item_id}`;
    const reason = (feedbackReasons[reasonKey] || '').trim();
    try {
      await apiClient.submitAIHubRecommendationFeedback(String(job.id), {
        ...payload,
        reason: reason || undefined,
      } as any);
      toast.success('Feedback saved');
      queryClient.invalidateQueries(['agent-job', job.id, 'ai-hub', 'recommendation-feedback']);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to save feedback');
    }
  };

  const bulkDecision = async (decision: 'accept' | 'reject') => {
    if (
      !aiHubBundle ||
      !Array.isArray(aiHubBundle.selection_rationale) ||
      aiHubBundle.selection_rationale.length === 0
    ) {
      return;
    }
    setBulkSubmitting(true);
    try {
      const reason = bulkReason.trim();
      for (const rec of aiHubBundle.selection_rationale) {
        const itemType = rec?.type === 'dataset_preset' ? 'dataset_preset' : 'eval_template';
        const workflow = rec?.workflow as 'triage' | 'extraction' | 'literature';
        const itemId = rec?.id;
        if (!workflow || !itemId) continue;
        await apiClient.submitAIHubRecommendationFeedback(String(job.id), {
          workflow,
          item_type: itemType as any,
          item_id: itemId,
          decision,
          reason: reason || undefined,
        } as any);
      }
      toast.success(`Saved ${decision} for all`);
      queryClient.invalidateQueries(['agent-job', job.id, 'ai-hub', 'recommendation-feedback']);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to save bulk feedback');
    } finally {
      setBulkSubmitting(false);
    }
  };

  if (!aiHubBundle) return null;

  return (
    <div className="mb-4">
      <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
        <Sparkles className="w-4 h-4" />
        AI Hub Bundle
      </h3>
      <div className="bg-white border border-gray-200 rounded-lg p-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium text-gray-900">
              {aiHubBundle.bundle_name || 'Bundle'}
            </div>
            <div className="text-xs text-gray-500">
              Presets: {(aiHubBundle.enabled_dataset_presets || []).length} • Evals:{' '}
              {(aiHubBundle.enabled_eval_templates || []).length}
            </div>
          </div>
          <div className="flex gap-2">
            <Button size="sm" onClick={applyAIHubBundle}>
              Apply to AI Hub
            </Button>
            <Button size="sm" variant="secondary" onClick={() => navigate('/ai-hub?tab=datasets')}>
              Open AI Hub
            </Button>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
          <div className="bg-gray-50 rounded p-2">
            <div className="font-medium text-gray-700 mb-1">Enabled Dataset Presets</div>
            <div className="text-gray-600 break-words">
              {(aiHubBundle.enabled_dataset_presets || []).join(', ') || '(none)'}
            </div>
          </div>
          <div className="bg-gray-50 rounded p-2">
            <div className="font-medium text-gray-700 mb-1">Enabled Eval Templates</div>
            <div className="text-gray-600 break-words">
              {(aiHubBundle.enabled_eval_templates || []).join(', ') || '(none)'}
            </div>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => copyText(JSON.stringify(aiHubBundle, null, 2), 'Bundle JSON')}
          >
            Copy Bundle JSON
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => copyText(envText, 'Env Vars')}
            disabled={!envText}
            title="Use these for env-based configuration if you can’t apply via admin"
          >
            Copy Env Vars
          </Button>
        </div>

        {Array.isArray(aiHubBundle.recommended_new_plugins) &&
          aiHubBundle.recommended_new_plugins.length > 0 && (
            <div className="mt-4">
              <div className="text-sm font-medium text-gray-800 mb-2">Recommended new plugins</div>
              <label className="flex items-center gap-2 text-xs text-gray-600 mb-2">
                <input
                  type="checkbox"
                  checked={enableAfterCreate}
                  onChange={(e) => setEnableAfterCreate(e.target.checked)}
                />
                Enable after create (only affects allowlist mode; no-op if “all enabled”)
              </label>
              <div className="space-y-2">
                {aiHubBundle.recommended_new_plugins.map((rec: any, idx: number) => {
                  const skeleton = rec?.skeleton;
                  const pluginType =
                    rec?.type === 'dataset_preset'
                      ? ('dataset_preset' as const)
                      : ('eval_template' as const);
                  const suggestedId = rec?.id_suggestion || skeleton?.id || `plugin_${idx}`;
                  const plugin = {
                    ...(skeleton || {}),
                    id: suggestedId,
                    name: rec?.name_suggestion || skeleton?.name || suggestedId,
                  };
                  return (
                    <div
                      key={`${pluginType}:${suggestedId}:${idx}`}
                      className="border border-gray-200 rounded-lg p-3 bg-white"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {pluginType === 'dataset_preset' ? 'Dataset Preset' : 'Eval Template'} •{' '}
                            {rec?.workflow || 'workflow'}
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            Suggested id: <span className="font-mono">{suggestedId}</span>
                          </div>
                          {rec?.why && <div className="text-xs text-gray-600 mt-1">{rec.why}</div>}
                        </div>
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => copyText(JSON.stringify(plugin, null, 2), 'Plugin JSON')}
                          >
                            Copy JSON
                          </Button>
                          <Button
                            size="sm"
                            onClick={() => createPlugin(pluginType, plugin)}
                            disabled={creatingPluginId === String(plugin.id)}
                            title="Admin: persist this plugin JSON to disk"
                          >
                            {creatingPluginId === String(plugin.id) ? 'Creating…' : 'Create Plugin'}
                          </Button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="text-xs text-gray-500 mt-2">
                After creating, enable it in `Admin → AI Hub` (or rerun AI Scientist and Apply).
              </div>
            </div>
          )}

        {Array.isArray(aiHubBundle.selection_rationale) &&
          aiHubBundle.selection_rationale.length > 0 && (
            <div className="mt-4">
              <div className="text-sm font-medium text-gray-800 mb-2">
                Learning loop (accept/reject)
              </div>
              <div className="mb-3 border border-gray-200 rounded-lg p-3 bg-gray-50">
                <div className="text-xs text-gray-600 mb-2">Bulk actions</div>
                <div className="flex flex-wrap gap-2 items-center">
                  <input
                    className="flex-1 min-w-[220px] border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={bulkReason}
                    onChange={(e) => setBulkReason(e.target.value)}
                    placeholder="Optional shared reason (applies to all)"
                  />
                  <Button
                    size="sm"
                    onClick={() => bulkDecision('accept')}
                    disabled={bulkSubmitting}
                  >
                    {bulkSubmitting ? 'Saving…' : 'Accept all'}
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => bulkDecision('reject')}
                    disabled={bulkSubmitting}
                  >
                    {bulkSubmitting ? 'Saving…' : 'Reject all'}
                  </Button>
                </div>
              </div>
              <div className="space-y-2">
                {aiHubBundle.selection_rationale.map((rec: any, idx: number) => {
                  const itemType =
                    rec?.type === 'dataset_preset' ? 'dataset_preset' : 'eval_template';
                  const workflow = rec?.workflow as 'triage' | 'extraction' | 'literature';
                  const itemId = rec?.id;
                  const key = `${workflow}:${itemType}:${itemId}`;
                  const existing = feedbackIndex[key];
                  const isOpen = Boolean(detailsOpen[key]);
                  return (
                    <div
                      key={`${key}:${idx}`}
                      className="border border-gray-200 rounded-lg p-3 bg-white"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {workflow} • {itemType === 'dataset_preset' ? 'Preset' : 'Eval'} •{' '}
                            <span className="font-mono">{itemId}</span>
                          </div>
                          {Array.isArray(rec?.matched_terms) && rec.matched_terms.length > 0 && (
                            <div className="text-xs text-gray-500 mt-1">
                              Matched: {rec.matched_terms.slice(0, 8).join(', ')}
                            </div>
                          )}
                          {(rec?.feedback_accepts !== undefined ||
                            rec?.feedback_rejects !== undefined) && (
                            <div className="text-xs text-gray-500 mt-1">
                              Feedback: +{Number(rec.feedback_accepts || 0)} / -
                              {Number(rec.feedback_rejects || 0)}
                              {rec?.feedback_bias !== undefined && (
                                <>
                                  {' '}
                                  • bias {Number(rec.feedback_bias || 0) >= 0 ? '+' : ''}
                                  {Number(rec.feedback_bias || 0)}
                                </>
                              )}
                              {rec?.base_score !== undefined && (
                                <> • base {Number(rec.base_score || 0)}</>
                              )}
                            </div>
                          )}
                          {existing?.decision && (
                            <div className="text-xs text-gray-600 mt-1">
                              Your last decision:{' '}
                              <span className="font-medium">{existing.decision}</span>
                            </div>
                          )}
                        </div>
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() =>
                              setDetailsOpen((prev) => ({
                                ...prev,
                                [key]: !prev[key],
                              }))
                            }
                          >
                            {isOpen ? 'Hide' : 'Why'}
                          </Button>
                          <Button
                            size="sm"
                            variant={existing?.decision === 'accept' ? 'primary' : 'secondary'}
                            onClick={() =>
                              submitFeedback({
                                workflow,
                                item_type: itemType as any,
                                item_id: itemId,
                                decision: 'accept',
                              })
                            }
                          >
                            Accept
                          </Button>
                          <Button
                            size="sm"
                            variant={existing?.decision === 'reject' ? 'primary' : 'secondary'}
                            onClick={() =>
                              submitFeedback({
                                workflow,
                                item_type: itemType as any,
                                item_id: itemId,
                                decision: 'reject',
                              })
                            }
                          >
                            Reject
                          </Button>
                        </div>
                      </div>
                      {isOpen && (
                        <div className="mt-3 bg-gray-50 border border-gray-200 rounded-lg p-3 text-xs text-gray-700 space-y-1">
                          <div>
                            Score: <span className="font-medium">{Number(rec.score || 0)}</span>{' '}
                            (base {Number(rec.base_score || 0)} + bias{' '}
                            {Number(rec.feedback_bias || 0) >= 0 ? '+' : ''}
                            {Number(rec.feedback_bias || 0)})
                          </div>
                          {Array.isArray(rec?.matched_terms) && rec.matched_terms.length > 0 && (
                            <div>
                              Matched terms:{' '}
                              <span className="text-gray-600">{rec.matched_terms.join(', ')}</span>
                            </div>
                          )}
                          {Array.isArray((aiHubBundle as any)?.customer_keywords) && (
                            <div>
                              Customer keywords:{' '}
                              <span className="text-gray-600">
                                {(aiHubBundle as any).customer_keywords.slice(0, 12).join(', ')}
                              </span>
                            </div>
                          )}
                          <div className="pt-2 flex gap-2">
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() =>
                                copyText(JSON.stringify(rec, null, 2), 'Rationale JSON')
                              }
                            >
                              Copy rationale
                            </Button>
                          </div>
                        </div>
                      )}
                      <div className="mt-2">
                        <label className="block text-xs font-medium text-gray-700 mb-1">
                          Reason (optional)
                        </label>
                        <input
                          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                          value={feedbackReasons[key] ?? existing?.reason ?? ''}
                          onChange={(e) =>
                            setFeedbackReasons((prev) => ({
                              ...prev,
                              [key]: e.target.value,
                            }))
                          }
                          placeholder="E.g., 'Not relevant to our tooling' or 'Great default for weekly triage'"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="text-xs text-gray-500 mt-2">
                Feedback is stored per customer profile and will bias future AI Scientist
                recommendations.
              </div>
            </div>
          )}
      </div>
    </div>
  );
};

export default AIHubBundleSection;
