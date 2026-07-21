/**
 * Notification bell component with dropdown
 */

import React, { useState, useRef, useEffect } from 'react';
import { Bell, Check, CheckCheck, X, ExternalLink, Loader2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useNotifications } from '../../contexts/NotificationContext';
import { Notification } from '../../types';
import { formatDistanceToNow } from 'date-fns';
import toast from 'react-hot-toast';
import { apiClient } from '../../services/api';
import { summarizeAutonomyBudgetNotification, summarizeCustomerAutonomyBudgetNotification, summarizeExperimentNotification, summarizeFollowUpOutcomeNotification, summarizeHypothesisReevaluationNotification, summarizePolicyGuardrailNotification, summarizeQueueUrgencyNotification } from '../../utils/notificationSummary';
import RecoveryAuditPanel from '../agent/RecoveryAuditPanel';
import {
  NotificationBellFilterMode,
  getNotificationBellCounts,
  getNotificationBellFilterLabel,
  getNotificationBellHeaderActionsState,
  getVisibleNotificationsForBell,
} from '../../utils/notificationBellState';

const NotificationBell: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [filterMode, setFilterMode] = useState<NotificationBellFilterMode>('all');
  const [relaunchingNotificationIds, setRelaunchingNotificationIds] = useState<Record<string, boolean>>({});
  const dropdownRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const {
    notifications,
    unreadCount,
    isLoading,
    fetchNotifications,
    markAsRead,
    markAllAsRead,
    dismissNotification,
  } = useNotifications();

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  useEffect(() => {
    if (!isOpen) {
      setFilterMode('all');
    }
  }, [isOpen]);

  const handleNotificationClick = async (notification: Notification) => {
    if (!notification.is_read) {
      await markAsRead(notification.id);
    }
    if (notification.action_url) {
      navigate(notification.action_url);
      setIsOpen(false);
    }
  };

  const ensureNotificationRead = async (notification: Notification) => {
    if (!notification.is_read) {
      await markAsRead(notification.id);
    }
  };

  const markVisibleAsRead = async () => {
    const unreadVisibleNotifications = visibleNotifications.filter((notification) => !notification.is_read);
    for (const notification of unreadVisibleNotifications) {
      await markAsRead(notification.id);
    }
  };

  const dismissVisibleNotifications = async () => {
    for (const notification of visibleNotifications) {
      await dismissNotification(notification.id);
    }
  };

  const restartRecoveryJob = async (notification: Notification) => {
    const agentJobId = String((notification.data as any)?.agent_job_id || '').trim();
    if (!agentJobId) return;
    try {
      await ensureNotificationRead(notification);
      await apiClient.performAgentJobAction(agentJobId, 'restart', {});
      toast.success('Recovery job restarted');
    } catch (error: any) {
      toast.error(error?.message || 'Failed to restart recovery job');
    }
  };

  const relaunchRecoveryJob = async (notification: Notification) => {
    const agentJobId = String((notification.data as any)?.agent_job_id || '').trim();
    if (!agentJobId) return;
    try {
      await ensureNotificationRead(notification);
      await apiClient.performAgentJobAction(agentJobId, 'relaunch', {});
      toast.success('Recovery job relaunched');
    } catch (error: any) {
      toast.error(error?.message || 'Failed to relaunch recovery job');
    }
  };

  const openNotificationTarget = async (notification: Notification) => {
    await ensureNotificationRead(notification);
    if (notification.action_url) {
      navigate(notification.action_url);
      setIsOpen(false);
    }
  };

  const openFollowUpJob = async (notification: Notification) => {
    const jobId = String((notification.data as any)?.follow_up_last_job_id || (notification.data as any)?.follow_up_job_id || '').trim();
    if (!jobId) return;
    await ensureNotificationRead(notification);
    navigate(`/autonomous-agents?job=${encodeURIComponent(jobId)}`);
    setIsOpen(false);
  };

  const relaunchFollowUpFromNotification = async (notification: Notification) => {
    const notificationId = String(notification.id || '').trim();
    const inboxItemId = String((notification.data as any)?.inbox_item_id || '').trim();
    if (!notificationId || !inboxItemId) return;
    setRelaunchingNotificationIds((prev) => ({ ...prev, [notificationId]: true }));
    try {
      await ensureNotificationRead(notification);
      await apiClient.relaunchInboxFollowUp(inboxItemId, {});
      await fetchNotifications();
      toast.success('Follow-up relaunched');
    } catch (error: any) {
      toast.error(error?.message || 'Failed to relaunch follow-up');
    } finally {
      setRelaunchingNotificationIds((prev) => ({ ...prev, [notificationId]: false }));
    }
  };

  const openReevaluationJob = async (notification: Notification) => {
    const jobId = String((notification.data as any)?.reevaluation_job_id || '').trim();
    if (!jobId) return;
    await ensureNotificationRead(notification);
    navigate(`/synthesis?job=${encodeURIComponent(jobId)}`);
    setIsOpen(false);
  };

  const openOriginatingOpportunity = async (notification: Notification) => {
    const actionUrl = String((notification.data as any)?.origin_action_url || '').trim();
    if (!actionUrl) return;
    await ensureNotificationRead(notification);
    navigate(actionUrl);
    setIsOpen(false);
  };

  const restartReevaluation = async (notification: Notification) => {
    const noteId = String((notification.data as any)?.note_id || '').trim();
    const noteTitle = String((notification.data as any)?.note_title || '').trim();
    if (!noteId) return;
    try {
      await ensureNotificationRead(notification);
      const job = await apiClient.createSynthesisJob({
        job_type: 'hypothesis_reevaluation',
        title: `Hypothesis Re-evaluation · ${noteTitle || noteId}`.slice(0, 500),
        document_ids: [],
        research_note_id: noteId,
        output_format: 'markdown',
        output_style: 'technical',
      });
      await fetchNotifications();
      toast.success('Hypothesis reevaluation started');
      navigate(`/synthesis?job=${encodeURIComponent(job.id)}`);
      setIsOpen(false);
    } catch (error: any) {
      toast.error(error?.message || 'Failed to restart hypothesis reevaluation');
    }
  };

  const copyNotificationField = async (value: string, label: string) => {
    try {
      await navigator.clipboard.writeText(value);
      toast.success(`${label} copied`);
    } catch (error: any) {
      toast.error(error?.message || `Failed to copy ${label.toLowerCase()}`);
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent': return 'border-l-red-500';
      case 'high': return 'border-l-orange-500';
      case 'normal': return 'border-l-blue-500';
      default: return 'border-l-gray-300';
    }
  };

  const getTypeIcon = (type: string) => {
    if (type.includes('error')) return '!';
    if (type.includes('complete')) return '\u2713';
    if (type.includes('maintenance')) return '\uD83D\uDD27';
    if (type.includes('broadcast')) return '\uD83D\uDCE2';
    if (type.includes('citation')) return '\uD83D\uDCCC';
    if (type.includes('experiment')) return '\uD83E\uDDEA';
    return '\u2022';
  };

  const visibleNotifications = getVisibleNotificationsForBell(notifications, filterMode);
  const bellCounts = getNotificationBellCounts(notifications);
  const headerActionsState = getNotificationBellHeaderActionsState(filterMode, visibleNotifications);

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Bell Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative p-2 text-gray-500 hover:text-gray-700 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-lg"
        aria-label="Notifications"
      >
        <Bell className="w-6 h-6" />
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs font-bold rounded-full h-5 w-5 flex items-center justify-center">
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-96 bg-white rounded-lg shadow-lg border border-gray-200 z-50 max-h-[32rem] flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
            <h3 className="font-semibold text-gray-900">Notifications</h3>
            <div className="flex items-center gap-3">
              {headerActionsState.showMarkFilteredRead && (
                <button
                  onClick={markVisibleAsRead}
                  className="text-sm text-violet-600 hover:text-violet-700 flex items-center gap-1"
                >
                  <Check className="w-4 h-4" />
                  Mark filtered read
                </button>
              )}
              {headerActionsState.showDismissFiltered && (
                <button
                  onClick={dismissVisibleNotifications}
                  className="text-sm text-rose-600 hover:text-rose-700 flex items-center gap-1"
                >
                  <X className="w-4 h-4" />
                  Dismiss filtered
                </button>
              )}
              {unreadCount > 0 && (
                <button
                  onClick={markAllAsRead}
                  className="text-sm text-primary-600 hover:text-primary-700 flex items-center gap-1"
                >
                  <CheckCheck className="w-4 h-4" />
                  Mark all read
                </button>
              )}
            </div>
          </div>

          {notifications.length > 0 && (
            <div className="border-b border-gray-100 bg-gray-50">
              <div className="flex items-center gap-2 px-4 py-2">
                <button
                  type="button"
                  aria-pressed={filterMode === 'all'}
                  className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${
                    filterMode === 'all'
                      ? 'bg-gray-900 text-white'
                      : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-100'
                  }`}
                  onClick={() => setFilterMode('all')}
                >
                  All {notifications.length}
                </button>
                <button
                  type="button"
                  aria-pressed={filterMode === 'unread'}
                  className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${
                    filterMode === 'unread'
                      ? 'bg-violet-600 text-white'
                      : 'bg-white text-violet-700 border border-violet-200 hover:bg-violet-50'
                  }`}
                  onClick={() => setFilterMode('unread')}
                >
                  Unread {bellCounts.unread}
                </button>
                <button
                  type="button"
                  aria-pressed={filterMode === 'unread_recovery'}
                  className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${
                    filterMode === 'unread_recovery'
                      ? 'bg-rose-600 text-white'
                      : 'bg-white text-rose-700 border border-rose-200 hover:bg-rose-50'
                  }`}
                  onClick={() => setFilterMode('unread_recovery')}
                >
                  Unread recovery {bellCounts.unreadRecovery}
                </button>
                <button
                  type="button"
                  aria-pressed={filterMode === 'experiment_runs'}
                  className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${
                    filterMode === 'experiment_runs'
                      ? 'bg-sky-600 text-white'
                      : 'bg-white text-sky-700 border border-sky-200 hover:bg-sky-50'
                  }`}
                  onClick={() => setFilterMode('experiment_runs')}
                >
                  Experiment runs {bellCounts.experimentRuns}
                </button>
                <button
                  type="button"
                  aria-pressed={filterMode === 'reevaluations'}
                  className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${
                    filterMode === 'reevaluations'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-white text-indigo-700 border border-indigo-200 hover:bg-indigo-50'
                  }`}
                  onClick={() => setFilterMode('reevaluations')}
                >
                  Reevaluations {bellCounts.reevaluations}
                </button>
                <button
                  type="button"
                  aria-pressed={filterMode === 'queue_alerts'}
                  className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${
                    filterMode === 'queue_alerts'
                      ? 'bg-fuchsia-600 text-white'
                      : 'bg-white text-fuchsia-700 border border-fuchsia-200 hover:bg-fuchsia-50'
                  }`}
                  onClick={() => setFilterMode('queue_alerts')}
                >
                  Queue alerts {bellCounts.queueAlerts}
                </button>
                <button
                  type="button"
                  aria-pressed={filterMode === 'follow_up_outcomes'}
                  className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${
                    filterMode === 'follow_up_outcomes'
                      ? 'bg-emerald-600 text-white'
                      : 'bg-white text-emerald-700 border border-emerald-200 hover:bg-emerald-50'
                  }`}
                  onClick={() => setFilterMode('follow_up_outcomes')}
                >
                  Follow-ups {bellCounts.followUpOutcomes}
                </button>
                <button
                  type="button"
                  aria-pressed={filterMode === 'recovery_open'}
                  className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${
                    filterMode === 'recovery_open'
                      ? 'bg-amber-600 text-white'
                      : 'bg-white text-amber-700 border border-amber-200 hover:bg-amber-50'
                  }`}
                  onClick={() => setFilterMode('recovery_open')}
                >
                  Open recovery {bellCounts.recoveryOpen}
                </button>
              </div>
              {filterMode !== 'all' && (
                <div
                  aria-live="polite"
                  className="flex items-center justify-between px-4 pb-2 text-[11px] font-medium text-gray-500"
                >
                  <span>
                    Showing {visibleNotifications.length} of {notifications.length} notifications for {getNotificationBellFilterLabel(filterMode)}
                  </span>
                  <button
                    type="button"
                    className="text-primary-600 hover:text-primary-700"
                    onClick={() => setFilterMode('all')}
                  >
                    Clear filter
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Notification List */}
          <div className="overflow-y-auto flex-1">
            {isLoading && notifications.length === 0 ? (
              <div className="p-4 text-center text-gray-500">
                <Loader2 className="w-6 h-6 mx-auto animate-spin mb-2" />
                Loading...
              </div>
            ) : notifications.length === 0 ? (
              <div className="p-8 text-center text-gray-500">
                <Bell className="w-12 h-12 mx-auto mb-2 opacity-30" />
                <p>No notifications yet</p>
              </div>
            ) : visibleNotifications.length === 0 ? (
              <div className="p-8 text-center text-gray-500">
                <Bell className="w-12 h-12 mx-auto mb-2 opacity-30" />
                <p>No notifications match this filter</p>
                <button
                  type="button"
                  className="mt-3 inline-flex items-center rounded-md border border-gray-200 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50"
                  onClick={() => setFilterMode('all')}
                >
                  Show all notifications
                </button>
              </div>
            ) : (
              visibleNotifications.map((notification) => {
                const experimentSummary = summarizeExperimentNotification(notification);
                const queueSummary = summarizeQueueUrgencyNotification(notification);
                const followUpSummary = summarizeFollowUpOutcomeNotification(notification);
                const reevaluationSummary = summarizeHypothesisReevaluationNotification(notification);
                const policyGuardrailSummary = summarizePolicyGuardrailNotification(notification);
                const autonomyBudgetSummary = summarizeAutonomyBudgetNotification(notification);
                const customerAutonomyBudgetSummary = summarizeCustomerAutonomyBudgetNotification(notification);
                const recoveryOpen = Boolean((notification.data as any)?.recovery_open);
                const hasAgentJob = Boolean(String((notification.data as any)?.agent_job_id || '').trim());
                const noteId = String((notification.data as any)?.note_id || '').trim();
                const launchMode = String((notification.data as any)?.launch_mode || '').trim();
                const firstFailedCommand = String((notification.data as any)?.first_failed_command || '').trim();
                const recommendedAction = String((notification.data as any)?.recommended_action || '').trim();
                const followUpJobId = String((notification.data as any)?.follow_up_last_job_id || (notification.data as any)?.follow_up_job_id || '').trim();
                const followUpInboxItemId = String((notification.data as any)?.inbox_item_id || '').trim();
                const followUpOutcomeStatus = String((notification.data as any)?.follow_up_outcome_status || '').trim().toLowerCase();
                const canOpenFollowUpJob = Boolean(followUpJobId);
                const canRelaunchFollowUp = Boolean(followUpInboxItemId) && ['failed', 'cancelled'].includes(followUpOutcomeStatus);
                const isRelaunchingFollowUp = Boolean(relaunchingNotificationIds[notification.id]);
                const canRelaunchRecovery = ['quick_start_claude_backend', 'quick_start_repo_bug_triage', 'quick_start_role_workflow'].includes(launchMode);
                const reevaluationJobId = String((notification.data as any)?.reevaluation_job_id || '').trim();
                const reevaluationStatus = String((notification.data as any)?.reevaluation_status || '').trim().toLowerCase();
                const originActionUrl = String((notification.data as any)?.origin_action_url || '').trim();
                const reevaluationNoteAction = notification.action_url && String(notification.action_url).includes('/research-notes?')
                  ? notification.action_url
                  : '';
                const canRestartReevaluation = Boolean(noteId) && ['failed', 'stale'].includes(reevaluationStatus);
                return (
                <div
                  key={notification.id}
                  className={`
                    px-4 py-3 border-b border-gray-100 hover:bg-gray-50 cursor-pointer
                    border-l-4 ${getPriorityColor(notification.priority)}
                    ${!notification.is_read ? 'bg-blue-50' : ''}
                  `}
                  onClick={() => handleNotificationClick(notification)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-lg">{getTypeIcon(notification.notification_type)}</span>
                        <p className={`text-sm font-medium truncate ${!notification.is_read ? 'text-gray-900' : 'text-gray-600'}`}>
                          {notification.title}
                        </p>
                      </div>
                      <p className="text-sm text-gray-500 mt-1 line-clamp-2">
                        {notification.message}
                      </p>
                      {experimentSummary && experimentSummary.badges.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {experimentSummary.badges.slice(0, 5).map((badge) => (
                            <span
                              key={badge}
                              className="inline-flex items-center rounded-full bg-gray-100 px-2 py-0.5 text-[11px] font-medium text-gray-700"
                            >
                              {badge}
                            </span>
                          ))}
                        </div>
                      )}
                      {queueSummary && queueSummary.badges.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {queueSummary.badges.slice(0, 5).map((badge) => (
                            <span
                              key={badge}
                              className="inline-flex items-center rounded-full bg-fuchsia-50 px-2 py-0.5 text-[11px] font-medium text-fuchsia-700"
                            >
                              {badge}
                            </span>
                          ))}
                        </div>
                      )}
                      {followUpSummary && followUpSummary.badges.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {followUpSummary.badges.slice(0, 5).map((badge) => (
                            <span
                              key={badge}
                              className="inline-flex items-center rounded-full bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700"
                            >
                              {badge}
                            </span>
                          ))}
                        </div>
                      )}
                      {reevaluationSummary && reevaluationSummary.badges.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {reevaluationSummary.badges.slice(0, 5).map((badge) => (
                            <span
                              key={badge}
                              className="inline-flex items-center rounded-full bg-indigo-50 px-2 py-0.5 text-[11px] font-medium text-indigo-700"
                            >
                              {badge}
                            </span>
                          ))}
                        </div>
                      )}
                      {policyGuardrailSummary && policyGuardrailSummary.badges.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {policyGuardrailSummary.badges.slice(0, 5).map((badge) => (
                            <span
                              key={badge}
                              className="inline-flex items-center rounded-full bg-rose-50 px-2 py-0.5 text-[11px] font-medium text-rose-700"
                            >
                              {badge}
                            </span>
                          ))}
                        </div>
                      )}
                      {autonomyBudgetSummary && autonomyBudgetSummary.badges.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {autonomyBudgetSummary.badges.slice(0, 5).map((badge) => (
                            <span
                              key={badge}
                              className="inline-flex items-center rounded-full bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-700"
                            >
                              {badge}
                            </span>
                          ))}
                        </div>
                      )}
                      {customerAutonomyBudgetSummary && customerAutonomyBudgetSummary.badges.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {customerAutonomyBudgetSummary.badges.slice(0, 5).map((badge) => (
                            <span
                              key={badge}
                              className="inline-flex items-center rounded-full bg-orange-50 px-2 py-0.5 text-[11px] font-medium text-orange-700"
                            >
                              {badge}
                            </span>
                          ))}
                        </div>
                      )}
                      <RecoveryAuditPanel
                        className="mt-2"
                        textClassName="text-xs"
                        latestAction={experimentSummary?.latestOperator}
                        latestOutcome={experimentSummary?.latestOperatorOutcome}
                        latestOutcomeReason={experimentSummary?.latestOperatorOutcomeReason}
                        recoveryReason={experimentSummary?.reason || queueSummary?.reason}
                        nextStep={experimentSummary?.nextAction || queueSummary?.nextAction}
                        schedulerState={queueSummary?.schedulerState || null}
                      />
                      {experimentSummary?.latestOperatorNote && (
                        <p className="text-xs text-violet-600 mt-1 line-clamp-2">
                          Operator note: {experimentSummary.latestOperatorNote}
                        </p>
                      )}
                      {queueSummary?.evidenceSummary && (
                        <p className="text-xs text-fuchsia-700 mt-1 line-clamp-2">
                          Evidence: {queueSummary.evidenceSummary}
                        </p>
                      )}
                      {followUpSummary?.summary && (
                        <p className="text-xs text-emerald-700 mt-1 line-clamp-2">
                          Outcome: {followUpSummary.summary}
                        </p>
                      )}
                      {reevaluationSummary?.summary ? (
                        <p className="text-xs text-indigo-700 mt-1 line-clamp-2">
                          Reprioritization: {reevaluationSummary.summary}
                        </p>
                      ) : null}
                      {reevaluationSummary?.error ? (
                        <p className="text-xs text-rose-700 mt-1 line-clamp-2">
                          Error: {reevaluationSummary.error}
                        </p>
                      ) : null}
                      {policyGuardrailSummary?.reasons?.length ? (
                        <p className="text-xs text-rose-700 mt-1 line-clamp-2">
                          Safeguard: {policyGuardrailSummary.reasons.join(' · ')}
                        </p>
                      ) : null}
                      {autonomyBudgetSummary?.reasons?.length ? (
                        <p className="text-xs text-amber-700 mt-1 line-clamp-2">
                          Budget: {autonomyBudgetSummary.reasons.join(' · ')}
                        </p>
                      ) : null}
                      {customerAutonomyBudgetSummary?.reasons?.length ? (
                        <p className="text-xs text-orange-700 mt-1 line-clamp-2">
                          Customer budget: {customerAutonomyBudgetSummary.reasons.join(' · ')}
                        </p>
                      ) : null}
                      {(followUpSummary && (notification.action_url || canOpenFollowUpJob || canRelaunchFollowUp)) ? (
                        <div className="mt-2 flex flex-wrap gap-2">
                          {notification.action_url && (
                            <button
                              type="button"
                              className="inline-flex items-center rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-[11px] font-medium text-emerald-700 hover:bg-emerald-100"
                              onClick={async (e) => {
                                e.stopPropagation();
                                await openNotificationTarget(notification);
                              }}
                            >
                              Open target
                            </button>
                          )}
                          {canOpenFollowUpJob && (
                            <button
                              type="button"
                              className="inline-flex items-center rounded-md border border-sky-200 bg-sky-50 px-2 py-1 text-[11px] font-medium text-sky-700 hover:bg-sky-100"
                              onClick={async (e) => {
                                e.stopPropagation();
                                await openFollowUpJob(notification);
                              }}
                            >
                              Open follow-up job
                            </button>
                          )}
                          {canRelaunchFollowUp && (
                            <button
                              type="button"
                              disabled={isRelaunchingFollowUp}
                              className="inline-flex items-center rounded-md border border-indigo-200 bg-indigo-50 px-2 py-1 text-[11px] font-medium text-indigo-700 hover:bg-indigo-100 disabled:cursor-not-allowed disabled:opacity-60"
                              onClick={(e) => {
                                e.stopPropagation();
                                relaunchFollowUpFromNotification(notification);
                              }}
                            >
                              {isRelaunchingFollowUp ? (
                                <>
                                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                  Relaunching...
                                </>
                              ) : (
                                'Relaunch Follow-up'
                              )}
                            </button>
                          )}
                        </div>
                      ) : null}
                      {reevaluationSummary && (reevaluationNoteAction || reevaluationJobId || originActionUrl || canRestartReevaluation) ? (
                        <div className="mt-2 flex flex-wrap gap-2">
                          {reevaluationNoteAction && (
                            <button
                              type="button"
                              className="inline-flex items-center rounded-md border border-indigo-200 bg-indigo-50 px-2 py-1 text-[11px] font-medium text-indigo-700 hover:bg-indigo-100"
                              onClick={async (e) => {
                                e.stopPropagation();
                                await openNotificationTarget(notification);
                              }}
                            >
                              Open note
                            </button>
                          )}
                          {reevaluationJobId && (
                            <button
                              type="button"
                              className="inline-flex items-center rounded-md border border-sky-200 bg-sky-50 px-2 py-1 text-[11px] font-medium text-sky-700 hover:bg-sky-100"
                              onClick={async (e) => {
                                e.stopPropagation();
                                await openReevaluationJob(notification);
                              }}
                            >
                              Open reevaluation job
                            </button>
                          )}
                          {originActionUrl && (
                            <button
                              type="button"
                              className="inline-flex items-center rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-[11px] font-medium text-emerald-700 hover:bg-emerald-100"
                              onClick={async (e) => {
                                e.stopPropagation();
                                await openOriginatingOpportunity(notification);
                              }}
                            >
                              Open originating opportunity
                            </button>
                          )}
                          {canRestartReevaluation && (
                            <button
                              type="button"
                              className="inline-flex items-center rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-[11px] font-medium text-amber-700 hover:bg-amber-100"
                              onClick={(e) => {
                                e.stopPropagation();
                                restartReevaluation(notification);
                              }}
                            >
                              Restart reevaluation
                            </button>
                          )}
                        </div>
                      ) : null}
                      {recoveryOpen && hasAgentJob && (
                        <div className="mt-2 flex flex-wrap gap-2">
                          {noteId && (
                            <button
                              type="button"
                              className="inline-flex items-center rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-[11px] font-medium text-emerald-700 hover:bg-emerald-100"
                              onClick={async (e) => {
                                e.stopPropagation();
                                await ensureNotificationRead(notification);
                                navigate(`/research-notes?note=${encodeURIComponent(noteId)}`);
                                setIsOpen(false);
                              }}
                            >
                              Open note
                            </button>
                          )}
                          <button
                            type="button"
                            className="inline-flex items-center rounded-md border border-sky-200 bg-sky-50 px-2 py-1 text-[11px] font-medium text-sky-700 hover:bg-sky-100"
                            onClick={async (e) => {
                              e.stopPropagation();
                              await ensureNotificationRead(notification);
                              navigate(`/autonomous-agents?job=${encodeURIComponent(String((notification.data as any)?.agent_job_id || ''))}`);
                              setIsOpen(false);
                            }}
                          >
                            Open job
                          </button>
                          <button
                            type="button"
                            className="inline-flex items-center rounded-md border border-rose-200 bg-rose-50 px-2 py-1 text-[11px] font-medium text-rose-700 hover:bg-rose-100"
                            onClick={(e) => {
                              e.stopPropagation();
                              restartRecoveryJob(notification);
                            }}
                          >
                            Restart job
                          </button>
                          {canRelaunchRecovery && (
                            <button
                              type="button"
                              className="inline-flex items-center rounded-md border border-indigo-200 bg-indigo-50 px-2 py-1 text-[11px] font-medium text-indigo-700 hover:bg-indigo-100"
                              onClick={(e) => {
                                e.stopPropagation();
                                relaunchRecoveryJob(notification);
                              }}
                            >
                              Relaunch clean run
                            </button>
                          )}
                          {firstFailedCommand && (
                            <button
                              type="button"
                              className="inline-flex items-center rounded-md border border-gray-200 bg-white px-2 py-1 text-[11px] font-medium text-gray-700 hover:bg-gray-50"
                              onClick={async (e) => {
                                e.stopPropagation();
                                await ensureNotificationRead(notification);
                                copyNotificationField(firstFailedCommand, 'Failed command');
                              }}
                            >
                              Copy failed command
                            </button>
                          )}
                          {recommendedAction && (
                            <button
                              type="button"
                              className="inline-flex items-center rounded-md border border-gray-200 bg-white px-2 py-1 text-[11px] font-medium text-gray-700 hover:bg-gray-50"
                              onClick={async (e) => {
                                e.stopPropagation();
                                await ensureNotificationRead(notification);
                                copyNotificationField(recommendedAction, 'Recovery next step');
                              }}
                            >
                              Copy next step
                            </button>
                          )}
                        </div>
                      )}
                      <p className="text-xs text-gray-400 mt-1">
                        {formatDistanceToNow(new Date(notification.created_at), { addSuffix: true })}
                      </p>
                    </div>
                    <div className="flex items-center gap-1 ml-2">
                      {notification.action_url && (
                        <ExternalLink className="w-4 h-4 text-gray-400" />
                      )}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          dismissNotification(notification.id);
                        }}
                        className="p-1 text-gray-400 hover:text-gray-600 rounded"
                        aria-label="Dismiss"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
                );
              })
            )}
          </div>

          {/* Footer */}
          {notifications.length > 0 && (
            <div className="px-4 py-2 border-t border-gray-200 text-center">
              <button
                onClick={() => {
                  navigate('/settings?tab=notifications');
                  setIsOpen(false);
                }}
                className="text-sm text-primary-600 hover:text-primary-700"
              >
                Manage notification settings
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default NotificationBell;
