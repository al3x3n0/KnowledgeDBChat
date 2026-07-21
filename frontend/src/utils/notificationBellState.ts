import { Notification } from '../types';

export type NotificationBellFilterMode =
  | 'all'
  | 'unread'
  | 'experiment_runs'
  | 'reevaluations'
  | 'queue_alerts'
  | 'follow_up_outcomes'
  | 'recovery_open'
  | 'unread_recovery';

export interface NotificationBellCounts {
  unread: number;
  unreadRecovery: number;
  experimentRuns: number;
  reevaluations: number;
  queueAlerts: number;
  followUpOutcomes: number;
  recoveryOpen: number;
}

export interface NotificationBellHeaderActionsState {
  showMarkFilteredRead: boolean;
  showDismissFiltered: boolean;
}

export function getNotificationBellFilterLabel(filterMode: NotificationBellFilterMode): string {
  switch (filterMode) {
    case 'unread':
      return 'Unread';
    case 'unread_recovery':
      return 'Unread recovery';
    case 'experiment_runs':
      return 'Experiment runs';
    case 'reevaluations':
      return 'Reevaluations';
    case 'queue_alerts':
      return 'Queue alerts';
    case 'follow_up_outcomes':
      return 'Follow-up outcomes';
    case 'recovery_open':
      return 'Open recovery';
    default:
      return 'All';
  }
}

function isRecoveryOpen(notification: Notification): boolean {
  return Boolean((notification.data as any)?.recovery_open);
}

export function getNotificationBellRank(notification: Notification): number {
  const recoveryOpen = isRecoveryOpen(notification);
  if (recoveryOpen && !notification.is_read) return 0;
  if (recoveryOpen) return 1;
  if (!notification.is_read) return 2;
  return 3;
}

export function getNotificationBellCounts(notifications: Notification[]): NotificationBellCounts {
  return notifications.reduce<NotificationBellCounts>(
    (counts, notification) => {
      if (!notification.is_read) {
        counts.unread += 1;
      }
      if (notification.notification_type === 'experiment_run_update') {
        counts.experimentRuns += 1;
      }
      if (notification.notification_type === 'hypothesis_reevaluation_update') {
        counts.reevaluations += 1;
      }
      if (notification.notification_type === 'queue_urgency_alert') {
        counts.queueAlerts += 1;
      }
      if (notification.notification_type === 'policy_guardrail_alert') {
        counts.queueAlerts += 1;
      }
      if (notification.notification_type === 'autonomy_budget_alert') {
        counts.queueAlerts += 1;
      }
      if (notification.notification_type === 'customer_autonomy_budget_alert') {
        counts.queueAlerts += 1;
      }
      if (notification.notification_type === 'follow_up_outcome_alert') {
        counts.followUpOutcomes += 1;
      }
      if (isRecoveryOpen(notification)) {
        counts.recoveryOpen += 1;
        if (!notification.is_read) {
          counts.unreadRecovery += 1;
        }
      }
      return counts;
    },
    {
      unread: 0,
      unreadRecovery: 0,
      experimentRuns: 0,
      reevaluations: 0,
      queueAlerts: 0,
      followUpOutcomes: 0,
      recoveryOpen: 0,
    },
  );
}

export function getVisibleNotificationsForBell(
  notifications: Notification[],
  filterMode: NotificationBellFilterMode,
): Notification[] {
  return notifications
    .slice()
    .sort((a, b) => {
      const rankDelta = getNotificationBellRank(a) - getNotificationBellRank(b);
      if (rankDelta !== 0) return rankDelta;
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    })
    .filter((notification) => {
      if (filterMode === 'unread_recovery') {
        return !notification.is_read && isRecoveryOpen(notification);
      }
      if (filterMode === 'unread') {
        return !notification.is_read;
      }
      if (filterMode === 'experiment_runs') {
        return notification.notification_type === 'experiment_run_update';
      }
      if (filterMode === 'reevaluations') {
        return notification.notification_type === 'hypothesis_reevaluation_update';
      }
      if (filterMode === 'queue_alerts') {
        return notification.notification_type === 'queue_urgency_alert'
          || notification.notification_type === 'policy_guardrail_alert'
          || notification.notification_type === 'autonomy_budget_alert'
          || notification.notification_type === 'customer_autonomy_budget_alert';
      }
      if (filterMode === 'follow_up_outcomes') {
        return notification.notification_type === 'follow_up_outcome_alert';
      }
      if (filterMode === 'recovery_open') {
        return isRecoveryOpen(notification);
      }
      return true;
    })
    .slice(0, 10);
}

export function getNotificationBellHeaderActionsState(
  filterMode: NotificationBellFilterMode,
  visibleNotifications: Notification[],
): NotificationBellHeaderActionsState {
  const isFilteredView = filterMode !== 'all';
  return {
    showMarkFilteredRead: isFilteredView && visibleNotifications.some((notification) => !notification.is_read),
    showDismissFiltered: isFilteredView && visibleNotifications.length > 0,
  };
}
