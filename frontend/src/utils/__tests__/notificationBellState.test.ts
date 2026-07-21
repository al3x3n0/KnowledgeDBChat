import { Notification } from '../../types';
import {
  getNotificationBellCounts,
  getNotificationBellFilterLabel,
  getNotificationBellHeaderActionsState,
  getNotificationBellRank,
  getVisibleNotificationsForBell,
} from '../notificationBellState';

function makeNotifications(): Notification[] {
  return [
    {
      id: 'notif-1',
      notification_type: 'experiment_run_update',
      title: 'Experiment run failed',
      message: 'Recovery remains open.',
      priority: 'high',
      data: { recovery_open: true },
      is_read: false,
      created_at: '2026-03-11T12:00:00Z',
    },
    {
      id: 'notif-2',
      notification_type: 'experiment_run_update',
      title: 'Experiment run recovered',
      message: 'Fallback completed successfully.',
      priority: 'normal',
      data: { recovery_open: false },
      is_read: false,
      created_at: '2026-03-11T11:00:00Z',
    },
    {
      id: 'notif-3',
      notification_type: 'queue_urgency_alert',
      title: 'Queue alert',
      message: 'Approval overdue.',
      priority: 'high',
      data: { queue_item_type: 'approval_checkpoint' },
      is_read: true,
      created_at: '2026-03-11T10:00:00Z',
    },
    {
      id: 'notif-3b',
      notification_type: 'policy_guardrail_alert',
      title: 'Policy safeguard',
      message: 'Rollback recommended.',
      priority: 'high',
      data: { policy_guardrail_action: 'rollback' },
      is_read: true,
      created_at: '2026-03-11T09:30:00Z',
    },
    {
      id: 'notif-4',
      notification_type: 'follow_up_outcome_alert',
      title: 'Follow-up failed',
      message: 'The follow-up failed.',
      priority: 'high',
      data: { follow_up_outcome_status: 'failed' },
      is_read: true,
      created_at: '2026-03-11T09:00:00Z',
    },
    {
      id: 'notif-4b',
      notification_type: 'hypothesis_reevaluation_update',
      title: 'Reevaluation ready',
      message: 'A reevaluation is ready to review.',
      priority: 'normal',
      data: { reevaluation_status: 'completed' },
      is_read: false,
      created_at: '2026-03-11T08:30:00Z',
    },
    {
      id: 'notif-5',
      notification_type: 'system_maintenance',
      title: 'Planned maintenance window',
      message: 'Maintenance starts tonight.',
      priority: 'normal',
      is_read: true,
      created_at: '2026-03-11T08:00:00Z',
    },
    {
      id: 'notif-6',
      notification_type: 'system_maintenance',
      title: 'Secondary system message',
      message: 'FYI.',
      priority: 'normal',
      is_read: true,
      created_at: '2026-03-11T07:00:00Z',
    },
    {
      id: 'notif-7',
      notification_type: 'experiment_run_update',
      title: 'Older recovery still open',
      message: 'Another recovery remains open.',
      priority: 'high',
      data: { recovery_open: true },
      is_read: false,
      created_at: '2026-03-11T06:00:00Z',
    },
  ];
}

describe('notificationBellState', () => {
  it('computes bell counts for unread and recovery slices', () => {
    expect(getNotificationBellCounts(makeNotifications())).toEqual({
      unread: 4,
      unreadRecovery: 2,
      experimentRuns: 3,
      reevaluations: 1,
      queueAlerts: 2,
      followUpOutcomes: 1,
      recoveryOpen: 2,
    });
  });

  it('returns stable labels for bell filter modes', () => {
    expect(getNotificationBellFilterLabel('all')).toBe('All');
    expect(getNotificationBellFilterLabel('unread')).toBe('Unread');
    expect(getNotificationBellFilterLabel('unread_recovery')).toBe('Unread recovery');
    expect(getNotificationBellFilterLabel('experiment_runs')).toBe('Experiment runs');
    expect(getNotificationBellFilterLabel('reevaluations')).toBe('Reevaluations');
    expect(getNotificationBellFilterLabel('queue_alerts')).toBe('Queue alerts');
    expect(getNotificationBellFilterLabel('follow_up_outcomes')).toBe('Follow-up outcomes');
    expect(getNotificationBellFilterLabel('recovery_open')).toBe('Open recovery');
  });

  it('ranks unread recovery notifications ahead of other items', () => {
    const notifications = makeNotifications();
    expect(getNotificationBellRank(notifications.find((notification) => notification.id === 'notif-1')!)).toBe(0);
    expect(getNotificationBellRank(notifications.find((notification) => notification.id === 'notif-7')!)).toBe(0);
    expect(getNotificationBellRank(notifications.find((notification) => notification.id === 'notif-2')!)).toBe(2);
    expect(getNotificationBellRank(notifications.find((notification) => notification.id === 'notif-3')!)).toBe(3);
  });

  it('filters unread recovery notifications and preserves sort priority', () => {
    const visible = getVisibleNotificationsForBell(makeNotifications(), 'unread_recovery');
    expect(visible.map((notification) => notification.id)).toEqual(['notif-1', 'notif-7']);
  });

  it('sorts all notifications by recovery priority then recency', () => {
    const visible = getVisibleNotificationsForBell(makeNotifications(), 'all');
    expect(visible.map((notification) => notification.id)).toEqual([
      'notif-1',
      'notif-7',
      'notif-2',
      'notif-4b',
      'notif-3',
      'notif-3b',
      'notif-4',
      'notif-5',
      'notif-6',
    ]);
  });

  it('filters reevaluation notifications', () => {
    const visible = getVisibleNotificationsForBell(makeNotifications(), 'reevaluations');
    expect(visible.map((notification) => notification.id)).toEqual(['notif-4b']);
  });

  it('derives header action visibility from the filtered view', () => {
    const unreadRecoveryVisible = getVisibleNotificationsForBell(makeNotifications(), 'unread_recovery');
    expect(
      getNotificationBellHeaderActionsState('unread_recovery', unreadRecoveryVisible)
    ).toEqual({
      showMarkFilteredRead: true,
      showDismissFiltered: true,
    });

    const allVisible = getVisibleNotificationsForBell(makeNotifications(), 'all');
    expect(getNotificationBellHeaderActionsState('all', allVisible)).toEqual({
      showMarkFilteredRead: false,
      showDismissFiltered: false,
    });

    const emptyVisible = getVisibleNotificationsForBell(
      makeNotifications().map((notification) => ({ ...notification, is_read: true })),
      'unread',
    );
    expect(getNotificationBellHeaderActionsState('unread', emptyVisible)).toEqual({
      showMarkFilteredRead: false,
      showDismissFiltered: false,
    });
  });
});
