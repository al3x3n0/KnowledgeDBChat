import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import NotificationBell from '../NotificationBell';
import { Notification } from '../../../types';

const mockNavigate = jest.fn();
const mockMarkAllAsRead = jest.fn().mockResolvedValue(undefined);
const mockDismissNotification = jest.fn().mockResolvedValue(undefined);
const mockPerformAgentJobAction = jest.fn().mockResolvedValue({});
const mockRelaunchInboxFollowUp = jest.fn().mockResolvedValue({});
const mockCreateSynthesisJob = jest.fn().mockResolvedValue({ id: 'syn-new-1' });
const mockWriteText = jest.fn().mockResolvedValue(undefined);

let mockNotifications: Notification[] = [];
const mockMarkAsRead = jest.fn().mockImplementation(async (notificationId: string) => {
  mockNotifications = mockNotifications.map((notification) =>
    notification.id === notificationId
      ? { ...notification, is_read: true }
      : notification
  );
});

jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

jest.mock('../../../contexts/NotificationContext', () => ({
  useNotifications: () => ({
    notifications: mockNotifications,
    unreadCount: mockNotifications.filter((notification) => !notification.is_read).length,
    isLoading: false,
    preferences: null,
    fetchNotifications: jest.fn(),
    markAsRead: mockMarkAsRead,
    markAllAsRead: mockMarkAllAsRead,
    dismissNotification: mockDismissNotification,
    updatePreferences: jest.fn(),
    refreshUnreadCount: jest.fn(),
  }),
}));

jest.mock('../../../services/api', () => ({
  apiClient: {
    performAgentJobAction: (...args: any[]) => mockPerformAgentJobAction(...args),
    relaunchInboxFollowUp: (...args: any[]) => mockRelaunchInboxFollowUp(...args),
    createSynthesisJob: (...args: any[]) => mockCreateSynthesisJob(...args),
  },
}));

describe('NotificationBell', () => {
  beforeEach(() => {
    mockNotifications = [
      {
        id: 'notif-1',
        notification_type: 'experiment_run_update',
        title: 'Experiment run failed',
        message: 'Recovery remains open.',
        priority: 'high',
        related_entity_type: 'experiment_run',
        related_entity_id: 'run-1',
        data: {
          agent_job_id: 'job-1',
          note_id: 'note-1',
          launch_mode: 'quick_start_claude_backend',
          final_phase: 'fallback',
          source_name: 'Knowledge Repo',
          fallback_attempted: true,
          fallback_ok: false,
          failed_command_count: 2,
          first_failed_command: 'npm --prefix frontend test',
          recovery_open: true,
          recovery_reason: 'fallback verification still failing',
          recommended_action: 'Inspect failing fallback output',
          latest_operator_action: 'restart',
          latest_operator_status_before: 'failed',
          latest_operator_status_after: 'pending',
          latest_operator_note: 'Retry after fallback failure',
          latest_operator_outcome: 'unresolved',
          latest_operator_outcome_reason: 'Job failed after intervention',
        },
        action_url: '/research-notes?note=note-1',
        is_read: false,
        created_at: '2026-03-11T12:00:00Z',
      },
      {
        id: 'notif-2',
        notification_type: 'experiment_run_update',
        title: 'Experiment run recovered',
        message: 'Fallback completed successfully.',
        priority: 'normal',
        related_entity_type: 'experiment_run',
        related_entity_id: 'run-2',
        data: {
          agent_job_id: 'job-2',
          final_phase: 'retry_primary',
          source_name: 'Frontend Repo',
          fallback_attempted: true,
          fallback_ok: true,
          failed_command_count: 1,
          recovery_open: false,
        },
        action_url: '/autonomous-agents?job=job-2',
        is_read: false,
        created_at: '2026-03-11T11:00:00Z',
      },
      {
        id: 'notif-4',
        notification_type: 'experiment_run_update',
        title: 'Older recovery still open',
        message: 'Another recovery remains open.',
        priority: 'high',
        related_entity_type: 'experiment_run',
        related_entity_id: 'run-4',
        data: {
          agent_job_id: 'job-4',
          final_phase: 'fallback',
          source_name: 'Backend Repo',
          fallback_attempted: true,
          fallback_ok: false,
          failed_command_count: 1,
          recovery_open: true,
          recovery_reason: 'secondary fallback still failing',
          recommended_action: 'Inspect backend fallback output',
        },
        action_url: '/autonomous-agents?job=job-4',
        is_read: false,
        created_at: '2026-03-11T09:00:00Z',
      },
      {
        id: 'notif-3',
        notification_type: 'queue_urgency_alert',
        title: 'Queue alert: Approval Required Job',
        message: 'approval checkpoint · overdue · escalation high',
        priority: 'high',
        related_entity_type: 'agent_job',
        related_entity_id: 'job-3',
        data: {
          queue_key: 'approval:job-3:checkpoint-1',
          queue_item_type: 'approval_checkpoint',
          job_id: 'job-3',
          sla_bucket: 'overdue',
          escalation_level: 'high',
          priority_score: 140,
          recommended_action: 'approve',
          reason_label: 'Approval required',
          customer: 'Acme',
          age_minutes: 240,
          is_overdue: true,
          is_stale: false,
          evidence_summary: 'Human approval required before next action.',
          scheduler_state: {
            queue_reason: 'execution_failure',
            last_run_status: 'failed',
            failure_streak: 2,
            last_scheduled_at: '2026-03-11T09:40:00Z',
            last_dispatched_at: '2026-03-11T09:45:00Z',
            current_run_started_at: '2026-03-11T09:46:00Z',
            backoff_until: '2026-03-11T10:15:00Z',
          },
        },
        action_url: '/autonomous-agents?tab=queue&job=job-3&queue_item_type=approval_checkpoint&queue_sla=overdue',
        is_read: false,
        created_at: '2026-03-11T10:00:00Z',
      },
      {
        id: 'notif-5',
        notification_type: 'system_maintenance',
        title: 'Planned maintenance window',
        message: 'Maintenance starts tonight.',
        priority: 'normal',
        action_url: '/settings?tab=notifications',
        is_read: true,
        created_at: '2026-03-11T10:00:00Z',
      },
    ];
    Object.assign(navigator, {
      clipboard: {
        writeText: mockWriteText,
      },
    });
    mockNavigate.mockReset();
    mockMarkAsRead.mockClear();
    mockMarkAllAsRead.mockClear();
    mockDismissNotification.mockClear();
    mockPerformAgentJobAction.mockClear();
    mockRelaunchInboxFollowUp.mockClear();
    mockCreateSynthesisJob.mockClear();
    mockWriteText.mockClear();
  });

  it('renders experiment recovery badges and guidance in the dropdown', () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    const failedCard = screen.getByText('Experiment run failed').closest('.border-l-4');
    expect(failedCard).not.toBeNull();
    const failedCardQueries = within(failedCard as HTMLElement);

    expect(screen.getByText('Experiment run failed')).toBeInTheDocument();
    expect(screen.getByText('Experiment run recovered')).toBeInTheDocument();
    expect(screen.getByText('Older recovery still open')).toBeInTheDocument();
    expect(screen.getByText('Planned maintenance window')).toBeInTheDocument();
    expect(failedCardQueries.getByText('Phase fallback')).toBeInTheDocument();
    expect(failedCardQueries.getByText('Repo Knowledge Repo')).toBeInTheDocument();
    expect(failedCardQueries.getByText('Fallback attempted')).toBeInTheDocument();
    expect(failedCardQueries.getByText('Recovery open')).toBeInTheDocument();
    expect(failedCardQueries.getByText('Failed cmds 2')).toBeInTheDocument();
    expect(failedCardQueries.getByText('Recovery Audit')).toBeInTheDocument();
    expect(
      failedCardQueries.getByText((_, element) => element?.textContent === 'Latest action: restart (failed -> pending)')
    ).toBeInTheDocument();
    expect(
      failedCardQueries.getByText((_, element) => element?.textContent === 'Outcome: unresolved')
    ).toBeInTheDocument();
    expect(
      failedCardQueries.getByText((_, element) => element?.textContent === 'Outcome reason: Job failed after intervention')
    ).toBeInTheDocument();
    expect(
      failedCardQueries.getByText((_, element) => element?.textContent === 'Recovery reason: fallback verification still failing')
    ).toBeInTheDocument();
    expect(
      failedCardQueries.getByText((_, element) => element?.textContent === 'Next step: Inspect failing fallback output')
    ).toBeInTheDocument();
    expect(failedCardQueries.getByText('Operator note: Retry after fallback failure')).toBeInTheDocument();
    expect(failedCardQueries.getByText('Open note')).toBeInTheDocument();
    expect(failedCardQueries.getByText('Open job')).toBeInTheDocument();
    expect(failedCardQueries.getByText('Restart job')).toBeInTheDocument();
    expect(failedCardQueries.getByText('Relaunch clean run')).toBeInTheDocument();
    expect(failedCardQueries.getByText('Copy failed command')).toBeInTheDocument();
    expect(failedCardQueries.getByText('Copy next step')).toBeInTheDocument();

    fireEvent.click(failedCardQueries.getByText('Restart job'));

    return waitFor(() => {
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-1');
      expect(mockPerformAgentJobAction).toHaveBeenCalledWith('job-1', 'restart', {});
    });
  });

  it('can filter down to open recovery notifications', () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    expect(screen.getByText('All 5')).toBeInTheDocument();
    expect(screen.getByText('Unread 4')).toBeInTheDocument();
    expect(screen.getByText('Experiment runs 3')).toBeInTheDocument();
    expect(screen.getByText('Queue alerts 1')).toBeInTheDocument();
    expect(screen.getByText('Open recovery 2')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Open recovery 2'));

    expect(screen.getByText('Experiment run failed')).toBeInTheDocument();
    expect(screen.getByText('Older recovery still open')).toBeInTheDocument();
    expect(screen.queryByText('Experiment run recovered')).not.toBeInTheDocument();
    expect(screen.queryByText('Planned maintenance window')).not.toBeInTheDocument();
  });

  it('can mark only filtered notifications as read', async () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Open recovery 2'));
    fireEvent.click(screen.getByText('Mark filtered read'));

    await waitFor(() => {
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-1');
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-4');
    });

    expect(mockMarkAsRead).not.toHaveBeenCalledWith('notif-2');
    expect(mockMarkAsRead).not.toHaveBeenCalledWith('notif-3');
    expect(mockMarkAsRead).not.toHaveBeenCalledWith('notif-5');
  });

  it('can dismiss only filtered notifications', async () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Open recovery 2'));
    fireEvent.click(screen.getByText('Dismiss filtered'));

    await waitFor(() => {
      expect(mockDismissNotification).toHaveBeenCalledWith('notif-1');
      expect(mockDismissNotification).toHaveBeenCalledWith('notif-4');
    });

    expect(mockDismissNotification).not.toHaveBeenCalledWith('notif-2');
    expect(mockDismissNotification).not.toHaveBeenCalledWith('notif-3');
    expect(mockDismissNotification).not.toHaveBeenCalledWith('notif-5');
  });

  it('renders and filters queue urgency alerts', async () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Queue alerts 1'));

    expect(screen.getByText('Queue alert: Approval Required Job')).toBeInTheDocument();
    expect(screen.getByText('approval checkpoint')).toBeInTheDocument();
    expect(screen.getByText('overdue')).toBeInTheDocument();
    expect(screen.getByText('Esc high')).toBeInTheDocument();
    expect(screen.getByText('Customer Acme')).toBeInTheDocument();
    expect(screen.getByText('Age 240m')).toBeInTheDocument();
    expect(screen.getByText('Recovery Audit')).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Recovery reason: Approval required')
    ).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Next step: approve')
    ).toBeInTheDocument();
    expect(screen.getByText('Scheduler state')).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Queue reason: execution failure')
    ).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Last run: failed')
    ).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Failure streak: 2')
    ).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Last scheduled: 2026-03-11T09:40:00Z')
    ).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Last dispatched: 2026-03-11T09:45:00Z')
    ).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Current run: 2026-03-11T09:46:00Z')
    ).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Backoff until: 2026-03-11T10:15:00Z')
    ).toBeInTheDocument();
    expect(screen.getByText('Evidence: Human approval required before next action.')).toBeInTheDocument();
    expect(screen.queryByText('Experiment run failed')).not.toBeInTheDocument();

    fireEvent.click(screen.getByText('Queue alert: Approval Required Job'));

    await waitFor(() => {
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-3');
      expect(mockNavigate).toHaveBeenCalledWith(
        '/autonomous-agents?tab=queue&job=job-3&queue_item_type=approval_checkpoint&queue_sla=overdue'
      );
    });
  });

  it('can reset back to all notifications from an empty filtered state', () => {
    mockNotifications = mockNotifications.map((notification) => ({
      ...notification,
      is_read: true,
    }));

    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Unread 0'));

    expect(screen.getByText('No notifications match this filter')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Show all notifications'));

    expect(screen.getByText('Experiment run failed')).toBeInTheDocument();
    expect(screen.getByText('Experiment run recovered')).toBeInTheDocument();
    expect(screen.getByText('Older recovery still open')).toBeInTheDocument();
    expect(screen.getByText('Planned maintenance window')).toBeInTheDocument();
  });

  it('can filter down to experiment run notifications', () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Experiment runs 3'));

    expect(screen.getByText('Experiment run failed')).toBeInTheDocument();
    expect(screen.getByText('Experiment run recovered')).toBeInTheDocument();
    expect(screen.getByText('Older recovery still open')).toBeInTheDocument();
    expect(screen.queryByText('Planned maintenance window')).not.toBeInTheDocument();
  });

  it('can filter down to unread notifications', () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Unread 4'));

    expect(screen.getByText('Experiment run failed')).toBeInTheDocument();
    expect(screen.getByText('Experiment run recovered')).toBeInTheDocument();
    expect(screen.getByText('Older recovery still open')).toBeInTheDocument();
    expect(screen.queryByText('Planned maintenance window')).not.toBeInTheDocument();
  });

  it('can filter down to unread recovery notifications', () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Unread recovery 2'));

    expect(screen.getByText('Experiment run failed')).toBeInTheDocument();
    expect(screen.getByText('Older recovery still open')).toBeInTheDocument();
    expect(screen.queryByText('Experiment run recovered')).not.toBeInTheDocument();
    expect(screen.queryByText('Planned maintenance window')).not.toBeInTheDocument();
  });

  it('exposes the active filter via aria-pressed', () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));

    const allFilter = screen.getByRole('button', { name: 'All 5' });
    const unreadRecoveryFilter = screen.getByRole('button', { name: 'Unread recovery 2' });

    expect(allFilter).toHaveAttribute('aria-pressed', 'true');
    expect(unreadRecoveryFilter).toHaveAttribute('aria-pressed', 'false');

    fireEvent.click(unreadRecoveryFilter);

    expect(allFilter).toHaveAttribute('aria-pressed', 'false');
    expect(unreadRecoveryFilter).toHaveAttribute('aria-pressed', 'true');
  });

  it('shows a live filter summary for non-default views', () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    expect(screen.queryByText('Showing 2 of 5 notifications for Unread recovery')).not.toBeInTheDocument();

    fireEvent.click(screen.getByText('Unread recovery 2'));
    expect(screen.getByText('Showing 2 of 5 notifications for Unread recovery')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Experiment runs 3'));
    expect(screen.getByText('Showing 3 of 5 notifications for Experiment runs')).toBeInTheDocument();
  });

  it('can clear the active filter from the summary row', () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Unread recovery 2'));

    expect(screen.getByText('Showing 2 of 5 notifications for Unread recovery')).toBeInTheDocument();
    expect(screen.queryByText('Experiment run recovered')).not.toBeInTheDocument();

    fireEvent.click(screen.getByText('Clear filter'));

    expect(screen.queryByText('Showing 2 of 5 notifications for Unread recovery')).not.toBeInTheDocument();
    expect(screen.getByText('Experiment run failed')).toBeInTheDocument();
    expect(screen.getByText('Older recovery still open')).toBeInTheDocument();
    expect(screen.getByText('Experiment run recovered')).toBeInTheDocument();
    expect(screen.getByText('Planned maintenance window')).toBeInTheDocument();
  });

  it('resets back to all notifications when the bell closes', () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Unread recovery 2'));
    expect(screen.queryByText('Experiment run recovered')).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByLabelText(/notifications/i));

    expect(screen.getByText('Experiment run failed')).toBeInTheDocument();
    expect(screen.getByText('Older recovery still open')).toBeInTheDocument();
    expect(screen.getByText('Experiment run recovered')).toBeInTheDocument();
    expect(screen.getByText('Planned maintenance window')).toBeInTheDocument();
  });

  it('closes on Escape and reopens on the default filter', () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Unread recovery 2'));
    expect(screen.queryByText('Experiment run recovered')).not.toBeInTheDocument();

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByText('Notifications')).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText(/notifications/i));
    expect(screen.getByText('Experiment run failed')).toBeInTheDocument();
    expect(screen.getByText('Older recovery still open')).toBeInTheDocument();
    expect(screen.getByText('Experiment run recovered')).toBeInTheDocument();
    expect(screen.getByText('Planned maintenance window')).toBeInTheDocument();
  });

  it('sorts unread open recovery notifications ahead of other items', () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));

    const titles = screen.getAllByText(
      /Experiment run failed|Older recovery still open|Experiment run recovered|Planned maintenance window/
    );

    expect(titles.map((node) => node.textContent)).toEqual([
      'Experiment run failed',
      'Older recovery still open',
      'Experiment run recovered',
      'Planned maintenance window',
    ]);
  });

  it('can relaunch a quick-start recovery job from the dropdown', async () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Relaunch clean run'));

    await waitFor(() => {
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-1');
      expect(mockPerformAgentJobAction).toHaveBeenCalledWith('job-1', 'relaunch', {});
    });
  });

  it('can open the linked autonomous job from the dropdown', async () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    const failedCard = screen.getByText('Experiment run failed').closest('.border-l-4');
    expect(failedCard).not.toBeNull();
    fireEvent.click(within(failedCard as HTMLElement).getByText('Open job'));

    await waitFor(() => {
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-1');
      expect(mockNavigate).toHaveBeenCalledWith('/autonomous-agents?job=job-1');
    });
  });

  it('can open the linked research note from the dropdown', async () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Open note'));

    await waitFor(() => {
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-1');
      expect(mockNavigate).toHaveBeenCalledWith('/research-notes?note=note-1');
    });
  });

  it('opens follow-up outcome notifications on the targeted domain opportunity deep link', async () => {
    mockNotifications = [
      {
        id: 'notif-follow-up-1',
        notification_type: 'follow_up_outcome_alert',
        title: 'Follow-up completed: Compiler hotspot',
        message: 'Validated the hotspot and documented next steps.',
        priority: 'normal',
        related_entity_type: 'research_inbox_item',
        related_entity_id: 'inbox-1',
        data: {
          inbox_item_id: 'inbox-1',
          follow_up_job_id: 'job-follow-up-1',
          follow_up_last_job_id: 'job-follow-up-1',
          follow_up_outcome_status: 'completed',
          follow_up_outcome_summary: 'Validated the hotspot and documented next steps.',
          origin_source_kind: 'profile',
          origin_source_id: 'profile-1',
          origin_opportunity_id: 'opp-profile-1',
        },
        action_url: '/autonomous-agents?tab=domain&profileId=profile-1&opportunityId=opp-profile-1',
        is_read: false,
        created_at: '2026-03-12T12:00:00Z',
      },
    ];

    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Follow-up completed: Compiler hotspot'));

    await waitFor(() => {
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-follow-up-1');
      expect(mockNavigate).toHaveBeenCalledWith(
        '/autonomous-agents?tab=domain&profileId=profile-1&opportunityId=opp-profile-1'
      );
    });
  });

  it('can open the linked follow-up job from a follow-up outcome notification', async () => {
    mockNotifications = [
      {
        id: 'notif-follow-up-job-1',
        notification_type: 'follow_up_outcome_alert',
        title: 'Follow-up failed: Compiler hotspot',
        message: 'Benchmark verification failed.',
        priority: 'high',
        related_entity_type: 'research_inbox_item',
        related_entity_id: 'inbox-2',
        data: {
          inbox_item_id: 'inbox-2',
          follow_up_job_id: 'job-follow-up-2',
          follow_up_last_job_id: 'job-follow-up-2',
          follow_up_outcome_status: 'failed',
          follow_up_outcome_summary: 'Benchmark verification failed.',
          origin_source_kind: 'profile',
          origin_source_id: 'profile-2',
          origin_opportunity_id: 'opp-profile-2',
        },
        action_url: '/autonomous-agents?tab=domain&profileId=profile-2&opportunityId=opp-profile-2',
        is_read: false,
        created_at: '2026-03-12T13:00:00Z',
      },
    ];

    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Open follow-up job'));

    await waitFor(() => {
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-follow-up-job-1');
      expect(mockNavigate).toHaveBeenCalledWith('/autonomous-agents?job=job-follow-up-2');
    });
  });

  it('can relaunch a failed follow-up outcome from the notification bell', async () => {
    mockNotifications = [
      {
        id: 'notif-follow-up-relaunch-1',
        notification_type: 'follow_up_outcome_alert',
        title: 'Follow-up failed: Compiler hotspot',
        message: 'Benchmark verification failed.',
        priority: 'high',
        related_entity_type: 'research_inbox_item',
        related_entity_id: 'inbox-3',
        data: {
          inbox_item_id: 'inbox-3',
          follow_up_job_id: 'job-follow-up-3',
          follow_up_last_job_id: 'job-follow-up-3',
          follow_up_outcome_status: 'failed',
          follow_up_outcome_summary: 'Benchmark verification failed.',
        },
        action_url: '/autonomous-agents?tab=inbox&inbox=inbox-3',
        is_read: false,
        created_at: '2026-03-12T14:00:00Z',
      },
    ];

    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Relaunch Follow-up'));

    await waitFor(() => {
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-follow-up-relaunch-1');
      expect(mockRelaunchInboxFollowUp).toHaveBeenCalledWith('inbox-3', {});
    });
  });

  it('does not render relaunch controls for completed follow-up outcome notifications', () => {
    mockNotifications = [
      {
        id: 'notif-follow-up-complete-1',
        notification_type: 'follow_up_outcome_alert',
        title: 'Follow-up completed: Compiler hotspot',
        message: 'Validated the hotspot and documented next steps.',
        priority: 'normal',
        related_entity_type: 'research_inbox_item',
        related_entity_id: 'inbox-4',
        data: {
          inbox_item_id: 'inbox-4',
          follow_up_job_id: 'job-follow-up-4',
          follow_up_last_job_id: 'job-follow-up-4',
          follow_up_outcome_status: 'completed',
          follow_up_outcome_summary: 'Validated the hotspot and documented next steps.',
        },
        action_url: '/autonomous-agents?tab=inbox&inbox=inbox-4',
        is_read: false,
        created_at: '2026-03-12T15:00:00Z',
      },
    ];

    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    expect(screen.queryByText('Relaunch Follow-up')).not.toBeInTheDocument();
  });

  it('opens completed reevaluation notifications on the research note and exposes origin/job links', async () => {
    mockNotifications = [
      {
        id: 'notif-reeval-1',
        notification_type: 'hypothesis_reevaluation_update',
        title: 'Reevaluation ready: Compiler note',
        message: 'Hypotheses were re-scored using the latest experiment evidence.',
        priority: 'normal',
        related_entity_type: 'research_note',
        related_entity_id: 'note-2',
        data: {
          note_id: 'note-2',
          note_title: 'Compiler note',
          reevaluation_job_id: 'syn-reeval-1',
          reevaluation_status: 'completed',
          source_run_ids: ['run-1'],
          reprioritization_summary: 'Hypothesis A moved ahead after the benchmark result.',
          origin_source_kind: 'profile',
          origin_source_id: 'profile-9',
          origin_opportunity_id: 'opp-9',
          origin_action_url: '/autonomous-agents?tab=domain&profileId=profile-9&opportunityId=opp-9',
        },
        action_url: '/research-notes?note=note-2',
        is_read: false,
        created_at: '2026-03-12T16:00:00Z',
      },
    ];

    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    expect(screen.getByText('Open note')).toBeInTheDocument();
    expect(screen.getByText('Open reevaluation job')).toBeInTheDocument();
    expect(screen.getByText('Open originating opportunity')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Open reevaluation job'));

    await waitFor(() => {
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-reeval-1');
      expect(mockNavigate).toHaveBeenCalledWith('/synthesis?job=syn-reeval-1');
    });
  });

  it('can restart a failed reevaluation notification from the bell', async () => {
    mockNotifications = [
      {
        id: 'notif-reeval-2',
        notification_type: 'hypothesis_reevaluation_update',
        title: 'Reevaluation failed: Compiler note',
        message: 'Model timeout during reevaluation.',
        priority: 'high',
        related_entity_type: 'research_note',
        related_entity_id: 'note-3',
        data: {
          note_id: 'note-3',
          note_title: 'Compiler note',
          reevaluation_job_id: 'syn-reeval-2',
          reevaluation_status: 'failed',
          reevaluation_error: 'Model timeout during reevaluation',
        },
        action_url: '/synthesis?job=syn-reeval-2',
        is_read: false,
        created_at: '2026-03-12T17:00:00Z',
      },
    ];

    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Restart reevaluation'));

    await waitFor(() => {
      expect(mockCreateSynthesisJob).toHaveBeenCalledWith({
        job_type: 'hypothesis_reevaluation',
        title: 'Hypothesis Re-evaluation · Compiler note',
        document_ids: [],
        research_note_id: 'note-3',
        output_format: 'markdown',
        output_style: 'technical',
      });
    });
  });

  it('can copy the failed command from the dropdown', async () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    fireEvent.click(screen.getByText('Copy failed command'));

    await waitFor(() => {
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-1');
      expect(mockWriteText).toHaveBeenCalledWith('npm --prefix frontend test');
    });
  });

  it('can copy the recommended next step from the dropdown', async () => {
    render(<NotificationBell />);

    fireEvent.click(screen.getByLabelText(/notifications/i));
    const failedCard = screen.getByText('Experiment run failed').closest('.border-l-4');
    expect(failedCard).not.toBeNull();
    fireEvent.click(within(failedCard as HTMLElement).getByText('Copy next step'));

    await waitFor(() => {
      expect(mockMarkAsRead).toHaveBeenCalledWith('notif-1');
      expect(mockWriteText).toHaveBeenCalledWith('Inspect failing fallback output');
    });
  });
});
