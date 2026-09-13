import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import toast from 'react-hot-toast';
import { NotificationProvider, useNotifications } from '../NotificationContext';
import { Notification } from '../../types';

const mockGetNotifications = jest.fn();
const mockGetNotificationPreferences = jest.fn();
const mockCreateNotificationsWebSocket = jest.fn();
const mockMarkNotificationRead = jest.fn().mockResolvedValue(undefined);
const mockRelaunchInboxFollowUp = jest.fn().mockResolvedValue({});
const mockCreateSynthesisJob = jest.fn().mockResolvedValue({ id: 'syn-new-1' });

type MockSocket = {
  readyState: number;
  onopen: null | (() => void);
  onmessage: null | ((event: { data: string }) => void);
  onclose: null | ((event: { code: number; reason: string }) => void);
  onerror: null | ((event: unknown) => void);
  close: jest.Mock<void, [number?, string?]>;
};

let mockSocket: MockSocket;

jest.mock('../AuthContext', () => ({
  useAuth: () => ({
    user: { id: 'user-1' },
  }),
}));

jest.mock('../../services/api', () => ({
  apiClient: {
    getNotifications: (...args: any[]) => mockGetNotifications(...args),
    getNotificationPreferences: (...args: any[]) => mockGetNotificationPreferences(...args),
    createNotificationsWebSocket: (...args: any[]) => mockCreateNotificationsWebSocket(...args),
    markNotificationRead: (...args: any[]) => mockMarkNotificationRead(...args),
    markAllNotificationsRead: jest.fn(),
    dismissNotification: jest.fn(),
    updateNotificationPreferences: jest.fn(),
    getUnreadCount: jest.fn(),
    relaunchInboxFollowUp: (...args: any[]) => mockRelaunchInboxFollowUp(...args),
    createSynthesisJob: (...args: any[]) => mockCreateSynthesisJob(...args),
  },
}));

jest.mock('react-hot-toast', () => {
  const toastFn = jest.fn();
  (toastFn as any).custom = jest.fn();
  (toastFn as any).dismiss = jest.fn();
  (toastFn as any).success = jest.fn();
  (toastFn as any).error = jest.fn();
  return {
    __esModule: true,
    default: toastFn,
  };
});

function NotificationStateProbe() {
  const { unreadCount, notifications, fetchNotifications } = useNotifications();
  return (
    <div>
      <span data-testid="unread-count">{unreadCount}</span>
      <span data-testid="notification-count">{notifications.length}</span>
      <button type="button" onClick={() => { void fetchNotifications(); }}>
        refresh notifications
      </button>
    </div>
  );
}

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

function makeNotification(): Notification {
  return {
    id: 'notif-1',
    notification_type: 'experiment_run_update',
    title: 'Experiment run failed',
    message: 'Recovery remains open.',
    priority: 'high',
    related_entity_type: 'experiment_run',
    related_entity_id: 'run-1',
    data: {
      final_phase: 'fallback',
      source_name: 'Knowledge Repo',
      fallback_attempted: true,
      fallback_ok: false,
      failed_command_count: 2,
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
    action_url: '/autonomous-agents?job=job-1',
    is_read: false,
    created_at: '2026-03-12T12:00:00Z',
  };
}

function makeFollowUpOutcomeNotification(overrides?: Partial<Notification>): Notification {
  return {
    id: 'notif-follow-up-1',
    notification_type: 'follow_up_outcome_alert',
    title: 'Follow-up failed: Compiler hotspot',
    message: 'Benchmark verification failed.',
    priority: 'normal',
    related_entity_type: 'research_inbox_item',
    related_entity_id: 'inbox-1',
    data: {
      inbox_item_id: 'inbox-1',
      follow_up_job_id: 'job-follow-up-1',
      follow_up_last_job_id: 'job-follow-up-1',
      follow_up_outcome_status: 'failed',
      follow_up_outcome_summary: 'Benchmark verification failed.',
      origin_source_kind: 'profile',
      origin_source_id: 'profile-1',
      origin_opportunity_id: 'opp-profile-1',
    },
    action_url: '/autonomous-agents?tab=domain&profileId=profile-1&opportunityId=opp-profile-1',
    is_read: false,
    created_at: '2026-03-12T12:00:00Z',
    ...overrides,
  };
}

function makeReevaluationNotification(overrides?: Partial<Notification>): Notification {
  return {
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
      origin_source_id: 'profile-1',
      origin_opportunity_id: 'opp-profile-1',
      origin_action_url: '/autonomous-agents?tab=domain&profileId=profile-1&opportunityId=opp-profile-1',
    },
    action_url: '/research-notes?note=note-2',
    is_read: false,
    created_at: '2026-03-12T12:00:00Z',
    ...overrides,
  };
}

describe('NotificationContext', () => {
  beforeEach(() => {
    mockSocket = {
      readyState: 1,
      onopen: null,
      onmessage: null,
      onclose: null,
      onerror: null,
      close: jest.fn(),
    };
    mockGetNotifications.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 20,
      unread_count: 0,
    });
    mockGetNotificationPreferences.mockResolvedValue({
      play_sound: false,
    });
    mockCreateNotificationsWebSocket.mockReturnValue(mockSocket);
    (toast as unknown as jest.Mock).mockClear();
    ((toast as any).custom as jest.Mock).mockClear();
    ((toast as any).dismiss as jest.Mock).mockClear();
    ((toast as any).success as jest.Mock).mockClear();
    ((toast as any).error as jest.Mock).mockClear();
    mockMarkNotificationRead.mockClear();
    mockRelaunchInboxFollowUp.mockClear();
    mockCreateSynthesisJob.mockClear();
    mockGetNotifications.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 20,
      unread_count: 0,
    });
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: {
        assign: jest.fn(),
      },
    });
  });

  it('emits an enriched toast summary for live experiment recovery notifications', async () => {
    renderWithProviders(
      <NotificationProvider>
        <NotificationStateProbe />
      </NotificationProvider>,
    );

    await waitFor(() => {
      expect(mockCreateNotificationsWebSocket).toHaveBeenCalled();
    });

    act(() => {
      mockSocket.onmessage?.({
        data: JSON.stringify({
          type: 'notification',
          notification: makeNotification(),
        }),
      });
    });

    await waitFor(() => {
      expect(screen.getByTestId('unread-count')).toHaveTextContent('1');
    });
    expect(screen.getByTestId('notification-count')).toHaveTextContent('1');

    expect(toast).toHaveBeenCalledWith(
      expect.stringContaining('Experiment run failed'),
      expect.objectContaining({
        icon: 'i',
        duration: 5000,
      }),
    );

    const toastMessage = (toast as unknown as jest.Mock).mock.calls[0]?.[0] as string;
    expect(toastMessage).toContain('Phase fallback');
    expect(toastMessage).toContain('Repo Knowledge Repo');
    expect(toastMessage).toContain('Fallback attempted');
    expect(toastMessage).toContain('Reason: fallback verification still failing');
    expect(toastMessage).toContain('Next: Inspect failing fallback output');
    expect(toastMessage).toContain('Last operator: restart (failed -> pending)');
    expect(toastMessage).toContain('Operator outcome: unresolved');
    expect(toastMessage).toContain('Outcome reason: Job failed after intervention');
  });

  it('renders an actionable custom toast for live follow-up outcome notifications', async () => {
    renderWithProviders(
      <NotificationProvider>
        <NotificationStateProbe />
      </NotificationProvider>,
    );

    await waitFor(() => {
      expect(mockCreateNotificationsWebSocket).toHaveBeenCalled();
    });

    act(() => {
      mockSocket.onmessage?.({
        data: JSON.stringify({
          type: 'notification',
          notification: makeFollowUpOutcomeNotification(),
        }),
      });
    });

    await waitFor(() => {
      expect(((toast as any).custom as jest.Mock)).toHaveBeenCalled();
    });
    expect(screen.getByTestId('unread-count')).toHaveTextContent('1');
    expect(screen.getByTestId('notification-count')).toHaveTextContent('1');

    const renderToast = ((toast as any).custom as jest.Mock).mock.calls[0]?.[0] as () => React.ReactElement;
    render(renderToast());

    expect(screen.getByText('Follow-up failed: Compiler hotspot')).toBeInTheDocument();
    expect(screen.getByText(/Benchmark verification failed\./i)).toBeInTheDocument();
    expect(screen.getByText('Open target')).toBeInTheDocument();
    expect(screen.getByText('Open follow-up job')).toBeInTheDocument();
    expect(screen.getByText('Relaunch Follow-up')).toBeInTheDocument();
  });

  it('opens the target from a live follow-up outcome toast', async () => {
    renderWithProviders(
      <NotificationProvider>
        <NotificationStateProbe />
      </NotificationProvider>,
    );

    await waitFor(() => {
      expect(mockCreateNotificationsWebSocket).toHaveBeenCalled();
    });

    act(() => {
      mockSocket.onmessage?.({
        data: JSON.stringify({
          type: 'notification',
          notification: makeFollowUpOutcomeNotification(),
        }),
      });
    });

    const renderToast = ((toast as any).custom as jest.Mock).mock.calls[0]?.[0] as () => React.ReactElement;
    render(renderToast());
    fireEvent.click(screen.getByText('Open target'));

    await waitFor(() => {
      expect(mockMarkNotificationRead).toHaveBeenCalledWith('notif-follow-up-1');
    });
    expect(window.location.assign).toHaveBeenCalledWith(
      '/autonomous-agents?tab=domain&profileId=profile-1&opportunityId=opp-profile-1'
    );
  });

  it('relaunches a failed live follow-up outcome from the custom toast', async () => {
    renderWithProviders(
      <NotificationProvider>
        <NotificationStateProbe />
      </NotificationProvider>,
    );

    await waitFor(() => {
      expect(mockCreateNotificationsWebSocket).toHaveBeenCalled();
    });

    act(() => {
      mockSocket.onmessage?.({
        data: JSON.stringify({
          type: 'notification',
          notification: makeFollowUpOutcomeNotification(),
        }),
      });
    });

    const renderToast = ((toast as any).custom as jest.Mock).mock.calls[0]?.[0] as () => React.ReactElement;
    render(renderToast());
    fireEvent.click(screen.getByText('Relaunch Follow-up'));

    await waitFor(() => {
      expect(mockMarkNotificationRead).toHaveBeenCalledWith('notif-follow-up-1');
    });
    expect(mockRelaunchInboxFollowUp).toHaveBeenCalledWith('inbox-1', {});
    expect(((toast as any).success as jest.Mock)).toHaveBeenCalledWith('Follow-up relaunched');
  });

  it('does not render relaunch for completed live follow-up outcomes', async () => {
    renderWithProviders(
      <NotificationProvider>
        <NotificationStateProbe />
      </NotificationProvider>,
    );

    await waitFor(() => {
      expect(mockCreateNotificationsWebSocket).toHaveBeenCalled();
    });

    act(() => {
      mockSocket.onmessage?.({
        data: JSON.stringify({
          type: 'notification',
          notification: makeFollowUpOutcomeNotification({
            title: 'Follow-up completed: Compiler hotspot',
            data: {
              inbox_item_id: 'inbox-1',
              follow_up_job_id: 'job-follow-up-1',
              follow_up_last_job_id: 'job-follow-up-1',
              follow_up_outcome_status: 'completed',
              follow_up_outcome_summary: 'Validated the hotspot and documented next steps.',
            },
          }),
        }),
      });
    });

    const renderToast = ((toast as any).custom as jest.Mock).mock.calls[0]?.[0] as () => React.ReactElement;
    render(renderToast());

    expect(screen.queryByText('Relaunch Follow-up')).not.toBeInTheDocument();
  });

  it('renders a dedicated custom toast for live reevaluation notifications', async () => {
    renderWithProviders(
      <NotificationProvider>
        <NotificationStateProbe />
      </NotificationProvider>,
    );

    await waitFor(() => {
      expect(mockCreateNotificationsWebSocket).toHaveBeenCalled();
    });

    act(() => {
      mockSocket.onmessage?.({
        data: JSON.stringify({
          type: 'notification',
          notification: makeReevaluationNotification(),
        }),
      });
    });

    await waitFor(() => {
      expect(((toast as any).custom as jest.Mock)).toHaveBeenCalled();
    });
    expect(screen.getByTestId('unread-count')).toHaveTextContent('1');
    expect(screen.getByTestId('notification-count')).toHaveTextContent('1');

    const renderToast = ((toast as any).custom as jest.Mock).mock.calls[0]?.[0] as () => React.ReactElement;
    render(renderToast());

    expect(screen.getByText('Reevaluation ready: Compiler note')).toBeInTheDocument();
    expect(screen.getByText('Open note')).toBeInTheDocument();
    expect(screen.getByText('Open reevaluation job')).toBeInTheDocument();
    expect(screen.getByText('Open originating opportunity')).toBeInTheDocument();
  });

  it('opens the note from a live reevaluation toast', async () => {
    renderWithProviders(
      <NotificationProvider>
        <NotificationStateProbe />
      </NotificationProvider>,
    );

    await waitFor(() => {
      expect(mockCreateNotificationsWebSocket).toHaveBeenCalled();
    });

    act(() => {
      mockSocket.onmessage?.({
        data: JSON.stringify({
          type: 'notification',
          notification: makeReevaluationNotification(),
        }),
      });
    });

    const renderToast = ((toast as any).custom as jest.Mock).mock.calls[0]?.[0] as () => React.ReactElement;
    render(renderToast());
    fireEvent.click(screen.getByText('Open note'));

    await waitFor(() => {
      expect(mockMarkNotificationRead).toHaveBeenCalledWith('notif-reeval-1');
    });
    expect(window.location.assign).toHaveBeenCalledWith('/research-notes?note=note-2');
  });

  it('restarts a failed live reevaluation from the custom toast', async () => {
    renderWithProviders(
      <NotificationProvider>
        <NotificationStateProbe />
      </NotificationProvider>,
    );

    await waitFor(() => {
      expect(mockCreateNotificationsWebSocket).toHaveBeenCalled();
    });

    act(() => {
      mockSocket.onmessage?.({
        data: JSON.stringify({
          type: 'notification',
          notification: makeReevaluationNotification({
            id: 'notif-reeval-2',
            title: 'Reevaluation failed: Compiler note',
            action_url: '/synthesis?job=syn-reeval-2',
            data: {
              note_id: 'note-2',
              note_title: 'Compiler note',
              reevaluation_job_id: 'syn-reeval-2',
              reevaluation_status: 'failed',
              reevaluation_error: 'Model timeout during reevaluation',
            },
          }),
        }),
      });
    });

    const renderToast = ((toast as any).custom as jest.Mock).mock.calls[0]?.[0] as () => React.ReactElement;
    render(renderToast());
    fireEvent.click(screen.getByText('Restart reevaluation'));

    await waitFor(() => {
      expect(mockCreateSynthesisJob).toHaveBeenCalledWith({
        job_type: 'hypothesis_reevaluation',
        title: 'Hypothesis Re-evaluation · Compiler note',
        document_ids: [],
        research_note_id: 'note-2',
        output_format: 'markdown',
        output_style: 'technical',
      });
    });
    expect(((toast as any).success as jest.Mock)).toHaveBeenCalledWith('Hypothesis reevaluation started');
  });

  it('dismisses a reevaluation toast when a refresh shows the notification was resolved', async () => {
    mockGetNotifications
      .mockResolvedValueOnce({
        items: [makeReevaluationNotification()],
        total: 1,
        page: 1,
        page_size: 20,
        unread_count: 1,
      })
      .mockResolvedValueOnce({
        items: [],
        total: 0,
        page: 1,
        page_size: 20,
        unread_count: 0,
      });

    renderWithProviders(
      <NotificationProvider>
        <NotificationStateProbe />
      </NotificationProvider>,
    );

    expect(await screen.findByTestId('notification-count')).toHaveTextContent('1');

    fireEvent.click(screen.getByText('refresh notifications'));

    await waitFor(() => {
      expect(((toast as any).dismiss as jest.Mock)).toHaveBeenCalledWith('notif-reeval-1');
    });
    expect(screen.getByTestId('notification-count')).toHaveTextContent('0');
  });
});
