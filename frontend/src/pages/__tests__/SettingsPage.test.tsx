import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import SettingsPage from '../SettingsPage';
import { NotificationPreferences } from '../../types';

const mockUpdatePreferences = jest.fn().mockResolvedValue(undefined);

jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useSearchParams: () => [new URLSearchParams('tab=notifications')],
}));

const preferences: NotificationPreferences = {
  id: 'prefs-1',
  user_id: 'user-1',
  notify_document_processing: true,
  notify_document_errors: true,
  notify_sync_complete: true,
  notify_ingestion_complete: true,
  notify_transcription_complete: true,
  notify_summarization_complete: true,
  notify_research_note_citation_issues: true,
  notify_experiment_run_updates: true,
  notify_hypothesis_reevaluation_updates: true,
  notify_queue_urgency_alerts: true,
  notify_follow_up_outcome_alerts: true,
  notify_policy_guardrail_alerts: true,
  notify_autonomy_budget_alerts: true,
  notify_customer_autonomy_budget_alerts: true,
  research_note_citation_coverage_threshold: 0.7,
  research_note_citation_notify_cooldown_hours: 12,
  queue_urgency_alert_reminder_cooldown_hours: 6,
  research_note_citation_notify_on_unknown_keys: true,
  research_note_citation_notify_on_low_coverage: true,
  research_note_citation_notify_on_missing_bibliography: true,
  notify_maintenance: true,
  notify_quota_warnings: true,
  notify_admin_broadcasts: true,
  notify_mentions: true,
  notify_shares: true,
  notify_comments: true,
  play_sound: false,
  show_desktop_notification: false,
  created_at: '2026-03-12T12:00:00Z',
  updated_at: '2026-03-12T12:00:00Z',
};

jest.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({
    user: {
      id: 'user-1',
      username: 'alex',
      email: 'alex@example.com',
      full_name: 'Alex Example',
      role: 'user',
      is_active: true,
      is_verified: true,
      created_at: '2026-03-01T12:00:00Z',
    },
    updateUser: jest.fn(),
  }),
}));

jest.mock('../../contexts/NotificationContext', () => ({
  useNotifications: () => ({
    preferences,
    updatePreferences: (...args: any[]) => mockUpdatePreferences(...args),
    isLoading: false,
    notifications: [],
    unreadCount: 0,
    fetchNotifications: jest.fn(),
    markAsRead: jest.fn(),
    markAllAsRead: jest.fn(),
    dismissNotification: jest.fn(),
    refreshUnreadCount: jest.fn(),
  }),
}));

jest.mock('../../services/api', () => ({
  apiClient: {},
}));

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

describe('SettingsPage notifications tab', () => {
  beforeEach(() => {
    mockUpdatePreferences.mockClear();
  });

  it('saves the experiment run notification preference', async () => {
    render(
      <MemoryRouter
        initialEntries={['/settings?tab=notifications']}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByText('Notification Preferences')).toBeInTheDocument();

    const experimentRunsCheckbox = screen.getByRole('checkbox', { name: /experiment runs/i });
    expect(experimentRunsCheckbox).toBeChecked();

    fireEvent.click(experimentRunsCheckbox);
    fireEvent.click(screen.getByRole('button', { name: /save preferences/i }));

    await waitFor(() => {
      expect(mockUpdatePreferences).toHaveBeenCalledTimes(1);
    });

    expect(mockUpdatePreferences).toHaveBeenCalledWith(
      expect.objectContaining({
        notify_experiment_run_updates: false,
      }),
    );
  });
});
