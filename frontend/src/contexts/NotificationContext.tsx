/**
 * Context for managing notifications state and real-time updates
 */

import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { useQueryClient } from 'react-query';
import { apiClient } from '../services/api';
import { Notification, NotificationPreferences } from '../types';
import { useAuth } from './AuthContext';
import toast from 'react-hot-toast';
import { buildNotificationToastSummary } from '../utils/notificationSummary';

interface NotificationContextValue {
  notifications: Notification[];
  unreadCount: number;
  isLoading: boolean;
  preferences: NotificationPreferences | null;
  fetchNotifications: (page?: number) => Promise<void>;
  markAsRead: (notificationId: string) => Promise<void>;
  markAllAsRead: () => Promise<void>;
  dismissNotification: (notificationId: string) => Promise<void>;
  updatePreferences: (prefs: Partial<NotificationPreferences>) => Promise<void>;
  refreshUnreadCount: () => Promise<void>;
}

const NotificationContext = createContext<NotificationContextValue | null>(null);

const isActionableFollowUpOutcome = (notification: Notification): boolean =>
  notification.notification_type === 'follow_up_outcome_alert';

const getFollowUpToastJobId = (notification: Notification): string =>
  String((notification.data as any)?.follow_up_last_job_id || (notification.data as any)?.follow_up_job_id || '').trim();

const getFollowUpToastInboxItemId = (notification: Notification): string =>
  String((notification.data as any)?.inbox_item_id || '').trim();

const getFollowUpToastOutcomeStatus = (notification: Notification): string =>
  String((notification.data as any)?.follow_up_outcome_status || '').trim().toLowerCase();

const canRelaunchFollowUpFromToast = (notification: Notification): boolean =>
  Boolean(getFollowUpToastInboxItemId(notification)) && ['failed', 'cancelled'].includes(getFollowUpToastOutcomeStatus(notification));

const isActionableReevaluationUpdate = (notification: Notification): boolean =>
  notification.notification_type === 'hypothesis_reevaluation_update';

const getReevaluationToastNoteId = (notification: Notification): string =>
  String((notification.data as any)?.note_id || '').trim();

const getReevaluationToastJobId = (notification: Notification): string =>
  String((notification.data as any)?.reevaluation_job_id || '').trim();

const getReevaluationToastStatus = (notification: Notification): string =>
  String((notification.data as any)?.reevaluation_status || '').trim().toLowerCase();

const getReevaluationToastOriginActionUrl = (notification: Notification): string =>
  String((notification.data as any)?.origin_action_url || '').trim();

const getReevaluationToastNoteTitle = (notification: Notification): string =>
  String((notification.data as any)?.note_title || '').trim();

const canRestartReevaluationFromToast = (notification: Notification): boolean =>
  Boolean(getReevaluationToastNoteId(notification)) && ['failed', 'stale'].includes(getReevaluationToastStatus(notification));

const navigateToUrl = (url: string) => {
  if (!url) return;
  window.location.assign(url);
};

const FollowUpOutcomeToast: React.FC<{
  notification: Notification;
  onOpenTarget: () => Promise<void>;
  onOpenJob: () => Promise<void>;
  onRelaunch: () => Promise<void>;
}> = ({ notification, onOpenTarget, onOpenJob, onRelaunch }) => {
  const [isRelaunching, setIsRelaunching] = useState(false);
  const toastSummary = buildNotificationToastSummary(notification);
  const hasActionUrl = Boolean(String(notification.action_url || '').trim());
  const hasFollowUpJob = Boolean(getFollowUpToastJobId(notification));
  const canRelaunch = canRelaunchFollowUpFromToast(notification);

  return (
    <div className="pointer-events-auto w-full max-w-md rounded-lg border border-emerald-200 bg-white p-3 shadow-lg">
      <div className="text-sm font-semibold text-gray-900">{toastSummary.title}</div>
      {toastSummary.description ? (
        <div className="mt-1 whitespace-pre-line text-xs text-gray-600">{toastSummary.description}</div>
      ) : null}
      {(hasActionUrl || hasFollowUpJob || canRelaunch) ? (
        <div className="mt-3 flex flex-wrap gap-2">
          {hasActionUrl ? (
            <button
              type="button"
              className="inline-flex items-center rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-[11px] font-medium text-emerald-700 hover:bg-emerald-100"
              onClick={() => {
                void onOpenTarget();
              }}
            >
              Open target
            </button>
          ) : null}
          {hasFollowUpJob ? (
            <button
              type="button"
              className="inline-flex items-center rounded-md border border-sky-200 bg-sky-50 px-2 py-1 text-[11px] font-medium text-sky-700 hover:bg-sky-100"
              onClick={() => {
                void onOpenJob();
              }}
            >
              Open follow-up job
            </button>
          ) : null}
          {canRelaunch ? (
            <button
              type="button"
              disabled={isRelaunching}
              className="inline-flex items-center rounded-md border border-indigo-200 bg-indigo-50 px-2 py-1 text-[11px] font-medium text-indigo-700 hover:bg-indigo-100 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={async () => {
                if (isRelaunching) return;
                setIsRelaunching(true);
                try {
                  await onRelaunch();
                } finally {
                  setIsRelaunching(false);
                }
              }}
            >
              {isRelaunching ? 'Relaunching...' : 'Relaunch Follow-up'}
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
};

const ReevaluationToast: React.FC<{
  notification: Notification;
  onOpenNote: () => Promise<void>;
  onOpenJob: () => Promise<void>;
  onOpenOrigin: () => Promise<void>;
  onRestart: () => Promise<void>;
}> = ({ notification, onOpenNote, onOpenJob, onOpenOrigin, onRestart }) => {
  const [isRestarting, setIsRestarting] = useState(false);
  const toastSummary = buildNotificationToastSummary(notification);
  const noteActionUrl = String(notification.action_url || '').includes('/research-notes?')
    ? String(notification.action_url || '').trim()
    : '';
  const hasJob = Boolean(getReevaluationToastJobId(notification));
  const hasOrigin = Boolean(getReevaluationToastOriginActionUrl(notification));
  const canRestart = canRestartReevaluationFromToast(notification);

  return (
    <div className="pointer-events-auto w-full max-w-md rounded-lg border border-indigo-200 bg-white p-3 shadow-lg">
      <div className="text-sm font-semibold text-gray-900">{toastSummary.title}</div>
      {toastSummary.description ? (
        <div className="mt-1 whitespace-pre-line text-xs text-gray-600">{toastSummary.description}</div>
      ) : null}
      {(noteActionUrl || hasJob || hasOrigin || canRestart) ? (
        <div className="mt-3 flex flex-wrap gap-2">
          {noteActionUrl ? (
            <button
              type="button"
              className="inline-flex items-center rounded-md border border-indigo-200 bg-indigo-50 px-2 py-1 text-[11px] font-medium text-indigo-700 hover:bg-indigo-100"
              onClick={() => {
                void onOpenNote();
              }}
            >
              Open note
            </button>
          ) : null}
          {hasJob ? (
            <button
              type="button"
              className="inline-flex items-center rounded-md border border-sky-200 bg-sky-50 px-2 py-1 text-[11px] font-medium text-sky-700 hover:bg-sky-100"
              onClick={() => {
                void onOpenJob();
              }}
            >
              Open reevaluation job
            </button>
          ) : null}
          {hasOrigin ? (
            <button
              type="button"
              className="inline-flex items-center rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-[11px] font-medium text-emerald-700 hover:bg-emerald-100"
              onClick={() => {
                void onOpenOrigin();
              }}
            >
              Open originating opportunity
            </button>
          ) : null}
          {canRestart ? (
            <button
              type="button"
              disabled={isRestarting}
              className="inline-flex items-center rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-[11px] font-medium text-amber-700 hover:bg-amber-100 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={async () => {
                if (isRestarting) return;
                setIsRestarting(true);
                try {
                  await onRestart();
                } finally {
                  setIsRestarting(false);
                }
              }}
            >
              {isRestarting ? 'Restarting...' : 'Restart reevaluation'}
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
};

export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [preferences, setPreferences] = useState<NotificationPreferences | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isConnectingRef = useRef(false);

  const isAuthenticated = !!user;

  const invalidateReevaluationTargets = useCallback(async (notification: Notification) => {
    const noteId = getReevaluationToastNoteId(notification);
    const originSourceKind = String((notification.data as any)?.origin_source_kind || '').trim().toLowerCase();
    const originSourceId = String((notification.data as any)?.origin_source_id || '').trim();
    const reevaluationJobId = getReevaluationToastJobId(notification);
    queryClient.invalidateQueries(['notifications']);
    queryClient.invalidateQueries(['notifications-unread-count']);
    if (noteId) {
      queryClient.invalidateQueries(['research-notes']);
      queryClient.invalidateQueries(['experiment-plans', noteId]);
    }
    if (reevaluationJobId) {
      queryClient.invalidateQueries(['synthesis-jobs']);
    }
    if (originSourceKind === 'profile' && originSourceId) {
      queryClient.invalidateQueries(['domain-research-profiles']);
    }
    if (originSourceKind === 'portfolio' && originSourceId) {
      queryClient.invalidateQueries(['research-portfolios']);
    }
  }, [queryClient]);

  const fetchNotifications = useCallback(async (page: number = 1) => {
    if (!isAuthenticated) return;
    setIsLoading(true);
    try {
      const response = await apiClient.getNotifications({ page, page_size: 20 });
      if (page === 1) {
        setNotifications(prev => {
          const nextItems = response.items;
          const nextIds = new Set(nextItems.map((item) => String(item.id || '').trim()));
          const resolvedReevaluationToastIds = prev
            .filter((item) => item.notification_type === 'hypothesis_reevaluation_update')
            .map((item) => String(item.id || '').trim())
            .filter((id) => id && !nextIds.has(id));
          if (typeof (toast as any).dismiss === 'function') {
            resolvedReevaluationToastIds.forEach((id) => (toast as any).dismiss(id));
          }
          return nextItems;
        });
      } else {
        setNotifications(prev => [...prev, ...response.items]);
      }
      setUnreadCount(response.unread_count);
    } catch (error) {
      console.error('Failed to fetch notifications:', error);
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated]);

  const refreshUnreadCount = useCallback(async () => {
    if (!isAuthenticated) return;
    try {
      const { unread_count } = await apiClient.getUnreadCount();
      setUnreadCount(unread_count);
    } catch (error) {
      console.error('Failed to refresh unread count:', error);
    }
  }, [isAuthenticated]);

  const markAsRead = useCallback(async (notificationId: string) => {
    try {
      await apiClient.markNotificationRead(notificationId);
      setNotifications(prev =>
        prev.map(n => n.id === notificationId ? { ...n, is_read: true } : n)
      );
      setUnreadCount(prev => Math.max(0, prev - 1));
    } catch (error) {
      console.error('Failed to mark notification as read:', error);
    }
  }, []);

  const markAllAsRead = useCallback(async () => {
    try {
      await apiClient.markAllNotificationsRead();
      setNotifications(prev => prev.map(n => ({ ...n, is_read: true })));
      setUnreadCount(0);
    } catch (error) {
      console.error('Failed to mark all notifications as read:', error);
    }
  }, []);

  const dismissNotification = useCallback(async (notificationId: string) => {
    try {
      await apiClient.dismissNotification(notificationId);
      const notification = notifications.find(n => n.id === notificationId);
      setNotifications(prev => prev.filter(n => n.id !== notificationId));
      if (notification && !notification.is_read) {
        setUnreadCount(prev => Math.max(0, prev - 1));
      }
    } catch (error) {
      console.error('Failed to dismiss notification:', error);
    }
  }, [notifications]);

  const updatePreferences = useCallback(async (prefs: Partial<NotificationPreferences>) => {
    try {
      const updated = await apiClient.updateNotificationPreferences(prefs);
      setPreferences(updated);
      toast.success('Notification preferences updated');
    } catch (error) {
      console.error('Failed to update preferences:', error);
      toast.error('Failed to update preferences');
    }
  }, []);

  // Set up WebSocket connection for real-time notifications
  const connectWebSocket = useCallback(() => {
    if (!isAuthenticated || isConnectingRef.current || wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    isConnectingRef.current = true;

    try {
      const ws = apiClient.createNotificationsWebSocket();
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('Notification WebSocket connected');
        isConnectingRef.current = false;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'notification') {
            const newNotification = data.notification as Notification;
            setNotifications(prev => [newNotification, ...prev]);
            setUnreadCount(prev => prev + 1);

            if (isActionableFollowUpOutcome(newNotification)) {
              const notificationId = String(newNotification.id || '').trim();
              const actionUrl = String(newNotification.action_url || '').trim();
              const followUpJobId = getFollowUpToastJobId(newNotification);
              const inboxItemId = getFollowUpToastInboxItemId(newNotification);

              const markNotificationRead = async () => {
                if (!newNotification.is_read && notificationId) {
                  try {
                    await apiClient.markNotificationRead(notificationId);
                    setNotifications(prev =>
                      prev.map(n => n.id === notificationId ? { ...n, is_read: true } : n)
                    );
                    setUnreadCount(prev => Math.max(0, prev - 1));
                  } catch (error) {
                    console.error('Failed to mark notification as read:', error);
                  }
                }
              };

              const dismissLiveToast = () => {
                if (typeof (toast as any).dismiss === 'function') {
                  (toast as any).dismiss(notificationId);
                }
              };

              if (typeof (toast as any).custom === 'function') {
                (toast as any).custom(
                  () => (
                    <FollowUpOutcomeToast
                      notification={newNotification}
                      onOpenTarget={async () => {
                        await markNotificationRead();
                        if (actionUrl) {
                          dismissLiveToast();
                          navigateToUrl(actionUrl);
                        }
                      }}
                      onOpenJob={async () => {
                        await markNotificationRead();
                        if (followUpJobId) {
                          dismissLiveToast();
                          navigateToUrl(`/autonomous-agents?job=${encodeURIComponent(followUpJobId)}`);
                        }
                      }}
                      onRelaunch={async () => {
                        if (!inboxItemId) return;
                        await markNotificationRead();
                        await apiClient.relaunchInboxFollowUp(inboxItemId, {});
                        await fetchNotifications();
                        dismissLiveToast();
                        toast.success('Follow-up relaunched');
                      }}
                    />
                  ),
                  {
                    id: notificationId || undefined,
                    duration: 7000,
                  },
                );
              }
            } else if (isActionableReevaluationUpdate(newNotification)) {
              const notificationId = String(newNotification.id || '').trim();
              const noteId = getReevaluationToastNoteId(newNotification);
              const noteActionUrl = String(newNotification.action_url || '').includes('/research-notes?')
                ? String(newNotification.action_url || '').trim()
                : '';
              const reevaluationJobId = getReevaluationToastJobId(newNotification);
              const originActionUrl = getReevaluationToastOriginActionUrl(newNotification);
              const noteTitle = getReevaluationToastNoteTitle(newNotification);

              const markNotificationRead = async () => {
                if (!newNotification.is_read && notificationId) {
                  try {
                    await apiClient.markNotificationRead(notificationId);
                    setNotifications(prev =>
                      prev.map(n => n.id === notificationId ? { ...n, is_read: true } : n)
                    );
                    setUnreadCount(prev => Math.max(0, prev - 1));
                  } catch (error) {
                    console.error('Failed to mark notification as read:', error);
                  }
                }
              };

              const dismissLiveToast = () => {
                if (typeof (toast as any).dismiss === 'function') {
                  (toast as any).dismiss(notificationId);
                }
              };

              if (typeof (toast as any).custom === 'function') {
                (toast as any).custom(
                  () => (
                    <ReevaluationToast
                      notification={newNotification}
                      onOpenNote={async () => {
                        await markNotificationRead();
                        await invalidateReevaluationTargets(newNotification);
                        if (noteActionUrl) {
                          dismissLiveToast();
                          navigateToUrl(noteActionUrl);
                        }
                      }}
                      onOpenJob={async () => {
                        await markNotificationRead();
                        await invalidateReevaluationTargets(newNotification);
                        if (reevaluationJobId) {
                          dismissLiveToast();
                          navigateToUrl(`/synthesis?job=${encodeURIComponent(reevaluationJobId)}`);
                        }
                      }}
                      onOpenOrigin={async () => {
                        await markNotificationRead();
                        await invalidateReevaluationTargets(newNotification);
                        if (originActionUrl) {
                          dismissLiveToast();
                          navigateToUrl(originActionUrl);
                        }
                      }}
                      onRestart={async () => {
                        if (!noteId) return;
                        await markNotificationRead();
                        const job = await apiClient.createSynthesisJob({
                          job_type: 'hypothesis_reevaluation',
                          title: `Hypothesis Re-evaluation · ${noteTitle || noteId}`.slice(0, 500),
                          document_ids: [],
                          research_note_id: noteId,
                          output_format: 'markdown',
                          output_style: 'technical',
                        });
                        await fetchNotifications();
                        await invalidateReevaluationTargets(newNotification);
                        dismissLiveToast();
                        toast.success('Hypothesis reevaluation started');
                        if (job?.id) {
                          navigateToUrl(`/synthesis?job=${encodeURIComponent(job.id)}`);
                        }
                      }}
                    />
                  ),
                  {
                    id: notificationId || undefined,
                    duration: 7000,
                  },
                );
              }
            } else if (newNotification.priority === 'high' || newNotification.priority === 'urgent') {
              // Show toast for high priority notifications
              const toastSummary = buildNotificationToastSummary(newNotification);
              toast(`${toastSummary.title}\n${toastSummary.description}`.trim(), {
                icon: newNotification.priority === 'urgent' ? '!' : 'i',
                duration: 5000,
              });
            }

            // Play sound if enabled
            if (preferences?.play_sound) {
              try {
                const audio = new Audio('/notification.mp3');
                audio.volume = 0.5;
                audio.play().catch(() => {});
              } catch (e) {
                // Sound not available
              }
            }
          }
        } catch (error) {
          console.error('Failed to parse notification:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('Notification WebSocket closed:', event.code, event.reason);
        isConnectingRef.current = false;
        wsRef.current = null;

        // Reconnect after delay if still authenticated
        if (isAuthenticated && event.code !== 1000) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, 5000);
        }
      };

      ws.onerror = (error) => {
        console.error('Notification WebSocket error:', error);
        isConnectingRef.current = false;
      };
    } catch (error) {
      console.error('Failed to connect notification WebSocket:', error);
      isConnectingRef.current = false;
    }
  }, [fetchNotifications, invalidateReevaluationTargets, isAuthenticated, preferences?.play_sound]);

  // Connect WebSocket when authenticated
  useEffect(() => {
    if (isAuthenticated) {
      connectWebSocket();
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting');
        wsRef.current = null;
      }
    };
  }, [isAuthenticated, connectWebSocket]);

  // Fetch initial data when authenticated
  useEffect(() => {
    if (isAuthenticated) {
      fetchNotifications();
      apiClient.getNotificationPreferences()
        .then(setPreferences)
        .catch(console.error);
    } else {
      // Clear state when logged out
      setNotifications([]);
      setUnreadCount(0);
      setPreferences(null);
    }
  }, [isAuthenticated, fetchNotifications]);

  return (
    <NotificationContext.Provider
      value={{
        notifications,
        unreadCount,
        isLoading,
        preferences,
        fetchNotifications,
        markAsRead,
        markAllAsRead,
        dismissNotification,
        updatePreferences,
        refreshUnreadCount,
      }}
    >
      {children}
    </NotificationContext.Provider>
  );
};

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within NotificationProvider');
  }
  return context;
};
