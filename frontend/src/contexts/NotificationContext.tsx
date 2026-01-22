/**
 * Context for managing notifications state and real-time updates
 */

import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { apiClient } from '../services/api';
import { Notification, NotificationPreferences } from '../types';
import { useAuth } from './AuthContext';
import toast from 'react-hot-toast';

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

export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user } = useAuth();
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [preferences, setPreferences] = useState<NotificationPreferences | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isConnectingRef = useRef(false);

  const isAuthenticated = !!user;

  const fetchNotifications = useCallback(async (page: number = 1) => {
    if (!isAuthenticated) return;
    setIsLoading(true);
    try {
      const response = await apiClient.getNotifications({ page, page_size: 20 });
      if (page === 1) {
        setNotifications(response.items);
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

            // Show toast for high priority notifications
            if (newNotification.priority === 'high' || newNotification.priority === 'urgent') {
              toast(newNotification.title, {
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
  }, [isAuthenticated, preferences?.play_sound]);

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
