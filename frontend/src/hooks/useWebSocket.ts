/**
 * Custom hook for WebSocket connections
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { WebSocketMessage } from '../types';

interface UseWebSocketOptions {
  onMessage?: (data: WebSocketMessage) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

export const useWebSocket = (url: string | null, options: UseWebSocketOptions = {}) => {
  const {
    onMessage,
    onOpen,
    onClose,
    onError,
    reconnectAttempts = 3,
    reconnectInterval = 3000,
  } = options;

  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const attemptCountRef = useRef(0);
  const intentionalDisconnectRef = useRef(false);
  const callbacksRef = useRef({ onMessage, onOpen, onClose, onError });
  callbacksRef.current = { onMessage, onOpen, onClose, onError };

  const connect = useCallback(() => {
    if (!url) return;

    try {
      intentionalDisconnectRef.current = false;
      setConnectionStatus('connecting');
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionStatus('connected');
        attemptCountRef.current = 0;
        callbacksRef.current.onOpen?.();
      };

      ws.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(data);
          callbacksRef.current.onMessage?.(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        if (wsRef.current === ws) {
          wsRef.current = null;
        }
        setConnectionStatus('disconnected');
        callbacksRef.current.onClose?.();

        // Attempt to reconnect
        if (!intentionalDisconnectRef.current && attemptCountRef.current < reconnectAttempts) {
          attemptCountRef.current += 1;
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (error) => {
        setConnectionStatus('error');
        callbacksRef.current.onError?.(error);
      };
    } catch (error) {
      setConnectionStatus('error');
      console.error('WebSocket connection error:', error);
    }
  }, [reconnectAttempts, reconnectInterval, url]);

  const disconnect = useCallback(() => {
    intentionalDisconnectRef.current = true;
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionStatus('disconnected');
  }, []);

  const sendMessage = useCallback((message: Record<string, unknown>): boolean => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        return false;
      }
    }
    return false;
  }, []);

  useEffect(() => {
    if (url) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [connect, disconnect, url]);

  return {
    connectionStatus,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
  };
};

