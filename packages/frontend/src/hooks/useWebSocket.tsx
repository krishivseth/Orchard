import { createContext, useContext, useEffect, useMemo, useRef, useState, ReactNode } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { backendBaseUrl } from '../api';
import { DeviceInfo, WebSocketEvent } from '../types';

interface WebSocketContextType {
  isConnected: boolean;
  devices: DeviceInfo[];
  sendMessage: (message: unknown) => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

interface WebSocketProviderProps {
  children: ReactNode;
}

const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30000;

function resolveWsUrl(): string {
  if (window.electronAPI) {
    // In Electron, use the backend URL directly
    return backendBaseUrl.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:') + '/ws';
  }
  // In web browser, use standard logic
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const backendHost =
    window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
      ? `${window.location.hostname}:8000`
      : window.location.host;
  return `${protocol}//${backendHost}/ws`;
}

export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const queryClient = useQueryClient();

  useEffect(() => {
    let cancelled = false;
    let backoff = INITIAL_BACKOFF_MS;
    const wsUrl = resolveWsUrl();

    const handleEvent = (data: WebSocketEvent) => {
      switch (data.type) {
        case 'device_update':
          setDevices((prev) => {
            const index = prev.findIndex((d) => d.id === data.device.id);
            if (index === -1) return [...prev, data.device];
            const updated = [...prev];
            updated[index] = data.device;
            return updated;
          });
          break;
        case 'device_removed':
          setDevices((prev) => prev.filter((d) => d.id !== data.device_id));
          break;
        case 'new_message':
          queryClient.invalidateQueries({ queryKey: ['chat-history'] });
          break;
      }
    };

    const connect = () => {
      if (cancelled) return;

      const ws = new WebSocket(wsUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        backoff = INITIAL_BACKOFF_MS;
        setIsConnected(true);
        console.log('WebSocket connected');
      };

      ws.onmessage = (event) => {
        try {
          handleEvent(JSON.parse(event.data) as WebSocketEvent);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.onclose = () => {
        setIsConnected(false);
        if (socketRef.current === ws) socketRef.current = null;
        if (cancelled) return;
        console.log(`WebSocket disconnected, reconnecting in ${backoff}ms`);
        reconnectTimerRef.current = setTimeout(connect, backoff);
        backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [queryClient]);

  const value = useMemo<WebSocketContextType>(
    () => ({
      isConnected,
      devices,
      sendMessage: (message: unknown) => {
        const socket = socketRef.current;
        if (socket && socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify(message));
        }
      },
    }),
    [isConnected, devices]
  );

  return <WebSocketContext.Provider value={value}>{children}</WebSocketContext.Provider>;
}

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
}
