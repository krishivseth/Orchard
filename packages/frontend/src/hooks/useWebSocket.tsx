import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { DeviceInfo, DeviceHealthMetrics, ChatMessage } from '../types';

interface WebSocketContextType {
  isConnected: boolean;
  devices: DeviceInfo[];
  lastMessage: any;
  sendMessage: (message: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

interface WebSocketProviderProps {
  children: ReactNode;
}

export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [lastMessage, setLastMessage] = useState<any>(null);

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // In development, frontend runs on 3000 but backend on 8000
    const backendHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
      ? `${window.location.hostname}:8000` 
      : window.location.host;
    const wsUrl = `${protocol}//${backendHost}/ws`;
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);
        
        // Handle different message types
        switch (data.type) {
          case 'device_update':
            setDevices(prev => {
              const updated = [...prev];
              const index = updated.findIndex(d => d.id === data.device.id);
              if (index >= 0) {
                updated[index] = data.device;
              } else {
                updated.push(data.device);
              }
              return updated;
            });
            break;
          case 'device_metrics':
            // Handle metrics update
            break;
          case 'new_message':
            // Handle new chat message
            break;
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setSocket(ws);

    return () => {
      ws.close();
    };
  }, []);

  const sendMessage = (message: any) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify(message));
    }
  };

  const value = {
    isConnected,
    devices,
    lastMessage,
    sendMessage,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
} 