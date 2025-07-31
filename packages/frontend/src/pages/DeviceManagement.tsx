import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { deviceApi } from '../api';
import { useWebSocket } from '../hooks/useWebSocket';
import { Monitor, Cpu, Thermometer, Activity, WifiOff } from 'lucide-react';
import { DeviceInfo } from '../types';

function DeviceCard({ device }: { device: DeviceInfo }) {
  const memoryUsage = ((device.total_memory_gb - device.available_memory_gb) / device.total_memory_gb) * 100;
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'busy': return 'bg-yellow-500';
      case 'offline': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'mac': return '🖥️';
      case 'iphone': return '📱';
      case 'ipad': return '📱';
      default: return '💻';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <div className="flex items-start justify-between">
        <div className="flex items-center space-x-3">
          <div className="text-2xl">{getDeviceIcon(device.type)}</div>
          <div>
            <h3 className="text-lg font-medium text-gray-900">{device.name}</h3>
            <p className="text-sm text-gray-500">{device.type.toUpperCase()}</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`h-2 w-2 rounded-full ${getStatusColor(device.status)}`} />
          <span className="text-sm text-gray-600 capitalize">{device.status}</span>
        </div>
      </div>

      <div className="mt-4 space-y-3">
        {/* Memory Usage */}
        <div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Memory Usage</span>
            <span className="text-gray-900">
              {(device.total_memory_gb - device.available_memory_gb).toFixed(1)} GB / 
              {device.total_memory_gb.toFixed(1)} GB
            </span>
          </div>
          <div className="mt-1 bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full"
              style={{ width: `${memoryUsage}%` }}
            />
          </div>
        </div>

        {/* CPU Usage */}
        <div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">CPU Usage</span>
            <span className="text-gray-900">{device.cpu_usage_percent.toFixed(1)}%</span>
          </div>
          <div className="mt-1 bg-gray-200 rounded-full h-2">
            <div
              className="bg-green-600 h-2 rounded-full"
              style={{ width: `${device.cpu_usage_percent}%` }}
            />
          </div>
        </div>

        {/* Temperature */}
        {device.temperature_celsius && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600 flex items-center">
              <Thermometer className="h-4 w-4 mr-1" />
              Temperature
            </span>
            <span className="text-gray-900">{device.temperature_celsius}°C</span>
          </div>
        )}

        {/* Network Info */}
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Network</span>
          <span className="text-gray-900">{device.ip_address}:{device.port}</span>
        </div>

        {/* Current Model */}
        {device.current_model && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Current Model</span>
            <span className="text-gray-900 font-medium">{device.current_model}</span>
          </div>
        )}

        {/* Last Heartbeat */}
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Last Seen</span>
          <span className="text-gray-900">
            {new Date(device.last_heartbeat).toLocaleTimeString()}
          </span>
        </div>
      </div>
    </div>
  );
}

export function DeviceManagement() {
  const { devices: wsDevices, isConnected } = useWebSocket();
  const { data: devices = [], refetch } = useQuery({
    queryKey: ['devices'],
    queryFn: deviceApi.getDevices,
    refetchInterval: 5000,
  });

  // Use WebSocket devices if available, otherwise fall back to query data
  const displayDevices = wsDevices.length > 0 ? wsDevices : devices;

  const onlineDevices = displayDevices.filter(d => d.status === 'online');
  const offlineDevices = displayDevices.filter(d => d.status === 'offline');
  const busyDevices = displayDevices.filter(d => d.status === 'busy');

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Device Management</h1>
          <p className="mt-1 text-sm text-gray-600">
            Monitor and manage your connected Apple devices
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="btn btn-secondary"
        >
          Refresh
        </button>
      </div>

      {/* Device Stats */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center">
            <Monitor className="h-8 w-8 text-green-600" />
            <div className="ml-4">
              <p className="text-2xl font-semibold text-gray-900">{onlineDevices.length}</p>
              <p className="text-sm text-gray-600">Online Devices</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center">
            <Activity className="h-8 w-8 text-yellow-600" />
            <div className="ml-4">
              <p className="text-2xl font-semibold text-gray-900">{busyDevices.length}</p>
              <p className="text-sm text-gray-600">Busy Devices</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center">
            <WifiOff className="h-8 w-8 text-red-600" />
            <div className="ml-4">
              <p className="text-2xl font-semibold text-gray-900">{offlineDevices.length}</p>
              <p className="text-sm text-gray-600">Offline Devices</p>
            </div>
          </div>
        </div>
      </div>

      {/* Device Grid */}
      {displayDevices.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm border p-12 text-center">
          <Cpu className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 text-lg font-medium text-gray-900">No devices connected</h3>
          <p className="mt-2 text-sm text-gray-600">
            Start a device agent on your Apple devices to see them here.
          </p>
          <div className="mt-4">
            <code className="text-sm bg-gray-100 px-2 py-1 rounded">
              python packages/device-agent/agent.py --port 8001
            </code>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {displayDevices.map((device) => (
            <DeviceCard key={device.id} device={device} />
          ))}
        </div>
      )}
    </div>
  );
} 