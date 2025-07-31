import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { deviceApi, modelApi } from '../api';
import { useWebSocket } from '../hooks/useWebSocket';
import { Monitor, Cpu, Activity, Thermometer } from 'lucide-react';

export function Dashboard() {
  const { devices: wsDevices, isConnected } = useWebSocket();
  const { data: devices = [] } = useQuery({
    queryKey: ['devices'],
    queryFn: deviceApi.getDevices,
    refetchInterval: 5000,
  });

  const { data: models = [] } = useQuery({
    queryKey: ['models'],
    queryFn: modelApi.getModels,
  });

  // Use WebSocket devices if available, otherwise fall back to query data
  const displayDevices = wsDevices.length > 0 ? wsDevices : devices;

  const onlineDevices = displayDevices.filter(d => d.status === 'online');
  const totalMemory = displayDevices.reduce((sum, d) => sum + d.total_memory_gb, 0);
  const availableMemory = displayDevices.reduce((sum, d) => sum + d.available_memory_gb, 0);
  const avgCpuUsage = displayDevices.length > 0 
    ? displayDevices.reduce((sum, d) => sum + d.cpu_usage_percent, 0) / displayDevices.length 
    : 0;

  const stats = [
    {
      name: 'Online Devices',
      value: onlineDevices.length,
      total: displayDevices.length,
      icon: Monitor,
      color: 'text-green-600',
    },
    {
      name: 'Available Memory',
      value: `${availableMemory.toFixed(1)} GB`,
      total: `${totalMemory.toFixed(1)} GB`,
      icon: Cpu,
      color: 'text-blue-600',
    },
    {
      name: 'Average CPU Usage',
      value: `${avgCpuUsage.toFixed(1)}%`,
      total: '100%',
      icon: Activity,
      color: 'text-orange-600',
    },
    {
      name: 'Available Models',
      value: models.length,
      total: models.length,
      icon: Thermometer,
      color: 'text-purple-600',
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-600">
          Overview of your distributed LLM platform
        </p>
      </div>

      {/* Connection Status */}
      <div className="rounded-lg bg-white p-6 shadow-sm border">
        <div className="flex items-center">
          <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="ml-2 text-sm font-medium text-gray-900">
            {isConnected ? 'Connected to backend' : 'Disconnected from backend'}
          </span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <div
            key={stat.name}
            className="relative overflow-hidden rounded-lg bg-white px-4 py-5 shadow-sm border sm:px-6"
          >
            <div>
              <dt className="truncate text-sm font-medium text-gray-500">
                {stat.name}
              </dt>
              <dd className="mt-1 text-3xl font-semibold tracking-tight text-gray-900">
                {stat.value}
                {stat.total && stat.value !== stat.total && (
                  <span className="text-sm text-gray-500 ml-1">/ {stat.total}</span>
                )}
              </dd>
            </div>
            <div className="absolute inset-y-0 right-0 flex items-center pr-4">
              <stat.icon className={`h-8 w-8 ${stat.color}`} />
            </div>
          </div>
        ))}
      </div>

      {/* Device Overview */}
      <div className="rounded-lg bg-white shadow-sm border">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">Device Status</h2>
        </div>
        <div className="px-6 py-4">
          {displayDevices.length === 0 ? (
            <p className="text-gray-500 text-center py-8">
              No devices connected. Start a device agent to see devices here.
            </p>
          ) : (
            <div className="space-y-4">
              {displayDevices.map((device) => (
                <div
                  key={device.id}
                  className="flex items-center justify-between py-3 border-b border-gray-100 last:border-b-0"
                >
                  <div className="flex items-center space-x-3">
                    <div className={`h-2 w-2 rounded-full ${
                      device.status === 'online' ? 'bg-green-500' :
                      device.status === 'busy' ? 'bg-yellow-500' : 'bg-red-500'
                    }`} />
                    <div>
                      <p className="text-sm font-medium text-gray-900">{device.name}</p>
                      <p className="text-xs text-gray-500">{device.type.toUpperCase()}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-900">
                      {device.available_memory_gb.toFixed(1)} GB / {device.total_memory_gb.toFixed(1)} GB
                    </p>
                    <p className="text-xs text-gray-500">
                      CPU: {device.cpu_usage_percent.toFixed(1)}%
                      {device.temperature_celsius && ` • ${device.temperature_celsius}°C`}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 