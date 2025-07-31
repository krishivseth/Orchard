import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelApi, deviceApi } from '../api';
import { useWebSocket } from '../hooks/useWebSocket';
import { Brain, Download, Play, CheckCircle, XCircle, Loader } from 'lucide-react';
import { LLMModel, DeviceInfo } from '../types';

function ModelCard({ 
  model, 
  devices, 
  onDeploy 
}: { 
  model: LLMModel; 
  devices: DeviceInfo[];
  onDeploy: (modelId: string, deviceIds: string[]) => void;
}) {
  const [selectedDevices, setSelectedDevices] = useState<string[]>([]);
  const [showDeployment, setShowDeployment] = useState(false);

  const compatibleDevices = devices.filter(device => 
    model.supported_devices.includes(device.type) && 
    device.available_memory_gb >= model.min_memory_gb &&
    device.status === 'online'
  );

  const devicesRunningModel = devices.filter(device => device.current_model === model.id);

  const handleDeploy = () => {
    if (selectedDevices.length > 0) {
      onDeploy(model.id, selectedDevices);
      setSelectedDevices([]);
      setShowDeployment(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="p-6">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <Brain className="h-8 w-8 text-purple-600" />
            <div>
              <h3 className="text-lg font-medium text-gray-900">{model.name}</h3>
              <p className="text-sm text-gray-500">{model.description}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-600">Size: {model.size_gb} GB</p>
            <p className="text-xs text-gray-500">Min RAM: {model.min_memory_gb} GB</p>
          </div>
        </div>

        <div className="mt-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Compatible Devices</span>
            <span className="text-sm text-gray-900">
              {compatibleDevices.length} available
            </span>
          </div>
          
          {devicesRunningModel.length > 0 && (
            <div className="mt-2 flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <span className="text-sm text-green-600">
                Running on {devicesRunningModel.length} device(s)
              </span>
            </div>
          )}
        </div>

        <div className="mt-4 flex space-x-2">
          <button
            onClick={() => setShowDeployment(!showDeployment)}
            className="btn btn-primary flex items-center space-x-2"
            disabled={compatibleDevices.length === 0}
          >
            <Play className="h-4 w-4" />
            <span>Deploy</span>
          </button>
          
          <div className="flex flex-wrap gap-1">
            {model.supported_devices.map(deviceType => (
              <span
                key={deviceType}
                className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
              >
                {deviceType.toUpperCase()}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Deployment Panel */}
      {showDeployment && (
        <div className="border-t border-gray-200 p-6 bg-gray-50">
          <h4 className="text-sm font-medium text-gray-900 mb-3">
            Select devices to deploy to:
          </h4>
          
          {compatibleDevices.length === 0 ? (
            <p className="text-sm text-gray-500">
              No compatible devices available. Ensure devices have sufficient memory and are online.
            </p>
          ) : (
            <div className="space-y-2">
              {compatibleDevices.map(device => (
                <label
                  key={device.id}
                  className="flex items-center space-x-3 p-2 rounded hover:bg-gray-100"
                >
                  <input
                    type="checkbox"
                    checked={selectedDevices.includes(device.id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedDevices([...selectedDevices, device.id]);
                      } else {
                        setSelectedDevices(selectedDevices.filter(id => id !== device.id));
                      }
                    }}
                    className="h-4 w-4 text-primary-600 border-gray-300 rounded"
                  />
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-900">
                        {device.name}
                      </span>
                      <span className="text-xs text-gray-500">
                        {device.available_memory_gb.toFixed(1)} GB available
                      </span>
                    </div>
                  </div>
                </label>
              ))}
              
              <div className="mt-4 flex space-x-2">
                <button
                  onClick={handleDeploy}
                  disabled={selectedDevices.length === 0}
                  className="btn btn-primary"
                >
                  Deploy to {selectedDevices.length} device(s)
                </button>
                <button
                  onClick={() => setShowDeployment(false)}
                  className="btn btn-secondary"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function ModelManagement() {
  const { devices: wsDevices } = useWebSocket();
  const queryClient = useQueryClient();

  const { data: models = [] } = useQuery({
    queryKey: ['models'],
    queryFn: modelApi.getModels,
  });

  const { data: devices = [] } = useQuery({
    queryKey: ['devices'],
    queryFn: deviceApi.getDevices,
    refetchInterval: 5000,
  });

  const deployMutation = useMutation({
    mutationFn: modelApi.deployModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['devices'] });
    },
  });

  // Use WebSocket devices if available, otherwise fall back to query data
  const displayDevices = wsDevices.length > 0 ? wsDevices : devices;

  const handleDeploy = (modelId: string, deviceIds: string[]) => {
    deployMutation.mutate({ model_id: modelId, device_ids: deviceIds });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Management</h1>
        <p className="mt-1 text-sm text-gray-600">
          Deploy and manage LLM models across your devices
        </p>
      </div>

      {/* Deployment Status */}
      {deployMutation.isPending && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center">
            <Loader className="h-5 w-5 text-blue-600 animate-spin" />
            <span className="ml-2 text-sm text-blue-800">
              Deploying model to selected devices...
            </span>
          </div>
        </div>
      )}

      {deployMutation.isError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <XCircle className="h-5 w-5 text-red-600" />
            <span className="ml-2 text-sm text-red-800">
              Deployment failed: {deployMutation.error?.message}
            </span>
          </div>
        </div>
      )}

      {deployMutation.isSuccess && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="ml-2 text-sm text-green-800">
              Model deployment initiated successfully
            </span>
          </div>
        </div>
      )}

      {/* Models Grid */}
      {models.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm border p-12 text-center">
          <Brain className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 text-lg font-medium text-gray-900">No models available</h3>
          <p className="mt-2 text-sm text-gray-600">
            Models will appear here when the backend is running.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {models.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              devices={displayDevices}
              onDeploy={handleDeploy}
            />
          ))}
        </div>
      )}
    </div>
  );
} 