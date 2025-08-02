import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelApi, deviceApi } from '../api';
import { useWebSocket } from '../hooks/useWebSocket';
import { Brain, Download, Play, CheckCircle, XCircle, Loader } from 'lucide-react';
import { LLMModel, DeviceInfo, ShardingStrategy } from '../types';

function ModelCard({ 
  model, 
  devices, 
  onDeploy,
  onDeploySharded 
}: { 
  model: LLMModel; 
  devices: DeviceInfo[];
  onDeploy: (modelId: string, deviceIds: string[]) => void;
  onDeploySharded: (modelId: string, deviceIds: string[], strategy: ShardingStrategy) => void;
}) {
  const [selectedDevices, setSelectedDevices] = useState<string[]>([]);
  const [showDeployment, setShowDeployment] = useState(false);
  const [showShardedDeployment, setShowShardedDeployment] = useState(false);
  const [shardingStrategy, setShardingStrategy] = useState<ShardingStrategy>('layer_split');

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

  const handleShardedDeploy = () => {
    if (selectedDevices.length > 1) {  // Need at least 2 devices for sharding
      onDeploySharded(model.id, selectedDevices, shardingStrategy);
      setSelectedDevices([]);
      setShowShardedDeployment(false);
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
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600">
                {compatibleDevices.length} compatible devices
              </span>
              {devicesRunningModel.length > 0 && (
                <span className="text-sm text-green-600">
                  Running on {devicesRunningModel.length} device(s)
                </span>
              )}
            </div>
            <div className="flex space-x-2">
              {compatibleDevices.length > 0 && (
                <button
                  onClick={() => setShowDeployment(!showDeployment)}
                  className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
                >
                  <Play className="w-3 h-3 mr-1" />
                  Deploy
                </button>
              )}
              {compatibleDevices.length > 1 && model.id === 'llama-3.2-1b' && (
                <button
                  onClick={() => setShowShardedDeployment(!showShardedDeployment)}
                  className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700"
                >
                  <Brain className="w-3 h-3 mr-1" />
                  Shard
                </button>
              )}
            </div>
          </div>
        </div>

        {showDeployment && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <h4 className="font-medium text-blue-900 mb-2">Deploy Model</h4>
            <div className="space-y-2">
              {compatibleDevices.map(device => (
                <label key={device.id} className="flex items-center">
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
                    className="mr-2"
                  />
                  <span className="text-sm text-blue-900">
                    {device.name} ({device.available_memory_gb} GB available)
                  </span>
                </label>
              ))}
            </div>
            <div className="flex space-x-2 mt-3">
              <button
                onClick={handleDeploy}
                disabled={selectedDevices.length === 0}
                className="px-3 py-1.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 text-sm"
              >
                Deploy
              </button>
              <button
                onClick={() => setShowDeployment(false)}
                className="px-3 py-1.5 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400 text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {showShardedDeployment && (
          <div className="mt-4 p-4 bg-purple-50 rounded-lg">
            <h4 className="font-medium text-purple-900 mb-2">Sharded Deployment</h4>
            
            <div className="mb-3">
              <label className="block text-sm font-medium text-purple-900 mb-1">
                Sharding Strategy
              </label>
              <select
                value={shardingStrategy}
                onChange={(e) => setShardingStrategy(e.target.value as ShardingStrategy)}
                className="w-full px-3 py-2 border border-purple-300 rounded-md bg-white"
              >
                <option value="layer_split">Layer Split</option>
                <option value="tensor_parallel">Tensor Parallel</option>
                <option value="pipeline_parallel">Pipeline Parallel</option>
              </select>
            </div>
            
            <div className="space-y-2 mb-3">
              {compatibleDevices.map(device => (
                <label key={device.id} className="flex items-center">
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
                    className="mr-2"
                  />
                  <span className="text-sm text-purple-900">
                    {device.name} ({device.available_memory_gb} GB available)
                  </span>
                </label>
              ))}
            </div>
            
            <div className="flex space-x-2">
              <button
                onClick={handleShardedDeploy}
                disabled={selectedDevices.length < 2}
                className="px-3 py-1.5 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 text-sm"
              >
                Deploy Sharded
              </button>
              <button
                onClick={() => setShowShardedDeployment(false)}
                className="px-3 py-1.5 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400 text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
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

  const deployShardedMutation = useMutation({
    mutationFn: modelApi.deployLlamaSharded,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['devices'] });
    },
  });

  // Use WebSocket devices if available, otherwise fall back to query data
  const displayDevices = wsDevices.length > 0 ? wsDevices : devices;

  const handleDeploy = (modelId: string, deviceIds: string[]) => {
    deployMutation.mutate({ model_id: modelId, device_ids: deviceIds });
  };

  const handleDeploySharded = (modelId: string, deviceIds: string[], strategy: ShardingStrategy) => {
    deployShardedMutation.mutate({ 
      model_id: modelId, 
      device_ids: deviceIds, 
      strategy: strategy 
    });
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
      {(deployMutation.isPending || deployShardedMutation.isPending) && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center">
            <Loader className="w-5 h-5 text-blue-600 animate-spin mr-2" />
            <span className="text-blue-800">
              {deployMutation.isPending ? 'Deploying model...' : 'Deploying sharded model...'}
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
              onDeploySharded={handleDeploySharded}
            />
          ))}
        </div>
      )}
    </div>
  );
} 