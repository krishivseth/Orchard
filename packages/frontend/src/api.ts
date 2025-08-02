import axios from 'axios';
import { 
  DeviceInfo, 
  LLMModel, 
  ChatMessage, 
  InferenceRequest, 
  InferenceResponse,
  ModelDeploymentRequest,
  DeviceHealthMetrics,
  DistributedInferenceRequest,
  ShardedInferenceResponse
} from './types';

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 30000,
});

export const deviceApi = {
  getDevices: (): Promise<DeviceInfo[]> =>
    api.get('/devices').then(res => res.data),
  
  getDeviceMetrics: (deviceId: string): Promise<DeviceHealthMetrics[]> =>
    api.get(`/devices/${deviceId}/metrics`).then(res => res.data),
};

export const modelApi = {
  getModels: (): Promise<LLMModel[]> =>
    api.get('/models').then(res => res.data),
  
  deployModel: (deployment: ModelDeploymentRequest) =>
    api.post('/models/deploy', deployment).then(res => res.data),
  
  deployLlamaSharded: (deployment: any) =>
    api.post('/models/deploy-llama-sharded', deployment).then(res => res.data),
};

export const chatApi = {
  sendMessage: (request: InferenceRequest): Promise<InferenceResponse> =>
    api.post('/chat', request).then(res => res.data),
  
  sendLlamaShardedMessage: (request: DistributedInferenceRequest): Promise<ShardedInferenceResponse> =>
    api.post('/chat/llama-sharded', request).then(res => res.data),
  
  getChatHistory: (): Promise<ChatMessage[]> =>
    api.get('/chat/history').then(res => res.data),
};

export default api; 