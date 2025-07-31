import axios from 'axios';
import {
  DeviceInfo,
  LLMModel,
  ChatMessage,
  InferenceRequest,
  InferenceResponse,
  DeviceHealthMetrics,
  ModelDeploymentRequest,
} from './types';

const api = axios.create({
  baseURL: '/api',
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
};

export const chatApi = {
  sendMessage: (request: InferenceRequest): Promise<InferenceResponse> =>
    api.post('/chat', request).then(res => res.data),
  
  getChatHistory: (): Promise<ChatMessage[]> =>
    api.get('/chat/history').then(res => res.data),
};

export default api; 