export type DeviceType = 'mac' | 'iphone' | 'ipad';
export type DeviceStatus = 'online' | 'offline' | 'busy' | 'error';

export interface LLMModel {
  id: string;
  name: string;
  size_gb: number;
  min_memory_gb: number;
  description: string;
  supported_devices: DeviceType[];
}

export interface DeviceInfo {
  id: string;
  name: string;
  type: DeviceType;
  status: DeviceStatus;
  ip_address: string;
  port: number;
  total_memory_gb: number;
  available_memory_gb: number;
  cpu_usage_percent: number;
  temperature_celsius?: number;
  last_heartbeat: string;
  current_model?: string;
}

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: string;
  device_id?: string;
}

export interface InferenceRequest {
  message: string;
  model_id: string;
  max_tokens?: number;
  temperature?: number;
}

export interface InferenceResponse {
  response: string;
  device_id: string;
  processing_time_ms: number;
}

export interface DeviceHealthMetrics {
  device_id: string;
  memory_usage_gb: number;
  cpu_usage_percent: number;
  temperature_celsius?: number;
  inference_count: number;
  average_response_time_ms: number;
  timestamp: string;
}

export interface ModelDeploymentRequest {
  model_id: string;
  device_ids: string[];
}

export interface ModelDeploymentStatus {
  model_id: string;
  device_id: string;
  status: 'loading' | 'ready' | 'failed';
  progress_percent: number;
  error_message?: string;
} 