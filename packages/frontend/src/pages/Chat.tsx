import React, { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { chatApi, modelApi, deviceApi } from '../api';
import { useWebSocket } from '../hooks/useWebSocket';
import { Send, Bot, User, Cpu, Clock } from 'lucide-react';
import { ChatMessage, InferenceRequest } from '../types';

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`flex max-w-xs lg:max-w-md ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        <div className={`flex-shrink-0 ${isUser ? 'ml-2' : 'mr-2'}`}>
          <div className={`h-8 w-8 rounded-full flex items-center justify-center ${
            isUser ? 'bg-primary-600' : 'bg-gray-600'
          }`}>
            {isUser ? (
              <User className="h-4 w-4 text-white" />
            ) : (
              <Bot className="h-4 w-4 text-white" />
            )}
          </div>
        </div>
        
        <div className={`px-4 py-2 rounded-lg ${
          isUser 
            ? 'bg-primary-600 text-white' 
            : 'bg-gray-100 text-gray-900'
        }`}>
          <p className="text-sm">{message.content}</p>
          <div className={`mt-1 text-xs ${isUser ? 'text-primary-100' : 'text-gray-500'}`}>
            {new Date(message.timestamp).toLocaleTimeString()}
            {!isUser && message.device_id && (
              <span className="ml-2">• {message.device_id.slice(0, 8)}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function Chat() {
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [messageInput, setMessageInput] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const messagesEndRef = useRef<HTMLDivElement>(null);
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

  const { data: chatHistory = [] } = useQuery({
    queryKey: ['chat-history'],
    queryFn: chatApi.getChatHistory,
    refetchInterval: 2000,
  });

  const sendMessageMutation = useMutation({
    mutationFn: chatApi.sendMessage,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chat-history'] });
      setMessageInput('');
    },
  });

  // Use WebSocket devices if available, otherwise fall back to query data
  const displayDevices = wsDevices.length > 0 ? wsDevices : devices;

  // Get available models (models that are deployed to at least one online device)
  const availableModels = models.filter(model => 
    displayDevices.some(device => 
      device.current_model === model.id && device.status === 'online'
    )
  );

  // Auto-select first available model
  useEffect(() => {
    if (!selectedModel && availableModels.length > 0) {
      setSelectedModel(availableModels[0].id);
    }
  }, [availableModels, selectedModel]);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const handleSendMessage = () => {
    if (!messageInput.trim() || !selectedModel) return;

    const request: InferenceRequest = {
      message: messageInput.trim(),
      model_id: selectedModel,
      temperature,
    };

    sendMessageMutation.mutate(request);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const devicesForModel = displayDevices.filter(d => d.current_model === selectedModel && d.status === 'online');

  return (
    <div className="flex h-[calc(100vh-8rem)] space-x-6">
      {/* Chat Interface */}
      <div className="flex-1 flex flex-col bg-white rounded-lg shadow-sm border">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div>
            <h1 className="text-lg font-medium text-gray-900">Chat Interface</h1>
            {selectedModel && (
              <p className="text-sm text-gray-600">
                Using {models.find(m => m.id === selectedModel)?.name}
              </p>
            )}
          </div>
          
          {devicesForModel.length > 0 && (
            <div className="flex items-center text-sm text-gray-600">
              <Cpu className="h-4 w-4 mr-1" />
              {devicesForModel.length} device(s) available
            </div>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {chatHistory.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Bot className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-4 text-lg font-medium text-gray-900">
                  Start a conversation
                </h3>
                <p className="mt-2 text-sm text-gray-600">
                  {availableModels.length === 0 
                    ? 'Deploy a model to start chatting'
                    : 'Send a message to begin chatting with the LLM'
                  }
                </p>
              </div>
            </div>
          ) : (
            <>
              {chatHistory.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input */}
        <div className="p-4 border-t border-gray-200">
          <div className="flex space-x-2">
            <textarea
              value={messageInput}
              onChange={(e) => setMessageInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                availableModels.length === 0 
                  ? 'Deploy a model first...'
                  : 'Type your message...'
              }
              disabled={availableModels.length === 0 || sendMessageMutation.isPending}
              className="flex-1 resize-none input"
              rows={2}
            />
            <button
              onClick={handleSendMessage}
              disabled={
                !messageInput.trim() || 
                !selectedModel || 
                availableModels.length === 0 || 
                sendMessageMutation.isPending
              }
              className="btn btn-primary"
            >
              {sendMessageMutation.isPending ? (
                <Clock className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <div className="w-80 space-y-6">
        {/* Model Selection */}
        <div className="bg-white rounded-lg shadow-sm border p-4">
          <h3 className="text-sm font-medium text-gray-900 mb-3">Model Settings</h3>
          
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-gray-600 mb-1">
                Selected Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="input w-full text-sm"
                disabled={availableModels.length === 0}
              >
                <option value="">Select a model...</option>
                {availableModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs text-gray-600 mb-1">
                Temperature: {temperature}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>Focused</span>
                <span>Creative</span>
              </div>
            </div>
          </div>
        </div>

        {/* Active Devices */}
        {selectedModel && devicesForModel.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border p-4">
            <h3 className="text-sm font-medium text-gray-900 mb-3">
              Active Devices ({devicesForModel.length})
            </h3>
            
            <div className="space-y-2">
              {devicesForModel.map((device) => (
                <div
                  key={device.id}
                  className="flex items-center justify-between p-2 bg-gray-50 rounded"
                >
                  <div>
                    <p className="text-sm font-medium text-gray-900">
                      {device.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      CPU: {device.cpu_usage_percent.toFixed(1)}%
                    </p>
                  </div>
                  <div className="h-2 w-2 bg-green-500 rounded-full" />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* No Models Available */}
        {availableModels.length === 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <h3 className="text-sm font-medium text-yellow-800 mb-2">
              No Models Available
            </h3>
            <p className="text-xs text-yellow-700">
              Deploy at least one model to a device to start chatting.
            </p>
          </div>
        )}
      </div>
    </div>
  );
} 