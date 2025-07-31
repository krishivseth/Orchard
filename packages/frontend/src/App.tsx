import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { DeviceManagement } from './pages/DeviceManagement';
import { ModelManagement } from './pages/ModelManagement';
import { Chat } from './pages/Chat';
import { WebSocketProvider } from './hooks/useWebSocket';

function App() {
  return (
    <WebSocketProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/devices" element={<DeviceManagement />} />
          <Route path="/models" element={<ModelManagement />} />
          <Route path="/chat" element={<Chat />} />
        </Routes>
      </Layout>
    </WebSocketProvider>
  );
}

export default App; 