# Orchard - Distributed LLM Platform 🌳

A distributed system for hosting and running Large Language Models (LLMs) across Apple devices (Mac, iPhone, iPad). This project enables you to leverage the combined computing power of your Apple ecosystem for LLM inference.

## Features

- **Multi-Device Support**: Deploy LLMs across Mac, iPhone, and iPad devices
- **Real-time Monitoring**: Live device health, memory usage, and performance metrics
- **Load Balancing**: Intelligent distribution of inference requests across available devices
- **Web Interface**: Modern React-based UI for device management, model deployment, and chat
- **WebSocket Integration**: Real-time updates for device status and chat messages
- **Distributed Architecture**: Built with modern distributed systems best practices

## Demo

https://drive.google.com/file/d/1gUjtkkiOiIw50rW0NhRqXaUvCm_uxBrI/view?usp=sharing

(https://drive.google.com/file/d/1gUjtkkiOiIw50rW0NhRqXaUvCm_uxBrI/view?usp=sharing)

## Architecture 

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │ Device Agents   │
│   (React/TS)    │◄──►│  (FastAPI/Python)│◄──►│   (Python)      │
│                 │    │                 │    │                 │
│ • Device Mgmt   │    │ • Orchestration │    │ • LLM Inference │
│ • Model Deploy  │    │ • Load Balance  │    │ • Health Metrics│
│ • Chat UI       │    │ • Health Monitor│    │ • Model Loading │
│ • Real-time     │    │ • WebSocket Hub │    │ • Auto Register │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start 

### Prerequisites

- Python 3.8+ 
- Node.js 18+
- npm or yarn

### 1. Install Dependencies

**Backend:**
```bash
cd packages/backend
pip install -r requirements.txt
```

**Device Agent:**
```bash
cd packages/device-agent  
pip install -r requirements.txt
```

**Frontend:**
```bash
cd packages/frontend
npm install
```

### 2. Start the Backend

```bash
cd packages/backend
python main.py
```

The backend will be available at `http://localhost:8000`

### 3. Start Device Agents

On each Apple device you want to use:

```bash
cd packages/device-agent
python agent.py --backend http://YOUR_BACKEND_IP:8000 --port 8001
```

**For multiple devices on the same machine (testing):**
```bash
# Terminal 1
python agent.py --port 8001

# Terminal 2  
python agent.py --port 8002

# Terminal 3
python agent.py --port 8003
```

### 4. Start the Frontend

```bash
cd packages/frontend
npm run dev
```

The web interface will be available at `http://localhost:3000`

## Usage Guide 📖

### 1. Dashboard
- Overview of connected devices and system health
- Real-time connection status
- Memory and CPU usage statistics

### 2. Device Management
- View all connected Apple devices
- Monitor device health metrics (CPU, memory, temperature)
- See device status and network information

### 3. Model Management
- Browse available LLM models
- Deploy models to compatible devices
- Monitor deployment status
- View which devices are running which models

### 4. Chat Interface
- Select from deployed models
- Real-time chat with distributed LLMs
- Adjust model parameters (temperature)
- See which device processed each response

## Available Models 🤖

The system comes with sample models:

- **Llama 2 7B**: Meta's 7B parameter model (Mac, iPad)
- **Mistral 7B**: Mistral AI's 7B model (Mac, iPad)  
- **Phi-3 Mini**: Microsoft's compact 3.8B model (Mac, iPad, iPhone)

## Development 🛠️

### Project Structure

```
Orchard/
├── packages/
│   ├── backend/           # FastAPI backend
│   │   ├── main.py       # Main application
│   │   └── requirements.txt
│   ├── device-agent/     # Device agent
│   │   ├── agent.py      # Agent implementation
│   │   └── requirements.txt
│   ├── frontend/         # React frontend
│   │   ├── src/
│   │   │   ├── pages/    # Main pages
│   │   │   ├── components/# UI components
│   │   │   └── hooks/    # React hooks
│   │   └── package.json
│   └── shared/           # Shared types/models
│       └── types.py
└── README.md
```

### API Endpoints

**Devices:**
- `GET /api/devices` - List all devices
- `POST /api/devices/register` - Register new device
- `POST /api/devices/{id}/heartbeat` - Device heartbeat

**Models:**
- `GET /api/models` - List available models
- `POST /api/models/deploy` - Deploy model to devices

**Chat:**
- `POST /api/chat` - Send chat message
- `GET /api/chat/history` - Get chat history

**WebSocket:**
- `WS /ws` - Real-time updates

### Extending the System

**Adding New Models:**
1. Update the `initialize_models()` function in `backend/main.py`
2. Implement model loading in `device-agent/agent.py`

**Adding New Device Types:**
1. Update `DeviceType` enum in `shared/types.py`
2. Add device detection logic in device agent

## Production Deployment 🌐

For production use:

1. **Replace In-Memory Storage**: Use Redis for caching and PostgreSQL for persistence
2. **Add Authentication**: Implement JWT-based auth for the API
3. **Use HTTPS**: Configure SSL certificates
4. **Load Balancer**: Use nginx or similar for frontend
5. **Container Deployment**: Create Docker containers for each service
6. **Model Loading**: Replace mock inference with actual LLM libraries (transformers, llama.cpp, etc.)

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License 📄

MIT License - see LICENSE file for details

## Troubleshooting 🔧

**Common Issues:**

1. **Device not connecting**: Check network connectivity and firewall settings
2. **Model deployment failed**: Ensure device has sufficient memory
3. **Chat not working**: Verify at least one model is deployed and device is online
4. **WebSocket disconnected**: Check if backend is running and accessible

**Debug Mode:**
Add `--log-level debug` to any Python service for verbose logging.

## Roadmap 🗺️

- [ ] Support for more LLM frameworks (llama.cpp, ONNX, etc.)
- [ ] Advanced load balancing strategies  
- [ ] Model caching and compression
- [ ] Device clustering and failover
- [ ] Performance analytics dashboard
- [ ] Mobile apps for device agents (iOS/iPadOS)
- [ ] Integration with cloud services

---

Built with ❤️ for the Apple ecosystem 
