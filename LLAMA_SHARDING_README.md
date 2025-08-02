# Llama 3.2 1B Distributed Sharding Implementation

This implementation allows you to split a single Llama 3.2 1B model across multiple Macs for distributed inference using Thunderbolt networking.

## 🎯 Overview

The system splits the 22 transformer layers of Llama 3.2 1B across your devices:

- **Mac 1 (Main)**: Layers 0-10 (11 layers) - ~750MB memory
- **Mac 2 (Thunderbolt)**: Layers 11-21 (11 layers) - ~750MB memory

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Backend dependencies
cd packages/backend
pip install -r requirements.txt

# Device agent dependencies (includes PyTorch and Transformers)
cd ../device-agent
pip install -r requirements.txt

# Frontend dependencies
cd ../frontend
npm install
```

### 2. Accept Llama 3.2 License

Visit [HuggingFace Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and click "I Accept" to agree to the license.

### 3. Start the System

```bash
# Start all services
./scripts/start-dev.sh
```

### 4. Access the Frontend

Open http://localhost:3000 in your browser.

## 🔧 How It Works

### Layer Split Strategy (Recommended)

The 22 transformer layers are divided across your devices:

```
Input → Mac 1 (Layers 0-10) → Mac 2 (Layers 11-21) → Output
```

### Sharding Configuration

- **Total Layers**: 22
- **Layers per Device**: 11 (for 2 devices)
- **Memory per Device**: ~750MB (vs 1.5GB for full model)
- **Network**: Thunderbolt for fast inter-device communication

### Inference Flow

1. **Tokenization**: Input text is tokenized
2. **Layer Processing**: Each device processes its assigned layers
3. **Data Transfer**: Hidden states are passed between devices
4. **Output Generation**: Final response is generated

## 📁 Implementation Files

### Backend
- `packages/backend/llama_sharding.py` - Sharding engine
- `packages/backend/main.py` - API endpoints for sharding

### Device Agent
- `packages/device-agent/llama_sharded_inference.py` - Model shard loader
- `packages/device-agent/agent.py` - Device agent with sharding support

### Frontend
- `packages/frontend/src/pages/ModelManagement.tsx` - Sharding UI
- `packages/frontend/src/api.ts` - Sharding API calls
- `packages/frontend/src/types.ts` - Sharding types

### Shared Types
- `packages/shared/types.py` - Sharding data models

## 🎮 Usage

### 1. Deploy Sharded Model

1. Go to **Model Management** in the frontend
2. Find **Llama 3.2 1B** model
3. Click **"Shard"** button
4. Select **2 or more devices**
5. Choose **"Layer Split"** strategy
6. Click **"Deploy Sharded"**

### 2. Test Sharded Inference

1. Go to **Chat** interface
2. Select **Llama 3.2 1B** model
3. Send a message
4. Watch the response come from multiple devices

### 3. Monitor Performance

- Check device metrics in **Device Management**
- View processing times and shard contributions
- Monitor memory usage across devices

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_sharding.py
```

This will test:
- Backend connectivity
- Device registration
- Sharded deployment
- Sharded inference

## 🔍 API Endpoints

### Sharded Deployment
```http
POST /api/models/deploy-llama-sharded
{
  "model_id": "llama-3.2-1b",
  "device_ids": ["device-1", "device-2"],
  "strategy": "layer_split"
}
```

### Sharded Inference
```http
POST /api/chat/llama-sharded
{
  "message": "Hello, how are you?",
  "model_id": "llama-3.2-1b",
  "max_tokens": 50,
  "temperature": 0.7,
  "sharding_strategy": "layer_split"
}
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │ Device Agents   │
│   (React/TS)    │◄──►│  (FastAPI)      │◄──►│   (Python)      │
│                 │    │                 │    │                 │
│ • Sharding UI   │    │ • Shard Config  │    │ • Layer Shards  │
│ • Model Deploy  │    │ • Load Balance  │    │ • Tensor Shards │
│ • Chat Interface│    │ • Orchestration │    │ • Pipeline      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Performance Benefits

### Memory Efficiency
- **Single Device**: 1.5GB memory usage
- **2 Devices**: 750MB per device
- **4 Devices**: 375MB per device

### Scalability
- Add more devices to reduce memory per device
- Thunderbolt networking for fast data transfer
- Automatic load balancing

### Flexibility
- Multiple sharding strategies
- Dynamic device discovery
- Real-time monitoring

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Ensure you've accepted the Llama 3.2 license
   - Check internet connection for model download
   - Verify sufficient disk space

2. **Device Connection Issues**
   - Check Thunderbolt connection
   - Verify device IP addresses
   - Ensure firewall allows connections

3. **Memory Issues**
   - Close other applications
   - Reduce batch size
   - Use more devices for smaller shards

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
./scripts/start-dev.sh
```

## 🔮 Future Enhancements

- **Tensor Parallelism**: Split attention heads across devices
- **Pipeline Parallelism**: Process different model stages
- **Dynamic Sharding**: Automatic shard redistribution
- **Model Quantization**: Further reduce memory usage
- **GPU Support**: CUDA acceleration for faster inference

## 📝 License

This implementation follows the Llama 3.2 Community License. See [Meta's license](https://huggingface.co/meta-llama/Llama-3.2-1B) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the test script output
- Check device logs in `logs/` directory
- Open an issue on GitHub

---

**Happy Sharding! 🚀** 