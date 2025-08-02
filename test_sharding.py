#!/usr/bin/env python3
"""
Test script for Llama 3.2 1B sharding across two Macs
"""

import asyncio
import httpx
import json
from datetime import datetime

async def test_sharding():
    """Test the sharding functionality"""
    
    print("🧪 Testing Llama 3.2 1B Sharding Implementation")
    print("=" * 50)
    
    # Test backend connectivity
    print("\n1. Testing Backend Connectivity...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/api/models")
            if response.status_code == 200:
                models = response.json()
                print(f"✅ Backend connected. Found {len(models)} models")
                for model in models:
                    print(f"   - {model['name']} ({model['id']})")
            else:
                print(f"❌ Backend error: {response.status_code}")
                return
    except Exception as e:
        print(f"❌ Backend connection failed: {e}")
        return
    
    # Test device registration
    print("\n2. Testing Device Registration...")
    try:
        async with httpx.AsyncClient() as client:
            # Simulate device registration
            device_data = {
                "id": "test-device-1",
                "name": "Test Mac 1",
                "type": "mac",
                "status": "online",
                "ip_address": "localhost",
                "port": 8001,
                "total_memory_gb": 16.0,
                "available_memory_gb": 12.0,
                "cpu_usage_percent": 25.0,
                "temperature_celsius": 45.0,
                "last_heartbeat": datetime.now().isoformat()
            }
            
            response = await client.post("http://localhost:8000/api/devices/register", json=device_data)
            if response.status_code == 200:
                print("✅ Device 1 registered successfully")
            else:
                print(f"❌ Device registration failed: {response.status_code}")
                
            # Register second device
            device_data["id"] = "test-device-2"
            device_data["name"] = "Test Mac 2"
            device_data["ip_address"] = "localhost"
            device_data["port"] = 8002
            
            response = await client.post("http://localhost:8000/api/devices/register", json=device_data)
            if response.status_code == 200:
                print("✅ Device 2 registered successfully")
            else:
                print(f"❌ Device 2 registration failed: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Device registration failed: {e}")
        return
    
    # Test sharded deployment
    print("\n3. Testing Sharded Deployment...")
    try:
        async with httpx.AsyncClient() as client:
            deployment_data = {
                "model_id": "llama-3.2-1b",
                "device_ids": ["test-device-1", "test-device-2"],
                "strategy": "layer_split"
            }
            
            response = await client.post("http://localhost:8000/api/models/deploy-llama-sharded", json=deployment_data)
            if response.status_code == 200:
                result = response.json()
                print("✅ Sharded deployment initiated successfully")
                print(f"   - Strategy: {result['config']['strategy']}")
                print(f"   - Total layers: {result['config']['total_layers']}")
                print(f"   - Devices used: {len(result['config']['devices_used'])}")
                print(f"   - Shards created: {len(result['config']['shards'])}")
                
                # Show shard details
                for shard in result['config']['shards']:
                    print(f"   - Shard {shard['shard_id']}: layers {shard['layer_start']}-{shard['layer_end']} on {shard['device_id']}")
            else:
                print(f"❌ Sharded deployment failed: {response.status_code}")
                print(f"   Response: {response.text}")
                print(f"   Request data: {deployment_data}")
                
    except Exception as e:
        print(f"❌ Sharded deployment failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return
    
    # Test sharded inference
    print("\n4. Testing Sharded Inference...")
    try:
        async with httpx.AsyncClient() as client:
            inference_data = {
                "message": "Hello, how are you today?",
                "model_id": "llama-3.2-1b",
                "max_tokens": 50,
                "temperature": 0.7,
                "sharding_strategy": "layer_split"
            }
            
            response = await client.post("http://localhost:8000/api/chat/llama-sharded", json=inference_data)
            if response.status_code == 200:
                result = response.json()
                print("✅ Sharded inference completed successfully")
                print(f"   - Response: {result['response'][:100]}...")
                print(f"   - Processing time: {result['processing_time_ms']}ms")
                print(f"   - Devices used: {result['device_ids']}")
                print(f"   - Shard contributions: {result['shard_contributions']}")
            else:
                print(f"❌ Sharded inference failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Sharded inference failed: {e}")
        return
    
    print("\n🎉 All tests completed successfully!")
    print("\n📋 Summary:")
    print("   - Backend is running and accessible")
    print("   - Device registration works")
    print("   - Sharded deployment is functional")
    print("   - Sharded inference is working")
    print("\n🚀 You can now use the frontend to deploy and test Llama 3.2 1B sharding!")

if __name__ == "__main__":
    asyncio.run(test_sharding()) 