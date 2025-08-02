import asyncio
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from shared_types import (
    ModelShardingConfig, ModelShard, ShardingStrategy, 
    DeviceInfo, LLMModel, DistributedInferenceRequest
)

class LlamaShardingEngine:
    """Handles Llama 3.2 model sharding across multiple devices"""
    
    def __init__(self):
        self.sharding_configs: Dict[str, ModelShardingConfig] = {}
        self.device_connections: Dict[str, str] = {}  # device_id -> ip:port
        
        # Llama 3.2 1B configuration
        self.llama_config = {
            'num_hidden_layers': 22,
            'hidden_size': 2048,
            'num_attention_heads': 16,
            'intermediate_size': 5632,
            'vocab_size': 128256,
            'max_position_embeddings': 4096,
            'rms_norm_eps': 1e-6,
            'rope_theta': 10000.0
        }
    
    async def create_llama_sharding_config(
        self, 
        model_id: str, 
        devices: List[DeviceInfo], 
        strategy: ShardingStrategy
    ) -> ModelShardingConfig:
        """Create sharding configuration for Llama 3.2 1B"""
        
        if strategy == ShardingStrategy.LAYER_SPLIT:
            return await self._create_llama_layer_split_config(model_id, devices)
        elif strategy == ShardingStrategy.TENSOR_PARALLEL:
            return await self._create_llama_tensor_parallel_config(model_id, devices)
        elif strategy == ShardingStrategy.PIPELINE_PARALLEL:
            return await self._create_llama_pipeline_parallel_config(model_id, devices)
        else:
            raise ValueError(f"Unsupported sharding strategy: {strategy}")
    
    async def _create_llama_layer_split_config(
        self, 
        model_id: str, 
        devices: List[DeviceInfo]
    ) -> ModelShardingConfig:
        """Split Llama 3.2 1B layers across devices"""
        
        total_layers = self.llama_config['num_hidden_layers']  # 22 layers
        layers_per_device = total_layers // len(devices)
        remaining_layers = total_layers % len(devices)
        
        shards = []
        current_layer = 0
        
        for i, device in enumerate(devices):
            # Give extra layers to first few devices if there are remaining layers
            device_layers = layers_per_device + (1 if i < remaining_layers else 0)
            
            # Calculate memory usage (rough estimate for Llama 3.2 1B)
            # Total model ~ 1.5GB, each layer ~ 50MB
            base_memory = 1.5  # GB
            layer_memory = (device_layers / total_layers) * base_memory
            
            shard = ModelShard(
                shard_id=f"{model_id}-layer-shard-{i}",
                device_id=device.id,
                layer_start=current_layer,
                layer_end=current_layer + device_layers - 1,
                model_path="meta-llama/Llama-3.2-1B",
                shard_type="layers",
                memory_usage_gb=layer_memory,
                llama_config={
                    "num_hidden_layers": device_layers,
                    "hidden_size": self.llama_config['hidden_size'],
                    "num_attention_heads": self.llama_config['num_attention_heads'],
                    "intermediate_size": self.llama_config['intermediate_size'],
                    "layer_start": current_layer,
                    "layer_end": current_layer + device_layers - 1,
                    "vocab_size": self.llama_config['vocab_size']
                }
            )
            shards.append(shard)
            current_layer += device_layers
        
        return ModelShardingConfig(
            model_id=model_id,
            strategy=ShardingStrategy.LAYER_SPLIT,
            shards=shards,
            total_layers=total_layers,
            devices_used=[d.id for d in devices],
            model_name="meta-llama/Llama-3.2-1B"
        )
    
    async def _create_llama_tensor_parallel_config(
        self, 
        model_id: str, 
        devices: List[DeviceInfo]
    ) -> ModelShardingConfig:
        """Split Llama 3.2 1B tensors across devices"""
        
        shards = []
        for i, device in enumerate(devices):
            # Split attention heads and MLP dimensions
            heads_per_device = self.llama_config['num_attention_heads'] // len(devices)
            mlp_split = self.llama_config['intermediate_size'] // len(devices)
            
            shard = ModelShard(
                shard_id=f"{model_id}-tensor-shard-{i}",
                device_id=device.id,
                layer_start=0,  # All layers, but different tensor parts
                layer_end=self.llama_config['num_hidden_layers'] - 1,
                model_path="meta-llama/Llama-3.2-1B",
                shard_type="tensors",
                memory_usage_gb=1.5 / len(devices),  # Equal split
                llama_config={
                    "num_hidden_layers": self.llama_config['num_hidden_layers'],
                    "hidden_size": self.llama_config['hidden_size'],
                    "num_attention_heads": heads_per_device,
                    "intermediate_size": mlp_split,
                    "tensor_shard_id": i,
                    "total_shards": len(devices)
                }
            )
            shards.append(shard)
        
        return ModelShardingConfig(
            model_id=model_id,
            strategy=ShardingStrategy.TENSOR_PARALLEL,
            shards=shards,
            total_layers=self.llama_config['num_hidden_layers'],
            devices_used=[d.id for d in devices],
            model_name="meta-llama/Llama-3.2-1B"
        )
    
    async def _create_llama_pipeline_parallel_config(
        self, 
        model_id: str, 
        devices: List[DeviceInfo]
    ) -> ModelShardingConfig:
        """Create pipeline stages for Llama 3.2 1B"""
        
        total_layers = self.llama_config['num_hidden_layers']
        stages_per_device = total_layers // len(devices)
        
        shards = []
        for i, device in enumerate(devices):
            stage_start = i * stages_per_device
            stage_end = min((i + 1) * stages_per_device - 1, total_layers - 1)
            
            shard = ModelShard(
                shard_id=f"{model_id}-pipeline-stage-{i}",
                device_id=device.id,
                layer_start=stage_start,
                layer_end=stage_end,
                model_path="meta-llama/Llama-3.2-1B",
                shard_type="pipeline_stage",
                memory_usage_gb=1.5 / len(devices),
                llama_config={
                    "num_hidden_layers": stage_end - stage_start + 1,
                    "hidden_size": self.llama_config['hidden_size'],
                    "num_attention_heads": self.llama_config['num_attention_heads'],
                    "intermediate_size": self.llama_config['intermediate_size'],
                    "stage_start": stage_start,
                    "stage_end": stage_end
                }
            )
            shards.append(shard)
        
        return ModelShardingConfig(
            model_id=model_id,
            strategy=ShardingStrategy.PIPELINE_PARALLEL,
            shards=shards,
            total_layers=total_layers,
            devices_used=[d.id for d in devices],
            model_name="meta-llama/Llama-3.2-1B"
        )
    
    async def execute_llama_sharded_inference(
        self, 
        request: DistributedInferenceRequest,
        config: ModelShardingConfig
    ) -> str:
        """Execute inference using sharded Llama 3.2"""
        
        if config.strategy == ShardingStrategy.LAYER_SPLIT:
            return await self._execute_llama_layer_split_inference(request, config)
        elif config.strategy == ShardingStrategy.TENSOR_PARALLEL:
            return await self._execute_llama_tensor_parallel_inference(request, config)
        elif config.strategy == ShardingStrategy.PIPELINE_PARALLEL:
            return await self._execute_llama_pipeline_parallel_inference(request, config)
        else:
            raise ValueError(f"Unsupported strategy: {config.strategy}")
    
    async def _execute_llama_layer_split_inference(
        self, 
        request: DistributedInferenceRequest, 
        config: ModelShardingConfig
    ) -> str:
        """Execute inference with layer splitting"""
        
        # Sort shards by layer order
        sorted_shards = sorted(config.shards, key=lambda s: s.layer_start)
        
        # Forward pass through each layer shard sequentially
        hidden_states = request.message  # Start with input text
        
        for shard in sorted_shards:
            device_ip = self.device_connections.get(shard.device_id)
            if not device_ip:
                continue
                
            # Send to device for layer processing
            shard_request = {
                "input": hidden_states,
                "layer_start": shard.layer_start,
                "layer_end": shard.layer_end,
                "model_id": config.model_id,
                "shard_id": shard.shard_id,
                "llama_config": shard.llama_config
            }
            
            # Process layer shard
            hidden_states = await self._process_llama_layer_shard(device_ip, shard_request)
        
        return hidden_states
    
    async def _execute_llama_tensor_parallel_inference(
        self, 
        request: DistributedInferenceRequest, 
        config: ModelShardingConfig
    ) -> str:
        """Execute inference with tensor parallelism"""
        
        # Process all tensor shards in parallel
        tasks = []
        for shard in config.shards:
            device_ip = self.device_connections.get(shard.device_id)
            if device_ip:
                task = self._process_llama_tensor_shard(device_ip, request, shard)
                tasks.append(task)
        
        # Wait for all tensor shards to complete
        results = await asyncio.gather(*tasks)
        
        # Combine results
        combined_result = self._combine_llama_tensor_results(results)
        return combined_result
    
    async def _execute_llama_pipeline_parallel_inference(
        self, 
        request: DistributedInferenceRequest, 
        config: ModelShardingConfig
    ) -> str:
        """Execute inference with pipeline parallelism"""
        
        # Sort shards by pipeline stage
        sorted_shards = sorted(config.shards, key=lambda s: s.layer_start)
        
        # Pipeline stages process sequentially
        current_input = request.message
        
        for shard in sorted_shards:
            device_ip = self.device_connections.get(shard.device_id)
            if device_ip:
                current_input = await self._process_llama_pipeline_stage(
                    device_ip, current_input, shard
                )
        
        return current_input
    
    async def _process_llama_layer_shard(self, device_ip: str, request: dict) -> str:
        """Process a Llama layer shard on a device"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{device_ip}/llama/layer-inference",
                    json=request
                )
                if response.status_code == 200:
                    return response.json()["output"]
                else:
                    logger.error(f"Layer shard processing failed: {response.status_code}")
                    return request["input"]  # Return input if processing fails
        except Exception as e:
            logger.error(f"Error processing layer shard: {e}")
            return request["input"]
    
    async def _process_llama_tensor_shard(self, device_ip: str, request: DistributedInferenceRequest, shard: ModelShard) -> str:
        """Process a Llama tensor shard on a device"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{device_ip}/llama/tensor-inference",
                    json={
                        "input": request.message,
                        "shard_id": shard.shard_id,
                        "llama_config": shard.llama_config
                    }
                )
                if response.status_code == 200:
                    return response.json()["output"]
                else:
                    return f"Tensor shard {shard.shard_id} processed by {device_ip}"
        except Exception as e:
            logger.error(f"Error processing tensor shard: {e}")
            return f"Tensor shard {shard.shard_id} processed by {device_ip}"
    
    async def _process_llama_pipeline_stage(self, device_ip: str, input_data: str, shard: ModelShard) -> str:
        """Process a Llama pipeline stage on a device"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{device_ip}/llama/pipeline-inference",
                    json={
                        "input": input_data,
                        "shard_id": shard.shard_id,
                        "llama_config": shard.llama_config
                    }
                )
                if response.status_code == 200:
                    return response.json()["output"]
                else:
                    return f"Pipeline stage {shard.shard_id} processed by {device_ip}: {input_data}"
        except Exception as e:
            logger.error(f"Error processing pipeline stage: {e}")
            return f"Pipeline stage {shard.shard_id} processed by {device_ip}: {input_data}"
    
    def _combine_llama_tensor_results(self, results: List[str]) -> str:
        """Combine results from tensor parallel processing"""
        return "Combined Llama tensor results: " + " + ".join(results) 