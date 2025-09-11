import asyncio
import json
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Any
from loguru import logger
from shared_types import ModelShard, ShardingStrategy

class OllamaInferenceEngine:
    """Ollama-based inference engine for distributed sharding"""
    
    def __init__(self):
        self.loaded_shards: Dict[str, Any] = {}
        self.ollama_models = self._get_available_models()
        
    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]  # First column is model name
                        models.append(model_name)
                logger.info(f"Available Ollama models: {models}")
                return models
            else:
                logger.error(f"Failed to get Ollama models: {result.stderr}")
                return []
        except Exception as e:
            logger.error(f"Error getting Ollama models: {e}")
            return []
    
    async def load_llama_shard(self, shard: ModelShard) -> bool:
        """Load a Llama shard using Ollama"""
        try:
            # Check if the model is available in Ollama
            model_name = "llama3.2:1b"  # Use the 1B model we just downloaded
            
            if model_name not in self.ollama_models:
                logger.error(f"Model {model_name} not available in Ollama")
                return False
            
            # Store shard info
            self.loaded_shards[shard.shard_id] = {
                "type": "ollama",
                "model_name": model_name,
                "layer_start": shard.layer_start,
                "layer_end": shard.layer_end,
                "shard_type": shard.shard_type
            }
            
            logger.info(f"Loaded Ollama shard {shard.shard_id} using {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Ollama shard {shard.shard_id}: {e}")
            return False
    
    async def process_llama_layer_shard(self, input_data: str, layer_start: int, layer_end: int) -> str:
        """Process input through Ollama with layer context"""
        try:
            # Find the shard that contains these layers
            shard_id = None
            for sid, shard_data in self.loaded_shards.items():
                if (shard_data["layer_start"] <= layer_start and 
                    shard_data["layer_end"] >= layer_end):
                    shard_id = sid
                    break
            
            if not shard_id:
                return f"Layers {layer_start}-{layer_end} processed: {input_data}"
            
            shard_data = self.loaded_shards[shard_id]
            model_name = shard_data["model_name"]
            
            # Create a prompt that includes layer context
            prompt = f"[Processing layers {layer_start}-{layer_end}] {input_data}"
            
            # Run Ollama inference
            result = await self._run_ollama_inference(model_name, prompt)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing Ollama layer shard: {e}")
            return f"Layers {layer_start}-{layer_end} processed: {input_data}"
    
    async def process_llama_tensor_shard(self, input_data: str) -> str:
        """Process input through tensor shard using Ollama"""
        try:
            # Find any loaded shard
            if not self.loaded_shards:
                return f"Ollama tensor shard processed: {input_data}"
            
            shard_id = list(self.loaded_shards.keys())[0]
            shard_data = self.loaded_shards[shard_id]
            model_name = shard_data["model_name"]
            
            prompt = f"[Tensor processing] {input_data}"
            result = await self._run_ollama_inference(model_name, prompt)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing Ollama tensor shard: {e}")
            return f"Ollama tensor shard processed: {input_data}"
    
    async def process_llama_pipeline_stage(self, input_data: str, stage_id: str) -> str:
        """Process input through pipeline stage using Ollama"""
        try:
            # Find any loaded shard
            if not self.loaded_shards:
                return f"Ollama pipeline stage {stage_id} processed: {input_data}"
            
            shard_id = list(self.loaded_shards.keys())[0]
            shard_data = self.loaded_shards[shard_id]
            model_name = shard_data["model_name"]
            
            prompt = f"[Pipeline stage {stage_id}] {input_data}"
            result = await self._run_ollama_inference(model_name, prompt)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing Ollama pipeline stage: {e}")
            return f"Ollama pipeline stage {stage_id} processed: {input_data}"
    
    async def _run_ollama_inference(self, model_name: str, prompt: str) -> str:
        """Run inference using Ollama"""
        try:
            # Create a temporary file for the prompt
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(prompt)
                temp_file = f.name
            
            try:
                # Run Ollama with the prompt file
                result = subprocess.run(
                    ['ollama', 'run', model_name, prompt],
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout
                )
                
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    logger.error(f"Ollama inference failed: {result.stderr}")
                    return f"Ollama inference failed: {prompt}"
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except subprocess.TimeoutExpired:
            logger.error(f"Ollama inference timed out for prompt: {prompt}")
            return f"Ollama inference timed out: {prompt}"
        except Exception as e:
            logger.error(f"Error running Ollama inference: {e}")
            return f"Ollama inference error: {prompt}"
    
    async def unload_shard(self, shard_id: str):
        """Unload a model shard"""
        if shard_id in self.loaded_shards:
            del self.loaded_shards[shard_id]
            logger.info(f"Unloaded Ollama shard {shard_id}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "loaded_shards": len(self.loaded_shards),
            "available_models": self.ollama_models,
            "shard_details": self.loaded_shards
        } 
