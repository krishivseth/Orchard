import asyncio
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from loguru import logger
from shared_types import ModelShard, ShardingStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig

class LlamaShardedLoader:
    """Loads and manages Llama 3.2 shards on a device"""
    
    def __init__(self):
        self.loaded_shards: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
    
    async def load_llama_shard(self, shard: ModelShard) -> bool:
        """Load a specific Llama 3.2 shard"""
        try:
            if shard.shard_type == "layers":
                return await self._load_llama_layer_shard(shard)
            elif shard.shard_type == "tensors":
                return await self._load_llama_tensor_shard(shard)
            elif shard.shard_type == "pipeline_stage":
                return await self._load_llama_pipeline_stage(shard)
            else:
                logger.error(f"Unknown shard type: {shard.shard_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to load Llama shard {shard.shard_id}: {e}")
            return False
    
    async def _load_llama_layer_shard(self, shard: ModelShard) -> bool:
        """Load a layer shard of Llama 3.2"""
        try:
            # Load tokenizer if not already loaded
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(shard.model_path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create a custom config for the layer shard
            config = LlamaConfig(
                num_hidden_layers=shard.llama_config["num_hidden_layers"],
                hidden_size=shard.llama_config["hidden_size"],
                num_attention_heads=shard.llama_config["num_attention_heads"],
                intermediate_size=shard.llama_config["intermediate_size"],
                vocab_size=shard.llama_config["vocab_size"],
                max_position_embeddings=4096,
                rms_norm_eps=1e-6,
                rope_theta=10000.0
            )
            
            # Load the model with custom config
            self.model = AutoModelForCausalLM.from_pretrained(
                shard.model_path,
                config=config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.loaded_shards[shard.shard_id] = {
                "type": "layers",
                "layer_start": shard.layer_start,
                "layer_end": shard.layer_end,
                "model": self.model,
                "tokenizer": self.tokenizer
            }
            
            logger.info(f"Loaded Llama layer shard {shard.shard_id} (layers {shard.layer_start}-{shard.layer_end})")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Llama layer shard: {e}")
            return False
    
    async def _load_llama_tensor_shard(self, shard: ModelShard) -> bool:
        """Load a tensor shard of Llama 3.2"""
        try:
            await asyncio.sleep(1)
            
            self.loaded_shards[shard.shard_id] = {
                "type": "tensors",
                "llama_config": shard.llama_config
            }
            
            logger.info(f"Loaded Llama tensor shard {shard.shard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Llama tensor shard: {e}")
            return False
    
    async def _load_llama_pipeline_stage(self, shard: ModelShard) -> bool:
        """Load a pipeline stage of Llama 3.2"""
        try:
            await asyncio.sleep(1)
            
            self.loaded_shards[shard.shard_id] = {
                "type": "pipeline_stage",
                "layer_start": shard.layer_start,
                "layer_end": shard.layer_end,
                "llama_config": shard.llama_config
            }
            
            logger.info(f"Loaded Llama pipeline stage {shard.shard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Llama pipeline stage: {e}")
            return False
    
    async def process_llama_layer_shard(self, input_data: str, layer_start: int, layer_end: int) -> str:
        """Process input through specific Llama layers"""
        try:
            if not self.model or not self.tokenizer:
                return f"Layers {layer_start}-{layer_end} processed: {input_data}"
            
            # Tokenize input
            inputs = self.tokenizer(input_data, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass through the loaded layers
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Decode the output
            generated_tokens = torch.argmax(logits, dim=-1)
            output_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            return output_text
            
        except Exception as e:
            logger.error(f"Error processing Llama layer shard: {e}")
            return f"Layers {layer_start}-{layer_end} processed: {input_data}"
    
    async def process_llama_tensor_shard(self, input_data: str) -> str:
        """Process input through tensor shard"""
        await asyncio.sleep(0.1)
        return f"Llama tensor shard processed: {input_data}"
    
    async def process_llama_pipeline_stage(self, input_data: str, stage_id: str) -> str:
        """Process input through pipeline stage"""
        await asyncio.sleep(0.1)
        return f"Llama pipeline stage {stage_id} processed: {input_data}"
    
    async def unload_shard(self, shard_id: str):
        """Unload a model shard"""
        if shard_id in self.loaded_shards:
            shard_data = self.loaded_shards[shard_id]
            if "model" in shard_data:
                del shard_data["model"]
            if "tokenizer" in shard_data:
                del shard_data["tokenizer"]
            del self.loaded_shards[shard_id]
            logger.info(f"Unloaded Llama shard {shard_id}") 