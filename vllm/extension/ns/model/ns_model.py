from typing import List, Optional
import os
import torch
from torch import nn

from transformers import PretrainedConfig
from transformers import LlamaConfig

from vllm.config import LoRAConfig, ModelConfig, SchedulerConfig, ParallelConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.sampler import Sampler
from vllm.attention import AttentionMetadata

from inference_engine import Model as IE_Model

class NSModel(nn.Module):
    def __init__(self, config: PretrainedConfig,
                   linear_method: Optional[LinearMethodBase],
                   lora_config: Optional[LoRAConfig] = None):
        super(NSModel, self).__init__()
        self.config = config
        self.linear_method = linear_method
        self.lora_config = lora_config
        self.ie_model = None
        self.tokenizer = None

        self.sampler = Sampler()

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata):
        
        return self.ie_model(input_ids, positions, kv_caches, attn_metadata)
    
    def init_inference_engine(self, model_config: ModelConfig, parallel_config: ParallelConfig, scheduler_config: SchedulerConfig):
        self.ie_model = IE_Model(self.config.name_or_path, max_batch_size=scheduler_config.max_num_seqs)
        self.tokenizer = self.ie_model.tokenizer

    def load_weights(self, weights):
        assert sum(1 for _ in weights) > 0
        
        qc = self.linear_method.quant_config
        self.ie_model.check_and_quantize(weight_dtype=qc.weight_dtype,
                                         alg=qc.alg,
                                         group_size=qc.group_size,
                                         scale_dtype=qc.scale_dtype,
                                         compute_dtype=qc.compute_dtype,
                                        )
        self.ie_model.load_model()

    
class NSLLamaModel(NSModel):
    def __init__(self, config: LlamaConfig,
                   linear_method: Optional[LinearMethodBase],
                   lora_config: Optional[LoRAConfig] = None):
        super().__init__(config, linear_method, lora_config)