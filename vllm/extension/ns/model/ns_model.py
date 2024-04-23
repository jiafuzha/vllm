from typing import Optional
from torch import nn

from transformers import PretrainedConfig
from transformers import LlamaConfig

from vllm.config import LoRAConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.sampler import Sampler

class NSModel(nn.Module):
    def __init__(self, config: PretrainedConfig,
                   linear_method: Optional[LinearMethodBase],
                   lora_config: Optional[LoRAConfig] = None):
        super(NSModel, self).__init__()
        self.config = config
        self.linear_method = linear_method
        self.lora_config = lora_config

        self.ns_model = None

        self.sampler = Sampler()

    def forward(self, x):
        return self.fc(x)

    def load_weights(self, weights):
        assert sum(weights) > 0
        pass
    
class NSLLamaModel(NSModel):
    def __init__(self, config: LlamaConfig,
                   linear_method: Optional[LinearMethodBase],
                   lora_config: Optional[LoRAConfig] = None):
        super().__init__(config, linear_method, lora_config)