from typing import Optional
from torch import nn

from transformers import PretrainedConfig
from transformers import LlamaConfig

from vllm.config import LoRAConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.sampler import Sampler

from inference_engine import Model as IE_Model

class NSModel(nn.Module):
    def __init__(self, config: PretrainedConfig,
                   linear_method: Optional[LinearMethodBase],
                   lora_config: Optional[LoRAConfig] = None):
        super(NSModel, self).__init__()
        self.config = config
        self.linear_method = linear_method
        self.lora_config = lora_config
        # TODO: determine pad token
        self.ie_model = IE_Model(config.name_or_path, max_batch_size=10)

        self.sampler = Sampler()

    def forward(self, x):
        return self.fc(x)

    def load_weights(self, weights):
        assert sum(weights) > 0
        
        qc = self.linear_method.quant_config
        self.ie_model.check_and_quantize(weight_dtype=qc.weight_dtype,
                                         alg=qc.alg,
                                         group_size=qc.group_size,
                                         scale_dtype=qc.scale_dtype,
                                         compute_dtype=qc.compute_dtype,
                                        )
        self.ie_model.load_weights()

    
class NSLLamaModel(NSModel):
    def __init__(self, config: LlamaConfig,
                   linear_method: Optional[LinearMethodBase],
                   lora_config: Optional[LoRAConfig] = None):
        super().__init__(config, linear_method, lora_config)