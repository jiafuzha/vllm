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

from vllm.sequence import SamplerOutput, SequenceGroupMetadata


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
        assert len(kv_caches) == 1, "kv_caches should have 1 element here"
        # use data_ptr and torch type in str to avoid inference engine model depends on pytorch and vllm types
        return self.ie_model(input_ids.data_ptr(),
                             positions.data_ptr(),
                             kv_caches[0].data_ptr(), # kv cache type is fixed, int32
                             attn_metadata.is_prompt,
                             attn_metadata.block_tables.data_ptr(),
                             attn_metadata.slot_mapping.data_ptr(),
                             attn_metadata.prompt_lens
                             )
    
    def init_inference_engine(self, model_config: ModelConfig, parallel_config: ParallelConfig, scheduler_config: SchedulerConfig):
        self.ie_model = IE_Model(self.config.name_or_path, max_batch_size=scheduler_config.max_num_seqs, ctx_size=model_config.max_model_len, max_new_tokens=model_config.max_model_len)
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


# modified execute_model in cpu_model_runner.py to pass sequence_id and convert tensor to int32 for now
def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata
         ) = self.prepare_input_tensors(seq_group_metadata_list)
        
        # set seq id to first element of block in kv cache
        # one sequence one block
        if attn_metadata.is_prompt:
            kv_cache = kv_caches[0]
            # 1 for block_size
            block_tables = torch.zeros((len(seq_group_metadata_list) + 1), dtype=torch.int)
            block_tables[0] = self.block_size
            i = 1
            for seq_g_meta in seq_group_metadata_list:
                for seq_id, block_nbrs in seq_g_meta.block_tables.items():
                    block_nbr = block_nbrs[0]
                    kv_cache[block_nbr][0][0] = seq_id
                    block_tables[i] = block_nbr
                    i = i + 1
            assert i == block_tables.shape[0], "inconsistent block tables and sequences"
            attn_metadata.block_tables = block_tables

        model_executable = self.model
        execute_model_kwargs = {
            "input_ids": input_tokens.to(torch.int32),
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }

        hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not sampling_metadata.perform_sampling:
            return None

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return output