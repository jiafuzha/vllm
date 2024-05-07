from typing import List, Dict
import os
import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, DeviceConfig, SchedulerConfig
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

from vllm.logger import init_logger

logger = init_logger(__name__)

class NSCPUCacheEngine:
    """Origin:
    Manages the KV cache for CPU backend.

    This class is responsible for initializing and managing CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as copying.

    New:
    ======Change to map vllm seq_id to native KV cache slot_id======
    KV cache is managed in native. 
    """

    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig,
                 parallel_config: ParallelConfig,
                 device_config: DeviceConfig) -> None:
        assert device_config.device_type == "cpu"
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        # set 
        self.block_size = cache_config.block_size
        # Note: In CacheConfig, num_gpu_blocks actual is num_cpu_blocks
        # for CPU backend, because we want to reuse KV cache management
        # in the scheduler.
        self.num_cpu_blocks = cache_config.num_gpu_blocks

        # if cache_config.cache_dtype == "auto":
        #     self.dtype = model_config.dtype
        # else:
        #     self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        # int32 should be enough to hold slot_id
        self.dtype = torch.int32

        # Get attention backend.
        # self.attn_backend = get_attn_backend(model_config.dtype)

        # Initialize the cache.
        # Note: fake kv cache here. We only store native KV cache slot_id here
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks)

    def _allocate_kv_cache(
        self,
        num_blocks: int,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on CPU."""
        # kv_cache_shape = self.attn_backend.get_kv_cache_shape(
        #     num_blocks, self.block_size, self.num_heads, self.head_size)

        # single tensor would be enough to store the sequence id/slot_id
        kv_cache_shape = (num_blocks, self.block_size, 2)
        kv_cache: List[torch.Tensor] = []
        # for _ in range(self.num_layers):
        kv_cache.append(
            torch.empty(kv_cache_shape, dtype=self.dtype, device="cpu"))

        return kv_cache

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError("Swap is not supported in CPUCacheEngine.")

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError("Swap is not supported in CPUCacheEngine.")

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        # self.attn_backend.copy_blocks(self.cpu_cache, src_to_dsts)
        pass

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig
    ) -> int:
        # head_size = model_config.get_head_size()
        # num_heads = model_config.get_num_kv_heads(parallel_config)
        # num_layers = model_config.get_num_layers(parallel_config)

        # key_cache_block = block_size * num_heads * head_size
        # value_cache_block = key_cache_block
        # total = num_layers * (key_cache_block + value_cache_block)

        # VLLM_CPU_KVCACHE_SPACE env and block_size are used to calculate number of cpu blocks
        # model_config.max_model_len must be no greater than block_size,
        # and  VLLM_CPU_KVCACHE_SPACE *_GB/(block_size*element_size) = num_cpu_blocks <= max_num_seqs as verfied in NSCPUCacheEngine
        # Otherwise, native kv cache may run out of slots.
        # Set VLLM_CPU_KVCACHE_SPACE to at least 1GB here before number of cpu blocks being caclulated in vllm. block_size
        # may be adjusted to meet the requirement.
        _GB = 1 << 30
        block_size = cache_config.block_size
        assert block_size >= model_config.max_model_len, "kv cache block size should be equal to max_model_len"
        space_key = "VLLM_CPU_KVCACHE_SPACE"
        space_value = block_size * scheduler_config.max_num_seqs * 4 # int32
        cache_config.cpu_kvcache_space_bytes = space_value
        os.environ[space_key] = str(float(space_value)/_GB)
        logger.info("reset cache_config.cpu_kvcache_space_bytes to %s GB", os.environ[space_key])
        
        total = block_size
        # if cache_dtype == "auto":
        #     dtype = model_config.dtype
        # else:
        #     dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        # dtype_size = torch.tensor([], dtype=dtype).element_size()

        # we use int32 to store native kv cache slot_id
        return 4 * total
