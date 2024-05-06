# extension for integrate with neural-speed
import importlib
from typing import Optional
import os

from vllm.model_executor.model_loader.weight_utils import get_quant_config

from vllm.logger import init_logger

logger = init_logger(__name__)

# allow ns quantization
vllm_config = importlib.import_module('vllm.config')
old_verify_quantization = vllm_config.ModelConfig._verify_quantization

def _verify_quntization(self):
    if self.quantization is not None and self.quantization == "ns":
        os.environ["NS_QUANTIZATION"] = "1"
        return
    return old_verify_quantization(self)

vllm_config.ModelConfig._verify_quantization = _verify_quntization

## add new loadformat ns
# FORMAT_DICT = {e.name : e.value for e in vllm_config.LoadFormat}
# FORMAT_DICT.update({"NS": "ns"})
# LF = enum.Enum("LoadFormat", FORMAT_DICT)

# vllm_config.LoadFormat = LF

# register ns quant config, 
# TODO add quant_ns_config.json under huggingface cache folder like below,
# {
# "quantization_config": {
#     "quant_method": "ns",
#     "weight_dtype": "int4",
#     "alg": "sym",
#     "group_size": 128,
#     "scale_dtype": "fp32",
#     "compute_dtype": "int8",
#     "version": "v1"
# }
# }
quant = importlib.import_module('vllm.model_executor.layers.quantization')

from vllm.extension.ns.quantization.cpu_ns_config import NSQuantConfig

quant._QUANTIZATION_CONFIG_REGISTRY["ns"] = NSQuantConfig

logger.info("__ns extension: add ns quantization config, %s", NSQuantConfig.__name__)

# use ns model loader for ns
vllm_loader = importlib.import_module('vllm.model_executor.model_loader.loader')
old_get_model_loader = vllm_loader.get_model_loader

from vllm.extension.ns.model.ns_loader import NSModelLoader
from vllm.config import LoadConfig

def get_model_loader_ns(load_config: LoadConfig) -> vllm_loader.BaseModelLoader:
    if os.environ["NS_QUANTIZATION"] == "1":
        return NSModelLoader(load_config)
    return old_get_model_loader(load_config)

vllm_loader.get_model_loader = get_model_loader_ns

def _get_linear_method_ns(model_config: vllm_config.ModelConfig,
        load_config: LoadConfig) -> Optional["LinearMethodBase"]:
    """Get the (maybe quantized) linear method."""
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config, load_config)
        linear_method = quant_config.get_linear_method()
    return linear_method

vllm_loader._get_linear_method = _get_linear_method_ns

# reload to make above changes take effect
model_loader = importlib.import_module('vllm.model_executor.model_loader')
importlib.reload(model_loader)

logger.info("__ns extension: use ns model loader for ns model, %s", NSModelLoader.__name__)

# register ns model
from vllm.model_executor.models import ModelRegistry
from vllm.extension.ns.model.ns_model import NSLLamaModel

ModelRegistry.register_model("LlamaForCausalLM", NSLLamaModel)

logger.info("__ns extension: register ns model, %s", NSLLamaModel.__name__)

# use our CPUCacheEngine for ns
cpu_worker = importlib.import_module('vllm.worker.cpu_worker')

from vllm.extension.ns.kv_cache.ns_cache import NSCPUCacheEngine
cpu_worker.CPUCacheEngine = NSCPUCacheEngine

def get_cache_block_size_bytes(self) -> int:
    """Return the size in bytes of a single KV cache block.
    """
    return NSCPUCacheEngine.get_cache_block_size(self.cache_config, self.model_config, self.parallel_config, self.scheduler_config)

cpu_worker.CPUWorker.get_cache_block_size_bytes = get_cache_block_size_bytes

logger.info("__ns extension: use ns cache engine for ns, %s", NSCPUCacheEngine.__name__)

# use our execute_model method to do some conversion and pass more parameters
cpu_model_runner = importlib.import_module('vllm.worker.cpu_model_runner')

from vllm.extension.ns.model.ns_model import execute_model
cpu_model_runner.CPUModelRunner.execute_model = execute_model

logger.info("__ns extension: replace execute_model in cpu_model_runner, %s", execute_model.__name__)


