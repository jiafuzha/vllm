# extension for integrate with neural-speed
import enum
import importlib
from typing import Optional

from vllm.model_executor.model_loader.weight_utils import get_quant_config

# allow ns quantization
vllm_config = importlib.import_module('vllm.config')
old_verify_quantization = vllm_config.ModelConfig._verify_quantization

def _verify_quntization(self):
    if self.quantization is not None and self.quantization == "ns":
        return
    return old_verify_quantization(self)

vllm_config.ModelConfig._verify_quantization = _verify_quntization

## add new loadformat ns
FORMAT_DICT = {e.name : e.value for e in vllm_config.LoadFormat}
FORMAT_DICT.update({"NS": "ns"})
LF = enum.Enum("LoadFormat", FORMAT_DICT)

vllm_config.LoadFormat = LF

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

# use ns model loader for ns
vllm_loader = importlib.import_module('vllm.model_executor.model_loader.loader')

from vllm.extension.ns.model.ns_loader import NSModelLoader
from vllm.config import LoadConfig

def get_model_loader_ns(load_config: LoadConfig) -> vllm_loader.BaseModelLoader:
    if load_config.load_format == "ns":
        return NSModelLoader(load_config)
    return vllm_loader.DefaultModelLoader(load_config)

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

# register ns model
from vllm.model_executor.models import ModelRegistry
from vllm.extension.ns.model.ns_model import NSLLamaModel

ModelRegistry.register_model("LlamaForCausalLM", NSLLamaModel)

