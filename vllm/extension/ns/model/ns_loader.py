import glob
import importlib
import os
from typing import Generator, List, Optional, Tuple
import torch
from torch import nn
from vllm.config import VLLM_USE_MODELSCOPE, DeviceConfig, LoRAConfig, LoadConfig, LoadFormat, ModelConfig, ParallelConfig, SchedulerConfig, VisionLanguageConfig
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf, filter_files_not_needed_for_inference, np_cache_weights_iterator, pt_weights_iterator, safetensors_weights_iterator

# extend get_model_loader to load ns model
vllm_loader = importlib.import_module('vllm.model_executor.model_loader.loader')

class NSModelLoader(vllm_loader.BaseModelLoader):
    """Model loader that can load different file types from disk."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def _maybe_download_from_modelscope(
            self, model: str, revision: Optional[str]) -> Optional[str]:
        """Download model from ModelScope hub if VLLM_USE_MODELSCOPE is True.
        
        Returns the path to the downloaded model, or None if the model is not
        downloaded from ModelScope."""
        if VLLM_USE_MODELSCOPE:
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            # pylint: disable=C.
            from modelscope.hub.snapshot_download import snapshot_download

            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=self.load_config.download_dir,
                    revision=revision)
            else:
                model_path = model
            return model_path
        return None

    def _prepare_weights(self, model_name_or_path: str,
                         revision: Optional[str],
                         fall_back_to_pt: bool) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = self._maybe_download_from_modelscope(
            model_name_or_path, revision) or model_name_or_path

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        # Some quantized models use .pt files for storing the weights.
        if load_format == LoadFormat.AUTO:
            allow_patterns = ["*.safetensors", "*.bin"]
        elif load_format == LoadFormat.SAFETENSORS:
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == LoadFormat.PT:
            allow_patterns = ["*.pt"]
        elif load_format == LoadFormat.NPCACHE:
            allow_patterns = ["*.bin"]
        elif load_format == LoadFormat.NS: # ====NS changed====
            allow_patterns = ["*.safetensors", "*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if not is_local:
            hf_folder = download_weights_from_hf(model_name_or_path,
                                                 self.load_config.download_dir,
                                                 allow_patterns)
        else:
            hf_folder = model_name_or_path

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if not use_safetensors:
            hf_weights_files = filter_files_not_needed_for_inference(
                hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, model_name_or_path: str, revision: Optional[str],
        fall_back_to_pt: bool
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            model_name_or_path, revision, fall_back_to_pt)
        if self.load_config.load_format == LoadFormat.NPCACHE:
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            return np_cache_weights_iterator(model_name_or_path,
                                             self.load_config.download_dir,
                                             hf_folder, hf_weights_files)
        if use_safetensors:
            return safetensors_weights_iterator(hf_weights_files)
        return pt_weights_iterator(hf_weights_files)

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig) -> nn.Module:
        if (model_config.quantization is None) or (model_config.quantization != 'ns'):
            raise ValueError(f"Model {model_config.model} is not a NS model")
        
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = vllm_loader._initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config)
            # initialize native inference engine
            model.init_inference_engine(model_config, parallel_config, scheduler_config)
            model.load_weights(
                self._get_weights_iterator(model_config.model,
                                           model_config.revision,
                                           fall_back_to_pt=getattr(
                                               model,
                                               "fall_back_to_pt_during_load",
                                               True)), )

        return model.eval()
    
class NSModelLoaderV2(vllm_loader.DefaultModelLoader):

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig) -> nn.Module:
        if (model_config.quantization is None) or (model_config.quantization != 'ns'):
            raise ValueError(f"Model {model_config.model} is not a NS model")
        
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                vllm_model = vllm_loader._initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config)
            # initialize native inference engine
            vllm_model.model.init_inference_engine(model_config, parallel_config, scheduler_config)
            vllm_model.model.load_weights([0])
            weights_generator = self._get_weights_iterator(model_config.model,
                                           model_config.revision,
                                           fall_back_to_pt=getattr(
                                               vllm_model,
                                               "fall_back_to_pt_during_load",
                                               True))
            
            filtered_weights = [(name, weight) for name, weight in weights_generator if (name == "lm_head.weight" or name == "model.embed_tokens.weight")]
            vllm_model.load_weights(filtered_weights)

        return vllm_model.eval()
