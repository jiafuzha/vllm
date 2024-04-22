from vllm.config import (DeviceConfig, LoadConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from typing import Optional

_REGISTRY = {"name": "calf"}

class DummyModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        *args,
        **kwargs,
    ):
        self.name = "dummy_model_runner"
        print("I am dummy model runner")

def test():
    from vllm.worker.cpu_model_runner import CPUModelRunner
    runner = CPUModelRunner(None, None, None, None, None, None)
    print(_REGISTRY["name"])