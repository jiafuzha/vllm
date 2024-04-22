import enum

_REGISTRY = {"name": "calf"}

class Base:
    def __init__(self, x):
        self.x = x

class LF(str, enum.Enum):
    AUTO = "auto"
    PT = "pt"
    SAFETENSORS = "safetensors"
    NPCACHE = "npcache"
    DUMMY = "dummy"
    TENSORIZER = "tensorizer"

def test():
    for k, v in _REGISTRY.items():
        print(k, v)