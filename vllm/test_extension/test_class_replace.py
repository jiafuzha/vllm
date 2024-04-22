import importlib
import enum

sub = importlib.import_module('vllm.test_extension.sub')

sub._REGISTRY["niu"] = "dfdfd"

FORMAT_DICT = {e.name : e.value for e in sub.LF}
FORMAT_DICT.update({"NS": "ns"})
LF = enum.Enum("LF", FORMAT_DICT)

sub.LF = LF

def test_method_return_type_hint() -> sub.Base:
    b = sub.Base(1)
    print(b.x)
    return b

if __name__ == "__main__":
    from vllm.test_extension import sub
    sub.test()
    test_method_return_type_hint()
    print(sub.LF.AUTO)
    print(sub.LF.NS)
    print([ (e.name, e.value) for e in sub.LF ])
    # print([ (e.name, e.value) for e in LOAD_FORMAT_WITH_NS ])
    

