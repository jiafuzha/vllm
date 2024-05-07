import ctypes

class Generation(ctypes.Structure):
    _fields_ = [("query_id", ctypes.c_int), ("n_prompt_tokens", ctypes.c_int),
                ("n_generated_tokens", ctypes.c_int), ("max_new_tokens", ctypes.c_int),
                ("receive_time", ctypes.c_int64), ("end_time", ctypes.c_int64),
                ("status", ctypes.c_int), ("prompt_ids", ctypes.POINTER(ctypes.c_int32)), ("generated_ids", ctypes.POINTER(ctypes.c_int32))]