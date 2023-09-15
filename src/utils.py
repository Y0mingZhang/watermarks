import pickle
import torch
from hashlib import sha256
from typing import Any


def obj_to_hash(*obj: Any) -> int:
    digest = sha256(pickle.dumps(obj, protocol=0)).hexdigest()
    # take the first 8 bytes
    return int(digest[:16], 16)


def dict_to_device(d: dict, device: str):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
