"""Avoid the Safetensors/Torch mmap access violation observed on Windows.

Only call this inside the owned model/training subprocess, before importing Laya.
The Rust reader validates the file first. Read-only NumPy mappings then avoid
Torch storage slicing and the full temporary byte buffer on memory-limited
Windows systems. Laya 0.4.6 copies these source tensors into its model; callers
must never mutate the returned tensors. No pickle or precision conversion.
"""
import json
import os
import struct
import warnings


def load_checkpoint_numpy(filename, device="cpu"):
    import numpy as np
    import torch
    from safetensors import safe_open

    # Validate offsets, sizes, dtypes, and overlapping tensors with the upstream parser.
    with safe_open(filename, framework="np") as source:
        names = source.keys()
    with open(filename, "rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
    dtypes = {"F16": "<f2", "F32": "<f4", "F64": "<f8", "I64": "<i8", "I32": "<i4",
              "I16": "<i2", "I8": "i1", "U8": "u1", "BOOL": "?"}
    tensors = {}
    for name in names:
        item = header[name]
        if item["dtype"] not in dtypes:
            raise ValueError(f"Windows checkpoint loader does not support {item['dtype']}; use an F16/F32 Laya checkpoint.")
        shape = tuple(item["shape"])
        dtype = np.dtype(dtypes[item["dtype"]])
        if item["data_offsets"][0] == item["data_offsets"][1]:
            array = np.empty(shape, dtype=dtype)
        else:
            array = np.memmap(filename, dtype=dtype, mode="r", shape=shape,
                              offset=8 + header_size + item["data_offsets"][0])
        # These tensors are read only sources for load_state_dict, never mutated.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            tensors[name] = torch.from_numpy(array).to(device)
    return tensors


def configure_checkpoint_loading():
    if os.name == "nt":
        import safetensors.torch

        safetensors.torch.load_file = load_checkpoint_numpy
