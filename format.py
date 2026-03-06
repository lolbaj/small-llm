"""
.toon binary format implementation for model serialization.
"""

import json
import os
import struct
from typing import Dict, Any
import torch


class ToonFormat:
    """
    .toon format specification:
    1. MAGIC: b"TOON" (4 bytes)
    2. VERSION: 0x01 (1 byte)
    3. CONFIG_LEN: uint32 (4 bytes)
    ...
    """

    MAGIC = b"TOON"
    VERSION = 1

    @staticmethod
    # pylint: disable=too-many-locals
    def save_toon(model_state: Dict[str, torch.Tensor], config: Any, file_path: str):
        """Saves a quantized model in the .toon format."""
        # 1. Prepare Config
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
        config_json = json.dumps(config_dict).encode("utf-8")

        # 2. Prepare Vocab (from tokenizer)
        vocab_path = "tokenizer/tokenizer.json"
        if os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f_in:
                vocab_data = f_in.read()
        else:
            vocab_data = b"{}"

        # 3. Prepare Tensors
        tensor_data = bytearray()
        tensor_metadata = []

        for name, tensor in model_state.items():
            # Support quantized or standard tensors
            if tensor.dtype in (torch.qint8, torch.uint8):
                raw = tensor.numpy().tobytes()
                dtype = "qint8"
            elif tensor.dtype == torch.float16:
                raw = tensor.numpy().tobytes()
                dtype = "float16"
            else:
                raw = tensor.float().numpy().tobytes()
                dtype = "float32"

            offset = len(tensor_data)
            tensor_data.extend(raw)
            # Alignment to 16 bytes
            padding = (16 - (len(tensor_data) % 16)) % 16
            tensor_data.extend(b"\x00" * padding)

            tensor_metadata.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": dtype,
                    "offset": offset,
                    "length": len(raw),
                }
            )

        metadata_json = json.dumps(tensor_metadata).encode("utf-8")

        # 4. Write to File
        with open(file_path, "wb") as f_out:
            f_out.write(ToonFormat.MAGIC)
            f_out.write(struct.pack("B", ToonFormat.VERSION))
            f_out.write(struct.pack("I", len(config_json)))
            f_out.write(config_json)
            f_out.write(struct.pack("I", len(vocab_data)))
            f_out.write(vocab_data)
            f_out.write(struct.pack("I", len(tensor_metadata)))
            f_out.write(struct.pack("I", len(metadata_json)))
            f_out.write(metadata_json)
            f_out.write(tensor_data)

        print(
            f"[*] Model saved to {file_path}. Size: {os.path.getsize(file_path) / 1e6:.2f} MB"
        )

    @staticmethod
    # pylint: disable=too-many-locals
    def load_toon(file_path: str) -> Dict[str, Any]:
        """Loads a model and config from the .toon format."""
        with open(file_path, "rb") as f_in:
            if f_in.read(4) != ToonFormat.MAGIC:
                raise ValueError("Invalid .toon file magic.")

            # Skip version
            f_in.read(1)

            # Load Config
            config_len = struct.unpack("I", f_in.read(4))[0]
            config_json = json.loads(f_in.read(config_len).decode("utf-8"))

            # Load Vocab
            vocab_len = struct.unpack("I", f_in.read(4))[0]
            vocab_data = f_in.read(vocab_len)

            # Load Metadata
            _ = struct.unpack("I", f_in.read(4))[0]  # tensor_count
            metadata_len = struct.unpack("I", f_in.read(4))[0]
            metadata = json.loads(f_in.read(metadata_len).decode("utf-8"))

            # Map Tensors
            data_start = f_in.tell()
            state_dict = {}
            # pylint: disable=import-outside-toplevel
            from numpy import frombuffer, float32, int8, float16

            for m in metadata:
                f_in.seek(data_start + m["offset"])
                raw = f_in.read(m["length"])
                if m["dtype"] == "qint8":
                    arr = frombuffer(raw, dtype=int8)
                    state_dict[m["name"]] = torch.from_numpy(arr).view(m["shape"])
                elif m["dtype"] == "float16":
                    arr = frombuffer(raw, dtype=float16)
                    state_dict[m["name"]] = torch.from_numpy(arr).view(m["shape"])
                else:
                    arr = frombuffer(raw, dtype=float32)
                    state_dict[m["name"]] = torch.from_numpy(arr).view(m["shape"])

        return {"config": config_json, "vocab": vocab_data, "state_dict": state_dict}


if __name__ == "__main__":
    from config import SmallLLMConfig

    test_cfg = SmallLLMConfig(fast_test=True)
    test_state = {"layer.weight": torch.randn(10, 10)}
    ToonFormat.save_toon(test_state, test_cfg, "test.toon")
    loaded = ToonFormat.load_toon("test.toon")
    print(f"Loaded weights: {loaded['state_dict']['layer.weight'].shape}")
