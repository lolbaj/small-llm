"""
Export utilities for quantizing and converting models to .toon and Safetensors.
"""

import copy
import os
import torch

# pylint: disable=import-error
from safetensors.torch import save_file
from model import MoETransformer
from format import ToonFormat


def export_model(aligned_path: str, output_name: str = "small_moe"):
    """Quantize and export the final model to various formats."""
    print(f"[*] Exporting model from {aligned_path}...")

    # 1. Load the original model
    checkpoint = torch.load(aligned_path, map_location="cpu")
    config = checkpoint["config"]
    model = MoETransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. INT8 Quantization (Dynamic)
    print("[*] Applying INT8 quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # 3. Export to .toon
    os.makedirs("export", exist_ok=True)
    toon_path = f"export/{output_name}.toon"
    ToonFormat.save_toon(quantized_model.state_dict(), config, toon_path)

    # 4. Export to Safetensors (float32 and bfloat16)
    print("[*] Saving Safetensors...")
    save_file(model.state_dict(), f"export/{output_name}_fp32.safetensors")

    # Save a bfloat16 version
    model_bf16 = copy_model_to_dtype(model, torch.bfloat16)
    save_file(model_bf16.state_dict(), f"export/{output_name}_bf16.safetensors")

    # 5. Instructions for GGUF
    print("\n" + "=" * 50)
    print(" EXPORT COMPLETE ")
    print("=" * 50)
    print(f"1. Custom Format (.toon): {toon_path}")
    print(f"2. Standard FP32: export/{output_name}_fp32.safetensors")
    print(f"3. Standard BF16: export/{output_name}_bf16.safetensors")
    print("\nTo convert to GGUF for llama.cpp, run:")
    print(
        f"python llama.cpp/convert_hf_to_gguf.py export/{output_name}_fp32.safetensors "
        f"--outfile export/{output_name}.gguf --vocab-type hf"
    )
    print("=" * 50)


def copy_model_to_dtype(model, dtype):
    """Deep copies a model and converts it to a specific dtype."""
    new_model = copy.deepcopy(model)
    new_model.to(dtype)
    return new_model


if __name__ == "__main__":
    if os.path.exists("checkpoints/final_aligned.pt"):
        export_model("checkpoints/final_aligned.pt")
    else:
        print("[-] final_aligned.pt not found. Train first!")
