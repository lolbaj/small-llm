"""
Utility functions for SmallLLM training, RAM estimation, and checkpoint management.
"""

import os
import torch

# pylint: disable=import-error
import psutil
from config import SmallLLMConfig


def estimate_ram_usage(config: SmallLLMConfig):
    """
    Estimates the projected RAM and VRAM usage for the given model configuration.
    Ryzen 5 5600GT iGPU shares system RAM, so we track total usable RAM.
    """
    # Simple estimate per layer
    d_dim = config.d_model
    e_experts = config.n_experts
    params_per_layer = (
        (4 * d_dim * d_dim)
        + (e_experts * 4 * d_dim * d_dim)
        + (4 * d_dim * d_dim)
        + (d_dim * e_experts)
    )
    total_params = params_per_layer * config.n_layers + (config.vocab_size * d_dim)

    # Each parameter is 4 bytes
    param_mem_gb = (total_params * 4) / (1024**3)

    # AdamW stores 2 states per parameter (8 bytes total)
    optim_mem_gb = (total_params * 8) / (1024**3)

    # Activation memory
    act_mem_gb = (
        config.context_length
        * config.n_layers
        * config.d_model
        * config.pretrain_batch_size
        * 10
        * 4
    ) / (1024**3)

    projected_gb = param_mem_gb + optim_mem_gb + act_mem_gb
    available_gb = psutil.virtual_memory().available / (1024**3)

    print("-" * 50)
    print(f"[*] RAM ESTIMATOR - Projected for {total_params / 1e6:.2f}M parameters")
    print(f" - Model Weights: {param_mem_gb:.2f} GB")
    print(f" - Optimizer State: {optim_mem_gb:.2f} GB")
    print(f" - Peak Activations: ~{act_mem_gb:.2f} GB")
    print(f" - TOTAL PROJECTED: {projected_gb:.2f} GB")
    print(f" - AVAILABLE RAM: {available_gb:.2f} GB")
    print("-" * 50)

    if projected_gb > 13.0:
        print("[!] ERROR: Projected RAM usage exceeds 13GB (max usable). ABORTING.")
        return False
    if projected_gb > 10.0:
        print(
            "[!] WARNING: Projected RAM usage exceeds 10GB. You may experience OOM or swapping."
        )

    return True


def get_autocast_context(device, config):
    """
    Returns a torch.amp.autocast context with the correct dtype based on config.
    """
    return torch.amp.autocast(
        device_type=str(device),
        dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
    )


def save_checkpoint(model, optimizer, step, path="checkpoints/latest.pt"):
    """Saves a model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model, optimizer=None, path="checkpoints/latest.pt"):
    """Loads a model checkpoint."""
    if not os.path.exists(path):
        return 0
    checkpoint = torch.load(path, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"]


def cleanup_checkpoints(directory="checkpoints", keep=3):
    """Removes old checkpoints, keeping the latest N."""
    files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    if len(files) > keep:
        for f_path in files[keep:]:
            os.remove(f_path)
            print(f"[*] Removed old checkpoint: {f_path}")


if __name__ == "__main__":
    test_config = SmallLLMConfig()
    estimate_ram_usage(test_config)
