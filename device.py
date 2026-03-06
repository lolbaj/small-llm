"""
Device detection and ROCm/CUDA initialization for SmallLLM.
"""

import os
import platform
import torch


def get_device():
    """
    Detects the best available device (ROCm/CUDA/CPU).
    For the Ryzen 5 5600GT (Vega 7 iGPU), it attempts to force ROCm compatibility
    using the HSA_OVERRIDE_GFX_VERSION environment variable.
    """
    # AMD Ryzen 5600GT iGPU is GFX90C.
    # ROCm doesn't support it officially, but setting this to 9.0.0
    # (Vega 10) usually allows PyTorch ROCm to work.
    if platform.system() == "Linux":
        # Setting environment variable before any torch.cuda calls
        # This is critical for the driver to recognize the iGPU
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.0"

    if torch.cuda.is_available():
        # In ROCm builds, torch.cuda.is_available() returns True for AMD GPUs
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"[*] Graphics Acceleration Detected: {device_name}")

        # Verify if it's actually working (simple tensor op)
        try:
            torch.ones(1).to(device)
            print("[+] ROCm/CUDA initialized successfully.")
        except RuntimeError as e:
            print(f"[-] ROCm initialization failed: {e}")
            print("[!] Falling back to CPU. Check your ROCm installation.")
            device = torch.device("cpu")
    else:
        print("[*] No GPU detected. Using CPU.")
        device = torch.device("cpu")

    return device


if __name__ == "__main__":
    dev = get_device()
    print(f"Active Device: {dev}")
