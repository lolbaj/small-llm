# 🎮 iGPU Utilization Guide: AMD Ryzen 5 5600GT

This document details how we leverage the **AMD Radeon Vega 7 (GFX90C)** integrated GPU for training and inference, despite it lacking official ROCm support.

## 1. Hardware Context
- **Architecture:** GFX90C (Vega 2nd Gen)
- **Compute Units:** 7
- **Memory:** Shared System RAM (Unified Memory Architecture)
- **Shared Limit:** Typically ~2GB-4GB assigned in BIOS, but ROCm can access more of the system's 16GB (up to ~12-13GB usable for this project).

## 2. The ROCm Workaround
The Vega 7 is not officially supported by modern ROCm (which targets GFX906/908/90A/1030+). However, it is architecturally similar to the Vega 10 (GFX900). 

We "mask" the device to force compatibility by setting:
```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0
```

### Automatic Handling
In this project, `device.py` handles this before initializing `torch.cuda`. It checks the OS and applies the override if Linux is detected, ensuring the iGPU is visible to PyTorch without manual environment exports.

## 3. Installation Requirements (Arch Linux)
To use the iGPU, you must install the ROCm-enabled version of PyTorch:

```bash
# Recommendation: Use the official Arch 'python-pytorch-rocm' package or AUR
sudo pacman -S rocm-core hip-runtime-amd python-pytorch-rocm
```

## 4. Shared RAM Management
Because the iGPU shares RAM with the CPU, training a 60M parameter model requires careful monitoring:
- **Total System RAM:** 16GB
- **OS/Background Overhead:** ~2GB-3GB
- **Usable for Training:** ~12GB-13GB
- **Optimization:** We use `utils.estimate_ram_usage()` to calculate the projected footprint of the Model + Optimizer + Activations before each stage begins.

## 5. Performance Expectations
- **Speedup:** Expect **3x to 5x** faster training compared to pure CPU execution for these small MoE models.
- **Mixed Precision:** Use `torch.amp` with `bfloat16` if supported by the driver, or `float16` for maximum speed on older Vega architectures.
- **No Flash Attention:** Vega 7 does not reliably support Flash Attention kernels in ROCm; we use standard scaled dot-product attention instead.

## 6. Verification
To verify the iGPU is active and masked correctly, run:
```bash
python device.py
```
**Expected Output:**
`[*] Graphics Acceleration Detected: AMD Radeon Graphics`
`[+] ROCm/CUDA initialized successfully.`
