# 🧸 Small MoE LLM (with GRPO Alignment)

Build and train a tiny Mixture of Experts (MoE) LLM from scratch, aligned using Group Relative Policy Optimization (GRPO), and optimized for AMD Ryzen 5600GT (Vega 7 iGPU).

## 🚀 Overview

- **Architecture:** Sparse MoE Transformer (20M–60M params)
- **Features:** RoPE, GQA, SwiGLU, RMSNorm, 8 experts (top-2 routing)
- **Training Stages:**
  1. **Pre-training:** Causal LM on TinyStories.
  2. **SFT:** Instruction fine-tuning on Dolly-15k.
  3. **GRPO Alignment:** DeepSeek-R1 style RLHF from scratch (Format/Length/Coherence rewards).
- **Optimization:** Optimized for Arch Linux + ROCm (Vega 7) with 16GB shared RAM.

---

## 🛠️ Environment Setup (Arch Linux)

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure ROCm for Ryzen 5600GT (Vega 7)
The Ryzen 5600GT iGPU (GFX90C) is not officially supported by ROCm but works with an override.
Ensure you have `rocm-core`, `hip-runtime-amd`, and `pytorch-rocm` (from AUR or official) installed.

**Critical Environment Variable:**
```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0
```
*This is handled automatically by `device.py` during execution.*

---

## 📈 Training Pipeline

### Phase 0: Fast Test (Pipeline Validation)
Ensure everything works in < 5 minutes on CPU:
```bash
# Set fast_test=True in config.py, then:
python train.py
python sft.py
python grpo.py
```

### Phase 1: Pre-training (train.py)
```bash
python train.py
```
Trains on `roneneldan/TinyStories`. Logs loss and tokens/sec to `training_log.csv`.

### Phase 2: Supervised Fine-Tuning (sft.py)
```bash
python sft.py
```
Loads the best pre-trained checkpoint and tunes on `databricks/databricks-dolly-15k`.

### Phase 3: GRPO Alignment (grpo.py)
```bash
python grpo.py
```
Implements Group Relative Policy Optimization (GRPO) to teach the model step-by-step reasoning using a `<|think|>` template.

---

## 📦 Export & Inference

### Export to .toon and Safetensors
```bash
python export.py
```
Quantizes the final model to **INT8** and saves it in the custom `.toon` binary format.

### Run Inference (Chat)
```bash
python inference.py
```
Starts a terminal-based chat loop.
- Use `/reset` to clear context.
- Use `/quit` to exit.
- Supports streaming and verbose mode (shows expert activation).

---

## 🔧 Troubleshooting

- **OOM (Out of Memory):** 
  - Reduce `pretrain_batch_size` or `context_length` in `config.py`.
  - Ensure `HSA_OVERRIDE_GFX_VERSION=9.0.0` is set if using iGPU.
- **Expert Collapse:** 
  - If `expert utilization %` is low, increase `moe_aux_loss_coeff`.
- **ROCm Initialization Failed:**
  - Verify your user is in the `render` and `video` groups: `sudo usermod -aG render,video $USER`.
  - Re-login for changes to take effect.
