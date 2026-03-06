You are a senior ML engineer and LLM architect. Build a complete Python project
that trains a tiny MoE (Mixture of Experts) LLM from scratch, aligned using GRPO
(DeepSeek-R1-style), and saved in the best portable quantized format for CPU/iGPU
inference — all optimized for constrained consumer hardware.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## HARDWARE & OS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- CPU: AMD Ryzen 5 5600GT (6 cores, 12 threads)
- iGPU: AMD Radeon Vega 7 (integrated, shares system RAM, gfx90c)
- RAM: 16GB DDR4 shared between CPU and iGPU (~12GB usable for training)
- OS: Arch Linux (rolling release, latest kernel)
- ROCm optional: use PyTorch ROCm build if available, with HSA_OVERRIDE_GFX_VERSION=9.0.0
  to unlock Vega 7. Gracefully fall back to CPU if ROCm init fails.
- All setup uses Python venv (NOT conda). System Python 3.11+.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## CHECKPOINT / MODEL FORMAT — .toon
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Research whether ".toon" is a real, documented LLM serialization format.
2. If it is real and documented, implement it exactly and cite its spec.
3. If it does NOT exist or is undocumented, do NOT invent it. Instead:
   - Use GGUF as the primary export format (industry standard, used by llama.cpp).
   - Implement a lightweight custom binary format called ".toon" as a thin wrapper
     around safetensors metadata + quantized weights, and document its spec clearly
     in a FORMAT_SPEC.md file so it is reproducible and human-readable.
   - The .toon format must store: model config JSON, tokenizer vocab, INT8/INT4
     quantized weights in row-major order, and a magic header b"TOON" + version byte.
   - Provide both a save_toon() and load_toon() function in a dedicated format.py.
   - Also export to GGUF via llama.cpp's convert script for real-world compatibility.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MODEL ARCHITECTURE — Tiny MoE Transformer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Build a GPT-style decoder-only transformer with Sparse Mixture of Experts FFN layers:

- Total parameters: 20M–60M (sparse activation means only ~30% fire per token)
- Embedding dim (d_model): 256 or 384
- Transformer blocks: 6
- Attention heads: 4 or 6
- Context length: 512 tokens
- Positional encoding: RoPE (Rotary Position Embeddings)
- Activation: SwiGLU (modern standard, used in LLaMA/Mistral)
- Weight tying: share input embedding and output projection weights

### MoE FFN Layer (replace standard FFN with this):
- Number of experts: 8
- Top-K routing: K=2 (each token activates 2 experts per layer)
- Router: linear layer → softmax → top-k selection
- Load balancing loss: auxiliary loss term (coefficient 0.01) to prevent
  expert collapse — add to main loss during training
- Expert capacity buffer: implement capacity factor of 1.25 to handle
  token overflow without dropping gradients
- Use a shared "dense" expert in addition to routed experts (like DeepSeek-V2)
  so every token always passes through at least one non-gated path

### Attention:
- Grouped Query Attention (GQA) with num_kv_heads = num_heads // 2
- Causal mask (decoder-only)
- No Flash Attention (not reliably supported on Vega iGPU via ROCm)

### Normalization:
- RMSNorm instead of LayerNorm (faster, used in LLaMA/Mistral/DeepSeek)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## TRAINING PIPELINE — 3 Stages
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Stage 1 — Pretraining (train.py)
Objective: Causal Language Modeling (next-token prediction)
- Dataset: TinyStories or OpenWebText-10k via HuggingFace datasets (streamed,
  never fully loaded into RAM)
- Chat template: "<|user|> {text} <|assistant|> {response} <|end|>"
- Batch size: 4, gradient accumulation: 16 steps (effective batch = 64)
- Optimizer: AdamW (lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
- LR schedule: cosine decay with linear warmup over first 2% of steps
- Gradient clipping: 1.0
- Mixed precision: bfloat16 via torch.amp
- DataLoader: num_workers=2, pin_memory=False
- Checkpoint: save every 500 steps, keep best 3 by val loss, fully resumable
- torch.compile(): attempt with try/except fallback
- MoE auxiliary load balancing loss added to CLM loss every step
- Log: loss, aux_loss, lr, tokens/sec, expert utilization % → training_log.csv

### Stage 2 — Supervised Fine-Tuning / SFT (sft.py)
Objective: teach the model to follow instructions in chat format
- Dataset: a small instruction-following dataset (e.g. databricks-dolly-15k
  or OpenAssistant/oasst1 — whichever is smaller and easier to stream)
- Same optimizer and precision settings as Stage 1
- Only train on assistant response tokens (mask user tokens in loss)
- Run for 1–2 epochs maximum
- Save SFT checkpoint separately

### Stage 3 — GRPO Alignment (grpo.py)
Objective: align model responses using Group Relative Policy Optimization
(as introduced in DeepSeek-R1 / used instead of PPO to avoid needing a
separate critic/value network — critical for low-RAM hardware)

Implement GRPO from scratch as follows:
1. For each prompt, sample G=4 responses from the current policy model
   (G is the group size — keep low for RAM reasons)
2. Score each response with a reward function (implement 3 reward signals
   that can be combined via weighted sum):
   a. Format reward: +1.0 if response follows "<think>...</think><answer>...</answer>" 
      template (DeepSeek-R1 style chain-of-thought), 0.0 otherwise
   b. Length reward: soft reward penalizing responses shorter than 20 tokens
      or longer than 300 tokens
   c. Coherence reward: simple n-gram overlap score between prompt and response
      as a proxy for relevance (no external reward model needed)
3. Compute group-relative advantage:
   A_i = (r_i - mean(r)) / (std(r) + 1e-8)
   where r is the vector of rewards for the G responses in the group
4. GRPO policy loss:
   L = -mean[ min( ratio * A, clip(ratio, 1-eps, 1+eps) * A ) ]
   where ratio = exp(log_prob_policy - log_prob_ref), eps=0.2
5. KL penalty: add β * KL(policy || reference) to loss, β=0.01
   Use the frozen SFT checkpoint as the reference model
   Load reference model in inference mode with torch.no_grad() to save RAM
6. Gradient accumulation: 8 steps
7. GRPO training: 1000–2000 steps maximum (hardware limit)
8. Save final aligned checkpoint

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## TOKENIZER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Train a custom BPE tokenizer using HuggingFace tokenizers library
- Vocabulary size: 8000–12000 tokens (small = faster on CPU)
- Special tokens: <|user|>, <|assistant|>, <|end|>, <|think|>, <|/think|>,
  <|answer|>, <|/answer|> (for GRPO chain-of-thought template)
- Save/load from tokenizer/ directory
- tokenizer.py handles training and loading

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MODEL EXPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export.py must:
1. Load the final GRPO-aligned checkpoint
2. Quantize weights to INT8 using torch.quantization.quantize_dynamic
   (CPU-compatible, no CUDA required)
3. Save in .toon format via format.py's save_toon()
4. Also save as safetensors (float32 and bfloat16 variants)
5. Print final model size in MB for each format
6. Provide instructions for converting to GGUF using llama.cpp's
   convert_hf_to_gguf.py script (include the exact CLI command)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## INFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
inference.py must:
- Load from .toon or safetensors checkpoint
- Implement KV cache for efficient autoregressive generation
- Support: greedy, top-p (nucleus), top-k, and temperature sampling
- Use the <|think|>...</|think|> chain-of-thought format from GRPO
- Stream tokens to terminal character by character
- Print which experts were activated per layer (--verbose flag)
- Terminal chat loop with /reset, /save, /quit commands

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROJECT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
small-llm/
├── config.py          # Master config dataclass for all 3 stages + MoE params
├── model.py           # MoE Transformer (RoPE, GQA, SwiGLU, RMSNorm, MoE FFN)
├── tokenizer.py       # BPE tokenizer training & loading
├── dataset.py         # Streaming dataset, chat template formatting
├── train.py           # Stage 1: Pretraining loop
├── sft.py             # Stage 2: Supervised fine-tuning
├── grpo.py            # Stage 3: GRPO alignment from scratch
├── format.py          # .toon format: save_toon() / load_toon() + spec
├── export.py          # INT8 quantization + multi-format export
├── inference.py       # KV cache chat loop with sampling + expert visibility
├── device.py          # ROCm/CPU detection + HSA env var guidance
├── utils.py           # Checkpointing, logging, LR schedule, RAM estimator
├── FORMAT_SPEC.md     # Human-readable .toon binary format specification
├── README.md          # Full Arch Linux setup guide, all 3 training stages
└── requirements.txt   # All dependencies pinned

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## ADDITIONAL REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- config.py: fast_test: bool = False flag → tiny model + 50 steps, completes
  in under 5 minutes on CPU for pipeline validation
- utils.py: estimate_ram_usage() → print projected RAM before each stage,
  warn if > 10GB, abort if > 13GB
- Every file fully commented explaining what each component does and WHY
  (architecture decisions, math behind GRPO, why RMSNorm over LayerNorm, etc.)
- README.md must cover:
  - venv creation and pip install (CPU PyTorch default)
  - Optional ROCm PyTorch install for Arch (with HSA_OVERRIDE_GFX_VERSION=9.0.0)
  - How to run all 3 stages in order
  - How to run fast_test sanity check
  - How to export and run inference
  - Troubleshooting: OOM errors, ROCm init failures, expert collapse symptoms

Generate ALL files completely. Do not truncate, summarize, or skip any file.
Generate in this order: device.py → config.py → model.py → tokenizer.py →
dataset.py → train.py → sft.py → grpo.py → format.py → export.py →
inference.py → utils.py → FORMAT_SPEC.md → README.md → requirements.txt
