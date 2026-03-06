"""
Configuration parameters for SmallLLM model architecture and training.
"""

from dataclasses import dataclass


# pylint: disable=too-many-instance-attributes
@dataclass
class SmallLLMConfig:
    """
    Master configuration for model architecture and all 3 training stages.
    """

    # --- MODEL ARCHITECTURE (Tiny MoE Transformer) ---
    d_model: int = 256  # Embedding dimension
    n_layers: int = 6  # Transformer blocks
    n_heads: int = 4  # Attention heads
    n_kv_heads: int = 2  # GQA: num_heads // 2
    context_length: int = 512  # Max sequence length
    vocab_size: int = 8192  # Tokenizer vocabulary size
    norm_eps: float = 1e-5  # RMSNorm epsilon

    # --- MoE (Mixture of Experts) ---
    n_experts: int = 8  # Total experts per MoE layer
    top_k: int = 2  # Number of active experts per token
    moe_aux_loss_coeff: float = 0.01  # Expert balance loss weight
    capacity_factor: float = 1.25  # MoE token capacity multiplier
    use_shared_expert: bool = True  # DeepSeek-V2 style shared path

    # --- TRAINING: Stage 1 (Pretraining) ---
    pretrain_lr: float = 3e-4
    pretrain_batch_size: int = 4
    pretrain_grad_accum: int = 16  # Effective batch size = 64
    pretrain_steps: int = 10000
    pretrain_warmup_pct: float = 0.02
    pretrain_weight_decay: float = 0.1

    # --- TRAINING: Stage 2 (SFT) ---
    sft_lr: float = 5e-5
    sft_batch_size: int = 4
    sft_epochs: int = 2

    # --- TRAINING: Stage 3 (GRPO) ---
    grpo_lr: float = 1e-6
    grpo_group_size: int = 4  # Number of samples (G) per prompt
    grpo_eps: float = 0.2  # PPO-style clipping epsilon
    grpo_kl_beta: float = 0.01  # KL divergence penalty coefficient
    grpo_steps: int = 1500
    grpo_grad_accum: int = 8

    # --- HARDWARE & DEBUG ---
    fast_test: bool = False  # Set to True for tiny model & quick sanity check
    mixed_precision: str = "bf16"  # "fp16" or "bf16"
    seed: int = 42

    def __post_init__(self):
        """Overrides parameters for quick validation in fast_test mode."""
        if self.fast_test:
            self.d_model = 64
            self.n_layers = 2
            self.n_heads = 2
            self.n_kv_heads = 1
            self.pretrain_steps = 50
            self.grpo_steps = 20
            self.vocab_size = 500
            print(
                "[!] FAST_TEST mode enabled: architecture and steps significantly reduced."
            )
