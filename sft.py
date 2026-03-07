"""
Stage 2: Supervised Fine-Tuning (SFT) for SmallLLM.
"""

import os
import torch
from tqdm import tqdm
from torch import amp
from torch.optim import AdamW
from config import SmallLLMConfig
from model import MoETransformer
from tokenizer import CustomBPETokenizer
from dataset import get_dataloader
from device import get_device
from utils import estimate_ram_usage, get_autocast_context


def mask_user_tokens(x_ids, tokenizer):
    """
    Vectorized mask for assistant tokens.
    User tokens (between <|user|> and <|assistant|>) are masked out (-100).
    """
    user_id = tokenizer.get_special_token_id("<|user|>")
    asst_id = tokenizer.get_special_token_id("<|assistant|>")
    pad_id = tokenizer.get_special_token_id("<|pad|>")

    targets = x_ids.clone()

    # Vectorized logic:
    # 1. Find where user/assistant tags are
    is_user_tag = x_ids == user_id
    is_asst_tag = x_ids == asst_id

    # 2. Compute "turn" state using cumulative sums
    # Every time we see a user tag, the state increases.
    # Every time we see an asst tag, it effectively "decreases" if we logic it right.
    # Logic: tokens after <|user|> but before <|assistant|> are user tokens.
    user_cum = is_user_tag.cumsum(dim=1)
    asst_cum = is_asst_tag.cumsum(dim=1)

    # In user turn if we've seen more user tags than assistant tags
    # OR if we are ON the assistant tag itself (since we only train on RESPONSE tokens)
    mask = (user_cum > asst_cum) | is_asst_tag | (x_ids == pad_id)

    targets[mask] = -100
    return targets


# pylint: disable=too-many-locals
def train_stage2(config: SmallLLMConfig, pretrain_path: str):
    """Stage 2: Supervised Fine-Tuning loop."""
    if not estimate_ram_usage(config):
        return

    device = get_device()
    os.makedirs("checkpoints/sft", exist_ok=True)

    tokenizer = CustomBPETokenizer(vocab_size=config.vocab_size)
    tokenizer.load("tokenizer")

    model = MoETransformer(config).to(device)
    checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    print(f"[*] Loaded pre-trained checkpoint from {pretrain_path}")

    optimizer = AdamW(model.parameters(), lr=config.sft_lr)
    scaler = amp.GradScaler(enabled=config.mixed_precision == "fp16")

    dataloader = get_dataloader(
        config, tokenizer, "databricks/databricks-dolly-15k", split="train", stage="sft"
    )

    print(f"[*] Starting SFT: {config.sft_epochs} epochs...")

    model.train()
    for epoch in range(config.sft_epochs):
        step = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for x_ids, _ in pbar:
            x_ids = x_ids.to(device)
            targets = mask_user_tokens(x_ids, tokenizer).to(device)

            input_ids = x_ids[:, :-1]
            target_ids = targets[:, 1:]

            with get_autocast_context(device, config):
                _, loss, _, _, _ = model(input_ids, targets=target_ids)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if step % 50 == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            step += 1

        path = f"checkpoints/sft/sft_epoch_{epoch}.pt"
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config},
            path,
        )
        print(f"[*] SFT Checkpoint saved: {path}")


if __name__ == "__main__":
    # Use the latest checkpoint from pre-training test
    checkpoint_path = "checkpoints/pretrain_step_50.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/pretrain_step_0.pt"
        
    if os.path.exists(checkpoint_path):
        train_stage2(SmallLLMConfig(fast_test=True), checkpoint_path)
    else:
        print(f"[!] Pre-trained checkpoint not found at {checkpoint_path}. Run train.py first.")
