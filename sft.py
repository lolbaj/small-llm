"""
Stage 2: Supervised Fine-Tuning (SFT) for SmallLLM.
"""

import os
import torch
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
    Creates a mask for the assistant tokens in the target sequence.
    User tokens (between <|user|> and <|assistant|>) are masked out (set to -100).
    """
    user_id = tokenizer.get_special_token_id("<|user|>")
    asst_id = tokenizer.get_special_token_id("<|assistant|>")

    targets = x_ids.clone()

    for i in range(x_ids.shape[0]):
        is_user_turn = False
        for j in range(x_ids.shape[1]):
            token = x_ids[i, j].item()

            if token == user_id:
                is_user_turn = True
            elif token == asst_id:
                is_user_turn = False

            if is_user_turn or token == tokenizer.get_special_token_id("<|pad|>"):
                targets[i, j] = -100

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
    checkpoint = torch.load(pretrain_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
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
        for x_ids, _ in dataloader:
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
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

            step += 1

        path = f"checkpoints/sft/sft_epoch_{epoch}.pt"
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config},
            path,
        )
        print(f"[*] SFT Checkpoint saved: {path}")


if __name__ == "__main__":
    train_stage2(SmallLLMConfig(fast_test=True), "checkpoints/pretrain_step_0.pt")
