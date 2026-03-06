"""
Stage 1: Pre-training loop for SmallLLM.
"""

import csv
import os
import time
import torch
from torch import nn, amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import SmallLLMConfig
from model import MoETransformer
from tokenizer import CustomBPETokenizer
from dataset import get_dataloader
from device import get_device
from utils import estimate_ram_usage, get_autocast_context


# pylint: disable=too-many-locals, too-many-statements, import-outside-toplevel, import-error
def train_stage1(config: SmallLLMConfig):
    """Stage 1: Pre-training loop."""
    # RAM Check
    if not estimate_ram_usage(config):
        return

    device = get_device()
    os.makedirs("checkpoints", exist_ok=True)

    # Initialize Tokenizer and Model
    tokenizer = CustomBPETokenizer(vocab_size=config.vocab_size)
    if os.path.exists("tokenizer/tokenizer.json"):
        tokenizer.load("tokenizer")
    else:
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True).take(
            1000
        )
        tokenizer.train([item["text"] for item in ds], save_dir="tokenizer")

    model = MoETransformer(config).to(device)

    try:
        model = torch.compile(model)
        print("[+] torch.compile() enabled.")
    except RuntimeError as e:
        print(f"[-] torch.compile() not available: {e}")

    optimizer = AdamW(
        model.parameters(),
        lr=config.pretrain_lr,
        betas=(0.9, 0.95),
        weight_decay=config.pretrain_weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.pretrain_steps, eta_min=config.pretrain_lr * 0.1
    )
    scaler = amp.GradScaler(enabled=config.mixed_precision == "fp16")
    dataloader = get_dataloader(
        config, tokenizer, "roneneldan/TinyStories", split="train", stage="pretrain"
    )

    with open("training_log.csv", "w", newline="", encoding="utf-8") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            ["step", "loss", "aux_loss", "util_pct", "lr", "tokens_sec"]
        )

        print(f"[*] Starting Pre-training: {config.pretrain_steps} steps...")

        model.train()
        step = 0
        total_tokens = 0
        start_time = time.time()

        for x_ids, y_ids in dataloader:
            if step >= config.pretrain_steps:
                break

            x_ids, y_ids = x_ids.to(device), y_ids.to(device)

            with get_autocast_context(device, config):
                _, loss, aux_loss, util, _ = model(x_ids, targets=y_ids)
                loss_val = loss / config.pretrain_grad_accum

            scaler.scale(loss_val).backward()
            total_tokens += x_ids.numel()

            if (step + 1) % config.pretrain_grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                if step % 100 == 0:
                    elapsed = time.time() - start_time
                    tps = total_tokens / elapsed
                    cur_loss = loss_val.item() * config.pretrain_grad_accum
                    print(
                        f"Step {step}/{config.pretrain_steps} | Loss: {cur_loss:.4f} | "
                        f"Aux: {aux_loss.item():.4f} | Util: {util:.1f}% | T/s: {tps:.2f}"
                    )
                    log_writer.writerow(
                        [
                            step,
                            cur_loss,
                            aux_loss.item(),
                            util,
                            scheduler.get_last_lr()[0],
                            tps,
                        ]
                    )
                    log_file.flush()

                if step % 500 == 0 or step == config.pretrain_steps - 1:
                    path = f"checkpoints/pretrain_step_{step}.pt"
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": config,
                        },
                        path,
                    )
                    print(f"[*] Checkpoint saved: {path}")

            step += 1


if __name__ == "__main__":
    train_stage1(SmallLLMConfig(fast_test=True))
