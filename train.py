"""
Stage 1: Pre-training loop for SmallLLM.
"""

import csv
import os
import time
import torch
from tqdm import tqdm
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

    # torch.compile() fallback for constrained environments
    if os.environ.get("DISABLE_COMPILE", "0") == "1":
        print("[*] torch.compile() disabled via environment variable.")
    else:
        try:
            # We use 'aot_eager' or 'stock' backends which are safer on CPU
            model = torch.compile(model)
            print("[+] torch.compile() enabled.")
        except (RuntimeError, ImportError, Exception) as e:
            print(f"[-] torch.compile() failed or not supported: {e}")
            print("[!] Continuing with standard execution mode.")

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

        print(f"[*] Starting Pre-training: {config.pretrain_steps} update steps...")

        model.train()
        update_step = 0
        global_step = 0
        total_tokens = 0
        start_time = time.time()

        pbar = tqdm(total=config.pretrain_steps, desc="Pre-training")
        for x_ids, y_ids in dataloader:
            if update_step >= config.pretrain_steps:
                break

            x_ids, y_ids = x_ids.to(device), y_ids.to(device)

            with get_autocast_context(device, config):
                _, loss, aux_loss, util, _ = model(x_ids, targets=y_ids)
                loss_val = loss / config.pretrain_grad_accum

            scaler.scale(loss_val).backward()
            total_tokens += x_ids.numel()

            if (global_step + 1) % config.pretrain_grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                update_step += 1
                pbar.update(1)

                # More frequent logging, especially for fast_test
                log_interval = 10 if config.fast_test else 100
                if update_step < 10 or update_step % log_interval == 0:
                    elapsed = time.time() - start_time
                    tps = total_tokens / (elapsed + 1e-8)
                    cur_loss = loss_val.item() * config.pretrain_grad_accum
                    util_val = util.item()
                    pbar.set_postfix({
                        "Loss": f"{cur_loss:.4f}", 
                        "Aux": f"{aux_loss.item():.4f}", 
                        "Util": f"{util_val:.1f}%", 
                        "T/s": f"{tps:.2f}"
                    })
                    log_writer.writerow(
                        [
                            update_step,
                            cur_loss,
                            aux_loss.item(),
                            util_val,
                            scheduler.get_last_lr()[0],
                            tps,
                        ]
                    )
                    log_file.flush()

                if update_step % 500 == 0 or update_step == config.pretrain_steps:
                    path = f"checkpoints/pretrain_step_{update_step}.pt"
                    torch.save(
                        {
                            "step": update_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": config,
                        },
                        path,
                    )
                    # pbar.write instead of print so we don't break the progress bar
                    pbar.write(f"[*] Checkpoint saved: {path}")

            global_step += 1
            
        pbar.close()


if __name__ == "__main__":
    train_stage1(SmallLLMConfig(fast_test=True))
