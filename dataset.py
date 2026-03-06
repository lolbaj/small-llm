"""
Streaming dataset wrappers for TinyStories and instruction fine-tuning.
"""

from typing import Optional, Callable, Dict
import torch
from torch.utils.data import IterableDataset, DataLoader

# pylint: disable=import-error
from datasets import load_dataset
from config import SmallLLMConfig


# pylint: disable=abstract-method, too-many-arguments, too-many-positional-arguments, too-few-public-methods
class StreamingDatasetWrapper(IterableDataset):
    """
    Wraps HuggingFace streaming datasets for training.
    Optimized for low RAM environments by never loading the full dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: any,
        split: str = "train",
        context_len: int = 512,
        template_func: Optional[Callable] = None,
    ):
        super().__init__()
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.template_func = template_func

    def __iter__(self):
        for item in self.dataset:
            # Apply formatting (e.g., chat template)
            if self.template_func:
                text = self.template_func(item)
            else:
                # Default for pre-training (e.g., TinyStories)
                text = item.get("text", "")

            # Encode text
            ids = self.tokenizer.encode(text)

            # Chunking/Padding
            for idx in range(0, len(ids), self.context_len):
                chunk = ids[idx : idx + self.context_len]
                # Need at least 2 tokens for next-token prediction
                if len(chunk) < 2:
                    continue

                # Padding
                if len(chunk) < self.context_len:
                    pad_id = self.tokenizer.get_special_token_id("<|pad|>")
                    chunk = chunk + [pad_id] * (self.context_len - len(chunk))

                # Prepare inputs (x) and targets (y)
                # For causal language modeling, target is x shifted by 1
                x_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                y_ids = torch.tensor(chunk[1:], dtype=torch.long)

                yield x_ids, y_ids


def pretrain_template(item: Dict) -> str:
    """Pre-training template: simple causal text."""
    return item.get("text", "")


def sft_template(item: Dict) -> str:
    """SFT template: Instruction/Response in chat format."""
    instr = item.get("instruction", "")
    context = item.get("context", "")
    resp = item.get("response", "")

    prompt = f"<|user|> {instr} {context} <|assistant|> "
    return f"{prompt}{resp} <|end|>"


def get_dataloader(
    config: SmallLLMConfig,
    tokenizer: any,
    dataset_name: str,
    split: str = "train",
    stage: str = "pretrain",
) -> DataLoader:
    """Creates a DataLoader for the specified stage."""
    template_map = {
        "pretrain": pretrain_template,
        "sft": sft_template,
    }

    ds_obj = StreamingDatasetWrapper(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split=split,
        context_len=config.context_length + 1,  # +1 because we split into x, y
        template_func=template_map.get(stage),
    )

    return DataLoader(
        ds_obj,
        batch_size=(
            config.pretrain_batch_size if stage == "pretrain" else config.sft_batch_size
        ),
        num_workers=2,
        pin_memory=False,
    )


if __name__ == "__main__":
    from tokenizer import CustomBPETokenizer

    test_cfg = SmallLLMConfig(fast_test=True)
    test_tok = CustomBPETokenizer()
    test_tok.load("test_tokenizer")

    # Test streaming TinyStories
    dl_obj = get_dataloader(
        test_cfg, test_tok, "roneneldan/TinyStories", split="train", stage="pretrain"
    )

    for batch_idx, (test_x, test_y) in enumerate(dl_obj):
        print(f"Batch {batch_idx}: x shape {test_x.shape}, y shape {test_y.shape}")
        if batch_idx >= 2:
            break
