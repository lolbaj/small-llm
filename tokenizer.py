"""
Custom BPE Tokenizer training and loading using HuggingFace tokenizers.
"""

import os
from typing import List, Union
import torch

# pylint: disable=import-error
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders


class CustomBPETokenizer:
    """
    Custom BPE Tokenizer for SmallLLM.
    Handles special tokens for chat and GRPO chain-of-thought logic.
    """

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.special_tokens = [
            "<|pad|>",
            "<|unk|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>",
            "<|think|>",
            "</|think|>",
            "<|answer|>",
            "</|answer|>",
        ]

    def train(self, texts: List[str], save_dir: str = "tokenizer"):
        """Trains a new BPE tokenizer on a list of texts."""
        os.makedirs(save_dir, exist_ok=True)

        # Initialize BPE model
        model = models.BPE(unk_token="<|unk|>")
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2,
            show_progress=True,
        )

        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        # Train on provided text
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Post-processing: ensure special tokens are correctly handled
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        # Save
        self.tokenizer = tokenizer
        self.tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
        print(f"[*] Tokenizer trained and saved to {save_dir}")

    def load(self, save_dir: str = "tokenizer"):
        """Loads a pre-trained tokenizer."""
        path = os.path.join(save_dir, "tokenizer.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Tokenizer not found at {path}. Run train() first."
            )

        self.tokenizer = Tokenizer.from_file(path)
        print(f"[*] Tokenizer loaded from {save_dir}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes text to token IDs."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded or trained.")
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: Union[List[int], torch.Tensor]) -> str:
        """Decodes token IDs back to text."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded or trained.")
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def get_vocab_size(self) -> int:
        """Returns the total vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_special_token_id(self, token: str) -> int:
        """Returns the token ID for a specific special token."""
        return self.tokenizer.token_to_id(token)


if __name__ == "__main__":
    # Quick test training
    sample_texts = [
        "This is a test of the small-llm tokenizer.",
        "<|user|> Hello! <|assistant|> Hi there! <|end|>",
        "<|think|> I need to calculate 2+2 </|think|> <|answer|> 4 </|answer|>",
    ]
    t = CustomBPETokenizer(vocab_size=500)
    t.train(sample_texts, save_dir="test_tokenizer")

    encoded = t.encode("<|user|> Hello world! <|end|>")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {t.decode(encoded)}")
