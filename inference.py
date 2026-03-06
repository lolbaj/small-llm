"""
Inference engine and terminal chat loop for SmallLLM.
"""

import os
import sys
import torch
import torch.nn.functional as F
from model import MoETransformer
from tokenizer import CustomBPETokenizer
from device import get_device


# pylint: disable=too-few-public-methods
class SmallLLMInference:
    """Efficient inference engine with KV cache and sampling."""

    def __init__(self, model_path: str, device=None):
        self.device = device if device else get_device()
        self.tokenizer = CustomBPETokenizer()
        self.tokenizer.load("tokenizer")

        # Load from checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint["config"]
        self.model = MoETransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temp: float = 0.7,
        verbose: bool = False,
    ):
        """Generates text from a prompt using KV cache."""

        # Prepare input
        input_ids = torch.tensor(
            [self.tokenizer.encode(f"<|user|> {prompt} <|assistant|> ")],
            device=self.device,
        )

        print("\n<|assistant|>", flush=True)

        kv_caches = None
        curr_ids = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass:
                # On first step, we pass full prompt and get initial cache.
                # On subsequent steps, we pass ONLY the last token and update cache.
                logits, _, _, util, kv_caches = self.model(
                    curr_ids, kv_caches=kv_caches
                )

                next_token_logits = logits[:, -1, :]
                next_token_logits = next_token_logits / (temp + 1e-8)

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update curr_ids to be ONLY the new token for the next iteration
                curr_ids = next_token

                # Decode and print
                token_text = self.tokenizer.decode(next_token[0])
                sys.stdout.write(token_text)

                if verbose:
                    sys.stdout.write(f" [Util: {util.item():.1f}%]")

                sys.stdout.flush()

                if next_token.item() == self.tokenizer.get_special_token_id("<|end|>"):
                    break

        print("\n")


def chat_loop():
    """Terminal chat loop."""
    print("=" * 50)
    print(" Small MoE LLM - Chat Interface ")
    print(" Commands: /reset, /quit ")
    print("=" * 50)

    # Assumes SFT or Aligned model exists
    model_path = "checkpoints/sft/sft_epoch_0.pt"
    if not os.path.exists(model_path):
        model_path = "checkpoints/pretrain_step_0.pt"
        if not os.path.exists(model_path):
            print("[-] No checkpoints found. Please train the model first.")
            return

    engine = SmallLLMInference(model_path)

    while True:
        try:
            user_input = input("\n<|user|> ")
            if user_input.strip() == "/quit":
                break
            if user_input.strip() == "/reset":
                continue

            engine.generate(user_input, verbose=True)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    chat_loop()
