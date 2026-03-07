"""
Stage 3: Group Relative Policy Optimization (GRPO) alignment for SmallLLM.
"""

import random
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import AdamW
from config import SmallLLMConfig
from model import MoETransformer
from tokenizer import CustomBPETokenizer
from device import get_device
from utils import estimate_ram_usage


class GRPORewards:
    """Implement 3 reward signals for GRPO alignment."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def format_reward(self, response_text: str) -> float:
        """DeepSeek-R1 style CoT template: <|think|>...<|/think|><|answer|>...<|/answer|>"""
        if (
            "<|think|>" in response_text
            and "</|think|>" in response_text
            and "<|answer|>" in response_text
            and "</|answer|>" in response_text
        ):
            # Check for correct order
            if (
                response_text.find("<|think|>")
                < response_text.find("</|think|>")
                < response_text.find("<|answer|>")
                < response_text.find("</|answer|>")
            ):
                return 1.0
        return 0.0

    def length_reward(self, response_text: str) -> float:
        """Reward responses that provide balanced reasoning lengths."""
        length = len(response_text.split())
        if 20 <= length <= 300:
            return 1.0
        if length < 20:
            return 0.1 * (length / 20)
        # length > 300
        return 0.1 * (300 / length)

    def coherence_reward(self, prompt_text: str, response_text: str) -> float:
        """Simple n-gram overlap score as a proxy for relevance/coherence."""
        prompt_words = set(prompt_text.lower().split())
        response_words = set(response_text.lower().split())
        if not prompt_words:
            return 0.5
        overlap = len(prompt_words.intersection(response_words))
        # Offset by 0.3 base relevance
        return min(1.0, overlap / (len(prompt_words) + 1e-8) + 0.3)

    def get_total_reward(self, prompt: str, response: str) -> float:
        """Weighted sum of all rewards."""
        r1 = self.format_reward(response)
        r2 = self.length_reward(response)
        r3 = self.coherence_reward(prompt, response)
        return 0.5 * r1 + 0.2 * r2 + 0.3 * r3


def sample_responses(model, prompt_ids, group_size, tokenizer, max_new_tokens=256):
    """Samples G responses for a single prompt using current policy model with KV cache."""
    model.eval()
    responses_ids = []
    # Old logprobs from sampling phase (no-grad) are not used for training
    # but kept for legacy if needed. We will re-evaluate them with grads later.
    
    for _ in range(group_size):
        with torch.no_grad():
            curr_ids = prompt_ids.clone()
            kv_caches = None
            resp_ids = []

            for i in range(max_new_tokens):
                # Use KV cache for efficiency
                # On first step, pass full prompt. Subsequently, pass only the last token.
                input_ids = curr_ids if i == 0 else curr_ids[:, -1:]
                logits, _, _, _, kv_caches = model(input_ids, kv_caches=kv_caches)
                
                next_token_logits = logits[:, -1, :] / 0.8
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                resp_ids.append(next_token)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)

                if next_token.item() == tokenizer.get_special_token_id("<|end|>"):
                    break

            responses_ids.append(torch.cat(resp_ids, dim=1))

    model.train()
    return responses_ids


# pylint: disable=too-many-locals
def train_stage3_grpo(config: SmallLLMConfig, sft_path: str):
    """Stage 3: GRPO Alignment loop."""
    if not estimate_ram_usage(config):
        return

    device = get_device()
    tokenizer = CustomBPETokenizer(vocab_size=config.vocab_size)
    tokenizer.load("tokenizer")

    reward_gen = GRPORewards(tokenizer)

    policy_model = MoETransformer(config).to(device)
    checkpoint = torch.load(sft_path, map_location=device, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    policy_model.load_state_dict(state_dict)

    ref_model = MoETransformer(config).to(device)
    ref_model.load_state_dict(state_dict)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(policy_model.parameters(), lr=config.grpo_lr)

    prompts = [
        "Explain the benefit of Sparse MoE in transformers.",
        "How do I optimize Python code for Ryzen processors?",
        "What is Group Relative Policy Optimization (GRPO)?",
        "Explain quantum entanglement like I'm 5.",
    ]

    print(f"[*] Starting GRPO Alignment: {config.grpo_steps} steps...")

    pbar = tqdm(range(config.grpo_steps), desc="GRPO Alignment")
    for step in pbar:
        prompt = random.choice(prompts)
        prompt_ids = torch.tensor(
            [tokenizer.encode(f"<|user|> {prompt} <|assistant|> ")], device=device
        )

        responses = sample_responses(
            policy_model, prompt_ids, config.grpo_group_size, tokenizer
        )

        rewards = []
        for resp_ids in responses:
            resp_text = tokenizer.decode(resp_ids[0])
            rewards.append(reward_gen.get_total_reward(prompt, resp_text))

        rewards_tensor = torch.tensor(rewards, device=device)

        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_r) / std_r

        # Optimization step: we need to re-evaluate logprobs with gradients
        optimizer.zero_grad()
        
        step_loss = 0
        for i in range(config.grpo_group_size):
            input_ids = torch.cat([prompt_ids, responses[i]], dim=1)
            resp_len = responses[i].shape[1]
            
            # Policy logprobs (with gradients)
            policy_logits, _, _, _, _ = policy_model(input_ids[:, :-1])
            policy_logprobs_all = torch.log_softmax(policy_logits, dim=-1)
            policy_logprobs = (
                policy_logprobs_all[:, -resp_len:, :]
                .gather(-1, responses[i].unsqueeze(-1))
                .squeeze(-1)
            )

            # Reference logprobs (no gradients)
            with torch.no_grad():
                ref_logits, _, _, _, _ = ref_model(input_ids[:, :-1])
                ref_logprobs_all = torch.log_softmax(ref_logits, dim=-1)
                ref_logprobs = (
                    ref_logprobs_all[:, -resp_len:, :]
                    .gather(-1, responses[i].unsqueeze(-1))
                    .squeeze(-1)
                )

            ratio = torch.exp(policy_logprobs - ref_logprobs)

            surr1 = ratio * advantages[i]
            surr2 = (
                torch.clamp(ratio, 1 - config.grpo_eps, 1 + config.grpo_eps)
                * advantages[i]
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            kl_div = (policy_logprobs - ref_logprobs).mean()
            
            # Combine losses
            loss = policy_loss + config.grpo_kl_beta * kl_div
            loss = loss / config.grpo_group_size / config.grpo_grad_accum
            loss.backward()
            step_loss += loss.item()

        if (step + 1) % config.grpo_grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

            if step % 20 == 0:
                pbar.set_postfix({
                    "Mean Reward": f"{mean_r:.4f}", 
                    "Loss": f"{step_loss:.6f}"
                })

    final_path = "checkpoints/final_aligned.pt"
    torch.save(
        {"model_state_dict": policy_model.state_dict(), "config": config}, final_path
    )
    print(f"[*] GRPO Alignment complete. Final model saved: {final_path}")


if __name__ == "__main__":
    train_stage3_grpo(SmallLLMConfig(fast_test=True), "checkpoints/sft/sft_epoch_0.pt")
