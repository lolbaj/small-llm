import torch
import pytest
from model import MoETransformer
from config import SmallLLMConfig
from sft import mask_user_tokens

def test_model_forward():
    config = SmallLLMConfig(fast_test=True)
    model = MoETransformer(config)
    x = torch.randint(0, config.vocab_size, (1, 8))
    logits, loss, aux, util, cache = model(x, targets=x)
    assert logits.shape == (1, 8, config.vocab_size)
    assert loss is not None
    assert util >= 0

class MockTokenizer:
    def get_special_token_id(self, token):
        mapping = {"<|user|>": 1, "<|assistant|>": 2, "<|pad|>": 0}
        return mapping.get(token)

def test_sft_masking():
    tokenizer = MockTokenizer()
    # [user, hello, asst, hi, pad]
    x = torch.tensor([[1, 10, 2, 20, 0]])
    targets = mask_user_tokens(x, tokenizer)
    # Expected: user(1) masked, hello(10) masked, asst(2) masked, hi(20) kept, pad(0) masked
    assert targets[0, 0] == -100
    assert targets[0, 1] == -100
    assert targets[0, 2] == -100
    assert targets[0, 3] == 20
    assert targets[0, 4] == -100
