"""ZImageModel.encode_text on a tiny random Qwen3ForCausalLM (CPU, fp32).

The text encoder call skips the KV cache and keeps one position of logits,
and ``pad_to_longest`` pads a batch to its longest prompt. The invariant: the
kept real-token hidden states match the legacy call (512-token padding, full
logits, cache on) for every prompt, whether a chunk mixes lengths or has no
padding row at all."""

from types import SimpleNamespace

from modules.model.ZImageModel import PROMPT_MAX_LENGTH, ZImageModel
from modules.util.enum.ModelType import ModelType

import torch
from torch.testing import assert_close

import pytest

VOCAB = 128
PAD_ID = 0
CPU = torch.device("cpu")


class _StubTokenizer:
    """Character tokenizer with a tiny chat template. Pads on the right like
    the Qwen2 tokenizer; ids stay below the tiny model's vocabulary."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=False):
        assert tokenize is False
        assert add_generation_prompt is True and enable_thinking is True
        return "<u>" + messages[0]["content"] + "<a>"

    def __call__(self, text, max_length, padding, truncation, return_tensors):
        assert return_tensors == "pt"
        self.calls.append({"padding": padding, "max_length": max_length, "truncation": truncation})
        ids = [[1 + ord(ch) % (VOCAB - 1) for ch in t] for t in text]
        if truncation:
            ids = [row[:max_length] for row in ids]
        if padding == "max_length":
            width = max_length
        elif padding == "longest":
            width = max(len(row) for row in ids)
        else:
            raise AssertionError(f"unexpected padding {padding!r}")
        input_ids = torch.full((len(ids), width), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, row in enumerate(ids):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[i, : len(row)] = 1
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


@pytest.fixture(scope="module")
def tiny_qwen():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config).eval()


def _model(tiny_qwen) -> ZImageModel:
    model = ZImageModel(ModelType.Z_IMAGE)
    model.text_encoder = tiny_qwen
    model.tokenizer = _StubTokenizer()
    return model


def _legacy_encode(model: ZImageModel, prompts: list[str]) -> list[torch.Tensor]:
    """The pre-change encode: 512-token padding, default cache and logits."""
    templated = [
        model.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        for p in prompts
    ]
    out = model.tokenizer(
        templated, max_length=PROMPT_MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        result = model.text_encoder(
            out.input_ids, attention_mask=out.attention_mask.float(), output_hidden_states=True, return_dict=True
        )
    # the legacy call really did build the cache and the full-sequence logits
    assert result.past_key_values is not None
    assert result.logits.shape == (len(prompts), PROMPT_MAX_LENGTH, VOCAB)
    hidden = result.hidden_states[-2]
    mask = out.attention_mask.bool()
    return [hidden[i][mask[i]] for i in range(len(prompts))]


MIXED = ['Neon sign text reading "Open 24h"', "a", 'Chalkboard text reading "Soup of the day"', ""]
EQUAL = ["abcd", "efgh", "ijkl"]  # same token count: no padding row under pad_to_longest


def _hook_kwargs(module):
    seen = {}

    def pre_hook(_module, args, kwargs):
        seen["kwargs"] = dict(kwargs)

    def post_hook(_module, args, kwargs, output):
        seen["logits_shape"] = tuple(output.logits.shape)
        seen["past_key_values"] = output.past_key_values

    handles = [
        module.register_forward_pre_hook(pre_hook, with_kwargs=True),
        module.register_forward_hook(post_hook, with_kwargs=True),
    ]
    return seen, handles


@pytest.mark.parametrize("prompts", [MIXED, EQUAL], ids=["mixed_lengths", "equal_lengths"])
@pytest.mark.parametrize("pad_to_longest", [False, True], ids=["pad_max_length", "pad_longest"])
def test_encode_text_matches_legacy_real_token_states(tiny_qwen, prompts, pad_to_longest):
    model = _model(tiny_qwen)
    expected = _legacy_encode(model, prompts)

    text = list(prompts)
    with torch.no_grad():
        got = model.encode_text(train_device=CPU, batch_size=len(text), text=text, pad_to_longest=pad_to_longest)

    call = model.tokenizer.calls[-1]
    assert call["padding"] == ("longest" if pad_to_longest else "max_length")
    assert call["max_length"] == PROMPT_MAX_LENGTH
    assert call["truncation"] is True
    assert len(got) == len(expected)
    for g, e in zip(got, expected, strict=True):
        assert g.shape == e.shape
        assert_close(g, e, atol=1e-5, rtol=1e-4)


def test_equal_length_chunk_has_no_padding_under_pad_to_longest(tiny_qwen):
    model = _model(tiny_qwen)
    seen, handles = _hook_kwargs(tiny_qwen)
    try:
        with torch.no_grad():
            model.encode_text(train_device=CPU, batch_size=3, text=list(EQUAL), pad_to_longest=True)
    finally:
        for handle in handles:
            handle.remove()
    mask = seen["kwargs"]["attention_mask"]
    assert mask.shape == (3, len("<u>abcd<a>"))
    assert bool((mask == 1).all())


def test_encode_text_templates_the_callers_list_in_place(tiny_qwen):
    model = _model(tiny_qwen)
    text = ["one", "two words"]
    with torch.no_grad():
        model.encode_text(train_device=CPU, batch_size=2, text=text, pad_to_longest=True)
    assert text == ["<u>one<a>", "<u>two words<a>"]

    text = ["one"]
    with torch.no_grad():
        model.encode_text(train_device=CPU, batch_size=1, text=text)
    assert text == ["<u>one<a>"]


@pytest.mark.parametrize("pad_to_longest", [False, True], ids=["pad_max_length", "pad_longest"])
def test_encode_text_skips_cache_and_keeps_one_logit_position(tiny_qwen, pad_to_longest):
    model = _model(tiny_qwen)
    seen, handles = _hook_kwargs(tiny_qwen)
    try:
        with torch.no_grad():
            model.encode_text(train_device=CPU, batch_size=2, text=["abc", "de"], pad_to_longest=pad_to_longest)
    finally:
        for handle in handles:
            handle.remove()

    assert seen["kwargs"]["use_cache"] is False
    assert seen["kwargs"]["logits_to_keep"] == 1
    assert seen["kwargs"]["output_hidden_states"] is True
    assert seen["logits_shape"] == (2, 1, VOCAB)
    assert seen["past_key_values"] is None


def test_encode_text_token_path_is_unaffected_by_pad_to_longest(tiny_qwen):
    """Pre-tokenized batches (the training path) ignore ``pad_to_longest``."""
    model = _model(tiny_qwen)
    out = model.tokenizer(
        ["<u>xyz<a>", "<u>q<a>"], max_length=16, padding="max_length", truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        a = model.encode_text(train_device=CPU, tokens=out.input_ids, tokens_mask=out.attention_mask)
        b = model.encode_text(
            train_device=CPU, tokens=out.input_ids, tokens_mask=out.attention_mask, pad_to_longest=True
        )
    assert [t.shape[0] for t in a] == [9, 7]
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)
