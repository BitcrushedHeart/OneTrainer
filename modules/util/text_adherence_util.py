"""On-image text adherence for DPO curation: is the text the caption asks for
actually rendered, and rendered right?

A wrong sign in a *chosen* image is poison: the model learns that "a sign
reading OPEN" is satisfied by "OPEM". This axis exists so that can never win a
pair by looking photographic. It is consulted only when the caption mentions
on-image text — :func:`caption_text_spans` decides that with Bitcrush Studio's
Text Tool rule (quoted strings, "reads" / "printed with" / ... cues) — so the
OCR model is never loaded for the ordinary caption.

How an image is scored:

1. **OCR** reads the whole image to plain text with one of Studio's two Text
   Tool readers, PaddleOCR-VL (primary, ~0.9B) or GOT-OCR2 (~580M fallback).
   Both run through ``transformers`` with the same load recipe Studio validated
   on the pinned 4.57.x line; neither localises, so the reading is one string.
   Readings are cached per image and model, so re-scoring a group against an
   edited caption is free.

2. **Each caption span is aligned to its best OCR window** — Studio's
   ``CandidateSearch``: every token window of about the span's length is
   normalised and compared (Levenshtein ratio, token-set ratio, take the max).
   With several spans the windows are assigned disjointly (a shop sign and its
   tagline never both claim the same tokens).

3. **Each span gets a verdict, and the image's text score is the mean of
   them.** Verbatim (zero edit distance after normalisation) is 1.0. A span the
   OCR cannot find at all is :data:`ABSENT_SCORE`. A span that is there but
   wrong — "Kama" for "Karma" — scores *below* absent: for DPO a chosen image
   with no sign under-delivers on the prompt, which the adherence axis already
   penalises, while a chosen image with a misspelled sign teaches the model
   that the misspelling *is* that caption. Within "wrong", a closer miss still
   ranks above a garbled one. Per-span readings are reported as propositions
   so the panel can say *which* text went wrong and what the model read.

The model half is injected (``backend``) so the pure half is testable without
weights; ``load_ocr_backend`` is the real one.
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from modules.util.caption_text_extract import (
    extract_identified_text,
    levenshtein_ratio,
    normalize_for_match,
    token_set_ratio,
)

TEXT_OCR_AXIS = "text_ocr"
#: Cache field for the raw OCR reading; stamped with the backend's version.
OCR_CACHE_FIELD = "ocr_text"

OCR_BACKENDS = ("paddleocr-vl", "got-ocr2")
DEFAULT_OCR_BACKEND = "paddleocr-vl"
#: Free VRAM below which "auto" keeps the reader on the CPU. PaddleOCR-VL is
#: ~1.8 GB of bf16 weights plus a long vision sequence; this leaves room for a
#: generation job sharing the card, the same policy as the SigLIP embedder.
_AUTO_CUDA_MIN_FREE_BYTES = 4_000_000_000
#: Below this similarity the best OCR window is unrelated text, not a bad
#: rendering of the span: the span is reported as absent rather than given the
#: small credit any two strings share. Studio's weak-reading bound.
ABSENT_RATIO = 0.3
#: The score of a span the image does not render at all. Sits between
#: verbatim (1.0) and every "present but wrong" rendering.
ABSENT_SCORE = 0.5
#: A wrong rendering scores its similarity times this, so the worst case that
#: still counts as present (``ABSENT_RATIO``) lands at 0.12 and the best
#: near-miss just under 0.4 — always below :data:`ABSENT_SCORE`.
WRONG_SCALE = 0.4
#: Longest reading we let a model produce. A poster's worth of text is a few
#: hundred tokens; past this the model is looping on a texture, not reading.
_MAX_NEW_TOKENS = 512


# --------------------------------------------------------------------------- #
# Pure half — caption spans and OCR alignment (Studio's text_autocorrect /
# text_agreement, trimmed to what a scorer needs)
# --------------------------------------------------------------------------- #


def caption_text_spans(prompt: str) -> list[str]:
    """The on-image text strings a caption claims, in caption order, or ``[]``
    when it claims none (so OCR is never consulted)."""
    if not prompt:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in extract_identified_text(prompt):
        key = normalize_for_match(item.text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.text)
    return out


def _has_digit_token(norm: str) -> bool:
    return any(any(ch.isdigit() for ch in token) for token in norm.split())


def token_sequence_ratio(a: str, b: str) -> float:
    """Order-aware counterpart of :func:`token_set_ratio`: the longest common
    token subsequence over the token union, so ``7 1`` vs ``1 7`` is 0.5, not
    1.0."""
    ta, tb = a.split(), b.split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    previous = [0] * (len(tb) + 1)
    for token in ta:
        current = [0]
        for j, other in enumerate(tb, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[j - 1]))
        previous = current
    union = len(set(ta) | set(tb))
    return previous[-1] / union if union else 1.0


def similarity(span_norm: str, candidate_norm: str) -> float:
    """Blend of edit-distance and token similarity on normalised text.

    Studio's rule takes the token-*set* ratio so a two-line sign read in
    column order ("24 HOURS OPEN" for "OPEN 24 HOURS") is not punished. That
    tolerance is wrong for numbers: a scoreline of 1-7 is not a rendering of
    7-1, and 30:10 is not 10:30. A span with a digit in any token therefore
    uses the order-aware token ratio instead.
    """
    if not span_norm and not candidate_norm:
        return 1.0
    if not span_norm or not candidate_norm:
        return 0.0
    token_ratio = (
        token_sequence_ratio(span_norm, candidate_norm)
        if _has_digit_token(span_norm)
        else token_set_ratio(span_norm, candidate_norm)
    )
    return max(levenshtein_ratio(span_norm, candidate_norm), token_ratio)


@dataclass(frozen=True)
class IndexedCandidate:
    """The best OCR token window for a span, including its source range."""

    raw: str
    ratio: float
    gap: float
    tok_start: int
    tok_end: int


class CandidateSearch:
    """Score each distinct OCR window once and reuse it across blocked searches."""

    def __init__(self, span_text: str, ocr_text: str) -> None:
        self.span_norm = normalize_for_match(span_text)
        tokens = ocr_text.split()
        span_len = max(1, len(self.span_norm.split()))
        min_size = max(1, span_len - 2)
        max_size = min(len(tokens), span_len + 2)
        by_norm: dict[str, tuple[float, list[tuple[str, int, int]]]] = {}

        for start in range(len(tokens)):
            for end in range(start + min_size, min(len(tokens), start + max_size) + 1):
                raw = " ".join(tokens[start:end])
                normalized = normalize_for_match(raw)
                if not normalized:
                    continue
                existing = by_norm.get(normalized)
                if existing is None:
                    by_norm[normalized] = (similarity(self.span_norm, normalized), [(raw, start, end)])
                else:
                    existing[1].append((raw, start, end))

        self._ranked = sorted(by_norm.values(), key=lambda item: item[0], reverse=True)

    @property
    def best_ratio(self) -> float:
        return self._ranked[0][0] if self._ranked else 0.0

    def best(self, blocked: set[int] | None = None) -> IndexedCandidate | None:
        """The best admissible window and its gap to the next distinct one."""
        blocked = blocked or set()
        winner: tuple[float, tuple[str, int, int]] | None = None
        runner_score: float | None = None
        for score, occurrences in self._ranked:
            occurrence = next(
                (item for item in occurrences if blocked.isdisjoint(range(item[1], item[2]))),
                None,
            )
            if occurrence is None:
                continue
            if winner is None:
                winner = score, occurrence
            else:
                runner_score = score
                break
        if winner is None:
            return None
        score, (raw, start, end) = winner
        return IndexedCandidate(
            raw=raw,
            ratio=score,
            gap=1.0 if runner_score is None else score - runner_score,
            tok_start=start,
            tok_end=end,
        )


@dataclass(frozen=True)
class SpanMatch:
    """One caption span and how well the OCR reading supports it."""

    span: str
    #: What the reader saw for this span, ``None`` when nothing resembled it.
    read: str | None
    #: Raw similarity of the reading in [0, 1]; 0 when absent.
    ratio: float
    #: ``"verbatim"``, ``"wrong"`` or ``"absent"``.
    verdict: str
    #: The axis value for this span under the verdict policy (see
    #: :func:`span_score`).
    score: float


def span_score(ratio: float, read: str | None) -> tuple[str, float]:
    """``(verdict, score)`` for one aligned span.

    Verbatim is a similarity of 1.0 — zero edit distance on the normalised
    text (or Studio's layout tolerance for a word-only span). One wrong
    letter is wrong whether the word has five characters or fourteen, so the
    threshold is exactness, not a high ratio. The reader's own slips fall on
    the wrong side of it; the panel shows what was read so they can be seen.
    """
    if read is None:
        return "absent", ABSENT_SCORE
    if ratio >= 1.0 - 1e-9:
        return "verbatim", 1.0
    return "wrong", ratio * WRONG_SCALE


def align_spans(spans: Sequence[str], ocr_text: str) -> list[SpanMatch]:
    """Assign each span its best disjoint OCR window, preserving input order.

    Spans are served best-match-first so the clearest reading claims its
    tokens before an ambiguous one can steal them.
    """
    if not spans:
        return []
    searches = [CandidateSearch(span, ocr_text) for span in spans]
    order = sorted(range(len(spans)), key=lambda i: searches[i].best_ratio, reverse=True)
    blocked: set[int] = set()
    assigned: dict[int, IndexedCandidate | None] = {}
    for i in order:
        candidate = searches[i].best(blocked)
        if candidate is None or candidate.ratio < ABSENT_RATIO:
            assigned[i] = None
            continue
        assigned[i] = candidate
        blocked.update(range(candidate.tok_start, candidate.tok_end))
    out: list[SpanMatch] = []
    for i, span in enumerate(spans):
        cand = assigned[i]
        read = cand.raw if cand else None
        ratio = cand.ratio if cand else 0.0
        verdict, score = span_score(ratio, read)
        out.append(SpanMatch(span=span, read=read, ratio=ratio, verdict=verdict, score=score))
    return out


def score_text(spans: Sequence[str], ocr_text: str | None) -> dict:
    """Per-image summary for ``spans`` against one OCR reading.

    ``text`` is the mean span score under :func:`span_score` (the axis
    value), ``text_min`` the worst span, ``matches`` the per-span detail. An
    empty or ``None`` reading marks every span absent — the caption asked for
    text and none was rendered.
    """
    matches = align_spans(spans, ocr_text or "")
    scores = [m.score for m in matches]
    return {
        "text": sum(scores) / len(scores) if scores else None,
        "text_min": min(scores) if scores else None,
        "n_text": len(scores),
        "ocr_text": (ocr_text or "").strip(),
        "matches": matches,
    }


# --------------------------------------------------------------------------- #
# The model half — Studio's Text Tool readers
# --------------------------------------------------------------------------- #


def _pick_device(device: str) -> str:
    device = str(device).lower()
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    if device == "cuda":
        return "cuda"
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return "cpu"
    return "cuda" if free >= _AUTO_CUDA_MIN_FREE_BYTES else "cpu"


def _from_pretrained_cache_first(loader, repo: str, **kwargs):
    """Local HF cache first, network only on a miss (every ``from_pretrained``
    without ``local_files_only`` issues a HEAD per file, and anonymous HEADs
    get rate-limited into minutes-long backoffs)."""
    try:
        return loader(repo, local_files_only=True, **kwargs)
    except OSError:
        return loader(repo, **kwargs)


def _release(model) -> None:
    if model is None:
        return
    with contextlib.suppress(Exception):
        model.to("cpu")
    del model
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


class OCRBackendBase:
    """Lazy load, explicit unload, one reading per image.

    The interface the scorer needs: ``model_version`` (cache key), ``loaded``,
    ``load_model() -> bool``, ``unload_model()``, ``read(pil) -> str``.
    """

    display_name = "OCR"
    repo = ""
    revision: str | None = None

    @classmethod
    def class_version(cls) -> str:
        """The cache key readings from this backend are stored under — known
        without loading, so a cache can be consulted first."""
        return f"{cls.repo}@{cls.revision or 'main'}"

    def __init__(self, device: str = "auto"):
        self.requested_device = device
        self.device: str | None = None
        self.dtype = None
        self.model = None
        self.processor = None
        self._load_lock = threading.RLock()
        self.last_error: str | None = None

    @property
    def model_version(self) -> str:
        return self.class_version()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load_model(self) -> bool:
        with self._load_lock:
            if self.loaded:
                return True
            return self._load_impl()

    def _load_impl(self) -> bool:
        raise NotImplementedError

    def _select_device(self) -> None:
        import torch

        self.device = _pick_device(self.requested_device)
        if self.device == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = torch.float32

    def unload_model(self) -> None:
        model, self.model = self.model, None
        self.processor = None
        _release(model)

    def _run_one(self, pil) -> str:
        raise NotImplementedError

    def read(self, pil) -> str:
        """Plain-text reading of an already-decoded RGB image. Raises
        ``RuntimeError`` when the model cannot be loaded."""
        if not self.loaded and not self.load_model():
            raise RuntimeError(f"{self.display_name} could not load: {self.last_error or 'unknown error'}")
        return self._run_one(pil)


_PADDLE_ATTN_IMPLEMENTATIONS: tuple[str | None, ...] = ("sdpa", "eager", None)
_paddle_shim_state = {"installed": False}


def _install_paddle_causal_mask_shim() -> None:
    """Let PaddleOCR-VL's remote modeling code run on transformers 4.57.x.

    Its forward calls ``create_causal_mask(inputs_embeds=...)``; 4.57 names
    that parameter ``input_embeds`` and rejects unknown kwargs. Translate the
    kwarg (and drop other unexpected ones). Idempotent; a no-op when the
    installed transformers already accepts ``inputs_embeds``.
    """
    if _paddle_shim_state["installed"]:
        return
    try:
        import inspect

        import transformers.masking_utils as mu
    except Exception:
        return
    orig = mu.create_causal_mask
    params = inspect.signature(orig).parameters
    if "inputs_embeds" in params:
        _paddle_shim_state["installed"] = True
        return
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    target = "input_embeds" if "input_embeds" in params else None

    def shim(*args, **kwargs):
        if "inputs_embeds" in kwargs:
            value = kwargs.pop("inputs_embeds")
            if target:
                kwargs.setdefault(target, value)
        if not has_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in params}
        return orig(*args, **kwargs)

    mu.create_causal_mask = shim
    # The remote module imported the symbol by value; patch that binding too.
    for name, module in list(sys.modules.items()):
        if "modeling_paddleocr_vl" in name and hasattr(module, "create_causal_mask"):
            module.create_causal_mask = shim
    _paddle_shim_state["installed"] = True


class PaddleOCRVLBackend(OCRBackendBase):
    """PaddleOCR-VL through ``transformers`` — Studio's primary Text Tool reader.

    On transformers < 5.12 the remote code registers under
    ``AutoModelForCausalLM``; candidate Auto* classes are tried in order and the
    first that accepts the config wins. SDPA attention first: eager over the
    vision tower's ~14k patches allocates a multi-GiB fp32 softmax per layer.
    """

    display_name = "PaddleOCR-VL"
    repo = "PaddlePaddle/PaddleOCR-VL"
    #: Commit pinned for remote-code reproducibility; the recipe below was
    #: validated against this snapshot (Studio ISSUE-0909).
    revision = "baee27eebcbf26cdeab160116679d765f13a3f27"
    _prompt = "OCR:"

    @staticmethod
    def _candidate_model_classes() -> list:
        import transformers as tf

        out = []
        for attr in (
            "PaddleOCRVLForConditionalGeneration",
            "AutoModelForCausalLM",
            "AutoModel",
            "AutoModelForImageTextToText",
        ):
            cls = getattr(tf, attr, None)
            if cls is not None:
                out.append(cls)
        return out

    def _load_impl(self) -> bool:
        try:
            from transformers import AutoProcessor
        except ImportError as ex:
            self.last_error = f"transformers unavailable: {ex}"
            return False
        try:
            self._select_device()
            self.processor = _from_pretrained_cache_first(
                AutoProcessor.from_pretrained,
                self.repo,
                trust_remote_code=True,
                revision=self.revision,
                use_fast=True,
            )
            self.model = self._load_weights()
            if self.model is None:
                self.unload_model()
                return False
            self.model.eval()
            _install_paddle_causal_mask_shim()
            return True
        except Exception as ex:
            self.last_error = str(ex)
            self.unload_model()
            return False

    def _load_weights(self):
        last: Exception | None = None
        for cls in self._candidate_model_classes():
            for attn in _PADDLE_ATTN_IMPLEMENTATIONS:
                kwargs = {"trust_remote_code": True, "revision": self.revision, "torch_dtype": self.dtype}
                if attn is not None:
                    kwargs["attn_implementation"] = attn
                try:
                    model = _from_pretrained_cache_first(cls.from_pretrained, self.repo, **kwargs)
                    return model.to(self.device)
                except (ValueError, TypeError, OSError) as ex:
                    last = ex
        self.last_error = f"no compatible loader ({type(last).__name__}: {last})" if last else "no loader"
        return None

    def _build_inputs(self, pil):
        apply_chat = getattr(self.processor, "apply_chat_template", None)
        if callable(apply_chat):
            messages = [
                {"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": self._prompt}]}
            ]
            return apply_chat(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
        return self.processor(text=self._prompt, images=pil, return_tensors="pt")

    def _run_one(self, pil) -> str:
        import torch

        with torch.no_grad():
            inputs = self._build_inputs(pil).to(self.device)
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=_MAX_NEW_TOKENS,
                # The repo's generation_config ships use_cache=false; with the
                # cache off the vision tower re-encodes the image per token.
                use_cache=True,
            )
        input_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]


class GotOcr2Backend(OCRBackendBase):
    """GOT-OCR2 (~580M) through its mainline ``transformers`` class — the
    lighter fallback reader."""

    display_name = "GOT-OCR2"
    repo = "stepfun-ai/GOT-OCR-2.0-hf"
    _stop_string = "<|im_end|>"

    def _load_impl(self) -> bool:
        try:
            from transformers import AutoProcessor

            try:
                from transformers import GotOcr2ForConditionalGeneration as model_cls
            except ImportError:
                from transformers import AutoModelForImageTextToText as model_cls
        except ImportError as ex:
            self.last_error = f"transformers unavailable: {ex}"
            return False
        try:
            self._select_device()
            self.processor = _from_pretrained_cache_first(AutoProcessor.from_pretrained, self.repo, use_fast=True)
            self.model = _from_pretrained_cache_first(model_cls.from_pretrained, self.repo, torch_dtype=self.dtype).to(
                self.device
            )
            self.model.eval()
            return True
        except Exception as ex:
            self.last_error = str(ex)
            self.unload_model()
            return False

    def _run_one(self, pil) -> str:
        import torch

        with torch.no_grad():
            inputs = self.processor(pil, return_tensors="pt").to(self.device)
            if self.dtype != torch.float32:
                inputs = inputs.to(self.dtype)
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings=self._stop_string,
                max_new_tokens=_MAX_NEW_TOKENS,
            )
        return self.processor.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


_BACKEND_CLASSES = {"paddleocr-vl": PaddleOCRVLBackend, "got-ocr2": GotOcr2Backend}


class OCRReader:
    """Studio's reader chain behind one lazy handle.

    Nothing loads until the first ``read``: readings are cached per image and
    per backend version, and ``candidate_versions`` names every key a reading
    may sit under (preferred backend first), so a group whose readings are all
    cached never pays for the weights. On a miss the preferred backend is
    tried first and the other is the fallback, so a broken remote-code load
    still leaves the axis working; when neither loads ``read`` raises the
    ``RuntimeError`` the UI shows.
    """

    def __init__(self, device: str = "auto", backend: str = DEFAULT_OCR_BACKEND):
        self.requested_device = device
        self._order = [backend, *[b for b in OCR_BACKENDS if b != backend]]
        self._active: OCRBackendBase | None = None
        self._load_lock = threading.RLock()
        self.last_error: str | None = None

    @property
    def candidate_versions(self) -> list[str]:
        """Cache keys a reading may be stored under, best first."""
        versions: list[str] = []
        if self._active is not None:
            versions.append(self._active.model_version)
        for name in self._order:
            cls = _BACKEND_CLASSES.get(name)
            if cls is not None and cls.class_version() not in versions:
                versions.append(cls.class_version())
        return versions

    @property
    def model_version(self) -> str:
        """The key new readings are stored under: the loaded backend's, or the
        preferred one's before anything has loaded."""
        versions = self.candidate_versions
        return versions[0] if versions else "none"

    @property
    def display_name(self) -> str:
        return self._active.display_name if self._active is not None else "OCR"

    @property
    def loaded(self) -> bool:
        return self._active is not None and self._active.loaded

    def load_model(self) -> bool:
        with self._load_lock:
            if self.loaded:
                return True
            failures: list[str] = []
            for name in self._order:
                cls = _BACKEND_CLASSES.get(name)
                if cls is None:
                    failures.append(f"{name}: unknown backend")
                    continue
                reader = cls(device=self.requested_device)
                if reader.load_model():
                    self._active = reader
                    self.last_error = None
                    return True
                failures.append(f"{reader.display_name}: {reader.last_error or 'load failed'}")
            self.last_error = "; ".join(failures)
            return False

    def unload_model(self) -> None:
        with self._load_lock:
            active, self._active = self._active, None
            if active is not None:
                active.unload_model()

    def read(self, pil) -> str:
        if not self.loaded and not self.load_model():
            raise RuntimeError(f"Text adherence scoring could not load an OCR model ({self.last_error})")
        return self._active.read(pil)


def load_ocr_backend(device: str = "auto", backend: str = DEFAULT_OCR_BACKEND) -> OCRReader:
    """A reader handle, or a ``RuntimeError`` the UI can show. Nothing loads
    until a reading the cache cannot answer is asked for (see
    :class:`OCRReader`)."""
    try:
        import torch  # noqa: F401

        import transformers  # noqa: F401
    except ImportError as ex:
        raise RuntimeError("Text adherence scoring needs torch and transformers.") from ex
    return OCRReader(device=device, backend=backend)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def read_image_text(
    paths: Sequence[str],
    *,
    backend,
    cache=None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, str]:
    """The OCR reading of every readable path, cache first, model for the rest.

    Unreadable files and readings that raise are skipped (the image ends up
    with no text score rather than a wrong one).

    A reading cached under any of the backend's ``candidate_versions`` counts:
    a reading is a reading whichever of Studio's two readers produced it, and
    accepting the fallback's means a session that once fell back to GOT-OCR2
    does not re-read everything when PaddleOCR-VL loads again.
    """
    from PIL import Image

    versions = list(getattr(backend, "candidate_versions", None) or [backend.model_version])
    readings: dict[str, str] = {}
    misses: list[str] = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        cached = None
        if cache is not None:
            for version in versions:
                cached = cache.peek_versioned(path, OCR_CACHE_FIELD, version)
                if isinstance(cached, str):
                    break
        if isinstance(cached, str):
            readings[path] = cached
        else:
            misses.append(path)
    done = len(readings)
    total = len(paths)
    if progress is not None:
        progress(done, total)
    for path in misses:
        try:
            with Image.open(path) as handle:
                pil = handle.convert("RGB")
            text = backend.read(pil)
        except RuntimeError:
            raise
        except Exception:
            text = None
        if text is not None:
            text = text.strip()
            readings[path] = text
            if cache is not None:
                # Stamped with the backend that actually read it, which after a
                # fallback is not the preferred one.
                cache.store_versioned(path, OCR_CACHE_FIELD, text, backend.model_version)
        done += 1
        if progress is not None:
            progress(done, total)
    return readings


def _span_sentence(m: SpanMatch) -> str:
    if m.verdict == "absent":
        return f"text: {m.span!r} not found"
    if m.verdict == "verbatim":
        return f"text: {m.span!r} rendered verbatim"
    return f"text: {m.span!r} misrendered as {m.read!r}"


def score_images_text(
    paths: Sequence[str],
    prompt: str,
    *,
    backend=None,
    cache=None,
    device: str = "auto",
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, dict]:
    """Text adherence of each readable path to the text ``prompt`` claims.

    Returns ``{path: {text, text_min, n_text, ocr_text, propositions: [{text,
    axis, label, prob, read}]}}``. A caption that claims no on-image text
    returns ``{}`` without touching the model. Raises ``RuntimeError`` when no
    OCR model can be loaded.
    """
    spans = caption_text_spans(prompt)
    if not spans or not paths:
        return {}
    if backend is None:
        backend = load_ocr_backend(device=device)
    readings = read_image_text(list(paths), backend=backend, cache=cache, progress=progress)
    results: dict[str, dict] = {}
    for path, ocr_text in readings.items():
        summary = score_text(spans, ocr_text)
        matches = summary.pop("matches")
        summary["propositions"] = [
            {
                "text": _span_sentence(m),
                "axis": TEXT_OCR_AXIS,
                "label": m.span,
                "prob": m.score,
                "read": m.read,
                "verdict": m.verdict,
                "ratio": m.ratio,
            }
            for m in matches
        ]
        results[path] = summary
    return results
