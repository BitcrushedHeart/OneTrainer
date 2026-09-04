"""Dense-transport warm starts for non-OFT PEFT adapters (LoRA / DoRA / LoKr).

Steps 1-2 of the dfit pipeline (modules/util/oft_dense_fit.py) never look at
the adapter's parameterization: the dense transport ``W_v ~= R_L W_b R_R`` is
fit from the base/variant weights alone, and the education-anchored target
``T = W_v + R_L (M_b - W_b) R_R`` only needs the ORIGINAL adapter's merged
teacher weight ``M_b``. This module supplies what the other adapter families
need around those shared steps:

- ``detect_adapter_kind``: classify an adapter file by its tensor-key suffixes
  (``oft_R.weight`` / ``lora_down.weight`` [+ ``dora_scale``] / ``lokr_w1*``),
  never by metadata, which lies about requested-vs-actual values elsewhere.
- Merged-form builders that mirror the modules' forward/bake math EXACTLY
  (LoRAModule / DoRAModule / LoKrModule in modules/module/LoRAModule.py):
  LoRA ``W + (alpha/rank) B A``; DoRA ``dora_scale * WP / (norm(WP) + eps)``
  with the norm over rows (decompose_output_axis) or columns (default, and the
  axis is DERIVED from the stored dora_scale shape, not config); LoKr
  ``W + (alpha/dim) kron(w1, w2)`` with each factor possibly itself low-rank.
- Student fits of each deployed form against a dense target:
  * LoRA: one truncated SVD of ``(T - W_v)/s`` -- Eckart-Young, globally
    optimal, no iteration and no refinement needed.
  * DoRA: alternating least squares over (axis scale c, low-rank BA). Both
    half-steps are exactly optimal given the other (the scale folds into the
    low-rank fit as ``X = s BA diag(c)``, so the constrained problem stays a
    plain truncated SVD), so the objective is monotone. The stored
    ``dora_scale`` is recovered from the effective scale as
    ``c * (norm(WP) + eps)``, which reproduces ``c`` bit-exactly through the
    module's decode.
  * LoKr: nearest-Kronecker-product via the Van Loan rearrangement (rank-1 SVD
    of the rearranged delta), factor-rank projection for decomposed w1/w2, and
    a few rank-1 ALS sweeps. With a DoRA scale attached the c half-step stays
    exact but the Kronecker half-step is fit against ``T/c`` (exact only for
    constant c); ``refine_peft_fit`` closes the remainder.
- ``refine_peft_fit``: generic LBFGS polish over the RAW stored tensors
  through the exact merged form, keep-best on the true deployed error (never
  returns worse than its input). Unlike the OFT path there is no encode/decode
  mismatch to fight -- the stored tensors ARE the deployed parameterization --
  so this only mops up ALS suboptimality and the DoRA-LoKr approximation.

Linear tensors only (the late families are all nn.Linear); Conv/Tucker layouts
raise. Pure tensor math -- no file I/O. Callers on this Windows box must obey
the MKL discipline (CLAUDE.md): materialize every safetensors tensor and close
the files BEFORE calling anything here, ``torch.set_num_threads(1)`` in
scripts, TF32 off on CUDA.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from modules.util.oft_dense_fit import DenseTransportFit, dense_transport_fit

import torch

# ---------------------------------------------------------------------------
# Detection

_KIND_MARKERS = {
    "oft": (".oft_R.weight",),
    "lora": (".lora_down.weight",),
    "lokr": (".lokr_w1", ".lokr_w1_a"),
}


def detect_adapter_kind(keys: Iterable[str]) -> str:
    """Adapter family from tensor-key suffixes alone.

    Returns one of ``oft``, ``dora_oft``, ``lora``, ``dora``, ``lokr``,
    ``dora_lokr``. LoHa is recognized and rejected explicitly; an empty or
    mixed-family file raises ValueError.
    """
    keys = list(keys)
    found = set()
    for kind, suffixes in _KIND_MARKERS.items():
        if any(k.endswith(s) for k in keys for s in suffixes):
            found.add(kind)
    if any(k.endswith(".hada_w1_a") for k in keys):
        raise NotImplementedError("LoHa adapters are not supported by the dfit pipeline")
    if not found:
        raise ValueError("no recognizable adapter tensors (oft_R / lora_down / lokr_w1)")
    if len(found) > 1:
        raise ValueError(f"adapter mixes families {sorted(found)}; refusing to guess")
    kind = found.pop()
    if kind == "oft":
        return "dora_oft" if any(k.endswith(".dora_multiplier") for k in keys) else "oft"
    has_dora = any(k.endswith(".dora_scale") for k in keys)
    if kind == "lora":
        return "dora" if has_dora else "lora"
    return "dora_lokr" if has_dora else "lokr"


def marker_suffixes(kind: str) -> tuple[str, ...]:
    """The per-module marker suffix(es) that identify one adapted stem."""
    base = kind.removeprefix("dora_") if kind.startswith("dora_") else ("lora" if kind == "dora" else kind)
    return _KIND_MARKERS[base]


# ---------------------------------------------------------------------------
# Per-stem parameter bundles (fp32, Linear only)


@dataclass
class LoRAStemParams:
    down: torch.Tensor  # (rank, in)
    up: torch.Tensor  # (out, rank)
    alpha: float
    dora_scale: torch.Tensor | None = None  # stored shape: (out, 1) row axis / (1, in) column axis
    norm_eps: bool = True  # DoRAModule's norm_epsilon flag (from ot_config)

    @property
    def rank(self) -> int:
        return self.down.shape[0]

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @property
    def output_axis(self) -> bool:
        assert self.dora_scale is not None
        return self.dora_scale.shape[0] != 1


@dataclass
class LoKrStemParams:
    """Stored-layout LoKr factors; exactly one of (w1, w1_a+w1_b) and one of
    (w2, w2_a+w2_b) is present, mirroring LoKrModule's use_w1/use_w2 split."""

    alpha: float
    w1: torch.Tensor | None = None  # (out_l, in_m)
    w1_a: torch.Tensor | None = None  # (out_l, dim)
    w1_b: torch.Tensor | None = None  # (dim, in_m)
    w2: torch.Tensor | None = None  # (out_k, in_n)
    w2_a: torch.Tensor | None = None  # (out_k, dim)
    w2_b: torch.Tensor | None = None  # (dim, in_n)
    dora_scale: torch.Tensor | None = None
    norm_eps: bool = True  # LoKrModule applies eps unconditionally

    def factor1(self) -> torch.Tensor:
        return self.w1 if self.w1 is not None else self.w1_a @ self.w1_b

    def factor2(self) -> torch.Tensor:
        return self.w2 if self.w2 is not None else self.w2_a @ self.w2_b

    @property
    def dim(self) -> int | None:
        """The lokr_dim, derivable only when at least one factor is decomposed."""
        if self.w1_a is not None:
            return self.w1_a.shape[1]
        if self.w2_a is not None:
            return self.w2_a.shape[1]
        return None

    @property
    def scale(self) -> float:
        # use_w1 and use_w2 both full: LoKrModule fills alpha to dim, scale 1.
        dim = self.dim
        return 1.0 if dim is None else self.alpha / dim

    @property
    def output_axis(self) -> bool:
        assert self.dora_scale is not None
        return self.dora_scale.shape[0] != 1


def lora_params_from_tensors(
    tensors: dict[str, torch.Tensor], stem_prefix: str, *, norm_eps: bool = True, dtype=torch.float32, device="cpu"
) -> LoRAStemParams:
    """Bundle for one adapted module; ``stem_prefix`` includes the trailing dot
    (e.g. ``transformer.layers.22.attention.to_q.``)."""

    def t(suffix):
        return tensors[stem_prefix + suffix].to(device=device, dtype=dtype)

    dora = tensors.get(stem_prefix + "dora_scale")
    return LoRAStemParams(
        down=t("lora_down.weight"),
        up=t("lora_up.weight"),
        alpha=float(tensors[stem_prefix + "alpha"]),
        dora_scale=dora.to(device=device, dtype=dtype) if dora is not None else None,
        norm_eps=norm_eps,
    )


def lokr_params_from_tensors(
    tensors: dict[str, torch.Tensor], stem_prefix: str, *, norm_eps: bool = True, dtype=torch.float32, device="cpu"
) -> LoKrStemParams:
    if stem_prefix + "lokr_t2" in tensors:
        raise NotImplementedError(f"{stem_prefix}: Tucker LoKr layout is Conv-only and unsupported here")

    def opt(suffix):
        v = tensors.get(stem_prefix + suffix)
        return v.to(device=device, dtype=dtype) if v is not None else None

    p = LoKrStemParams(
        alpha=float(tensors[stem_prefix + "alpha"]),
        w1=opt("lokr_w1"),
        w1_a=opt("lokr_w1_a"),
        w1_b=opt("lokr_w1_b"),
        w2=opt("lokr_w2"),
        w2_a=opt("lokr_w2_a"),
        w2_b=opt("lokr_w2_b"),
        dora_scale=opt("dora_scale"),
        norm_eps=norm_eps,
    )
    if (p.w1 is None) == (p.w1_a is None) or (p.w2 is None) == (p.w2_a is None):
        raise ValueError(f"{stem_prefix}: inconsistent LoKr factor layout")
    return p


# ---------------------------------------------------------------------------
# Merged forms (must mirror the modules' math exactly)


def _dora_norm(wp: torch.Tensor, output_axis: bool, norm_eps: bool) -> torch.Tensor:
    eps = torch.finfo(wp.dtype).eps if norm_eps else 0.0
    if output_axis:
        return wp.norm(dim=1, keepdim=True) + eps  # (out, 1)
    return wp.norm(dim=0, keepdim=True) + eps  # (1, in)


def merged_lora(w: torch.Tensor, p: LoRAStemParams) -> torch.Tensor:
    """LoRAModule / DoRAModule deployed weight for a Linear layer."""
    wp = w + p.scale * (p.up @ p.down)
    if p.dora_scale is None:
        return wp
    return p.dora_scale * (wp / _dora_norm(wp, p.output_axis, p.norm_eps))


def merged_lokr(w: torch.Tensor, p: LoKrStemParams) -> torch.Tensor:
    """LoKrModule deployed weight for a Linear layer (make_kron + view)."""
    wp = w + p.scale * torch.kron(p.factor1().contiguous(), p.factor2().contiguous())
    if p.dora_scale is None:
        return wp
    return p.dora_scale * (wp / _dora_norm(wp, p.output_axis, p.norm_eps))


def merged_peft(kind: str, w: torch.Tensor, p) -> torch.Tensor:
    if kind in ("lora", "dora"):
        return merged_lora(w, p)
    if kind in ("lokr", "dora_lokr"):
        return merged_lokr(w, p)
    raise ValueError(f"unsupported kind {kind!r} (OFT goes through oft_dense_fit)")


# ---------------------------------------------------------------------------
# Student fits


def _truncated_svd_factors(x: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced (down, up) with up @ down = best rank-``rank`` approx of x."""
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    r = min(rank, s.numel())
    sq = s[:r].clamp_min(0).sqrt()
    return sq.view(-1, 1) * vh[:r], u[:, :r] * sq.view(1, -1)


def _axis_scale_view(c: torch.Tensor, output_axis: bool) -> torch.Tensor:
    return c.view(-1, 1) if output_axis else c.view(1, -1)


def _axis_scale_init(w: torch.Tensor, target: torch.Tensor, output_axis: bool) -> torch.Tensor:
    """Closed-form per-row/column scale minimizing ||w (x) c - target||: the
    exact c half-step at zero adapter delta."""
    axis = 1 if output_axis else 0
    num = (w * target).sum(dim=axis)
    den = (w * w).sum(dim=axis)
    tiny = torch.finfo(w.dtype).tiny
    return torch.where(den > 0.0, num / den.clamp_min(tiny), torch.ones_like(num))


@dataclass
class LoRAStudentFit:
    down: torch.Tensor
    up: torch.Tensor
    axis_scale: torch.Tensor | None  # effective c ((out,) or (in,)), None without DoRA
    objective_history: list[float]  # ||deployed - target||_F per iteration


def fit_student_lora(w_variant: torch.Tensor, target: torch.Tensor, rank: int, scale: float) -> LoRAStudentFit:
    """Exact (Eckart-Young) rank-``rank`` fit of ``W_v + scale * up@down`` to
    the target: one truncated SVD, no iteration."""
    down, up = _truncated_svd_factors((target - w_variant) / scale, rank)
    err = float((w_variant + scale * (up @ down) - target).norm())
    return LoRAStudentFit(down=down, up=up, axis_scale=None, objective_history=[err])


def fit_student_dora(
    w_variant: torch.Tensor,
    target: torch.Tensor,
    rank: int,
    scale: float,
    output_axis: bool,
    iters: int = 4,
) -> LoRAStudentFit:
    """ALS fit of the DoRA deployed form ``c (x) (W_v + scale * up@down)``.

    Low-rank half-step (given c): with ``X = scale * (up@down) (x) c`` the
    problem is an UNWEIGHTED truncated SVD of ``target - W_v (x) c`` (the axis
    scale is invertible, so the rank constraint carries through) -- exactly
    optimal. Scale half-step (given up/down): per-row/column closed-form least
    squares. Both halves exact, so the objective never increases. Axis entries
    where ``c`` underflows keep their previous factors (that row/column's
    residual is ``||target_row||`` regardless of BA there).

    ``c`` starts at its closed-form optimum for BA = 0 rather than at ones:
    starting at ones leaves the scale component of the delta inside the first
    truncated SVD, which then burns rank on it and parks the alternation in a
    visibly worse stationary point (~100x on representable targets).
    """
    tiny = torch.finfo(w_variant.dtype).tiny
    c = _axis_scale_init(w_variant, target, output_axis)
    down, up, history = None, None, []
    for _ in range(iters):
        cv = _axis_scale_view(c, output_axis)
        safe = torch.where(c.abs() > tiny, c, torch.ones_like(c))
        x = target - w_variant * cv
        down_x, up_x = _truncated_svd_factors(x, rank)
        # BA = X / c / scale; fold the un-scaling into the up factor (row axis)
        # or the down factor (column axis) so up@down stays rank-exact.
        if output_axis:
            up = up_x / _axis_scale_view(safe, True) / scale
            down = down_x
        else:
            down = down_x / _axis_scale_view(safe, False) / scale
            up = up_x
        wp = w_variant + scale * (up @ down)
        axis = 1 if output_axis else 0
        num = (wp * target).sum(dim=axis)
        den = (wp * wp).sum(dim=axis)
        c = torch.where(den > 0.0, num / den.clamp_min(tiny), torch.ones_like(num))
        history.append(float((wp * _axis_scale_view(c, output_axis) - target).norm()))
    return LoRAStudentFit(down=down, up=up, axis_scale=c, objective_history=history)


def dora_scale_from_axis_scale(c: torch.Tensor, wp: torch.Tensor, output_axis: bool, norm_eps: bool) -> torch.Tensor:
    """The stored dora_scale that makes the module's decode reproduce the
    effective axis scale ``c`` exactly: ``dora_scale = c * (norm(WP) + eps)``
    (the decode divides by that same norm)."""
    return _axis_scale_view(c, output_axis) * _dora_norm(wp, output_axis, norm_eps)


@dataclass
class LoKrStudentFit:
    w1: torch.Tensor  # (out_l, in_m) full factor (rank-projected if decomposed)
    w2: torch.Tensor  # (out_k, in_n)
    axis_scale: torch.Tensor | None
    objective_history: list[float]


def _van_loan_rearrange(d: torch.Tensor, out_l: int, out_k: int, in_m: int, in_n: int) -> torch.Tensor:
    """R with R[i*in_m+j, k*in_n+n] = d[i*out_k+k, j*in_n+n]: nearest Kronecker
    product of d == rank-1 approximation of R (Van Loan & Pitsianis)."""
    return d.reshape(out_l, out_k, in_m, in_n).permute(0, 2, 1, 3).reshape(out_l * in_m, out_k * in_n)


def _project_rank(w: torch.Tensor, rank: int | None) -> torch.Tensor:
    if rank is None or min(w.shape) <= rank:
        return w
    down, up = _truncated_svd_factors(w, rank)
    return up @ down


def fit_student_lokr(
    w_variant: torch.Tensor,
    target: torch.Tensor,
    p: LoKrStemParams,
    iters: int = 8,
) -> LoKrStudentFit:
    """Nearest-Kronecker fit of the LoKr deployed form to a dense target.

    Rank-1 SVD of the Van Loan rearrangement seeds w1/w2; when a factor is
    stored decomposed (w1_a@w1_b) it is projected to that inner rank, then
    rank-1 ALS sweeps (closed-form given the other factor, followed by the
    rank projection) re-tighten the pair. With a DoRA scale attached the c
    half-step is exact but the Kronecker half is fit against ``target / c``
    -- exact only for constant c; refine_peft_fit mops up the difference.
    """
    out_l, in_m = p.factor1().shape
    out_k, in_n = p.factor2().shape
    if (out_l * out_k, in_m * in_n) != tuple(w_variant.shape):
        raise ValueError(
            f"LoKr factor shapes ({out_l}x{in_m}, {out_k}x{in_n}) do not tile weight {tuple(w_variant.shape)}"
        )
    scale = p.scale
    rank1 = p.w1_a.shape[1] if p.w1_a is not None else None
    rank2 = p.w2_a.shape[1] if p.w2_a is not None else None
    tiny = torch.finfo(w_variant.dtype).tiny
    dora = p.dora_scale is not None
    # same closed-form scale init as fit_student_dora (c half-step at zero delta)
    c = _axis_scale_init(w_variant, target, p.output_axis) if dora else None

    def kron_fit_target() -> torch.Tensor:
        if c is None:
            return (target - w_variant) / scale
        safe = torch.where(c.abs() > tiny, c, torch.ones_like(c))
        return (target / _axis_scale_view(safe, p.output_axis) - w_variant) / scale

    r = _van_loan_rearrange(kron_fit_target(), out_l, out_k, in_m, in_n)
    u, s, vh = torch.linalg.svd(r, full_matrices=False)
    sq = s[0].clamp_min(0).sqrt()
    w1 = _project_rank((sq * u[:, 0]).reshape(out_l, in_m), rank1)
    w2 = _project_rank((sq * vh[0]).reshape(out_k, in_n), rank2)

    history: list[float] = []
    for _ in range(iters):
        r = _van_loan_rearrange(kron_fit_target(), out_l, out_k, in_m, in_n)
        v2 = w2.reshape(-1)
        w1 = _project_rank((r @ v2 / v2.dot(v2).clamp_min(tiny)).reshape(out_l, in_m), rank1)
        v1 = w1.reshape(-1)
        w2 = _project_rank((r.transpose(-2, -1) @ v1 / v1.dot(v1).clamp_min(tiny)).reshape(out_k, in_n), rank2)
        wp = w_variant + scale * torch.kron(w1.contiguous(), w2.contiguous())
        if dora:
            axis = 1 if p.output_axis else 0
            num = (wp * target).sum(dim=axis)
            den = (wp * wp).sum(dim=axis)
            c = torch.where(den > 0.0, num / den.clamp_min(tiny), torch.ones_like(num))
            history.append(float((wp * _axis_scale_view(c, p.output_axis) - target).norm()))
        else:
            history.append(float((wp - target).norm()))
    return LoKrStudentFit(w1=w1, w2=w2, axis_scale=c, objective_history=history)


# ---------------------------------------------------------------------------
# Generic LBFGS refinement over the raw stored tensors


@dataclass
class PeftRefineResult:
    tensors: dict[str, torch.Tensor]  # refined stored tensors, keyed by suffix
    err_init: float
    err_final: float  # never above err_init (keep-best)
    err_history: list[float]
    steps: int


def refine_peft_fit(
    tensors: dict[str, torch.Tensor],
    build_merged: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    target: torch.Tensor,
    *,
    iters: int = 60,
    tol: float = 1e-10,
    history_size: int = 10,
    stall_limit: int = 8,
) -> PeftRefineResult:
    """LBFGS polish of the stored tensors through the exact merged form.

    The stored tensors ARE the deployed parameterization for these families
    (no encode/decode mismatch, no clip), so the loop is plain keep-best
    LBFGS on ``||build_merged(tensors) - target||``. max_eval must be raised
    past torch's max_iter*5//4 default for the same reason as
    oft_dense_fit.refine_deployed_fit: with max_iter=1 the entry evaluation
    consumes the whole budget and strong-Wolfe can never move.
    """
    leaves = {k: v.detach().clone().contiguous().requires_grad_(True) for k, v in tensors.items()}
    with torch.no_grad():
        err_init = float((build_merged(leaves) - target).norm())
    best = {k: v.detach().clone() for k, v in leaves.items()}
    best_err = err_init

    opt = torch.optim.LBFGS(
        list(leaves.values()),
        lr=1.0,
        max_iter=1,
        max_eval=30,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = (build_merged(leaves) - target).square().sum()
        loss.backward()
        for v in leaves.values():
            # broadcast-heavy builders (DoRA norms) can leave non-contiguous
            # grads, which LBFGS's flat-grad gather rejects
            if v.grad is not None and not v.grad.is_contiguous():
                v.grad = v.grad.contiguous()
        return loss

    history: list[float] = []
    stall = 0
    for _ in range(iters):
        opt.step(closure)
        with torch.no_grad():
            err = float((build_merged(leaves) - target).norm())
        history.append(err)
        if err < best_err - tol * max(err_init, torch.finfo(target.dtype).tiny):
            stall = 0
        else:
            stall += 1
        if err < best_err:  # NaN never accepts
            best_err = err
            best = {k: v.detach().clone() for k, v in leaves.items()}
        if stall >= stall_limit:
            break
    return PeftRefineResult(
        tensors=best, err_init=err_init, err_final=best_err, err_history=history, steps=len(history)
    )


# ---------------------------------------------------------------------------
# Full per-tensor pipeline


@dataclass
class PeftDenseWarmstart:
    tensors_new: dict[str, torch.Tensor]  # suffix -> fitted stored tensor (fp32)
    target: torch.Tensor  # T = W_v + R_L (M_b - W_b) R_R
    dense: DenseTransportFit
    fit_history: list[float]
    deployed_err: float  # ||merged(fitted) - T||_F through the exact merged form
    deployed_err_pre_refine: float
    refine: PeftRefineResult | None


def _lora_suffix_tensors(p: LoRAStemParams) -> dict[str, torch.Tensor]:
    out = {"lora_down.weight": p.down, "lora_up.weight": p.up}
    if p.dora_scale is not None:
        out["dora_scale"] = p.dora_scale
    return out


def _lokr_suffix_tensors(p: LoKrStemParams) -> dict[str, torch.Tensor]:
    out = {}
    for suffix in ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "dora_scale"):
        v = getattr(p, suffix.removeprefix("lokr_") if suffix != "dora_scale" else "dora_scale")
        if v is not None:
            out[suffix] = v
    return out


def _params_with_suffix_tensors(kind: str, p, tensors: dict[str, torch.Tensor]):
    """A copy of ``p`` with its stored tensors replaced from a suffix dict."""
    if kind in ("lora", "dora"):
        return LoRAStemParams(
            down=tensors["lora_down.weight"],
            up=tensors["lora_up.weight"],
            alpha=p.alpha,
            dora_scale=tensors.get("dora_scale"),
            norm_eps=p.norm_eps,
        )
    return LoKrStemParams(
        alpha=p.alpha,
        w1=tensors.get("lokr_w1"),
        w1_a=tensors.get("lokr_w1_a"),
        w1_b=tensors.get("lokr_w1_b"),
        w2=tensors.get("lokr_w2"),
        w2_a=tensors.get("lokr_w2_a"),
        w2_b=tensors.get("lokr_w2_b"),
        dora_scale=tensors.get("dora_scale"),
        norm_eps=p.norm_eps,
    )


def dense_warmstart_for_peft_tensor(
    kind: str,
    p,
    w_base: torch.Tensor,
    w_variant: torch.Tensor,
    *,
    dense_iters: int = 30,
    dense_tol: float = 1e-9,
    als_iters: int = 4,
    refine_iters: int = 60,
) -> PeftDenseWarmstart:
    """dfit for one Linear tensor of a LoRA / DoRA / LoKr adapter: dense
    transport -> education-anchored target -> deployed-form student fit ->
    optional LBFGS refinement. Same target math as
    oft_dense_fit.dense_warmstart_for_tensor; only the student differs.

    ``refine_iters`` is skipped for plain LoRA regardless of its value: the
    truncated SVD is already the global optimum of that deployed form.
    """
    m_b = merged_peft(kind, w_base, p)
    dense = dense_transport_fit(w_base, w_variant, iters=dense_iters, tol=dense_tol)
    target = w_variant + dense.r_left @ (m_b - w_base) @ dense.r_right

    if kind == "lora":
        fit = fit_student_lora(w_variant, target, p.rank, p.scale)
        p_new = LoRAStemParams(down=fit.down, up=fit.up, alpha=p.alpha)
    elif kind == "dora":
        fit = fit_student_dora(w_variant, target, p.rank, p.scale, p.output_axis, iters=als_iters)
        wp = w_variant + p.scale * (fit.up @ fit.down)
        scale_t = dora_scale_from_axis_scale(fit.axis_scale, wp, p.output_axis, p.norm_eps)
        p_new = LoRAStemParams(down=fit.down, up=fit.up, alpha=p.alpha, dora_scale=scale_t, norm_eps=p.norm_eps)
    elif kind in ("lokr", "dora_lokr"):
        fit = fit_student_lokr(w_variant, target, p, iters=max(als_iters, 2))
        p_new = _lokr_params_from_fit(p, fit, w_variant)
    else:
        raise ValueError(f"unsupported kind {kind!r} (OFT goes through oft_dense_fit)")

    tensors_new = _lora_suffix_tensors(p_new) if kind in ("lora", "dora") else _lokr_suffix_tensors(p_new)
    pre_err = float((merged_peft(kind, w_variant, p_new) - target).norm())

    refine = None
    final_err = pre_err
    if refine_iters > 0 and kind != "lora":
        refine = refine_peft_fit(
            tensors_new,
            lambda leaves: merged_peft(kind, w_variant, _params_with_suffix_tensors(kind, p, leaves)),
            target,
            iters=refine_iters,
        )
        if refine.err_final <= pre_err:
            tensors_new = refine.tensors
            final_err = refine.err_final

    return PeftDenseWarmstart(
        tensors_new=tensors_new,
        target=target,
        dense=dense,
        fit_history=fit.objective_history,
        deployed_err=final_err,
        deployed_err_pre_refine=pre_err,
        refine=refine,
    )


def _lokr_params_from_fit(p: LoKrStemParams, fit: LoKrStudentFit, w_variant: torch.Tensor) -> LoKrStemParams:
    """Fitted full factors re-split into the adapter's STORED layout (exact:
    the fit already projected each decomposed factor to its inner rank)."""
    if p.w1 is not None:
        w1_kw = {"w1": fit.w1}
    else:
        b, a = _truncated_svd_factors(fit.w1, p.w1_a.shape[1])
        w1_kw = {"w1_a": a, "w1_b": b}
    if p.w2 is not None:
        w2_kw = {"w2": fit.w2}
    else:
        b, a = _truncated_svd_factors(fit.w2, p.w2_a.shape[1])
        w2_kw = {"w2_a": a, "w2_b": b}
    scale_t = None
    if p.dora_scale is not None:
        wp = w_variant + p.scale * torch.kron(fit.w1.contiguous(), fit.w2.contiguous())
        scale_t = dora_scale_from_axis_scale(fit.axis_scale, wp, p.output_axis, p.norm_eps)
    return LoKrStemParams(alpha=p.alpha, dora_scale=scale_t, norm_eps=p.norm_eps, **w1_kw, **w2_kw)
