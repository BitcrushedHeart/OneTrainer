"""Dense-transport least-squares warm start for OFT / DoRA-OFT adapter transfer.

The SoReal! distillation delta on the late qkv/mlp tensors is ~99.95% a DENSE
two-sided rotation ``W_v ~= R_L W_b R_R`` but only ~10-20% of the delta ENERGY
is block-expressible at the adapter partition (blockwise Procrustes residuals
0.87-0.94, modules/util/oft_conjugate.py). Conjugating the skew generator by
the block-diagonal part of the transport therefore barely moves the adapter.
This module goes through the dense transport instead:

1. ``dense_transport_fit``: recover the dense ``R_L`` (out x out) and ``R_R``
   (in x in) by damped Gauss-Newton in the singular frame of W_b. Naive SVD
   column alignment (R_L = U_v U_b^T) breaks inside clustered/degenerate
   singular subspaces, and plain alternating two-sided Procrustes crawls
   through the same near-gauge valleys (each near-equal singular pair lets
   R_L and R_R trade rotation without moving the residual). In W_b's singular
   coordinates the linearized problem A' S + S B' = delta' decouples into
   independent 2x2 systems per singular pair whose closed-form solution splits
   into a (d1+d2)/((s_i-s_j)^2 + lam) and a (d1-d2)/((s_i+s_j)^2 + lam) term:
   cancellation-free, and the Tikhonov damping lam sends exactly-degenerate
   pairs to ZERO rotation (the identity-biased minimal-norm answer) instead of
   an arbitrary gauge pick. On non-square tensors the factor components acting
   entirely outside W_b's column/row space are unidentified and stay at
   identity for the same reason. Steps are exponentials of the solved skews
   with backtracking, so the recorded residual is monotone and convergence is
   quadratic away from the damped directions.

2. The caller forms the education-anchored teacher target
   ``T = W_v + R_L @ (M_b - W_b) @ R_R`` (``M_b`` = the ORIGINAL adapter's
   merged teacher weight, DoRA rows included). Only the education delta rides
   the fitted transport, so the dense fit's residual -- distillation residue,
   not education -- never leaks into the target. The residual runs 2-3% of the
   distill delta while the education is only a fraction of that delta
   (merge/distill 0.14-0.59 across the lineup), so the old ``R_L M_b R_R``
   form polluted the target by up to ~5-20% of the education signal; the two
   coincide exactly when the transport fit is exact.

3. ``fit_student_als``: fit the DEPLOYED student form
   ``D(B, m) = rowscale(m) * (W_v @ blockdiag(B)^T)`` to ``T`` by alternating
   least squares. Block half-step: per input-block orthogonal Procrustes,
   ``B_j = polar(T_j^T A_j)`` with ``A = rowscale(m) W_v`` (batched
   Newton-Schulz polar -- pure matmul, no batched LAPACK; see oft_polar).
   Row half-step: closed-form per-row least squares.

4. ``encode_rotation_to_vec``: re-encode the fitted blocks as a stored oft_R
   vector via the exact Cayley inverse ``Q = (B - I)(B + I)^{-1}``
   (antisymmetrized), reporting the Cayley-Neumann-5 truncation gap and any
   block past the spectral clip (whose clip would bake in on reload).

5. ``refine_deployed_fit``: polish ``(vec, dora_multiplier)`` by LBFGS through
   the module's ACTUAL decode (scatter -> skew -> Neumann-5), minimizing the
   deployed error against ``T`` directly. The spectral clip is enforced by
   projecting iterates onto the clip ball (the projection is what the module's
   reload produces anyway, so kept iterates are always reload-faithful and the
   in-graph decode needs no clip). This removes the exact-Cayley vs Neumann-5
   encode mismatch and any ALS suboptimality; keep-best acceptance through the
   true decode guarantees the result is never worse than its input.

Pure tensor math -- no file I/O. Callers on this Windows box must obey the MKL
discipline (CLAUDE.md): materialize every safetensors tensor and close the
files BEFORE calling anything here, ``torch.set_num_threads(1)`` in scripts,
and keep TF32 off on CUDA (the ~1e-3 TF32 matmul error is not below the
signal these fits chase).
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.util.oft_align import AdapterOFTConfig, block_size_from_elements, merged_full, reconstruct_oft_rotation
from modules.util.oft_polar import polar_orthogonal

import torch

NEUMANN_TERMS = 5  # keep in lock-step with oft_align / OFTRotationModule


def neumann5_rotation(q: torch.Tensor) -> torch.Tensor:
    """The module's truncated Cayley map R(Q) = I + 2Q + 2Q^2 + 2Q^3 + Q^4 for
    a (..., b, b) skew stack -- identical coefficients to
    oft_align.reconstruct_oft_rotation (last term 1, not 2)."""
    b = q.shape[-1]
    r = torch.eye(b, dtype=q.dtype, device=q.device).expand_as(q).clone()
    qp = q
    r = r + 2.0 * qp
    for _ in range(2, NEUMANN_TERMS - 1):
        qp = qp @ q
        r = r + 2.0 * qp
    qp = qp @ q
    return r + qp


def _complete_orthonormal_basis(u: torch.Tensor, n: int) -> torch.Tensor:
    """Extend an orthonormal n x k matrix to a full n x n orthogonal basis.

    Complete QR of an orthonormal matrix reproduces its columns up to signs;
    only the complement columns are taken, so the leading k columns stay
    EXACTLY ``u`` (the singular frame the tangent solve indexes into)."""
    k = u.shape[-1]
    if k == n:
        return u
    q = torch.linalg.qr(u, mode="complete")[0]
    return torch.cat([u, q[:, k:]], dim=-1)


@dataclass
class DenseTransportFit:
    r_left: torch.Tensor  # (out, out) orthogonal
    r_right: torch.Tensor  # (in, in) orthogonal
    rel_residual: float  # ||R_L W_b R_R - W_v||_F / ||W_v - W_b||_F
    residual_history: list[float]  # absolute residual after each accepted step


def dense_transport_fit(
    w_base: torch.Tensor,
    w_variant: torch.Tensor,
    iters: int = 30,
    tol: float = 1e-9,
    damping_rcond: float | None = None,
) -> DenseTransportFit:
    """Damped Gauss-Newton fit of the dense two-sided rotation W_v ~= R_L W_b R_R.

    One economy SVD of W_b up front; afterwards the current fit's singular
    frame is tracked by rotation (W_cur = (R_L U) S (R_R^T V)^T -- the singular
    values never change), so each iteration is matmuls plus two matrix
    exponentials of solved skew steps. In the singular frame the linearization
    ``delta' = A' S + S B'`` decouples per singular pair (i, j):

        A'_ij = (s_j - s_i)(d1 + d2) / (2 ((s_i - s_j)^2 + lam))
              + (s_i + s_j)(d1 - d2) / (2 ((s_i + s_j)^2 + lam))
        B'_ij = (s_i - s_j)(d1 + d2) / (2 ((s_i - s_j)^2 + lam))
              + (s_i + s_j)(d1 - d2) / (2 ((s_i + s_j)^2 + lam))

    with d1 = delta'_ij, d2 = delta'_ji. The split is cancellation-free. The
    damping is Levenberg-Marquardt style, ``lam = lam_floor + (resid/||W_b|| *
    s_max)^2`` with ``lam_floor = (damping_rcond * s_max)^2`` (default rcond =
    sqrt(machine eps)): exactly-degenerate pairs get ZERO rotation (the
    identity-biased minimal-norm answer), near-degenerate pairs are deferred
    until the residual is small enough for their identification signal to
    stand above the linearization noise, everything else resolves in one or
    two iterations. Rectangular tails (i or j past
    min(out, in)) reduce to single damped divisions, and components acting
    entirely outside W_b's column/row space are unidentified and stay at
    identity. Each step is accepted only if it lowers the residual
    (backtracking halves it up to 6 times), so ``residual_history`` is
    monotone; the loop exits once an accepted step improves less than
    ``tol * ||W_v - W_b||`` or no step improves at all.
    """
    if w_base.shape != w_variant.shape:
        raise ValueError(f"shape mismatch: base {tuple(w_base.shape)} vs variant {tuple(w_variant.shape)}")
    out_f, in_f = w_base.shape
    k = min(out_f, in_f)
    dtype, device = w_base.dtype, w_base.device
    eye_l = torch.eye(out_f, dtype=dtype, device=device)
    eye_r = torch.eye(in_f, dtype=dtype, device=device)
    delta = float((w_variant - w_base).norm())
    if delta == 0.0:
        return DenseTransportFit(r_left=eye_l, r_right=eye_r, rel_residual=0.0, residual_history=[0.0])

    u_econ, s, vh_econ = torch.linalg.svd(w_base, full_matrices=False)  # single (non-batched) LAPACK call
    u_full = _complete_orthonormal_basis(u_econ, out_f)
    v_full = _complete_orthonormal_basis(vh_econ.transpose(-2, -1), in_f)

    rcond = float(torch.finfo(dtype).eps) ** 0.5 if damping_rcond is None else damping_rcond
    lam_floor = (rcond * float(s[0])) ** 2
    base_norm = float(w_base.norm())
    si = s.view(-1, 1)
    sj = s.view(1, -1)

    r_left = eye_l.clone()
    r_right = eye_r.clone()
    resid_cur = delta
    history: list[float] = []
    for _ in range(iters):
        # Levenberg-Marquardt damping tied to the CURRENT residual: the
        # linearization error in d_p is O(theta^2) with theta ~ resid/||W_b||,
        # so a singular pair is only resolvable once |s_i - s_j| stands above
        # theta * s_max -- damp anything finer to (identity-biased) zero now
        # and pick it up in a later iteration when the residual is small
        # enough for its signal to be real. lam ~ resid^2 is exactly the LM
        # schedule that keeps convergence quadratic on the resolvable part.
        theta_est = resid_cur / base_norm
        lam = lam_floor + (theta_est * float(s[0])) ** 2
        inv_minus = 1.0 / ((si - sj).square() + lam)
        inv_plus = 1.0 / ((si + sj).square() + lam)
        tail_l = (s / (s.square() + lam)).view(1, -1) if out_f > k else None  # per-column 1/s_j, damped
        tail_r = (s / (s.square() + lam)).view(-1, 1) if in_f > k else None  # per-row 1/s_i, damped

        u_cur = r_left @ u_full
        v_cur = r_right.transpose(-2, -1) @ v_full
        diff = w_variant - (r_left @ w_base @ r_right)
        d_p = u_cur.transpose(-2, -1) @ diff @ v_cur  # delta in the current singular frame

        d1 = d_p[:k, :k]
        d2 = d1.transpose(-2, -1)
        sym = (d1 + d2) * inv_minus * 0.5
        asym = (d1 - d2) * inv_plus * 0.5
        a_core = (sj - si) * sym + (si + sj) * asym
        b_core = (si - sj) * sym + (si + sj) * asym

        a_p = torch.zeros(out_f, out_f, dtype=dtype, device=device)
        a_p[:k, :k] = a_core
        if tail_l is not None:
            a_p[k:, :k] = d_p[k:, :k] * tail_l
            a_p[:k, k:] = -a_p[k:, :k].transpose(-2, -1)
        b_p = torch.zeros(in_f, in_f, dtype=dtype, device=device)
        b_p[:k, :k] = b_core
        if tail_r is not None:
            b_p[:k, k:] = d_p[:k, k:] * tail_r
            b_p[k:, :k] = -b_p[:k, k:].transpose(-2, -1)
        a_p = 0.5 * (a_p - a_p.transpose(-2, -1))  # fp hygiene: exact skewness
        b_p = 0.5 * (b_p - b_p.transpose(-2, -1))
        a_amb = u_cur @ a_p @ u_cur.transpose(-2, -1)
        b_amb = v_cur @ b_p @ v_cur.transpose(-2, -1)

        step = 1.0
        accepted = False
        for _bt in range(6):
            r_left_try = torch.matrix_exp(step * a_amb) @ r_left
            r_right_try = r_right @ torch.matrix_exp(step * b_amb)
            resid_try = float((r_left_try @ w_base @ r_right_try - w_variant).norm())
            if resid_try < resid_cur:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        improvement = resid_cur - resid_try
        r_left, r_right, resid_cur = r_left_try, r_right_try, resid_try
        history.append(resid_cur)
        if improvement <= tol * delta:
            break

    if not history:
        history = [resid_cur]
    return DenseTransportFit(
        r_left=r_left,
        r_right=r_right,
        rel_residual=resid_cur / delta,
        residual_history=history,
    )


def student_merged(w: torch.Tensor, r_blocks: torch.Tensor, mult: torch.Tensor | None = None) -> torch.Tensor:
    """The deployed student form rowscale(m) * (W @ blockdiag(B)^T), matching
    DoRAOFTModule.apply_to_module (rotation first, per-output-row scale after)."""
    merged = merged_full(w, r_blocks)
    if mult is not None:
        merged = merged * mult.view(-1, 1)
    return merged


@dataclass
class StudentFit:
    r_blocks: torch.Tensor  # (r, b, b) block-orthogonal
    multiplier: torch.Tensor  # (out,) per-output-row, ones when dora=False
    objective_history: list[float]  # ||D - T||_F after each full iteration


def fit_student_als(
    w_variant: torch.Tensor,
    target: torch.Tensor,
    block_size: int,
    dora: bool = True,
    iters: int = 4,
) -> StudentFit:
    """Alternating least squares fit of the student form to a dense target.

    Block half-step (given m): each input-block objective decouples;
    ``||A_j B_j^T - T_j||`` with ``A = rowscale(m) W_v`` is orthogonal
    Procrustes with solution ``B_j = polar(T_j^T A_j)`` -- computed as a
    batched Newton-Schulz polar (SVD fallback per matrix on failure).
    Row half-step (given B): per-row closed-form least squares
    ``m_i = <D0_i, T_i> / <D0_i, D0_i>`` with ``D0 = W_v blockdiag(B)^T``
    (rows of W_v with zero norm keep m_i = 1). Both half-steps are exactly
    optimal given the other, so the recorded objective never increases.
    """
    out_f, in_f = w_variant.shape
    if in_f % block_size:
        raise ValueError(f"block_size {block_size} does not divide in_features {in_f}")
    r = in_f // block_size
    mult = torch.ones(out_f, dtype=w_variant.dtype, device=w_variant.device)
    w_blocks = w_variant.reshape(out_f, r, block_size)
    t_blocks = target.reshape(out_f, r, block_size)
    history: list[float] = []
    r_blocks = torch.eye(block_size, dtype=w_variant.dtype, device=w_variant.device).repeat(r, 1, 1)
    for _ in range(iters):
        a_blocks = w_blocks * mult.view(-1, 1, 1)
        m = torch.einsum("orp,orq->rpq", t_blocks, a_blocks)  # M_j = T_j^T A_j
        r_blocks = polar_orthogonal(m)
        d0 = torch.einsum("orc,rkc->ork", w_blocks, r_blocks).reshape(out_f, in_f)
        if dora:
            num = (d0 * target).sum(dim=-1)
            den = (d0 * d0).sum(dim=-1)
            mult = torch.where(den > 0.0, num / den.clamp_min(torch.finfo(den.dtype).tiny), torch.ones_like(num))
        history.append(float((mult.view(-1, 1) * d0 - target).norm()))
    return StudentFit(r_blocks=r_blocks, multiplier=mult, objective_history=history)


@dataclass
class EncodeResult:
    vec: torch.Tensor  # (r, b*(b-1)//2) stored oft_R params
    neumann_dev: float  # max ||N5(Q') - B||_F over blocks: the truncation gap
    sigma_max: float  # max per-block spectral norm of the encoded generator
    clipped_blocks: int  # blocks whose sigma exceeds cfg.clip (clip bakes in)


def encode_rotation_to_vec(r_blocks: torch.Tensor, cfg: AdapterOFTConfig) -> EncodeResult:
    """Exact Cayley inverse ``Q = (B - I)(B + I)^{-1}`` -> stored skew vector.

    The module decodes with the Neumann-5 polynomial, not the exact Cayley
    map; at warm-start angles the gap is ~||Q||^4 and is measured and returned
    (``neumann_dev``), never assumed. Solves run one block at a time -- never
    batched LAPACK (Windows-MKL discipline). Blocks whose generator exceeds
    the spectral clip are counted: their clip re-applies on reload, so the
    written rotation is NOT faithful for them.
    """
    bs = r_blocks.shape[-1]
    eye = torch.eye(bs, dtype=r_blocks.dtype, device=r_blocks.device)
    q = torch.empty_like(r_blocks)
    for i in range(r_blocks.shape[0]):
        b = r_blocks[i]
        # Q (B + I) = (B - I)  ->  (B + I)^T Q^T = (B - I)^T
        q[i] = torch.linalg.solve((b + eye).transpose(-2, -1), (b - eye).transpose(-2, -1)).transpose(-2, -1)
    q = 0.5 * (q - q.transpose(-2, -1))  # skew up to fp roundoff: antisymmetrize

    neumann_dev = float((neumann5_rotation(q) - r_blocks).flatten(1).norm(dim=1).max())
    # exact spectral norm of the skew generator, same eigvalsh path oft_align uses
    ev = torch.linalg.eigvalsh(-torch.matmul(q, q))
    sigma = ev[:, -1].clamp_min(0).sqrt()
    sigma_max = float(sigma.max())
    clipped = int((sigma > cfg.clip).sum()) if cfg.clip is not None else 0

    rows, cols = torch.triu_indices(bs, bs, 1, device=q.device)
    scaling = 2.0 * (bs - 1) ** 0.5 if cfg.scaled else 1.0
    return EncodeResult(
        vec=q[:, rows, cols] * scaling,
        neumann_dev=neumann_dev,
        sigma_max=sigma_max,
        clipped_blocks=clipped,
    )


def _scatter_skew(vec: torch.Tensor, bs: int, scaled: bool) -> torch.Tensor:
    """Differentiable stored-vec -> skew stack (NO clip): un-scale, scatter the
    strict upper triangle, antisymmetrize. Autograd-safe counterpart of the
    first half of oft_conjugate.clipped_skew_from_vec."""
    scaling = 2.0 * (bs - 1) ** 0.5 if scaled else 1.0
    v = vec / scaling
    r = v.shape[0]
    rows, cols = torch.triu_indices(bs, bs, 1, device=v.device)
    bidx = torch.arange(r, device=v.device)[:, None]
    q = torch.zeros(r, bs, bs, device=v.device, dtype=v.dtype).index_put((bidx, rows, cols), v)
    return q - q.transpose(-2, -1)


@dataclass
class RefineResult:
    vec: torch.Tensor  # refined oft_R params; clip-projected, so reload-faithful
    multiplier: torch.Tensor | None  # refined dora_multiplier, None for plain OFT
    err_init: float  # deployed err of the (clip-projected) init through the true decode
    err_final: float  # deployed err of the returned iterate; never above err_init
    err_history: list[float]  # exact deployed err after each LBFGS step
    projected_blocks: int  # blocks of the returned iterate sitting ON the clip
    steps: int  # LBFGS steps actually taken


def refine_deployed_fit(
    vec_init: torch.Tensor,
    mult_init: torch.Tensor | None,
    w_variant: torch.Tensor,
    target: torch.Tensor,
    cfg: AdapterOFTConfig,
    *,
    iters: int = 60,
    tol: float = 1e-10,
    history_size: int = 10,
    stall_limit: int = 8,
) -> RefineResult:
    """LBFGS polish of ``(vec, mult)`` through the module's actual decode path.

    The in-graph forward is scatter -> skew -> Neumann-5 -> W_v blockdiag ->
    rowscale, WITHOUT the spectral clip: instead every iterate (including the
    init) is projected onto the clip ball under ``no_grad`` (per-block
    ``Q *= clip/sigma`` for over-clip blocks -- exactly what the module's
    reload computes, so the projection changes nothing the deployed form can
    see and kept iterates are always reload-faithful). Inside the ball the
    smooth decode IS the exact decode, so the recorded errors are honest
    deployed errors, not surrogate values. A cheap certificate skips the
    eigen-solve when possible: for skew Q, ``sigma_max <= ||Q||_F / sqrt(2)``
    (eigenvalues pair as +-i*sigma), so blocks certified inside the clip never
    touch LAPACK. Acceptance is keep-best on the exact deployed error, so
    ``err_final <= err_init`` always; the loop stops after ``stall_limit``
    steps without a meaningful (``tol``-relative) improvement.

    In-place projection of the LBFGS parameters mildly perturbs the optimizer's
    curvature memory; in the warm-start regime (sigma far below the clip) the
    projection never fires and LBFGS runs undisturbed. Windows-MKL discipline:
    the only LAPACK call (``eigvalsh``) runs under ``no_grad`` on materialized
    tensors, batched exactly like oft_align's reconstruction path.
    """
    bs = block_size_from_elements(vec_init.shape[-1])

    def _project_(v_param: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Clip-ball projection of the stored params, in place. Returns
        (projected block count, post-projection skew stack)."""
        q = _scatter_skew(v_param, bs, cfg.scaled)
        if cfg.clip is None:
            return 0, q
        fro_bound = q.flatten(1).norm(dim=1) * (0.5**0.5)  # sigma_max <= ||Q||_F/sqrt(2)
        if not bool((fro_bound > cfg.clip).any()):
            return 0, q
        ev = torch.linalg.eigvalsh(-torch.matmul(q, q))
        sigma = ev[:, -1].clamp_min(0).sqrt()
        over = sigma > cfg.clip
        n_over = int(over.sum())
        if n_over:
            factor = torch.where(over, cfg.clip / sigma, torch.ones_like(sigma))
            v_param.mul_(factor.view(-1, 1))
            q = q * factor.view(-1, 1, 1)
        return n_over, q

    def _deployed_err(q: torch.Tensor, m: torch.Tensor | None) -> float:
        return float((student_merged(w_variant, neumann5_rotation(q), m) - target).norm())

    v = vec_init.detach().clone().requires_grad_(True)
    m = mult_init.detach().clone().requires_grad_(True) if mult_init is not None else None
    with torch.no_grad():
        proj_n, q0 = _project_(v)
        err_init = _deployed_err(q0, m)
    best_v = v.detach().clone()
    best_m = m.detach().clone() if m is not None else None
    best_err, best_proj = err_init, proj_n

    params = [v] if m is None else [v, m]
    # max_iter=1 so projection + keep-best interleave with every LBFGS
    # iteration. max_eval MUST be raised: torch defaults it to max_iter*5//4=1,
    # which the entry evaluation consumes entirely, leaving _strong_wolfe a
    # zero-eval budget -- it then returns t=0 whenever the raw trial step is
    # uphill and the optimizer never moves.
    opt = torch.optim.LBFGS(
        params, lr=1.0, max_iter=1, max_eval=30, history_size=history_size, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        q = _scatter_skew(v, bs, cfg.scaled)
        loss = (student_merged(w_variant, neumann5_rotation(q), m) - target).square().sum()
        loss.backward()
        return loss

    history: list[float] = []
    stall = 0
    for _ in range(iters):
        opt.step(closure)
        with torch.no_grad():
            proj_n, q = _project_(v)
            err = _deployed_err(q, m)
        history.append(err)
        if err < best_err - tol * max(err_init, torch.finfo(v.dtype).tiny):
            stall = 0
        else:
            stall += 1
        if err < best_err:  # NaN never accepts: NaN < x is False
            best_err, best_proj = err, proj_n
            best_v = v.detach().clone()
            if m is not None:
                best_m = m.detach().clone()
        if stall >= stall_limit:
            break

    return RefineResult(
        vec=best_v,
        multiplier=best_m,
        err_init=err_init,
        err_final=best_err,
        err_history=history,
        projected_blocks=best_proj,
        steps=len(history),
    )


@dataclass
class DenseWarmstart:
    """Everything the retarget/benchmark tooling needs for one late tensor."""

    vec_new: torch.Tensor  # fitted oft_R params, compute dtype
    mult_new: torch.Tensor | None  # fitted dora_multiplier, None for plain OFT
    target: torch.Tensor  # T = W_v + R_L (M_b - W_b) R_R, the education-anchored teacher
    dense: DenseTransportFit  # step-1 fit (rotations still attached; caller frees)
    student: StudentFit
    encode: EncodeResult
    deployed_err: float  # ||D(vec_new, mult_new) - T||_F through the decode path (final)
    deployed_err_pre_refine: float  # same, for the ALS+encode result before refinement
    refine: RefineResult | None  # None when refine_iters == 0


def dense_warmstart_for_tensor(
    vec: torch.Tensor,
    mult: torch.Tensor | None,
    w_base: torch.Tensor,
    w_variant: torch.Tensor,
    cfg: AdapterOFTConfig,
    *,
    dense_iters: int = 30,
    dense_tol: float = 1e-9,
    als_iters: int = 4,
    refine_iters: int = 0,
) -> DenseWarmstart:
    """The full per-tensor pipeline: dense transport -> teacher target ->
    student ALS -> Cayley re-encode -> deployed-form self-check -> optional
    LBFGS refinement through the true decode path (``refine_iters > 0``).

    ``deployed_err`` is computed through the same decode the module runs
    (clip + Neumann-5) on the written vector, so it is the honest error of
    what actually gets written, not of the pre-encode fit. Refinement is
    keep-best on that same error, so ``deployed_err <=
    deployed_err_pre_refine`` always; the ALS+encode result is kept whenever
    refinement cannot improve on it.
    """
    from modules.util.oft_conjugate import clipped_skew_from_vec

    bs = block_size_from_elements(vec.shape[-1])
    q_b, _ = clipped_skew_from_vec(vec, cfg)
    m_b = student_merged(w_base, neumann5_rotation(q_b), mult)

    dense = dense_transport_fit(w_base, w_variant, iters=dense_iters, tol=dense_tol)
    # Education-anchored target: only the education delta (M_b - W_b) rides
    # the fitted transport; the transport's own residual never enters it.
    target = w_variant + dense.r_left @ (m_b - w_base) @ dense.r_right

    student = fit_student_als(w_variant, target, bs, dora=mult is not None, iters=als_iters)
    encode = encode_rotation_to_vec(student.r_blocks, cfg)

    mult_new = student.multiplier if mult is not None else None
    deployed = student_merged(w_variant, reconstruct_oft_rotation(encode.vec, cfg), mult_new)
    pre_err = float((deployed - target).norm())

    vec_out, mult_out, final_err, refine = encode.vec, mult_new, pre_err, None
    if refine_iters > 0:
        refine = refine_deployed_fit(encode.vec, mult_new, w_variant, target, cfg, iters=refine_iters)
        # err_final <= err_init = pre_err by keep-best (the init projection is
        # deployed-identical to the module's reload clip); the guard only
        # covers fp roundoff between the two evaluation paths.
        if refine.err_final <= pre_err:
            vec_out, mult_out, final_err = refine.vec, refine.multiplier, refine.err_final
    return DenseWarmstart(
        vec_new=vec_out,
        mult_new=mult_out,
        target=target,
        dense=dense,
        student=student,
        encode=encode,
        deployed_err=final_err,
        deployed_err_pre_refine=pre_err,
        refine=refine,
    )
