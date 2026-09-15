import math

from modules.util.bf16_stochastic_rounding import add_stochastic_

import torch
import torch.nn as nn

# Shared upper-triangle index tables, keyed by (block_size, device). Every
# adapted Linear of a given block size reads the same two int64 tensors instead
# of registering its own copy: on Z-Image (210 adapted Linears, blocks 768/1024)
# per-module buffers cost ~1 GiB of VRAM for data that only has 2 distinct values.
# Modules hold plain attribute references (not buffers), filled eagerly in
# __init__ and re-bound in _apply after .to()/.cuda(); the forward path only
# reads attributes, so no dict lookup ever happens inside a compiled region.
_TRIU_INDEX_CACHE: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

# Power-iteration steps applied to the spectral-clip state (u_state/v_state) once
# per optimizer update, by OFTRotationModule.advance_spectral_state. Forwards only
# read the state, so every forward of one update window (checkpoint recompute, DPO
# reference passes, validation) sees the same clip. See
# tests/util/test_oft_spectral_state.py for the drift-tracking evidence.
SPECTRAL_STEPS_PER_UPDATE = 2

# Rotation cache (config oft_rotation_cache): the Linear rotation R(Q) is computed once
# per optimizer update into a bf16 leaf that the forward reads instead of running the
# Cayley per micro-step; dL/dR accumulates (stochastically rounded) until the update,
# where it is pulled back to dL/dQ through one fp32 R(Q). See OFTRotationModule.
ROTATION_CACHE_SR_SEED_SALT = 0x5EED0F7
# Stochastic-rounding adds are sliced along the block dim to at most this many
# elements, bounding the float32 + int32 transients of the largest modules.
ROTATION_CACHE_SR_CHUNK_NUMEL = 1 << 22
# Stale-cache guard threshold, in units of the leaf dtype's half-ulp at each entry
# of the freshly computed R (see OFTRotationModule.rotation_cache_staleness).
ROTATION_CACHE_STALE_RATIO = 1.5
_SIGNIFICAND_BITS = {torch.bfloat16: 8, torch.float16: 11, torch.float32: 24, torch.float64: 53}

# Dedicated generator for the accumulator's stochastic rounding, separate from the
# optimizer's (bf16_stochastic_rounding.generator) so enabling the cache leaves the
# optimizer's rounding stream untouched.
_rotation_cache_generator: torch.Generator | None = None


def _normalized_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def set_rotation_cache_seed(seed: int, device: torch.device | str) -> None:
    """Reseed the accumulator's rounding generator; the trainer calls this once per micro-step."""
    global _rotation_cache_generator
    device = _normalized_device(device)
    if _rotation_cache_generator is None or _rotation_cache_generator.device != device:
        _rotation_cache_generator = torch.Generator(device=device)
    _rotation_cache_generator.manual_seed(seed ^ ROTATION_CACHE_SR_SEED_SALT)


def rotation_cache_generator(device: torch.device | str) -> torch.Generator:
    global _rotation_cache_generator
    device = _normalized_device(device)
    if _rotation_cache_generator is None or _rotation_cache_generator.device != device:
        _rotation_cache_generator = torch.Generator(device=device)
        _rotation_cache_generator.manual_seed(ROTATION_CACHE_SR_SEED_SALT)
    return _rotation_cache_generator


def shared_triu_indices(block_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """(rows, cols) of the strict upper triangle of a block_size square, shared per device."""
    device = torch.device(device)
    key = (block_size, device)
    cached = _TRIU_INDEX_CACHE.get(key)
    if cached is None:
        rows, cols = torch.triu_indices(block_size, block_size, 1, device=device)
        cached = (rows, cols)
        _TRIU_INDEX_CACHE[key] = cached
    return cached


class MultiplicativeDropoutLayer(nn.Module):
    """
    Implements the multiplicative dropout layer for OFT.
    This layer randomly replaces the learned rotation blocks with identity matrices during training.
    """

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            if x.shape[-1] != x.shape[-2]:
                raise ValueError("The last two dimensions of input should be the same!")

            D, H, _ = x.shape

            if D == 1:
                return x

            keep_prob = 1.0 - self.p

            # This 'stochastic_mask' has 1s for blocks to keep, and 0s for blocks to replace with Identity.
            stochastic_mask = torch.empty(D, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(p=keep_prob)

            eye_matrix = torch.eye(H, device=x.device, dtype=x.dtype).repeat(D, 1, 1)

            x = stochastic_mask * x + (1 - stochastic_mask) * eye_matrix

        return x


class OFTRotationModule(nn.Module):
    rows: torch.Tensor  # shared triu row indices, see _TRIU_INDEX_CACHE
    cols: torch.Tensor

    def __init__(
        self,
        r,
        n_elements,
        block_size,
        in_features,
        block_share=False,
        oft_scaled=False,
        use_cayley_neumann=True,
        num_cayley_neumann_terms=5,
        oft_cans=False,
        dropout_probability=0.0,
        oft_clipped_norm: float | None = 0.95,
    ):
        super().__init__()
        self.r = r
        self.n_elements = n_elements
        self.block_size = block_size
        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty(r, n_elements))
        self.block_share = block_share
        if oft_scaled:
            # Register a persistent buffer to indicate this module uses Scaled OFT.
            # This embeds the scaling configuration directly into the state_dict,
            # allowing inference tools to automatically detect scaled oft.
            self.register_buffer("scaled_oft", torch.tensor(True))
        self.oft_scaled = oft_scaled
        self.use_cayley_neumann = use_cayley_neumann and not oft_cans
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        self.oft_cans = oft_cans
        if oft_cans:
            self.register_buffer("cans_oft", torch.tensor(True))
        # Indices for the upper triangle (excluding diagonal). Plain attributes
        # referencing the shared per-(block_size, device) table, NOT buffers:
        # nn.Module.to() would otherwise materialize a private copy per module.
        # Re-bound to the weight's device by _apply.
        self._bind_triu_indices(torch.device("cpu"))
        self.dropout = MultiplicativeDropoutLayer(p=dropout_probability)
        # Spectral clipping is a Cayley-Neumann convergence safeguard only: it
        # exists because the Neumann series diverges once ||Q||_2 >= 1. CANS
        # self-normalizes (g_norm) and the exact solver is unconditionally
        # stable, so neither needs the clip. Gate it to the Neumann path so our
        # merge of PR #1492 + #1512 reproduces each PR exactly in its own mode
        # (and so the clipped_oft marker isn't written for non-clipped adapters).
        self.oft_clipped_norm = oft_clipped_norm if self.use_cayley_neumann else None
        if self.oft_clipped_norm is not None:
            # Persistent buffer: embeds the clip value in the state_dict so
            # inference tools can detect and reproduce the clipped rotation.
            self.register_buffer("clipped_oft", torch.tensor(self.oft_clipped_norm))
            # Power-iteration state for spectral-norm estimation; one state per block.
            u = torch.randn(r, block_size)
            u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.register_buffer("u_state", u, persistent=False)
            v = torch.randn(r, block_size)
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.register_buffer("v_state", v, persistent=False)
        if not self.use_cayley_neumann:
            id_mat = torch.eye(block_size).unsqueeze(0).expand(r, block_size, block_size)
            self.register_buffer("id_mat", id_mat, persistent=False)
        # Bumped whenever u_state/v_state change; the rotation cache records the value
        # it was refreshed against, so a clip-state change without a refresh is caught.
        self._spectral_state_version = 0
        # Rotation cache, off unless enable_rotation_cache() is called. Plain attributes,
        # not a Parameter or buffer: a Parameter would leak into state_dict, the optimizer,
        # clipping, EMA and the DPO reference snapshot; a buffer stops being a leaf when
        # .to() moves it. _apply moves them instead, keeping the leaf's identity and hook.
        self.rotation_leaf: torch.Tensor | None = None
        self.rotation_grad_acc: torch.Tensor | None = None
        # True while the forward reads the leaf. Per-module Python bool: flipping it
        # (reference passes) specializes the compiled graph, exactly like the weight
        # dtype swap those passes already cause, so the graph count is unchanged.
        self.rotation_cache_active = False
        self._rotation_leaf_hook = None
        self._rotation_cache_state_version = -1

    def _bind_triu_indices(self, device: torch.device) -> None:
        self.rows, self.cols = shared_triu_indices(self.block_size, device)

    def _apply(self, fn, recurse=True):
        # .to()/.cuda()/.cpu()/.to_empty() all funnel through _apply. The index
        # tables are integer attributes outside the buffer registry, so they are
        # never dtype-cast; only their device follows the weight. The rotation
        # cache likewise follows the device only (its dtype is the train dtype).
        result = super()._apply(fn, recurse)
        self._bind_triu_indices(self.weight.device)
        self._move_rotation_cache(self.weight.device)
        return result

    def _move_rotation_cache(self, device: torch.device) -> None:
        leaf = self.rotation_leaf
        if leaf is not None and leaf.device != device:
            moved = leaf.detach().to(device)
            if torch._has_compatible_shallow_copy_type(leaf, moved):
                # Same Python object: the grad hook stays registered and compiled
                # graphs (which lift the attribute as an input) see the new storage.
                leaf.data = moved
            else:
                self._set_rotation_leaf(moved.requires_grad_(leaf.requires_grad))
        if self.rotation_grad_acc is not None and self.rotation_grad_acc.device != device:
            self.rotation_grad_acc = self.rotation_grad_acc.to(device)

    def _triu_indices_on(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # Fast path (always taken after a normal .to()): the bound attributes.
        # The fallback covers a weight moved without _apply (e.g. load_state_dict
        # with assign=True onto another device).
        if self.rows.device == device:
            return self.rows, self.cols
        return shared_triu_indices(self.block_size, device)

    @torch.no_grad()
    def warm_start_spectral_state(self, steps: int = 256):
        """Re-converge the power-iteration state (u_state/v_state) after a load.

        The state is a non-persistent buffer, so resuming from a backup restarts
        it from random vectors. At init that is harmless (Q == 0), but on a
        trained adapter a random restart underestimates sigma, which silently
        lifts the spectral clip until the per-update iteration re-converges —
        every previously clipped block snaps to a stronger rotation for hundreds
        of steps. Called from OFTModule.load_state_dict.

        Pure batched matmul on purpose: no LAPACK, so it is safe to run while
        safetensors files may still be open (Windows MKL segfaults when mmap IO
        interleaves with SVD/eigh).
        """
        self._iterate_spectral_state(steps)

    @torch.no_grad()
    def advance_spectral_state(self, steps: int = SPECTRAL_STEPS_PER_UPDATE):
        """Advance the power-iteration state against the current weight.

        The forward only reads u_state/v_state, so the trainer calls this once per
        optimizer update (BaseModelSetup.after_optimizer_step_adapters) to track
        the weight as it trains. No-op when the spectral clip is disabled.
        """
        self._iterate_spectral_state(steps)

    def _iterate_spectral_state(self, steps: int):
        if self.oft_clipped_norm is None:
            return
        self._spectral_state_version += 1
        # Same scaled Q as the forward, iterated in float32 whatever the weight dtype.
        scaling_factor = 2 * math.sqrt(self.block_size - 1) if self.oft_scaled else 1
        Q = (self.weight.detach() / scaling_factor).to(torch.float32)
        Q_skew = self._pytorch_skew_symmetric(Q, self.block_size)
        u = self.u_state.to(device=Q_skew.device, dtype=torch.float32).unsqueeze(-1)
        v = self.v_state.to(device=Q_skew.device, dtype=torch.float32).unsqueeze(-1)
        for _ in range(steps):
            v, u = self._power_iteration_step(Q_skew, u, v)
        self.v_state.copy_(v.squeeze(-1).to(self.v_state.dtype))
        self.u_state.copy_(u.squeeze(-1).to(self.u_state.dtype))

    @staticmethod
    def _power_iteration_step(
        Q_skew: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One power-iteration step on (B, n, 1) vectors; returns (next_v, next_u).

        Degeneracy guard: a (near-)zero block keeps its current unit vectors
        instead of collapsing the state to zero.
        """
        # Update v (right singular vector)
        v_raw = torch.bmm(Q_skew.mT, u)
        v_norm = torch.linalg.vector_norm(v_raw, dim=1, keepdim=True)
        next_v = torch.where(v_norm >= 1e-6, v_raw / v_norm.clamp_min(1e-8), v)
        # Update u (left singular vector)
        u_raw = torch.bmm(Q_skew, next_v)
        u_norm = torch.linalg.vector_norm(u_raw, dim=1, keepdim=True)
        next_u = torch.where(u_norm >= 1e-6, u_raw / u_norm.clamp_min(1e-8), u)
        return next_v, next_u

    def _pytorch_skew_symmetric(self, vec, block_size):
        batch_size = vec.shape[0]
        matrix = torch.zeros(batch_size, block_size, block_size, device=vec.device, dtype=vec.dtype)

        # the following lines are equivalent to "matrix[:, rows, cols] = vec",
        # but they work around a pytorch issue: https://github.com/pytorch/pytorch/issues/169179
        rows, cols = self._triu_indices_on(vec.device)
        batch_idx = torch.arange(batch_size, device=vec.device)[:, None]
        matrix = matrix.index_put((batch_idx, rows, cols), vec)

        matrix = matrix - matrix.transpose(-2, -1)
        return matrix

    def _pytorch_skew_symmetric_inv(self, matrix, block_size):
        # Extract the upper triangular elements
        rows, cols = self._triu_indices_on(matrix.device)
        vec = matrix[:, rows, cols]
        return vec

    @torch.no_grad()
    def _spectral_norm(self, Q_skew):
        """One read-only power-iteration step; returns the (right, left) singular vectors per block.

        Never writes u_state/v_state, in any mode, so a forward and its checkpoint
        recompute, DPO reference passes and validation all see the same clip. The
        state advances once per optimizer update (advance_spectral_state).
        """
        u = self.u_state.unsqueeze(-1).to(Q_skew.dtype)
        v = self.v_state.unsqueeze(-1).to(Q_skew.dtype)
        return self._power_iteration_step(Q_skew, u, v)

    def _cans_newton_schulz_iteration(
        self,
        G: torch.Tensor,
        steps: int = 7,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Chebyshev-Optimized Newton-Schulz iteration with a dynamically computed Chebyshev lower bound.
        Optimized for G = I + Q (where Q is skew-symmetric).
        """
        original_dtype = G.dtype
        X = G

        # Max row sum is guaranteed to be >= the maximum singular value of X.
        # detach: the normalization is a preconditioning scale only; letting grads
        # flow through g_norm (a function of Q) creates a feedback term in the
        # rotation gradient that spirals as Q grows (Koratahiu PR #1512).
        g_norm = X.abs().sum(dim=-1, keepdim=True).amax(dim=-2, keepdim=True).clamp_min(eps).detach()
        X = X / g_norm

        # Since min_singular_value(I + Q) >= 1, the min_singular_value of normalized X
        # is guaranteed to be >= 1 / ||G||_F.
        lower_bound = 1.0 / g_norm
        upper_bound = 1

        for _ in range(steps):
            lb, ub = lower_bound, upper_bound
            lb_ub = lb * ub
            e_sq = (lb**2 + lb_ub + ub**2) / 3.0
            K = 2.0 * e_sq**1.5
            L = lb_ub * (lb + ub)
            denom = K + L
            alpha = 6.0 / denom
            c1 = alpha * e_sq
            c3 = -alpha / 3.0

            A = torch.bmm(X, X.mT)
            X = c1 * X + c3 * torch.bmm(A, X)

            # Dynamically update bounds for the next step
            eps_val = (K - L) / denom

            # bmm acts as an opaque boundary, forcing Inductor to
            # materialize eps_val and severing the exponential AST tree.
            # Shape is (B, 1, 1), so ones_like acts as an identity.
            eps_val = torch.bmm(eps_val, torch.ones_like(eps_val))

            lower_bound = 1 - eps_val
            upper_bound = 1 + eps_val

        return X.to(original_dtype)

    def _cayley_batch(
        self,
        Q: torch.Tensor,
        block_size: int,
        use_cayley_neumann: bool = True,
        num_neumann_terms: int = 5,
        oft_cans: bool = False,
    ) -> torch.Tensor:
        """
        Perform the Cayley parametrization on a batch of skew-symmetric matrices.
        """
        b, _ = Q.shape
        previous_dtype = Q.dtype

        Q_skew = self._pytorch_skew_symmetric(Q, block_size)

        if self.oft_clipped_norm is not None:
            # The Neumann series only converges if the spectral norm ||Q||_2 < 1.
            # Estimate it with one read-only power-iteration step and clip Q to the
            # configured max norm; this also bounds the learned rotation angle
            # (angle ~= 2*arctan(sigma)) regardless of orthogonalization method.
            v_vec, u_vec = self._spectral_norm(Q_skew)
            u_raw = torch.bmm(Q_skew, v_vec)
            sigma = torch.sum(u_vec * u_raw, dim=1, keepdim=True)
            max_norm = self.oft_clipped_norm
            Q_skew = Q_skew * (max_norm / torch.clamp(sigma, min=max_norm))

        if use_cayley_neumann and num_neumann_terms == 5:
            # R = I + 2Q + 2Q^2 + 2Q^3 + Q^4, Horner-factored as
            # R = I + 2Q + Q^2 (2I + 2Q + Q^2): two batched matmuls instead of three.
            Q_squared = torch.bmm(Q_skew, Q_skew)
            inner = torch.add(Q_squared, Q_skew, alpha=2.0)
            inner.diagonal(dim1=-2, dim2=-1).add_(2.0)
            R = torch.baddbmm(Q_skew, Q_squared, inner, beta=2.0, alpha=1.0)
            R.diagonal(dim1=-2, dim2=-1).add_(1.0)
        elif use_cayley_neumann:
            R = torch.eye(block_size, device=Q.device, dtype=Q.dtype).repeat(b, 1, 1)
            if num_neumann_terms > 1:
                R.add_(Q_skew, alpha=2.0)
                if num_neumann_terms > 2:
                    Q_squared = torch.bmm(Q_skew, Q_skew)
                    R.add_(Q_squared, alpha=2.0)

                    Q_power = Q_squared
                    for _ in range(3, num_neumann_terms - 1):
                        Q_power = torch.bmm(Q_power, Q_skew)
                        R.add_(Q_power, alpha=2.0)
                    Q_power = torch.bmm(Q_power, Q_skew)
                    R.add_(Q_power)
        elif oft_cans:
            G = self.id_mat + Q_skew
            # Empirically, BF16 requires 5 steps to converge to ortho error ~1e-2 (its limit)
            # While FP32 takes 7 steps to converge to ortho error ~1e-6
            steps = 5 if G.dtype == torch.bfloat16 else 7
            R_half = self._cans_newton_schulz_iteration(G=G, steps=steps)
            # Squaring the matrix doubles the rotation range from (-90°, 90°) to (-180°, 180°) and
            # matches Cayley (I + 2Q).
            R = torch.bmm(R_half, R_half)
        else:
            R = torch.linalg.solve(self.id_mat + Q_skew, self.id_mat - Q_skew, left=False)

        return R.to(previous_dtype)

    # ------------------------------------------------------------------ #
    # Rotation cache
    # ------------------------------------------------------------------ #

    @property
    def rotation_cache_enabled(self) -> bool:
        return self.rotation_leaf is not None

    def _set_rotation_leaf(self, leaf: torch.Tensor | None) -> None:
        if self._rotation_leaf_hook is not None:
            self._rotation_leaf_hook.remove()
            self._rotation_leaf_hook = None
        self.rotation_leaf = leaf
        if leaf is not None:
            self._rotation_leaf_hook = leaf.register_post_accumulate_grad_hook(self._accumulate_rotation_grad)

    def enable_rotation_cache(self, dtype: torch.dtype) -> None:
        """Allocate the cached rotation in ``dtype`` (the train dtype) and fill it from Q."""
        if self.rotation_leaf is None or self.rotation_leaf.dtype != dtype:
            shape = (self.weight.shape[0], self.block_size, self.block_size)
            leaf = torch.empty(shape, dtype=dtype, device=self.weight.device)
            self._set_rotation_leaf(leaf.requires_grad_(self.weight.requires_grad))
            self.rotation_grad_acc = None
        self.rotation_cache_active = True
        self.refresh_rotation_cache()

    def disable_rotation_cache(self) -> None:
        self._set_rotation_leaf(None)
        self.rotation_grad_acc = None
        self.rotation_cache_active = False
        self._rotation_cache_state_version = -1

    def _fp32_rotation(self, weight: torch.Tensor) -> torch.Tensor:
        """R(Q) in float32 outside autocast; differentiable when ``weight`` requires grad."""
        scaling_factor = 2 * math.sqrt(self.block_size - 1) if self.oft_scaled else 1
        with torch.autocast(device_type=weight.device.type, enabled=False):
            Q = weight.to(torch.float32) / scaling_factor
            return self._cayley_batch(
                Q, self.block_size, self.use_cayley_neumann, self.num_cayley_neumann_terms, self.oft_cans
            )

    @torch.no_grad()
    def refresh_rotation_cache(self) -> None:
        """Recompute the cached rotation from the current Q and clip state.

        Must not run with dL/dR still pending: the accumulator belongs to the
        rotation it was computed against, so materialize it first.
        """
        leaf = self.rotation_leaf
        if leaf is None:
            return
        if self.rotation_grad_acc is not None:
            raise RuntimeError(
                "OFT rotation cache refreshed with un-materialized rotation gradients; "
                "call materialize_rotation_grad() before changing Q or the clip state."
            )
        if leaf.requires_grad != self.weight.requires_grad:
            leaf.requires_grad_(self.weight.requires_grad)
        leaf.copy_(self._fp32_rotation(self.weight.detach()))
        self._rotation_cache_state_version = self._spectral_state_version

    def _accumulate_rotation_grad(self, leaf: torch.Tensor) -> None:
        # Post-accumulate-grad hook: fires once per backward that reaches the leaf.
        grad = leaf.grad
        leaf.grad = None
        if grad is None:
            return
        with torch.no_grad():
            if self.rotation_grad_acc is None:
                self.rotation_grad_acc = torch.zeros(leaf.shape, dtype=leaf.dtype, device=leaf.device)
            if self.rotation_grad_acc.dtype == torch.bfloat16:
                add_stochastic_(
                    self.rotation_grad_acc,
                    grad,
                    generator=rotation_cache_generator(leaf.device),
                    chunk_numel=ROTATION_CACHE_SR_CHUNK_NUMEL,
                )
            else:
                self.rotation_grad_acc.add_(grad)

    @staticmethod
    def rotation_cache_staleness(leaf: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        """max |leaf - R| in units of the leaf dtype's half-ulp at each entry of R.

        A leaf filled from this same R by copy_ differs from it by at most half an
        ulp of its dtype (round to nearest), so a fresh cache scores <= 1 (0 for a
        float32 leaf). Entries where R is exactly 0 get a bound of float32 tiny: they
        round exactly. Any Q or clip change since the refresh shows up as entries
        many half-ulps off. Returns a 0-dim tensor (no host sync).
        """
        rotation = rotation.detach().to(torch.float32)
        bits = _SIGNIFICAND_BITS.get(leaf.dtype, 24)
        magnitude = rotation.abs()
        # half-ulp(x) = 2 ** (floor(log2 |x|) - bits); small slack for exp() and for
        # float32 kernel noise between the refresh and this recomputation.
        bound = magnitude.log2().floor_().sub_(bits).mul_(math.log(2.0)).exp_()
        bound.mul_(1.0 + 1e-3).add_(magnitude, alpha=1e-6).add_(torch.finfo(torch.float32).tiny)
        del magnitude
        # copy=True: a float32 leaf must never be modified in place here.
        diff = leaf.detach().to(dtype=torch.float32, copy=True).sub_(rotation).abs_()
        return diff.div_(bound).amax()

    def materialize_rotation_grad(self) -> torch.Tensor | None:
        """Pull the accumulated dL/dR back into weight.grad through one fp32 R(Q).

        Releases the accumulator. Returns the stale-cache score of this module
        (rotation_cache_staleness; a 0-dim tensor, no sync), or None when there was
        nothing to materialize. A clip-state change since the last refresh scores inf.
        """
        acc = self.rotation_grad_acc
        self.rotation_grad_acc = None
        if acc is None or self.rotation_leaf is None or not self.weight.requires_grad:
            return None
        with torch.enable_grad():
            rotation = self._fp32_rotation(self.weight)
            if self._rotation_cache_state_version != self._spectral_state_version:
                staleness = torch.full((), float("inf"), device=rotation.device)
            else:
                staleness = self.rotation_cache_staleness(self.rotation_leaf, rotation)
            upstream = acc.to(torch.float32)
            del acc
            rotation.backward(upstream)
        return staleness

    def forward(self, x):
        required_dtype = x.dtype
        orig_shape = x.shape

        if self.rotation_cache_active:
            # Refreshed once per optimizer update; dL/dR reaches the accumulator
            # through the leaf's post-accumulate-grad hook.
            orth_rotate = self.rotation_leaf
        else:
            scaling_factor = 2 * math.sqrt(self.block_size - 1) if self.oft_scaled else 1
            effective_weight = self.weight / scaling_factor

            # The rotation runs under the caller's autocast (bf16 in mixed-precision
            # training). A float32 Cayley was tried and reverted: the applied rotation
            # is cast back to the activation dtype anyway, so it bought no output
            # precision while giving up the bf16 matmul speed.
            orth_rotate = self._cayley_batch(
                effective_weight,
                self.block_size,
                self.use_cayley_neumann,
                self.num_cayley_neumann_terms,
                self.oft_cans,
            )
        orth_rotate = self.dropout(orth_rotate)
        # Cast the rotation once instead of upcasting the activation: under autocast
        # the einsum runs in the autocast dtype either way, so this is numerically
        # identical and skips an activation-sized copy.
        orth_rotate = orth_rotate.to(required_dtype)

        rank = self.in_features // self.block_size if self.block_share else self.r
        batch_dims = x.shape[:-1]
        x_reshaped = x.reshape(*batch_dims, rank, self.block_size)

        if self.block_share:
            orth_rotate = orth_rotate.repeat(rank, 1, 1)
            x_rotated_reshaped = torch.einsum("...rk,rkc->...rc", x_reshaped, orth_rotate)
        else:
            x_rotated_reshaped = torch.einsum("...rk,rkc->...rc", x_reshaped, orth_rotate)

        x_rotated = x_rotated_reshaped.reshape(*orig_shape)

        return x_rotated.to(required_dtype)
