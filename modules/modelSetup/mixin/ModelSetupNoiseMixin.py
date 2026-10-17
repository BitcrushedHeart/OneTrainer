import math
from abc import ABCMeta
from random import Random

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.noising_override_util import NoisingParams, resolve_noising_overrides

import torch
import torch.distributions
from torch import Generator, Tensor

CONTINUOUS_TIMESTEP_DISTRIBUTIONS = (
    TimestepDistribution.UNIFORM,
    TimestepDistribution.LOGIT_NORMAL,
    TimestepDistribution.HEAVY_TAIL,
    TimestepDistribution.BETA,
)


class ModelSetupNoiseMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        self.__weights = None
        # Discrete-distribution CDFs of the per-concept override path, keyed by
        # their parameters (one per distinct concept setting and resolution).
        self.__override_cdfs: dict[tuple, Tensor] = {}
        self._offset_noise_psi_schedule: Tensor | None = None

    def _compute_and_cache_offset_noise_psi_schedule(self, betas: Tensor) -> Tensor:
        """
        Computes the time-dependent psi_t coefficients for generalized offset noise.
        This implementation follows the paper "Generalized Diffusion Model with Adjusted Offset Noise",
        specifically Equation (34) and the logic of Algorithm 1 for the "balanced-phi_t, psi_t strategy".
        """
        if self._offset_noise_psi_schedule is not None and self._offset_noise_psi_schedule.shape[0] == betas.shape[0]:
            return self._offset_noise_psi_schedule.to(betas.device).to(torch.float64)

        betas = betas.to(torch.float64)
        T = betas.shape[0]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # From paper footnote 4: "we introduce α_0 = 1 for convenience".
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=betas.device, dtype=betas.dtype), alphas_cumprod[:-1]]
        )

        # --- Start of Algorithm 1 ---
        gammas = torch.zeros(T, device=betas.device, dtype=betas.dtype)

        # Step 1: Set gamma_1 = 1
        gammas[0] = 1.0

        # This sum is `Σ_{i=1 to t-1} γ_i/√¯αᵢ₋₁` which we build iteratively.
        cumulative_sum_term = gammas[0] / torch.sqrt(alphas_cumprod_prev[0])

        # Step 2-4: Loop for t = 2 to T (in code: t = 1 to T-1)
        for t in range(1, T):
            alpha_t = alphas[t]
            alpha_cumprod_tm1 = alphas_cumprod_prev[t]

            # Denominator from the paper's formula for C_t.
            c_t_denominator = alpha_t * (1 - alpha_cumprod_tm1)
            c_t = (1 - alpha_t) * torch.sqrt(alpha_cumprod_tm1) / c_t_denominator

            # Paper's recursive formula uses the full cumulative sum.
            gammas[t] = c_t * cumulative_sum_term

            # Update the sum for the next iteration.
            cumulative_sum_term += gammas[t] / torch.sqrt(alphas_cumprod_prev[t])

        # Step 5: Calculate normalization factor psi_T
        psi_T_denominator = torch.sqrt(1 - alphas_cumprod[-1])
        psi_T = cumulative_sum_term / psi_T_denominator

        # Step 6-8: Normalize gammas
        gammas_normalized = gammas / psi_T
        # --- End of Algorithm 1 ---

        # Finally, calculate the psi schedule for all timesteps t using Equation (22)
        terms = gammas_normalized / torch.sqrt(alphas_cumprod_prev)
        s_cumulative = torch.cumsum(terms, dim=0)
        psi_schedule = s_cumulative / torch.sqrt(1 - alphas_cumprod)

        self._offset_noise_psi_schedule = psi_schedule.to(betas.device)
        return self._offset_noise_psi_schedule

    def _apply_dpo_paired_rng(self, tensor: Tensor) -> Tensor:
        # During a batched DPO forward the batch is [chosen; rejected] along
        # dim 0, and each pair must be compared at the same timestep with the
        # same noise. Sequential RNG draws would give rejected[i] draw B+i
        # instead of draw i, so the second half is overwritten with the first.
        # calculate_dpo_loss sets _dpo_paired_half to B around its batched
        # predict() calls; the batch dict itself is not visible here.
        # Applied at every per-sample RNG draw site: _create_noise,
        # _get_timestep_discrete, and _apply_ciop (eps_in/eps_out). Any new
        # draw site must apply it too, or DPO pairs silently desynchronize.
        half = getattr(self, "_dpo_paired_half", None)
        if half is not None and tensor.shape[0] == 2 * half:
            tensor[half:] = tensor[:half]
        return tensor

    def _create_noise(
        self,
        source_tensor: Tensor,
        config: TrainConfig,
        generator: Generator,
        timestep: Tensor | None = None,
        betas: Tensor | None = None,
    ) -> Tensor:
        noise = torch.randn(
            source_tensor.shape, generator=generator, device=config.train_device, dtype=source_tensor.dtype
        )

        if config.offset_noise_weight > 0:
            offset_noise = torch.randn(
                (source_tensor.shape[0], source_tensor.shape[1], *[1 for _ in range(source_tensor.ndim - 2)]),
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype,
            )
            # Use the time-dependent generalized method if enabled.
            # This will only be true for Diffusion models (which uses betas)
            if config.generalized_offset_noise and timestep is not None and betas is not None:
                psi_schedule = self._compute_and_cache_offset_noise_psi_schedule(betas).to(timestep.device)
                psi_t = psi_schedule[timestep]
                psi_t = psi_t.view(psi_t.shape[0], *[1 for _ in range(source_tensor.ndim - 1)])
                # Scale by the time-dependent psi_t factor
                noise = noise + (psi_t * config.offset_noise_weight * offset_noise)
            else:  # Otherwise, use the normal offset noise.
                noise = noise + (config.offset_noise_weight * offset_noise)

        if config.perturbation_noise_weight > 0:
            perturbation_noise = torch.randn(
                source_tensor.shape, generator=generator, device=config.train_device, dtype=source_tensor.dtype
            )
            noise = noise + (config.perturbation_noise_weight * perturbation_noise)

        return self._apply_dpo_paired_rng(noise)

    def _apply_ciop(
        self,
        noisy_latent: Tensor,
        target_noise: Tensor,
        config: TrainConfig,
        generator: Generator,
        rand: Random,
    ) -> tuple[Tensor, Tensor]:
        """
        Applies Coordinated Input-Output Perturbation (CIOP) as per Equation (6) & (7).
        Paper: "Bidirectional Noise Injection: Enhancing Diffusion Models via Coordinated Input-Output Perturbation"

        With probability p, injects independent Gaussian noise to both the noisy latent
        and the target noise.
        """
        ciop_noise_weight = config.ciop_noise_weight
        if ciop_noise_weight != 0:
            prob_threshold = torch.rand(1, generator=generator, device=noisy_latent.device)

            apply_mask = (prob_threshold < config.ciop_p).to(noisy_latent.dtype)

            eps_in = torch.randn(
                noisy_latent.shape,
                generator=generator,
                device=noisy_latent.device,
                dtype=noisy_latent.dtype,
            ) * (ciop_noise_weight * apply_mask)
            # DPO pairing: see _apply_dpo_paired_rng (prob_threshold is batch-wide, no pairing needed)
            eps_in = self._apply_dpo_paired_rng(eps_in)

            eps_out = torch.randn(
                target_noise.shape,
                generator=generator,
                device=target_noise.device,
                dtype=target_noise.dtype,
            ) * (ciop_noise_weight * apply_mask)
            eps_out = self._apply_dpo_paired_rng(eps_out)

            noisy_latent = noisy_latent + eps_in
            target_noise = target_noise + eps_out

        return noisy_latent, target_noise

    def _get_timestep_discrete(
        self,
        num_train_timesteps: int,
        deterministic: bool,
        generator: Generator,
        batch_size: int,
        config: TrainConfig,
        shift: float = None,
        batch: dict | None = None,
    ) -> Tensor:
        """Per-sample training timesteps.

        ``batch`` carries the dataloader's per-concept override columns (see
        noising_override_util); when no sample in it overrides anything the
        draw is the run-wide one below, bit for bit, so resumes and runs
        without overrides are unaffected.
        """
        if shift is None:
            shift = config.timestep_shift

        if deterministic:
            # -1 is for zero-based indexing. The fraction is 0.5 (validation)
            # unless a diagnostics probe pinned another one (_probe_timestep_fraction).
            fraction = getattr(self, "_probe_timestep_fraction", None)
            fraction = 0.5 if fraction is None else float(fraction)
            return torch.tensor(
                max(0, int(num_train_timesteps * fraction) - 1),
                dtype=torch.long,
                device=generator.device,
            ).unsqueeze(0)

        overrides = resolve_noising_overrides(config, batch, batch_size, shift)
        if overrides is None:
            timestep = self.__sample_run_timesteps(num_train_timesteps, generator, batch_size, config, shift)
        else:
            timestep = self.__sample_concept_timesteps(num_train_timesteps, generator, batch_size, config, overrides)
        return self._apply_dpo_paired_rng(timestep.int())

    def __sample_run_timesteps(
        self,
        num_train_timesteps: int,
        generator: Generator,
        batch_size: int,
        config: TrainConfig,
        shift: float,
    ) -> Tensor:
        """The run-wide draw: every sample uses the TrainConfig settings."""
        min_timestep = int(num_train_timesteps * config.min_noising_strength)
        max_timestep = int(num_train_timesteps * config.max_noising_strength)
        num_timestep = max_timestep - min_timestep

        if config.timestep_distribution in CONTINUOUS_TIMESTEP_DISTRIBUTIONS:
            # continuous implementations
            if config.timestep_distribution == TimestepDistribution.UNIFORM:
                timestep = min_timestep + (max_timestep - min_timestep) * torch.rand(
                    batch_size, generator=generator, device=generator.device
                )
            elif config.timestep_distribution == TimestepDistribution.LOGIT_NORMAL:
                bias = config.noising_bias
                scale = config.noising_weight + 1.0

                normal = torch.normal(bias, scale, size=(batch_size,), generator=generator, device=generator.device)
                logit_normal = normal.sigmoid()
                timestep = logit_normal * num_timestep + min_timestep
            elif config.timestep_distribution == TimestepDistribution.HEAVY_TAIL:
                scale = config.noising_weight

                u = torch.rand(
                    size=(batch_size,),
                    generator=generator,
                    device=generator.device,
                )
                u = 1.0 - u - scale * (torch.cos(math.pi / 2.0 * u) ** 2.0 - 1.0 + u)
                timestep = u * num_timestep + min_timestep
            elif config.timestep_distribution == TimestepDistribution.BETA:
                # B-TTDM Configuration
                # Noising Weight -> Alpha
                # Noising Bias   -> Beta
                alpha = max(1e-4, config.noising_weight)
                beta = max(1e-4, config.noising_bias)

                # B-TTDM Paper optimization (Section 3.3):
                # They strictly recommend Beta=1 and Alpha < 1.
                # When Beta=1, we can use Inverse Transform Sampling (CDF inversion)
                # which allows us to use torch.rand with the generator
                # CDF^(-1)(u) = u^(1/alpha)
                if abs(beta - 1.0) < 1e-5:
                    u = torch.rand(batch_size, generator=generator, device=generator.device)
                    u = u.pow(1.0 / alpha)
                    timestep = u * num_timestep + min_timestep

                # Inverse case: Alpha=1, Beta != 1
                # x = 1 - u^(1/beta)
                elif abs(alpha - 1.0) < 1e-5:
                    u = torch.rand(batch_size, generator=generator, device=generator.device)
                    u = 1.0 - u.pow(1.0 / beta)
                    timestep = u * num_timestep + min_timestep

                else:
                    # Fallback for arbitrary Beta values (Beta != 1 and Alpha != 1).
                    # PyTorch's Beta distribution does not accept a generator directly.
                    # This path is mathematically correct for distribution shape, but
                    # technically bypasses the generator seed (uses global device seed).
                    # Since B-TTDM requires Beta=1, this path is rarely taken for this specific paper.
                    m = torch.distributions.Beta(
                        torch.tensor(alpha, device=generator.device), torch.tensor(beta, device=generator.device)
                    )
                    u = m.sample((batch_size,))
                    timestep = u * num_timestep + min_timestep

            timestep = num_train_timesteps * shift * timestep / ((shift - 1) * timestep + num_train_timesteps)
        else:
            if self.__weights is None:
                self.__weights = self.__discrete_weights(
                    config.timestep_distribution, config.noising_bias, config.noising_weight, shift, num_timestep
                ).to(device=generator.device)
            samples = (
                torch.multinomial(self.__weights, num_samples=batch_size, replacement=True, generator=generator)
                + min_timestep
            )
            timestep = samples.to(dtype=torch.long, device=generator.device)

        return timestep

    @staticmethod
    def __discrete_weights(
        distribution: TimestepDistribution, noising_bias: float, noising_weight: float, shift: float, num_timestep: int
    ) -> Tensor:
        # Shifting a discrete distribution is done in two steps:
        # 1. Apply the inverse shift to the linspace.
        #    This moves the sample points of the function to their shifted place.
        # 2. Multiply the result with the derivative of the inverse shift function.
        #    The derivative is an approximation of the distance between sample points.
        #    Or in other words, the size of a shifted bucket in the original function.
        linspace = torch.linspace(0, 1, num_timestep)
        linspace = linspace / (shift - shift * linspace + linspace)

        linspace_derivative = torch.linspace(0, 1, num_timestep)
        linspace_derivative = shift / (shift + linspace_derivative - (linspace_derivative * shift)).pow(2)

        # continuous implementations
        if distribution == TimestepDistribution.COS_MAP:
            weights = 2.0 / (math.pi - 2.0 * math.pi * linspace + 2.0 * math.pi * linspace**2.0)
        elif distribution == TimestepDistribution.SIGMOID:
            bias = noising_bias + 0.5
            weight = noising_weight

            weights = linspace / (shift - shift * linspace + linspace)
            weights = 1 / (1 + torch.exp(-weight * (weights - bias)))  # Sigmoid
        elif distribution == TimestepDistribution.INVERTED_PARABOLA:
            bias = noising_bias + 0.5
            weight = noising_weight

            weights = torch.clamp(-weight * ((linspace - bias) ** 2) + 2, min=0.0)
        else:
            raise ValueError(f"{distribution} is not a discrete timestep distribution")
        weights *= linspace_derivative
        return weights

    def __sample_concept_timesteps(
        self,
        num_train_timesteps: int,
        generator: Generator,
        batch_size: int,
        config: TrainConfig,
        params: NoisingParams,
    ) -> Tensor:
        """The per-concept draw: one base random number per sample, drawn
        exactly as the run's distribution draws it (same generator
        consumption), then shaped by each sample's own settings.

        Samples that inherit everything get the run's numbers back: the base
        draw is the run's and the transforms repeat its arithmetic. A sample
        overriding its distribution converts the base number (uniform <->
        normal through the normal CDF, uniform -> discrete bucket through the
        inverse CDF), so it follows its own distribution from the same draw.
        Runs on a discrete distribution (COS_MAP, SIGMOID, INVERTED_PARABOLA)
        draw a uniform here instead of the multinomial, so their inheriting
        samples keep the distribution but not the exact values of the run-wide
        path; the no-override case never reaches this method.

        Range vs shift: with range_after_shift off a sample's [min, max] is
        applied before its shift, like the run's values (the continuous
        distributions shift the ranged draw, the discrete ones fold the shift
        into their bucket weights). On, the draw covers the full schedule, is
        shifted, and the shifted value is mapped linearly into [min, max], so
        the bounds mean the same noise level whatever the shift.
        """
        device = generator.device
        T = num_train_timesteps
        run_distribution = config.timestep_distribution

        # Base draw: the run's own, so inheriting samples see the same numbers.
        normal = None
        if run_distribution == TimestepDistribution.LOGIT_NORMAL:
            run_bias = config.noising_bias
            run_scale = config.noising_weight + 1.0
            normal = torch.normal(run_bias, run_scale, size=(batch_size,), generator=generator, device=device)
            z = (normal - run_bias) / run_scale
            u = torch.special.ndtr(z)
        else:
            u = torch.rand(batch_size, generator=generator, device=device)
            z = torch.special.ndtri(u)

        min_t = torch.tensor([int(T * value) for value in params.min_strength], dtype=torch.float32, device=device)
        max_t = torch.tensor([int(T * value) for value in params.max_strength], dtype=torch.float32, device=device)
        after = torch.tensor(params.range_after_shift, dtype=torch.bool, device=device)
        # Before the shift the draw spans [min_t, max_t); after it, the full
        # schedule [0, T) and the range is applied to the shifted value.
        low = torch.where(after, torch.zeros_like(min_t), min_t)
        span = torch.where(after, torch.full_like(max_t, T), max_t - min_t)
        weight = torch.tensor(params.weight, dtype=torch.float32, device=device)
        bias = torch.tensor(params.bias, dtype=torch.float32, device=device)

        timestep = torch.zeros(batch_size, dtype=torch.float32, device=device)
        for distribution in dict.fromkeys(params.distribution):
            index = torch.tensor(
                [i for i, d in enumerate(params.distribution) if d == distribution], dtype=torch.long, device=device
            )
            if distribution in CONTINUOUS_TIMESTEP_DISTRIBUTIONS:
                unit = self.__continuous_unit_sample(
                    distribution,
                    u[index],
                    z[index],
                    normal[index] if normal is not None else None,
                    weight[index],
                    bias[index],
                    run_bias if normal is not None else None,
                    run_scale if normal is not None else None,
                )
                timestep[index] = unit * span[index] + low[index]
            else:
                # Inverse-CDF draw from the discrete weights, one CDF per
                # distinct (settings, range) group.
                for i in index.tolist():
                    key = (distribution, params.bias[i], params.weight[i], params.shift[i], int(span[i].item()))
                    cdf = self.__override_cdfs.get(key)
                    if cdf is None:
                        weights = self.__discrete_weights(
                            distribution, params.bias[i], params.weight[i], params.shift[i], key[-1]
                        )
                        cdf = (weights.cumsum(0) / weights.sum()).to(device=device)
                        self.__override_cdfs[key] = cdf
                    bucket = torch.searchsorted(cdf, u[i : i + 1], right=True).clamp(max=key[-1] - 1)
                    timestep[i] = bucket[0].to(torch.float32) + low[i]

        # The continuous distributions shift the draw; the discrete ones already
        # folded the shift into their weights.
        continuous = torch.tensor(
            [d in CONTINUOUS_TIMESTEP_DISTRIBUTIONS for d in params.distribution], dtype=torch.bool, device=device
        )
        if all(value == params.shift[0] for value in params.shift):
            shift = params.shift[0]  # python scalar: the run-wide arithmetic, bit for bit
        else:
            shift = torch.tensor(params.shift, dtype=torch.float32, device=device)
        shifted = T * shift * timestep / ((shift - 1) * timestep + T)
        timestep = torch.where(continuous, shifted, timestep)

        # Range after the shift: the shifted full-schedule value, mapped
        # linearly into this sample's [min_t, max_t).
        ranged = timestep / T * (max_t - min_t) + min_t
        return torch.where(after, ranged, timestep)

    @staticmethod
    def __continuous_unit_sample(
        distribution: TimestepDistribution,
        u: Tensor,
        z: Tensor,
        normal: Tensor | None,
        weight: Tensor,
        bias: Tensor,
        run_bias: float | None,
        run_scale: float | None,
    ) -> Tensor:
        """The run-wide formulas of __sample_run_timesteps with per-sample
        parameters, returning the draw in [0, 1] (before range and shift)."""
        if distribution == TimestepDistribution.UNIFORM:
            return u
        if distribution == TimestepDistribution.LOGIT_NORMAL:
            scale = weight + 1.0
            shaped = bias + scale * z
            if normal is not None:
                # Inheriting samples take the run's normal draw itself rather
                # than the (rounded) round trip through z.
                shaped = torch.where((bias == run_bias) & (scale == run_scale), normal, shaped)
            return shaped.sigmoid()
        if distribution == TimestepDistribution.HEAVY_TAIL:
            return 1.0 - u - weight * (torch.cos(math.pi / 2.0 * u) ** 2.0 - 1.0 + u)
        if distribution == TimestepDistribution.BETA:
            alpha = weight.clamp(min=1e-4)
            beta = bias.clamp(min=1e-4)
            beta_is_one = (beta - 1.0).abs() < 1e-5
            alpha_is_one = (alpha - 1.0).abs() < 1e-5
            unit = torch.where(beta_is_one, u.pow(1.0 / alpha), 1.0 - u.pow(1.0 / beta))
            fallback = ~beta_is_one & ~alpha_is_one
            if bool(fallback.any()):
                # Arbitrary alpha/beta: no closed-form inverse CDF, sampled
                # without the generator like the run-wide path.
                unit = unit.clone()
                unit[fallback] = torch.distributions.Beta(alpha[fallback], beta[fallback]).sample()
            return unit
        raise ValueError(f"{distribution} is not a continuous timestep distribution")

    def _get_timestep_continuous(
        self,
        deterministic: bool,
        generator: Generator,
        batch_size: int,
        config: TrainConfig,
        batch: dict | None = None,
    ) -> Tensor:
        if deterministic:
            fraction = getattr(self, "_probe_timestep_fraction", None)
            return torch.full(
                size=(batch_size,),
                fill_value=0.5 if fraction is None else float(fraction),
                device=generator.device,
            )
        else:
            discrete_timesteps = 10000  # Discretize to 10000 timesteps
            discrete = (
                self._get_timestep_discrete(
                    num_train_timesteps=discrete_timesteps,
                    deterministic=False,
                    generator=generator,
                    batch_size=batch_size,
                    config=config,
                    batch=batch,
                )
                + 1
            )

            continuous = discrete.float() / discrete_timesteps
            return continuous
