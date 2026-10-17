"""Per-concept timestep sampling overrides in ModelSetupNoiseMixin.

``_legacy_timesteps`` is the run-wide draw as it was before the overrides
existed; a batch in which no sample overrides anything must reproduce it bit
for bit (same generator consumption), and in a mixed batch the inheriting
samples keep following it while the overriding ones use their own settings.
"""

import math

from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_light_util import split_dpo_pair_batch
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.noising_override_util import (
    INHERIT_DISTRIBUTION,
    INHERIT_FLOAT,
    NOISING_OVERRIDE_COLUMNS,
    resolve_noising_overrides,
)

import torch

import pytest

T = 1000
NAN = INHERIT_FLOAT


class _Noise(ModelSetupNoiseMixin):
    pass


def _legacy_timesteps(weights_cache: dict, num_train_timesteps, generator, batch_size, config, shift) -> torch.Tensor:
    """ModelSetupNoiseMixin._get_timestep_discrete (non-deterministic branch)
    before per-concept overrides, verbatim apart from the weights cache."""
    min_timestep = int(num_train_timesteps * config.min_noising_strength)
    max_timestep = int(num_train_timesteps * config.max_noising_strength)
    num_timestep = max_timestep - min_timestep

    if config.timestep_distribution in [
        TimestepDistribution.UNIFORM,
        TimestepDistribution.LOGIT_NORMAL,
        TimestepDistribution.HEAVY_TAIL,
        TimestepDistribution.BETA,
    ]:
        if config.timestep_distribution == TimestepDistribution.UNIFORM:
            timestep = min_timestep + (max_timestep - min_timestep) * torch.rand(
                batch_size, generator=generator, device=generator.device
            )
        elif config.timestep_distribution == TimestepDistribution.LOGIT_NORMAL:
            bias = config.noising_bias
            scale = config.noising_weight + 1.0
            normal = torch.normal(bias, scale, size=(batch_size,), generator=generator, device=generator.device)
            timestep = normal.sigmoid() * num_timestep + min_timestep
        elif config.timestep_distribution == TimestepDistribution.HEAVY_TAIL:
            scale = config.noising_weight
            u = torch.rand(size=(batch_size,), generator=generator, device=generator.device)
            u = 1.0 - u - scale * (torch.cos(math.pi / 2.0 * u) ** 2.0 - 1.0 + u)
            timestep = u * num_timestep + min_timestep
        elif config.timestep_distribution == TimestepDistribution.BETA:
            alpha = max(1e-4, config.noising_weight)
            beta = max(1e-4, config.noising_bias)
            if abs(beta - 1.0) < 1e-5:
                u = torch.rand(batch_size, generator=generator, device=generator.device)
                u = u.pow(1.0 / alpha)
                timestep = u * num_timestep + min_timestep
            elif abs(alpha - 1.0) < 1e-5:
                u = torch.rand(batch_size, generator=generator, device=generator.device)
                u = 1.0 - u.pow(1.0 / beta)
                timestep = u * num_timestep + min_timestep
            else:
                m = torch.distributions.Beta(
                    torch.tensor(alpha, device=generator.device), torch.tensor(beta, device=generator.device)
                )
                u = m.sample((batch_size,))
                timestep = u * num_timestep + min_timestep

        timestep = num_train_timesteps * shift * timestep / ((shift - 1) * timestep + num_train_timesteps)
    else:
        linspace = torch.linspace(0, 1, num_timestep)
        linspace = linspace / (shift - shift * linspace + linspace)

        linspace_derivative = torch.linspace(0, 1, num_timestep)
        linspace_derivative = shift / (shift + linspace_derivative - (linspace_derivative * shift)).pow(2)

        if config.timestep_distribution == TimestepDistribution.COS_MAP:
            if "weights" not in weights_cache:
                weights = 2.0 / (math.pi - 2.0 * math.pi * linspace + 2.0 * math.pi * linspace**2.0)
                weights *= linspace_derivative
                weights_cache["weights"] = weights.to(device=generator.device)
        elif config.timestep_distribution == TimestepDistribution.SIGMOID:
            if "weights" not in weights_cache:
                bias = config.noising_bias + 0.5
                weight = config.noising_weight
                weights = linspace / (shift - shift * linspace + linspace)
                weights = 1 / (1 + torch.exp(-weight * (weights - bias)))
                weights *= linspace_derivative
                weights_cache["weights"] = weights.to(device=generator.device)
        elif config.timestep_distribution == TimestepDistribution.INVERTED_PARABOLA:
            if "weights" not in weights_cache:
                bias = config.noising_bias + 0.5
                weight = config.noising_weight
                weights = torch.clamp(-weight * ((linspace - bias) ** 2) + 2, min=0.0)
                weights *= linspace_derivative
                weights_cache["weights"] = weights.to(device=generator.device)
        samples = (
            torch.multinomial(weights_cache["weights"], num_samples=batch_size, replacement=True, generator=generator)
            + min_timestep
        )
        timestep = samples.to(dtype=torch.long, device=generator.device)

    return timestep.int()


def _config(
    distribution=TimestepDistribution.UNIFORM,
    *,
    min_strength=0.0,
    max_strength=1.0,
    weight=0.0,
    bias=0.0,
    shift=1.0,
) -> TrainConfig:
    config = TrainConfig.default_values()
    config.timestep_distribution = distribution
    config.min_noising_strength = min_strength
    config.max_noising_strength = max_strength
    config.noising_weight = weight
    config.noising_bias = bias
    config.timestep_shift = shift
    return config


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _draw(config, batch, batch_size=6, seed=0, shift=None, mixin=None):
    mixin = mixin or _Noise()
    generator = _generator(seed)
    timestep = mixin._get_timestep_discrete(T, False, generator, batch_size, config, shift=shift, batch=batch)
    return timestep, generator.get_state()


def _legacy(config, batch_size=6, seed=0, shift=None, cache=None):
    generator = _generator(seed)
    timestep = _legacy_timesteps(
        cache if cache is not None else {},
        T,
        generator,
        batch_size,
        config,
        config.timestep_shift if shift is None else shift,
    )
    return timestep, generator.get_state()


def _inherit_columns(batch_size: int) -> dict:
    """Every column present, every sample inheriting (what the dataloader
    sends for concepts that leave the settings alone)."""
    columns = {name: torch.full((batch_size,), NAN) for name in NOISING_OVERRIDE_COLUMNS}
    columns["timestep_distribution_override"] = [INHERIT_DISTRIBUTION] * batch_size
    columns["noising_range_after_shift"] = torch.zeros(batch_size, dtype=torch.bool)
    return columns


def _column(values: list) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


RUN_SETTINGS = {
    "uniform": _config(TimestepDistribution.UNIFORM, min_strength=0.05, max_strength=0.95),
    "uniform_shift": _config(TimestepDistribution.UNIFORM, shift=3.0),
    "logit_normal": _config(TimestepDistribution.LOGIT_NORMAL, weight=0.3, bias=-0.2, shift=2.5),
    "heavy_tail": _config(TimestepDistribution.HEAVY_TAIL, weight=0.4, min_strength=0.1),
    "beta_cdf": _config(TimestepDistribution.BETA, weight=0.5, bias=1.0),
    "beta_inverse": _config(TimestepDistribution.BETA, weight=1.0, bias=2.0),
    "cos_map": _config(TimestepDistribution.COS_MAP, shift=1.5),
    "sigmoid": _config(TimestepDistribution.SIGMOID, weight=2.0, bias=0.1),
    "inverted_parabola": _config(TimestepDistribution.INVERTED_PARABOLA, weight=4.0, bias=-0.1, max_strength=0.8),
}


@pytest.mark.parametrize("name", list(RUN_SETTINGS))
@pytest.mark.parametrize("seed", [0, 1, 12345])
@pytest.mark.parametrize("columns", ["absent", "inherit"], ids=["no_columns", "inherit_columns"])
def test_no_override_draw_is_bit_identical_to_the_legacy_draw(name, seed, columns):
    config = RUN_SETTINGS[name]
    batch = None if columns == "absent" else {"latent_image": torch.zeros(6), **_inherit_columns(6)}
    mixin = _Noise()
    for step in range(3):  # the discrete weights cache is filled on the first call
        expected, expected_state = _legacy(config, seed=seed + step, cache=mixin.__dict__.setdefault("_cache", {}))
        actual, state = _draw(config, batch, seed=seed + step, mixin=mixin)
        assert torch.equal(actual, expected), name
        assert torch.equal(state, expected_state), "the generator must be consumed exactly as before"


def test_all_inherit_batch_resolves_to_no_overrides():
    config = _config()
    assert resolve_noising_overrides(config, None, 4, 1.0) is None
    assert resolve_noising_overrides(config, {"latent_image": torch.zeros(4)}, 4, 1.0) is None
    assert resolve_noising_overrides(config, _inherit_columns(4), 4, 1.0) is None
    # Rows cached before the columns existed come through the collate filler
    # as the same sentinels; a partial column set inherits the rest.
    assert resolve_noising_overrides(config, {"noising_bias_override": _column([NAN, NAN])}, 2, 1.0) is None
    params = resolve_noising_overrides(config, {"noising_bias_override": _column([NAN, -1.0])}, 2, 1.0)
    assert params.bias == [0.0, -1.0]
    assert params.min_strength == [0.0, 0.0] and params.max_strength == [1.0, 1.0]
    assert params.distribution == [TimestepDistribution.UNIFORM] * 2
    assert params.shift == [1.0, 1.0] and params.range_after_shift == [False, False]


def test_mixed_batch_keeps_the_run_draw_for_inheriting_samples_and_ranges_the_rest():
    config = RUN_SETTINGS["uniform"]
    batch = {
        "min_noising_strength_override": _column([0.1, NAN, NAN, 0.3, NAN, NAN]),
        "max_noising_strength_override": _column([0.6, NAN, NAN, 0.9, NAN, NAN]),
    }
    for seed in range(20):
        expected, _ = _legacy(config, seed=seed)
        actual, _ = _draw(config, batch, seed=seed)
        inheriting = [1, 2, 4, 5]
        assert torch.equal(actual[inheriting], expected[inheriting])
        assert 100 <= int(actual[0]) < 600
        assert 300 <= int(actual[3]) < 900
        # The overriding samples are the same uniform draw remapped, not a new one.
        u = (expected[0].float() - 50) / 900  # the run's [0.05, 0.95) mapping, rounded by int()
        assert abs(float(actual[0]) - (100 + 500 * u)) <= 2


def test_bias_override_shifts_only_the_overriding_samples():
    config = RUN_SETTINGS["logit_normal"]
    batch = {"noising_bias_override": _column([-2.0, NAN, NAN, 1.5])}
    for seed in range(20):
        expected, _ = _legacy(config, batch_size=4, seed=seed)
        actual, _ = _draw(config, batch, batch_size=4, seed=seed)
        assert torch.equal(actual[[1, 2]], expected[[1, 2]])
        # Same normal draw, lower / higher bias: lower / higher timestep (up to
        # the int truncation of equal values).
        assert int(actual[0]) <= int(expected[0])
        assert int(actual[3]) >= int(expected[3])
    lower = sum(int(_draw(config, batch, batch_size=4, seed=s)[0][0]) for s in range(50))
    base = sum(int(_legacy(config, batch_size=4, seed=s)[0][0]) for s in range(50))
    assert lower < base


def test_weight_override_inherits_the_run_bias():
    config = RUN_SETTINGS["logit_normal"]
    actual, _ = _draw(config, {"noising_weight_override": _column([2.0, NAN])}, batch_size=2)
    expected, _ = _legacy(config, batch_size=2)
    assert torch.equal(actual[1:], expected[1:])
    assert not torch.equal(actual[:1], expected[:1])


def test_distribution_override_converts_the_run_draw_instead_of_consuming_more():
    config = RUN_SETTINGS["uniform"]
    batch = {"timestep_distribution_override": ["LOGIT_NORMAL", "", "SIGMOID", "", "HEAVY_TAIL", "BETA"]}
    expected, expected_state = _legacy(config)
    actual, state = _draw(config, batch)
    assert torch.equal(state, expected_state)
    assert torch.equal(actual[[1, 3]], expected[[1, 3]])
    u = (expected[0].float() - 50) / 900
    logit_normal = torch.special.ndtri(u).sigmoid() * 900 + 50
    assert abs(float(actual[0]) - float(logit_normal)) <= 2
    assert 50 <= int(actual[2]) < 950 and 50 <= int(actual[4]) < 950 and 50 <= int(actual[5]) < 950
    # Same seed, same batch: deterministic (the inverse-CDF draw included).
    again, _ = _draw(config, batch)
    assert torch.equal(actual, again)


def test_logit_normal_run_hands_other_distributions_a_uniform_through_the_normal_cdf():
    config = RUN_SETTINGS["logit_normal"]
    batch = {"timestep_distribution_override": ["UNIFORM", "", "COS_MAP"]}
    expected, expected_state = _legacy(config, batch_size=3)
    actual, state = _draw(config, batch, batch_size=3)
    assert torch.equal(state, expected_state)
    assert torch.equal(actual[1:2], expected[1:2])
    assert 0 <= int(actual[0]) < T and 0 <= int(actual[2]) < T


@pytest.mark.parametrize("name", ["cos_map", "sigmoid"])
def test_discrete_run_with_an_override_draws_the_same_distribution(name):
    config = RUN_SETTINGS[name]
    batch = {"min_noising_strength_override": _column([0.2] + [NAN] * 511)}
    actual, _ = _draw(config, batch, batch_size=512)
    assert int(actual[0]) >= 200
    low, high = int(T * config.min_noising_strength), int(T * config.max_noising_strength)
    assert low <= int(actual.min()) and int(actual.max()) < high
    # Inheriting samples follow the run's weights (inverse CDF of the same
    # weights): the empirical mean sits where the multinomial's does.
    legacy, _ = _legacy(config, batch_size=512)
    assert abs(float(actual[1:].float().mean()) - float(legacy[1:].float().mean())) < 60


def test_shift_override_applies_per_sample():
    config = _config(TimestepDistribution.UNIFORM, shift=1.0)
    expected, _ = _legacy(config, batch_size=3)
    actual, _ = _draw(config, {"timestep_shift_override": _column([3.0, NAN, 1.0])}, batch_size=3)
    assert torch.equal(actual[1:], expected[1:])
    t = expected[0].float()
    assert abs(float(actual[0]) - float(T * 3.0 * t / (2.0 * t + T))) <= 1
    # A dynamic (per-resolution) shift passed by the setup is what the inheriting samples use.
    shifted, _ = _draw(config, {"timestep_shift_override": _column([NAN, 2.0])}, batch_size=2, shift=4.0)
    reference, _ = _legacy(config, batch_size=2, shift=4.0)
    assert torch.equal(shifted[:1], reference[:1])


def test_range_before_the_shift_lands_at_a_different_sigma_per_shift():
    # Today's semantics (default): [min, max] is applied to the unshifted draw,
    # then shifted, so the floor moves with the (dynamic) shift.
    config = _config(TimestepDistribution.UNIFORM)
    batch = {
        "min_noising_strength_override": _column([0.1] * 256),
        "max_noising_strength_override": _column([0.6] * 256),
    }
    at_19, _ = _draw(config, batch, batch_size=256, shift=1.9)
    at_32, _ = _draw(config, batch, batch_size=256, shift=3.2)
    floor_19 = T * 1.9 * 100 / (0.9 * 100 + T)
    floor_32 = T * 3.2 * 100 / (2.2 * 100 + T)
    assert int(at_19.min()) >= int(floor_19) and int(at_32.min()) >= int(floor_32)
    assert int(at_32.min()) > int(floor_19) + 50  # the 3.2 floor is well above the 1.9 one


def test_range_after_the_shift_means_the_same_sigma_at_every_shift():
    config = _config(TimestepDistribution.UNIFORM)
    batch = {
        "min_noising_strength_override": _column([0.1] * 2048),
        "max_noising_strength_override": _column([0.6] * 2048),
        "noising_range_after_shift": torch.ones(2048, dtype=torch.bool),
    }
    for shift in (1.0, 1.9, 3.2):
        actual, _ = _draw(config, batch, batch_size=2048, shift=shift)
        # The whole [100, 600) range is covered, nothing outside it, no pile-up
        # at the bounds (a linear remap, not a clamp: the top bucket holds at
        # most the shifted density, ~3.2x of 2048/500, never a clamped mass).
        assert 100 <= int(actual.min()) <= 104
        assert 594 <= int(actual.max()) < 600
        assert int((actual == 100).sum()) < 40 and int((actual == 599).sum()) < 40
    # Same draw as the run-wide one (u drawn exactly as today), just remapped.
    plain, _ = _legacy(config, batch_size=2048, shift=3.2)
    after, _ = _draw(config, batch, batch_size=2048, shift=3.2)
    assert torch.allclose(after.float(), plain.float() / T * 500 + 100, atol=1.0)


def test_range_after_the_shift_with_a_discrete_distribution():
    config = _config(TimestepDistribution.SIGMOID, weight=2.0)
    batch = {
        "min_noising_strength_override": _column([0.1] * 512),
        "max_noising_strength_override": _column([0.6] * 512),
        "noising_range_after_shift": torch.ones(512, dtype=torch.bool),
    }
    actual, _ = _draw(config, batch, batch_size=512, shift=3.0)
    assert int(actual.min()) >= 100 and int(actual.max()) < 600


def test_dpo_fused_pairs_share_the_timestep_and_use_their_own_range():
    config = RUN_SETTINGS["uniform"]
    mixin = _Noise()
    mixin._dpo_paired_half = 3
    # _create_dpo_batched_batch tiles tensor columns to 2B and passes string
    # columns through at B; the rejected half rides on the chosen rows' values.
    batch = {
        "min_noising_strength_override": _column([0.1, NAN, 0.7] * 2),
        "max_noising_strength_override": _column([0.3, NAN, 0.8] * 2),
        "timestep_distribution_override": ["", "LOGIT_NORMAL", ""],
    }
    for seed in range(10):
        actual, _ = _draw(config, batch, batch_size=6, seed=seed, mixin=mixin)
        assert torch.equal(actual[:3], actual[3:])
        assert 100 <= int(actual[0]) < 300
        assert 700 <= int(actual[2]) < 800
        assert 50 <= int(actual[1]) < 950


def test_dpo_light_halves_resolve_the_same_settings_from_the_pairs_concept():
    config = RUN_SETTINGS["uniform"]
    paired = {
        "latent_image": torch.zeros(3, 2),
        "latent_image_rejected": torch.ones(3, 2),
        "min_noising_strength_override": _column([0.1, NAN, 0.7]),
        "max_noising_strength_override": _column([0.3, NAN, 0.8]),
    }
    chosen, rejected = split_dpo_pair_batch(paired)
    for seed in range(10):
        # Every predict() of the light path reseeds from global_step.
        chosen_t, _ = _draw(config, chosen, batch_size=3, seed=seed)
        rejected_t, _ = _draw(config, rejected, batch_size=3, seed=seed)
        assert torch.equal(chosen_t, rejected_t)
        assert 100 <= int(chosen_t[0]) < 300 and 700 <= int(chosen_t[2]) < 800


def test_column_length_must_match_the_batch_or_its_chosen_half():
    with pytest.raises(RuntimeError, match="noising_bias_override has 2 entries"):
        _draw(_config(), {"noising_bias_override": _column([1.0, 2.0])}, batch_size=3)


def test_continuous_timesteps_pass_the_batch_through():
    config = _config(TimestepDistribution.UNIFORM)
    mixin = _Noise()
    batch = {"min_noising_strength_override": _column([0.5, NAN]), "max_noising_strength_override": _column([0.6, NAN])}
    continuous = mixin._get_timestep_continuous(False, _generator(3), 2, config, batch=batch)
    assert 0.5 <= float(continuous[0]) <= 0.6001
    plain = _Noise()._get_timestep_continuous(False, _generator(3), 2, config)
    assert float(continuous[1]) == float(plain[1])


def test_deterministic_draws_ignore_the_overrides():
    mixin = _Noise()
    batch = {"min_noising_strength_override": _column([0.9, 0.9])}
    assert mixin._get_timestep_discrete(T, True, _generator(0), 2, _config(), batch=batch).tolist() == [499]
