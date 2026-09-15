"""Score, report on, and retire saturated DPO pairs.

Scores every pair of the enabled DPO concepts of a training config against an
*origin* checkpoint (the pre-DPO base of the lineage) and a *current*
checkpoint, using the trainer's own cached data loader and its own
``predict()`` so the per-sample denoising MSEs are the ones the DPO loss sees.
The two checkpoints are never resident together: the origin pass runs first,
its scalars are cached in the concept's table for good, then the current
checkpoint's transformer weights are copied in place and the current pass
runs. A rescore against a newer checkpoint only runs the current pass.

Two ways to name the models:

* ``--current CKPT [--origin CKPT]``  two full checkpoints; origin defaults
  to the config's transformer model. Trains no adapter (a rank-1 LoRA at
  zero is attached only because the RLHF paths require one).
* ``--adapter FILE``  origin = the config's base checkpoint without the
  adapter, current = with it. One model load, toggled per batch.

Draws: ``--draws N`` samples N (noise, timestep) pairs from the trainer's own
timestep distribution (each seed reproduces a training step exactly);
``--timesteps 0.1,0.5,0.9 --seeds K`` forces those noise levels instead, K
noise draws each, for controlled per-bucket coverage.

Nothing here reads dataset images directly: latents and prompt embeddings
come from the training cache (``--cache-dir`` to point at another one, for
example while a run is using the config's cache).

Usage:
    python scripts/util/dpo_saturation_score.py score --config training_presets/X.json \\
        --current path/to/Base_V1.4.safetensors [--origin path/to/Base_V1.3.safetensors] --draws 8
    python scripts/util/dpo_saturation_score.py report --config training_presets/X.json [--beta 750]
    python scripts/util/dpo_saturation_score.py retire --config training_presets/X.json --beta 750 --dead-weight 0.05
    python scripts/util/dpo_saturation_score.py restore --config training_presets/X.json [--keys a,b]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.util import dpo_saturation_util as sat  # noqa: E402
from modules.util.config.ConceptConfig import ConceptConfig  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.dpo_curation_util import dpo_pair_key, walk_skipping_dotted  # noqa: E402
from modules.util.dpo_pattern_util import (  # noqa: E402
    build_rejected_index,
    dpo_concept_pattern_dirs,
    match_chosen,
    resolve_rejected,
)
from modules.util.enum.BalancingStrategy import BalancingStrategy  # noqa: E402
from modules.util.enum.DPORefMode import DPORefMode  # noqa: E402
from modules.util.enum.ModelType import PeftType  # noqa: E402
from modules.util.path_util import is_supported_image_extension  # noqa: E402

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def load_train_config(path: str) -> TrainConfig:
    config = TrainConfig.default_values()
    with open(path, "r", encoding="utf-8") as f:
        config.from_dict(json.load(f))
    return config


def load_concepts(config: TrainConfig, concepts_path: str | None) -> list[ConceptConfig]:
    path = concepts_path or config.concept_file_name
    with open(path, "r", encoding="utf-8") as f:
        return [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]


def select_dpo_concepts(concepts: list[ConceptConfig], only: list[str] | None) -> list[ConceptConfig]:
    """Enabled DPO pattern concepts, optionally filtered by name or path
    substring. Balancing and variations are forced to one so a single epoch
    of the loader is exactly one visit per pair."""
    out = []
    for concept in concepts:
        if not concept.enabled or not concept.is_dpo():
            continue
        if only and not any(
            token.lower() in (concept.name or "").lower() or token.lower() in concept.path.lower() for token in only
        ):
            continue
        c = copy.deepcopy(concept)
        c.balancing = 1.0
        c.balancing_strategy = BalancingStrategy.REPEATS
        c.image_variations = 1
        c.text_variations = 1
        out.append(c)
    return out


def _label(concept: ConceptConfig) -> str:
    """Concept name plus its folder, since several concepts may share a name."""
    base = os.path.basename(os.path.normpath(concept.path))
    return f"{concept.name} ({base})" if concept.name else base


def build_scoring_config(
    config: TrainConfig,
    concepts: list[ConceptConfig],
    *,
    origin: str | None,
    adapter: str | None,
    batch_size: int,
    cache_dir: str | None,
    device: str | None,
) -> TrainConfig:
    """The training config, trimmed to a forwards-only DPO pass."""
    cfg = TrainConfig.default_values().from_dict(config.to_dict())
    cfg.concepts = concepts
    cfg.rlhf_enabled = True
    cfg.batch_size = batch_size
    cfg.gradient_accumulation_steps = 1
    cfg.continue_last_backup = False
    cfg.clear_cache_before_training = False
    cfg.compile = False
    cfg.enable_activation_offloading = False
    cfg.enable_async_offloading = False
    cfg.layer_offload_fraction = 0.0
    cfg.text_reward_enabled = False
    cfg.transfer_enabled = False
    cfg.redistill_enabled = False
    cfg.distillation_teacher_lora_model_name = ""
    cfg.validation = False
    for te in (cfg.text_encoder, cfg.text_encoder_2, cfg.text_encoder_3, cfg.text_encoder_4):
        te.dropout_probability = 0.0
    if cache_dir:
        cfg.cache_dir = cache_dir
    if device:
        cfg.train_device = device
    if adapter:
        cfg.lora_model_name = adapter
        cfg.rlhf_dpo_ref_mode = DPORefMode.NEW_ADAPTER
    else:
        # No adapter to compare: attach the smallest identity adapter the RLHF
        # paths accept. A rank-1 LoRA initialises B = 0, so it is exactly the
        # base model, unlike an identity OFT rotation cast to bf16.
        cfg.lora_model_name = ""
        cfg.peft_type = PeftType.LORA
        cfg.lora_rank = 1
        cfg.lora_alpha = 1.0
        cfg.dora_oft = False
    if origin:
        cfg.transformer.model_name = origin
    return cfg


# --------------------------------------------------------------------------- #
# Model side
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def forced_timestep(model_setup, t_index: int | None):
    """Force the trainer's timestep draw to ``t_index`` for the duration.
    The generator is still consumed exactly as in training (noise first,
    then the timestep draw), so the noise tensor is the one the same seed
    gives on the training path."""
    if t_index is None:
        yield
        return
    original = model_setup._get_timestep_discrete

    def patched(num_train_timesteps, deterministic, generator, batch_size, config, shift=None):
        drawn = original(num_train_timesteps, deterministic, generator, batch_size, config, shift=shift)
        return drawn.new_full(drawn.shape, int(t_index))

    model_setup._get_timestep_discrete = patched
    try:
        yield
    finally:
        del model_setup._get_timestep_discrete


@contextlib.contextmanager
def adapter_disabled(model):
    """Run the bare base model: unhook every adapter, rehook afterwards. The
    trainer's ``reference_model`` cannot be used here - with a loaded adapter
    it snapshots that adapter as the reference, which is the *current* side."""
    adapters = model.adapters()
    if not adapters:
        raise RuntimeError("adapter mode needs a loaded adapter, but the model has none")
    for adapter in adapters:
        adapter.remove_hook_from_module()
    try:
        yield
    finally:
        for adapter in adapters:
            adapter.hook_to_module()


def load_transformer_weights(model, path: str):
    """Copy a checkpoint's transformer tensors over the live transformer in
    place - no second model, no CPU copy of the whole file."""
    from modules.util.torch_util import torch_gc

    import torch

    from safetensors import safe_open
    from tqdm import tqdm

    state = model.transformer.state_dict()
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        missing = sorted(set(state) - keys)
        unexpected = sorted(keys - set(state))
        if missing or unexpected:
            raise RuntimeError(
                f"{path}: transformer key mismatch ({len(missing)} missing, {len(unexpected)} unexpected); "
                f"first missing: {missing[:3]}, first unexpected: {unexpected[:3]}"
            )
        with torch.no_grad():
            for key in tqdm(sorted(keys), desc="loading current weights", leave=False):
                state[key].copy_(f.get_tensor(key))
    torch_gc()


def _per_sample_mse(model_setup, out) -> list[float]:
    return model_setup._dpo_mse_per_sample(out["predicted"], out["target"]).tolist()


def _timesteps(out, batch_size: int) -> list[int]:
    t = out["timestep"]
    values = t.flatten().tolist()
    if len(values) == 1 and batch_size > 1:
        values = values * batch_size
    return [int(v) for v in values]


def score_batch_side(model_setup, model, config, progress, chosen_batch, rejected_batch, draws):
    """One side (origin or current) for every draw of a batch. Returns per
    draw: (t_used per sample, mse_chosen per sample, mse_rejected per sample)."""
    import torch

    results = []
    batch_size = chosen_batch["latent_image"].shape[0]
    with torch.no_grad():
        for draw in draws:
            progress.global_step = int(draw["seed"])
            with forced_timestep(model_setup, draw["t"]):
                out = model_setup.predict(model, chosen_batch, config, progress)
                t_c = _timesteps(out, batch_size)
                mse_c = _per_sample_mse(model_setup, out)
                del out
                out = model_setup.predict(model, rejected_batch, config, progress)
                t_r = _timesteps(out, batch_size)
                mse_r = _per_sample_mse(model_setup, out)
                del out
            if t_c != t_r:
                raise RuntimeError(f"chosen/rejected timesteps desynchronised for seed {draw['seed']}: {t_c} vs {t_r}")
            results.append((t_c, mse_c, mse_r))
    return results


class Scorer:
    def __init__(self, args):
        self.args = args
        self.base_config = load_train_config(args.config)
        concepts = select_dpo_concepts(load_concepts(self.base_config, args.concepts), args.concept)
        if not concepts:
            raise SystemExit("no enabled DPO concepts matched")
        self.concepts = concepts
        self.config = build_scoring_config(
            self.base_config,
            concepts,
            origin=args.origin,
            adapter=args.adapter,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            device=args.device,
        )
        self.concept_dirs = dpo_concept_pattern_dirs(concepts)
        self.adapter_mode = bool(args.adapter)
        if self.adapter_mode:
            self.origin_id = sat.checkpoint_identity(self.config.transformer.model_name)
            self.current_id = sat.checkpoint_identity(args.adapter)
        else:
            self.origin_id = sat.checkpoint_identity(self.config.transformer.model_name)
            self.current_id = sat.checkpoint_identity(args.current)
        self.tables: dict[int, dict] = {}
        self.draws = None
        self._rejected_index: dict[int, dict] = {}
        # The scheduler length is only known once the model is loaded, but the
        # draw list (fixed-timestep mode) and the origin-reuse decision are
        # needed before. Take it from an existing table, else the Z-Image
        # default, and verify against the loaded scheduler.
        self.num_train_timesteps = args.num_train_timesteps or self._table_num_train_timesteps() or 1000
        self._init_tables(self.num_train_timesteps)

    # ----- tables --------------------------------------------------------- #

    def _table_num_train_timesteps(self) -> int | None:
        for concept in self.concepts:
            existing = sat.load_table(concept.path)
            if existing and existing.get("num_train_timesteps"):
                return int(existing["num_train_timesteps"])
        return None

    def _init_tables(self, num_train_timesteps: int):
        if self.args.timesteps:
            timesteps = sat.parse_timestep_fractions(self.args.timesteps, num_train_timesteps)
            self.draws = sat.make_draws(timesteps=timesteps, seeds=self.args.seeds, seed_base=self.args.seed_base)
        else:
            self.draws = sat.make_draws(draws=self.args.draws, seed_base=self.args.seed_base)
        for i, concept in enumerate(self.concepts):
            chosen_dir, rejected_dir = self.concept_dirs[i]
            existing = sat.load_table(concept.path)
            table = sat.new_table(
                concept={
                    "name": concept.name,
                    "path": concept.path,
                    "chosen_dir": chosen_dir,
                    "rejected_dir": rejected_dir,
                    "beta_override": float(concept.dpo_beta_override or 0.0),
                },
                global_beta=float(self.base_config.rlhf_dpo_beta),
                num_train_timesteps=num_train_timesteps,
                origin=self.origin_id,
                draws=self.draws,
            )
            if not self.args.force_origin:
                table["pairs"] = sat.reusable_origin(existing, origin=self.origin_id, draws=self.draws)
            reused = len(table["pairs"])
            print(f"[{_label(concept)}] origin scalars reused for {reused} pairs")
            self.tables[i] = table

    def _save_tables(self, mark_current: bool):
        for i, concept in enumerate(self.concepts):
            table = self.tables[i]
            if mark_current:
                sat.mark_current(table, self.current_id)
            sat.save_table(concept.path, table)

    def _rejected_for(self, idx: int, chosen_path: str) -> str:
        """The rejected file the trainer pairs with this chosen image, via the
        same pattern helpers DeriveDPORejectedPath uses."""
        concept = self.concepts[idx]
        if idx not in self._rejected_index:
            self._rejected_index[idx] = build_rejected_index(concept.path, concept.include_subdirectories)
        stem = match_chosen(concept.dpo_chosen_pattern, concept.path, chosen_path)
        if stem is None:
            raise RuntimeError(f"{chosen_path} does not match the chosen pattern of {_label(concept)}")
        return resolve_rejected(
            concept.dpo_rejected_pattern,
            concept.path,
            stem,
            os.path.splitext(chosen_path)[1],
            self._rejected_index[idx],
        )

    def _locate(self, chosen_path: str) -> tuple[int, str]:
        idx = sat.concept_for_chosen_path(chosen_path, self.concept_dirs)
        if idx is None:
            raise RuntimeError(f"sample outside every DPO concept: {chosen_path}")
        chosen_dir, _ = self.concept_dirs[idx]
        return idx, dpo_pair_key(chosen_path, chosen_dir)

    def _pair_entries(self, batch) -> list[tuple[int, str, dict]]:
        """(concept index, pair key, table entry) per sample of a batch."""
        out = []
        for chosen_path in batch["image_path"]:
            idx, key = self._locate(chosen_path)
            entry = sat.ensure_pair(self.tables[idx], key, chosen_path, self._rejected_for(idx, chosen_path))
            out.append((idx, key, entry))
        return out

    def _chosen_files(self) -> list[tuple[int, str]]:
        """(concept index, chosen path) for every chosen image the loader will
        see: a plain walk of each chosen directory, dot-folders pruned like
        the trainer's path collector."""
        out = []
        for idx, (chosen_dir, _) in enumerate(self.concept_dirs):
            concept = self.concepts[idx]
            for root, files in walk_skipping_dotted(chosen_dir):
                for name in files:
                    if not is_supported_image_extension(os.path.splitext(name)[1]):
                        continue
                    path = os.path.join(root, name)
                    if match_chosen(concept.dpo_chosen_pattern, concept.path, path) is None:
                        continue
                    out.append((idx, path))
                if not concept.include_subdirectories:
                    break
        return out

    def origin_pass_needed(self) -> bool:
        """False when every pair in the training tree already has cached
        origin scalars for every draw."""
        for idx, chosen_path in self._chosen_files():
            key = dpo_pair_key(chosen_path, self.concept_dirs[idx][0])
            entry = self.tables[idx]["pairs"].get(key)
            if entry is None or sat.missing_draws(entry, "origin", self.draws):
                return True
        return False

    # ----- model ---------------------------------------------------------- #

    def run(self):
        from modules.util import create
        from modules.util.config.TrainConfig import QuantizationConfig
        from modules.util.dpo_light_util import split_dpo_pair_batch
        from modules.util.torch_util import torch_gc
        from modules.util.TrainProgress import TrainProgress

        import torch

        from tqdm import tqdm

        config = self.config
        train_device = torch.device(config.train_device)
        temp_device = torch.device(config.temp_device)
        if config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        origin_needed = self.adapter_mode or self.args.force_origin or self.origin_pass_needed()
        if not self.adapter_mode and not origin_needed:
            # Every origin scalar is cached: load the current checkpoint as
            # the model and skip both the origin pass and the weight swap.
            print("origin scalars complete for every pair; loading the current checkpoint directly")
            config.transformer.model_name = self.args.current

        model_loader = create.create_model_loader(config.model_type, config.training_method)
        model_setup = create.create_model_setup(
            config.model_type, train_device, temp_device, config.training_method, config.debug_mode
        )
        print(f"loading model: {config.transformer.model_name}")
        self._progress("loading model", 0, 0)
        model = model_loader.load(
            model_type=config.model_type,
            model_names=config.model_names(),
            weight_dtypes=config.weight_dtypes(),
            quantization=QuantizationConfig.default_values(),
        )
        model.train_config = config
        model_setup.setup_optimizations(model, config)
        model_setup.setup_train_device(model, config)
        model_setup.setup_model(model, config)
        model.to(temp_device)
        model.eval()
        torch_gc()

        progress = TrainProgress()
        model.train_progress = progress
        self._progress("preparing data loader / cache", 0, 0)
        data_loader = create.create_data_loader(
            train_device=train_device,
            temp_device=temp_device,
            model=model,
            model_type=config.model_type,
            model_setup=model_setup,
            training_method=config.training_method,
            config=config,
            train_progress=progress,
            is_validation=False,
        )
        data_set = data_loader.get_data_set()

        scheduler_steps = int(model.noise_scheduler.config["num_train_timesteps"])
        if scheduler_steps != self.num_train_timesteps:
            raise RuntimeError(
                f"scheduler has {scheduler_steps} train timesteps, tables assume {self.num_train_timesteps}; "
                f"pass --num-train-timesteps {scheduler_steps} (and --force-origin if tables exist)"
            )

        model_setup.setup_train_device(model, config)
        model.eval()

        def epoch(desc):
            data_set.start_next_epoch()
            total = data_set.approximate_length()
            self._progress(desc, 0, total)
            for done, batch in enumerate(tqdm(data_loader.get_data_loader(), total=total, desc=desc), start=1):
                yield batch
                self._progress(desc, done, total)

        def score(batch, side, draws):
            chosen_batch, rejected_batch = split_dpo_pair_batch(batch)
            entries = self._pair_entries(batch)
            if side == "origin":
                draws = [d for d in draws if any(sat.missing_draws(e, "origin", [d]) for _, _, e in entries)]
                if not draws:
                    return
            results = score_batch_side(model_setup, model, config, progress, chosen_batch, rejected_batch, draws)
            self._record(entries, side, draws, results)

        started = time.time()
        if self.adapter_mode:
            # Origin = base without the adapter, current = with it: both sides
            # per batch, one epoch.
            for batch in epoch("scoring (adapter off/on)"):
                with adapter_disabled(model):
                    score(batch, "origin", self.draws)
                score(batch, "current", self.draws)
            self._save_tables(mark_current=True)
        else:
            if origin_needed:
                for batch in epoch("origin pass"):
                    score(batch, "origin", self.draws)
                self._save_tables(mark_current=False)
                print(f"swapping in current weights: {self.args.current}")
                self._progress("loading current weights", 0, 0)
                load_transformer_weights(model, self.args.current)
            for batch in epoch("current pass"):
                score(batch, "current", self.draws)
            self._save_tables(mark_current=True)

        elapsed = time.time() - started
        print(f"done in {elapsed / 60:.1f} min")
        for i, concept in enumerate(self.concepts):
            print(f"  {_label(concept)}: {len(self.tables[i]['pairs'])} pairs -> {sat.table_path(concept.path)}")
        self._progress("done", 1, 1)

    def _progress(self, phase: str, done: int, total: int):
        """One JSON line per progress event on stdout when --progress-json is
        set, for a parent process (the web backend) to follow without parsing
        tqdm output."""
        if not getattr(self.args, "progress_json", False):
            return
        print(json.dumps({"progress": {"phase": phase, "done": int(done), "total": int(total)}}), flush=True)

    def _record(self, entries, side, draws, results):
        for draw, (t_used, mse_c, mse_r) in zip(draws, results, strict=True):
            for j, (idx, key, _) in enumerate(entries):
                sat.record_draw(self.tables[idx], key, side, draw, t_used[j], mse_c[j], mse_r[j])


# --------------------------------------------------------------------------- #
# Report / retire / restore
# --------------------------------------------------------------------------- #


def _tables_for(args) -> list[tuple[ConceptConfig, dict]]:
    config = load_train_config(args.config)
    concepts = select_dpo_concepts(load_concepts(config, args.concepts), args.concept)
    out = []
    for concept in concepts:
        table = sat.load_table(concept.path)
        if table is None:
            print(f"[{_label(concept)}] no saturation table; run `score` first")
            continue
        out.append((concept, table))
    return out


def _summary_kwargs(args) -> dict:
    return {
        "beta": args.beta,
        "dead_weight": args.dead_weight,
        "require_all_buckets": args.require_all_buckets,
        "min_draws": args.min_draws,
        "tails": args.tails,
    }


def cmd_report(args):
    tables = _tables_for(args)
    for concept, table in tables:
        label = _label(concept)
        retired = sat.retired_keys(concept.path)
        summary = sat.summarize(table, exclude_keys=retired, **_summary_kwargs(args))
        print(sat.format_report(summary, concept_label=label, show_keys=args.show_keys))
        if retired:
            print(f"retired so far: {len(retired)} pairs, excluded above (restore with `restore`)")
        current = table.get("current") or {}
        print(
            f"origin: {os.path.basename(table['origin']['path'])}  current: {os.path.basename(current.get('path', '-'))}"
        )
        print()
        if args.csv:
            path = args.csv if len(tables) == 1 else _suffixed(args.csv, label)
            sat.rows_to_csv(summary["rows"], path)
            print(f"wrote {path}")
        if args.json:
            path = args.json if len(tables) == 1 else _suffixed(args.json, label)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=1)
            print(f"wrote {path}")


def _suffixed(path: str, label: str) -> str:
    stem, ext = os.path.splitext(path)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    return f"{stem}_{safe}{ext}"


def cmd_retire(args):
    for concept, table in _tables_for(args):
        label = _label(concept)
        if args.keys:
            keys = [k.strip() for k in args.keys.split(",") if k.strip()]
        else:
            summary = sat.summarize(table, exclude_keys=sat.retired_keys(concept.path), **_summary_kwargs(args))
            keys = summary["saturated_keys"]
            print(
                f"[{label}] {len(keys)} saturated pairs at beta {summary['beta']:g}, "
                f"dead weight < {summary['dead_weight']:g}"
                + (" in every bucket" if summary["require_all_buckets"] else "")
            )
        reason = (
            args.reason or f"saturated at beta {sat.resolve_beta(table, args.beta):g}, weight < {args.dead_weight:g}"
        )
        moved = sat.retire_pairs(concept.path, table, keys, reason=reason, dry_run=args.dry_run)
        verb = "would retire" if args.dry_run else "retired"
        print(f"[{label}] {verb} {len(moved)} pairs -> {os.path.join(concept.path, '<side>', sat.RETIRED_DIRNAME)}")
        if args.show_keys:
            for record in moved:
                print("  " + record["key"])


def cmd_restore(args):
    config = load_train_config(args.config)
    concepts = select_dpo_concepts(load_concepts(config, args.concepts), args.concept)
    keys = [k.strip() for k in args.keys.split(",") if k.strip()] if args.keys else None
    for concept in concepts:
        restored = sat.restore_pairs(concept.path, keys)
        print(f"[{_label(concept)}] restored {len(restored)} pairs")


def cmd_score(args):
    if not args.adapter and not args.current:
        raise SystemExit("score needs --current CKPT (with optional --origin) or --adapter FILE")
    if args.adapter and (args.current or args.origin):
        raise SystemExit("--adapter cannot be combined with --current/--origin")
    if args.timesteps and args.draws:
        print("--timesteps given; ignoring --draws")
    if not args.timesteps and args.draws <= 0:
        raise SystemExit("--draws must be > 0 without --timesteps")
    scorer = Scorer(args)
    if args.dry_run:
        print("concepts:")
        for concept, (chosen_dir, rejected_dir) in zip(scorer.concepts, scorer.concept_dirs, strict=True):
            print(f"  {_label(concept)}: chosen={chosen_dir} rejected={rejected_dir}")
        print(f"origin: {scorer.origin_id['path']}")
        print(f"current: {scorer.current_id['path']}")
        print(f"cache: {scorer.config.cache_dir}")
        print(f"draws per pair: {len(scorer.draws)} -> {len(scorer.draws) * 2} forwards per pair per side")
        if not scorer.adapter_mode:
            print(f"origin pass needed: {'yes' if scorer.origin_pass_needed() else 'no (all origin scalars cached)'}")
        return
    scorer.run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", required=True, help="training config json (the run whose DPO concepts to score)")
        p.add_argument("--concepts", help="concept file; default: the config's concept_file_name")
        p.add_argument("--concept", action="append", help="only concepts whose name/path contains this (repeatable)")

    def analysis(p):
        p.add_argument(
            "--beta", type=float, help="override beta; default: concept override, else the config's global beta"
        )
        p.add_argument(
            "--dead-weight",
            type=float,
            default=sat.DEFAULT_DEAD_WEIGHT,
            help="pair is dead below this expected gradient weight",
        )
        p.add_argument(
            "--require-all-buckets",
            action="store_true",
            help="every populated timestep bucket must be dead, not just the mean",
        )
        p.add_argument("--min-draws", type=int, default=1)
        p.add_argument("--tails", type=int, default=10)
        p.add_argument(
            "--show-keys", action="store_true", help="print pair keys (file stems) in tails and retire lists"
        )

    p = sub.add_parser("score", help="run the origin/current passes and write the tables")
    common(p)
    p.add_argument("--current", help="current checkpoint (transformer safetensors)")
    p.add_argument("--origin", help="origin checkpoint; default: the config's transformer model")
    p.add_argument("--adapter", help="adapter mode: origin = base without it, current = base with it")
    p.add_argument("--draws", type=int, default=8, help="draws per pair from the training timestep distribution")
    p.add_argument("--timesteps", help="fixed noise levels, e.g. 0.1,0.3,0.5,0.7,0.9 (overrides --draws)")
    p.add_argument("--seeds", type=int, default=2, help="noise draws per fixed timestep")
    p.add_argument("--seed-base", type=int, default=1_000_000, help="first draw seed (a trainer global step)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-dir", help="latent cache to use; default: the config's cache_dir")
    p.add_argument("--device", help="train device override, e.g. cuda:1")
    p.add_argument("--force-origin", action="store_true", help="ignore cached origin scalars")
    p.add_argument("--num-train-timesteps", type=int, help="scheduler length; default: existing table, else 1000")
    p.add_argument("--dry-run", action="store_true", help="print the plan without loading a model")
    p.add_argument("--progress-json", action="store_true", help="emit one JSON progress line per batch on stdout")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("report", help="summarise the tables")
    common(p)
    analysis(p)
    p.add_argument("--csv", help="write per-pair rows here")
    p.add_argument("--json", help="write the summary here")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("retire", help="move saturated pairs out of the training tree")
    common(p)
    analysis(p)
    p.add_argument("--keys", help="retire exactly these pair keys instead of the saturated set")
    p.add_argument("--reason")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_retire)

    p = sub.add_parser("restore", help="put retired pairs back")
    common(p)
    p.add_argument("--keys", help="only these pair keys; default: all retired pairs")
    p.set_defaults(func=cmd_restore)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
