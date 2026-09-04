"""SFT + DPO mode helpers.

The standard finetune pipeline stays the primary loader (it drives epochs,
progress and caching exactly like a plain run) and a secondary loader of DPO
pairs is interleaved into it every N micro-batches.
"""


def derive_dpo_addon_interval(sft_batches: int, dpo_batches: int, override: int = 0) -> int:
    """Standard micro-batches per interleaved DPO pair.

    With ``override <= 0`` the interval is the rounded ratio of the two loader
    lengths, so a DPO concept with R repeats finishes its epoch together with
    the standard data (96k standard / (2k pairs * 10 repeats) -> every 5th).
    """
    if override > 0:
        return override
    if sft_batches <= 0 or dpo_batches <= 0:
        return 1
    return max(1, round(sft_batches / dpo_batches))


def is_dpo_addon_step(epoch_step: int, interval: int) -> bool:
    return interval >= 1 and epoch_step % interval == 0


def dpo_ref_rebase_due(global_step: int, interval: int, last_bucket: int) -> tuple[bool, int]:
    """Whether a periodic DPO reference rebase is due at ``global_step``.

    Steps are grouped into buckets of ``interval``; a rebase is due the first
    time the caller sees a new bucket. Checking at the optimizer step (not the
    micro-batch) keeps the reference fixed across a gradient-accumulation
    window, and seeding ``last_bucket`` from the resume step means a resumed
    run waits for the next boundary instead of rebasing immediately. Returns
    ``(due, bucket)``; ``interval <= 0`` disables and echoes ``last_bucket``.
    """
    if interval <= 0:
        return False, last_bucket
    bucket = global_step // interval
    return bucket > last_bucket, bucket


def accumulate_dpo_metrics(accumulated: dict | None, micro_metrics: dict, extra_keys=()) -> dict:
    """Sum per-micro-batch DPO metrics across a gradient-accumulation window.

    ``_count`` tracks how many DPO micro-batches contributed so the logger can
    average; ``extra_keys`` are seeded to 0.0 for callers that add their own
    terms (the SFT-anchor loss) outside the DPO metrics dict.
    """
    if accumulated is None:
        accumulated = dict.fromkeys(micro_metrics, 0.0)
        accumulated["_count"] = 0
    for key in extra_keys:
        accumulated.setdefault(key, 0.0)
    for key, value in micro_metrics.items():
        accumulated[key] = accumulated.get(key, 0.0) + value
    accumulated["_count"] += 1
    return accumulated


class CyclingBatchSource:
    """Pulls batches from a secondary data loader, restarting its dataset epoch
    when it runs dry.

    The primary and secondary loaders generally differ in length, so the
    secondary cycles independently of the primary epoch.
    """

    def __init__(self, data_loader, name: str, on_epoch_restart=None):
        self.data_loader = data_loader
        self.name = name
        # Called after a mid-step epoch restart (see next()). The restart runs
        # the loader's caching pass, which moves the model off the train device
        # and into eval mode; the caller uses this to put it back before the
        # next forward. Not called from start_next_epoch(), whose callers
        # already restore the device themselves.
        self.on_epoch_restart = on_epoch_restart
        self._iter = None

    def start_next_epoch(self):
        self.data_loader.get_data_set().start_next_epoch()
        self._iter = None

    def approximate_length(self) -> int:
        return self.data_loader.get_data_set().approximate_length()

    def next(self):
        for _ in range(2):
            if self._iter is None:
                self._iter = iter(self.data_loader.get_data_loader())
            try:
                return next(self._iter)
            except StopIteration:
                self.data_loader.get_data_set().start_next_epoch()
                self._iter = None
                if self.on_epoch_restart is not None:
                    self.on_epoch_restart()
        raise RuntimeError(
            f"{self.name} data loader produced no batches after a full epoch restart. "
            "Check that its concepts are enabled and contain images."
        )
