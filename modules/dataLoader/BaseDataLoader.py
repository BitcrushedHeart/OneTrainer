import copy
from abc import ABCMeta, abstractmethod

from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin
from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS, TrainDataLoader

import torch
from torch.utils.data._utils.collate import default_collate

_IDENTITY_KEYS = ('image_path', 'sample_prompt_path', 'concept')


def _shape_summary(value):
    if isinstance(value, torch.Tensor):
        return f"tensor{tuple(value.shape)}"
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__}{tuple(value)}"
    return type(value).__name__


def _identity(item: dict) -> str:
    for k in _IDENTITY_KEYS:
        v = item.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict) and v.get('path'):
            return v['path']
    return '<unknown>'


def _aspect_disagrees(spatial_shape, crop_resolution) -> bool:
    """True when the H/W ratio of a tensor's spatial dims disagrees with
    the cached crop_resolution tuple's H/W ratio.

    Catches the within-.pt inconsistency we hit when an old cache build
    dedup'd or augment-in-place rewrote crop_resolution out of sync with
    the actual cached tensor shapes.
    """
    if not isinstance(crop_resolution, (tuple, list)) or len(crop_resolution) < 2:
        return False
    try:
        cr_h, cr_w = int(crop_resolution[-2]), int(crop_resolution[-1])
        t_h, t_w = int(spatial_shape[-2]), int(spatial_shape[-1])
    except (TypeError, ValueError, IndexError):
        return False
    if cr_w == 0 or t_w == 0:
        return False
    cr_landscape = cr_h < cr_w
    t_landscape = t_h < t_w
    cr_square = cr_h == cr_w
    t_square = t_h == t_w
    if cr_square or t_square:
        return False
    return cr_landscape != t_landscape


def _shape_safe_collate(batch):
    """Default-collate with a helpful error when tensors don't stack.

    Why: a single mis-shaped cache entry crashes ``torch.stack`` deep inside
    the dataloader's collate with no indication of which file is at fault.
    On stack failure we walk the batch ourselves and raise a richer error
    naming the offending key, every item's full tensor shape map, the
    cached crop_resolution, and an explicit flag for entries whose
    crop_resolution disagrees in orientation with their tensor shapes
    (the smoking gun for stale-aggregate cache corruption).
    """
    try:
        return default_collate(batch)
    except RuntimeError as e:
        if 'stack expects each tensor to be equal size' not in str(e):
            raise
        if not batch or not isinstance(batch[0], dict):
            raise

        bad_keys = []
        for key in batch[0]:
            values = [item.get(key) for item in batch]
            if not all(isinstance(v, torch.Tensor) for v in values):
                continue
            shapes = {tuple(v.shape) for v in values}
            if len(shapes) > 1:
                bad_keys.append(key)

        lines = [
            f"Batch shape mismatch in collate (original: {e}).",
            f"Batch size: {len(batch)}.",
            "Per-entry tensor shapes vs cached crop_resolution:",
        ]
        for i, item in enumerate(batch):
            cr = item.get('crop_resolution')
            tensor_shapes = {
                k: tuple(v.shape)
                for k, v in item.items()
                if isinstance(v, torch.Tensor) and v.dim() >= 2
            }
            disagreeing = [
                k for k, s in tensor_shapes.items()
                if _aspect_disagrees(s, cr)
            ]
            mark = " <-- AGGREGATE/TENSOR ORIENTATION MISMATCH" if disagreeing else ""
            lines.append(
                f"  [{i}] crop_resolution={cr} source={_identity(item)}{mark}"
            )
            for k in sorted(tensor_shapes):
                marker = "  *" if k in disagreeing else "   "
                lines.append(f"    {marker} {k}={tensor_shapes[k]}")
        if bad_keys:
            lines.append(f"Keys with cross-entry shape disagreement: {bad_keys}")
        raise RuntimeError("\n".join(lines)) from e


class BaseDataLoader(
    DataLoaderMgdsMixin,
    metaclass=ABCMeta,
):

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
            is_sft_anchor: bool = False,
    ):
        super().__init__()

        self.train_device = train_device
        self.temp_device = temp_device
        self.stop_check_fun = lambda: False
        self.is_sft_anchor = is_sft_anchor

        if is_validation:
            config = copy.copy(config)
            config.batch_size = 1
            config.multi_gpu = False
            config.rlhf_enabled = config.rlhf_dpo_validation and config.rlhf_enabled
        elif is_sft_anchor:
            # Parallel SFT-anchor loader for DPO runs that include STANDARD
            # concepts: re-use the standard (non-DPO) data pipeline so the
            # PairByFilename module is skipped and STANDARD samples flow as
            # ordinary supervised batches.
            config = copy.copy(config)
            config.rlhf_enabled = False

        self.__ds = self._create_dataset(
            config=config,
            model=model,
            model_setup=model_setup,
            train_progress=train_progress,
            is_validation=is_validation,
        )
        self.__dl = TrainDataLoader(self.__ds, config.batch_size)
        self.__dl.collate_fn = _shape_safe_collate

    def get_data_set(self) -> MGDS:
        return self.__ds

    def get_data_loader(self) -> TrainDataLoader:
        return self.__dl

    @abstractmethod
    def _create_dataset(
            self,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation,
    ):
        pass
