import os

from modules.util.dpo_curation_util import dpo_pair_key

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.CollectPaths import CollectPaths
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.pipelineModuleTypes.SerialPipelineModule import SerialPipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import (
    SingleVariationRandomAccessPipelineModule,
)

import torch


def _to_resolution_tuple(value) -> tuple | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return tuple(value.detach().cpu().flatten().tolist())
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _resolutions_match(chosen, rejected) -> bool:
    chosen_tuple = _to_resolution_tuple(chosen)
    rejected_tuple = _to_resolution_tuple(rejected)
    if chosen_tuple is None or rejected_tuple is None:
        return chosen_tuple == rejected_tuple
    return chosen_tuple == rejected_tuple


class PairByFilename(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
        self,
        concept_pairs: list[tuple[str, str]],
        chosen_names: list[str | tuple[str, str]],
        rejected_names: list[str | tuple[str, str]],
    ):
        super().__init__()

        self.chosen_names = [x if isinstance(x, tuple) else (x, x) for x in chosen_names]
        self.rejected_names = [x if isinstance(x, tuple) else (x, x) for x in rejected_names]

        self.concept_lookup = {}
        for pair_id, (chosen_path, rejected_path) in enumerate(concept_pairs):
            self.concept_lookup[self.__canonical_path(chosen_path)] = (pair_id, True)
            self.concept_lookup[self.__canonical_path(rejected_path)] = (pair_id, False)

        self._pair_indices: list[tuple[int, int]] | None = None
        # Matched indices + paths shared between the lazy seed-override
        # provider (called from AspectBucketing during SmartDiskCache start)
        # and the lazy resolution-mismatch pre-filter (called from the first
        # length()/get_item). Both phases need the same chosen↔rejected
        # matching, so we build it once and reuse.
        self._matched_pairs: list[tuple[int, int]] | None = None
        self._image_paths_by_index: dict[int, str] | None = None

    def init(self, pipeline, base_seed: int, module_index: int, state):
        super().init(pipeline, base_seed, module_index, state)
        # Register the chosen↔rejected map provider on DPOAspectBucketing.
        # The provider is invoked lazily on the first rand lookup inside the
        # bucketing module — by then CollectPaths.start has populated paths so
        # the build can complete. Installing the map directly here would be
        # too early (CollectPaths is empty before start). Installing from our
        # own start() would be too late (SmartDiskCache.start runs first in
        # pipeline order and bakes its aggregate cache from bucketing output).
        from modules.dataLoader.dpo.DPOAspectBucketing import DPOAspectBucketing

        for module in pipeline.modules:
            if isinstance(module, DPOAspectBucketing):
                module._pair_map_provider = self.__build_pair_map
                break

    @staticmethod
    def __canonical_path(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def __ensure_matched_pairs(self) -> tuple[list[tuple[int, int]], dict[int, str]]:
        if self._matched_pairs is not None and self._image_paths_by_index is not None:
            return self._matched_pairs, self._image_paths_by_index

        chosen_indices: dict[tuple, int] = {}
        rejected_indices: dict[tuple, int] = {}
        image_paths_by_index: dict[int, str] = {}

        collect_paths = self.__find_collect_paths()
        for index in range(collect_paths.length()):
            item = collect_paths.get_item(0, index)
            concept = item["concept"]
            image_path = item["image_path"]
            concept_path = self.__canonical_path(concept["path"])
            pair_info = self.concept_lookup.get(concept_path)
            if pair_info is None:
                continue

            pair_id, is_chosen = pair_info
            key = (pair_id, dpo_pair_key(image_path, concept_path))
            image_paths_by_index[index] = image_path

            if is_chosen:
                chosen_indices[key] = index
            else:
                rejected_indices[key] = index

        missing_rejected = sorted(set(chosen_indices) - set(rejected_indices))
        missing_chosen = sorted(set(rejected_indices) - set(chosen_indices))
        if missing_rejected or missing_chosen:
            details = []
            if missing_rejected:
                details.append(f"{len(missing_rejected)} chosen files are missing rejected matches")
            if missing_chosen:
                details.append(f"{len(missing_chosen)} rejected files are missing chosen matches")
            raise RuntimeError("RLHF DPO concept pairs must match exactly by filename: " + ", ".join(details) + ".")

        matched: list[tuple[int, int]] = []
        for key, chosen_index in chosen_indices.items():
            rejected_index = rejected_indices.get(key)
            if rejected_index is not None:
                matched.append((chosen_index, rejected_index))
        matched.sort(key=lambda x: x[0])

        if not matched:
            raise RuntimeError(
                "No DPO pairs could be matched by filename between the configured chosen/rejected concepts."
            )

        self._matched_pairs = matched
        self._image_paths_by_index = image_paths_by_index
        return matched, image_paths_by_index

    def __build_pair_map(self) -> dict[int, int]:
        # {rejected_raw_index: chosen_raw_index}. DPOAspectBucketing uses
        # this to substitute the chosen index for all bucketing decisions
        # whenever an upstream lookup arrives for a rejected sample.
        matched, _ = self.__ensure_matched_pairs()
        pair_map: dict[int, int] = {rej: chosen for chosen, rej in matched}
        print(f"[PairByFilename] Built chosen-aligned pair map for {len(matched)} pairs.")
        return pair_map

    def __find_collect_paths(self) -> CollectPaths:
        # Why: SmartDiskCache exposes 'image_path' as an aggregate output and its
        # blank-sentinel fallback returns a fixed, unrelated path for any item
        # whose cache build failed. Going through `_get_previous_item` would
        # stop at the cache and yield that fixed path, collapsing many distinct
        # pair_NNNN stems into one key per side and falsely reporting missing
        # pairs. Pull image_path + concept directly from the CollectPaths
        # module that owns the filesystem enumeration.
        for module in self.pipeline.modules:
            if isinstance(module, CollectPaths):
                return module
        raise RuntimeError("PairByFilename could not locate the CollectPaths module in the pipeline.")

    def __build_pair_indices(self):
        matched, image_paths_by_index = self.__ensure_matched_pairs()
        # With DPOAspectBucketing in the pipeline this pre-filter should
        # report zero drops — alignment forces chosen and rejected onto
        # the same bucket+target at every variation. The check stays in as
        # a defense-in-depth net for pipelines that aren't bucket-aligned
        # (fixed-resolution training, or future cache backends).
        pair_indices, dropped = self.__drop_resolution_mismatches(list(matched), image_paths_by_index)
        if dropped:
            self.__log_dropped_pairs(dropped)
        if not pair_indices:
            raise RuntimeError(
                "All DPO pairs were dropped due to mismatched crop resolutions between chosen and rejected samples."
            )

        self._pair_indices = pair_indices

    def __resolve_upstream_variation(self, name: str) -> int | None:
        # Why: SmartDiskCache (and other SingleVariation modules) reject
        # _get_previous_item calls whose variation does not match their
        # current_variation. During start_next_epoch the epoch is already
        # written onto those upstream modules' current_variation; mirror that
        # value instead of guessing 0.
        for module in self.pipeline.modules:
            if module is self:
                break
            if name in module.get_outputs() and isinstance(
                module, (SingleVariationRandomAccessPipelineModule, SerialPipelineModule)
            ):
                return getattr(module, "current_variation", None)
        return None

    def __drop_resolution_mismatches(
        self,
        pair_indices: list[tuple[int, int]],
        image_paths_by_index: dict[int, str],
    ) -> tuple[list[tuple[int, int]], list[tuple[str, str, object, object]]]:
        kept: list[tuple[int, int]] = []
        dropped: list[tuple[str, str, object, object]] = []
        variation = self.__resolve_upstream_variation("crop_resolution")
        if variation is None:
            # Fall back to no pre-filter; the get_item runtime check still
            # protects training, just with a hard failure.
            return pair_indices, []
        for chosen_index, rejected_index in pair_indices:
            chosen_res = self._get_previous_item(variation, "crop_resolution", chosen_index)
            rejected_res = self._get_previous_item(variation, "crop_resolution", rejected_index)
            if _resolutions_match(chosen_res, rejected_res):
                kept.append((chosen_index, rejected_index))
            else:
                dropped.append(
                    (
                        image_paths_by_index.get(chosen_index, f"<index {chosen_index}>"),
                        image_paths_by_index.get(rejected_index, f"<index {rejected_index}>"),
                        chosen_res,
                        rejected_res,
                    )
                )
        return kept, dropped

    def __log_dropped_pairs(self, dropped: list[tuple[str, str, object, object]]):
        print(
            f"[PairByFilename] Dropped {len(dropped)} DPO pair(s) with mismatched crop resolutions "
            f"between chosen and rejected samples."
        )
        try:
            log_path = os.path.join(os.getcwd(), "dpo_dropped_resolution_mismatches.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(
                    "# DPO pairs dropped because chosen/rejected crop resolutions did not match.\n"
                    "# Format: <chosen_crop_resolution> <rejected_crop_resolution>\\t<chosen_path>\\t<rejected_path>\n"
                )
                for chosen_path, rejected_path, chosen_res, rejected_res in dropped:
                    f.write(
                        f"{tuple(chosen_res.tolist()) if isinstance(chosen_res, torch.Tensor) else chosen_res} "
                        f"{tuple(rejected_res.tolist()) if isinstance(rejected_res, torch.Tensor) else rejected_res}\t"
                        f"{chosen_path}\t{rejected_path}\n"
                    )
            print(f"[PairByFilename] Wrote mismatch list to {log_path}")
        except OSError as e:
            print(f"[PairByFilename] Could not write mismatch log: {e}")

    def __get_pair_indices(self) -> list[tuple[int, int]]:
        if self._pair_indices is None:
            self.__build_pair_indices()
        return self._pair_indices

    def length(self) -> int:
        return len(self.__get_pair_indices())

    def get_inputs(self) -> list[str]:
        names = ["concept.path", "image_path", "prompt", "crop_resolution"]
        names += [in_name for in_name, _ in self.chosen_names]
        names += [in_name for in_name, _ in self.rejected_names]
        return list(dict.fromkeys(names))

    def get_outputs(self) -> list[str]:
        return [out_name for _, out_name in self.chosen_names] + [out_name for _, out_name in self.rejected_names]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        chosen_index, rejected_index = self.__get_pair_indices()[index]

        chosen_prompt = self._get_previous_item(variation, "prompt", chosen_index)
        rejected_prompt = self._get_previous_item(variation, "prompt", rejected_index)
        if chosen_prompt != rejected_prompt:
            raise RuntimeError(
                "RLHF DPO paired samples must use identical prompts/captions in chosen and rejected concepts."
            )

        chosen_crop_resolution = self._get_previous_item(variation, "crop_resolution", chosen_index)
        rejected_crop_resolution = self._get_previous_item(variation, "crop_resolution", rejected_index)
        if not _resolutions_match(chosen_crop_resolution, rejected_crop_resolution):
            raise RuntimeError(
                "RLHF DPO paired samples must have matching crop resolutions in chosen and rejected concepts."
            )

        item = {}
        for in_name, out_name in self.chosen_names:
            item[out_name] = self._get_previous_item(variation, in_name, chosen_index)
        for in_name, out_name in self.rejected_names:
            item[out_name] = self._get_previous_item(variation, in_name, rejected_index)

        return item
