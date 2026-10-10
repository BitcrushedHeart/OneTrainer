from collections.abc import Iterable
from functools import cached_property

from modules.util.config.TrainConfig import TrainConfig

from torch.nn import Parameter


class NamedParameterGroup:
    def __init__(
        self,
        unique_name: str,
        parameters: Iterable[Parameter],
        learning_rate: float,
        display_name: str | None = None,
        key_space: str | None = None,
        key_indices: Iterable[int] | None = None,
    ):
        self.unique_name = unique_name
        self.display_name = display_name if display_name is not None else unique_name
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        # A parameter's identity is (key_space, key index): by default its position in
        # this group. A group split out of another one keeps the other's key space and
        # indices, so the split changes learning rates but not identities.
        self.key_space = key_space if key_space is not None else unique_name
        self.key_indices = list(key_indices) if key_indices is not None else list(range(len(self.parameters)))

    def split(
        self,
        unique_name: str,
        parameters: Iterable[Parameter],
        learning_rate: float | None,
    ) -> "NamedParameterGroup | None":
        """Move ``parameters`` out of this group into a new group with its own learning rate.

        Both groups keep the keys the parameters had before the split, so state saved
        under either layout (optimizer state, EMA, accumulated gradients) binds to the
        same parameters under the other. Returns None if none of them are in this group.
        """
        moved_ids = {id(p) for p in parameters}
        moved = [i for i, p in enumerate(self.parameters) if id(p) in moved_ids]
        if not moved:
            return None
        kept = [i for i, p in enumerate(self.parameters) if id(p) not in moved_ids]

        split_group = NamedParameterGroup(
            unique_name=unique_name,
            parameters=[self.parameters[i] for i in moved],
            learning_rate=learning_rate,
            key_space=self.key_space,
            key_indices=[self.key_indices[i] for i in moved],
        )
        self.parameters = [self.parameters[i] for i in kept]
        self.key_indices = [self.key_indices[i] for i in kept]
        return split_group


class NamedParameterGroupCollection:
    __groups: list[NamedParameterGroup]

    def __init__(self):
        self.__groups = []

    def add_group(self, group: NamedParameterGroup):
        self.__groups.append(group)

    def __keyed_parameters(self) -> list[tuple[str, Parameter]]:
        # Key order, not group order: groups sharing a key space (a split group) are
        # interleaved back by key index, so the order does not depend on the split.
        by_space: dict[str, list[tuple[int, Parameter]]] = {}
        for group in self.__groups:
            by_space.setdefault(group.key_space, []).extend(zip(group.key_indices, group.parameters, strict=True))
        return [
            (f"{key_space}.{index}", p)
            for key_space, entries in by_space.items()
            for index, p in sorted(entries, key=lambda entry: entry[0])
        ]

    def parameters(self) -> list[Parameter]:
        return [p for _, p in self.__keyed_parameters()]

    def iter_named_parameters(self) -> Iterable[tuple[str, Parameter]]:
        """Yield (stable_key, parameter) pairs in deterministic order.

        Key format is ``"{group.key_space}.{key_index}"`` -- for an unsplit group
        ``"{group.unique_name}.{index}"`` -- so the same key identifies the same
        parameter across stop/resume as long as ``unique_name`` and the group's
        parameter ordering are stable (both are byte-for-byte reproducible at
        construction time), whether or not the group is split.
        Used by the accumulator-state save/load path to map saved
        ``.grad`` tensors back onto live parameters.
        """
        yield from self.__keyed_parameters()

    def parameters_for_optimizer(self, config: TrainConfig) -> list[dict]:
        parameters = []

        for group in self.__groups:
            # Determine the learning rate
            lr = group.learning_rate if group.learning_rate is not None else config.learning_rate
            lr = lr * (
                (config.learning_rate_scaler.get_scale(config.batch_size, config.gradient_accumulation_steps)) ** 0.5
            )

            # Create a parameter group for the text encoder
            parameters.append(
                {
                    "name": group.display_name,
                    "params": list(group.parameters),
                    "lr": lr,
                    "initial_lr": lr,
                }
            )

        return parameters

    @cached_property
    def unique_name_mapping(self) -> list[str]:
        return [group.unique_name for group in self.__groups]

    @cached_property
    def display_name_mapping(self) -> list[str]:
        return [group.display_name for group in self.__groups]

    @property
    def key_mapping(self) -> list[tuple[str, list[int]]]:
        """(key space, key indices) of each group, in group order."""
        return [(group.key_space, list(group.key_indices)) for group in self.__groups]
