"""The dora_oft_learning_rate entry: config default, LoRA tab slot, queue labels and diff
groups, web tooltip. Parsed from source so no Tk / web stack is imported."""

from __future__ import annotations

import ast
import pathlib

from modules.util.config.TrainConfig import TrainConfig

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _tree(relative: str) -> ast.Module:
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def _const(node):
    return node.value if isinstance(node, ast.Constant) else None


def _component_calls(tree: ast.Module):
    """(kind, row, column, call) for every components.label/switch/entry/... grid call."""
    return [
        (node.func.attr, _const(node.args[1]), _const(node.args[2]), node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "components"
        and len(node.args) >= 3
    ]


def _oft_branch(tree: ast.Module) -> ast.If:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.comparators[0], ast.Attribute)
            and node.test.comparators[0].attr == "OFT_2"
            and _component_calls(ast.Module(body=node.body, type_ignores=[]))
        ):
            return node
    raise AssertionError("OFT_2 widget branch not found in LoraTab")


def test_config_defaults_to_no_separate_dora_learning_rate():
    config = TrainConfig.default_values()
    assert config.dora_oft_learning_rate is None

    config.dora_oft_learning_rate = 2e-5
    restored = TrainConfig.default_values().from_dict(config.to_dict())
    assert restored.dora_oft_learning_rate == 2e-5


def test_lora_tab_has_the_entry_in_its_own_oft_slot():
    branch = _oft_branch(_tree("modules/ui/LoraTab.py"))
    calls = _component_calls(ast.Module(body=branch.body, type_ignores=[]))

    entry = next(call for kind, _, _, call in calls if _const(call.args[-1]) == "dora_oft_learning_rate")
    assert entry.func.attr == "entry"
    row, column = _const(entry.args[1]), _const(entry.args[2])

    at_slot = sorted((kind, r, c) for kind, r, c, _ in calls if r == row and c in (column - 1, column))
    assert at_slot == [("entry", row, column), ("label", row, column - 1)]  # nothing else shares the slot

    label = next(call for kind, r, c, call in calls if kind == "label" and (r, c) == (row, column - 1))
    assert _const(label.args[3]) == "DoRA Learning Rate"
    tooltip = next(_const(k.value) for k in label.keywords if k.arg == "tooltip")
    for phrase in ("DoRA magnitude", "OFT rotation", "Leave empty", "DoRA OFT is enabled"):
        assert phrase in tooltip, phrase


def _dict_with_key(tree: ast.Module, key: str) -> dict:
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and any(_const(k) == key for k in node.keys):
            return {_const(k): _const(v) for k, v in zip(node.keys, node.values, strict=True)}
    raise AssertionError(f"no dict literal with key {key!r}")


def _group_containing(tree: ast.Module, member: str) -> tuple[str, list[str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 2 and isinstance(node.elts[1], ast.List):
            names = [_const(e) for e in node.elts[1].elts]
            if member in names:
                return _const(node.elts[0]), names
    raise AssertionError(f"no diff group containing {member!r}")


def test_queue_labels_and_diff_groups_list_it_next_to_dora_oft():
    for relative in ("modules/ui/QueueWindow.py", "web/backend/services/_queue_diff.py"):
        tree = _tree(relative)
        labels = _dict_with_key(tree, "dora_oft")
        assert labels["dora_oft_learning_rate"] == "DoRA Learning Rate", relative
        group, names = _group_containing(tree, "dora_oft")
        assert group == "LoRA / OFT" and "dora_oft_learning_rate" in names, relative


def test_web_metadata_has_the_tooltip():
    tooltips = _dict_with_key(_tree("web/scripts/ui_metadata.py"), "oft_block_share")
    assert "dora_oft_learning_rate" in tooltips
