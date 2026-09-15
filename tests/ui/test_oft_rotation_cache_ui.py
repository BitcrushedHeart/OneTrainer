"""The oft_rotation_cache toggle: LoRA tab slot and tooltip, queue labels and diff groups,
web tooltip. Parsed from source so no Tk / web stack is imported."""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _tree(relative: str) -> ast.Module:
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def _const(node):
    return node.value if isinstance(node, ast.Constant) else None


def _oft_branch(tree: ast.Module) -> list[ast.stmt]:
    """Body of the `elif peft_type == PeftType.OFT_2:` branch that lays out the OFT widgets."""
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.comparators[0], ast.Attribute)
                and test.comparators[0].attr == "OFT_2"
                and _grid_calls(node.body)
            ):
                return node.body
    raise AssertionError("OFT_2 widget branch not found in LoraTab")


def _grid_calls(body: list[ast.stmt]):
    """(kind, row, column, call) for components.label/switch/entry/... calls in the branch."""
    nodes = [node for stmt in body for node in ast.walk(stmt)]
    return [
        (node.func.attr, _const(node.args[1]), _const(node.args[2]), node)
        for node in nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "components"
        and len(node.args) >= 3
    ]


def test_lora_tab_places_the_toggle_in_the_free_oft_slot_with_the_tooltip():
    calls = _grid_calls(_oft_branch(_tree("modules/ui/LoraTab.py")))

    at_slot = [(kind, row, col) for kind, row, col, _ in calls if row == 3 and col in (3, 4)]
    assert sorted(at_slot) == [("label", 3, 3), ("switch", 3, 4)]  # nothing else shares the slot

    switch = next(call for kind, row, col, call in calls if kind == "switch" and (row, col) == (3, 4))
    assert _const(switch.args[-1]) == "oft_rotation_cache"

    label = next(call for kind, row, col, call in calls if kind == "label" and (row, col) == (3, 3))
    tooltip = next(_const(k.value) for k in label.keywords if k.arg == "tooltip")
    for phrase in (
        "1.6 GiB",
        "block_size^2",
        "Faster steps",
        "stochastic rounding",
        "unbiased",
        "bfloat16",
        "single GPU",
    ):
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


def test_queue_labels_and_diff_groups_follow_the_other_oft_booleans():
    for relative in ("modules/ui/QueueWindow.py", "web/backend/services/_queue_diff.py"):
        tree = _tree(relative)
        labels = _dict_with_key(tree, "oft_block_share")
        assert labels["oft_rotation_cache"] == "OFT Rotation Cache", relative
        group, names = _group_containing(tree, "oft_block_share")
        assert group == "LoRA / OFT" and "oft_rotation_cache" in names, relative


def test_web_metadata_has_the_tooltip():
    tooltips = _dict_with_key(_tree("web/scripts/ui_metadata.py"), "oft_block_share")
    assert "oft_rotation_cache" in tooltips
