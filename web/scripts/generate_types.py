import ast
import importlib
import inspect
import json
import os
import sys
from enum import Enum
from typing import Any, get_args, get_origin

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from modules.util.config.BaseConfig import BaseConfig
from modules.util.type_util import issubclass_safe

_incomplete_generation = False

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "web", "gui", "src", "renderer", "types", "generated"
)

ENUM_DIR = os.path.join(PROJECT_ROOT, "modules", "util", "enum")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "modules", "util", "config")

_KNOWN_ENUM_MODULES = {
    "modules.util.enum.AudioFormat",
    "modules.util.enum.DataType",
    "modules.util.enum.ModelType",
    "modules.util.enum.Optimizer",
    "modules.util.enum.TrainingMethod",
}

_KNOWN_CONFIG_CLASSES = {
    "TrainConfig", "TrainOptimizerConfig", "TrainModelPartConfig",
    "TrainEmbeddingConfig", "QuantizationConfig", "ConceptConfig",
    "ConceptImageConfig", "ConceptTextConfig", "SampleConfig",
    "CloudConfig", "CloudSecretsConfig", "SecretsConfig",
}


def discover_enum_modules() -> list[str]:
    modules = []
    for filename in sorted(os.listdir(ENUM_DIR)):
        if filename.startswith("_") or not filename.endswith(".py"):
            continue
        module_name = filename[:-3]
        module_path = f"modules.util.enum.{module_name}"
        modules.append(module_path)
    return modules


def discover_config_classes() -> list[tuple[str, type]]:
    configs = []
    seen = set()
    for filename in sorted(os.listdir(CONFIG_DIR)):
        if filename.startswith("_") or not filename.endswith(".py"):
            continue
        module_name = filename[:-3]
        module_path = f"modules.util.config.{module_name}"
        try:
            mod = importlib.import_module(module_path)
        except Exception:
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                isinstance(obj, type)
                and issubclass_safe(obj, BaseConfig)
                and obj is not BaseConfig
                and hasattr(obj, "default_values")
                and attr_name not in seen
            ):
                seen.add(attr_name)
                configs.append((attr_name, obj))
    return configs


ENUM_MODULES = discover_enum_modules()

_missing_enums = _KNOWN_ENUM_MODULES - set(ENUM_MODULES)
if _missing_enums:
    raise RuntimeError(f"Dynamic enum scan missed known modules: {_missing_enums}")


def collect_enums() -> list[tuple[str, type]]:
    enums = []
    for module_path in ENUM_MODULES:
        mod = importlib.import_module(module_path)
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, Enum) and obj is not Enum:
                enums.append((name, obj))
    return enums


def collect_configs() -> list[tuple[str, type]]:
    configs = discover_config_classes()
    found_names = {name for name, _ in configs}
    missing = _KNOWN_CONFIG_CLASSES - found_names
    if missing:
        raise RuntimeError(f"Dynamic config scan missed known classes: {missing}")
    return configs


def python_type_to_ts(py_type: type, nullable: bool, enum_names: set[str]) -> str:
    if py_type is str:
        ts = "string"
    elif py_type is bool:
        ts = "boolean"
    elif py_type is int or py_type is float:
        ts = "number"
    elif py_type is dict:
        ts = "Record<string, unknown>"
    elif issubclass_safe(py_type, Enum) or issubclass_safe(py_type, BaseConfig):
        ts = py_type.__name__
    elif py_type is list or get_origin(py_type) is list:
        args = get_args(py_type)
        if args:
            inner = args[0]
            if issubclass_safe(inner, BaseConfig) or issubclass_safe(inner, Enum):
                ts = f"{inner.__name__}[]"
            elif inner is str:
                ts = "string[]"
            elif inner is int or inner is float:
                ts = "number[]"
            elif inner is bool:
                ts = "boolean[]"
            elif get_origin(inner) is dict:
                dict_args = get_args(inner)
                if dict_args and len(dict_args) == 2:
                    key_ts = python_type_to_ts(dict_args[0], False, enum_names)
                    val_ts = python_type_to_ts(dict_args[1], False, enum_names)
                    ts = f"Record<{key_ts}, {val_ts}>[]"
                else:
                    ts = "Record<string, unknown>[]"
            else:
                ts = "unknown[]"
        else:
            ts = "unknown[]"
    elif get_origin(py_type) is dict:
        args = get_args(py_type)
        if args and len(args) == 2:
            key_ts = python_type_to_ts(args[0], False, enum_names)
            val_ts = python_type_to_ts(args[1], False, enum_names)
            ts = f"Record<{key_ts}, {val_ts}>"
        else:
            ts = "Record<string, unknown>"
    else:
        ts = "unknown"

    if nullable:
        ts = f"{ts} | null"

    return ts


def generate_enums_ts(enums: list[tuple[str, type]]) -> str:
    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when backend enums change.",
        "",
    ]

    for name, enum_cls in sorted(enums, key=lambda x: x[0]):
        members = list(enum_cls)
        lines.append(f"export type {name} =")
        for i, member in enumerate(members):
            separator = ";" if i == len(members) - 1 else ""
            lines.append(f"  | '{member.value}'{separator}")
        lines.append("")

        lines.append(f"export const {name}Values: {name}[] = [")
        lines.extend(f"  '{member.value}'," for member in members)
        lines.append("];")
        lines.append("")

    return "\n".join(lines)


def generate_config_ts(
    configs: list[tuple[str, type]], enum_names: set[str]
) -> str:
    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when backend config classes change.",
        "",
        "import type {",
    ]

    used_enums = set()
    for _class_name, cls in configs:
        instance = cls.default_values()
        for field_type in instance.types.values():
            if issubclass_safe(field_type, Enum):
                used_enums.add(field_type.__name__)
            elif get_origin(field_type) is list:
                args = get_args(field_type)
                if args and issubclass_safe(args[0], Enum):
                    used_enums.add(args[0].__name__)
            elif get_origin(field_type) is dict:
                args = get_args(field_type)
                if args:
                    for arg in args:
                        if issubclass_safe(arg, Enum):
                            used_enums.add(arg.__name__)

    lines.extend(f"  {enum_name}," for enum_name in sorted(used_enums))
    lines.append("} from './enums';")
    lines.append("")

    generated = set()

    def generate_interface(class_name: str, cls: type):
        if class_name in generated:
            return ""
        generated.add(class_name)

        result_lines = []

        instance = cls.default_values()
        for field_type in instance.types.values():
            if issubclass_safe(field_type, BaseConfig) and field_type.__name__ not in generated:
                for cn, cc in configs:
                    if cn == field_type.__name__:
                        dep = generate_interface(cn, cc)
                        if dep:
                            result_lines.append(dep)
                        break
            elif get_origin(field_type) is list:
                args = get_args(field_type)
                if args and issubclass_safe(args[0], BaseConfig) and args[0].__name__ not in generated:
                    for cn, cc in configs:
                        if cn == args[0].__name__:
                            dep = generate_interface(cn, cc)
                            if dep:
                                result_lines.append(dep)
                            break
            elif get_origin(field_type) is dict:
                args = get_args(field_type)
                if args and len(args) > 1 and issubclass_safe(args[1], BaseConfig) and args[1].__name__ not in generated:
                    for cn, cc in configs:
                        if cn == args[1].__name__:
                            dep = generate_interface(cn, cc)
                            if dep:
                                result_lines.append(dep)
                            break

        iface_lines = [f"export interface {class_name} {{"]

        for field_name in instance.types:
            field_type = instance.types[field_name]
            nullable = instance.nullables.get(field_name, False)
            ts_type = python_type_to_ts(field_type, nullable, enum_names)
            iface_lines.append(f"  {field_name}: {ts_type};")

        iface_lines.append("}")
        iface_lines.append("")

        result_lines.append("\n".join(iface_lines))
        return "\n".join(result_lines)

    for class_name, cls in configs:
        result = generate_interface(class_name, cls)
        if result:
            lines.append(result)

    return "\n".join(lines)


def generate_metadata_ts(
    configs: list[tuple[str, type]], enum_names: set[str]
) -> str:
    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when backend config classes change.",
        "",
        "export interface FieldMetadata {",
        "  type: 'string' | 'number' | 'boolean' | 'enum' | 'config' | 'list' | 'dict';",
        "  nullable: boolean;",
        "  enumType?: string;",
        "  configType?: string;",
        "  defaultValue: unknown;",
        "}",
        "",
        "export type ConfigMetadata = Record<string, FieldMetadata>;",
        "",
    ]

    for class_name, cls in configs:
        instance = cls.default_values()
        lines.append(f"export const {class_name}Metadata: ConfigMetadata = {{")

        for field_name in instance.types:
            field_type = instance.types[field_name]
            nullable = instance.nullables.get(field_name, False)
            default_value = instance.default_values.get(field_name)

            if field_type is str:
                meta_type = "string"
                enum_type = None
                config_type = None
            elif field_type is bool:
                meta_type = "boolean"
                enum_type = None
                config_type = None
            elif field_type is int or field_type is float:
                meta_type = "number"
                enum_type = None
                config_type = None
            elif issubclass_safe(field_type, Enum):
                meta_type = "enum"
                enum_type = field_type.__name__
                config_type = None
            elif issubclass_safe(field_type, BaseConfig):
                meta_type = "config"
                enum_type = None
                config_type = field_type.__name__
            elif field_type is list or get_origin(field_type) is list:
                meta_type = "list"
                enum_type = None
                config_type = None
                args = get_args(field_type)
                if args and issubclass_safe(args[0], BaseConfig):
                    config_type = args[0].__name__
            elif field_type is dict or get_origin(field_type) is dict:
                meta_type = "dict"
                enum_type = None
                config_type = None
                args = get_args(field_type)
                if args and len(args) > 1 and issubclass_safe(args[1], BaseConfig):
                    config_type = args[1].__name__
            else:
                meta_type = "string"
                enum_type = None
                config_type = None

            if default_value is None:
                default_json = "null"
            elif isinstance(default_value, bool):
                default_json = "true" if default_value else "false"
            elif isinstance(default_value, (int, float)):
                if default_value == float("inf"):
                    default_json = '"Infinity"'
                elif default_value == float("-inf"):
                    default_json = '"-Infinity"'
                else:
                    default_json = json.dumps(default_value)
            elif isinstance(default_value, str):
                default_json = json.dumps(default_value)
            elif isinstance(default_value, Enum):
                default_json = json.dumps(str(default_value))
            elif isinstance(default_value, BaseConfig):
                default_json = "null"
            elif isinstance(default_value, list):
                default_json = "[]"
            elif isinstance(default_value, dict):
                default_json = "{}"
            else:
                default_json = "null"

            enum_str = f', enumType: "{enum_type}"' if enum_type else ""
            config_str = f', configType: "{config_type}"' if config_type else ""

            lines.append(
                f'  {field_name}: {{ type: "{meta_type}", nullable: {str(nullable).lower()}'
                f"{enum_str}{config_str}, defaultValue: {default_json} }},"
            )

        lines.append("};")
        lines.append("")

    return "\n".join(lines)


def _safe_call_method(obj: object, method_name: str) -> bool:
    try:
        return bool(getattr(obj, method_name)())
    except TypeError:
        return False


def generate_model_type_info_ts(enums: list[tuple[str, type]]) -> str:
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TrainingMethod import TrainingMethod

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when backend ModelType or TopBar logic changes.",
        "",
        "import type { ModelType, TrainingMethod } from './enums';",
        "",
    ]

    # inspect.getmembers doesn't find these due to Enum metaclass; use __dict__
    group_methods = []
    for name, obj in ModelType.__dict__.items():
        if name.startswith(("is_", "has_")) and callable(obj):
            group_methods.append(name)
    group_methods.sort()

    lines.append("/** Model type groupings derived from ModelType.is_*() / has_*() methods. */")
    lines.append("export const MODEL_TYPE_GROUPS: Record<string, ModelType[]> = {")
    for method_name in group_methods:
        members_in_group = [
            mt.value for mt in ModelType if _safe_call_method(mt, method_name)
        ]
        if members_in_group:
            members_str = ", ".join(f'"{m}"' for m in members_in_group)
            lines.append(f"  {method_name}: [{members_str}],")
    lines.append("};")
    lines.append("")

    lines.append("/** Reverse lookup: for any ModelType, which groups it belongs to. */")
    lines.append("export const MODEL_TYPE_FLAGS: Record<ModelType, string[]> = {")
    for mt in ModelType:
        flags = [
            method_name for method_name in group_methods
            if _safe_call_method(mt, method_name)
        ]
        flags_str = ", ".join(f'"{f}"' for f in flags)
        lines.append(f'  "{mt.value}": [{flags_str}],')
    lines.append("};")
    lines.append("")

    lines.append("/** Allowed training methods per model type (from TopBar.py). */")
    lines.append("export const TRAINING_METHODS_BY_MODEL: Record<ModelType, TrainingMethod[]> = {")
    for mt in ModelType:
        if mt.is_stable_diffusion():
            methods = [TrainingMethod.FINE_TUNE, TrainingMethod.LORA, TrainingMethod.EMBEDDING, TrainingMethod.FINE_TUNE_VAE]
        elif (mt.is_stable_diffusion_3() or mt.is_stable_diffusion_xl() or mt.is_wuerstchen()
              or mt.is_pixart() or mt.is_flux_1() or mt.is_sana()
              or mt.is_hunyuan_video() or mt.is_hi_dream() or mt.is_chroma()):
            methods = [TrainingMethod.FINE_TUNE, TrainingMethod.LORA, TrainingMethod.EMBEDDING]
        elif mt.is_qwen() or mt.is_z_image() or mt.is_flux_2():
            methods = [TrainingMethod.FINE_TUNE, TrainingMethod.LORA]
        else:
            methods = [TrainingMethod.FINE_TUNE, TrainingMethod.LORA, TrainingMethod.EMBEDDING]
        methods_str = ", ".join(f'"{m.value}"' for m in methods)
        lines.append(f'  "{mt.value}": [{methods_str}],')
    lines.append("};")
    lines.append("")

    return "\n".join(lines)


def _find_layer_presets_attr(cls: type) -> dict | None:
    for ancestor in cls.__mro__:
        if "LAYER_PRESETS" in ancestor.__dict__:
            value = ancestor.__dict__["LAYER_PRESETS"]
            if isinstance(value, dict):
                return value
    return None


def _normalize_layer_preset_value(value: Any) -> dict[str, Any]:
    # BaseZImageSetup uses {'patterns': [...], 'regex': True}; others use list[str].
    if isinstance(value, dict):
        patterns = value.get("patterns", []) or []
        return {"patterns": [str(p) for p in patterns], "regex": bool(value.get("regex", False))}
    if isinstance(value, (list, tuple)):
        return {"patterns": [str(p) for p in value], "regex": False}
    return {"patterns": [str(value)], "regex": False}


def generate_layer_presets_ts() -> str:
    """Emit LAYER_PRESETS_BY_MODEL derived from Base*Setup.LAYER_PRESETS.

    Keyed by ModelType — presets live on the Base*Setup family class and are
    inherited by every LoRA/FineTune/Embedding subclass, so TrainingMethod is
    irrelevant here. Mirrors CTk behavior in modules/ui/TrainingTab.py and
    modules/ui/ModelTab.py which both read `cls.LAYER_PRESETS` from
    `create.get_model_setup_class(model_type, training_method)`.
    """
    global _incomplete_generation

    from modules.util.enum.ModelType import ModelType

    presets_by_model: dict[str, dict[str, dict[str, Any]]] = {}

    try:
        from modules.modelSetup.BaseModelSetup import BaseModelSetup
        from modules.util import create as _create_side_effects  # noqa: F401
        from modules.util import factory

        registry = factory.__dict__.get("__registry", {}).get(BaseModelSetup, [])
        seen: set[ModelType] = set()
        for args, _kwargs, cls in registry:
            if not args:
                continue
            mt = args[0]
            if not isinstance(mt, ModelType) or mt in seen:
                continue
            raw = _find_layer_presets_attr(cls)
            if raw is None:
                continue
            presets_by_model[mt.value] = {
                str(k): _normalize_layer_preset_value(v) for k, v in raw.items()
            }
            seen.add(mt)
    except Exception as e:
        print(f"  WARNING: Could not enumerate LAYER_PRESETS from factory registry ({e})")
        print("  Layer preset map will be empty.")
        print("  Run from the OneTrainer venv for complete output: python -m web.scripts.generate_types")
        _incomplete_generation = True

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when backend Base*Setup.LAYER_PRESETS change.",
        "",
        "import type { ModelType } from './enums';",
        "",
        "export interface LayerPresetDef {",
        "  patterns: string[];",
        "  regex: boolean;",
        "}",
        "",
        "export type LayerPresetMap = Record<string, LayerPresetDef>;",
        "",
        "/** Fallback preset map used when a ModelType has no registered BaseSetup. */",
        "export const EMPTY_PRESETS: LayerPresetMap = {",
        "  full: { patterns: [], regex: false },",
        "};",
        "",
        "/**",
        " * Per-ModelType layer-filter presets sourced from Base*Setup.LAYER_PRESETS.",
        " * Selecting a preset should write both the joined patterns to layer_filter",
        " * and the regex flag to layer_filter_regex, matching the CTk UI behavior.",
        " */",
        "export const LAYER_PRESETS_BY_MODEL: Partial<Record<ModelType, LayerPresetMap>> = {",
    ]

    for mt_value in sorted(presets_by_model.keys()):
        preset_map = presets_by_model[mt_value]
        lines.append(f"  {json.dumps(mt_value)}: {{")
        for preset_name in preset_map:
            defn = preset_map[preset_name]
            patterns_str = ", ".join(json.dumps(p) for p in defn["patterns"])
            regex_str = "true" if defn["regex"] else "false"
            lines.append(
                f"    {json.dumps(preset_name)}: {{ patterns: [{patterns_str}], regex: {regex_str} }},"
            )
        lines.append("  },")

    lines.append("};")
    lines.append("")

    return "\n".join(lines)


def generate_loss_weight_info_ts() -> str:
    """Emit a map of LossWeight -> supports_flow_matching boolean.

    Mirrors CTk behavior in modules/ui/TrainingTab.py, which filters the
    loss_weight_fn dropdown so only entries whose supports_flow_matching()
    matches the current model's is_flow_matching() (plus CONSTANT) are shown.
    """
    from modules.util.enum.LossWeight import LossWeight

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when LossWeight.supports_flow_matching changes.",
        "",
        "import type { LossWeight } from './enums';",
        "",
        "/**",
        " * Maps each LossWeight value to whether it supports flow-matching models.",
        " * Sourced from LossWeight.supports_flow_matching() in",
        " * modules/util/enum/LossWeight.py. Used by the schema renderer to filter",
        " * the loss_weight_fn dropdown per the current model_type.",
        " */",
        "export const LOSS_WEIGHT_SUPPORTS_FLOW_MATCHING: Record<LossWeight, boolean> = {",
    ]

    for member in LossWeight:
        flag = "true" if member.supports_flow_matching() else "false"
        lines.append(f'  "{member.value}": {flag},')

    lines.append("};")
    lines.append("")

    return "\n".join(lines)


def generate_optimizer_info_ts(enums: list[tuple[str, type]]) -> str:
    from modules.util.enum.Optimizer import Optimizer

    # optimizer_util has a circular import; extract defaults from source via AST
    global _incomplete_generation
    try:
        OPTIMIZER_DEFAULT_PARAMETERS = _extract_optimizer_defaults()
    except Exception as e:
        print(f"  WARNING: Could not extract OPTIMIZER_DEFAULT_PARAMETERS ({e})")
        print("  Optimizer defaults will be empty.")
        print("  Run from the OneTrainer venv for complete output: python -m web.scripts.generate_types")
        OPTIMIZER_DEFAULT_PARAMETERS = {}
        _incomplete_generation = True

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when backend optimizer definitions change.",
        "",
        "import type { Optimizer } from './enums';",
        "",
    ]

    adaptive = [opt for opt in Optimizer if opt.is_adaptive]
    schedule_free = [opt for opt in Optimizer if opt.is_schedule_free]
    fused_back_pass = [opt for opt in Optimizer if opt.supports_fused_back_pass()]

    def opt_array(name: str, description: str, opts: list) -> None:
        lines.append(f"/** {description} */")
        lines.append(f"export const {name}: Optimizer[] = [")
        lines.extend(f'  "{opt.value}",' for opt in opts)
        lines.append("];")
        lines.append("")

    opt_array("ADAPTIVE_OPTIMIZERS", "Optimizers with adaptive learning rates.", adaptive)
    opt_array("SCHEDULE_FREE_OPTIMIZERS", "Schedule-free optimizers.", schedule_free)
    opt_array("FUSED_BACK_PASS_OPTIMIZERS", "Optimizers that support fused backward pass.", fused_back_pass)

    lines.append("/** Per-optimizer boolean property flags. */")
    lines.append("export const OPTIMIZER_FLAGS: Record<Optimizer, {")
    lines.append("  isAdaptive: boolean;")
    lines.append("  isScheduleFree: boolean;")
    lines.append("  supportsFusedBackPass: boolean;")
    lines.append("}> = {")
    for opt in Optimizer:
        a = "true" if opt.is_adaptive else "false"
        sf = "true" if opt.is_schedule_free else "false"
        fb = "true" if opt.supports_fused_back_pass() else "false"
        lines.append(f'  "{opt.value}": {{ isAdaptive: {a}, isScheduleFree: {sf}, supportsFusedBackPass: {fb} }},')
    lines.append("};")
    lines.append("")

    def serialize_value(v: Any) -> str:
        if v is None:
            return "null"
        elif isinstance(v, bool):
            return "true" if v else "false"
        elif isinstance(v, (int, float)):
            if v == float("inf"):
                return "Infinity"
            elif v == float("-inf"):
                return "-Infinity"
            return json.dumps(v)
        elif isinstance(v, str):
            return json.dumps(v)
        elif isinstance(v, dict):
            if not v:
                return "{}"
            pairs = ", ".join(f"{json.dumps(str(k))}: {serialize_value(vv)}" for k, vv in v.items())
            return f"{{ {pairs} }}"
        elif isinstance(v, list):
            if not v:
                return "[]"
            items = ", ".join(serialize_value(item) for item in v)
            return f"[{items}]"
        elif isinstance(v, Enum):
            return json.dumps(str(v))
        else:
            return json.dumps(str(v))

    lines.append("/** Default parameter values per optimizer (from optimizer_util.py). */")
    lines.append("// eslint-disable-next-line @typescript-eslint/no-explicit-any")
    lines.append("export const OPTIMIZER_DEFAULTS: Record<Optimizer, Record<string, any>> = {")
    for opt in Optimizer:
        defaults = OPTIMIZER_DEFAULT_PARAMETERS.get(opt, {})
        if not defaults:
            lines.append(f'  "{opt.value}": {{}},')
            continue
        lines.append(f'  "{opt.value}": {{')
        for key, val in defaults.items():
            lines.append(f"    {json.dumps(key)}: {serialize_value(val)},")
        lines.append("  },")
    lines.append("};")
    lines.append("")

    return "\n".join(lines)


def _extract_dict_from_source(source: str, variable_name: str) -> dict:
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == variable_name:
                        value_source = ast.get_source_segment(source, node.value)
                        if value_source:
                            return ast.literal_eval(value_source)
        return {}
    except (SyntaxError, ValueError) as e:
        print(f"  WARNING: Could not parse {variable_name}: {e}")
        return {}


def _extract_optimizer_defaults() -> dict:
    """AST-based extraction to avoid circular import in optimizer_util."""
    from modules.util.enum.Optimizer import Optimizer

    source_path = os.path.join(PROJECT_ROOT, "modules", "util", "optimizer_util.py")
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  WARNING: Could not parse optimizer_util.py: {e}")
        return {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "OPTIMIZER_DEFAULT_PARAMETERS":
                    if not isinstance(node.value, ast.Dict):
                        continue
                    result = {}
                    for key_node, val_node in zip(node.value.keys, node.value.values, strict=False):
                        if isinstance(key_node, ast.Attribute) and isinstance(key_node.value, ast.Name):
                            if key_node.value.id == "Optimizer":
                                try:
                                    opt_member = Optimizer[key_node.attr]
                                except KeyError:
                                    continue
                                val_source = ast.get_source_segment(source, val_node)
                                if val_source:
                                    try:
                                        val = ast.literal_eval(val_source)
                                    except (ValueError, SyntaxError):
                                        val = {}
                                else:
                                    val = {}
                                result[opt_member] = val
                    return result

    print("  WARNING: Could not find OPTIMIZER_DEFAULT_PARAMETERS in optimizer_util.py")
    return {}


def generate_optimizer_defaults_json(enums: list[tuple[str, type]]) -> str:
    from modules.util.enum.Optimizer import Optimizer

    global _incomplete_generation
    try:
        OPTIMIZER_DEFAULT_PARAMETERS = _extract_optimizer_defaults()
    except Exception as e:
        print(f"  WARNING: Could not extract OPTIMIZER_DEFAULT_PARAMETERS ({e})")
        OPTIMIZER_DEFAULT_PARAMETERS = {}
        _incomplete_generation = True

    def clean_value(v: Any) -> Any:
        if v is None:
            return None
        elif isinstance(v, bool):
            return v
        elif isinstance(v, (int, float)):
            if v == float("inf"):
                return "Infinity"
            elif v == float("-inf"):
                return "-Infinity"
            return v
        elif isinstance(v, str):
            return v
        elif isinstance(v, dict):
            return {str(k): clean_value(vv) for k, vv in v.items()}
        elif isinstance(v, list):
            return [clean_value(item) for item in v]
        elif isinstance(v, Enum):
            return str(v)
        else:
            return str(v)

    result: dict[str, dict] = {}
    for opt in Optimizer:
        defaults = OPTIMIZER_DEFAULT_PARAMETERS.get(opt, {})
        result[str(opt)] = {str(k): clean_value(v) for k, v in defaults.items()}

    return json.dumps(result, indent=2)


def _extract_key_detail_map() -> dict:
    """AST-based extraction to avoid importing tkinter-dependent UI module."""
    source_path = os.path.join(PROJECT_ROOT, "modules", "ui", "OptimizerParamsWindow.py")
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    return _extract_dict_from_source(source, "KEY_DETAIL_MAP")


def generate_optimizer_key_details_json() -> str:
    key_detail_map = _extract_key_detail_map()
    return json.dumps(key_detail_map, indent=2)


def generate_optimizer_key_details_ts() -> str:
    key_detail_map = _extract_key_detail_map()

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Source: modules/ui/OptimizerParamsWindow.py KEY_DETAIL_MAP",
        "// Do not edit manually.",
        "",
        "export interface OptimizerKeyDetail {",
        "  title: string;",
        "  tooltip: string;",
        "  type: \"bool\" | \"float\" | \"int\" | \"str\" | \"dict\" | string;",
        "}",
        "",
        "export const OPTIMIZER_KEY_DETAILS: Record<string, OptimizerKeyDetail> = {",
    ]

    for key in sorted(key_detail_map.keys()):
        detail = key_detail_map[key]
        title = json.dumps(detail["title"])
        tooltip = json.dumps(detail["tooltip"])
        detail_type = json.dumps(detail["type"])
        lines.append(f"  {json.dumps(key)}: {{ title: {title}, tooltip: {tooltip}, type: {detail_type} }},")

    lines.append("};")
    lines.append("")
    return "\n".join(lines)


_ACRONYMS = {
    "SDXL", "VAE", "LORA", "GAN", "FP16", "FP32", "BF16", "NF4",
    "CPU", "GPU", "TPU", "EMA", "LR", "GGUF", "BNBFP4", "BNBNF4",
    "RGB", "RGBA", "HDR", "SRT", "JSON", "CSV", "MP3", "MP4", "FLAC",
    "WAV", "OGG", "AVI", "MKV", "WEBM", "GIF", "PNG", "JPG", "JPEG",
    "WEBP", "BMP", "TIFF",
}

_SPECIAL_LABELS = {
    "ADAMW": "AdamW",
    "LORA": "LoRA",
    "ADAFACTOR": "Adafactor",
}

_VERSION_SUFFIXES = {"15": "1.5", "20": "2.0", "21": "2.1", "30": "3.0", "35": "3.5"}


def _auto_label(value: str) -> str:
    if value in _SPECIAL_LABELS:
        return _SPECIAL_LABELS[value]

    parts = value.split("_")
    result = []
    for part in parts:
        if part in _ACRONYMS:
            result.append(part)
        elif part in _VERSION_SUFFIXES:
            result.append(_VERSION_SUFFIXES[part])
        elif part.isdigit():
            result.append(part)
        else:
            result.append(part.capitalize())
    return " ".join(result)


def _auto_tooltip(field_path: str) -> str:
    readable = field_path.replace(".", " ").replace("_", " ")
    return readable[:1].upper() + readable[1:] if readable else ""


def generate_enum_labels_ts(enums: list[tuple[str, type]]) -> str:
    from web.scripts.ui_metadata import ENUM_DISPLAY_LABELS

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Update labels in web/scripts/ui_metadata.py.",
        "",
    ]

    flat_labels: dict[str, str] = {}
    for enum_name, enum_cls in enums:
        overrides = ENUM_DISPLAY_LABELS.get(enum_name, {})
        for member in enum_cls:
            flat_labels[member.value] = overrides.get(member.value, _auto_label(member.value))

    lines.append("const labels: Record<string, string> = {")
    lines.extend(
        f"  {json.dumps(value)}: {json.dumps(flat_labels[value])},"
        for value in sorted(flat_labels.keys())
    )
    lines.append("};")
    lines.append("")

    lines.append("function formatFallback(value: string): string {")
    lines.append("  return value")
    lines.append('    .replace(/_/g, " ")')
    lines.append('    .replace(/\\b([A-Za-z])([A-Za-z]*)\\b/g, (_match, first: string, rest: string) =>')
    lines.append("      first.toUpperCase() + rest.toLowerCase(),")
    lines.append("    );")
    lines.append("}")
    lines.append("")

    lines.append("/**")
    lines.append(" * Returns a human-friendly display label for any enum value string.")
    lines.append(" * Looks up the value in a curated map. Falls back to title-casing.")
    lines.append(" */")
    lines.append("export function enumLabel(value: string): string {")
    lines.append("  return labels[value] ?? formatFallback(value);")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_data_type_subsets_ts() -> str:
    from web.scripts.ui_metadata import DTYPE_SUBSETS

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Update subsets in web/scripts/ui_metadata.py.",
        "",
        "import type { DataType } from './enums';",
        "",
        "export interface DTypeOption {",
        "  label: string;",
        "  value: DataType;",
        "}",
        "",
    ]

    lines.append("export const DTYPE_SUBSETS: Record<string, DTypeOption[]> = {")
    for subset_name, options in DTYPE_SUBSETS.items():
        lines.append(f"  {subset_name}: [")
        for label, value in options:
            lines.append(f"    {{ label: {json.dumps(label)}, value: {json.dumps(value)} }},")
        lines.append("  ],")
    lines.append("};")
    lines.append("")

    return "\n".join(lines)


def generate_tooltips_ts(configs: list[tuple[str, type]]) -> str:
    from web.scripts.ui_metadata import FIELD_TOOLTIPS, WIDE_TOOLTIPS

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Update tooltips in web/scripts/ui_metadata.py.",
        "",
    ]

    all_tooltips: dict[str, str] = {}
    all_tooltips.update(FIELD_TOOLTIPS)

    for _class_name, cls in configs:
        try:
            inst = cls.default_values()
            for field_name in inst.to_dict():
                if field_name.startswith("__"):
                    continue
                if field_name not in all_tooltips:
                    auto = _auto_tooltip(field_name)
                    if auto:
                        all_tooltips[field_name] = auto
        except Exception:
            continue

    lines.append("/** Tooltip text for config fields, keyed by dot-notation field path. */")
    lines.append("export const FIELD_TOOLTIPS: Record<string, string> = {")
    for key in sorted(all_tooltips.keys()):
        tooltip = all_tooltips[key]
        lines.append(f"  {json.dumps(key)}: {json.dumps(tooltip)},")
    lines.append("};")
    lines.append("")

    lines.append("/** Field keys that require wide tooltip display. */")
    lines.append("export const WIDE_TOOLTIP_KEYS: Set<string> = new Set([")
    lines.extend(f"  {json.dumps(key)}," for key in sorted(WIDE_TOOLTIPS))
    lines.append("]);")
    lines.append("")

    lines.append("/** Get tooltip text for a config field key. Returns undefined if not found. */")
    lines.append("export function getTooltip(fieldKey: string): string | undefined {")
    lines.append("  return FIELD_TOOLTIPS[fieldKey];")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def write_file(filename: str, content: str) -> str:
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return filepath


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Collecting enums...")
    enums = collect_enums()
    print(f"  Found {len(enums)} enum types")

    print("Collecting config classes...")
    configs = collect_configs()
    print(f"  Found {len(configs)} config classes")

    enum_names = {name for name, _ in enums}

    print("Generating enums.ts...")
    print(f"  Wrote {write_file('enums.ts', generate_enums_ts(enums))}")

    print("Generating config.ts...")
    print(f"  Wrote {write_file('config.ts', generate_config_ts(configs, enum_names))}")

    print("Generating metadata.ts...")
    print(f"  Wrote {write_file('metadata.ts', generate_metadata_ts(configs, enum_names))}")

    new_generators = [
        ("modelTypeInfo.ts", lambda: generate_model_type_info_ts(enums)),
        ("optimizerInfo.ts", lambda: generate_optimizer_info_ts(enums)),
        ("optimizerKeyDetails.ts", lambda: generate_optimizer_key_details_ts()),
        ("enumLabels.ts", lambda: generate_enum_labels_ts(enums)),
        ("dataTypeSubsets.ts", lambda: generate_data_type_subsets_ts()),
        ("tooltips.ts", lambda: generate_tooltips_ts(configs)),
        ("layerPresets.ts", lambda: generate_layer_presets_ts()),
        ("lossWeightInfo.ts", lambda: generate_loss_weight_info_ts()),
    ]

    global _incomplete_generation
    for filename, generator in new_generators:
        print(f"Generating {filename}...")
        try:
            print(f"  Wrote {write_file(filename, generator())}")
        except Exception as e:
            print(f"  ERROR generating {filename}: {e}")
            import traceback
            traceback.print_exc()
            _incomplete_generation = True

    backend_generated_dir = os.path.join(PROJECT_ROOT, "web", "backend", "generated")
    os.makedirs(backend_generated_dir, exist_ok=True)

    print("Generating optimizer_defaults.json (backend)...")
    try:
        json_content = generate_optimizer_defaults_json(enums)
        json_path = os.path.join(backend_generated_dir, "optimizer_defaults.json")
        with open(json_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json_content)
        print(f"  Wrote {json_path}")
    except Exception as e:
        print(f"  ERROR generating optimizer_defaults.json: {e}")
        import traceback
        traceback.print_exc()
        _incomplete_generation = True

    print("Generating optimizer_key_details.json (backend)...")
    try:
        json_content = generate_optimizer_key_details_json()
        json_path = os.path.join(backend_generated_dir, "optimizer_key_details.json")
        with open(json_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json_content)
        print(f"  Wrote {json_path}")
    except Exception as e:
        print(f"  ERROR generating optimizer_key_details.json: {e}")
        import traceback
        traceback.print_exc()
        _incomplete_generation = True

    total_enum_values = sum(len(list(cls)) for _, cls in enums)
    total_config_fields = 0
    for _class_name, cls in configs:
        instance = cls.default_values()
        total_config_fields += len(instance.types)

    print("\nSummary:")
    print(f"  {len(enums)} enum types with {total_enum_values} total values")
    print(f"  {len(configs)} config interfaces with {total_config_fields} total fields")
    print("  11 generated TypeScript files")

    if _incomplete_generation:
        print("\nWARNING: Some files were not generated successfully.")
        print("Check the errors above and fix the underlying issues.")
        print("Re-run from the OneTrainer venv: python -m web.scripts.generate_types")
        sys.exit(1)
    else:
        print("Done!")


if __name__ == "__main__":
    main()
