from __future__ import annotations

from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Type, get_origin, get_type_hints

import yaml


def load_experiment_config(path: Path) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load a YAML config and return dataset, config dict, runner dict, and raw data."""
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level in {path}, got {type(data)}")

    dataset = data.get("dataset")
    if not dataset:
        raise ValueError(f"Missing 'dataset' in {path}")

    if "config" in data:
        config = data["config"] or {}
    else:
        config = {k: v for k, v in data.items() if k not in {"dataset", "runner"}}
    if not isinstance(config, dict):
        raise ValueError(f"Expected 'config' to be a mapping in {path}")

    runner = data.get("runner", {}) or {}
    if not isinstance(runner, dict):
        raise ValueError(f"Expected 'runner' to be a mapping in {path}")

    return str(dataset), config, runner, data


def expand_sweeps(config: Dict[str, Any], config_cls: Type) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Expand list-valued sweep fields into a cartesian product.

    Returns a list of (resolved_config, sweep_values) pairs.
    """
    try:
        field_types = get_type_hints(config_cls)
    except Exception:
        field_types = {f.name: f.type for f in fields(config_cls)}
    sweep_keys = []
    base_config = {}

    for key, value in config.items():
        is_list_field = _is_list_type(field_types.get(key))
        if isinstance(value, list) and not is_list_field:
            sweep_keys.append(key)
        else:
            base_config[key] = value

    if not sweep_keys:
        return [(config, {})]

    sweep_values = [config[key] for key in sweep_keys]
    expanded = []
    for combo in _cartesian_product(sweep_values):
        resolved = dict(base_config)
        sweep = {}
        for key, value in zip(sweep_keys, combo):
            resolved[key] = value
            sweep[key] = value
        expanded.append((resolved, sweep))

    return expanded


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_latest_results_dir(
    config_path: Path,
    results_root: Path = Path("results"),
) -> Optional[Path]:
    dataset, _, _, _ = load_experiment_config(config_path)
    config_name = config_path.stem
    base_dir = results_root / dataset / config_name
    if not base_dir.exists():
        return None
    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_latest_results(
    config_path: Path,
    results_root: Path = Path("results"),
) -> Tuple[Dict[str, Any], Path]:
    latest_dir = find_latest_results_dir(config_path, results_root=results_root)
    if latest_dir is None:
        raise FileNotFoundError(f"No results found for {config_path}")
    results_path = latest_dir / "results.pkl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.pkl in {latest_dir}")
    import pickle

    with open(results_path, "rb") as f:
        payload = pickle.load(f)
    return payload, latest_dir


def select_results(payload: Dict[str, Any], sweep_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    runs = payload.get("runs")
    if not runs:
        return payload
    if sweep_filter is None:
        if len(runs) == 1:
            return runs[0]["results"]
        raise ValueError("Multiple sweep runs found; provide sweep_filter to select one.")
    for run in runs:
        if run.get("sweep") == sweep_filter:
            return run.get("results")
    raise ValueError(f"No run found for sweep_filter={sweep_filter}")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _is_list_type(field_type: Any) -> bool:
    if field_type is list:
        return True
    if isinstance(field_type, str):
        return field_type == "list" or field_type.startswith("list[")
    origin = get_origin(field_type)
    return origin is list


def _cartesian_product(values: Iterable[Iterable[Any]]) -> List[Tuple[Any, ...]]:
    product = [()]
    for pool in values:
        product = [x + (y,) for x in product for y in pool]
    return product
