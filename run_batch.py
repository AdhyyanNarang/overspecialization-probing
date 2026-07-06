from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

from experiment_io import expand_sweeps, load_experiment_config, timestamp_now

# Force a non-interactive backend for batch jobs to avoid blocking on plt.show().
os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure local imports resolve when running from repo root.
SCRIPT_DIR = Path(__file__).resolve().parent

# Configure batch runs here instead of via CLI.
# Use paths relative to msgd/ or absolute paths.
CONFIG_PATHS = [
    Path("configs/census_bad.yaml"),
    Path("configs/amazon_bad.yaml"),
    Path("configs/movielens_bad.yaml"),
]
RESULTS_ROOT = SCRIPT_DIR / "results"
SAVE_MODE = "slim"  # 'slim' removes large trajectory tensors from saved payloads

import sys

sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR / "utils"))

from utils_clustering import create_rankings_from_clusters  # noqa: E402


def main() -> None:
    os.chdir(SCRIPT_DIR)
    config_paths = [Path(p) for p in CONFIG_PATHS]

    if not config_paths:
        raise SystemExit("CONFIG_PATHS is empty. Add YAML paths to run.")

    results_root = RESULTS_ROOT.resolve()

    for config_path in config_paths:
        run_config_file(config_path, results_root)


def run_config_file(config_path: Path, results_root: Path) -> None:
    dataset, config_dict, runner, _ = load_experiment_config(config_path)
    config_cls = _get_config_class(dataset)

    expanded = expand_sweeps(config_dict, config_cls)
    output_dir = _select_or_create_output_dir(results_root, dataset, config_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = output_dir.name

    config_copy_path = output_dir / "config.yaml"
    if not config_copy_path.exists():
        config_copy_path.write_text(config_path.read_text())

    runs = []
    partial_paths = []
    for run_idx, (resolved_config, sweep) in enumerate(expanded):
        cfg = config_cls(**resolved_config)
        partial_path = output_dir / f"results.partial.run{run_idx}.pkl"
        partial_paths.append(partial_path)
        results = _run_dataset(dataset, cfg, runner, checkpoint_path=partial_path)
        results = _apply_save_mode(results, SAVE_MODE)
        runs.append({"sweep": sweep, "results": results})

    payload = {
        "dataset": dataset,
        "config_file": config_path.name,
        "timestamp": timestamp,
        "base_config": config_dict,
        "runner": runner,
        "runs": runs,
    }

    results_path = output_dir / "results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(payload, f)

    for p in partial_paths:
        if p.exists():
            p.unlink()

    print(f"Saved results to {results_path}")


def _run_dataset(
    dataset: str,
    config: Any,
    runner: Dict[str, Any],
    checkpoint_path: Path | None = None,
) -> Dict[str, Any]:
    if dataset == "census":
        return _run_census(config, runner)
    if dataset == "census_nn":
        return _run_census_nn(config, runner, checkpoint_path=checkpoint_path)
    if dataset == "amazon":
        return _run_amazon(config, runner, checkpoint_path=checkpoint_path)
    if dataset == "movielens":
        return _run_movielens(config)
    raise ValueError(f"Unknown dataset '{dataset}'.")


def _select_or_create_output_dir(results_root: Path, dataset: str, config_stem: str) -> Path:
    config_root = results_root / dataset / config_stem
    config_root.mkdir(parents=True, exist_ok=True)

    existing = [p for p in config_root.iterdir() if p.is_dir()]
    unfinished = []
    for d in existing:
        has_final = (d / "results.pkl").exists()
        has_partial = any(d.glob("results.partial.run*.pkl"))
        if (not has_final) and has_partial:
            unfinished.append(d)

    if unfinished:
        resume_dir = sorted(unfinished)[-1]
        print(f"Resuming unfinished run directory: {resume_dir}")
        return resume_dir

    timestamp = timestamp_now()
    return config_root / timestamp


def _run_census(config: Any, runner: Dict[str, Any]) -> Dict[str, Any]:
    from census_final_plots import (  # noqa: E402
        load_and_preprocess_data,
        perform_clustering as census_perform_clustering,
        run_msgd_experiments as run_census_msgd,
    )
    from utils_msgd_census import initialize_theta_gd  # noqa: E402

    X_train, X_test, y_train, y_test, X_original_train, _ = load_and_preprocess_data()
    cluster_labels, kmeans = census_perform_clustering(X_train, X_original_train, config)
    rankings = create_rankings_from_clusters(
        X_train, kmeans.cluster_centers_, config.n, cluster_labels
    )

    n_features = X_train.shape[1]
    cache_name = (
        f"bar_theta_gd_mode={config.clustering_mode}_n{config.n}_"
        f"reg{config.reg_lambda}_T{config.T}_eta{config.eta}.npy"
    )
    cache_path = SCRIPT_DIR / "cache" / cache_name
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not config.force_recompute_bar_theta:
        bar_Theta = np.load(cache_path)
    else:
        bar_Theta = initialize_theta_gd(
            X_train,
            y_train,
            rankings,
            config.n,
            n_features,
            T=config.T,
            eta=config.eta,
            reg_lambda=config.reg_lambda,
            return_diagnostics=False,
        )
        np.save(cache_path, bar_Theta)

    num_seeds = int(runner.get("num_seeds", 15))
    results_dict = run_census_msgd(
        X_train, y_train, X_test, y_test, rankings, bar_Theta, config, num_seeds=num_seeds
    )
    return results_dict


def _run_census_nn(
    config: Any,
    runner: Dict[str, Any],
    checkpoint_path: Path | None = None,
) -> Dict[str, Any]:
    from census_final_plots import (  # noqa: E402
        load_and_preprocess_data,
        perform_clustering as census_perform_clustering,
        train_baseline_lr,
    )
    from census_nn_final_plots import (  # noqa: E402
        maybe_subsample_census_data,
        run_msgd_experiments as run_census_nn_msgd,
        train_baseline_mlp,
    )
    from utils_msgd_census_nn import pretrain_partition_models  # noqa: E402

    X_train, X_test, y_train, y_test, X_original_train, X_original_test = load_and_preprocess_data()
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_original_train,
        X_original_test,
    ) = maybe_subsample_census_data(
        X_train,
        X_test,
        y_train,
        y_test,
        X_original_train,
        X_original_test,
        config,
    )

    full_lr_test_acc = train_baseline_lr(X_train, y_train, X_test, y_test, config.reg_lambda)
    full_mlp_test_acc = None
    if config.run_pooled_mlp_baseline:
        full_mlp_test_acc = train_baseline_mlp(X_train, y_train, X_test, y_test, config)

    cluster_labels, kmeans = census_perform_clustering(X_train, X_original_train, config)
    rankings = create_rankings_from_clusters(
        X_train,
        kmeans.cluster_centers_,
        config.n,
        cluster_labels,
    )

    init_models = None
    if config.init_method == "partition_pretrain":
        init_models = pretrain_partition_models(X_train, y_train, rankings, config)

    num_seeds = int(runner.get("num_seeds", 3))

    initial_results = None
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_payload = pickle.load(f)
            if isinstance(checkpoint_payload, dict) and "results" in checkpoint_payload:
                initial_results = checkpoint_payload["results"]
            print(
                f"Resuming Census NN from checkpoint {checkpoint_path} "
                f"({0 if initial_results is None else len(initial_results)} runs loaded)"
            )
        except Exception as exc:
            print(f"Warning: failed to load checkpoint {checkpoint_path}: {exc}")
            initial_results = None

    def checkpoint_fn(results_dict: Dict[str, Any], meta: Dict[str, Any]) -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": "census_nn",
            "results": results_dict,
            "meta": meta,
        }
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(checkpoint_path.parent), delete=False
        ) as tmpf:
            pickle.dump(payload, tmpf, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_name = tmpf.name
        os.replace(tmp_name, checkpoint_path)

    results_dict = run_census_nn_msgd(
        X_train,
        y_train,
        X_test,
        y_test,
        rankings,
        init_models,
        config,
        num_seeds=num_seeds,
        initial_results=initial_results,
        checkpoint_every=1,
        checkpoint_fn=checkpoint_fn,
        baseline_mlp_acc=full_mlp_test_acc,
        baseline_lr_acc=full_lr_test_acc,
    )
    return results_dict


def _run_amazon(
    config: Any,
    runner: Dict[str, Any],
    checkpoint_path: Path | None = None,
) -> Dict[str, Any]:
    from amazon_final_plots import (  # noqa: E402
        add_intercept_feature,
        load_and_preprocess_amazon_data,
        perform_clustering as amazon_perform_clustering,
        run_msgd_experiments as run_amazon_msgd,
    )
    from utils_msgd_census import initialize_theta_gd  # noqa: E402

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_original_train,
        _,
        categories_train,
        _,
    ) = load_and_preprocess_amazon_data(config)

    X_train_base, X_test_base = X_train, X_test
    if config.add_intercept_feature:
        X_train = add_intercept_feature(X_train_base)
        X_test = add_intercept_feature(X_test_base)

    cluster_labels, kmeans, rankings = amazon_perform_clustering(
        X_train, X_original_train, config, categories_train=categories_train
    )

    n_features = X_train.shape[1]
    cache_name = (
        f"bar_theta_gd_mode={config.clustering_mode}_n{config.n}_"
        f"d{n_features}_reg{config.reg_lambda}_T{config.T}_eta{config.eta}.npy"
    )
    cache_path = Path(config.models_cache_dir) / cache_name
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not config.force_recompute_bar_theta:
        bar_Theta = np.load(cache_path)
    else:
        bar_Theta = initialize_theta_gd(
            X_train,
            y_train,
            rankings,
            config.n,
            n_features,
            T=config.T,
            eta=config.eta,
            reg_lambda=config.reg_lambda,
            return_diagnostics=False,
        )
        np.save(cache_path, bar_Theta)

    num_seeds = int(runner.get("num_seeds", 1))

    initial_results = None
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_payload = pickle.load(f)
            if isinstance(checkpoint_payload, dict) and "results" in checkpoint_payload:
                initial_results = checkpoint_payload["results"]
            print(
                f"Resuming Amazon from checkpoint {checkpoint_path} "
                f"({0 if initial_results is None else len(initial_results)} runs loaded)"
            )
        except Exception as exc:
            print(f"Warning: failed to load checkpoint {checkpoint_path}: {exc}")
            initial_results = None

    def checkpoint_fn(results_dict: Dict[str, Any], meta: Dict[str, Any]) -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": "amazon",
            "results": results_dict,
            "meta": meta,
        }
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(checkpoint_path.parent), delete=False
        ) as tmpf:
            pickle.dump(payload, tmpf, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_name = tmpf.name
        os.replace(tmp_name, checkpoint_path)

    results_dict = run_amazon_msgd(
        X_train,
        y_train,
        X_test,
        y_test,
        rankings,
        bar_Theta,
        config,
        num_seeds=num_seeds,
        store_trajectories=(SAVE_MODE == "full"),
        initial_results=initial_results,
        checkpoint_every=1,
        checkpoint_fn=checkpoint_fn,
    )
    return results_dict


def _run_movielens(config: Any) -> Dict[str, Any]:
    from movielens_final_plots import (  # noqa: E402
        initialize_theta_from_erm,
        load_movielens_data,
        perform_clustering as movielens_perform_clustering,
        run_msgd_experiments as run_movielens_msgd,
    )

    data = load_movielens_data(config)
    X_train = data["X_train"]
    y_train = data["y_train"]
    mask_train = data["mask_train"]
    num_movies = data["num_movies"]
    num_emb = data["num_emb"]

    cluster_labels, kmeans, rankings = movielens_perform_clustering(X_train, config)

    if config.init_method == "erm":
        Theta_init_erm = initialize_theta_from_erm(
            X_train,
            y_train,
            mask_train,
            rankings,
            config.n,
            num_movies,
            num_emb,
            reg_lambda=config.reg_lambda,
        )
    else:
        Theta_init_erm = np.random.rand(config.n, num_movies, num_emb)

    results = run_movielens_msgd(data, config, cluster_labels, kmeans, rankings, Theta_init_erm)
    return results


def _get_config_class(dataset: str):
    if dataset == "census":
        from census_final_plots import CensusConfig  # noqa: E402

        return CensusConfig
    if dataset == "census_nn":
        from census_nn_final_plots import CensusNNConfig  # noqa: E402

        return CensusNNConfig
    if dataset == "amazon":
        from amazon_final_plots import AmazonConfig  # noqa: E402

        return AmazonConfig
    if dataset == "movielens":
        from movielens_final_plots import MovieLensConfig  # noqa: E402

        return MovieLensConfig
    raise ValueError(f"Unknown dataset '{dataset}'.")


def _apply_save_mode(results: Dict[str, Any], save_mode: str) -> Dict[str, Any]:
    if save_mode == "full":
        return results
    if save_mode != "slim":
        raise ValueError(f"Unknown SAVE_MODE '{save_mode}'. Use 'slim' or 'full'.")

    for _, run_result in results.items():
        if isinstance(run_result, dict):
            run_result.pop("Theta", None)
            run_result.pop("Theta_full", None)
    return results


if __name__ == "__main__":
    main()
