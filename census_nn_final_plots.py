"""
Census MSGD experiments with 2-layer neural learners.

This module mirrors the existing Census pipeline but replaces each linear learner
with a small 2-layer ReLU MLP while keeping the same train/test split,
clustering, rankings, probing semantics, and plotting interface.
"""

import random
import sys
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

sys.path.append("./utils")

from census_final_plots import (  # noqa: E402
    load_and_preprocess_data,
    perform_clustering,
    train_baseline_lr,
)
from utils_clustering import create_rankings_from_clusters  # noqa: E402
from utils_msgd_census_nn import (  # noqa: E402
    clone_models,
    make_random_models,
    pretrain_partition_models,
    resolve_device,
    run_msgd_census_nn_with_probing,
    train_pooled_mlp_baseline,
)
from utils_plotting import (  # noqa: E402
    plot_assignment_fraction,
    plot_final_accuracy_vs_p,
    plot_individual_model_accuracies,
    plot_user_assignment_map,
)


@dataclass
class CensusNNConfig:
    # SGD parameters
    eta: float = 0.01
    n: int = 5
    T: int = 1000
    tau: float = 0.3
    reg_lambda: float = 1e-9
    num_sample: int = 150

    # Probing parameters
    N_probe: int = 100
    probe_num_samples: int = 150
    probing_set: list = None

    # Experiment parameters
    p_values: list = None
    kappa_values: list = None
    clustering_method: str = "kmeans"
    clustering_mode: str = "no_majority"
    feature_name: str = "AGEP"
    feature_index: int = 0
    feature_binning: str = "threshold"
    lr_schedule: str = "constant"
    max_plot_iterations: int = 1000

    # MLP parameters
    hidden_dim: int = 64
    activation: str = "relu"
    init_method: str = "partition_pretrain"
    device: str = "cpu"
    eval_every: int = 100
    run_pooled_mlp_baseline: bool = True
    baseline_epochs: int = 5
    baseline_batch_size: int = 512
    baseline_lr: float = 0.01
    pretrain_epochs: int = 5
    pretrain_batch_size: int = 512
    pretrain_lr: float = 0.01
    force_recompute_bar_models: bool = False
    cache_dir: str = "cache/census_nn"
    models_cache_dir: str = "cache/census_nn/models"
    max_train_users: int | None = None
    max_test_users: int | None = None

    # Plotting / compatibility options
    plot_seeds_separately: bool = False
    plot_distance_separately: bool = False
    rankings_only: bool = False
    visualize_clusters: bool = False

    def __post_init__(self) -> None:
        if self.probing_set is None:
            self.probing_set = [2]
        if self.p_values is None:
            self.p_values = [0, 0.1, 0.2, 0.4, 0.6, 0.8]
        if self.kappa_values is None:
            self.kappa_values = [0.0]
        if self.clustering_mode != "no_majority":
            raise ValueError("census_nn currently supports only clustering_mode='no_majority'.")
        if self.init_method not in {"partition_pretrain", "random"}:
            raise ValueError(
                f"Unknown init_method '{self.init_method}'. Expected 'partition_pretrain' or 'random'."
            )
        if self.activation != "relu":
            raise ValueError("census_nn currently supports only activation='relu'.")


def _subsample_arrays(
    X: np.ndarray,
    y: np.ndarray,
    limit: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if limit is None or limit >= len(X):
        return X, y, None
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X), size=limit, replace=False)
    indices.sort()
    return X[indices], y[indices], indices


def maybe_subsample_census_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_original_train: np.ndarray,
    X_original_test: np.ndarray,
    config: CensusNNConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_sub, y_train_sub, train_indices = _subsample_arrays(
        X_train,
        y_train,
        config.max_train_users,
        seed=0,
    )
    if train_indices is not None:
        X_original_train = X_original_train[train_indices]
    else:
        X_original_train = X_original_train.copy()

    X_test_sub, y_test_sub, test_indices = _subsample_arrays(
        X_test,
        y_test,
        config.max_test_users,
        seed=1,
    )
    if test_indices is not None:
        X_original_test = X_original_test[test_indices]
    else:
        X_original_test = X_original_test.copy()

    return (
        X_train_sub,
        X_test_sub,
        y_train_sub,
        y_test_sub,
        X_original_train,
        X_original_test,
    )


def train_baseline_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: CensusNNConfig,
) -> float:
    print("\nTraining pooled 2-layer MLP baseline on full Census train split...")
    _, test_acc = train_pooled_mlp_baseline(X_train, y_train, X_test, y_test, config)
    print(f"Full dataset MLP - Test accuracy: {test_acc:.4f}")
    return test_acc


def run_msgd_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rankings: np.ndarray,
    init_models,
    config: CensusNNConfig,
    num_seeds: int = 3,
    initial_results: Dict[Any, Dict[str, Any]] | None = None,
    checkpoint_every: int = 0,
    checkpoint_fn=None,
    baseline_mlp_acc: float | None = None,
    baseline_lr_acc: float | None = None,
) -> Dict[Any, Dict[str, Any]]:
    results_dict = {} if initial_results is None else dict(initial_results)
    input_dim = X_train.shape[1]
    device = resolve_device(config.device)
    new_runs = 0

    for seed in range(num_seeds):
        print(f"\n{'=' * 80}")
        print(f"Processing seed {seed}")
        print(f"{'=' * 80}")

        for p in config.p_values:
            kappa_list = [0.0] if p == 0 else config.kappa_values
            for kappa in kappa_list:
                run_key = (seed, p, kappa)
                if run_key in results_dict:
                    print(f"  Skipping existing run seed={seed}, p={p}, kappa={kappa}")
                    continue

                print(f"  Running with probing_p={p}, ranking_noise={kappa}")
                random.seed(seed)
                np.random.seed(seed)

                if config.init_method == "partition_pretrain":
                    if init_models is None:
                        raise ValueError("partition_pretrain requires precomputed init_models.")
                    run_init_models = clone_models(init_models)
                    print("    Using partition-pretrained initialization")
                else:
                    run_init_models = make_random_models(
                        config.n,
                        input_dim=input_dim,
                        hidden_dim=config.hidden_dim,
                        activation=config.activation,
                        device=device,
                        seed=seed,
                    )
                    print("    Using random initialization")

                result = run_msgd_census_nn_with_probing(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    rankings,
                    run_init_models,
                    config,
                    seed=seed,
                    probing_p=p,
                    ranking_noise=kappa,
                )

                result.update(
                    {
                        "init_seed": seed,
                        "probing_p": p,
                        "ranking_noise": kappa,
                        "n": config.n,
                        "tau": config.tau,
                        "probing_set": config.probing_set if p > 0 else [],
                        "baseline_mlp_acc": baseline_mlp_acc,
                        "baseline_lr_acc": baseline_lr_acc,
                    }
                )
                results_dict[run_key] = result
                new_runs += 1

                print(f"    Final accuracies: {result['model_acc'][-1]}")

                if checkpoint_fn is not None and checkpoint_every > 0 and new_runs % checkpoint_every == 0:
                    checkpoint_fn(
                        results_dict,
                        {"seed": seed, "p": p, "kappa": kappa, "new_runs": new_runs},
                    )

    if checkpoint_fn is not None and new_runs > 0:
        checkpoint_fn(results_dict, {"final": True, "new_runs": new_runs})

    print("\nTraining complete!")
    return results_dict


def print_final_accuracies(
    results_dict: Dict[Any, Dict[str, Any]],
    full_mlp_test_acc: float | None,
    full_lr_test_acc: float | None = None,
) -> None:
    print("\n" + "=" * 80)
    print("FINAL MODEL ACCURACIES")
    print("=" * 80)

    for (seed, p, kappa), result in sorted(results_dict.items()):
        final_acc = result["model_acc"][-1]
        final_acc_full = result["model_acc_full"][-1]
        final_ensemble = result["ensemble_acc"][-1]

        print(f"\nSeed {seed}, p={p}, κ={kappa}:")
        print(
            f"  Update_all: avg={final_acc_full.mean():.4f}, "
            f"min={final_acc_full.min():.4f}, max={final_acc_full.max():.4f}"
        )
        print(
            f"  MSGD:       avg={final_acc.mean():.4f}, "
            f"min={final_acc.min():.4f}, max={final_acc.max():.4f}"
        )
        print(f"  Ensemble:   {final_ensemble:.4f}")
        if full_mlp_test_acc is not None:
            print(f"  Reference (Full MLP): {full_mlp_test_acc:.4f}")
        if full_lr_test_acc is not None:
            print(f"  Reference (Full LR):  {full_lr_test_acc:.4f}")
        if full_mlp_test_acc is not None:
            print(f"  Update_all vs Full MLP: {final_acc_full.mean() - full_mlp_test_acc:+.4f}")

    print("\n" + "=" * 80)


def generate_plots(
    results_dict: Dict[Any, Dict[str, Any]],
    X_train: np.ndarray,
    config: CensusNNConfig,
    baseline_acc: float | None = None,
    prefix: str | None = None,
) -> None:
    base_name = prefix or (
        "census_nn_good_outcome"
        if config.init_method == "partition_pretrain" and any(p > 0 for p in config.p_values)
        else "census_nn_bad_outcome"
    )
    plot_individual_model_accuracies(
        results_dict,
        config,
        only_p0=False,
        show_title=False,
        save_file_name=base_name,
        baseline_acc=baseline_acc,
    )
    if any(p > 0 for p in config.p_values):
        plot_final_accuracy_vs_p(
            results_dict,
            config,
            title=None,
            save_file_name=f"{base_name}_final_accuracy_vs_p",
            baseline_acc=baseline_acc,
        )
    plot_assignment_fraction(
        results_dict,
        config,
        title=f"Learner assignment fraction ({base_name})",
        max_iter=config.max_plot_iterations,
    )
    plot_user_assignment_map(
        results_dict,
        X_train,
        p_value=0.0,
        kappa=0.0,
        seed=0,
        title=f"User assignment map ({base_name})",
    )


def main():
    config = CensusNNConfig()

    print("=" * 80)
    print("CENSUS NN MSGD EXPERIMENTS")
    print("=" * 80)
    print("Configuration:")
    print(f"  n={config.n}, T={config.T}, eta={config.eta}, tau={config.tau}")
    print(f"  Clustering: {config.clustering_method} (mode={config.clustering_mode})")
    print(f"  Initialization: {config.init_method}")
    print(f"  Hidden dim: {config.hidden_dim}, device={config.device}")
    print(f"  Probing set: {config.probing_set}")
    print("=" * 80)

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
    full_mlp_test_acc = (
        train_baseline_mlp(X_train, y_train, X_test, y_test, config)
        if config.run_pooled_mlp_baseline
        else None
    )

    cluster_labels, kmeans = perform_clustering(X_train, X_original_train, config)

    print("\nCreating rankings from clusters...")
    rankings = create_rankings_from_clusters(
        X_train,
        kmeans.cluster_centers_,
        config.n,
        cluster_labels,
    )

    init_models = None
    if config.init_method == "partition_pretrain":
        init_models = pretrain_partition_models(X_train, y_train, rankings, config)

    results_dict = run_msgd_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        rankings,
        init_models,
        config,
        num_seeds=3,
        baseline_mlp_acc=full_mlp_test_acc,
        baseline_lr_acc=full_lr_test_acc,
    )

    print_final_accuracies(results_dict, full_mlp_test_acc, full_lr_test_acc)
    generate_plots(results_dict, X_train, config, baseline_acc=full_mlp_test_acc)

    return results_dict, config, full_mlp_test_acc, full_lr_test_acc


if __name__ == "__main__":
    main()
