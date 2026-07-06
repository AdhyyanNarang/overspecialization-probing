"""
Smoke test for the Census NN MSGD pipeline.
"""

import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.append("./utils")

from census_nn_final_plots import (  # noqa: E402
    CensusNNConfig,
    maybe_subsample_census_data,
    run_msgd_experiments,
    train_baseline_mlp,
)
from census_final_plots import (  # noqa: E402
    load_and_preprocess_data,
    perform_clustering,
    train_baseline_lr,
)
from utils_clustering import create_rankings_from_clusters  # noqa: E402
from utils_msgd_census_nn import (  # noqa: E402
    binary_loss_matrix,
    generate_probe_batch_labels,
    make_random_models,
    pretrain_partition_models,
    resolve_device,
    set_all_seeds,
    stack_logits,
)
from utils_plotting import plot_assignment_fraction, plot_individual_model_accuracies  # noqa: E402


def _select_models_for_test(ranking_batch, loss_matrix, tau, rankings_only=False):
    rank_choices = np.argmin(ranking_batch, axis=1)
    if rankings_only:
        return rank_choices
    loss_choices = np.argmin(loss_matrix, axis=0)
    if tau <= 0:
        return loss_choices
    if tau >= 1:
        return rank_choices
    raise ValueError("Synthetic selection helper supports only tau in {0, 1}.")


def run_synthetic_backend_checks() -> None:
    print("\n[0/6] Running synthetic backend checks...")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the census_nn smoke test.") from exc

    device = resolve_device("cpu")
    set_all_seeds(123)
    models = make_random_models(3, input_dim=4, hidden_dim=8, activation="relu", device=device, seed=123)
    x_batch = torch.randn(5, 4, device=device)
    y_batch = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float32, device=device)

    logits = stack_logits(models, x_batch)
    losses = binary_loss_matrix(logits, y_batch)
    assert logits.shape == (3, 5), f"Unexpected logits shape: {logits.shape}"
    assert losses.shape == (3, 5), f"Unexpected loss matrix shape: {losses.shape}"

    ranking_batch = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
    loss_matrix = np.array([[0.3, 0.4, 0.1], [0.1, 0.5, 0.2], [0.6, 0.1, 0.3]])
    assert np.array_equal(_select_models_for_test(ranking_batch, loss_matrix, tau=1), np.array([0, 1, 2]))
    assert np.array_equal(_select_models_for_test(ranking_batch, loss_matrix, tau=0), np.array([1, 2, 0]))

    probe_rankings = torch.tensor(ranking_batch[:2], dtype=torch.long, device=device)
    probe_x = torch.randn(2, 4, device=device)
    set_all_seeds(999)
    labels_a = generate_probe_batch_labels(models, probe_x, probe_rankings, "no_majority", ranking_noise=0.5)
    set_all_seeds(999)
    labels_b = generate_probe_batch_labels(models, probe_x, probe_rankings, "no_majority", ranking_noise=0.5)
    assert labels_a.shape == (2,), f"Unexpected probe label shape: {labels_a.shape}"
    assert torch.equal(labels_a, labels_b), "Probe labels should be deterministic under a fixed seed."
    print("✓ Synthetic backend checks passed")


config = CensusNNConfig(
    n=5,
    T=50,
    tau=0.7,
    eta=0.01,
    num_sample=64,
    N_probe=32,
    probe_num_samples=32,
    probing_set=[2],
    p_values=[0],
    kappa_values=[0.0],
    init_method="partition_pretrain",
    hidden_dim=32,
    device="cpu",
    pretrain_epochs=1,
    baseline_epochs=1,
    eval_every=10,
    max_train_users=4096,
    max_test_users=512,
)

print("=" * 80)
print("TESTING CENSUS NN MSGD IMPLEMENTATION")
print("Testing with reduced Census split, 1 seed, p=0")
print("=" * 80)

try:
    run_synthetic_backend_checks()
except Exception as exc:
    print(f"✗ Synthetic backend checks failed: {exc}")
    raise

print("\n[1/6] Loading Census data...")
try:
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
    print("✓ Data loaded successfully")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
except Exception as exc:
    print(f"✗ Data loading failed: {exc}")
    sys.exit(1)

print("\n[2/6] Training pooled baselines...")
try:
    baseline_lr_acc = train_baseline_lr(X_train, y_train, X_test, y_test, config.reg_lambda)
    baseline_mlp_acc = train_baseline_mlp(X_train, y_train, X_test, y_test, config)
    assert np.isfinite(baseline_mlp_acc), "MLP baseline accuracy must be finite."
    print(f"✓ Baselines completed (LR={baseline_lr_acc:.4f}, MLP={baseline_mlp_acc:.4f})")
except Exception as exc:
    print(f"✗ Baseline training failed: {exc}")
    sys.exit(1)

print("\n[3/6] Clustering and rankings...")
try:
    cluster_labels, kmeans = perform_clustering(X_train, X_original_train, config)
    rankings = create_rankings_from_clusters(X_train, kmeans.cluster_centers_, config.n, cluster_labels)
    assert len(np.unique(cluster_labels)) == config.n, "Wrong number of clusters"
    assert rankings.shape == (len(X_train), config.n), "Wrong rankings shape"
    assert np.all((rankings >= 0) & (rankings < config.n)), "Invalid rankings"
    print("✓ Clustering verification passed")
except Exception as exc:
    print(f"✗ Clustering failed: {exc}")
    sys.exit(1)

print("\n[4/6] Pretraining partition models...")
try:
    init_models = pretrain_partition_models(X_train, y_train, rankings, config)
    import torch

    for model in init_models:
        flat = torch.cat([param.detach().reshape(-1) for param in model.parameters()])
        assert torch.isfinite(flat).all(), "Found non-finite parameter in pretrained model."
    print("✓ Partition-pretrained models initialized")
except Exception as exc:
    print(f"✗ Partition pretraining failed: {exc}")
    sys.exit(1)

print("\n[5/6] Running neural MSGD experiment...")
try:
    results_dict = run_msgd_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        rankings,
        init_models,
        config,
        num_seeds=1,
        baseline_mlp_acc=baseline_mlp_acc,
        baseline_lr_acc=baseline_lr_acc,
    )
    result = results_dict[(0, 0, 0.0)]
    assert result["model_acc"].shape[0] == len(result["eval_points"]), "Model accuracy shape mismatch"
    assert result["model_losses"].shape[0] == len(result["eval_points"]), "Model loss shape mismatch"
    assert result["distances_from_erm"].shape == (len(result["eval_points"]), config.n)
    assert np.isfinite(result["model_acc"]).all(), "Found non-finite model accuracy"
    assert np.isfinite(result["model_losses"]).all(), "Found non-finite model loss"
    assert np.allclose(result["assignment_fraction"].sum(axis=1), 1.0, atol=1e-6)
    print("✓ MSGD run completed and basic invariants hold")
except Exception as exc:
    print(f"✗ MSGD experiment failed: {exc}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n[6/6] Checking plotting compatibility...")
try:
    plot_individual_model_accuracies(
        results_dict,
        config,
        show_title=False,
        save_file_name="census_nn_smoke_accuracy",
        baseline_acc=baseline_mlp_acc,
    )
    plot_assignment_fraction(
        results_dict,
        config,
        title="Census NN smoke assignment fraction",
        max_iter=config.max_plot_iterations,
    )
    print("✓ Plotting functions accepted the neural result payload")
except Exception as exc:
    print(f"✗ Plot compatibility failed: {exc}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL CENSUS NN SMOKE TESTS PASSED")
print("=" * 80)
