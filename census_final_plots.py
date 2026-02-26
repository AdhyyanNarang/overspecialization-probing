"""
Census MSGD Experiments - Main Script

This script performs multi-learner strategic gradient descent (MSGD) experiments
on the Census ACS Employment dataset with optional offline probing.

Replaces the functionality of census_final_plots.ipynb.
"""

import numpy as np
import random
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSEmployment
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sys

# Add utils to path
sys.path.append("./utils")

from utils_msgd_census import (
    MSGD_census_with_probing,
    logistic_loss,
    all_models_with_ensemble_population_loss_census,
    initialize_theta_gd,
    initialize_theta_from_erm_sklearn
)
from utils_clustering import (
    cluster_users_majority_minority,
    cluster_users_by_feature,
    create_rankings_from_clusters,
    visualize_individual_clusters_2d
)
from utils_plotting import (
    plot_individual_model_accuracies,
    plot_individual_model_accuracies_with_kappa,
    plot_assignment_fraction,
    plot_user_assignment_map
)


@dataclass
class CensusConfig:
    """Configuration for census MSGD experiments."""
    # SGD parameters
    eta: float = 0.01  # Constant learning rate for SGD
    init_seed: int = 6  # Random seed to control initial Theta_0
    n: int = 5  # The number of models
    T: int = 2000  # Total number of rounds
    tau: float = 0.3  # Ranking weight
    reg_lambda: float = 1e-9  # L2 regularization
    num_sample: int = 150  # Mini-batch size

    # Probing parameters
    N_probe: int = 1000  # Offline probe dataset size
    probe_num_samples: int = 150  # Probe samples per iteration
    probing_set: list = None  # Which learners will use probing (e.g., [2])

    # Experiment parameters
    p_values: list = None  # Values of p to test (e.g., [0, 0.2, 0.5, 1.0])
    kappa_values: list = None  # Ranking noise values to test (e.g., [0.0])

    # Initialization method: 'random' or 'erm'
    init_method: str = 'erm'  # Toggle between 'random' and 'erm' (bar_Theta)

    # Plotting options
    plot_seeds_separately: bool = False  # Toggle for accuracy plots: False=aggregated, True=separate
    plot_distance_separately: bool = False  # Toggle for distance plots: False=aggregated, True=separate

    # Clustering configuration
    clustering_method: str = 'kmeans'  # 'kmeans' or 'feature'
    clustering_mode: str = 'no_majority'  # For kmeans: 'single_majority', 'half_majority', or 'no_majority'

    # Feature-based clustering parameters
    feature_name: str = 'AGEP'  # Feature name for clustering (e.g., 'AGEP', 'SCHL', 'SEX')
    feature_index: int = 0  # Column index of feature (AGEP is first column, index 0)
    feature_binning: str = 'threshold'  # 'quantile' (equal-sized) or 'equal-width'

    # Learning rate schedule
    lr_schedule: str = 'constant'  # 'constant' or 'sqrt'

    # Cache and computation control
    force_recompute_bar_theta: bool = False
    max_plot_iterations: int = 2000
    visualize_clusters: bool = False  # If True, show PCA/t-SNE cluster plots (blocking in interactive backends)

    def __post_init__(self):
        if self.probing_set is None:
            self.probing_set = [2]
        if self.p_values is None:
            self.p_values = [0, 0.2, 0.5, 1.0]
        if self.kappa_values is None:
            self.kappa_values = [0.0]


def load_and_preprocess_data():
    """Load and preprocess the Census ACS Employment dataset."""
    print("Loading Census ACS Employment data...")
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)

    # Save original features before scaling (needed for feature-based clustering)
    X_original = features.copy()

    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        features, label, group, test_size=0.01, random_state=0)

    # Also split original features identically
    X_original_train, X_original_test, _, _, _, _ = train_test_split(
        X_original, label, group, test_size=0.01, random_state=0)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Positive labels in training: {np.sum(y_train)}")

    return X_train, X_test, y_train, y_test, X_original_train, X_original_test


def train_baseline_lr(X_train, y_train, X_test, y_test, reg_lambda):
    """Train baseline logistic regression on full dataset."""
    print("\nTraining baseline logistic regression on full dataset...")
    clf_full = LogisticRegression(
        penalty='l2',
        C=1.0/(2*reg_lambda) if reg_lambda > 0 else 1e10,
        solver='lbfgs',
        max_iter=1000,
        fit_intercept=False,
        random_state=42
    )

    clf_full.fit(X_train, y_train)

    # Evaluate on training set
    train_pred = clf_full.predict(X_train)
    train_acc = (train_pred == y_train).mean()
    print(f"Full dataset LR - Train accuracy: {train_acc:.4f}")

    # Evaluate on test set
    test_pred = clf_full.predict(X_test)
    test_acc = (test_pred == y_test).mean()
    print(f"Full dataset LR - Test accuracy: {test_acc:.4f}")
    print(f"\nThis will serve as the reference accuracy (upper bound): {test_acc:.4f}")

    return test_acc


def perform_clustering(X_train, X_original_train, args):
    """Cluster users based on chosen method."""
    if args.clustering_method == 'kmeans':
        print(f"\nUsing K-means clustering with mode: {args.clustering_mode}")
        cluster_labels, kmeans = cluster_users_majority_minority(
            X_train,
            n_clusters=args.n,
            majority_percentage=0.80,  # Ignored when mode='no_majority'
            mode=args.clustering_mode
        )
    elif args.clustering_method == 'feature':
        print(f"\nUsing feature-based clustering on: {args.feature_name}")
        cluster_labels, kmeans = cluster_users_by_feature(
            X_original_train,  # Original (unscaled) features for interpretable bins
            X_train,           # Scaled features for cluster centers
            args.feature_name,
            args.feature_index,
            args.n,
            method=args.feature_binning
        )
    else:
        raise ValueError(f"Unknown clustering_method: {args.clustering_method}. "
                       f"Must be 'kmeans' or 'feature'")

    # Optional cluster visualization (can block in interactive backends).
    if getattr(args, "visualize_clusters", False):
        visualize_individual_clusters_2d(X_train, cluster_labels, kmeans.cluster_centers_, method='pca')

    return cluster_labels, kmeans


def run_msgd_experiments(X_train, y_train, X_test, y_test, rankings, bar_Theta, args, num_seeds=15):
    """Run MSGD experiments for multiple seeds and p values."""
    n_features = X_train.shape[1]
    p_values = args.p_values  # Values of p to test
    kappa_values = args.kappa_values  # Ranking noise values to test when p > 0
    results_dict = {}  # Dictionary to store results with (seed, p, kappa) as key

    for init_seed in range(num_seeds):
        print(f"\n{'='*80}")
        print(f"Processing seed {init_seed}")
        print(f"{'='*80}")
        random.seed(init_seed)
        np.random.seed(init_seed)

        for p in p_values:
            if p == 0:
                # Baseline: no probing, no ranking noise
                kappa_list = [0.0]  # Only one run for baseline
            else:
                # With probing: test multiple ranking noise values
                kappa_list = kappa_values

            for kappa in kappa_list:
                print(f"  Running with probing_p={p}, ranking_noise={kappa}")

                # Choose initialization method based on config
                if args.init_method == 'random':
                    print(f"    Using random initialization")
                    Theta_init = np.random.randn(args.n, n_features) * 0.01
                elif args.init_method == 'erm':
                    print(f"    Using ERM (bar_Theta) initialization")
                    Theta_init = bar_Theta.copy()
                else:
                    raise ValueError(f"Unknown init_method: {args.init_method}. Must be 'random' or 'erm'")

                # Run MSGD with current p and kappa values
                result = MSGD_census_with_probing(
                    X_train, Theta_init, args.n, y_train, args.T,
                    loss_func=logistic_loss,
                    eta=args.eta,
                    rankings=rankings,
                    tau=args.tau,
                    rankings_only=False,
                    probing_set=args.probing_set if p > 0 else [],  # Only probe if p > 0
                    probing_p=p,
                    reg_lambda=args.reg_lambda,
                    num_sample=args.num_sample,
                    seed=init_seed,
                    N_probe=args.N_probe,
                    probe_num_samples=args.probe_num_samples,
                    mode=args.clustering_mode,
                    ranking_noise=kappa,
                    lr_schedule=args.lr_schedule
                )

                Theta, Theta_full = result["Theta_traj"], result["Update_all_Theta_traj"]
                chosen_models = result["chosen_models"]
                user_assignment_counts = result["user_assignment_counts"]

                # Track fraction of users assigned to each learner per iteration
                T_iters = chosen_models.shape[0]
                assignment_counts = np.zeros((T_iters, args.n), dtype=int)
                assignment_fraction = np.zeros((T_iters, args.n))
                for learner_idx in range(args.n):
                    mask = (chosen_models == learner_idx)
                    learner_counts = mask.sum(axis=1)
                    assignment_counts[:, learner_idx] = learner_counts
                    assignment_fraction[:, learner_idx] = learner_counts / mask.shape[1]

                # Evaluate on TEST set
                model_losses, ensemble_losses, eval_pts, model_acc, ensemble_acc = \
                    all_models_with_ensemble_population_loss_census(
                        X_test, args.n, Theta, y_test, loss_func=logistic_loss,
                        jump=100, return_accuracy=True
                    )

                model_losses_full, ensemble_losses_full, _, model_acc_full, ensemble_acc_full = \
                    all_models_with_ensemble_population_loss_census(
                        X_test, args.n, Theta_full, y_test, loss_func=logistic_loss,
                        jump=100, return_accuracy=True
                    )

                # Compute distance from ERM solution (bar_Theta)
                distances_from_erm = np.zeros((len(eval_pts), args.n))
                for i, t_idx in enumerate(eval_pts):
                    for learner_idx in range(args.n):
                        distances_from_erm[i, learner_idx] = np.linalg.norm(
                            Theta[learner_idx, :, t_idx] - bar_Theta[learner_idx, :],
                            ord=2
                        )

                # Store results with (seed, p, kappa) as key
                results_dict[(init_seed, p, kappa)] = {
                    "init_seed": init_seed,
                    "probing_p": p,
                    "ranking_noise": kappa,
                    "n": args.n,
                    "tau": args.tau,
                    "probing_set": args.probing_set if p > 0 else [],
                    "model_losses": model_losses,
                    "ensemble_losses": ensemble_losses,
                    "model_acc": model_acc,
                    "ensemble_acc": ensemble_acc,
                    "Theta": Theta,
                    "Theta_full": Theta_full,
                    "model_losses_full": model_losses_full,
                    "ensemble_losses_full": ensemble_losses_full,
                    "model_acc_full": model_acc_full,
                    "ensemble_acc_full": ensemble_acc_full,
                    "eval_points": eval_pts,
                    "distances_from_erm": distances_from_erm,
                    "chosen_models": chosen_models,
                    "assignment_fraction": assignment_fraction,
                    "assignment_counts": assignment_counts,
                    "user_assignment_counts": user_assignment_counts,
                }

                print(f"    Final accuracies: {model_acc[-1]}")

    print("\nTraining complete!")
    return results_dict


def print_final_accuracies(results_dict, full_lr_test_acc):
    """Print final update_all accuracies for all configurations."""
    print("\n" + "="*80)
    print("FINAL UPDATE_ALL MODEL ACCURACIES")
    print("="*80)

    for key in sorted(results_dict.keys()):
        seed, p, kappa = key
        res = results_dict[key]

        # Get final update_all accuracies
        final_update_all_avg = res["model_acc_full"][-1].mean()
        final_update_all_min = res["model_acc_full"][-1].min()
        final_update_all_max = res["model_acc_full"][-1].max()

        # Get final MSGD accuracies
        final_msgd_avg = res["model_acc"][-1].mean()
        final_msgd_min = res["model_acc"][-1].min()
        final_msgd_max = res["model_acc"][-1].max()

        # Get final ensemble accuracy
        final_ensemble = res["ensemble_acc"][-1]

        print(f"\nSeed {seed}, p={p}, κ={kappa}:")
        print(f"  Update_all: avg={final_update_all_avg:.4f}, min={final_update_all_min:.4f}, max={final_update_all_max:.4f}")
        print(f"  MSGD:       avg={final_msgd_avg:.4f}, min={final_msgd_min:.4f}, max={final_msgd_max:.4f}")
        print(f"  Ensemble:   {final_ensemble:.4f}")
        print(f"  Reference (Full LR): {full_lr_test_acc:.4f}")
        print(f"  Update_all vs Reference: {final_update_all_avg - full_lr_test_acc:+.4f}")

    print("\n" + "="*80)


def generate_plots(results_dict, X_train, args):
    """Generate all plots for the experiment."""
    print("\nGenerating plots...")

    # Plot 1: Individual model accuracies
    plot_individual_model_accuracies(results_dict, args, only_p0=False, show_title=False)

    # Plot 2: Assignment fractions
    title_assign = f'Learner assignment fraction (mode={args.clustering_mode}, init={args.init_method})'
    plot_assignment_fraction(results_dict, args, title_assign, max_iter=args.max_plot_iterations)

    # Plot 3: User assignment map
    seed_to_plot = 0
    p_value_to_plot = 0.0
    title_map = f'User assignment map (p={p_value_to_plot}, seed={seed_to_plot}, mode={args.clustering_mode})'
    plot_user_assignment_map(results_dict, X_train, p_value=p_value_to_plot, kappa=0.0,
                            seed=seed_to_plot, title=title_map)

    print("Plots saved successfully!")


def run_decoupled_gd_analysis(results_dict, X_train, y_train, X_test, y_test, args, seed=0, p_value=0.0):
    """Optional: Train decoupled GD on MSGD-induced partitions for comparison."""
    print("\n" + "="*80)
    print("Running decoupled GD analysis on MSGD-induced partitions")
    print("="*80)

    key = (seed, p_value, 0.0)
    res = results_dict.get(key)
    if res is None:
        print(f'Missing results for seed={seed}, p={p_value}')
        return

    assignments = res['user_assignment_counts'].argmax(axis=1)
    n_features = X_train.shape[1]

    T_gd_part = args.T
    eta_gd_part = 0.01
    reg_lambda_part = args.reg_lambda
    bar_theta_part = np.zeros((args.n, n_features))

    partition_test_acc = []
    for learner_idx in range(args.n):
        mask = assignments == learner_idx
        X_partition = X_train[mask]
        y_partition = y_train[mask]
        if len(X_partition) == 0:
            print(f'Learner {learner_idx}: no users assigned, skipping')
            partition_test_acc.append(np.nan)
            continue
        theta = np.zeros(n_features)
        for t in range(T_gd_part):
            theta_expanded = theta.reshape(1, -1)
            _, batch_grad = logistic_loss(X_partition, theta_expanded, y_partition)
            grad = batch_grad[0].mean(axis=0) + reg_lambda_part * theta
            theta -= eta_gd_part * grad
        bar_theta_part[learner_idx] = theta
        pred = theta @ X_test.T
        acc = ((pred > 0) == y_test).mean()
        partition_test_acc.append(acc)
        print(f'Learner {learner_idx}: test acc {acc:.4f} (users={len(X_partition)})')

    print('\nDecoupled GD accuracies:', partition_test_acc)
    out = res['model_acc'][-1]
    print('MSGD final accuracies:', out)
    print("="*80)


def main():
    """Main execution function."""
    # Configuration
    args = CensusConfig()

    print("="*80)
    print("CENSUS MSGD EXPERIMENTS")
    print("="*80)
    print(f"Configuration:")
    print(f"  n={args.n}, T={args.T}, eta={args.eta}, tau={args.tau}")
    print(f"  Clustering: {args.clustering_method} (mode={args.clustering_mode})")
    print(f"  Initialization: {args.init_method}")
    print(f"  Probing set: {args.probing_set}")
    print("="*80)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, X_original_train, X_original_test = load_and_preprocess_data()
    n_features = X_train.shape[1]

    # Train baseline
    FULL_LR_TEST_ACC = train_baseline_lr(X_train, y_train, X_test, y_test, args.reg_lambda)

    # Clustering
    cluster_labels, kmeans = perform_clustering(X_train, X_original_train, args)

    # Create rankings from clusters
    print("\nCreating rankings from clusters...")
    n_clusters = args.n
    rankings = create_rankings_from_clusters(X_train, kmeans.cluster_centers_, n_clusters, cluster_labels)

    # Initialize bar_Theta using sklearn L-BFGS (for reference)
    cache_path = Path('cache/bar_theta_sklearn.npy')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not args.force_recompute_bar_theta:
        print(f'\nLoading bar_Theta_sklearn from {cache_path}')
        bar_Theta_sklearn = np.load(cache_path)
    else:
        print('\nComputing bar_Theta using sklearn L-BFGS...')
        bar_Theta_sklearn = initialize_theta_from_erm_sklearn(
            X_train, y_train, rankings, args.n, n_features, reg_lambda=args.reg_lambda)
        np.save(cache_path, bar_Theta_sklearn)

    print("="*80)
    print("bar_Theta (sklearn lbfgs) test accuracies:")
    test_preds = bar_Theta_sklearn @ X_test.T
    test_accs = ((test_preds > 0) == y_test).mean(axis=1)
    print(f"  {test_accs}")
    print(f"  Mean: {test_accs.mean():.4f}, Best: {test_accs.max():.4f}, Worst: {test_accs.min():.4f}")
    print("="*80)

    # Initialize bar_Theta using GD (main method)
    bar_Theta = initialize_theta_gd(
        X_train, y_train, rankings, args.n, n_features,
        T=args.T, eta=0.01, reg_lambda=args.reg_lambda,
        clustering_mode=args.clustering_mode,
        cache_dir='cache',
        force_recompute=args.force_recompute_bar_theta,
        return_diagnostics=False
    )

    print("\n" + "="*80)
    print("bar_Theta (GD constant lr) test accuracies:")
    test_preds = bar_Theta @ X_test.T
    test_accs = ((test_preds > 0) == y_test).mean(axis=1)
    print(f"  {test_accs}")
    print(f"  Mean: {test_accs.mean():.4f}, Best: {test_accs.max():.4f}, Worst: {test_accs.min():.4f}")
    print("="*80)

    # Run MSGD experiments
    results_dict = run_msgd_experiments(X_train, y_train, X_test, y_test, rankings, bar_Theta, args, num_seeds=15)

    # Print final accuracies
    print_final_accuracies(results_dict, FULL_LR_TEST_ACC)

    # Generate plots
    generate_plots(results_dict, X_train, args)

    # Optional: Run decoupled GD analysis
    run_decoupled_gd_analysis(results_dict, X_train, y_train, X_test, y_test, args, seed=0, p_value=0.0)

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
