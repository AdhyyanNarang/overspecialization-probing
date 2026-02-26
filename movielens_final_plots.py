"""
MovieLens MSGD Experiments - Main Script

This script performs multi-learner strategic gradient descent (MSGD) experiments
on the MovieLens 10M dataset with optional offline probing.

Replaces the functionality of movieLens_final_plots.ipynb.
"""

import numpy as np
import random
import pickle
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# Add utils to path
sys.path.append("./utils")

from utils_msgd_movielens import (
    MSGD_with_probing,
    square_loss,
    all_models_population_loss
)
from utils_clustering import (
    cluster_users_majority_minority,
    create_rankings_from_clusters,
    visualize_individual_clusters_2d
)


@dataclass
class MovieLensConfig:
    """Configuration for MovieLens MSGD experiments."""
    # Data parameters
    data_source_file: str = "./dataset/MovieLens10M_200_5.pkl"
    test_size: float = 0.01  # Train-test split ratio

    # Model parameters
    n: int = 5  # Number of models/learners
    T: int = 4000  # Number of training iterations
    eta: float = 1.0  # Learning rate
    num_sample: int = 50  # Batch size
    reg_lambda: float = 0.001  # L2 regularization
    init_seed: int = 3  # Random seed

    # Experiment parameters
    init_seeds: list = None  # Seeds to run experiments with
    p_values: list = None  # Probing weight values to test
    fixed_probing_set: list = None  # Which learners will probe
    tau: float = 0.3  # Ranking weight
    rankings_only: bool = False  # Use only rankings for model selection

    # Clustering parameters
    majority_percentage: float = 0.80  # Percentage for majority cluster
    clustering_mode: str = 'no_majority'  # 'single_majority', 'half_majority', or 'no_majority'

    # Probing parameters
    N_probe: int = 5000  # Size of offline probe dataset
    probe_num_samples: int = 100  # Batch size for probe gradients
    kappa_values: list = None  # Ranking noise values to test when p > 0

    # Evaluation parameters
    jump: int = 500  # Evaluation frequency (iterations)

    # Initialization method: 'random' or 'erm'
    init_method: str = 'erm'

    # Output parameters
    output_file: str = 'movieLens_probing_results_multi_seed_p.pkl'
    save_mode: str = 'slim'  # 'slim' drops large trajectories from saved results; 'full' keeps them

    def __post_init__(self):
        if self.init_seeds is None:
            self.init_seeds = list(range(1))
        if self.p_values is None:
            self.p_values = [0, 0.05, 0.1, 0.5, 1.5]
        if self.kappa_values is None:
            self.kappa_values = [0.0]
        if self.fixed_probing_set is None:
            self.fixed_probing_set = [3, 4]


def load_movielens_data(config):
    """
    Load and preprocess MovieLens data.

    Parameters
    ----------
    config : MovieLensConfig
        Configuration object

    Returns
    -------
    dict
        Dictionary containing:
        - X_train, X_test: user embeddings
        - y_train, y_test: ratings
        - mask_train, mask_test: rating masks
        - item_embeddings: movie embeddings
        - num_movies: number of movies
        - num_emb: embedding dimension
    """
    print("Loading MovieLens data...")
    with open(config.data_source_file, 'rb') as file:
        loaded_data = pickle.load(file)

    # Extract data
    user_embeddings = loaded_data["user_embeddings"]
    item_embeddings = loaded_data["item_embeddings"]
    ratings = loaded_data["ratings"]
    masks = loaded_data["mask"]
    num_movies = item_embeddings.shape[0]
    num_emb = user_embeddings.shape[1]

    # Create train-test split
    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
        user_embeddings, ratings, masks,
        test_size=config.test_size,
        random_state=0
    )

    print(f"Train set: {X_train.shape[0]} users")
    print(f"Test set: {X_test.shape[0]} users")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'mask_train': mask_train,
        'mask_test': mask_test,
        'item_embeddings': item_embeddings,
        'num_movies': num_movies,
        'num_emb': num_emb
    }


def fit_population_risk_minimiser(X, Y, Mask, lambd=0.0):
    """
    Returns Theta_opt that minimises the masked MSE on (X, Y, Mask).

    Parameters
    ----------
    X     : (N_users, d)   user-embedding matrix
    Y     : (N_users, m)   observed ratings
    Mask  : (N_users, m)   1 = rating present, 0 = missing
    lambd : float          ridge penalty λ (0 ⇒ pure OLS)

    Returns
    -------
    Theta_opt : (m, d)     weight matrix with one row per movie
    """
    N, d = X.shape
    m = Y.shape[1]
    Theta_opt = np.zeros((m, d))

    # Pre-compute per-user outer products once
    XtX_per_user = np.einsum('ni,nj->nij', X, X)  # (N, d, d)

    for k in range(m):
        idx = Mask[:, k].astype(bool)
        if not idx.any():  # no ratings for this movie
            continue

        # Sum_{j : mask=1} X_j X_j^T  and  Sum_{j : mask=1} X_j Y_{jk}
        XtX = XtX_per_user[idx].sum(axis=0)  # (d, d)
        Xty = (X[idx].T @ Y[idx, k])  # (d,)

        # Ridge:  (XtX + λI) w = Xty
        Theta_opt[k] = np.linalg.solve(
            XtX + lambd * np.eye(d), Xty
        )

    return Theta_opt


def initialize_theta_from_erm(X_train, y_train, mask_train, rankings, n,
                               num_movies, num_emb, reg_lambda=0.001):
    """
    Initialize each learner's parameters using ERM solution on users who rank that learner highest.

    Parameters
    ----------
    X_train : (N_users, d) user embeddings
    y_train : (N_users, m) ratings
    mask_train : (N_users, m) rating masks
    rankings : (N_users, n) user rankings of learners (0 = highest rank)
    n : int, number of learners
    num_movies : int, number of movies
    num_emb : int, embedding dimension
    reg_lambda : float, regularization parameter

    Returns
    -------
    Theta_init : (n, num_movies, num_emb) initial parameters
    """
    Theta_init = np.zeros((n, num_movies, num_emb))

    for i in range(n):
        # Find users who rank learner i as their top choice
        top_users_mask = (rankings[:, i] == 0)
        top_users_indices = np.where(top_users_mask)[0]

        if len(top_users_indices) == 0:
            print(f"Warning: Learner {i} has no top-ranked users. Using random initialization.")
            Theta_init[i, :, :] = np.random.rand(num_movies, num_emb)
            continue

        # Extract subset of data for this learner
        X_subset = X_train[top_users_indices]
        y_subset = y_train[top_users_indices]
        mask_subset = mask_train[top_users_indices]

        # Solve ERM for this learner's subset
        Theta_init[i, :, :] = fit_population_risk_minimiser(
            X_subset, y_subset, mask_subset, lambd=reg_lambda
        )

        # Count how many ratings this learner was trained on
        num_ratings = mask_subset.sum()
        print(f"Learner {i}: Initialized from {len(top_users_indices)} users "
              f"({num_ratings} total ratings)")

    return Theta_init


def compute_baseline(X_train, y_train, mask_train):
    """
    Compute the baseline empirical risk minimiser's loss.

    Parameters
    ----------
    X_train : (N_users, d) user embeddings
    y_train : (N_users, m) ratings
    mask_train : (N_users, m) rating masks

    Returns
    -------
    float
        The baseline loss
    """
    Theta_star = fit_population_risk_minimiser(X_train, y_train, mask_train, lambd=0.0)
    # square_loss expects shape (n_models, m, d), so add a leading axis
    min_loss_per_user, _ = square_loss(X_train, Theta_star[None, ...], y_train, mask_train)
    global_min_loss = min_loss_per_user.mean()
    print(f"Empirical risk minimiser's loss = {global_min_loss}")
    return global_min_loss


def perform_clustering(X_train, config):
    """
    Perform user clustering and create rankings.

    Parameters
    ----------
    X_train : (N_users, d) user embeddings
    config : MovieLensConfig
        Configuration object

    Returns
    -------
    tuple
        (cluster_labels, kmeans, rankings)
    """
    print("Performing user clustering...")
    cluster_labels, kmeans = cluster_users_majority_minority(
        X_train,
        n_clusters=config.n,
        majority_percentage=config.majority_percentage,
        mode=config.clustering_mode
    )

    rankings = create_rankings_from_clusters(
        X_train, kmeans.cluster_centers_, config.n, cluster_labels
    )

    print(f"Clustering complete. Mode: {config.clustering_mode}")
    return cluster_labels, kmeans, rankings


def run_msgd_experiments(data, config, cluster_labels, kmeans, rankings, Theta_init_erm):
    """
    Run MSGD experiments across all seeds and p values.

    Parameters
    ----------
    data : dict
        Dictionary containing train/test data
    config : MovieLensConfig
        Configuration object
    cluster_labels : array
        Cluster assignments for users
    kmeans : KMeans
        Fitted KMeans object
    rankings : array
        User rankings of learners
    Theta_init_erm : array
        ERM-initialized parameters

    Returns
    -------
    dict
        Results dictionary with (seed, p, kappa) as keys
    """
    all_results = {}
    if config.save_mode not in {"slim", "full"}:
        raise ValueError(f"Unknown save_mode '{config.save_mode}'. Expected 'slim' or 'full'.")

    for init_seed in config.init_seeds:
        print(f"\nProcessing seed {init_seed}")
        random.seed(init_seed)
        np.random.seed(init_seed)

        # Use ERM initialization (same for all p values within a seed)
        Theta_init = Theta_init_erm.copy()

        for p in config.p_values:
            # For p=0, ranking_noise does nothing; run once with kappa=0.0 only.
            kappa_list = [0.0] if p == 0 else config.kappa_values

            for kappa in kappa_list:
                print(f"Seed {init_seed}: Running with p={p}, kappa={kappa}")

                # Use fixed probing set, only probe if p > 0
                probing_set = config.fixed_probing_set if p > 0 else []

                # Run MSGD with current (p, kappa)
                result = MSGD_with_probing(
                    data['X_train'], Theta_init.copy(), config.n,
                    data['y_train'], data['mask_train'], config.T,
                    square_loss, config.eta,
                    rankings=rankings,
                    rankings_only=config.rankings_only,
                    tau=config.tau,
                    num_sample=config.num_sample,
                    probing_set=probing_set,
                    probing_p=p,
                    reg_lambda=config.reg_lambda,
                    N_probe=config.N_probe,
                    probe_num_samples=config.probe_num_samples,
                    mode=config.clustering_mode,
                    ranking_noise=kappa
                )

                # Reshape Theta trajectories
                Theta = result['Theta_traj']
                Theta_full = result['Update_all_Theta_traj']

                # Compute assignment fraction from chosen models
                chosen_models = result['chosen_models']  # (T, num_sample)
                T_iters = chosen_models.shape[0]
                assignment_fraction = np.zeros((T_iters, config.n))
                for learner_idx in range(config.n):
                    mask = (chosen_models == learner_idx)  # (T, num_sample) boolean mask
                    learner_counts = mask.sum(axis=1)  # (T,) - count per iteration
                    assignment_fraction[:, learner_idx] = learner_counts / chosen_models.shape[1]  # Normalize by batch size

                # Compute individual model losses on TEST set
                model_losses, eval_points = all_models_population_loss(
                    data['X_test'], config.n, Theta, data['y_test'],
                    data['mask_test'], square_loss, jump=config.jump
                )
                model_losses_full, _ = all_models_population_loss(
                    data['X_test'], config.n, Theta_full, data['y_test'],
                    data['mask_test'], square_loss, jump=config.jump
                )

                # Store results
                run_result = {
                    "init_seed": init_seed,
                    "probing_p": p,
                    "ranking_noise": kappa,
                    "n": config.n,
                    "model_losses": model_losses,
                    "model_losses_full": model_losses_full,
                    "eval_points": eval_points,
                    "probing_set": probing_set,
                    "assignment_fraction": assignment_fraction
                }
                if config.save_mode == 'full':
                    run_result["Theta"] = Theta.reshape(
                        (Theta.shape[0], Theta.shape[1] * Theta.shape[2], Theta.shape[3])
                    )
                    run_result["Theta_full"] = Theta_full.reshape(
                        (Theta_full.shape[0], Theta_full.shape[1] * Theta_full.shape[2], Theta_full.shape[3])
                    )
                all_results[(init_seed, p, kappa)] = run_result

    print(
        f"\nCompleted runs for {len(config.init_seeds)} seeds, "
        f"{len(config.p_values)} p values, and up to {len(config.kappa_values)} kappa values"
    )

    # Save all results to a file
    with open(config.output_file, 'wb') as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved to {config.output_file}")

    return all_results


def print_final_results(all_results):
    """
    Print summary of final results.

    Parameters
    ----------
    all_results : dict
        Results dictionary with (seed, p, kappa) as keys
    """
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    for (seed, p, kappa), result in all_results.items():
        final_losses = result['model_losses'][-1]
        final_losses_full = result['model_losses_full'][-1]

        print(f"\nSeed {seed}, p={p}, κ={kappa}:")
        print(f"  Individual models - Mean: {final_losses.mean():.4f}, "
              f"Best: {final_losses.min():.4f}, Worst: {final_losses.max():.4f}")
        print(f"  Update_all - Mean: {final_losses_full.mean():.4f}, "
              f"Best: {final_losses_full.min():.4f}, Worst: {final_losses_full.max():.4f}")


def generate_plots(all_results, config):
    """
    Generate plots from results.

    Parameters
    ----------
    all_results : dict
        Results dictionary with (seed, p) as keys
    config : MovieLensConfig
        Configuration object
    """
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    # Import plotting utilities
    from utils_plotting_movielens import (
        plot_individual_model_losses,
        plot_final_loss_vs_p,
        plot_assignment_fraction,
        compute_parameter_differences
    )

    # Plot individual model losses
    title = 'Ranking Based Probing, MovieLens'
    plot_individual_model_losses(all_results, title, save_file_name="movielens_losses")
    print("Generated individual model losses plot")

    # Plot final loss vs p value
    title_loss_vs_p = 'Final Loss vs Probing Weight'
    plot_final_loss_vs_p(all_results, config, title=title_loss_vs_p, save_file_name="movielens_final_loss_vs_p")
    print("Generated final loss vs p plot")

    # Plot assignment fraction
    title_assignment = 'Learner Assignment Fraction'
    plot_assignment_fraction(all_results, config, title=title_assignment, save_file_name="movielens_assignment_fraction")
    print("Generated assignment fraction plot")


def main():
    """Main execution function."""
    # Create configuration
    config = MovieLensConfig()

    print("="*80)
    print("MovieLens MSGD Experiments")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Models: {config.n}")
    print(f"  Iterations: {config.T}")
    print(f"  Learning rate: {config.eta}")
    print(f"  Batch size: {config.num_sample}")
    print(f"  Regularization: {config.reg_lambda}")
    print(f"  Probing dataset size: {config.N_probe}")
    print(f"  Probe batch size: {config.probe_num_samples}")
    print(f"  Seeds: {config.init_seeds}")
    print(f"  p values: {config.p_values}")
    print(f"  kappa values: {config.kappa_values}")
    print(f"  Probing set: {config.fixed_probing_set}")
    print(f"  Clustering mode: {config.clustering_mode}")
    print(f"  Initialization method: {config.init_method}")
    print()

    # Load data
    data = load_movielens_data(config)

    # Compute baseline
    baseline_loss = compute_baseline(
        data['X_train'], data['y_train'], data['mask_train']
    )

    # Perform clustering
    cluster_labels, kmeans, rankings = perform_clustering(data['X_train'], config)

    # Visualize clusters (optional)
    # visualize_individual_clusters_2d(
    #     data['X_train'], cluster_labels, kmeans.cluster_centers_, method='pca'
    # )

    # Initialize parameters
    if config.init_method == 'erm':
        print("\nInitializing theta using ERM solution on highest-ranked users...")
        Theta_init_erm = initialize_theta_from_erm(
            data['X_train'], data['y_train'], data['mask_train'],
            rankings, config.n, data['num_movies'], data['num_emb'],
            reg_lambda=config.reg_lambda
        )

        # Evaluate initial performance on test set
        print("\nEvaluating initial ERM-initialized learners on test set:")
        test_losses, _ = square_loss(data['X_test'], Theta_init_erm,
                                      data['y_test'], data['mask_test'])
        test_losses_per_model = test_losses.mean(axis=1)
        print(f"Initial test losses: {test_losses_per_model}")
        print(f"Test - Mean: {test_losses_per_model.mean():.4f}, "
              f"Best: {test_losses_per_model.min():.4f}, "
              f"Worst: {test_losses_per_model.max():.4f}")
    else:
        # Random initialization
        print("\nUsing random initialization...")
        Theta_init_erm = np.random.rand(config.n, data['num_movies'], data['num_emb'])

    # Run experiments
    all_results = run_msgd_experiments(
        data, config, cluster_labels, kmeans, rankings, Theta_init_erm
    )

    # Print final results
    print_final_results(all_results)

    # Generate plots
    generate_plots(all_results, config)

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
