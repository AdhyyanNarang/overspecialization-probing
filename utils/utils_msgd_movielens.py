import numpy as np
import argparse
import pickle
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import ipdb

import numpy as np

def square_loss_correct_scale(x, theta, y, mask, eps: float = 1e-8, loss_only=False):
    """
    Mask-aware mean-squared error and gradient.

    Parameters
    ----------
    x     : (b, d)   batch of user-embedding row-vectors
    theta : (n, m, d) model parameters  (n models, m movies, d dims)
    y     : (b, m)   true ratings
    mask  : (b, m)   1 if (user, movie) rating is present, else 0
    eps   : float    small constant to avoid divide-by-zero

    Returns
    -------
    batched_loss : (n, b)   MSE for every (model, user) pair,
                            averaged over *rated* entries only
    batched_grad : (n, b, m, d) gradient w.r.t. theta
    """
    # ---------- forward pass ----------
    # predictions: (n, m, b)  →  (n, b, m)
    pred = np.matmul(theta, x.T).transpose(0, 2, 1)
    err  = pred - y                               # broadcasts y → (1,b,m)
    sq_err = err**2 * mask                        # zero out unrated entries

    # number of rated items per user (shape (b,))
    rated_counts = mask.sum(axis=-1)
    rated_counts = np.maximum(rated_counts, 1)    # avoid division by 0

    # masked mean-squared error: (n,b)
    batched_loss = sq_err.sum(axis=-1) / rated_counts

    if loss_only:
        return batched_loss, None

    # ---------- backward pass ----------
    # scale factor 2/rated_counts for each user
    scale = (2.0 / rated_counts)[None, :, None]   #  (1,b,1) for broadcast
    weighted_err = err * mask * scale             #  (n,b,m)
    # einsum: (n,b,m) × (b,d)  →  (n,b,m,d)
    batched_grad = np.einsum('nbm,bd->nbmd', weighted_err, x)

    return batched_loss, batched_grad


def square_loss(x, theta, y, mask, loss_only=False):
    """
    Compute square loss and optionally its gradient.
    
    Parameters:
    - x: input features
    - theta: model parameters
    - y: target values
    - mask: mask for valid entries
    - loss_only: if True, only compute and return the loss (skip gradient computation)
    
    Returns:
    - batched_loss: computed loss
    - batched_grad: gradient (if loss_only=False), otherwise None
    """
    # Compute predictions and loss
    predictions = np.matmul(theta, x.T).transpose(0, 2, 1)
    squared_errors = (predictions - y)**2
    batched_loss = np.mean(np.multiply(squared_errors, mask), axis=-1)
    
    # Skip gradient computation if only loss is needed
    if loss_only:
        return batched_loss, None
    
    # Compute gradient if needed
    batched_grad = np.einsum('ijk,jl->ijkl', np.multiply(2 * (predictions - y), mask), x)
    return batched_loss, batched_grad

def MSGD_with_probing(X, Theta, n, Y, Mask, T, loss_func, eta, rankings=None, rankings_only=True, tau=1.0,
                       num_sample=1, probing_set=None, probing_p=0.5, reg_lambda=0.0, N_probe=500, probe_num_samples=10, mode='single_majority', ranking_noise=0.0):
    """
    Multi‐model SGD for MovieLens with notebook‐friendly progress bar and user rankings.
    
    Parameters:
    - X: input features
    - Theta: model parameters
    - n: number of models
    - Y: target values
    - Mask: mask for valid entries
    - T: number of iterations
    - zeta: exploration parameter
    - loss_func: loss function
    - eta: learning rate
    - rankings: array of shape (num_users, n) containing each user's ranking of each model
               (smaller values = higher preference)
    - tau: weight for the ranking term in model selection
    - num_sample: number of samples
    - reg_lambda: L2 regularization parameter (default=0.0)
    
    Returns:
    - Dictionary containing model trajectories
    """
    num_movies = Y.shape[1]
    num_emb    = X.shape[1]
    num_users  = X.shape[0]
    # Pre-sample user indices for batching: shape (T, num_sample)
    user_indices = np.random.choice(X.shape[0], size=(T, num_sample), replace=True)
    zeta = 0.0

    if probing_set is None:
        probing_set = set()

    # Initialize rankings if not provided
    if rankings is None:
        # Default: no preference (all zeros)
        rankings = np.zeros((num_users, n))

    # Pre-collect offline probe datasets for each learner in probing_set
    probe_datasets = {}
    if probing_set:
        print(f"Collecting offline probe datasets (N_probe={N_probe})...")
        for mid in probing_set:
            # Sample N_probe users randomly
            probe_user_indices = np.random.choice(num_users, size=N_probe, replace=True)
            probe_X = X[probe_user_indices, :]          # (N_probe, d)
            probe_Mask = Mask[probe_user_indices, :]    # (N_probe, m)

            # Get probe labels based on mode
            preds = np.matmul(Theta, probe_X.T)  # (n, m, d) × (d, N_probe) → (n, m, N_probe)
            preds = preds.transpose(2, 0, 1)     # → (N_probe, n, m)

            if mode == 'single_majority':
                # Use Model 0's predictions as probe labels
                probe_Y = preds[:, 0, :]  # (N_probe, m)
            elif mode == 'half_majority':
                # Use median of all models' predictions as probe labels
                probe_Y = np.median(preds, axis=1)  # (N_probe, m)
            elif mode == 'no_majority':
                # Use each probe user's top-ranked learner's prediction (with optional noise)
                probe_Y = np.zeros((N_probe, preds.shape[2]))  # (N_probe, m)
                for i, user_idx in enumerate(probe_user_indices):
                    # Find the top-ranked learner for this user (smallest ranking value)
                    top_ranked_learner = np.argmin(rankings[user_idx])

                    # Apply ranking noise: with probability ranking_noise, use wrong learner
                    if ranking_noise > 0 and np.random.random() < ranking_noise:
                        # Select a different learner uniformly at random
                        other_learners = [l for l in range(n) if l != top_ranked_learner]
                        selected_learner = np.random.choice(other_learners)
                    else:
                        # Use the correct top-ranked learner
                        selected_learner = top_ranked_learner

                    # Use that learner's predictions for all movies
                    probe_Y[i, :] = preds[i, selected_learner, :]
            else:
                raise ValueError(f"Unknown mode: {mode}. Must be 'single_majority', 'half_majority', or 'no_majority'")

            probe_datasets[mid] = {
                'X': probe_X,
                'Y': probe_Y,
                'Mask': probe_Mask,
                'user_indices': probe_user_indices
            }

            # Print stats
            num_ratings = probe_Mask.sum()
            print(f"  Learner {mid}: {N_probe} probe samples ({num_ratings} total ratings)")

    # Preallocate trajectories (T+1 to include initial parameters)
    Theta_traj            = np.zeros((n, num_movies, num_emb, T + 1))
    Update_all_Theta_traj = np.zeros((n, num_movies, num_emb, T + 1))
    Update_all_Theta      = Theta.copy()

    # Store initial parameters at position 0
    Theta_traj[:, :, :, 0]            = Theta
    Update_all_Theta_traj[:, :, :, 0] = Update_all_Theta

    # Track which model was chosen for each user at each iteration
    chosen_models = np.zeros((T, num_sample), dtype=int)
    user_indices_per_iter = np.zeros((T, num_sample), dtype=int)
    
    # Wrap the range in tqdm so each iteration auto‐updates
    progress_bar = tqdm(
        range(T),
        desc=f"MSGD (ζ={zeta}, τ={tau})",
        dynamic_ncols=True,
        leave=True,
    )

    for t in progress_bar:
        # Get batch of users
        batch_users = user_indices[t]
        x    = X[batch_users, :]
        y    = Y[batch_users, :]
        mask = Mask[batch_users, :]

        # stochastic choice for which model to update
        p_zeta = np.random.rand(num_sample)

        # get losses + grads under both schemes
        batch_loss, batch_grad               = loss_func(x, Theta, y, mask)
        _,          batch_grad_Update_all    = loss_func(x, Update_all_Theta, y, mask)

        # Incorporate rankings into model selection: M(x; Θ) = argmin_θᵢ ℓ(x, θ) + τ·π(x, i)
        # Add the ranking term to the loss for model selection

        ranking_penalty = tau * rankings[batch_users].T        # shape (n, num_sample)

        if rankings_only:
            selection_score = ranking_penalty
            model_id = np.argmin(selection_score, axis=0)      # shape (num_sample,)

        else:
            best_loss_model = np.argmin(batch_loss, axis=0)
            best_ranked_model = np.argmin(ranking_penalty, axis=0)
            model_id = np.where(np.random.random(num_sample) < tau,
                              best_ranked_model,
                              best_loss_model)


        flip     = (p_zeta < zeta)
        model_id[flip] = np.random.randint(0, n, size=flip.sum())

        # Track chosen models and user indices
        chosen_models[t] = model_id
        user_indices_per_iter[t] = batch_users

        # gather grads per‐model
        grad_Theta = np.zeros_like(Theta)
        grad_selected = batch_grad[model_id, np.arange(num_sample), :, :]
        for mid in np.unique(model_id):
            # main gradient
            grad_Theta[mid] = grad_selected[model_id == mid].mean(axis=0)

        # Offline probing updates
        for mid in probing_set:
            # Sample from pre-collected offline dataset
            probe_data = probe_datasets[mid]
            probe_indices = np.random.choice(N_probe, size=probe_num_samples, replace=True)

            probe_x_batch = probe_data['X'][probe_indices]
            probe_y_batch = probe_data['Y'][probe_indices]
            probe_msk_batch = probe_data['Mask'][probe_indices]

            # Compute gradient on probe samples
            _, probe_grad_full = loss_func(probe_x_batch, Theta, probe_y_batch, probe_msk_batch)
            probe_grad = probe_grad_full[mid].mean(axis=0)  # Average over probe batch

            # Mix gradients: weighted average instead of simple addition
            grad_Theta[mid] = (grad_Theta[mid] + probing_p * probe_grad) / (1 + probing_p)
        
        # overall‐average update for the "baseline" model set
        Update_all_grad_Theta = batch_grad_Update_all.mean(axis=1)
        
        # Add L2 regularization to gradients
        if reg_lambda > 0:
            grad_Theta += reg_lambda * Theta
            Update_all_grad_Theta += reg_lambda * Update_all_Theta

        # SGD step with 1/√t learning rate decay
        lr = eta / np.sqrt(t + 1)
        Theta             -= lr * grad_Theta
        Update_all_Theta  -= lr * Update_all_grad_Theta

        # record trajectories (at t+1 since t=0 stores initial params)
        Theta_traj[:, :, :, t + 1]            = Theta
        Update_all_Theta_traj[:, :, :, t + 1] = Update_all_Theta
        
        # occasionally show current loss
        if (t % 100 == 0) or (t == T - 1):
            avg_loss = batch_loss.mean()
            progress_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
    
    return {
        'Theta_traj': Theta_traj,
        'Update_all_Theta_traj': Update_all_Theta_traj,
        'probe_datasets': probe_datasets,
        'chosen_models': chosen_models,
        'user_indices': user_indices_per_iter
    }

def MSGD_with_probingv2(
        X, Theta, n, Y, Mask, T,
        loss_func, eta,
        rankings=None, rankings_only=True, tau=1.0,
        num_sample=1,
        probing_set=None, probing_p=0.5,
        reg_lambda=0.0, N_probe=500, probe_num_samples=10, mode='single_majority', ranking_noise=0.0):
    num_users, num_emb = X.shape
    num_movies = Y.shape[1]

    # --------  deterministic RNG draws made once  --------
    rng = np.random.default_rng()              # use Generator for clarity
    user_idx_samples = rng.integers(num_users, size=(T, num_sample))  # main users (T, num_sample)

    # Fallbacks
    if rankings is None:
        rankings = np.zeros((num_users, n))
    if probing_set is None:
        probing_set = set()

    # Pre-collect offline probe datasets for each learner in probing_set
    probe_datasets = {}
    if probing_set:
        print(f"Collecting offline probe datasets (N_probe={N_probe})...")
        for mid in probing_set:
            # Sample N_probe users randomly
            probe_user_indices = np.random.choice(num_users, size=N_probe, replace=True)
            probe_X = X[probe_user_indices, :]          # (N_probe, d)
            probe_Mask = Mask[probe_user_indices, :]    # (N_probe, m)

            # Get probe labels based on mode
            preds = np.matmul(Theta, probe_X.T)  # (n, m, d) × (d, N_probe) → (n, m, N_probe)
            preds = preds.transpose(2, 0, 1)     # → (N_probe, n, m)

            if mode == 'single_majority':
                # Use Model 0's predictions as probe labels
                probe_Y = preds[:, 0, :]  # (N_probe, m)
            elif mode == 'half_majority':
                # Use median of all models' predictions as probe labels
                probe_Y = np.median(preds, axis=1)  # (N_probe, m)
            elif mode == 'no_majority':
                # Use each probe user's top-ranked learner's prediction (with optional noise)
                probe_Y = np.zeros((N_probe, preds.shape[2]))  # (N_probe, m)
                for i, user_idx in enumerate(probe_user_indices):
                    # Find the top-ranked learner for this user (smallest ranking value)
                    top_ranked_learner = np.argmin(rankings[user_idx])

                    # Apply ranking noise: with probability ranking_noise, use wrong learner
                    if ranking_noise > 0 and np.random.random() < ranking_noise:
                        # Select a different learner uniformly at random
                        other_learners = [l for l in range(n) if l != top_ranked_learner]
                        selected_learner = np.random.choice(other_learners)
                    else:
                        # Use the correct top-ranked learner
                        selected_learner = top_ranked_learner

                    # Use that learner's predictions for all movies
                    probe_Y[i, :] = preds[i, selected_learner, :]
            else:
                raise ValueError(f"Unknown mode: {mode}. Must be 'single_majority', 'half_majority', or 'no_majority'")

            probe_datasets[mid] = {
                'X': probe_X,
                'Y': probe_Y,
                'Mask': probe_Mask,
                'user_indices': probe_user_indices
            }

            # Print stats
            num_ratings = probe_Mask.sum()
            print(f"  Learner {mid}: {N_probe} probe samples ({num_ratings} total ratings)")

    # Trajectory storage (T+1 to include initial parameters)
    Theta_traj = np.zeros((n, num_movies, num_emb, T + 1))
    Update_all_Theta_traj = np.zeros_like(Theta_traj)
    Update_all_Theta = Theta.copy()

    # Store initial parameters at position 0
    Theta_traj[:, :, :, 0] = Theta
    Update_all_Theta_traj[:, :, :, 0] = Update_all_Theta

    bar = tqdm(range(T), desc="MSGD-P v2", dynamic_ncols=True)

    for t in bar:
        # ----------  main batch of users (draw fixed) ----------
        batch_users = user_idx_samples[t]
        x   = X[batch_users]          # shape (num_sample, d)
        y   = Y[batch_users]
        msk = Mask[batch_users]

        # Loss & gradient for every model
        batch_loss,  batch_grad  = loss_func(x, Theta,          y, msk)
        _,           batch_gradA = loss_func(x, Update_all_Theta, y, msk)

        # Select best model purely by (loss + τ·ranking)
        ranking_penalty = tau * rankings[batch_users].T        # shape (n, num_sample)

        if rankings_only:
            selection_score = ranking_penalty
            model_id = np.argmin(selection_score, axis=0)      # shape (num_sample,)

        else:
            best_loss_model = np.argmin(batch_loss, axis=0)
            best_ranked_model = np.argmin(ranking_penalty, axis=0)
            model_id = np.where(np.random.random(num_sample) < tau,
                              best_ranked_model,
                              best_loss_model)      # (num_sample,)

        # --------  accumulate gradients per model --------
        grad_Theta = np.zeros_like(Theta)
        grad_selected = batch_grad[model_id, np.arange(num_sample), :, :]
        unique_ids = np.unique(model_id)
        for mid in unique_ids:
            # main gradient
            grad_Theta[mid] = grad_selected[model_id == mid].mean(axis=0)

        # Offline probing updates
        for mid in probing_set:
            # Sample from pre-collected offline dataset
            probe_data = probe_datasets[mid]
            probe_indices = np.random.choice(N_probe, size=probe_num_samples, replace=True)

            probe_x_batch = probe_data['X'][probe_indices]
            probe_y_batch = probe_data['Y'][probe_indices]
            probe_msk_batch = probe_data['Mask'][probe_indices]

            # Compute gradient on probe samples
            _, probe_grad_full = loss_func(probe_x_batch, Theta, probe_y_batch, probe_msk_batch)
            probe_grad = probe_grad_full[mid].mean(axis=0)  # Average over probe batch

            # Mix gradients: weighted average instead of simple addition
            grad_Theta[mid] = (grad_Theta[mid] + probing_p * probe_grad) / (1 + probing_p)

        # baseline all-models gradient
        grad_all = batch_gradA.mean(axis=1)

        # L2 regularisation (if any)
        if reg_lambda > 0:
            grad_Theta     += reg_lambda * Theta
            grad_all       += reg_lambda * Update_all_Theta

        # SGD step with 1/√t learning rate decay
        lr = eta / np.sqrt(t + 1)
        Theta           -= lr * grad_Theta
        Update_all_Theta -= lr * grad_all

        # log trajectories (at t+1 since t=0 stores initial params)
        Theta_traj[:, :, :, t + 1] = Theta
        Update_all_Theta_traj[:, :, :, t + 1] = Update_all_Theta

        if (t % 100 == 0) or (t == T - 1):
            bar.set_postfix(loss=f"{batch_loss.mean():.4f}")

    return {
        "Theta_traj": Theta_traj,
        "Update_all_Theta_traj": Update_all_Theta_traj,
        "probe_datasets": probe_datasets
    }


def population_loss_movieLens(X, n, zeta, Theta_traj, Y, Mask, loss_func, jump=2000):
    """
    Compute population loss using the training set with skipping to improve performance.
    
    Parameters:
    - X: input features
    - n: number of models
    - zeta: exploration parameter
    - Theta_traj: trajectory of model parameters
    - Y: target values
    - Mask: mask for valid entries
    - loss_func: loss function
    - jump: number of iterations to skip between evaluations (default=25)
    
    Returns:
    - total_loss: computed loss at evaluated time steps
    """
    num_user = X.shape[0]
    T = Theta_traj.shape[-1]
    
    # Only evaluate at every 'jump' iterations
    eval_points = range(0, T, jump)
    if (T-1) not in eval_points:
        eval_points = list(eval_points) + [T-1]  # Always include the last point
    
    total_loss = np.zeros(len(eval_points))
    
    for i, t in enumerate(eval_points):
        # Use loss_only=True to skip gradient computation
        batched_loss, _ = loss_func(X, Theta_traj[:, :, :, t], Y, Mask, loss_only=True)
        model_id = np.argmin(batched_loss, axis=0)
        total_loss[i] = (1-zeta) * np.mean(batched_loss[model_id, range(num_user)]) + zeta * np.mean(batched_loss)
    
    return total_loss, eval_points

def all_models_population_loss(X, n, Theta_traj, Y, Mask, loss_func, jump=2000):
    T = Theta_traj.shape[-1]
    
    # Only evaluate at every 'jump' iterations
    eval_points = range(0, T, jump)
    if (T-1) not in eval_points:
        eval_points = list(eval_points) + [T-1]  # Always include the last point
    
    # Initialize array to store each model's loss at each evaluated time step
    model_losses = np.zeros((len(eval_points), n))
    
    for i, t in enumerate(eval_points):
        # Use loss_only=True to skip gradient computation
        batched_loss, _ = loss_func(X, Theta_traj[:, :, :, t], Y, Mask, loss_only=True)
        
        # Calculate the average loss for each model across all users
        model_losses[i, :] = np.mean(batched_loss, axis=1)
    
    return model_losses, eval_points