import numpy as np
import argparse
import pickle
import random
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSEmployment
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.linear_model import LogisticRegression 

def _compute_lr(eta, t, schedule):
    if schedule == 'sqrt':
        return eta / np.sqrt(t + 1)
    if schedule == 'constant':
        return eta
    raise ValueError(f"Unknown lr_schedule '{schedule}'. Expected 'constant' or 'sqrt'.")

def logistic_loss(x, theta, y):
    predict_f = lambda x,theta: np.matmul(theta, x.T) 
    sigmoid_f = lambda yhat: 1/(1+np.exp(-yhat))  
    loss_f = lambda y, sigmoid: -(y.T * np.log(np.clip(sigmoid, 1e-7, None))+(1-y).T * np.log(np.clip(1- sigmoid, 1e-7, None)))
    grad_f = lambda x,y, sigmoid: np.einsum('ij,ki->kij', x, sigmoid -y)
    loss = loss_f(y, sigmoid_f(predict_f(x,theta)))
    gradient = grad_f(x,y, sigmoid_f(predict_f(x,theta)))
    return  loss, gradient

def MSGD_census(X, Theta, n, Y, T, loss_func, eta, num_sample = 1, probing_set=None, reg_lambda=0.0, lr_schedule='sqrt'):
    num_emb = X.shape[1] 
    Theta_traj = np.zeros((n, num_emb, T))
    Update_all_Theta_traj = np.zeros((n, num_emb, T))
    Update_all_Theta = Theta
    for t in range(T):
        user = list(range(num_sample * t, num_sample *(t+1)))
        x = X[user, :] 
        y = Y[user]
       
        batch_loss,  batch_grad = loss_func(x, Theta, y) # x.shape: (1, 16); Theta.shape: (3, 16), y.shape: (1, )
        _,  batch_grad_Update_all = loss_func(x, Update_all_Theta, y) 
        model_id = np.argmin(batch_loss, axis = 0) 
        grad_model_to_update  = batch_grad[model_id, range(num_sample),:] 
        grad_Theta = np.zeros_like(Theta)
        values, _ = np.unique(model_id, return_counts=True)
        for model_index in values:
            grad_Theta[model_index,:] = np.mean(grad_model_to_update[model_id == model_index, :], axis = 0)
        Update_all_grad_Theta = np.mean(batch_grad_Update_all, axis = 1)
        lr = _compute_lr(eta, t, lr_schedule)
        Theta = Theta - lr * grad_Theta
        Update_all_Theta = Update_all_Theta - lr * Update_all_grad_Theta
        Theta_traj[:, :, t] = Theta
        Update_all_Theta_traj[:, :, t] = Update_all_Theta
    result = {'Theta_traj': Theta_traj, 'Update_all_Theta_traj': Update_all_Theta_traj}
    return result  
# notebook/console–friendly progress bar

def MSGD_census_with_probing(
    X, Theta, n, Y, T, loss_func, eta, *,
    rankings=None, tau=1.0, rankings_only=False, probing_set=None,
    probing_p=0.5, reg_lambda=0.0, num_sample=1, seed=0,
    N_probe=500, probe_num_samples=10, mode='single_majority', ranking_noise=0.0,
    lr_schedule='sqrt'
):
    print("lr_schedule: ", lr_schedule)
    random.seed(seed)
    np.random.seed(seed)
    num_users, d = X.shape
    
    rankings = np.zeros((num_users, n)) if rankings is None else rankings
    probing_set = set(probing_set) if probing_set else set()
    
    # Pre-collect offline probe datasets for each learner in probing_set
    probe_datasets = {}
    if probing_set:
        print(f"Collecting offline probe datasets (N_probe={N_probe})...")
        for mid in probing_set:
            # Sample N_probe users randomly with replacement
            probe_user_indices = np.random.choice(num_users, size=N_probe, replace=True)
            probe_X = X[probe_user_indices]

            # Get probe labels based on mode
            if mode == 'single_majority':
                # Use Model 0's predictions as probe labels
                probe_preds = (Theta[0] @ probe_X.T)  # Shape: (N_probe,)
            elif mode == 'half_majority':
                # Use median of all models' predictions as probe labels
                all_preds = Theta @ probe_X.T  # Shape: (n, N_probe)
                probe_preds = np.median(all_preds, axis=0)  # Shape: (N_probe,)
            elif mode == 'no_majority':
                # Use each probe user's top-ranked learner's prediction (with optional noise)
                probe_preds = np.zeros(N_probe)
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

                    # Use that learner's prediction
                    probe_preds[i] = Theta[selected_learner] @ probe_X[i]
            else:
                raise ValueError(f"Unknown mode: {mode}. Must be 'single_majority', 'half_majority', or 'no_majority'")

            probe_Y = (probe_preds > 0).astype(float)  # Binary predictions

            probe_datasets[mid] = {
                'X': probe_X,
                'Y': probe_Y,
                'user_indices': probe_user_indices
            }
            print(f"  Learner {mid}: Collected {N_probe} probe samples "
                  f"(label distribution: {np.bincount(probe_Y.astype(int))})")
    
    # Pre-sample random indices for training
    user_indices = np.random.choice(num_users, size=(T, num_sample), replace=True)
    chosen_models = np.zeros_like(user_indices)
    user_assignment_counts = np.zeros((num_users, n), dtype=np.int64)
    
    # Initialize trajectories (T+1 to include initial parameters)
    Theta_traj = np.zeros((n, d, T + 1))
    Update_all_Theta_traj = np.zeros((n, d, T + 1))
    Update_all_Theta = Theta.copy()
    
    # Store initial parameters at position 0
    Theta_traj[:, :, 0] = Theta
    Update_all_Theta_traj[:, :, 0] = Update_all_Theta
    
    pbar = tqdm(range(T), desc=f"MSGD_Census (τ={tau}, probing_p={probing_p})", 
                dynamic_ncols=True, leave=True)
    
    for t in pbar:
        # Get batch
        batch_users = user_indices[t]
        x, y = X[batch_users], Y[batch_users]
        
        # Compute losses and gradients
        batch_loss, batch_grad = loss_func(x, Theta, y)
        _, batch_grad_all = loss_func(x, Update_all_Theta, y)
        
        # Model selection
        ranking_penalty = tau * rankings[batch_users].T  # shape: (n, num_sample)
        rank_choices = np.argmin(ranking_penalty, axis=0)
        if rankings_only:
            model_id = rank_choices
        else:
            loss_choices = np.argmin(batch_loss, axis=0)  # shape: (num_sample,)
            pick_rank = (np.random.random(num_sample) < tau)
            model_id = np.where(pick_rank, rank_choices, loss_choices)
        
        chosen_models[t] = model_id
        np.add.at(user_assignment_counts, (batch_users, model_id), 1)
        
        # Accumulate gradients
        grad_Theta = np.zeros_like(Theta)
        grad_selected = batch_grad[model_id, np.arange(num_sample), :]
        for mid in np.unique(model_id):
            grad_Theta[mid] = grad_selected[model_id == mid].mean(axis=0)
        
        # Offline probing updates
        for mid in probing_set:
            # Sample probe_num_samples from the offline dataset
            probe_data = probe_datasets[mid]
            probe_sample_indices = np.random.choice(N_probe, size=probe_num_samples, replace=True)
            probe_x_batch = probe_data['X'][probe_sample_indices]
            probe_y_batch = probe_data['Y'][probe_sample_indices]
            
            # Compute gradient on probe samples for this learner
            _, probe_grad_full = loss_func(probe_x_batch, Theta, probe_y_batch)
            probe_grad = probe_grad_full[mid].mean(axis=0)  # Average over probe samples
            
            # Mix real gradient with probe gradient (weighted average)
            grad_Theta[mid] = (grad_Theta[mid] + probing_p * probe_grad) / (1 + probing_p)
        
        # Regularization
        grad_Theta += reg_lambda * Theta
        grad_all = batch_grad_all.mean(axis=1) + reg_lambda * Update_all_Theta
        
        # SGD updates with sqrt-decaying learning rate
        lr = _compute_lr(eta, t, lr_schedule)
        Theta -= lr * grad_Theta
        Update_all_Theta -= lr * grad_all
        
        # Record trajectories (at t+1 since t=0 stores initial params)
        Theta_traj[:, :, t + 1] = Theta
        Update_all_Theta_traj[:, :, t + 1] = Update_all_Theta
        
        if t % 100 == 0 or t == T - 1:
            pbar.set_postfix({'avg_loss': f'{batch_loss.mean():.4f}'})
    
    return {
        'Theta_traj': Theta_traj, 
        'Update_all_Theta_traj': Update_all_Theta_traj, 
        'chosen_models': chosen_models,
        'probe_datasets': probe_datasets,
        'user_assignment_counts': user_assignment_counts,
        'user_indices': user_indices,
    }

def all_models_with_ensemble_population_loss_census(
    X_test, n, Theta_traj, y_test, loss_func, *, jump=2000, return_accuracy=False
):
    """
    Returns: (model_losses, ensemble_losses, eval_pts[, model_acc, ensemble_acc])
    """
    T = Theta_traj.shape[-1]
    eval_pts = list(range(0, T, jump)) + ([T - 1] if (T - 1) % jump != 0 else [])
    
    model_losses = np.zeros((len(eval_pts), n))
    ensemble_losses = np.zeros(len(eval_pts))
    model_acc = np.zeros_like(model_losses) if return_accuracy else None
    ensemble_acc = np.zeros(len(eval_pts)) if return_accuracy else None
    
    y_test = y_test.astype(int)
    
    for i, t in enumerate(eval_pts):
        Θ_t = Theta_traj[:, :, t]
        
        # Individual model losses
        batched_loss, _ = loss_func(X_test, Θ_t, y_test)
        model_losses[i] = batched_loss.mean(axis=1)
        
        # Ensemble predictions (median)
        z = np.median(Θ_t @ X_test.T, axis=0)
        
        # Numerically stable logistic loss for ensemble
        log_sigmoid = np.where(z >= 0, -np.log(1 + np.exp(-z)), z - np.log(1 + np.exp(z)))
        log_1m_sigmoid = np.where(z >= 0, -z - np.log(1 + np.exp(-z)), -np.log(1 + np.exp(z)))
        ensemble_losses[i] = (-y_test * log_sigmoid - (1 - y_test) * log_1m_sigmoid).mean()
        
        if return_accuracy:
            model_acc[i] = (((Θ_t @ X_test.T) > 0) == y_test).mean(axis=1)
            ensemble_acc[i] = ((z > 0) == y_test).mean()
    
    return (model_losses, ensemble_losses, eval_pts, model_acc, ensemble_acc) if return_accuracy \
           else (model_losses, ensemble_losses, eval_pts)


def initialize_theta_from_erm_sklearn(X_train, y_train, rankings, n, n_features, reg_lambda=1e-3):
    """
    Initialize theta using sklearn's L-BFGS logistic regression on partitions.
    This function is kept for reference but not used in the main pipeline.
    We now use GD with constant learning rate instead.

    Args:
        X_train: Training features
        y_train: Training labels
        rankings: User rankings of learners (n_users, n_learners)
        n: Number of learners
        n_features: Number of features
        reg_lambda: L2 regularization parameter

    Returns:
        Theta_init: Initialized parameters (n, n_features)
    """
    Theta_init = np.zeros((n, n_features))

    for i in range(n):
        # Find users who rank learner i as their top choice
        top_users_mask = (rankings[:, i] == 0)
        top_users_indices = np.where(top_users_mask)[0]

        if len(top_users_indices) == 0:
            print(f"Warning: Learner {i} has no top-ranked users. Using random initialization.")
            Theta_init[i, :] = np.random.rand(n_features)
            continue

        # Extract subset of data for this learner
        X_subset = X_train[top_users_indices]
        y_subset = y_train[top_users_indices]

        # Check if we have both classes in the subset
        unique_labels = np.unique(y_subset)
        if len(unique_labels) < 2:
            print(f"Warning: Learner {i} subset has only one class. Using random initialization.")
            Theta_init[i, :] = np.random.rand(n_features)
            continue

        # Solve logistic regression ERM
        # Using C = 1/(2*reg_lambda) to match the regularization used in MSGD
        clf = LogisticRegression(
            penalty='l2',
            C=1.0/reg_lambda if reg_lambda > 0 else 1e10,
            solver='lbfgs',
            max_iter=1000,
            fit_intercept=False,  # No intercept to match the MSGD formulation
            random_state=42
        )

        clf.fit(X_subset, y_subset)
        Theta_init[i, :] = clf.coef_[0]

        print(f"Learner {i}: Initialized from {len(top_users_indices)} users "
              f"(class distribution: {np.bincount(y_subset.astype(int))})")

    return Theta_init


def initialize_theta_gd(
    X_train, y_train, rankings, n, n_features,
    T=2000, eta=0.01, reg_lambda=1e-9,
    clustering_mode='no_majority',
    cache_dir='cache',
    force_recompute=False,
    return_diagnostics=False
):
    """
    Initialize theta using gradient descent with constant learning rate on user partitions.
    This matches MSGD's optimization but runs decoupled on each partition.

    Args:
        X_train: Training features (n_users, n_features)
        y_train: Training labels (n_users,)
        rankings: User rankings of learners (n_users, n_learners)
        n: Number of learners
        n_features: Number of features
        T: Number of GD iterations
        eta: Learning rate (constant)
        reg_lambda: L2 regularization parameter
        clustering_mode: Clustering mode string for cache naming
        cache_dir: Directory for caching results
        force_recompute: If True, recompute even if cache exists
        return_diagnostics: If True, return diagnostic information

    Returns:
        bar_Theta: Initialized parameters (n, n_features)
        diagnostics (optional): Dict with loss traces and gradient norms if return_diagnostics=True
    """
    # Setup caching
    cache_name = f"bar_theta_gd_mode={clustering_mode}_n{n}_d{n_features}_reg{reg_lambda}_T{T}_eta{eta}.npy"
    cache_path = Path(cache_dir) / cache_name
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_recompute:
        print(f"Loading bar_Theta_gd from {cache_path}")
        bar_Theta_gd = np.load(cache_path)
        if return_diagnostics:
            return bar_Theta_gd, None
        return bar_Theta_gd

    print("="*80)
    print("Computing bar_Theta using GD with constant learning rate")
    print("This matches MSGD's optimization but runs decoupled on each partition")
    print("="*80)

    bar_Theta_gd = np.random.randn(n, n_features) * 0.01
    log_every = max(1, T // 200)

    # Optional diagnostics
    if return_diagnostics:
        gd_loss_traces = [[] for _ in range(n)]
        gd_grad_norm_traces = [[] for _ in range(n)]
        gd_log_iters = [[] for _ in range(n)]

    for i in range(n):
        partition_mask = (rankings[:, i] == 0)
        partition_indices = np.where(partition_mask)[0]
        X_partition = X_train[partition_indices]
        y_partition = y_train[partition_indices]
        n_partition = len(partition_indices)

        print(f"Learner {i}: Training on {n_partition} users (class dist: {np.bincount(y_partition.astype(int))})")

        theta_i = bar_Theta_gd[i].copy()

        for t in range(T):
            theta_expanded = theta_i.reshape(1, -1)
            batch_loss, batch_grad = logistic_loss(X_partition, theta_expanded, y_partition)
            grad = batch_grad[0].mean(axis=0)
            grad += reg_lambda * theta_i

            if return_diagnostics and (t % log_every == 0 or t == T - 1):
                reg_obj = batch_loss.mean() + 0.5 * reg_lambda * np.linalg.norm(theta_i)**2
                gd_loss_traces[i].append(reg_obj)
                gd_grad_norm_traces[i].append(np.linalg.norm(grad))
                gd_log_iters[i].append(t)

            theta_i -= eta * grad

        bar_Theta_gd[i] = theta_i
        print(f"  Completed GD for Learner {i}")

    # Save to cache
    np.save(cache_path, bar_Theta_gd)
    print(f"Saved bar_Theta_gd to {cache_path}")

    if return_diagnostics:
        diagnostics = {
            'loss_traces': gd_loss_traces,
            'grad_norm_traces': gd_grad_norm_traces,
            'log_iters': gd_log_iters
        }
        return bar_Theta_gd, diagnostics

    return bar_Theta_gd
