"""
MovieLens-specific plotting utilities

Functions for visualizing MovieLens MSGD experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from datetime import datetime
from pathlib import Path


def plot_individual_model_losses(results_dict, title=None, save_file_name=None, baseline_loss=None):
    """
    Plots individual model losses for all seeds and p values for the MovieLens dataset.

    Args:
        results_dict: Dictionary of results with (seed, p) as keys
        title: Optional title for the figure
        save_file_name: Base name for the saved file (without extension). If None, defaults to 'movielens_losses'.
        baseline_loss: Optional scalar for a dashed black horizontal reference line.
    """
    # Apply census plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'figure.dpi': 150,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
    })

    all_keys_raw = list(results_dict.keys())
    keys_have_kappa = len(all_keys_raw[0]) == 3
    if keys_have_kappa:
        kappas = sorted({k for (_, _, k) in all_keys_raw})
        selected_kappa = 0.0 if 0.0 in kappas else kappas[0]
        filtered_keys = [(s, p) for (s, p, k) in all_keys_raw if k == selected_kappa]
        key_lookup = lambda seed, p: (seed, p, selected_kappa)
    else:
        filtered_keys = all_keys_raw
        key_lookup = lambda seed, p: (seed, p)

    # Extract unique p values and seeds
    p_values = sorted(list({p for (_, p) in filtered_keys}))
    seeds = sorted(list({seed for (seed, _) in filtered_keys}))

    # Get number of models from the first result
    n_models = results_dict[list(results_dict.keys())[0]]["n"]

    # Use sophisticated color palette matching census plots
    rgb_palette = [
        (0.890, 0.102, 0.110),     # Vibrant Red
        (0.216, 0.494, 0.722),     # Deep Blue
        (0.302, 0.686, 0.290),     # Green
        (0.596, 0.306, 0.639),     # Purple
        (1.000, 0.498, 0.000),     # Orange
        (1.000, 0.765, 0.000),     # Gold
        (0.651, 0.337, 0.157),     # Brown
        (0.969, 0.506, 0.749),     # Pink
        (0.600, 0.600, 0.600),     # Gray
        (0.094, 0.745, 0.804),     # Cyan
    ]
    # If n_models > length of palette, cycle
    colors = [c for c, _ in zip(itertools.cycle(rgb_palette), range(n_models))]

    # Check if we should aggregate across seeds
    aggregate_seeds = len(seeds) > 1

    if aggregate_seeds:
        # Aggregated mode: one subplot per p value, averaging across seeds
        fig, axes = plt.subplots(1, len(p_values),
                                figsize=(7*len(p_values), 5.5),
                                sharex=True, sharey=True, squeeze=False)
        axes = axes[0]  # Get the row

        for j, p in enumerate(p_values):
            ax = axes[j] if len(p_values) > 1 else axes[0]

            # Get all results for this p value across all seeds
            p_results = [
                results_dict[key_lookup(seed, p)]
                for seed in seeds
                if key_lookup(seed, p) in results_dict
            ]

            if not p_results:
                continue

            ep = np.array(p_results[0]["eval_points"])
            probing_set = p_results[0].get("probing_set", [])

            # Aggregate model losses across seeds
            all_model_losses = np.array([res["model_losses"] for res in p_results])  # (n_seeds, n_eval_pts, n_models)
            mean_model_losses = np.mean(all_model_losses, axis=0)  # (n_eval_pts, n_models)
            stderr_model_losses = np.std(all_model_losses, axis=0) / np.sqrt(len(p_results))  # (n_eval_pts, n_models)

            # Plot individual models with error bars
            for k in range(n_models):
                lw = 3.5 if k in probing_set else 2.5
                label = f"Learner {k}"
                ax.plot(ep, mean_model_losses[:, k],
                        label=label,
                        color=colors[k], linewidth=lw,
                        marker='o', markersize=6, markevery=max(1, len(ep)//15))
                ax.fill_between(
                    ep,
                    mean_model_losses[:, k] - stderr_model_losses[:, k],
                    mean_model_losses[:, k] + stderr_model_losses[:, k],
                    alpha=0.25, color=colors[k])
            if baseline_loss is not None and len(ep) > 0:
                ax.hlines(
                    baseline_loss,
                    xmin=ep[0],
                    xmax=ep[-1],
                    color='black',
                    linewidth=1.5,
                    linestyle='--',
                )

            # Grid styling (matching census plots)
            ax.grid(True, which='major', alpha=0.4, linewidth=1.2, linestyle='-')
            ax.grid(True, which='minor', alpha=0.15, linewidth=0.8, linestyle=':')
            ax.minorticks_on()

            # Clean up spines for modern look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

            # Set title with proper styling
            if title is not None:
                ax.set_title(f"{title} (p={p})", fontweight='semibold', pad=12)

            # Add labels
            ax.set_xlabel("Iteration", fontweight='semibold')
            if j == 0:
                ax.set_ylabel("MSE Loss", fontweight='semibold')

    else:
        # Single seed mode: grid with one row per seed (usually just 1), one column per p value
        fig, axes = plt.subplots(len(seeds), len(p_values),
                                figsize=(7*len(p_values), 5.5*len(seeds)),
                                sharex=True, sharey=True)

        # Iterate through all combinations of seeds and p values
        for i, seed in enumerate(seeds):
            for j, p in enumerate(p_values):
                key = key_lookup(seed, p)
                if key not in results_dict:
                    continue

                # Get the current axis - handle both 1D and 2D axes arrays
                if len(seeds) == 1 and len(p_values) == 1:
                    ax = axes
                elif len(seeds) == 1:
                    ax = axes[j]
                elif len(p_values) == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]

                res = results_dict[key]
                ep = res["eval_points"]

                # Get probing set for this configuration
                probing_set = res.get("probing_set", [])

                # Plot individual model losses with markers
                for k in range(n_models):
                    # Use thicker lines for probing learners (matching census style)
                    lw = 3.5 if k in probing_set else 2.5
                    label = f"Learner {k}" if i==0 and j==0 else None
                    ax.plot(ep, res["model_losses"][:, k],
                            label=label,
                            color=colors[k], linewidth=lw,
                            marker='o', markersize=6, markevery=max(1, len(ep)//15))
                if baseline_loss is not None and len(ep) > 0:
                    ax.hlines(
                        baseline_loss,
                        xmin=ep[0],
                        xmax=ep[-1],
                        color='black',
                        linewidth=1.5,
                        linestyle='--',
                    )

                # # Also plot the Update_all average
                # ax.plot(ep, res["model_losses_full"].mean(axis=1),
                #        label="Update_all (avg)" if i==0 and j==0 else None,
                #        color='black', ls="--", lw=2, alpha=0.7)

                # # Add a horizontal reference line at 2.92
                # ax.axhline(y=2.92, color='red', linestyle=':', lw=1, alpha=0.7,
                #           label="Reference (2.92)" if i==0 and j==0 else None)

                # Grid styling (matching census plots)
                ax.grid(True, which='major', alpha=0.4, linewidth=1.2, linestyle='-')
                ax.grid(True, which='minor', alpha=0.15, linewidth=0.8, linestyle=':')
                ax.minorticks_on()

                # Clean up spines for modern look
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)

                # Set title with proper styling
                if title is not None:
                    ax.set_title(f"{title} (p={p})", fontweight='semibold', pad=12)

                # Add x and y labels only on the left and bottom edges
                if j == 0:
                    ax.set_ylabel("MSE Loss", fontweight='semibold')
                if i == len(seeds) - 1:
                    ax.set_xlabel("Iteration", fontweight='semibold')

    # Create a single legend for the entire figure (matching census style)
    if len(seeds) == 1 and len(p_values) == 1:
        handles, labels = axes.get_legend_handles_labels()
    else:
        # axes is either 1D or 2D array, use flat[0] to get first axis
        handles, labels = axes.flat[0].get_legend_handles_labels()

    # Display legend in a single row
    ncol_legend = len(handles)
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, 0.03),
              ncol=ncol_legend, fontsize=13, frameon=True,
              fancybox=True, shadow=True, borderpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # Create final_figs directory if it doesn't exist
    save_dir = Path("final_figs")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use provided name or default
    base_name = save_file_name if save_file_name else "movielens_losses"

    # Add timestamp to filename (date + time)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"{base_name}_{timestamp_str}.pdf"

    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Figure saved to: {save_path}")
    plt.show()


def plot_final_loss_vs_p(
    results_dict,
    config,
    title=None,
    save_file_name=None,
    baseline_loss=None,
    flag_single_row=False,
    flag_kappa=False,
):
    """
    Plots final MSE loss vs p value for each learner.
    Shows mean ± stderr across seeds when multiple seeds exist.

    Args:
        results_dict: Dictionary of results with (seed, p) or (seed, p, kappa) as keys
        config: Configuration object with probing_set attribute
        title: Optional title for the figure
        save_file_name: Base name for the saved file (without extension). If None, defaults to 'final_loss_vs_p'.
        baseline_loss: Optional scalar for a dashed black horizontal reference line.
        flag_single_row (bool): If True, render legend as a single row.
        flag_kappa (bool): If True, include '(κ=...)' in probing-learner legend labels.
    """
    # Apply census plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'figure.dpi': 150,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
    })

    # Normalize keys to (seed, p, kappa) for backward compatibility.
    all_keys_raw = list(results_dict.keys())
    keys_have_kappa = len(all_keys_raw[0]) == 3
    if keys_have_kappa:
        all_keys = all_keys_raw
    else:
        all_keys = [(seed, p, 0.0) for (seed, p) in all_keys_raw]

    seeds = sorted(list({seed for (seed, _, _) in all_keys}))
    p_values = sorted(list({p for (_, p, _) in all_keys}))
    kappa_values = sorted(list({kappa for (_, _, kappa) in all_keys}))

    n_models = results_dict[all_keys[0]]["n"]
    probing_set = getattr(config, 'fixed_probing_set', [])

    # Use the same color palette as plot_individual_model_losses
    rgb_palette = [
        (0.890, 0.102, 0.110),     # Vibrant Red
        (0.216, 0.494, 0.722),     # Deep Blue
        (0.302, 0.686, 0.290),     # Green
        (0.596, 0.306, 0.639),     # Purple
        (1.000, 0.498, 0.000),     # Orange
        (1.000, 0.765, 0.000),     # Gold
        (0.651, 0.337, 0.157),     # Brown
        (0.969, 0.506, 0.749),     # Pink
        (0.600, 0.600, 0.600),     # Gray
        (0.094, 0.745, 0.804),     # Cyan
    ]
    colors = [c for c, _ in zip(itertools.cycle(rgb_palette), range(n_models))]

    # Linestyles for different kappa values
    linestyles = ['-', '--', '-.', ':']

    # Organize data: final_losses[learner_idx][p][kappa] = list of losses across seeds
    final_losses = {}
    for learner_idx in range(n_models):
        final_losses[learner_idx] = {}
        for p in p_values:
            final_losses[learner_idx][p] = {}
            for kappa in kappa_values:
                final_losses[learner_idx][p][kappa] = []

    # Extract final losses from results_dict
    for (seed, p, kappa) in all_keys:
        key = (seed, p, kappa) if keys_have_kappa else (seed, p)
        res = results_dict[key]
        final_loss_array = res["model_losses"][-1]  # Shape: (n_models,)
        for learner_idx in range(n_models):
            final_losses[learner_idx][p][kappa].append(final_loss_array[learner_idx])

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot each learner
    for learner_idx in range(n_models):
        color = colors[learner_idx]

        if learner_idx in probing_set:
            # Probing learner: plot one curve per kappa.
            for kappa_idx, kappa in enumerate(kappa_values):
                p_vals_for_plot = []
                mean_losses = []
                stderr_losses = []
                for p in p_values:
                    losses_for_cfg = final_losses[learner_idx][p][kappa]
                    if losses_for_cfg:
                        losses = np.array(losses_for_cfg)
                        p_vals_for_plot.append(p)
                        mean_losses.append(np.mean(losses))
                        stderr_losses.append(np.std(losses) / np.sqrt(len(losses)))

                if p_vals_for_plot:
                    linestyle = linestyles[kappa_idx % len(linestyles)]
                    label = f"Learner {learner_idx} (κ={kappa})" if flag_kappa else f"Learner {learner_idx}"
                    ax.plot(
                        p_vals_for_plot,
                        mean_losses,
                        color=color,
                        linestyle=linestyle,
                        linewidth=2.5,
                        marker='^',
                        markersize=8,
                        label=label,
                    )
                    ax.fill_between(
                        p_vals_for_plot,
                        np.array(mean_losses) - np.array(stderr_losses),
                        np.array(mean_losses) + np.array(stderr_losses),
                        alpha=0.2,
                        color=color,
                    )
        else:
            # Non-probing learner: single curve at kappa=0.0.
            kappa = 0.0 if 0.0 in kappa_values else kappa_values[0]
            p_vals_for_plot = []
            mean_losses = []
            stderr_losses = []

            for p in p_values:
                losses_for_cfg = final_losses[learner_idx][p][kappa]
                if losses_for_cfg:
                    losses = np.array(losses_for_cfg)
                    p_vals_for_plot.append(p)
                    mean_losses.append(np.mean(losses))
                    stderr_losses.append(np.std(losses) / np.sqrt(len(losses)))

            if p_vals_for_plot:
                label = f"Learner {learner_idx}"
                ax.plot(
                    p_vals_for_plot,
                    mean_losses,
                    color=color,
                    linestyle='-',
                    linewidth=2.5,
                    marker='o',
                    markersize=7,
                    label=label,
                )
                ax.fill_between(
                    p_vals_for_plot,
                    np.array(mean_losses) - np.array(stderr_losses),
                    np.array(mean_losses) + np.array(stderr_losses),
                    alpha=0.2,
                    color=color,
                )

    if baseline_loss is not None and p_values:
        ax.hlines(
            baseline_loss,
            xmin=min(p_values),
            xmax=max(p_values),
            color='black',
            linewidth=1.5,
            linestyle='--',
        )

    # Styling
    ax.grid(True, which='major', alpha=0.4, linewidth=1.2, linestyle='-')
    ax.grid(True, which='minor', alpha=0.15, linewidth=0.8, linestyle=':')
    ax.minorticks_on()

    # Clean up spines for modern look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.set_xlabel("Probing Weight (p)", fontweight='semibold')
    ax.set_ylabel("Final MSE Loss", fontweight='semibold')

    if title:
        ax.set_title(title, fontweight='semibold', pad=12)

    # Legend layout control.
    handles, labels = ax.get_legend_handles_labels()
    if not flag_kappa:
        # De-duplicate repeated labels when multiple kappa curves share one learner label.
        seen = set()
        unique_handles = []
        unique_labels = []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
        handles, labels = unique_handles, unique_labels

    ncol_legend = max(1, len(handles)) if flag_single_row else min(4, max(1, len(handles)))
    legend_y = -0.10 if flag_single_row else -0.15
    legend = ax.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, legend_y),
        ncol=ncol_legend,
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16 if flag_single_row else 0.22)

    # Create final_figs directory if it doesn't exist
    save_dir = Path("final_figs")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use provided name or default
    base_name = save_file_name if save_file_name else "final_loss_vs_p"

    # Add timestamp to filename (date + time)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"{base_name}_{timestamp_str}.pdf"

    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Figure saved to: {save_path}")
    plt.show()


def plot_assignment_fraction(results_dict, config, title=None, max_iter=None, save_file_name=None, window_size=50):
    """
    Plots the fraction of users assigned to each learner over iterations using a running average.

    Args:
        results_dict: Dictionary of results with (seed, p) as keys
        config: Configuration object
        title: Optional title for the figure
        max_iter: Maximum iteration to plot (defaults to config.T)
        save_file_name: Base name for the saved file (without extension). If None, defaults to 'assignment_fraction'.
        window_size: Size of the running average window (default: 50)
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    def running_average(data, window):
        """Compute running average using convolution."""
        if window <= 1:
            return data
        kernel = np.ones(window) / window
        # Use 'same' mode to keep the same length, pad at edges
        return np.convolve(data, kernel, mode='same')

    all_keys_raw = list(results_dict.keys())
    keys_have_kappa = len(all_keys_raw[0]) == 3
    if keys_have_kappa:
        kappas = sorted({k for (_, _, k) in all_keys_raw})
        selected_kappa = 0.0 if 0.0 in kappas else kappas[0]
        filtered_keys = [(s, p) for (s, p, k) in all_keys_raw if k == selected_kappa]
        key_lookup = lambda seed, p: (seed, p, selected_kappa)
    else:
        filtered_keys = all_keys_raw
        key_lookup = lambda seed, p: (seed, p)

    max_iter = max_iter or getattr(config, 'T', None)
    p_values = sorted({p for (_, p) in filtered_keys})
    seeds = sorted({seed for (seed, _) in filtered_keys})
    n_models = results_dict[list(results_dict.keys())[0]]["n"]
    colors = plt.cm.Dark2(np.linspace(0, 1, n_models))

    fig, axes = plt.subplots(1, len(p_values), figsize=(6.5*len(p_values), 4.5), sharey=True)
    if len(p_values) == 1:
        axes = [axes]

    for j, p in enumerate(p_values):
        ax = axes[j]
        p_results = [
            results_dict[key_lookup(seed, p)]
            for seed in seeds
            if key_lookup(seed, p) in results_dict
        ]
        if not p_results:
            continue
        frac_arrays = np.array([res["assignment_fraction"] for res in p_results])  # (n_seeds, T, n_models)
        iters = np.arange(frac_arrays.shape[1])
        if max_iter is not None:
            mask = iters <= max_iter
            frac_arrays = frac_arrays[:, mask]
            iters = iters[mask]
        mean_frac = frac_arrays.mean(axis=0)
        stderr_frac = frac_arrays.std(axis=0) / np.sqrt(len(p_results))

        # Apply running average to smooth the curves
        for k in range(n_models):
            smoothed_mean = running_average(mean_frac[:, k], window_size)
            smoothed_stderr = running_average(stderr_frac[:, k], window_size)

            ax.plot(iters, smoothed_mean, color=colors[k], linewidth=2, label=f"Learner {k}")
            ax.fill_between(iters, smoothed_mean - smoothed_stderr, smoothed_mean + smoothed_stderr,
                          color=colors[k], alpha=0.15)
        ax.set_title(f"p = {p}")
        ax.set_xlabel("Iteration")
        if j == 0:
            ax.set_ylabel("Fraction of users per batch")
        ax.set_ylim(0, 0.6)
        ax.grid(alpha=0.25)
    axes[0].legend(loc='upper right')
    if title:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Create final_figs directory if it doesn't exist
    save_dir = Path("final_figs")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use provided name or default
    base_name = save_file_name if save_file_name else "assignment_fraction"

    # Add timestamp to filename (date + time)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"{base_name}_{timestamp_str}.pdf"

    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Figure saved to: {save_path}")
    plt.show()


def compute_parameter_differences(theta_traj):
    """
    Compute the Frobenius norm of parameter differences between consecutive iterations.

    Parameters:
    theta_traj: numpy array of shape (n, num_params, T)

    Returns:
    norm_diffs: numpy array of shape (n, T-1) containing the norm differences
    """
    n, num_params, T = theta_traj.shape
    norm_diffs = np.zeros((n, T-1))

    for i in range(n):
        for t in range(1, T):
            # Compute the Frobenius norm of the difference
            diff = theta_traj[i, :, t] - theta_traj[i, :, t-1]
            norm_diffs[i, t-1] = np.linalg.norm(diff)

    return norm_diffs


def plot_parameter_convergence(results_dict, selected_seed=0):
    """
    Plot parameter convergence for a selected seed across different p values.

    Args:
        results_dict: Dictionary of results with (seed, p) as keys
        selected_seed: Seed to plot (default: 0)
    """
    # Filter results for the selected seed
    seed_results = {key: value for key, value in results_dict.items() if key[0] == selected_seed}

    # Plot results for each p value (for the selected seed)
    plt.figure(figsize=(6, 4))

    for result_idx, ((seed, p), result) in enumerate(seed_results.items()):

        # Get the flattened parameter trajectories
        theta_traj = result['Theta']  # Shape: (n, num_movies*num_emb, T)

        # Compute norm differences
        norm_diffs = compute_parameter_differences(theta_traj)

        # Compute cumulative sum of differences
        n, T_minus_1 = norm_diffs.shape
        cumsum_diffs = np.cumsum(norm_diffs, axis=1)

        # Create a subplot for each p value
        plt.subplot(len(seed_results), 1, result_idx + 1)

        for i in range(n):
            plt.plot(range(1, T_minus_1+1), cumsum_diffs[i], label=f"Model {i+1}")

        plt.xlabel("Time t")
        plt.ylabel(r"$\sum_{t=0}^{T} \|\Theta_t - \Theta_{t-1}\|$")
        plt.title(f"Seed {seed}, p={p}")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig("msgd_convergence.png", dpi=300)
    plt.show()


def plot_cumulative_parameter_differences(results_dict, selected_seed=0):
    """
    Plot cumulative parameter differences across p values for a selected seed.

    Args:
        results_dict: Dictionary of results with (seed, p) as keys
        selected_seed: Seed to plot (default: 0)
    """
    # Filter results for the selected seed
    seed_results = {key: value for key, value in results_dict.items() if key[0] == selected_seed}

    # Create a single combined plot for all p values (for the selected seed)
    plt.figure(figsize=(5, 3))

    # Different colors for different p values
    colors = ['purple', 'blue', 'red', 'green', 'orange']

    # Calculate cumulative sum for each p value
    for result_idx, ((seed, p), result) in enumerate(seed_results.items()):
        theta_traj = result['Theta']
        norm_diffs = compute_parameter_differences(theta_traj)

        # Average across models
        avg_norm_diff = np.mean(norm_diffs, axis=0)

        # Compute cumulative sum
        cumsum_diff = np.cumsum(avg_norm_diff)

        # Plot with different colors for different p values
        plt.plot(range(1, len(cumsum_diff)+1), cumsum_diff,
                 label=f"p={p}", color=colors[result_idx % len(colors)])

    plt.xlabel("Time t")
    plt.ylabel(r"$\sum_{t=0}^{T} \|\Theta_t - \Theta_{t-1}\|$")
    plt.title(f"Movie Rec (Seed {selected_seed})")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("msgd_cumulative_diff.png", dpi=300)
    plt.show()


def plot_model_performance(results_dict, selected_seed=0, selected_p=0):
    """
    Plot the performance of each model and the Update_all model on the full population.

    Parameters:
    - results_dict: Dictionary of results with (seed, p) as keys
    - selected_seed: Seed to plot (default: 0)
    - selected_p: p value to plot (default: 0)
    """
    if (selected_seed, selected_p) not in results_dict:
        print(f"Warning: No results found for seed={selected_seed}, p={selected_p}")
        return

    result = results_dict[(selected_seed, selected_p)]
    model_losses = result['model_losses']
    model_losses_full = result['model_losses_full']
    eval_points = result['eval_points']
    n = result['n']

    plt.figure(figsize=(12, 6))

    # Plot individual model losses
    for i in range(n):
        plt.plot(eval_points, model_losses[:, i], label=f"Model {i}", linestyle='-')

    # Plot Update_all model losses
    for i in range(1):
        plt.plot(eval_points, model_losses_full[:, i], label=f"Update_all Model",
                 linestyle='--', alpha=0.7)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Average Loss", fontsize=12)
    plt.title(f"Model Performance on Full Population (Seed {selected_seed}, p={selected_p})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"model_performance_seed{selected_seed}_p{selected_p}.png", dpi=300)
    plt.show()

    # Optional: Create a second plot showing only the best model from each approach
    plt.figure(figsize=(10, 5))

    # Find the best performing model at the final evaluation point
    best_model_idx = np.argmin(model_losses[-1, :])
    best_update_all_idx = np.argmin(model_losses_full[-1, :])

    plt.plot(eval_points, model_losses[:, best_model_idx],
             label=f"Best MSGD Model ({best_model_idx})", linewidth=2)
    plt.plot(eval_points, model_losses_full[:, best_update_all_idx],
             label=f"Best Update_all Model ({best_update_all_idx})",
             linewidth=2, linestyle='--')

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Average Loss", fontsize=12)
    plt.title(f"Best Model Performance Comparison (Seed {selected_seed}, p={selected_p})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"best_model_comparison_seed{selected_seed}_p{selected_p}.png", dpi=300)
    plt.show()
