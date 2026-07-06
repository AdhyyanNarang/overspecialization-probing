"""
Plotting utilities for census MSGD experiments.

Contains functions for visualizing model accuracies, user assignments,
and other metrics from multi-learner strategic learning experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import itertools
from datetime import datetime
from pathlib import Path


def plot_individual_model_accuracies(
    results_dict,
    args,
    title=None,
    only_p0=False,
    show_title=True,
    show_legend=True,
    save_file_name=None,
    baseline_acc=None,
    y_lim=None,
):
    """
    Plots individual model accuracies aggregated across seeds (mean ± stderr).
    
    Layout:
    - If only kappa=0.0 exists: Single row with columns for different p values (backward compatible)
    - If multiple kappa values exist for p>0: Grid with rows=kappa values, columns=p values
    - For p=0 column: always uses kappa=0.0
    - For p>0 columns: shows each kappa value as a separate row
    
    Supports both separate (per-seed) and aggregated (across seeds) plotting modes.

    Args:
        results_dict: Dictionary of results with (seed, p, kappa) as keys
        args: Configuration object with plot_seeds_separately and max_plot_iterations attributes
        title: Optional title for the figure
        only_p0 (bool): If True, plot only for p=0; else plot all available p.
        show_title (bool): If False, do not display any subplot or main title.
        show_legend (bool): If False, do not display the legend.
        save_file_name (str): Base name for the saved file (without extension). If None, defaults to 'bad_outcome'.
        baseline_acc (float or None): If provided, draws a horizontal reference line at this accuracy.
        y_lim (tuple or None): Optional (lower, upper) y-axis limits. Defaults to (0.4, 0.85).
    """
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

    max_iter = getattr(args, 'max_plot_iterations', None)
    # Choose p values: restrict to p=0 if only_p0 is True
    if only_p0:
        p_values = sorted({p for (_, p, k) in results_dict.keys() if p == 0 and k == 0.0})
    else:
        p_values = sorted({p for (_, p, k) in results_dict.keys() if k == 0.0})

    seeds = sorted({seed for (seed, _, k) in results_dict.keys() if k == 0.0})
    
    # Get all kappa values for p > 0 to detect if we need multiple rows
    kappa_values_p_gt_0 = sorted({k for (_, p, k) in results_dict.keys() if p > 0})
    # Check if we have multiple kappas
    has_multiple_kappas = len(kappa_values_p_gt_0) > 1

    if not p_values:
        print("No p=0 results found in results_dict (for kappa=0.0).")
        return

    n_models = results_dict[list(results_dict.keys())[0]]["n"]
    y_lim = y_lim or (0.4, 0.85)

    # Use a sophisticated color palette with better contrast and aesthetics
    # Based on ColorBrewer qualitative schemes and designed for clarity
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

    if args.plot_seeds_separately:
        # SEPARATE MODE: Create a grid of subplots for the selected p values
        fig, axes = plt.subplots(len(seeds), len(p_values),
                                figsize=(4*len(p_values), 5*len(seeds)),
                                sharex=True, sharey=True)

        for i, seed in enumerate(seeds):
            for j, p in enumerate(p_values):
                if (seed, p, 0.0) not in results_dict:
                    continue

                # Get the current axis
                if len(seeds) == 1 and len(p_values) == 1:
                    ax = axes
                elif len(seeds) == 1:
                    ax = axes[j]
                elif len(p_values) == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]

                res = results_dict[(seed, p, 0.0)]
                ep = res["eval_points"]
                mask = np.ones_like(ep, dtype=bool)
                if max_iter is not None:
                    mask = np.array(ep) <= max_iter
                ep = np.array(ep)[mask]
                probing_set = res.get("probing_set", [])

                # Plot individual model accuracies
                for k in range(n_models):
                    lw = 3.5 if k in probing_set else 2.5
                    ax.plot(
                        ep, res["model_acc"][mask, k],
                        label=f"Learner {k}" if i==0 and j==0 else None,
                        color=colors[k], linewidth=lw, marker='o', markersize=6, markevery=max(1, len(ep)//15))
                if baseline_acc is not None and len(ep) > 0:
                    ax.hlines(
                        baseline_acc,
                        xmin=ep[0],
                        xmax=ep[-1],
                        color='black',
                        linewidth=1.5,
                        linestyle='--',
                    )

                # Grid styling
                ax.grid(True, which='major', alpha=0.4, linewidth=1.2, linestyle='-')
                ax.grid(True, which='minor', alpha=0.15, linewidth=0.8, linestyle=':')
                ax.minorticks_on()
                
                # Clean up spines for modern look
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                
                ax.set_ylim(*y_lim)
                # Set x-axis ticks every 500
                ax.set_xticks(np.arange(0, max(ep)+1, 500))

                # Bring p value to subplot titles unless titles are excluded
                if show_title:
                    if title is None:
                        ax.set_title(f"p = {p}", fontweight='semibold', pad=12)
                    else:
                        ax.set_title(f"{title} (p={p})", fontweight='semibold', pad=12)

                if j == 0:
                    ax.set_ylabel("Accuracy", fontweight='semibold')
                if i == len(seeds) - 1:
                    ax.set_xlabel("Iteration", fontweight='semibold')

        # Get first axis for legend
        if show_legend:
            if len(seeds) == 1 and len(p_values) == 1:
                first_ax = axes
            elif len(seeds) == 1 or len(p_values) == 1:
                first_ax = axes[0]
            else:
                first_ax = axes[0, 0]

            handles, labels = first_ax.get_legend_handles_labels()
            # Display legend in a single row
            #ncol_legend = (n_models + 1) // 2  # This gives us roughly half, rounding up
            ncol_legend = n_models
            legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.70),
                      ncol=ncol_legend, fontsize=13, frameon=True,
                      fancybox=True, shadow=True, borderpad=1)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.95)
            legend.get_frame().set_edgecolor('gray')
            legend.get_frame().set_linewidth(1.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

    else:
        # AGGREGATED MODE: Plot with mean ± stderr across seeds
        # If multiple kappas exist, create rows for each kappa, columns for each p
        # Otherwise, keep single row with columns for each p
        
        if has_multiple_kappas:
            # Multiple rows (one per kappa) x columns (one per p value)
            n_rows = len(kappa_values_p_gt_0)
            n_cols = len(p_values)
            fig, axes = plt.subplots(n_rows, n_cols,
                                    figsize=(7*n_cols, 5.5*n_rows),
                                    sharex=True, sharey=True, squeeze=False)
            
            # Iterate over kappa values (rows) and p values (columns)
            for i, kappa in enumerate(kappa_values_p_gt_0):
                for j, p in enumerate(p_values):
                    ax = axes[i, j]
                    
                    # For p=0, always use kappa=0.0; for p>0, use current kappa
                    current_kappa = 0.0 if p == 0 else kappa
                    
                    # Get all results for this (p, kappa) across all seeds
                    p_results = [results_dict[(seed, p, current_kappa)] 
                                for seed in seeds 
                                if (seed, p, current_kappa) in results_dict]
                    
                    if not p_results:
                        continue
                    
                    ep = np.array(p_results[0]["eval_points"])
                    mask = np.ones_like(ep, dtype=bool)
                    if max_iter is not None:
                        mask = ep <= max_iter
                    ep = ep[mask]
                    probing_set = p_results[0].get("probing_set", [])
                    
                    # Aggregate model accuracies across seeds
                    all_model_accs = np.array([res["model_acc"][mask] for res in p_results])
                    mean_model_accs = np.mean(all_model_accs, axis=0)
                    stderr_model_accs = np.std(all_model_accs, axis=0) / np.sqrt(len(p_results))
                    
                    # Plot individual models with error bars
                    for k in range(n_models):
                        lw = 3.5 if k in probing_set else 2.5
                        # Only add label in first subplot for legend
                        label = f"Learner {k}" if i == 0 and j == 0 else None
                        ax.plot(
                            ep, mean_model_accs[:, k], label=label,
                            color=colors[k], linewidth=lw, marker='o', markersize=6, 
                            markevery=max(1, len(ep)//15))
                        ax.fill_between(
                            ep,
                            mean_model_accs[:, k] - stderr_model_accs[:, k],
                            mean_model_accs[:, k] + stderr_model_accs[:, k],
                            alpha=0.25, color=colors[k])
                    if baseline_acc is not None and len(ep) > 0:
                        ax.hlines(
                            baseline_acc,
                            xmin=ep[0],
                            xmax=ep[-1],
                            color='black',
                            linewidth=1.5,
                            linestyle='--',
                        )
                    
                    # Grid styling
                    ax.grid(True, which='major', alpha=0.4, linewidth=1.2, linestyle='-')
                    ax.grid(True, which='minor', alpha=0.15, linewidth=0.8, linestyle=':')
                    ax.minorticks_on()
                    
                    # Clean up spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_linewidth(1.5)
                    ax.spines['bottom'].set_linewidth(1.5)
                    
                    ax.set_ylim(*y_lim)
                    ax.set_xticks(np.arange(0, max(ep)+1, 1000))
                    
                    # Add titles for top row
                    if i == 0 and show_title:
                        if title is None:
                            ax.set_title(f"p = {p}", fontweight='semibold', pad=12)
                        else:
                            ax.set_title(f"{title} (p={p})", fontweight='semibold', pad=12)
                    
                    # Add x-label for bottom row
                    if i == n_rows - 1:
                        ax.set_xlabel("Iteration", fontweight='semibold')
                    
                    # Add y-label for leftmost column with kappa value
                    if j == 0:
                        ax.set_ylabel(f"κ={kappa}\nAccuracy", fontweight='semibold')
            
            # Add legend at the top center
            if show_legend:
                handles, labels = axes[0, 0].get_legend_handles_labels()
                ncol_legend = n_models
                legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                          ncol=ncol_legend, fontsize=13, frameon=True, 
                          fancybox=True, shadow=True, borderpad=1)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.95)
                legend.get_frame().set_edgecolor('gray')
                legend.get_frame().set_linewidth(1.5)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
        else:
            # Single row: backward compatible behavior when only kappa=0.0 exists
            fig, axes = plt.subplots(1, len(p_values),
                                    figsize=(7*len(p_values), 5.5),
                                    sharex=True, sharey=True, squeeze=False)
            axes = axes[0]  # Get the row

            for j, p in enumerate(p_values):
                ax = axes[j] if len(p_values) > 1 else axes[0]

                # Get all results for this p value across all seeds
                p_results = [results_dict[(seed, p, 0.0)] for seed in seeds if (seed, p, 0.0) in results_dict]

                if not p_results:
                    continue

                ep = np.array(p_results[0]["eval_points"])
                mask = np.ones_like(ep, dtype=bool)
                if max_iter is not None:
                    mask = ep <= max_iter
                ep = ep[mask]
                probing_set = p_results[0].get("probing_set", [])

                # Aggregate model accuracies across seeds
                all_model_accs = np.array([res["model_acc"][mask] for res in p_results])  # (n_seeds, n_eval_pts, n_models)
                mean_model_accs = np.mean(all_model_accs, axis=0)  # (n_eval_pts, n_models)
                stderr_model_accs = np.std(all_model_accs, axis=0) / np.sqrt(len(p_results))  # (n_eval_pts, n_models)

                # Plot individual models with error bars
                for k in range(n_models):
                    lw = 3.5 if k in probing_set else 2.5
                    ax.plot(
                        ep, mean_model_accs[:, k], label=f"Learner {k}",
                        color=colors[k], linewidth=lw, marker='o', markersize=6, markevery=max(1, len(ep)//15))
                    ax.fill_between(
                        ep,
                        mean_model_accs[:, k] - stderr_model_accs[:, k],
                        mean_model_accs[:, k] + stderr_model_accs[:, k],
                        alpha=0.25, color=colors[k])
                if baseline_acc is not None and len(ep) > 0:
                    ax.hlines(
                        baseline_acc,
                        xmin=ep[0],
                        xmax=ep[-1],
                        color='black',
                        linewidth=1.5,
                        linestyle='--',
                    )

                # Grid styling
                ax.grid(True, which='major', alpha=0.4, linewidth=1.2, linestyle='-')
                ax.grid(True, which='minor', alpha=0.15, linewidth=0.8, linestyle=':')
                ax.minorticks_on()
                
                # Clean up spines for modern look
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                
                ax.set_xlabel("Iteration", fontweight='semibold')
                ax.set_ylim(*y_lim)
                # Set x-axis ticks every 500
                ax.set_xticks(np.arange(0, max(ep)+1, 500))
                # Bring p value to subplot titles unless titles are excluded
                if show_title:
                    if title is None:
                        ax.set_title(f"p = {p}", fontweight='semibold', pad=12)
                    else:
                        ax.set_title(f"{title} (p={p})", fontweight='semibold', pad=12)
                if j == 0:
                    ax.set_ylabel("Accuracy", fontweight='semibold')

            # Add legend at the top center of the plot with improved styling
            if show_legend:
                handles, labels = axes[0].get_legend_handles_labels()
                # Display legend in a single row
                ncol_legend = n_models
                legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, 0.0),
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
    base_name = save_file_name if save_file_name else "bad_outcome"
    
    # Add timestamp to filename (date + time)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"{base_name}_{timestamp_str}.pdf"
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Figure saved to: {save_path}")
    plt.show()


def plot_final_accuracy_vs_p(
    results_dict,
    args,
    title=None,
    save_file_name=None,
    baseline_acc=None,
    flag_single_row=False,
    flag_kappa=False,
):
    """
    Plots final accuracy vs p value for each learner.
    Probing learners show multiple curves for different kappa values.
    
    Args:
        results_dict: Dictionary of results with (seed, p, kappa) as keys
        args: Configuration object with probing_set attribute
        title: Optional title for the figure
        save_file_name: Base name for the saved file (without extension). If None, defaults to 'final_accuracy_vs_p'.
        baseline_acc (float or None): If provided, draws a horizontal reference line at this accuracy.
        flag_single_row (bool): If True, render legend as a single row.
        flag_kappa (bool): If True, include '(κ=...)' in probing-learner legend labels.
    """
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
    
    # Extract unique values from results_dict
    all_keys = list(results_dict.keys())
    seeds = sorted(list(set([seed for (seed, _, _) in all_keys])))
    p_values = sorted(list(set([p for (_, p, _) in all_keys])))
    kappa_values = sorted(list(set([k for (_, _, k) in all_keys])))
    
    n_models = results_dict[all_keys[0]]["n"]
    probing_set = getattr(args, 'probing_set', [])
    
    # Use the same color palette as plot_individual_model_accuracies
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
    
    # Organize data: final_accs[learner_idx][p][kappa] = list of accuracies across seeds
    final_accs = {}
    for learner_idx in range(n_models):
        final_accs[learner_idx] = {}
        for p in p_values:
            final_accs[learner_idx][p] = {}
            for kappa in kappa_values:
                final_accs[learner_idx][p][kappa] = []
    
    # Extract final accuracies from results_dict
    for (seed, p, kappa) in all_keys:
        res = results_dict[(seed, p, kappa)]
        final_acc_array = res["model_acc"][-1]  # Shape: (n_models,)
        for learner_idx in range(n_models):
            final_accs[learner_idx][p][kappa].append(final_acc_array[learner_idx])
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot each learner
    for learner_idx in range(n_models):
        color = colors[learner_idx]
        
        if learner_idx in probing_set:
            # Probing learner: plot curves for each kappa value
            for kappa_idx, kappa in enumerate(kappa_values):
                p_vals_for_plot = []
                mean_accs = []
                stderr_accs = []
                
                for p in p_values:
                    if final_accs[learner_idx][p][kappa]:
                        accs = np.array(final_accs[learner_idx][p][kappa])
                        p_vals_for_plot.append(p)
                        mean_accs.append(np.mean(accs))
                        stderr_accs.append(np.std(accs) / np.sqrt(len(accs)))
                
                if p_vals_for_plot:
                    linestyle = linestyles[kappa_idx % len(linestyles)]
                    label = f"Learner {learner_idx} (κ={kappa})" if flag_kappa else f"Learner {learner_idx}"
                    
                    ax.plot(p_vals_for_plot, mean_accs, 
                           color=color, linestyle=linestyle, linewidth=2.5,
                           marker='^', markersize=8, label=label)
                    ax.fill_between(p_vals_for_plot, 
                                   np.array(mean_accs) - np.array(stderr_accs),
                                   np.array(mean_accs) + np.array(stderr_accs),
                                   alpha=0.2, color=color)
        else:
            # Non-probing learner: plot single curve at kappa=0.0
            kappa = 0.0
            p_vals_for_plot = []
            mean_accs = []
            stderr_accs = []
            
            for p in p_values:
                if final_accs[learner_idx][p][kappa]:
                    accs = np.array(final_accs[learner_idx][p][kappa])
                    p_vals_for_plot.append(p)
                    mean_accs.append(np.mean(accs))
                    stderr_accs.append(np.std(accs) / np.sqrt(len(accs)))
            
            if p_vals_for_plot:
                label = f"Learner {learner_idx}"
                ax.plot(p_vals_for_plot, mean_accs,
                       color=color, linestyle='-', linewidth=2.5,
                       marker='o', markersize=7, label=label)
                ax.fill_between(p_vals_for_plot,
                               np.array(mean_accs) - np.array(stderr_accs),
                               np.array(mean_accs) + np.array(stderr_accs),
                               alpha=0.2, color=color)
    
    if baseline_acc is not None and p_values:
        ax.hlines(
            baseline_acc,
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
    ax.set_ylabel("Final Accuracy", fontweight='semibold')
    
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
    base_name = save_file_name if save_file_name else "final_accuracy_vs_p"
    
    # Add timestamp to filename (date + time)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"{base_name}_{timestamp_str}.pdf"
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Figure saved to: {save_path}")
    plt.show()


def plot_individual_model_accuracies_with_kappa(results_dict, title=None):
    """
    Plots individual model accuracies for all (seed, p, kappa) combinations.
    Creates a grid where columns represent different kappa values for p=0.5,
    plus one column for the p=0 baseline.

    Args:
        results_dict: Dictionary of results with (seed, p, kappa) as keys
        title: Optional title for the figure
    """
    # Extract unique values
    all_keys = list(results_dict.keys())
    seeds = sorted(list(set([seed for (seed, _, _) in all_keys])))

    # Get baseline (p=0) keys
    baseline_keys = [(s, p, k) for (s, p, k) in all_keys if p == 0]

    # Get p=0.5 keys and extract kappa values
    p05_keys = [(s, p, k) for (s, p, k) in all_keys if p == 0.5]
    kappa_values = sorted(list(set([k for (_, p, k) in p05_keys if p == 0.5])))

    # Total columns: 1 for baseline + len(kappa_values) for p=0.5
    n_cols = 1 + len(kappa_values)
    n_rows = len(seeds)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols,
                            figsize=(5*n_cols, 5*n_rows),
                            sharex=True, sharey=True)

    # Use a colormap for the different models
    n_models = results_dict[all_keys[0]]["n"]
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    # Iterate through seeds
    for i, seed in enumerate(seeds):
        # Column 0: Baseline (p=0)
        if n_rows == 1 and n_cols == 1:
            ax = axes
        elif n_rows == 1:
            ax = axes[0]
        elif n_cols == 1:
            ax = axes[i]
        else:
            ax = axes[i, 0]

        # Find baseline result for this seed
        baseline_key = [(s, p, k) for (s, p, k) in baseline_keys if s == seed]
        if baseline_key:
            res = results_dict[baseline_key[0]]
            ep = res["eval_points"]
            probing_set = res.get("probing_set", [])

            # Plot individual model accuracies
            for k in range(n_models):
                lw = 3.0 if k in probing_set else 1.5
                ax.plot(ep, res["model_acc"][:, k],
                       label=f"Model {k}" if i==0 else None,
                       color=colors[k], linewidth=lw)

            # Plot Update_all average
            ax.plot(ep, res["model_acc_full"].mean(axis=1),
                   label="Update_all (avg)" if i==0 else None,
                   color='black', ls="--", lw=2, alpha=0.7)

            ax.set_title(f"Seed {seed}, p=0 (no probing)")
            ax.grid(alpha=0.3)
            ax.set_ylim([0.0, 0.82])

            if i == n_rows - 1:
                ax.set_xlabel("Iteration")
            ax.set_ylabel("Accuracy")

        # Remaining columns: p=0.5 with different kappa values
        for j, kappa in enumerate(kappa_values):
            if n_rows == 1 and n_cols == 1:
                ax = axes
            elif n_rows == 1:
                ax = axes[j + 1]
            elif n_cols == 1:
                ax = axes[i]
            else:
                ax = axes[i, j + 1]

            # Find result for this (seed, p=0.5, kappa)
            key = (seed, 0.5, kappa)
            if key in results_dict:
                res = results_dict[key]
                ep = res["eval_points"]
                probing_set = res.get("probing_set", [])

                # Plot individual model accuracies
                for k in range(n_models):
                    lw = 3.0 if k in probing_set else 1.5
                    ax.plot(ep, res["model_acc"][:, k],
                           label=f"Model {k}" if i==0 else None,
                           color=colors[k], linewidth=lw)

                # Plot Update_all average
                ax.plot(ep, res["model_acc_full"].mean(axis=1),
                       label="Update_all (avg)" if i==0 else None,
                       color='black', ls="--", lw=2, alpha=0.7)

                ax.set_title(f"Seed {seed}, p=0.5, κ={kappa}")
                ax.grid(alpha=0.3)
                ax.set_ylim([0.0, 0.82])

                if i == n_rows - 1:
                    ax.set_xlabel("Iteration")
                if j == 0:
                    ax.set_ylabel("Accuracy")

    # Create a single legend for the entire figure
    if n_rows == 1 and n_cols == 1:
        first_ax = axes
    elif n_rows == 1 or n_cols == 1:
        first_ax = axes[0]
    else:
        first_ax = axes[0, 0]

    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=n_models+1, fontsize=10)

    # Add title to the figure
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    plt.savefig("census_individual_model_accuracies_by_kappa.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_assignment_fraction(results_dict, args, title=None, max_iter=None):
    """
    Plots the fraction of users assigned to each learner over iterations.

    Args:
        results_dict: Dictionary of results with (seed, p, kappa) as keys
        args: Configuration object
        title: Optional title for the figure
        max_iter: Maximum iteration to plot (defaults to args.max_plot_iterations)
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    max_iter = max_iter or getattr(args, 'max_plot_iterations', None)
    p_values = sorted({p for (_, p, k) in results_dict.keys() if k == 0.0})
    seeds = sorted({seed for (seed, _, k) in results_dict.keys() if k == 0.0})
    n_models = results_dict[list(results_dict.keys())[0]]["n"]
    colors = plt.cm.Dark2(np.linspace(0, 1, n_models))

    fig, axes = plt.subplots(1, len(p_values), figsize=(6.5*len(p_values), 4.5), sharey=True)
    if len(p_values) == 1:
        axes = [axes]

    for j, p in enumerate(p_values):
        ax = axes[j]
        p_results = [results_dict[(seed, p, 0.0)] for seed in seeds if (seed, p, 0.0) in results_dict]
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
        for k in range(n_models):
            ax.plot(iters, mean_frac[:, k], color=colors[k], linewidth=2, label=f"Learner {k}")
            ax.fill_between(iters, mean_frac[:, k] - stderr_frac[:, k], mean_frac[:, k] + stderr_frac[:, k], color=colors[k], alpha=0.15)
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
    plt.savefig("census_model_assignment_fraction.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_user_assignment_map(results_dict, X_train, *, p_value, kappa=0.0, seed=None, title=None, method='pca'):
    """
    Plots a 2D visualization of which users are assigned to which learner.

    Args:
        results_dict: Dictionary of results with (seed, p, kappa) as keys
        X_train: Training features for dimensionality reduction
        p_value: The p value to plot
        kappa: The kappa value to plot (default: 0.0)
        seed: Specific seed to plot (default: first available seed)
        title: Optional title for the figure
        method: Dimensionality reduction method ('pca' or 'tsne')
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    seeds = sorted({s for (s, p, k) in results_dict.keys() if p == p_value and k == kappa})
    if not seeds:
        print(f"No runs found for p={p_value}, kappa={kappa}")
        return
    seed = seeds[0] if seed is None else seed
    key = (seed, p_value, kappa)
    res = results_dict.get(key)
    if res is None:
        print(f"Missing results for seed={seed}, p={p_value}")
        return
    n_models = res["n"]
    assignments = res["user_assignment_counts"].argmax(axis=1)

    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=0)
    else:
        reducer = PCA(n_components=2, random_state=0)
    reduced = reducer.fit_transform(X_train)

    colors = {'highlight': '#006ad1', 'background': '#d0d0d0'}
    fig, axes = plt.subplots(1, n_models, figsize=(3.8*n_models, 4), sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        mask = assignments == i
        ax.scatter(reduced[~mask, 0], reduced[~mask, 1], c=colors['background'], s=6, alpha=0.25, linewidths=0)
        ax.scatter(reduced[mask, 0], reduced[mask, 1], c=colors['highlight'], s=6, alpha=0.85, linewidths=0)
        ax.set_title(f"Learner {i}")
        if i == 0:
            ax.set_ylabel("PC 2" if method.lower() == 'pca' else 'Dim 2')
        ax.set_xlabel("PC 1" if method.lower() == 'pca' else 'Dim 1')
        ax.grid(alpha=0.2)

    if title:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"census_user_assignment_map_p{p_value}.png", dpi=300, bbox_inches='tight')
    plt.show()
