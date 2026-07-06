from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
_CACHE_ROOT = Path(tempfile.gettempdir()) / "uai_code_mpl_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import numpy as np
import yaml

from experiment_io import find_latest_results_dir


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "final_figs"
mpl = None
plt = None

MODEL_COLORS = [
    (0.890, 0.102, 0.110),  # red
    (0.216, 0.494, 0.722),  # blue
    (0.302, 0.686, 0.290),  # green
    (0.596, 0.306, 0.639),  # purple
    (1.000, 0.498, 0.000),  # orange
    (1.000, 0.765, 0.000),  # gold
    (0.651, 0.337, 0.157),  # brown
    (0.969, 0.506, 0.749),  # pink
    (0.600, 0.600, 0.600),  # gray
    (0.094, 0.745, 0.804),  # cyan
]
KAPPA_LINESTYLES = ["-", "--", "-.", ":"]


def _set_style() -> None:
    _ensure_matplotlib()
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )


def _ensure_matplotlib() -> None:
    global mpl, plt
    if plt is not None:
        return
    import matplotlib as _mpl
    import matplotlib.pyplot as _plt

    mpl = _mpl
    plt = _plt


def _read_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {config_path}")
    config = raw.get("config")
    if isinstance(config, dict):
        return config
    return {k: v for k, v in raw.items() if k not in {"dataset", "runner"}}


def _load_latest_results(config_path: Path, results_root: Path) -> dict[Any, dict[str, Any]]:
    latest_dir = find_latest_results_dir(config_path, results_root=results_root)
    if latest_dir is None:
        raise FileNotFoundError(
            f"No results found for {config_path}. Run the experiment before generating figures."
        )

    results_path = latest_dir / "results.pkl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.pkl in {latest_dir}")

    with open(results_path, "rb") as f:
        payload = pickle.load(f)

    runs = payload.get("runs") if isinstance(payload, dict) else None
    if runs:
        if len(runs) != 1:
            raise ValueError(
                f"{results_path} contains {len(runs)} sweep runs; expected one run for figure generation."
            )
        return runs[0]["results"]
    return payload


def _normalise_keys(results: dict[Any, dict[str, Any]]) -> dict[tuple[int, float, float], dict[str, Any]]:
    normalised = {}
    for key, value in results.items():
        if len(key) == 2:
            seed, p = key
            kappa = 0.0
        elif len(key) == 3:
            seed, p, kappa = key
        else:
            raise ValueError(f"Unexpected result key shape: {key}")
        normalised[(int(seed), float(p), float(kappa))] = value
    return normalised


def _colors(n_models: int) -> list[tuple[float, float, float]]:
    return [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(n_models)]


def _stderr(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=0) / np.sqrt(len(values)))


def _first_baseline(results: dict[tuple[int, float, float], dict[str, Any]], key: str) -> float | None:
    for result in results.values():
        value = result.get(key)
        if value is not None:
            return float(value)
    return None


def _plot_final_metric_vs_p(
    ax: plt.Axes,
    results_raw: dict[Any, dict[str, Any]],
    config: dict[str, Any],
    metric_key: str,
    title: str,
    ylabel: str,
    baseline: float | None = None,
) -> tuple[list[Any], list[str]]:
    results = _normalise_keys(results_raw)
    all_keys = sorted(results)
    p_values = sorted({p for (_, p, _) in all_keys})
    kappa_values = sorted({kappa for (_, _, kappa) in all_keys})
    n_models = int(results[all_keys[0]]["n"])
    probing_set = set(config.get("probing_set", config.get("fixed_probing_set", [])) or [])
    colors = _colors(n_models)

    handles: list[Any] = []
    labels: list[str] = []

    for learner_idx in range(n_models):
        if learner_idx in probing_set:
            kappas_to_plot = kappa_values
        else:
            kappas_to_plot = [0.0 if 0.0 in kappa_values else kappa_values[0]]

        for kappa_idx, kappa in enumerate(kappas_to_plot):
            xs = []
            means = []
            errs = []
            for p in p_values:
                values = []
                for (seed, key_p, key_kappa), result in results.items():
                    if key_p == p and key_kappa == kappa:
                        values.append(float(np.asarray(result[metric_key])[-1, learner_idx]))
                if values:
                    arr = np.asarray(values, dtype=float)
                    xs.append(p)
                    means.append(float(np.mean(arr)))
                    errs.append(_stderr(arr))

            if not xs:
                continue

            label = (
                f"Learner {learner_idx} (kappa={kappa:g})"
                if learner_idx in probing_set and len(kappas_to_plot) > 1
                else f"Learner {learner_idx}"
            )
            linestyle = (
                KAPPA_LINESTYLES[kappa_idx % len(KAPPA_LINESTYLES)]
                if learner_idx in probing_set
                else "-"
            )
            marker = "^" if learner_idx in probing_set else "o"
            line = ax.plot(
                xs,
                means,
                color=colors[learner_idx],
                linestyle=linestyle,
                linewidth=1.6,
                marker=marker,
                markersize=3.5,
                label=label,
            )[0]
            ax.fill_between(
                xs,
                np.asarray(means) - np.asarray(errs),
                np.asarray(means) + np.asarray(errs),
                color=colors[learner_idx],
                alpha=0.16,
                linewidth=0,
            )
            handles.append(line)
            labels.append(label)

    if baseline is not None and p_values:
        baseline_line = ax.hlines(
            baseline,
            xmin=min(p_values),
            xmax=max(p_values),
            color="black",
            linewidth=1.1,
            linestyle="--",
            label="Pooled baseline",
        )
        handles.append(baseline_line)
        labels.append("Pooled baseline")

    ax.set_title(title)
    ax.set_xlabel("Probing Weight (p)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return handles, labels


def _plot_accuracy_over_time(
    ax: plt.Axes,
    results_raw: dict[Any, dict[str, Any]],
    title: str,
    baseline: float | None,
) -> tuple[list[Any], list[str]]:
    results = _normalise_keys(results_raw)
    p0_results = [result for (seed, p, kappa), result in sorted(results.items()) if p == 0.0 and kappa == 0.0]
    if not p0_results:
        raise ValueError("No p=0, kappa=0 Census NN bad-outcome results found.")

    eval_points = np.asarray(p0_results[0]["eval_points"])
    acc = np.asarray([result["model_acc"] for result in p0_results], dtype=float)
    mean_acc = np.mean(acc, axis=0)
    err_acc = np.std(acc, axis=0, ddof=0) / np.sqrt(acc.shape[0])
    n_models = mean_acc.shape[1]
    colors = _colors(n_models)

    handles: list[Any] = []
    labels: list[str] = []
    for learner_idx in range(n_models):
        label = f"Learner {learner_idx}"
        line = ax.plot(
            eval_points,
            mean_acc[:, learner_idx],
            color=colors[learner_idx],
            linewidth=1.8,
            marker="o",
            markersize=2.5,
            markevery=max(1, len(eval_points) // 8),
            label=label,
        )[0]
        ax.fill_between(
            eval_points,
            mean_acc[:, learner_idx] - err_acc[:, learner_idx],
            mean_acc[:, learner_idx] + err_acc[:, learner_idx],
            color=colors[learner_idx],
            alpha=0.16,
            linewidth=0,
        )
        handles.append(line)
        labels.append(label)

    if baseline is not None:
        baseline_line = ax.hlines(
            baseline,
            xmin=float(eval_points.min()),
            xmax=float(eval_points.max()),
            color="black",
            linewidth=1.1,
            linestyle="--",
            label="Pooled MLP baseline",
        )
        handles.append(baseline_line)
        labels.append("Pooled MLP baseline")

    y_values = [mean_acc - err_acc, mean_acc + err_acc]
    if baseline is not None:
        y_values.append(np.asarray([baseline]))
    y_min = float(np.min([np.min(v) for v in y_values]))
    y_max = float(np.max([np.max(v) for v in y_values]))
    pad = max(0.02, 0.08 * (y_max - y_min))
    ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.grid(True, which="major", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return handles, labels


def _dedupe_legend(handles: list[Any], labels: list[str]) -> tuple[list[Any], list[str]]:
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    return unique_handles, unique_labels


def generate_good_kappa(results_root: Path, output_dir: Path) -> Path:
    configs = [
        ("Census", SCRIPT_DIR / "configs" / "census_good_kappa.yaml", "model_acc", "Final Accuracy"),
        ("Amazon", SCRIPT_DIR / "configs" / "amazon_good_kappa.yaml", "model_acc", "Final Accuracy"),
        ("MovieLens", SCRIPT_DIR / "configs" / "movielens_good_kappa.yaml", "model_losses", "Final MSE"),
    ]
    loaded = [
        (title, _load_latest_results(config_path, results_root), _read_config(config_path), metric_key, ylabel)
        for title, config_path, metric_key, ylabel in configs
    ]

    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))
    all_handles: list[Any] = []
    all_labels: list[str] = []

    for ax, (title, results, config, metric_key, ylabel) in zip(axes, loaded):
        handles, labels = _plot_final_metric_vs_p(
            ax,
            results,
            config,
            metric_key=metric_key,
            title=title,
            ylabel=ylabel,
        )
        all_handles.extend(handles)
        all_labels.extend(labels)

    handles, labels = _dedupe_legend(all_handles, all_labels)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(6, len(labels)),
        frameon=True,
        fontsize=6.5,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.34, wspace=0.32)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "good_kappa.pdf"
    fig.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return output_path


def generate_census_nn(results_root: Path, output_dir: Path) -> Path:
    bad_config = SCRIPT_DIR / "configs" / "census_nn_bad.yaml"
    good_config = SCRIPT_DIR / "configs" / "census_nn_good.yaml"
    bad_results = _normalise_keys(_load_latest_results(bad_config, results_root))
    good_results_raw = _load_latest_results(good_config, results_root)
    good_config_dict = _read_config(good_config)

    baseline = _first_baseline(bad_results, "baseline_mlp_acc")
    if baseline is None:
        baseline = _first_baseline(_normalise_keys(good_results_raw), "baseline_mlp_acc")

    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    handles_left, labels_left = _plot_accuracy_over_time(
        axes[0],
        bad_results,
        title="Random initialization, no probing",
        baseline=baseline,
    )
    handles_right, labels_right = _plot_final_metric_vs_p(
        axes[1],
        good_results_raw,
        good_config_dict,
        metric_key="model_acc",
        title="Partition-pretrained initialization with probing",
        ylabel="Final Accuracy",
        baseline=baseline,
    )

    handles, labels = _dedupe_legend(handles_left + handles_right, labels_left + labels_right)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(6, len(labels)),
        frameon=True,
        fontsize=7,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28, wspace=0.28)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "census_nn_camera_ready_joint.pdf"
    fig.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate camera-ready UAI figure PDFs.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--which",
        choices=["all", "good-kappa", "census-nn"],
        default="all",
        help="Which camera-ready figure to generate.",
    )
    args = parser.parse_args()

    outputs = []
    if args.which in {"all", "good-kappa"}:
        outputs.append(generate_good_kappa(args.results_root, args.output_dir))
    if args.which in {"all", "census-nn"}:
        outputs.append(generate_census_nn(args.results_root, args.output_dir))

    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
