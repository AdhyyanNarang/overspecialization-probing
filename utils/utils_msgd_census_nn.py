from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only when torch is missing
    torch = None
    nn = None
    F = None


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError(
            "The census_nn experiment path requires PyTorch. Install `torch` to use "
            "`dataset: census_nn`."
        )


if nn is not None:

    class TwoLayerCensusMLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            activation: str = "relu",
        ) -> None:
            super().__init__()
            if activation != "relu":
                raise ValueError(f"Unsupported activation '{activation}'. Expected 'relu'.")
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, 1)
            self.activation_name = activation

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            hidden = F.relu(self.hidden(x))
            return self.output(hidden).squeeze(-1)

else:

    class TwoLayerCensusMLP:  # pragma: no cover - used only without torch installed
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def resolve_device(device_str: str) -> "torch.device":
    _require_torch()
    normalized = device_str.lower()
    mps_backend = getattr(torch.backends, "mps", None)
    has_mps = bool(mps_backend is not None and mps_backend.is_available())
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
        return torch.device("cuda")
    if normalized == "mps":
        if not has_mps:
            raise RuntimeError("Requested device 'mps' but Apple MPS is not available.")
        return torch.device("mps")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if has_mps:
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unknown device '{device_str}'. Expected cpu, cuda, mps, or auto.")


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover - older torch versions
            torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def _compute_lr(eta: float, t: int, schedule: str) -> float:
    if schedule == "sqrt":
        return eta / np.sqrt(t + 1)
    if schedule == "constant":
        return eta
    raise ValueError(f"Unknown lr_schedule '{schedule}'. Expected 'constant' or 'sqrt'.")


def _to_tensor(data: np.ndarray, device: "torch.device") -> "torch.Tensor":
    _require_torch()
    return torch.as_tensor(np.asarray(data), dtype=torch.float32, device=device)


def _regularization_term(model: "TwoLayerCensusMLP", reg_lambda: float) -> "torch.Tensor":
    if reg_lambda <= 0:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return 0.5 * reg_lambda * sum(param.pow(2).sum() for param in model.parameters())


def _set_optimizer_lr(optimizer: "torch.optim.Optimizer", lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _binary_mean_loss(logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
    return F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="mean")


def make_random_models(
    n_models: int,
    input_dim: int,
    hidden_dim: int,
    activation: str,
    device: "torch.device",
    seed: int | None = None,
) -> List["TwoLayerCensusMLP"]:
    _require_torch()
    if seed is not None:
        set_all_seeds(seed)
    return [
        TwoLayerCensusMLP(input_dim=input_dim, hidden_dim=hidden_dim, activation=activation).to(device)
        for _ in range(n_models)
    ]


def clone_models(models: Sequence["TwoLayerCensusMLP"]) -> List["TwoLayerCensusMLP"]:
    return [copy.deepcopy(model) for model in models]


def flatten_model_params(model: "TwoLayerCensusMLP") -> "torch.Tensor":
    return torch.cat([param.detach().reshape(-1) for param in model.parameters()])


def stack_logits(
    models: Sequence["TwoLayerCensusMLP"],
    x_batch: "torch.Tensor",
) -> "torch.Tensor":
    logits = [model(x_batch) for model in models]
    return torch.stack(logits, dim=0)


def binary_loss_matrix(logits: "torch.Tensor", y_batch: "torch.Tensor") -> "torch.Tensor":
    target_matrix = y_batch.float().unsqueeze(0).expand_as(logits)
    return F.binary_cross_entropy_with_logits(logits, target_matrix, reduction="none")


def generate_probe_batch_labels(
    models: Sequence["TwoLayerCensusMLP"],
    probe_x: "torch.Tensor",
    probe_rankings: "torch.Tensor",
    mode: str,
    ranking_noise: float,
) -> "torch.Tensor":
    with torch.no_grad():
        logits = stack_logits(models, probe_x)
        if mode == "single_majority":
            selected_logits = logits[0]
        elif mode == "half_majority":
            selected_logits = torch.median(logits, dim=0).values
        elif mode == "no_majority":
            top_ranked = torch.argmin(probe_rankings, dim=1)
            selected = top_ranked.clone()
            if ranking_noise > 0:
                noisy_mask = torch.rand(probe_x.shape[0], device=probe_x.device) < ranking_noise
                noisy_count = int(noisy_mask.sum().item())
                if noisy_count > 0:
                    random_other = torch.randint(
                        0,
                        len(models) - 1,
                        (noisy_count,),
                        device=probe_x.device,
                    )
                    current = selected[noisy_mask]
                    selected[noisy_mask] = random_other + (random_other >= current).long()
            per_user_logits = logits.transpose(0, 1)
            selected_logits = per_user_logits[
                torch.arange(probe_x.shape[0], device=probe_x.device),
                selected,
            ]
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Must be 'single_majority', 'half_majority', or 'no_majority'."
            )
        return (selected_logits > 0).float()


def _train_model_on_dataset(
    model: "TwoLayerCensusMLP",
    X_data: np.ndarray,
    y_data: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    reg_lambda: float,
    seed: int,
    device: "torch.device",
) -> None:
    if len(X_data) == 0:
        return
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    X_np = np.asarray(X_data, dtype=np.float32)
    y_np = np.asarray(y_data, dtype=np.float32)
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        permutation = rng.permutation(len(X_np))
        for start in range(0, len(X_np), batch_size):
            batch_idx = permutation[start : start + batch_size]
            x_batch = _to_tensor(X_np[batch_idx], device)
            y_batch = _to_tensor(y_np[batch_idx], device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = _binary_mean_loss(logits, y_batch) + _regularization_term(model, reg_lambda)
            loss.backward()
            optimizer.step()


def _evaluate_accuracy(
    model: "TwoLayerCensusMLP",
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    device: "torch.device",
) -> float:
    x_tensor = _to_tensor(X_eval, device)
    y_tensor = _to_tensor(y_eval, device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)
        acc = ((logits > 0) == y_tensor.bool()).float().mean().item()
    model.train(was_training)
    return float(acc)


def train_pooled_mlp_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Any,
    seed: int = 0,
) -> tuple["TwoLayerCensusMLP", float]:
    device = resolve_device(config.device)
    model = make_random_models(
        1,
        input_dim=X_train.shape[1],
        hidden_dim=config.hidden_dim,
        activation=config.activation,
        device=device,
        seed=seed,
    )[0]
    _train_model_on_dataset(
        model,
        X_train,
        y_train,
        epochs=config.baseline_epochs,
        batch_size=config.baseline_batch_size,
        lr=config.baseline_lr,
        reg_lambda=config.reg_lambda,
        seed=seed,
        device=device,
    )
    test_acc = _evaluate_accuracy(model, X_test, y_test, device)
    return model, test_acc


def pretrain_partition_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rankings: np.ndarray,
    config: Any,
) -> List["TwoLayerCensusMLP"]:
    _require_torch()
    device = resolve_device(config.device)
    input_dim = X_train.shape[1]
    cache_dir = Path(config.models_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"partition_pretrain_mode={config.clustering_mode}_n{config.n}_d{input_dim}_"
        f"h{config.hidden_dim}_act={config.activation}_reg{config.reg_lambda}_"
        f"epochs{config.pretrain_epochs}_lr{config.pretrain_lr}_train{len(X_train)}.pt"
    )
    cache_path = cache_dir / cache_name

    if cache_path.exists() and not config.force_recompute_bar_models:
        payload = torch.load(cache_path, map_location=device)
        models = make_random_models(
            config.n,
            input_dim,
            config.hidden_dim,
            config.activation,
            device,
            seed=0,
        )
        for model, state_dict in zip(models, payload["state_dicts"]):
            model.load_state_dict(state_dict)
        print(f"Loading partition-pretrained models from {cache_path}")
        return models

    dataset_name = getattr(config, "dataset_name", config.__class__.__name__)
    print("=" * 80)
    print(f"Pretraining partition models for {dataset_name}")
    print("=" * 80)
    models = make_random_models(
        config.n,
        input_dim,
        config.hidden_dim,
        config.activation,
        device,
        seed=0,
    )

    for learner_idx, model in enumerate(models):
        partition_indices = np.where(rankings[:, learner_idx] == 0)[0]
        if len(partition_indices) == 0:
            print(
                f"Warning: learner {learner_idx} has no top-ranked users. "
                "Leaving this model randomly initialized."
            )
            continue
        X_partition = X_train[partition_indices]
        y_partition = y_train[partition_indices]
        print(
            f"Learner {learner_idx}: pretraining on {len(partition_indices)} users "
            f"(class dist: {np.bincount(y_partition.astype(int), minlength=2)})"
        )
        _train_model_on_dataset(
            model,
            X_partition,
            y_partition,
            epochs=config.pretrain_epochs,
            batch_size=config.pretrain_batch_size,
            lr=config.pretrain_lr,
            reg_lambda=config.reg_lambda,
            seed=learner_idx,
            device=device,
        )

    payload = {
        "state_dicts": [copy.deepcopy(model.state_dict()) for model in models],
        "meta": {
            "clustering_mode": config.clustering_mode,
            "n": config.n,
            "input_dim": input_dim,
            "hidden_dim": config.hidden_dim,
            "activation": config.activation,
            "reg_lambda": config.reg_lambda,
            "pretrain_epochs": config.pretrain_epochs,
            "pretrain_lr": config.pretrain_lr,
            "train_users": len(X_train),
        },
    }
    torch.save(payload, cache_path)
    print(f"Saved partition-pretrained models to {cache_path}")
    return models


def evaluate_models_census_nn(
    models: Sequence["TwoLayerCensusMLP"],
    models_full: Sequence["TwoLayerCensusMLP"],
    X_test: np.ndarray,
    y_test: np.ndarray,
    eval_step: int,
    results_accumulator: Dict[str, List[np.ndarray]],
    reference_models: Sequence["TwoLayerCensusMLP"],
) -> None:
    device = next(models[0].parameters()).device
    x_tensor = _to_tensor(X_test, device)
    y_tensor = _to_tensor(y_test, device)

    all_models = list(models) + list(models_full) + list(reference_models)
    original_modes = [model.training for model in all_models]
    for model in all_models:
        model.eval()

    with torch.no_grad():
        logits = stack_logits(models, x_tensor)
        logits_full = stack_logits(models_full, x_tensor)

        model_losses = binary_loss_matrix(logits, y_tensor).mean(dim=1).cpu().numpy()
        model_losses_full = binary_loss_matrix(logits_full, y_tensor).mean(dim=1).cpu().numpy()

        ensemble_logits = torch.median(logits, dim=0).values
        ensemble_logits_full = torch.median(logits_full, dim=0).values
        ensemble_loss = _binary_mean_loss(ensemble_logits, y_tensor).item()
        ensemble_loss_full = _binary_mean_loss(ensemble_logits_full, y_tensor).item()

        y_bool = y_tensor.bool().unsqueeze(0)
        model_acc = ((logits > 0) == y_bool).float().mean(dim=1).cpu().numpy()
        model_acc_full = ((logits_full > 0) == y_bool).float().mean(dim=1).cpu().numpy()
        ensemble_acc = ((ensemble_logits > 0) == y_tensor.bool()).float().mean().item()
        ensemble_acc_full = ((ensemble_logits_full > 0) == y_tensor.bool()).float().mean().item()

        reference_flat = [flatten_model_params(model).cpu() for model in reference_models]
        current_flat = [flatten_model_params(model).cpu() for model in models]
        distances = np.array(
            [(current_flat[i] - reference_flat[i]).norm().item() for i in range(len(models))],
            dtype=np.float32,
        )

    for model, original_mode in zip(all_models, original_modes):
        model.train(original_mode)

    results_accumulator["eval_points"].append(eval_step)
    results_accumulator["model_losses"].append(model_losses.astype(np.float32))
    results_accumulator["ensemble_losses"].append(np.float32(ensemble_loss))
    results_accumulator["model_acc"].append(model_acc.astype(np.float32))
    results_accumulator["ensemble_acc"].append(np.float32(ensemble_acc))
    results_accumulator["model_losses_full"].append(model_losses_full.astype(np.float32))
    results_accumulator["ensemble_losses_full"].append(np.float32(ensemble_loss_full))
    results_accumulator["model_acc_full"].append(model_acc_full.astype(np.float32))
    results_accumulator["ensemble_acc_full"].append(np.float32(ensemble_acc_full))
    results_accumulator["distances_from_erm"].append(distances)


def _make_results_accumulator() -> Dict[str, List[np.ndarray]]:
    return {
        "eval_points": [],
        "model_losses": [],
        "ensemble_losses": [],
        "model_acc": [],
        "ensemble_acc": [],
        "model_losses_full": [],
        "ensemble_losses_full": [],
        "model_acc_full": [],
        "ensemble_acc_full": [],
        "distances_from_erm": [],
    }


def _finalize_results_accumulator(results_accumulator: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {
        "eval_points": np.asarray(results_accumulator["eval_points"], dtype=np.int64),
        "model_losses": np.asarray(results_accumulator["model_losses"], dtype=np.float32),
        "ensemble_losses": np.asarray(results_accumulator["ensemble_losses"], dtype=np.float32),
        "model_acc": np.asarray(results_accumulator["model_acc"], dtype=np.float32),
        "ensemble_acc": np.asarray(results_accumulator["ensemble_acc"], dtype=np.float32),
        "model_losses_full": np.asarray(results_accumulator["model_losses_full"], dtype=np.float32),
        "ensemble_losses_full": np.asarray(results_accumulator["ensemble_losses_full"], dtype=np.float32),
        "model_acc_full": np.asarray(results_accumulator["model_acc_full"], dtype=np.float32),
        "ensemble_acc_full": np.asarray(results_accumulator["ensemble_acc_full"], dtype=np.float32),
        "distances_from_erm": np.asarray(results_accumulator["distances_from_erm"], dtype=np.float32),
    }


def run_msgd_census_nn_with_probing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rankings: np.ndarray,
    init_models: Sequence["TwoLayerCensusMLP"],
    config: Any,
    seed: int,
    probing_p: float,
    ranking_noise: float,
    store_snapshots: bool = False,
) -> Dict[str, Any]:
    del store_snapshots  # Neural trajectories are intentionally not serialized in v1.
    _require_torch()
    set_all_seeds(seed)
    device = resolve_device(config.device)
    num_users = X_train.shape[0]
    probing_set = set(config.probing_set) if probing_p > 0 and config.probing_set else set()

    X_train_tensor = _to_tensor(X_train, device)
    y_train_tensor = _to_tensor(y_train, device)
    rankings_tensor = torch.as_tensor(np.asarray(rankings), dtype=torch.long, device=device)

    models = clone_models(init_models)
    models_full = clone_models(init_models)
    reference_models = clone_models(init_models)
    for model in list(models) + list(models_full) + list(reference_models):
        model.to(device)
    for model in list(models) + list(models_full):
        model.train()
    for model in reference_models:
        model.eval()

    optimizers = [torch.optim.SGD(model.parameters(), lr=config.eta) for model in models]
    optimizers_full = [torch.optim.SGD(model.parameters(), lr=config.eta) for model in models_full]

    probe_datasets: Dict[int, Dict[str, torch.Tensor]] = {}
    if probing_set:
        print(f"Collecting offline probe datasets (N_probe={config.N_probe})...")
        for learner_idx in probing_set:
            probe_user_indices = np.random.choice(num_users, size=config.N_probe, replace=True)
            probe_x = X_train_tensor[probe_user_indices]
            probe_rankings = rankings_tensor[probe_user_indices]
            probe_mode = getattr(config, "probing_mode", config.clustering_mode)
            probe_y = generate_probe_batch_labels(
                reference_models,
                probe_x,
                probe_rankings,
                mode=probe_mode,
                ranking_noise=ranking_noise,
            )
            probe_datasets[learner_idx] = {
                "X": probe_x,
                "Y": probe_y,
                "user_indices": torch.as_tensor(
                    probe_user_indices,
                    dtype=torch.long,
                    device=device,
                ),
            }
            print(
                f"  Learner {learner_idx}: Collected {config.N_probe} probe samples "
                f"(label distribution: {np.bincount(probe_y.cpu().numpy().astype(int), minlength=2)})"
            )

    user_indices = np.random.choice(num_users, size=(config.T, config.num_sample), replace=True)
    chosen_models = np.zeros((config.T, config.num_sample), dtype=np.int64)
    user_assignment_counts = np.zeros((num_users, config.n), dtype=np.int64)
    assignment_counts = np.zeros((config.T, config.n), dtype=np.int64)
    assignment_fraction = np.zeros((config.T, config.n), dtype=np.float32)
    results_accumulator = _make_results_accumulator()

    evaluate_models_census_nn(
        models,
        models_full,
        X_test,
        y_test,
        eval_step=0,
        results_accumulator=results_accumulator,
        reference_models=reference_models,
    )

    for t in range(config.T):
        lr = _compute_lr(config.eta, t, config.lr_schedule)
        for optimizer in list(optimizers) + list(optimizers_full):
            _set_optimizer_lr(optimizer, lr)

        batch_users = user_indices[t]
        x_batch = X_train_tensor[batch_users]
        y_batch = y_train_tensor[batch_users]
        ranking_batch = rankings_tensor[batch_users]

        with torch.no_grad():
            logits = stack_logits(models, x_batch)
            loss_matrix = binary_loss_matrix(logits, y_batch)
            rank_choices = torch.argmin(ranking_batch, dim=1)
            if getattr(config, "rankings_only", False):
                model_id = rank_choices
            else:
                loss_choices = torch.argmin(loss_matrix, dim=0)
                if config.tau <= 0:
                    model_id = loss_choices
                elif config.tau >= 1:
                    model_id = rank_choices
                else:
                    pick_rank = torch.rand(config.num_sample, device=device) < config.tau
                    model_id = torch.where(pick_rank, rank_choices, loss_choices)

        model_id_np = model_id.cpu().numpy()
        chosen_models[t] = model_id_np
        np.add.at(user_assignment_counts, (batch_users, model_id_np), 1)
        for learner_idx in range(config.n):
            learner_count = int(np.sum(model_id_np == learner_idx))
            assignment_counts[t, learner_idx] = learner_count
            assignment_fraction[t, learner_idx] = learner_count / config.num_sample

        for learner_idx, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad()
            assigned_mask = model_id == learner_idx
            if torch.any(assigned_mask):
                task_logits = model(x_batch[assigned_mask])
                task_loss = _binary_mean_loss(task_logits, y_batch[assigned_mask])
            else:
                task_loss = torch.zeros((), device=device)

            combined_loss = task_loss
            if learner_idx in probing_set:
                probe_data = probe_datasets[learner_idx]
                probe_indices = np.random.choice(
                    config.N_probe,
                    size=config.probe_num_samples,
                    replace=True,
                )
                probe_indices = torch.as_tensor(probe_indices, dtype=torch.long, device=device)
                probe_x_batch = probe_data["X"][probe_indices]
                probe_y_batch = probe_data["Y"][probe_indices]
                probe_logits = model(probe_x_batch)
                probe_loss = _binary_mean_loss(probe_logits, probe_y_batch)
                combined_loss = (task_loss + probing_p * probe_loss) / (1 + probing_p)

            total_loss = combined_loss + _regularization_term(model, config.reg_lambda)
            total_loss.backward()
            optimizer.step()

        for model_full, optimizer_full in zip(models_full, optimizers_full):
            optimizer_full.zero_grad()
            full_logits = model_full(x_batch)
            full_loss = _binary_mean_loss(full_logits, y_batch) + _regularization_term(
                model_full,
                config.reg_lambda,
            )
            full_loss.backward()
            optimizer_full.step()

        should_eval = ((t + 1) % config.eval_every == 0) or (t == config.T - 1)
        if should_eval:
            evaluate_models_census_nn(
                models,
                models_full,
                X_test,
                y_test,
                eval_step=t + 1,
                results_accumulator=results_accumulator,
                reference_models=reference_models,
            )

    final_results = _finalize_results_accumulator(results_accumulator)
    final_results.update(
        {
            "chosen_models": chosen_models,
            "assignment_fraction": assignment_fraction,
            "assignment_counts": assignment_counts,
            "user_assignment_counts": user_assignment_counts,
        }
    )
    return final_results
