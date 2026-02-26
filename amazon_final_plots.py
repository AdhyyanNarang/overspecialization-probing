"""
Amazon Reviews MSGD Experiments - Main Script

This script performs multi-learner strategic gradient descent (MSGD) experiments
on the Amazon Reviews Multi dataset with binary sentiment classification.

Binary classification task: stars 1-3 → negative (0), stars 4-5 → positive (1)
Uses all-MiniLM-L6-v2 embeddings (384 dimensions)
Reuses MSGD algorithm from Census experiments (identical logistic loss)
"""

import numpy as np
import random
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import sys

# Add utils to path
sys.path.append("./utils")

# Reuse Census MSGD implementation (same loss function)
from utils_msgd_census import (
    MSGD_census_with_probing,
    logistic_loss,
    all_models_with_ensemble_population_loss_census,
    initialize_theta_gd,
    initialize_theta_from_erm_sklearn
)

# Reuse clustering utilities
from utils_clustering import (
    cluster_users_majority_minority,
    create_rankings_from_clusters,
    create_rankings_from_labels,
    visualize_individual_clusters_2d
)


@dataclass
class AmazonConfig:
    """Configuration for Amazon reviews MSGD experiments."""
    # Dataset selection
    dataset_source: str = 'mcauley_multi'  # 'amazon_reviews_multi', 'mcauley_multi', or 'legacy'
    dataset_config: str = 'en'                    # e.g., 'en' for amazon_reviews_multi
    dataset_split: str = 'train'
    dataset_revision: str = None                  # Revision (optional, usually not needed)
    legacy_dataset_name: str = 'McAuley-Lab/Amazon-Reviews-2023'
    legacy_dataset_config: str = 'raw_review_All_Beauty'
    # Categories for mcauley_multi source (loads multiple categories to enable category-based ranking)
    mcauley_categories: list = None  # If None, uses default set of categories
    mcauley_category_counts: dict = None  # Explicit counts per category, e.g. {'Electronics': 3000, ...}
                                          # If None, samples proportionally from natural distribution

    # Data parameters
    max_reviews: int = 50000              # Subsample size
    test_size: float = 0.01               # 99/1 split (like Census)
    random_state: int = 0                 # Data split seed

    # Embedding parameters
    embedding_model: str = 'all-MiniLM-L6-v2'
    embedding_dim: int = 384              # Fixed for MiniLM-L6-v2
    embedding_batch_size: int = 256       # Batch size for embedding generation
    force_recompute_embeddings: bool = False

    # Model parameters (following Census)
    n: int = 5                            # Number of learners
    T: int = 4000                         # Training iterations
    eta: float = 0.01                     # Learning rate
    num_sample: int = 150                 # Minibatch size
    reg_lambda: float = 1e-9              # L2 regularization
    lr_schedule: str = 'constant'         # Learning rate schedule

    # Baseline options
    add_intercept_feature: bool = False   # Append a 1.0 bias feature to inputs
    fit_intercept: bool = True
    run_sgd_baseline: bool = True
    sgd_max_iter: int = 1000
    sgd_eta: float = None                 # Defaults to eta / num_sample if None
    sgd_lr_schedule: str = 'constant'     # 'constant' or 'sqrt'

    # Clustering parameters
    clustering_method: str = 'kmeans'     # 'kmeans' or 'category'
    clustering_mode: str = 'no_majority'  # Start with balanced clusters

    # Category-based ranking parameters
    category_field: str = None            # Dataset field to use (auto-detect if None)
    category_groups: dict = None          # Optional mapping: group_name -> list of categories
    category_other_name: str = 'other'    # Name for the catch-all bucket
    category_group_names: list = None     # Populated at runtime for category-based rankings
    category_to_group: dict = None        # Populated at runtime for category-based rankings

    # Probing parameters
    N_probe: int = 500                    # Offline probe dataset size
    probe_num_samples: int = 150          # Probe batch size
    probing_set: list = None              # Default: [2]
    p_values: list = None                 # Default: [0, 0.2, 0.5, 1.0]
    kappa_values: list = None             # Default: [0.0]
    tau: float = 0.3                      # Ranking weight

    # Experiment parameters
    init_method: str = 'erm'              # 'erm' or 'random'
    plot_seeds_separately: bool = False
    plot_distance_separately: bool = False

    # Cache paths
    cache_dir: str = 'cache/amazon'
    embeddings_cache_dir: str = 'cache/amazon/embeddings'
    models_cache_dir: str = 'cache/amazon/models'

    # Output
    max_plot_iterations: int = 4000
    force_recompute_bar_theta: bool = False

    def __post_init__(self):
        if self.probing_set is None:
            self.probing_set = [2]
        if self.p_values is None:
            self.p_values = [0, 0.2, 0.5, 1.0]
        if self.kappa_values is None:
            self.kappa_values = [0.0]
        if self.mcauley_categories is None:
            # Default set of categories for mcauley_multi source
            # Using smaller categories (<2GB) for faster loading
            self.mcauley_categories = [
                'All_Beauty',              # 327 MB
                'Gift_Cards',              # 50 MB
                'Magazine_Subscriptions',  # 33 MB
                'Digital_Music',           # 79 MB
                'Musical_Instruments',     # 1.56 GB
                'Handmade_Products',       # 289 MB
                'Software',                # 1.87 GB
                'Industrial_and_Scientific',  # 2.35 GB (largest default)
                'Subscription_Boxes',      # 9 MB
            ]

        # Create cache directories
        Path(self.embeddings_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_cache_dir).mkdir(parents=True, exist_ok=True)


def load_and_preprocess_amazon_data(config):
    """
    Load Amazon reviews and generate embeddings.

    Steps:
    1. Check cache for embeddings
    2. If not cached:
       - Load from HuggingFace: amazon_reviews_multi, language='en'
       - Subsample to max_reviews
       - Generate embeddings with SentenceTransformer
       - Cache embeddings + metadata
    3. Normalize with StandardScaler
    4. Train-test split (stratified on labels)

    Parameters
    ----------
    config : AmazonConfig
        Configuration object

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, X_original_train, X_original_test,
        categories_train, categories_test
    """
    def normalize_category_value(value):
        if value is None:
            return "unknown"
        if isinstance(value, (list, tuple)):
            if not value:
                return "unknown"
            return " > ".join(str(v) for v in value if v is not None)
        return str(value)

    def infer_category_field(dataset):
        if config.category_field and config.category_field in dataset.features:
            return config.category_field
        if config.category_field and config.category_field not in dataset.features:
            print(f"⚠ Category field '{config.category_field}' not found in dataset features.")

        candidates = [
            "category",
            "categories",
            "main_category",
            "product_category",
            "category_name",
            "category_path",
            "product_type",
            "product_group",
        ]
        for candidate in candidates:
            if candidate in dataset.features:
                return candidate
        return None

    def extract_categories(dataset):
        category_field = infer_category_field(dataset)
        if category_field:
            raw = dataset[category_field]
            categories = [normalize_category_value(v) for v in raw]
        else:
            config_name = getattr(dataset.info, "config_name", None)
            builder_name = getattr(dataset.info, "builder_name", None) or "unknown"
            fallback = config_name or builder_name
            categories = [fallback] * len(dataset)
        return categories, category_field

    def cache_prefix():
        if config.dataset_source == 'legacy':
            return f"amazon_en_{config.max_reviews}_seed{config.random_state}"
        if config.dataset_source == 'mcauley_multi':
            n_cats = len(config.mcauley_categories)
            return f"mcauley_multi_{n_cats}cats_{config.max_reviews}_seed{config.random_state}"
        source = config.dataset_source.replace('/', '_')
        config_name = (config.dataset_config or "default").replace('/', '_')
        return f"{source}_{config_name}_{config.max_reviews}_seed{config.random_state}"

    cache_name = cache_prefix()
    embedding_cache_path = Path(config.embeddings_cache_dir) / f"{cache_name}.npy"
    metadata_cache_path = Path(config.embeddings_cache_dir) / "metadata" / f"{cache_name}.json"

    # Load embeddings from cache or generate
    if embedding_cache_path.exists() and not config.force_recompute_embeddings:
        print(f"Loading cached embeddings from {embedding_cache_path}")
        embeddings = np.load(embedding_cache_path)

        # Load metadata (labels, etc.)
        with open(metadata_cache_path) as f:
            metadata = json.load(f)
        labels = np.array(metadata['labels'])
        categories = metadata.get('categories')
        category_field = metadata.get('category_field')
        if category_field:
            config.category_field = category_field

        print(f"Loaded {len(labels)} reviews from cache")
    else:
        print("Loading Amazon reviews from HuggingFace...")
        if config.dataset_source == 'mcauley_multi':
            # Load from multiple McAuley categories via JSONL files (streaming)
            print(f"Loading from McAuley-Lab/Amazon-Reviews-2023 with {len(config.mcauley_categories)} categories...")
            print("Using streaming to load JSONL files from HuggingFace...")

            import pandas as pd

            # Base URL for JSONL files
            base_url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories"

            all_reviews = []
            all_ratings = []
            all_categories = []

            # If explicit counts provided, use them; otherwise we'll do proportional after loading
            if config.mcauley_category_counts is not None:
                samples_per_category = config.mcauley_category_counts
                print(f"Using explicit category counts")
            else:
                # Equal sampling initially, will be adjusted if needed
                samples_per_category = {cat: config.max_reviews // len(config.mcauley_categories)
                                        for cat in config.mcauley_categories}

            for category in config.mcauley_categories:
                n_samples = samples_per_category.get(category, config.max_reviews // len(config.mcauley_categories))
                url = f"{base_url}/{category}.jsonl"
                print(f"  Loading {category} ({n_samples} samples)...", end=" ")

                try:
                    # Use pandas to read JSONL with nrows limit (efficient streaming)
                    # Read more than needed to allow for random sampling
                    read_limit = min(n_samples * 3, 50000)  # Read up to 3x or 50k rows
                    df = pd.read_json(url, lines=True, nrows=read_limit)

                    # Subsample randomly
                    if len(df) > n_samples:
                        df = df.sample(n=n_samples, random_state=config.random_state)

                    # Extract text and rating
                    texts = df['text'].tolist()
                    ratings = df['rating'].tolist()

                    all_reviews.extend(texts)
                    all_ratings.extend(ratings)
                    all_categories.extend([category] * len(texts))

                    print(f"loaded {len(texts)} reviews")

                except Exception as e:
                    print(f"Warning: Failed to load {category}: {e}")
                    continue

            if not all_reviews:
                raise ValueError("Failed to load any reviews from McAuley categories")

            # Convert to arrays
            reviews = all_reviews
            labels = (np.array(all_ratings) >= 4.0).astype(int)
            categories = all_categories
            category_field = 'product_category'
            config.category_field = category_field

            print(f"\nTotal reviews loaded: {len(reviews)}")
            print(f"Categories: {list(set(categories))}")

        elif config.dataset_source == 'amazon_reviews_multi':
            # Try loading from mteb version (more reliable)
            try:
                print("Trying mteb/amazon_reviews_multi...")
                dataset = load_dataset(
                    "mteb/amazon_reviews_multi",
                    config.dataset_config,
                    split=config.dataset_split
                )
            except Exception as e1:
                print(f"mteb version failed: {e1}")
                # Try original without revision
                try:
                    print("Trying original amazon_reviews_multi...")
                    load_kwargs = {
                        "path": "amazon_reviews_multi",
                        "name": config.dataset_config,
                        "split": config.dataset_split
                    }
                    if config.dataset_revision:
                        load_kwargs["revision"] = config.dataset_revision
                    dataset = load_dataset(**load_kwargs)
                except Exception as e2:
                    raise ValueError(
                        f"Failed to load amazon_reviews_multi dataset. "
                        f"Consider using dataset_source='mcauley_multi' for category-based rankings. "
                        f"Errors: mteb: {e1}, original: {e2}"
                    )
        elif config.dataset_source == 'legacy':
            # Fallback to amazon_polarity (no category metadata)
            print("Loading amazon_polarity dataset...")
            dataset = load_dataset("amazon_polarity", split='train')
        else:
            raise ValueError(f"Unknown dataset_source: {config.dataset_source}")

        # For mcauley_multi, reviews/labels/categories are already set above
        # For other sources, we need to extract them from the dataset
        if config.dataset_source != 'mcauley_multi':
            print(f"Dataset size: {len(dataset)} reviews")

            # Subsample if needed
            if len(dataset) > config.max_reviews:
                print(f"Subsampling {config.max_reviews} reviews from {len(dataset)}...")
                indices = np.random.RandomState(config.random_state).choice(
                    len(dataset), size=config.max_reviews, replace=False
                )
                dataset = dataset.select(indices.tolist())

            # Extract reviews, labels, and categories (handle different dataset formats)
            if config.dataset_source == 'amazon_reviews_multi':
                if 'text' in dataset.features:
                    reviews = dataset['text']
                elif 'review_body' in dataset.features:
                    reviews = dataset['review_body']
                elif 'review_title' in dataset.features:
                    reviews = dataset['review_title']
                else:
                    raise ValueError("No text field found in amazon_reviews_multi dataset.")

                # Handle label field (mteb uses 'label', original uses 'stars')
                if 'label' in dataset.features:
                    # mteb version: label is 0-4 (representing 1-5 stars)
                    label_vals = np.array(dataset['label'])
                    labels = (label_vals >= 3).astype(int)  # 0-2 -> negative, 3-4 -> positive
                elif 'stars' in dataset.features:
                    stars = np.array(dataset['stars'])
                    labels = (stars >= 4).astype(int)
                else:
                    raise ValueError("No label/stars field found in amazon_reviews_multi dataset.")

                # Check for product_category (may not exist in mteb version)
                if 'product_category' in dataset.features:
                    category_field = 'product_category'
                    categories = [normalize_category_value(v) for v in dataset[category_field]]
                    config.category_field = category_field
                else:
                    print("Warning: product_category not found in dataset. Using 'unknown' for all reviews.")
                    print("Consider using dataset_source='mcauley_multi' for category-based rankings.")
                    categories = ["unknown"] * len(labels)
                    category_field = None
            else:
                if 'text' in dataset.features:  # McAuley format
                    reviews = dataset['text']
                    ratings = np.array(dataset['rating'])
                    labels = (ratings >= 4.0).astype(int)  # Binary: 0 = negative (1-3), 1 = positive (4-5)
                elif 'content' in dataset.features:  # amazon_polarity format
                    reviews = dataset['content']
                    labels = np.array(dataset['label'])  # Already binary (0/1)
                else:  # Original format (if it works)
                    reviews = dataset['review_body']
                    stars = np.array(dataset['stars'])
                    labels = (stars >= 4).astype(int)  # Binary: 0 = negative (1-3), 1 = positive (4-5)

                categories, category_field = extract_categories(dataset)
                if category_field:
                    config.category_field = category_field

        print(f"\nLabel distribution:")
        print(f"  Negative (0): {np.sum(labels == 0)} ({100*np.mean(labels == 0):.1f}%)")
        print(f"  Positive (1): {np.sum(labels == 1)} ({100*np.mean(labels == 1):.1f}%)")

        # Generate embeddings
        print(f"\nGenerating embeddings using {config.embedding_model}...")
        model = SentenceTransformer(config.embedding_model)

        # Check for GPU (CUDA or Apple MPS)
        import torch
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("Using CUDA GPU for embedding generation")
        elif torch.backends.mps.is_available():
            model = model.to('mps')
            print("Using Apple Silicon GPU (MPS) for embedding generation")

        embeddings = model.encode(
            reviews,
            batch_size=config.embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"Generated embeddings with shape: {embeddings.shape}")

        # Cache embeddings and metadata
        embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embedding_cache_path, embeddings)

        # Get dataset info if available (not available for mcauley_multi)
        dataset_name = None
        dataset_config_name = None
        try:
            dataset_name = getattr(dataset.info, "builder_name", None)
            dataset_config_name = getattr(dataset.info, "config_name", None)
        except (NameError, UnboundLocalError):
            pass  # dataset variable doesn't exist for mcauley_multi

        with open(metadata_cache_path, 'w') as f:
            json.dump({
                'labels': labels.tolist(),
                'categories': categories,
                'category_field': category_field,
                'dataset_name': dataset_name,
                'dataset_config': dataset_config_name,
                'dataset_source': config.dataset_source,
                'dataset_source_config': config.dataset_config,
                'n_samples': len(labels),
                'embedding_model': config.embedding_model,
                'max_reviews': config.max_reviews,
                'random_state': config.random_state
            }, f)
        print(f"Cached embeddings to {embedding_cache_path}")

    if categories is not None and len(categories) != len(labels):
        raise ValueError("Category list length does not match labels length.")
    if categories is None:
        if config.clustering_method == 'category':
            raise ValueError(
                "Category-based ranking requested, but categories are missing in cache metadata. "
                "Regenerate embeddings with force_recompute_embeddings=True."
            )
        categories = ["unknown"] * len(labels)

    # Save original embeddings for clustering
    X_original = embeddings.copy()

    # Normalize embeddings (following Census pattern)
    print("\nNormalizing embeddings...")
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)

    # Train-test split
    categories = np.array(categories, dtype=object)

    X_train, X_test, y_train, y_test, X_original_train, X_original_test, categories_train, categories_test = \
        train_test_split(
            embeddings_normalized, labels, X_original, categories,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=labels  # Maintain label balance
        )

    print(f"\nTrain set: {X_train.shape[0]} reviews")
    print(f"Test set: {X_test.shape[0]} reviews")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Train labels - Negative: {np.sum(y_train == 0)}, Positive: {np.sum(y_train == 1)}")
    print(f"Test labels - Negative: {np.sum(y_test == 0)}, Positive: {np.sum(y_test == 1)}")

    return X_train, X_test, y_train, y_test, X_original_train, X_original_test, categories_train, categories_test


def train_baseline_lr(X_train, y_train, X_test, y_test, reg_lambda, fit_intercept=True):
    """
    Train baseline logistic regression on full dataset.

    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    reg_lambda : L2 regularization parameter
    fit_intercept : Whether to fit an intercept term

    Returns
    -------
    float
        Test accuracy of baseline model
    """
    print("\nTraining baseline logistic regression on full dataset...")

    # Train with L2 regularization (C = 1/lambda)
    C = 1.0 / reg_lambda if reg_lambda > 0 else 1e9
    lr = LogisticRegression(C=C, max_iter=1000, random_state=0, fit_intercept=fit_intercept)
    lr.fit(X_train, y_train)

    train_acc = lr.score(X_train, y_train)
    test_acc = lr.score(X_test, y_test)

    print(f"Full dataset LR - Train accuracy: {train_acc:.4f}")
    print(f"Full dataset LR - Test accuracy: {test_acc:.4f}")
    print(f"\nThis will serve as the reference accuracy (upper bound): {test_acc:.4f}")

    return test_acc


def train_baseline_sgd(
    X_train, y_train, X_test, y_test, reg_lambda, eta, lr_schedule,
    max_iter=1000, fit_intercept=True
):
    """
    Train SGD baseline logistic regression to mirror MSGD optimization settings.

    Note: SGDClassifier applies per-sample updates; use a learning rate scaled
    by batch size (eta / num_sample) for closer parity with MSGD.
    """
    print("\nTraining baseline SGD (logistic loss) on full dataset...")

    if lr_schedule == 'constant':
        learning_rate = 'constant'
        power_t = 0.5
    elif lr_schedule == 'sqrt':
        learning_rate = 'invscaling'
        power_t = 0.5
    else:
        raise ValueError(f"Unknown sgd_lr_schedule '{lr_schedule}'.")

    clf = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=reg_lambda,
        learning_rate=learning_rate,
        eta0=eta,
        power_t=power_t,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=None,
        random_state=0
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"SGD baseline - Train accuracy: {train_acc:.4f}")
    print(f"SGD baseline - Test accuracy: {test_acc:.4f}")

    return test_acc


def add_intercept_feature(X):
    """Append a bias feature to the input matrix."""
    intercept = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.hstack([X, intercept])

def get_probing_mode(config):
    """
    Map clustering configuration to a probing mode supported by MSGD.
    """
    if config.clustering_method == 'category':
        return 'no_majority'
    if config.clustering_mode in {'single_majority', 'half_majority', 'no_majority'}:
        return config.clustering_mode
    return 'no_majority'

def assign_subpopulation_labels(X_original, categories, config, kmeans):
    """
    Assign subpopulation labels for evaluation.
    """
    if config.clustering_method == 'category':
        if config.category_to_group is None or config.category_group_names is None:
            raise ValueError("Category mapping not available. Run perform_clustering first.")
        mapping = config.category_to_group
        other_idx = mapping.get(config.category_other_name)
        labels = []
        for category in categories:
            if category in mapping:
                labels.append(mapping[category])
            elif other_idx is not None:
                labels.append(other_idx)
            else:
                raise ValueError(f"Unknown category '{category}' with no '{config.category_other_name}' group.")
        return np.array(labels, dtype=int), list(config.category_group_names)

    from scipy.spatial.distance import cdist
    distances = cdist(X_original, kmeans.cluster_centers_)
    labels = np.argmin(distances, axis=1)
    group_names = [f"cluster_{i}" for i in range(kmeans.cluster_centers_.shape[0])]
    return labels, group_names


def compute_cross_generalization_matrix(Theta, X_test, y_test, subpop_labels, n_groups):
    """
    Compute M_ij = accuracy of Theta[i] on subpopulation j.
    """
    preds = Theta @ X_test.T
    preds_binary = (preds > 0).astype(int)
    y_test = y_test.astype(int)

    n_models = Theta.shape[0]
    matrix = np.full((n_models, n_groups), np.nan, dtype=float)

    for j in range(n_groups):
        mask = (subpop_labels == j)
        if not np.any(mask):
            continue
        y_group = y_test[mask]
        preds_group = preds_binary[:, mask]
        matrix[:, j] = (preds_group == y_group).mean(axis=1)

    return matrix


def print_cross_generalization_matrix(matrix, group_names):
    """
    Pretty-print the cross-generalization matrix.
    """
    model_names = [f"model_{i}" for i in range(matrix.shape[0])]
    try:
        import pandas as pd
        df = pd.DataFrame(matrix, index=model_names, columns=group_names)
        print(df.round(4).to_string())
    except Exception:
        header = " " * 12 + " ".join(f"{name:>10}" for name in group_names)
        print(header)
        for i, row in enumerate(matrix):
            values = " ".join(f"{val:10.4f}" if not np.isnan(val) else " " * 10 for val in row)
            print(f"{model_names[i]:<12}{values}")


def perform_clustering(X_train, X_original_train, config, categories_train=None):
    """
    Perform user clustering and create rankings.

    Parameters
    ----------
    X_train : Normalized embeddings for training
    X_original_train : Original (unnormalized) embeddings
    config : AmazonConfig

    Returns
    -------
    tuple
        (cluster_labels, kmeans, rankings)
    """
    print("\nPerforming clustering...")

    if config.clustering_method == 'category':
        if categories_train is None:
            raise ValueError("categories_train is required for category-based ranking.")
        categories_train = np.asarray(categories_train, dtype=object)
        category_counts = Counter(categories_train)

        if config.category_groups:
            group_names = list(config.category_groups.keys())
            category_to_group = {}
            for idx, name in enumerate(group_names):
                for category in config.category_groups[name]:
                    category_to_group[category] = idx
            unknown = [c for c in category_counts if c not in category_to_group]
            if unknown:
                category_to_group.update({c: len(group_names) for c in unknown})
                group_names.append(config.category_other_name)
            if len(group_names) != config.n:
                print(f"Adjusting config.n from {config.n} to {len(group_names)} to match category groups.")
                config.n = len(group_names)
        else:
            if config.n <= 1:
                group_names = [config.category_other_name]
                category_to_group = {category: 0 for category in category_counts}
                config.n = 1
            elif len(category_counts) > config.n:
                top_categories = [
                    category for category, _ in category_counts.most_common(config.n - 1)
                ]
                group_names = top_categories + [config.category_other_name]
                category_to_group = {category: idx for idx, category in enumerate(top_categories)}
                category_to_group.update(
                    {category: len(group_names) - 1 for category in category_counts if category not in category_to_group}
                )
            else:
                group_names = [category for category, _ in category_counts.most_common()]
                if len(group_names) != config.n:
                    print(f"Adjusting config.n from {config.n} to {len(group_names)} to match categories.")
                    config.n = len(group_names)
                category_to_group = {category: idx for idx, category in enumerate(group_names)}

        cluster_labels = np.array([category_to_group[c] for c in categories_train], dtype=int)
        config.category_group_names = list(group_names)
        config.category_to_group = dict(category_to_group)
        cluster_centers = np.zeros((config.n, X_original_train.shape[1]))
        for idx, name in enumerate(group_names):
            mask = cluster_labels == idx
            if not np.any(mask):
                raise ValueError(f"No samples for category group '{name}'")
            cluster_centers[idx] = X_original_train[mask].mean(axis=0)

        class DummyKMeans:
            def __init__(self, centers):
                self.cluster_centers_ = centers

        kmeans = DummyKMeans(cluster_centers)
        rankings = create_rankings_from_labels(cluster_labels, config.n)
        config.clustering_mode = 'category'

        print(f"Category field: {config.category_field or 'auto'}")
        print(f"Category groups: {group_names}")
        print("Category distribution:")
        for idx, name in enumerate(group_names):
            count = np.sum(cluster_labels == idx)
            pct = 100 * count / len(cluster_labels)
            print(f"  {name}: {count} ({pct:.1f}%)")
    else:
        # Use original embeddings for clustering (not normalized)
        cluster_labels, kmeans = cluster_users_majority_minority(
            X_original_train,
            n_clusters=config.n,
            majority_percentage=0.80,
            mode=config.clustering_mode
        )
        rankings = create_rankings_from_clusters(
            X_original_train, kmeans.cluster_centers_, config.n, cluster_labels
        )

    print(f"Clustering mode: {config.clustering_mode}")
    print(f"Cluster distribution: {np.bincount(cluster_labels)}")

    return cluster_labels, kmeans, rankings


def run_msgd_experiments(
    X_train,
    y_train,
    X_test,
    y_test,
    rankings,
    bar_Theta,
    config,
    num_seeds=15,
    store_trajectories=True,
    initial_results=None,
    checkpoint_every=0,
    checkpoint_fn=None,
):
    """
    Run MSGD experiments for multiple seeds and p values.

    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    rankings : User rankings of learners
    bar_Theta : Initialized parameters
    config : AmazonConfig
    num_seeds : Number of random seeds to run
    store_trajectories : If False, do not keep Theta trajectories in results_dict
        (useful for long runs / batch mode to avoid memory blow-up)
    initial_results : Optional dict of existing {(seed, p, kappa): result} to resume from
    checkpoint_every : Save checkpoint every N new runs (0 disables periodic checkpointing)
    checkpoint_fn : Callable invoked as checkpoint_fn(results_dict, meta) when checkpointing

    Returns
    -------
    dict
        Results dictionary with (seed, p, kappa) as keys
    """
    results_dict = {} if initial_results is None else dict(initial_results)
    n_features = X_train.shape[1]
    new_runs = 0

    for seed in range(num_seeds):
        print(f"\n{'='*80}")
        print(f"Processing seed {seed}")
        print(f"{'='*80}")

        for p in config.p_values:
            # For p=0, ranking noise has no effect because probing_set is empty.
            # Run baseline once at kappa=0.0 to avoid redundant work.
            kappa_list = [0.0] if p == 0 else config.kappa_values
            for kappa in kappa_list:
                run_key = (seed, p, kappa)
                if run_key in results_dict:
                    print(f"  Skipping existing run seed={seed}, p={p}, kappa={kappa}")
                    continue

                print(f"  Running with probing_p={p}, ranking_noise={kappa}")
                probing_mode = get_probing_mode(config)

                # Set random seeds
                random.seed(seed)
                np.random.seed(seed)

                # Use bar_Theta initialization
                if config.init_method == 'erm':
                    Theta_init = bar_Theta.copy()
                    print("    Using ERM initialization (bar_Theta)")
                else:
                    Theta_init = np.random.randn(config.n, n_features) * 0.01
                    print("    Using random initialization")

                # Run MSGD with probing (import from utils_msgd_census)
                result = MSGD_census_with_probing(
                    X_train, Theta_init, config.n, y_train,
                    config.T, logistic_loss, config.eta,
                    rankings=rankings,
                    tau=config.tau,
                    rankings_only=False,
                    probing_set=config.probing_set if p > 0 else [],
                    probing_p=p,
                    N_probe=config.N_probe,
                    probe_num_samples=config.probe_num_samples,
                    mode=probing_mode,
                    ranking_noise=kappa,
                    reg_lambda=config.reg_lambda,
                    num_sample=config.num_sample,
                    lr_schedule=config.lr_schedule,
                    seed=seed
                )

                # Extract trajectories
                Theta_traj = result['Theta_traj']  # (n, d, T)
                Theta_full_traj = result['Update_all_Theta_traj']  # (n, d, T)
                chosen_models = result['chosen_models']  # (T, num_sample)

                # Compute assignment fractions
                T_iters = chosen_models.shape[0]
                assignment_fraction = np.zeros((T_iters, config.n))
                for learner_idx in range(config.n):
                    mask = (chosen_models == learner_idx)
                    learner_counts = mask.sum(axis=1)
                    assignment_fraction[:, learner_idx] = learner_counts / chosen_models.shape[1]

                # Extract user assignment counts from MSGD result
                user_assignment_counts = result["user_assignment_counts"]

                # Evaluate on test set
                model_losses, ensemble_losses, eval_points, model_acc, ensemble_acc = \
                    all_models_with_ensemble_population_loss_census(
                        X_test, config.n, Theta_traj, y_test, logistic_loss, jump=100, return_accuracy=True
                    )

                model_losses_full, ensemble_losses_full, _, model_acc_full, ensemble_acc_full = \
                    all_models_with_ensemble_population_loss_census(
                        X_test, config.n, Theta_full_traj, y_test, logistic_loss, jump=100, return_accuracy=True
                    )

                # Compute distances from initialization
                distances_from_erm = np.zeros((len(eval_points), config.n))
                for i, t in enumerate(eval_points):
                    for j in range(config.n):
                        distances_from_erm[i, j] = np.linalg.norm(
                            Theta_traj[j, :, t] - bar_Theta[j]
                        )

                # Store results
                run_result = {
                    "init_seed": seed,
                    "probing_p": p,
                    "ranking_noise": kappa,
                    "n": config.n,
                    "model_losses": model_losses,
                    "model_acc": model_acc,
                    "ensemble_losses": ensemble_losses,
                    "ensemble_acc": ensemble_acc,
                    "model_losses_full": model_losses_full,
                    "model_acc_full": model_acc_full,
                    "ensemble_losses_full": ensemble_losses_full,
                    "ensemble_acc_full": ensemble_acc_full,
                    "eval_points": eval_points,
                    "chosen_models": chosen_models,
                    "assignment_fraction": assignment_fraction,
                    "user_assignment_counts": user_assignment_counts,
                    "distances_from_erm": distances_from_erm,
                    "probing_set": config.probing_set
                }
                if store_trajectories:
                    run_result["Theta"] = Theta_traj
                    run_result["Theta_full"] = Theta_full_traj
                results_dict[run_key] = run_result
                new_runs += 1

                # Print final accuracies
                final_acc = model_acc[-1]
                print(f"    Final accuracies: {final_acc}")

                # Explicitly release large trajectory tensors in slim mode.
                if not store_trajectories:
                    del Theta_traj, Theta_full_traj, result

                if checkpoint_fn is not None and checkpoint_every > 0 and new_runs % checkpoint_every == 0:
                    checkpoint_fn(
                        results_dict,
                        {"seed": seed, "p": p, "kappa": kappa, "new_runs": new_runs},
                    )

    if checkpoint_fn is not None and new_runs > 0:
        checkpoint_fn(results_dict, {"final": True, "new_runs": new_runs})

    print("\nTraining complete!")
    return results_dict


def print_final_accuracies(results_dict, baseline_acc, baseline_sgd_acc=None):
    """
    Print final accuracies for all configurations.

    Parameters
    ----------
    results_dict : dict
        Results from run_msgd_experiments
    baseline_acc : float
        Baseline accuracy for reference
    baseline_sgd_acc : float or None
        SGD baseline accuracy if computed
    """
    print("\n" + "="*80)
    print("FINAL MODEL ACCURACIES")
    print("="*80)

    for (seed, p, kappa), result in sorted(results_dict.items()):
        final_acc = result['model_acc'][-1]
        final_acc_full = result['model_acc_full'][-1]
        final_ensemble = result['ensemble_acc'][-1]

        print(f"\nSeed {seed}, p={p}, κ={kappa}:")
        print(f"  Update_all: avg={final_acc_full.mean():.4f}, min={final_acc_full.min():.4f}, max={final_acc_full.max():.4f}")
        print(f"  MSGD:       avg={final_acc.mean():.4f}, min={final_acc.min():.4f}, max={final_acc.max():.4f}")
        print(f"  Ensemble:   {final_ensemble:.4f}")
        print(f"  Reference (Full LR): {baseline_acc:.4f}")
        if baseline_sgd_acc is not None:
            print(f"  Reference (SGD): {baseline_sgd_acc:.4f}")
        print(f"  Update_all vs Reference: {final_acc_full.mean() - baseline_acc:+.4f}")

    print("\n" + "="*80)


def main():
    """Main execution function."""
    # Create configuration
    config = AmazonConfig()

    print("="*80)
    print("AMAZON REVIEWS MSGD EXPERIMENTS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset source: {config.dataset_source}")
    print(f"  Dataset config: {config.dataset_config}")
    print(f"  Max reviews: {config.max_reviews}")
    print(f"  Models: {config.n}")
    print(f"  Iterations: {config.T}")
    print(f"  Learning rate: {config.eta}")
    print(f"  Batch size: {config.num_sample}")
    print(f"  Regularization: {config.reg_lambda}")
    print(f"  Probing dataset size: {config.N_probe}")
    print(f"  Probe batch size: {config.probe_num_samples}")
    print(f"  Probing set: {config.probing_set}")
    print(f"  p values: {config.p_values}")
    print(f"  Clustering method: {config.clustering_method}")
    print(f"  Clustering mode: {config.clustering_mode}")
    print(f"  Probing mode: {get_probing_mode(config)}")
    print(f"  Add intercept feature: {config.add_intercept_feature}")
    print(f"  Fit intercept (baseline): {config.fit_intercept}")
    print(f"  Run SGD baseline: {config.run_sgd_baseline}")
    print(f"  Initialization method: {config.init_method}")
    print()

    # 1. Load data and generate embeddings
    X_train, X_test, y_train, y_test, X_original_train, X_original_test, \
        categories_train, categories_test = load_and_preprocess_amazon_data(config)

    X_train_base, X_test_base = X_train, X_test
    if config.add_intercept_feature:
        X_train = add_intercept_feature(X_train_base)
        X_test = add_intercept_feature(X_test_base)

    n_features = X_train.shape[1]

    # 2. Train baseline
    effective_fit_intercept = config.fit_intercept and not config.add_intercept_feature
    if config.add_intercept_feature and config.fit_intercept:
        print("Note: add_intercept_feature=True, overriding fit_intercept to False for baselines.")
    FULL_LR_TEST_ACC = train_baseline_lr(
        X_train, y_train, X_test, y_test,
        reg_lambda=config.reg_lambda,
        fit_intercept=effective_fit_intercept
    )
    if config.run_sgd_baseline:
        sgd_eta = (config.eta / config.num_sample) if config.sgd_eta is None else config.sgd_eta
        FULL_SGD_TEST_ACC = train_baseline_sgd(
            X_train, y_train, X_test, y_test,
            reg_lambda=config.reg_lambda,
            eta=sgd_eta,
            lr_schedule=config.sgd_lr_schedule,
            max_iter=config.sgd_max_iter,
            fit_intercept=effective_fit_intercept
        )
    else:
        FULL_SGD_TEST_ACC = None

    # 3. Cluster and create rankings
    cluster_labels, kmeans, rankings = perform_clustering(
        X_train, X_original_train, config, categories_train=categories_train
    )

    # 4. Initialize bar_Theta using gradient descent (caching handled internally)
    print("\n" + "="*80)
    print("Initializing bar_Theta using GD with constant learning rate")
    print("="*80)

    bar_Theta = initialize_theta_gd(
        X_train, y_train, rankings, config.n, n_features,
        T=config.T, eta=config.eta, reg_lambda=config.reg_lambda,
        clustering_mode=config.clustering_mode,
        cache_dir=config.models_cache_dir,
        force_recompute=config.force_recompute_bar_theta,
        return_diagnostics=False
    )

    print("="*80)
    print("bar_Theta (GD constant lr) test accuracies:")
    test_preds = bar_Theta @ X_test.T
    test_accs = ((test_preds > 0) == y_test).mean(axis=1)
    print(f"  {test_accs}")
    print(f"  Mean: {test_accs.mean():.4f}, Best: {test_accs.max():.4f}, Worst: {test_accs.min():.4f}")
    print("="*80)

    # Cross-generalization matrix for bar_Theta
    try:
        subpop_labels_test, group_names = assign_subpopulation_labels(
            X_original_test, categories_test, config, kmeans
        )
        cross_gen = compute_cross_generalization_matrix(
            bar_Theta, X_test, y_test, subpop_labels_test, len(group_names)
        )
        print("\n" + "="*80)
        print("CROSS-GENERALIZATION (bar_Theta vs subpopulations)")
        print("="*80)
        print_cross_generalization_matrix(cross_gen, group_names)
        print("="*80)
    except Exception as exc:
        print(f"Warning: could not compute cross-generalization matrix: {exc}")

    # 5. Run MSGD experiments
    num_seeds = 15  # Number of random initializations to average
    results_dict = run_msgd_experiments(
        X_train, y_train, X_test, y_test, rankings, bar_Theta, config, num_seeds=num_seeds
    )

    # 6. Print final results
    print_final_accuracies(results_dict, FULL_LR_TEST_ACC, baseline_sgd_acc=FULL_SGD_TEST_ACC)

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("Results stored in results_dict variable")
    print("Use plotting functions to visualize results")
    print("="*80)

    return results_dict, config


if __name__ == "__main__":
    results_dict, config = main()
