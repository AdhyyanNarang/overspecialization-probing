import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist

def cluster_users(user_embeddings, n_clusters=3, random_state=42):
    """
    Cluster users based on their embeddings.
    
    Parameters:
    - user_embeddings: User embedding matrix
    - n_clusters: Number of clusters to create
    - random_state: Random seed for reproducibility
    
    Returns:
    - cluster_labels: Cluster assignment for each user
    - kmeans: The fitted KMeans model (contains cluster centers)
    """
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(user_embeddings)
    
    return cluster_labels, kmeans

def cluster_users_with_good_learners(user_embeddings, n_clusters=3, random_state=42):
    """
    Cluster users with a mix of random and k-means clustering.
    
    Parameters:
    - user_embeddings: User embedding matrix
    - n_clusters: Number of clusters (must be odd)
    - random_state: Random seed for reproducibility
    
    Returns:
    - cluster_labels: Cluster assignment for each user
    - kmeans_info: A dummy KMeans object with cluster_centers_ attribute for visualization
    """
    assert n_clusters % 2 == 1, "n_clusters must be odd"
    
    # Calculate h = (n_clusters - 1)/2
    h = (n_clusters - 1) // 2
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Calculate sizes for the two chunks
    n_samples = user_embeddings.shape[0]
    first_chunk_size = int(n_samples * ((h + 1) / n_clusters))
    second_chunk_size = n_samples - first_chunk_size
    
    # Randomly shuffle indices
    indices = np.random.permutation(n_samples)
    first_chunk_indices = indices[:first_chunk_size]
    second_chunk_indices = indices[first_chunk_size:]
    
    # Initialize final cluster labels array
    cluster_labels = np.zeros(n_samples, dtype=int)
    
    # First chunk: Random assignment to h+1 clusters
    first_chunk_labels = np.random.randint(0, h + 1, size=first_chunk_size)
    cluster_labels[first_chunk_indices] = first_chunk_labels
    
    # Initialize cluster centers array
    n_features = user_embeddings.shape[1]
    all_centers = np.zeros((n_clusters, n_features))
    
    # Calculate centroids for randomly assigned clusters
    for i in range(h + 1):
        mask = cluster_labels == i
        if np.any(mask):
            all_centers[i] = user_embeddings[mask].mean(axis=0)
    
    # Second chunk: K-means clustering for h clusters
    if second_chunk_size > 0:
        second_chunk_data = user_embeddings[second_chunk_indices]
        kmeans = KMeans(n_clusters=h, random_state=random_state, n_init=10)
        second_chunk_labels = kmeans.fit_predict(second_chunk_data)
        # Offset the labels by h+1 to avoid overlap with first chunk
        cluster_labels[second_chunk_indices] = second_chunk_labels + (h + 1)
        # Store the cluster centers
        all_centers[h+1:] = kmeans.cluster_centers_
    
    # Create a dummy KMeans object to store all centers
    class DummyKMeans:
        def __init__(self, centers):
            self.cluster_centers_ = centers
    
    kmeans_info = DummyKMeans(all_centers)
    
    return cluster_labels, kmeans_info

def cluster_users_majority_minority(user_embeddings, n_clusters=3, majority_percentage=0.90, random_state=42, mode='single_majority'):
    """
    This function can definitely be slimmed down a lot.
    """
    assert n_clusters >= 2, "n_clusters must be at least 2"
    assert 0 < majority_percentage < 1, "majority_percentage must be between 0 and 1"
    assert mode in ['single_majority', 'half_majority', 'no_majority'], "mode must be 'single_majority', 'half_majority', or 'no_majority'"

    # Set random seed for reproducibility
    np.random.seed(random_state)

    n_samples = user_embeddings.shape[0]
    n_features = user_embeddings.shape[1]

    # Initialize cluster labels and centers
    cluster_labels = np.zeros(n_samples, dtype=int)
    all_centers = np.zeros((n_clusters, n_features))

    # Randomly shuffle indices
    indices = np.random.permutation(n_samples)

    if mode == 'single_majority':
        # Original behavior: one cluster gets majority_percentage
        majority_size = int(n_samples * majority_percentage)
        minority_size = n_samples - majority_size

        majority_indices = indices[:majority_size]
        minority_indices = indices[majority_size:]

        # Majority chunk: All assigned to learner 0
        cluster_labels[majority_indices] = 0
        all_centers[0] = user_embeddings[majority_indices].mean(axis=0)

        # Minority chunk: K-means clustering for remaining clusters
        if minority_size > 0 and n_clusters > 1:
            minority_data = user_embeddings[minority_indices]
            n_minority_clusters = n_clusters - 1

            # Handle case where minority size is smaller than number of clusters
            if minority_size < n_minority_clusters:
                # If we have fewer minority users than clusters, assign them randomly
                minority_labels = np.random.randint(0, n_minority_clusters, size=minority_size)
                # Calculate centers for clusters that have users assigned
                for i in range(n_minority_clusters):
                    mask = minority_labels == i
                    if np.any(mask):
                        all_centers[i + 1] = minority_data[mask].mean(axis=0)
                    else:
                        # For empty clusters, use random point from minority data
                        all_centers[i + 1] = minority_data[np.random.randint(0, minority_size)]
            else:
                # Standard k-means clustering
                kmeans = KMeans(n_clusters=n_minority_clusters, random_state=random_state, n_init=10)
                minority_labels = kmeans.fit_predict(minority_data)
                # Store the cluster centers (offset by 1 since learner 0 is taken)
                all_centers[1:] = kmeans.cluster_centers_

            # Assign minority users to clusters 1, 2, ..., n_clusters-1
            cluster_labels[minority_indices] = minority_labels + 1

    elif mode == 'half_majority':
        # New behavior: ceil(n/2) clusters equally share majority_percentage
        import math
        n_majority_clusters = math.ceil(n_clusters / 2)
        n_minority_clusters = n_clusters - n_majority_clusters

        # Calculate sizes
        majority_size = int(n_samples * majority_percentage)
        minority_size = n_samples - majority_size

        majority_indices = indices[:majority_size]
        minority_indices = indices[majority_size:]

        # Assign majority users randomly to first n_majority_clusters
        # Size per majority cluster (divide equally among first ceil(n/2) clusters)
        size_per_majority_cluster = majority_size // n_majority_clusters

        current_idx = 0
        for i in range(n_majority_clusters):
            # Calculate size for this cluster (handle remainder in last cluster)
            if i == n_majority_clusters - 1:
                cluster_size = majority_size - (size_per_majority_cluster * i)
            else:
                cluster_size = size_per_majority_cluster

            cluster_indices = majority_indices[current_idx:current_idx + cluster_size]
            cluster_labels[cluster_indices] = i
            all_centers[i] = user_embeddings[cluster_indices].mean(axis=0)
            current_idx += cluster_size

        # Use KMeans to cluster minority users into remaining clusters
        if minority_size > 0 and n_minority_clusters > 0:
            minority_data = user_embeddings[minority_indices]

            # Handle case where minority size is smaller than number of clusters
            if minority_size < n_minority_clusters:
                # If we have fewer minority users than clusters, assign them randomly
                minority_labels = np.random.randint(0, n_minority_clusters, size=minority_size)
                # Calculate centers for clusters that have users assigned
                for i in range(n_minority_clusters):
                    mask = minority_labels == i
                    if np.any(mask):
                        all_centers[n_majority_clusters + i] = minority_data[mask].mean(axis=0)
                    else:
                        # For empty clusters, use random point from minority data
                        all_centers[n_majority_clusters + i] = minority_data[np.random.randint(0, minority_size)]
            else:
                # Standard k-means clustering
                kmeans = KMeans(n_clusters=n_minority_clusters, random_state=random_state, n_init=10)
                minority_labels = kmeans.fit_predict(minority_data)
                # Store the cluster centers (offset by n_majority_clusters)
                all_centers[n_majority_clusters:] = kmeans.cluster_centers_

            # Assign minority users to clusters n_majority_clusters, ..., n_clusters-1
            cluster_labels[minority_indices] = minority_labels + n_majority_clusters

    elif mode == 'no_majority':
        # Pure k-means clustering on all users (no majority/minority split)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(user_embeddings)
        all_centers = kmeans.cluster_centers_

    # Create a dummy KMeans object to store all centers
    class DummyKMeans:
        def __init__(self, centers):
            self.cluster_centers_ = centers

    kmeans_info = DummyKMeans(all_centers)

    return cluster_labels, kmeans_info

def visualize_clusters_2d(user_embeddings, cluster_labels, cluster_centers=None, method='pca'):
    """
    Visualize user clusters in 2D.
    
    Parameters:
    - user_embeddings: User embedding matrix
    - cluster_labels: Cluster assignment for each user
    - cluster_centers: Cluster centers (optional)
    - method: Dimensionality reduction method ('pca' or 'tsne')
    """
    # Reduce dimensionality for visualization
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        user_2d = reducer.fit_transform(user_embeddings)
        method_name = 'PCA'
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        user_2d = reducer.fit_transform(user_embeddings)
        method_name = 't-SNE'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    # Create a DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame({
        'x': user_2d[:, 0],
        'y': user_2d[:, 1],
        'cluster': cluster_labels
    })
    
    # Visualize the clusters
    plt.figure(figsize=(12, 10))
    
    # Plot with Seaborn for better aesthetics
    sns.scatterplot(x='x', y='y', hue='cluster', data=df, palette='viridis', 
                   s=100, alpha=0.7)
    
    # Add cluster centers if provided
    if cluster_centers is not None and method.lower() == 'pca':
        centers_2d = reducer.transform(cluster_centers)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], s=300, c='red', 
                   marker='X', edgecolor='black', label='Cluster Centers')
    
    plt.title(f'User Clusters Visualized with {method_name}', fontsize=16)
    plt.xlabel(f'{method_name} Dimension 1', fontsize=14)
    plt.ylabel(f'{method_name} Dimension 2', fontsize=14)
    plt.legend(title='Cluster', fontsize=12, title_fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'user_clusters_{method.lower()}.png', dpi=300)
    plt.show()
    
    # Create a second visualization showing the distribution of users in each cluster
    plt.figure(figsize=(10, 6))
    cluster_counts = df['cluster'].value_counts().sort_index()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
    plt.title('Number of Users in Each Cluster', fontsize=16)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Number of Users', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('cluster_distribution.png', dpi=300)
    plt.show()

def visualize_split_clusters_2d(user_embeddings, cluster_labels, cluster_centers, method='pca', random_state=42):
    """
    Visualize the two groups of clusters (random and k-means) separately in two subplots.
    
    Parameters:
    - user_embeddings: User embedding matrix
    - cluster_labels: Cluster assignments for each user
    - cluster_centers: Centers of all clusters
    - method: Dimensionality reduction method ('pca' or 't-sne')
    - random_state: Random seed for reproducibility
    """
    # Perform dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_data = reducer.fit_transform(user_embeddings)
        reduced_centers = reducer.transform(cluster_centers)
    elif method.lower() == 't-sne':
        reducer = TSNE(n_components=2, random_state=random_state)
        reduced_data = reducer.fit_transform(user_embeddings)
        reduced_centers = reducer.fit_transform(cluster_centers)
    
    # Calculate h based on total number of clusters
    n_clusters = len(np.unique(cluster_labels))
    h = (n_clusters - 1) // 2
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot first group (random clusters) - clusters 0 to h
    first_mask = cluster_labels <= h
    first_data = reduced_data[first_mask]
    first_labels = cluster_labels[first_mask]
    first_centers = reduced_centers[:h+1]
    
    scatter1 = ax1.scatter(first_data[:, 0], first_data[:, 1], 
                          c=first_labels, cmap='tab10', 
                          alpha=0.6)
    ax1.scatter(first_centers[:, 0], first_centers[:, 1], 
                c='black', marker='x', s=200, linewidths=3,
                label='Cluster Centers')
    ax1.set_title('Random Clusters')
    ax1.legend()
    
    # Plot second group (k-means clusters) - clusters h+1 to n_clusters-1
    second_mask = cluster_labels > h
    second_data = reduced_data[second_mask]
    second_labels = cluster_labels[second_mask] - (h + 1)  # Shift labels to start from 0
    second_centers = reduced_centers[h+1:]
    
    scatter2 = ax2.scatter(second_data[:, 0], second_data[:, 1], 
                          c=second_labels, cmap='tab10', 
                          alpha=0.6)
    ax2.scatter(second_centers[:, 0], second_centers[:, 1], 
                c='black', marker='x', s=200, linewidths=3,
                label='Cluster Centers')
    ax2.set_title('K-means Clusters')
    ax2.legend()
    
    # Add colorbars
    plt.colorbar(scatter1, ax=ax1, label='Cluster Label')
    plt.colorbar(scatter2, ax=ax2, label='Cluster Label')
    
    plt.tight_layout()
    plt.show()

def create_rankings_from_clusters(user_embeddings, cluster_centers, n_models, cluster_labels):
    """
    Create rankings for each user based on their distance to cluster centers,
    ensuring that each user's assigned cluster is always ranked first.
    
    Parameters:
    - user_embeddings: User embedding matrix
    - cluster_centers: Cluster centers from KMeans
    - n_models: Number of models (should match number of clusters)
    - cluster_labels: Array of cluster assignments for each user
    
    Returns:
    - rankings: Array of shape (num_users, n_models) containing rankings
    """
    # Ensure we have the right number of clusters
    if len(cluster_centers) != n_models:
        raise ValueError(f"Number of clusters ({len(cluster_centers)}) must match number of models ({n_models})")
    
    # Calculate distances from each user to each cluster center
    distances = cdist(user_embeddings, cluster_centers)
    
    # Create a mask to ensure assigned clusters get lowest distance
    num_users = len(user_embeddings)
    min_distance = np.min(distances) - 1  # Ensure it's smaller than any actual distance
    
    # Set the distance to the assigned cluster to be the minimum
    for user_idx in range(num_users):
        assigned_cluster = cluster_labels[user_idx]
        distances[user_idx, assigned_cluster] = min_distance
    
    # Convert distances to rankings (smaller distance = better rank)
    # We'll use argsort to get the indices that would sort the distances
    # Then argsort again to convert positions to ranks (0 = closest, n-1 = furthest)
    rankings = np.argsort(np.argsort(distances, axis=1), axis=1)
    
    return rankings

def create_rankings_from_labels(cluster_labels, n_models):
    """
    Create rankings from pre-assigned cluster labels.

    Each user gets their assigned cluster ranked first; all other clusters follow
    in ascending cluster index order to keep the ranking deterministic.
    """
    cluster_labels = np.asarray(cluster_labels)
    if cluster_labels.ndim != 1:
        raise ValueError("cluster_labels must be a 1D array")
    if np.any(cluster_labels < 0) or np.any(cluster_labels >= n_models):
        raise ValueError("cluster_labels must be in [0, n_models-1]")

    rankings = np.empty((len(cluster_labels), n_models), dtype=int)
    base_order = np.arange(n_models)

    for i, assigned in enumerate(cluster_labels):
        order = np.concatenate(([assigned], base_order[base_order != assigned]))
        rankings[i, order] = np.arange(n_models)

    return rankings

def visualize_individual_clusters_2d(user_embeddings, cluster_labels, cluster_centers, method='pca', random_state=42):
    """
    Visualize all clusters on a single plot with different colors.
    
    Parameters:
    - user_embeddings: User embedding matrix
    - cluster_labels: Cluster assignments for each user
    - cluster_centers: Centers of all clusters
    - method: Dimensionality reduction method ('pca' or 't-sne')
    - random_state: Random seed for reproducibility
    """
    # Perform dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_data = reducer.fit_transform(user_embeddings)
        reduced_centers = reducer.transform(cluster_centers)
    elif method.lower() == 't-sne':
        reducer = TSNE(n_components=2, random_state=random_state)
        reduced_data = reducer.fit_transform(user_embeddings)
        reduced_centers = reducer.fit_transform(cluster_centers)
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Define colors in the specified order
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    # Create a single figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each cluster with a different color
    for i, cluster_idx in enumerate(unique_clusters):
        # Get data for this cluster
        mask = cluster_labels == cluster_idx
        cluster_data = reduced_data[mask]
        center = reduced_centers[cluster_idx].reshape(1, -1)
        
        # Use color from the list (cycle if more clusters than colors)
        color = colors[i % len(colors)]
        
        # Plot cluster data points
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                  c=color, alpha=0.6, s=50, label=f'Cluster {cluster_idx}')
        
        # Add cluster center with 'X' marker
        ax.scatter(center[:, 0], center[:, 1], 
                  c=color, marker='X', s=300, linewidths=2,
                  edgecolors='black')
    
    # Set title and labels
    ax.set_title('User Clusters Visualization (PCA)', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    
    # Add legend
    ax.legend(fontsize=10, loc='best')
    
    # Add gridlines
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_rankings_2d(user_embeddings, rankings, method='pca', random_state=42):
    """
    Visualize users grouped by their top-ranked learner in separate plots.
    
    Parameters:
    - user_embeddings: User embedding matrix
    - rankings: Array of shape (num_users, n_models) where each row contains rankings
    - method: Dimensionality reduction method ('pca' or 't-sne')
    - random_state: Random seed for reproducibility
    """
    # Perform dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_data = reducer.fit_transform(user_embeddings)
    elif method.lower() == 't-sne':
        reducer = TSNE(n_components=2, random_state=random_state)
        reduced_data = reducer.fit_transform(user_embeddings)
    
    # Get the top-ranked learner for each user
    top_ranked = np.argmin(rankings, axis=1)
    
    # Get unique learners
    unique_learners = np.unique(top_ranked)
    n_learners = len(unique_learners)
    
    # Calculate grid dimensions for subplots
    n_cols = min(3, n_learners)  # Max 3 columns
    n_rows = (n_learners + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Flatten axes array for easier indexing if there are multiple rows and columns
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Convert to list for single subplot case
    
    # Create a plot for each learner
    for i, learner_idx in enumerate(unique_learners):
        # Get users who rank this learner highest
        mask = top_ranked == learner_idx
        learner_data = reduced_data[mask]
        
        # Plot all users in light gray
        axes[i].scatter(reduced_data[:, 0], reduced_data[:, 1], 
                        c='lightgray', alpha=0.2)
        
        # Highlight users who rank this learner highest
        axes[i].scatter(learner_data[:, 0], learner_data[:, 1], 
                        c='blue', alpha=0.6)
        
        # Set title
        axes[i].set_title(f'Learner {learner_idx}: Top Choice for {np.sum(mask)} Users')
        
        # Add gridlines
        axes[i].grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Also create a bar plot showing distribution
    counts = np.bincount(top_ranked, minlength=rankings.shape[1])
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counts)), counts)
    plt.xlabel('Learner ID')
    plt.ylabel('Number of Users Ranking it #1')
    plt.title('Distribution of Top-Ranked Learners')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def cluster_users_by_feature(X_original, X_scaled, feature_name, feature_index,
                             n_clusters, random_state=42, method='quantile'):
    """
    Cluster users based on a single feature value (e.g., age, education).
    Creates interpretable clusters by binning a single demographic feature.

    Parameters:
    - X_original: Original (unscaled) feature matrix for interpretable bin edges
    - X_scaled: Scaled feature matrix for computing cluster centers
    - feature_name: Name of the feature (e.g., 'AGEP', 'SCHL')
    - feature_index: Column index of the feature to cluster on
    - n_clusters: Number of clusters (bins) to create (ignored if method='threshold')
    - random_state: Random seed for reproducibility
    - method: Binning method
        - 'quantile': Equal-sized clusters based on percentiles
        - 'equal-width': Equal-width bins across feature range
        - 'threshold': Meaningful thresholds based on feature type (automatic)

    Returns:
    - cluster_labels: Cluster assignment for each user (0 to n_clusters-1)
    - cluster_info: DummyKMeans object with cluster_centers_ attribute
    """
    # Set random seed
    np.random.seed(random_state)

    n_samples = X_original.shape[0]
    n_features = X_scaled.shape[1]

    # Extract the feature values (unscaled for interpretability)
    feature_values = X_original[:, feature_index]

    # Define meaningful thresholds for different features
    FEATURE_THRESHOLDS = {
        'AGEP': [25, 40, 55, 65],  # Young adults, early career, mid-career, late career, retirement
        'SCHL': [16, 20, 21],  # Less than HS, HS/Some college, Bachelor's, Graduate
        # Add more features as needed
    }

    # Create bin edges based on method
    if method == 'quantile':
        # Equal-sized clusters based on quantiles
        percentiles = np.linspace(0, 100, n_clusters + 1)
        bin_edges = np.percentile(feature_values, percentiles)
    elif method == 'equal-width':
        # Equal-width bins across the range
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        bin_edges = np.linspace(min_val, max_val, n_clusters + 1)
    elif method == 'threshold':
        # Use predefined meaningful thresholds for this feature
        if feature_name not in FEATURE_THRESHOLDS:
            print(f"Warning: No predefined thresholds for feature '{feature_name}'. "
                  f"Falling back to 'quantile' method.")
            percentiles = np.linspace(0, 100, n_clusters + 1)
            bin_edges = np.percentile(feature_values, percentiles)
        else:
            thresholds = FEATURE_THRESHOLDS[feature_name]
            # Add min and max to create full bin edges
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            bin_edges = np.array([min_val] + thresholds + [max_val])
            # Update n_clusters based on thresholds
            n_clusters = len(bin_edges) - 1
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'quantile', 'equal-width', or 'threshold'")

    # Ensure bin edges are unique (can happen with discrete features)
    bin_edges = np.unique(bin_edges)

    # If we have fewer unique edges than n_clusters+1, adjust
    if len(bin_edges) < n_clusters + 1:
        print(f"Warning: Feature '{feature_name}' has only {len(bin_edges)-1} unique bins. "
              f"Requested {n_clusters} clusters.")
        n_clusters = len(bin_edges) - 1

    # Assign users to bins (clusters)
    # np.digitize returns 1-indexed bins, so subtract 1 for 0-indexed
    cluster_labels = np.digitize(feature_values, bin_edges, right=False) - 1

    # Handle edge case: values exactly equal to max get assigned to n_clusters
    # (one beyond last cluster), so reassign them to last cluster
    cluster_labels = np.clip(cluster_labels, 0, n_clusters - 1)

    # Compute cluster centers from scaled features
    all_centers = np.zeros((n_clusters, n_features))
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            all_centers[i] = X_scaled[mask].mean(axis=0)
        else:
            # If a cluster is empty, use random point (shouldn't happen with quantile method)
            print(f"Warning: Cluster {i} is empty!")
            all_centers[i] = X_scaled[np.random.randint(0, n_samples)]

    # Print cluster descriptions for interpretability
    print(f"\nClustering by feature: {feature_name} (method: {method})")
    print("=" * 60)
    for i in range(n_clusters):
        mask = cluster_labels == i
        n_users = np.sum(mask)

        # Get bin range
        if i == 0:
            lower_bound = bin_edges[i]
        else:
            lower_bound = bin_edges[i]

        if i == n_clusters - 1:
            upper_bound = bin_edges[i + 1]
            bracket = ']'  # Inclusive on right for last bin
        else:
            upper_bound = bin_edges[i + 1]
            bracket = ')'  # Exclusive on right

        print(f"Cluster {i}: {feature_name} [{lower_bound:.1f}, {upper_bound:.1f}{bracket} "
              f"({n_users} users, {100*n_users/n_samples:.1f}%)")
    print("=" * 60)

    # Create a dummy KMeans object to store cluster centers
    class DummyKMeans:
        def __init__(self, centers):
            self.cluster_centers_ = centers

    cluster_info = DummyKMeans(all_centers)

    return cluster_labels, cluster_info
