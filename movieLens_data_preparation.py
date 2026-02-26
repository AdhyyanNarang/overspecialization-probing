from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def prepare_movielens_dataset(
    ratings_path,
    movies_path,
    num_items,
    latent_dim,
    output_path,
    random_state,
    max_ratings_rows=None,
    max_users=None,
):
    """
    Build a compact MovieLens dataset used by the MSGD experiments.

    Parameters
    ----------
    ratings_path : str or Path
        Path to MovieLens `ratings.dat`.
    movies_path : str or Path
        Path to MovieLens `movies.dat` (validated/read for consistency).
    num_items : int
        Number of most-frequently-rated movies to retain.
    latent_dim : int
        SVD latent dimension.
    output_path : str or Path
        Output pickle path.
    random_state : int
        Random seed for sampling and SVD initialization.
    max_ratings_rows : int or None
        If provided, only read this many rows from ratings.dat (useful for smoke tests).
    max_users : int or None
        If provided, subsample to at most this many users after selecting top movies.
    """
    import numpy as np
    import pandas as pd
    from surprise import Dataset, Reader, SVD

    ratings_path = Path(ratings_path)
    movies_path = Path(movies_path)
    output_path = Path(output_path)

    print(f"Reading ratings from {ratings_path}")
    ratings_df = pd.read_csv(
        ratings_path,
        sep="::",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        nrows=max_ratings_rows,
    )
    print(f"Loaded {len(ratings_df)} ratings rows")

    # The movies file is not used in feature construction here, but we read it to
    # validate the expected MovieLens raw data layout and provide a helpful summary.
    print(f"Reading movies metadata from {movies_path}")
    movies_df = pd.read_csv(
        movies_path,
        sep="::",
        header=None,
        names=["movieId", "title", "genres"],
        engine="python",
    )
    print(f"Loaded {len(movies_df)} movies metadata rows")

    movie_counts = ratings_df["movieId"].value_counts()
    top_movie_ids = movie_counts.head(int(num_items)).index.tolist()
    top_n_df = ratings_df[ratings_df["movieId"].isin(top_movie_ids)].copy()

    if max_users is not None:
        unique_users = np.array(top_n_df["userId"].unique())
        if len(unique_users) > int(max_users):
            rng = np.random.RandomState(int(random_state))
            chosen_users = rng.choice(unique_users, size=int(max_users), replace=False)
            top_n_df = top_n_df[top_n_df["userId"].isin(chosen_users)].copy()
            print(f"Subsampled users to {int(max_users)}")

    if top_n_df.empty:
        raise ValueError("No ratings remain after filtering; adjust num_items/max_users/max_ratings_rows.")

    print(
        "Filtered dataset:"
        f" {top_n_df['userId'].nunique()} users,"
        f" {top_n_df['movieId'].nunique()} movies,"
        f" {len(top_n_df)} ratings"
    )

    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(top_n_df[["userId", "movieId", "rating"]], reader)
    trainset = surprise_data.build_full_trainset()

    model = SVD(n_factors=int(latent_dim), biased=True, verbose=True, random_state=int(random_state))
    model.fit(trainset)

    user_embeddings = model.pu
    item_embeddings = model.qi

    num_users = top_n_df["userId"].nunique()
    user_item_matrix = np.zeros((num_users, int(num_items)))
    user_item_matrix_mask = np.zeros((num_users, int(num_items)), dtype=bool)

    sorted_movie_ids = sorted(top_movie_ids)
    movie_id_to_sorted_index = {movie_id: idx for idx, movie_id in enumerate(sorted_movie_ids)}

    for user_row_idx, user_id in enumerate(top_n_df["userId"].unique()):
        user_df = top_n_df[top_n_df["userId"] == user_id]
        for _, row in user_df.iterrows():
            movie_id = row["movieId"]
            sorted_index = movie_id_to_sorted_index.get(movie_id)
            if sorted_index is None:
                continue
            user_item_matrix[user_row_idx][sorted_index] = row["rating"]
            user_item_matrix_mask[user_row_idx][sorted_index] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "user_embeddings": user_embeddings,
        "item_embeddings": item_embeddings,
        "ratings": user_item_matrix,
        "mask": user_item_matrix_mask,
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved prepared dataset to {output_path}")
    print(
        "Payload shapes:"
        f" user_embeddings={user_embeddings.shape},"
        f" item_embeddings={item_embeddings.shape},"
        f" ratings={user_item_matrix.shape},"
        f" mask={user_item_matrix_mask.shape}"
    )

    return payload


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare the MovieLens dataset for MSGD experiments.")
    parser.add_argument(
        "--ratings-path",
        default="./dataset/ml-10M100K/ratings.dat",
        help="Path to MovieLens ratings.dat",
    )
    parser.add_argument(
        "--movies-path",
        default="./dataset/ml-10M100K/movies.dat",
        help="Path to MovieLens movies.dat",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=200,
        help="Number of top-rated movies to retain",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=5,
        help="Latent dimension for Surprise SVD",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output pickle path (default: ./dataset/MovieLens10M_<num_items>_<latent_dim>.pkl)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for sampling and SVD initialization",
    )
    parser.add_argument(
        "--max-ratings-rows",
        type=int,
        default=None,
        help="Optional row limit when reading ratings.dat (smoke-test speedup)",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Optional cap on number of users after filtering to top movies (smoke-test speedup)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_path = args.output_path
    if output_path is None:
        output_path = f"./dataset/MovieLens10M_{args.num_items}_{args.latent_dim}.pkl"

    prepare_movielens_dataset(
        ratings_path=args.ratings_path,
        movies_path=args.movies_path,
        num_items=args.num_items,
        latent_dim=args.latent_dim,
        output_path=output_path,
        random_state=args.random_state,
        max_ratings_rows=args.max_ratings_rows,
        max_users=args.max_users,
    )


if __name__ == "__main__":
    main()
