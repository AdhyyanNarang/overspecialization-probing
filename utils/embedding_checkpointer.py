"""
Checkpoint management utilities for incremental embedding generation.

This module provides utilities for saving and resuming embedding generation
with checkpoint support. Useful for long-running embedding generation tasks
on GPU clusters where the process may be interrupted.
"""

import numpy as np
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List


@dataclass
class CheckpointMetadata:
    """Metadata for embedding generation checkpoints.

    Attributes:
        model_name: Name of the embedding model used
        embedding_dim: Dimension of output embeddings
        total_examples: Total number of examples to process
        examples_processed: Number of examples processed so far
        chunks_completed: List of chunk IDs that have been completed
        chunk_size: Number of examples per chunk
        min_score_ratio: Minimum score ratio filter (SHP-specific)
        random_state: Random seed for reproducibility
        timestamp: ISO format timestamp when generation started
        is_complete: Whether all chunks have been processed
    """
    model_name: str
    embedding_dim: int
    total_examples: int
    examples_processed: int
    chunks_completed: List[int]
    chunk_size: int
    min_score_ratio: float
    random_state: int
    timestamp: str
    is_complete: bool


class EmbeddingCheckpointer:
    """Manages checkpoints for incremental embedding generation.

    This class handles:
    - Saving embeddings chunk by chunk
    - Loading existing checkpoints for resume
    - Merging chunks into final embedding file
    - Cleanup of temporary checkpoint files

    Example:
        >>> checkpointer = EmbeddingCheckpointer(Path("cache"), "my_embeddings")
        >>> metadata = CheckpointMetadata(...)
        >>> checkpointer.save_checkpoint(0, embeddings_chunk, labels_chunk, metadata)
        >>> chunks, metadata = checkpointer.load_checkpoint()
        >>> embeddings, labels = checkpointer.merge_chunks(Path("cache"))
    """

    def __init__(self, cache_dir: Path, checkpoint_name: str):
        """Initialize checkpointer.

        Args:
            cache_dir: Base directory for caching
            checkpoint_name: Unique name for this checkpoint session
        """
        self.cache_dir = Path(cache_dir)
        self.checkpoint_name = checkpoint_name
        self.checkpoint_dir = self.cache_dir / "checkpoints" / checkpoint_name
        self.metadata_path = self.checkpoint_dir / "metadata.json"

    def save_checkpoint(self, chunk_id: int, embeddings: np.ndarray,
                       labels: np.ndarray, metadata: CheckpointMetadata):
        """Save embeddings for a specific chunk.

        Args:
            chunk_id: Integer ID of the chunk (0-indexed)
            embeddings: Embeddings array for this chunk (n_samples, embedding_dim)
            labels: Labels array for this chunk (n_samples,)
            metadata: Metadata object to update with progress
        """
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save chunk data (compressed to save disk space)
        chunk_path = self.checkpoint_dir / f"chunk_{chunk_id}.npz"
        np.savez_compressed(chunk_path, embeddings=embeddings, labels=labels)

        # Update metadata
        if chunk_id not in metadata.chunks_completed:
            metadata.chunks_completed.append(chunk_id)
            metadata.examples_processed += len(embeddings)

        # Save updated metadata
        self.save_metadata(metadata)

    def load_checkpoint(self) -> Tuple[Dict[int, Dict[str, np.ndarray]], CheckpointMetadata]:
        """Load all completed chunks and metadata.

        Returns:
            Tuple of (chunks_dict, metadata) where chunks_dict maps chunk_id to
            {'embeddings': array, 'labels': array}

        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist
        """
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        # Load metadata
        metadata = self.load_metadata()

        # Load all completed chunks
        chunks = {}
        for chunk_id in metadata.chunks_completed:
            chunk_path = self.checkpoint_dir / f"chunk_{chunk_id}.npz"
            if not chunk_path.exists():
                print(f"Warning: Chunk {chunk_id} listed in metadata but file not found")
                continue

            data = np.load(chunk_path)
            chunks[chunk_id] = {
                'embeddings': data['embeddings'],
                'labels': data['labels']
            }

        return chunks, metadata

    def is_complete(self) -> bool:
        """Check if generation is complete.

        Returns:
            True if all chunks have been processed and merged
        """
        if not self.checkpoint_dir.exists():
            return False

        try:
            metadata = self.load_metadata()
            return metadata.is_complete
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def merge_chunks(self, output_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Merge all chunks into final embedding file.

        Args:
            output_path: Directory where final embeddings will be saved

        Returns:
            Tuple of (embeddings, labels) as full arrays
        """
        chunks, metadata = self.load_checkpoint()

        # Sort chunks by ID to maintain order
        sorted_chunk_ids = sorted(chunks.keys())

        if not sorted_chunk_ids:
            raise ValueError("No chunks found to merge")

        # Concatenate embeddings and labels
        embeddings_list = [chunks[i]['embeddings'] for i in sorted_chunk_ids]
        labels_list = [chunks[i]['labels'] for i in sorted_chunk_ids]

        embeddings = np.vstack(embeddings_list)
        labels = np.concatenate(labels_list)

        print(f"Merged {len(sorted_chunk_ids)} chunks:")
        print(f"  Final embeddings shape: {embeddings.shape}")
        print(f"  Final labels shape: {labels.shape}")

        # Save final embeddings file
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        embeddings_file = output_path / f"{self.checkpoint_name}.npy"
        np.save(embeddings_file, embeddings)
        print(f"Saved embeddings to {embeddings_file}")

        # Save metadata JSON
        metadata_dir = output_path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = metadata_dir / f"{self.checkpoint_name}.json"

        with open(metadata_file, 'w') as f:
            json.dump({
                'labels': labels.tolist(),
                'n_samples': len(labels),
                'embedding_model': metadata.model_name,
                'embedding_dim': metadata.embedding_dim,
                'min_score_ratio': metadata.min_score_ratio,
                'random_state': metadata.random_state,
                'max_examples': metadata.total_examples,
                'use_large_model': 'Qwen' in metadata.model_name
            }, f, indent=2)
        print(f"Saved metadata to {metadata_file}")

        return embeddings, labels

    def cleanup_checkpoints(self):
        """Remove checkpoint directory after successful merge.

        This should be called after merge_chunks() to clean up
        temporary checkpoint files.
        """
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
            print(f"Cleaned up checkpoint directory: {self.checkpoint_dir}")

    def save_metadata(self, metadata: CheckpointMetadata):
        """Save metadata to JSON file.

        Args:
            metadata: CheckpointMetadata object to save
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with open(self.metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

    def load_metadata(self) -> CheckpointMetadata:
        """Load metadata from JSON file.

        Returns:
            CheckpointMetadata object

        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path) as f:
            data = json.load(f)

        return CheckpointMetadata(**data)

    def get_progress_summary(self) -> str:
        """Get a human-readable progress summary.

        Returns:
            Progress summary string
        """
        try:
            metadata = self.load_metadata()
            n_chunks = (metadata.total_examples + metadata.chunk_size - 1) // metadata.chunk_size
            n_completed = len(metadata.chunks_completed)
            pct_complete = 100 * n_completed / n_chunks if n_chunks > 0 else 0

            summary = f"Progress: {n_completed}/{n_chunks} chunks ({pct_complete:.1f}%)\n"
            summary += f"Examples processed: {metadata.examples_processed}/{metadata.total_examples}"

            return summary
        except (FileNotFoundError, json.JSONDecodeError):
            return "No checkpoint found"
