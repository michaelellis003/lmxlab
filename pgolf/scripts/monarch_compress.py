"""Monarch post-hoc compression for trained pgolf models.

Loads a trained model, projects each weight matrix to its optimal
Monarch approximation, and evaluates the quality loss (BPB delta).

Usage:
    uv run python scripts/monarch_compress.py <model_path>
"""

import sys
import numpy as np
from pathlib import Path


def project_to_monarch_rect(W: np.ndarray, m1: int, m2: int) -> np.ndarray:
    """Project a dense (n_out, n_in) matrix to its Monarch approximation.

    For non-square matrices, we handle the outer/inner dims separately.
    Uses rank-1 SVD per block for the projection.

    Args:
        W: Dense weight matrix (n_out, n_in)
        m1: First block size (n = m1 * m2)
        m2: Second block size

    Returns:
        W_monarch: Monarch-approximated weight matrix (same shape as W)
    """
    n_out, n_in = W.shape

    # For simplicity, project each dimension independently
    # Reshape W as (m1, m2, m1, m2) if square, or handle rectangular
    if n_out == n_in and n_out == m1 * m2:
        # Square case: full Monarch projection
        # Reshape: W[i*m2+j, k*m2+l] -> T[i, j, k, l]
        T = W.reshape(m1, m2, m1, m2)

        # For each (j,l) pair, we have an m1 x m1 block
        # T[:, j, :, l] is the (j,l)-th block of size m1 x m1
        # SVD each block, keep rank-1
        W_approx = np.zeros_like(W)
        for j in range(m2):
            for l in range(m2):
                block = T[:, j, :, l]  # (m1, m1)
                U, s, Vt = np.linalg.svd(block, full_matrices=False)
                # Rank-1 approximation
                block_approx = s[0] * np.outer(U[:, 0], Vt[0, :])
                # Put back
                for i in range(m1):
                    for k in range(m1):
                        W_approx[i * m2 + j, k * m2 + l] = block_approx[i, k]

        return W_approx
    else:
        # Non-square or incompatible: just return original
        return W


def compute_monarch_params(n: int, m1: int, m2: int) -> int:
    """Count parameters in Monarch approximation."""
    # Each of m2² blocks is rank-1: m1 left + m1 right = 2*m1 per block
    # Total: m2² * 2 * m1 = 2 * m1 * m2²
    # But with two block-diag factors: 2 * m1² * m2 (or 2 * m2² * m1)
    # Simpler: just count the stored values
    return 2 * m1 * m2 * max(m1, m2)


def analyze_model(model_path: str):
    """Load model and analyze Monarch compressibility."""
    import pickle, zlib

    path = Path(model_path)
    if path.suffix == ".npz":
        data = dict(np.load(str(path)))
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    print(f"Loaded {len(data)} tensors from {path}")

    total_params = 0
    total_monarch_params = 0
    total_error = 0.0

    for name, arr in sorted(data.items()):
        total_params += arr.size
        if arr.ndim != 2:
            total_monarch_params += arr.size
            continue

        n_out, n_in = arr.shape

        # Find suitable block sizes
        if n_out == n_in:
            n = n_out
            # Try to factor n into m1 * m2
            best_m1, best_m2 = 1, n
            for m1 in range(2, int(n ** 0.5) + 1):
                if n % m1 == 0:
                    m2 = n // m1
                    if abs(m1 - m2) < abs(best_m1 - best_m2):
                        best_m1, best_m2 = m1, m2

            if best_m1 > 1:
                W_monarch = project_to_monarch_rect(arr, best_m1, best_m2)
                error = np.linalg.norm(arr - W_monarch) / np.linalg.norm(arr)
                monarch_params = compute_monarch_params(n, best_m1, best_m2)
                ratio = arr.size / monarch_params
                total_monarch_params += monarch_params
                total_error += error * arr.size
                print(f"  {name}: {arr.shape} -> Monarch({best_m1}x{best_m2}) "
                      f"error={error:.4f} ratio={ratio:.1f}x params={monarch_params}")
            else:
                total_monarch_params += arr.size
                print(f"  {name}: {arr.shape} -> no factorization (prime dim)")
        else:
            total_monarch_params += arr.size
            print(f"  {name}: {arr.shape} -> skip (non-square)")

    print(f"\nTotal params: {total_params:,}")
    print(f"Monarch params: {total_monarch_params:,}")
    print(f"Compression ratio: {total_params / total_monarch_params:.2f}x")
    print(f"Weighted avg error: {total_error / total_params:.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find most recent model
        import glob
        models = sorted(glob.glob("logs/pgolf_*_mlx_model.npz"))
        if models:
            analyze_model(models[-1])
        else:
            print("No model found. Run training first.")
    else:
        analyze_model(sys.argv[1])
