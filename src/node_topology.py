"""
Node coordinates and physical graph topology for the LoRaWAN vineyard deployment.

Dataset: emanueleg/lora-rssi (vineyard-2021)
Location: Vineyard in Emilia-Romagna, Italy.
"""

import numpy as np

# Sensor node coordinates (tinovi-01 through tinovi-08)
# Ordered to match RSSI_01 through RSSI_08 in the dataset
NODE_COORDS = np.array([
    [44.822073400339981, 10.815745652887379],  # tinovi-01 (RSSI_01)
    [44.823122175066679, 10.81638157790559],   # tinovi-02 (RSSI_02)
    [44.82457105755545,  10.81725847759737],   # tinovi-03 (RSSI_03)
    [44.82375277777778,  10.81694166666667],   # tinovi-04 (RSSI_04)
    [44.82244371876236,  10.81619243363089],   # tinovi-05 (RSSI_05)
    [44.82197252644758,  10.81612277256111],   # tinovi-06 (RSSI_06)
    [44.82294444444445,  10.81670277777778],   # tinovi-07 (RSSI_07)
    [44.82445862158094,  10.8176270471526],    # tinovi-08 (RSSI_08)
], dtype=np.float64)

GATEWAY_COORDS = np.array([44.821957, 10.815304], dtype=np.float64)

# Gateway distances (meters) as reported in the dataset documentation
GW_DISTANCES_M = np.array([40, 160, 330, 240, 90, 65, 160, 340], dtype=np.float64)

NODE_NAMES = [f"RSSI_{i:02d}" for i in range(1, 9)]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in meters between two GPS points."""
    R = 6_371_000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def compute_distance_matrix(node_coords: np.ndarray = NODE_COORDS) -> np.ndarray:
    """Returns pairwise Haversine distance matrix (N x N) in meters."""
    node_coords = np.asarray(node_coords, dtype=np.float64)
    n = len(node_coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance(
                node_coords[i, 0], node_coords[i, 1],
                node_coords[j, 0], node_coords[j, 1]
            )
            D[i, j] = d
            D[j, i] = d
    return D


def compute_physical_adjacency(
    sigma: float = None,
    node_indices=None,
    node_coords: np.ndarray = NODE_COORDS,
) -> np.ndarray:
    """
    Computes a Gaussian-kernel adjacency matrix from physical GPS coordinates.

    A_ij = exp(-dist(i,j)^2 / sigma^2)

    If sigma is None, it defaults to the standard deviation of all pairwise distances.
    """
    D = compute_distance_matrix(node_coords)
    if node_indices is not None:
        D = D[np.ix_(node_indices, node_indices)]
    if sigma is None:
        # Use std of upper-triangle distances as bandwidth
        upper_dists = D[np.triu_indices_from(D, k=1)]
        sigma = max(float(np.std(upper_dists)), 1.0) if len(upper_dists) else 1.0

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Physical graph bandwidth must be positive and finite")

    A = np.exp(-(D / sigma) ** 2)
    np.fill_diagonal(A, 1.0)
    return A.astype(np.float32)
