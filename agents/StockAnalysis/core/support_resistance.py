import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple

# Type definition for the clustered result
LevelCluster = Dict[str, Union[float, str, int]]

def swing_levels(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    """
    Identify local swing highs and lows using a 3-bar pattern.

    A swing low is a bar where the Low is lower than the Low of the bars immediately
    preceding and succeeding it. A swing high is the opposite for Highs.

    Args:
        df: DataFrame with 'High' and 'Low' columns (OHLC data).

    Returns:
        A tuple containing lists of identified support and resistance levels.
    """
    # 1. Input Validation
    if df.empty or not all(col in df.columns for col in ["High", "Low"]):
        print("Error: DataFrame is empty or missing 'High'/'Low' columns.")
        return [], []

    highs = df["High"].values
    lows = df["Low"].values
    support: List[float] = []
    resistance: List[float] = []

    n = len(df)
    # Need at least 3 bars (L-M-R) to detect a turning point
    if n < 3:
        return support, resistance

    # 2. Swing Detection Loop
    # We iterate from the second bar (index 1) up to the second-to-last bar (index n-2)
    for i in range(1, n - 1):
        # Swing Low (V-shape): M < L and M < R
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            support.append(lows[i])

        # Swing High (Inverted V-shape): M > L and M > R
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            resistance.append(highs[i])

    return support, resistance

def cluster_levels(levels: List[float], threshold: float = 0.005) -> List[LevelCluster]:
    """
    Cluster price levels that are within `threshold` % of each other
    and grade the resulting zones by strength (hits).

    Args:
        levels: A list of price points (support or resistance) to cluster.
        threshold: Maximum percentage difference allowed for two levels to be
                   considered part of the same zone (e.g., 0.005 = 0.5%).

    Returns:
        A list of dictionaries, where each dictionary represents a graded
        support/resistance zone.
    """
    if not levels:
        return []

    # 1. Preparation: Sort the levels to enable sequential clustering
    levels = sorted(levels)
    clusters = [[levels[0]]]

    # 2. Clustering Logic (Single-pass on sorted data)
    for lvl in levels[1:]:
        last_lvl_in_cluster = clusters[-1][-1]

        # We compare the absolute difference relative to the existing cluster level.
        if abs(lvl - last_lvl_in_cluster) / last_lvl_in_cluster < threshold:
            clusters[-1].append(lvl)
        else:
            # Start a new cluster
            clusters.append([lvl])

    # 3. Grade Clusters by Strength (based on number of hits)
    graded_clusters: List[LevelCluster] = []
    for cluster in clusters:
        mean_price = float(np.mean(cluster))
        count = len(cluster)

        # Define strength based on the number of price "hits" (bounces)
        if count >= 4:
            strength = "very strong"
        elif count >= 2:
            strength = "strong"
        else:
            strength = "medium" # Single hit is still a level

        graded_clusters.append({
            "level": round(mean_price, 4), # Rounding for cleaner output
            "strength": strength,
            "hits": count
        })

    return graded_clusters

def support_resistance(df: pd.DataFrame, threshold: float = 0.005) -> Dict[str, List[LevelCluster]]:
    """
    Compute graded support and resistance zones from a DataFrame.

    Args:
        df: DataFrame with 'High' and 'Low' columns.
        threshold: The proximity threshold for clustering price levels.

    Returns:
        A dictionary containing "support_zones" and "resistance_zones".
    """
    # 1. Identify raw swing points
    support_raw, resistance_raw = swing_levels(df)

    # 2. Cluster and grade the identified levels
    support_zones = cluster_levels(support_raw, threshold=threshold)
    resistance_zones = cluster_levels(resistance_raw, threshold=threshold)

    return {
        "support_zones": support_zones,
        "resistance_zones": resistance_zones
    }