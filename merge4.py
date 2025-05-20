#!/usr/bin/env python3
"""
Enhanced model_merger.py with advanced distribution preservation techniques

This version includes:
- Improved multimodal detection via kernel density estimation
- Adaptive distribution calibration with multi-moment preservation
- Hierarchical fallback strategies for problematic layers
- Wasserstein distance-based quality metrics
- Sophisticated sign alignment with SVD-based permutation matching
- Memory-efficient processing for large tensors
"""

import os
import sys
import time
import json
import argparse
import logging
import importlib.util
import csv
import warnings
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from collections import Counter
from functools import partial

import numpy as np
import torch
import onnx
from onnx import numpy_helper, helper
import matplotlib.pyplot as plt

try:
    from scipy import stats
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logger = logging.getLogger("model_merger")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# ------------------------------------------------------------------------------
# Enhanced distribution analysis utilities
# ------------------------------------------------------------------------------

def estimate_density(samples: np.ndarray, kernel_width: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate probability density using kernel density estimation.
    
    Args:
        samples: Sample data points
        kernel_width: Width of the Gaussian kernel (if None, use Scott's rule)
        
    Returns:
        Tuple of (grid points, density values)
    """
    if len(samples) < 10:
        # For very small samples, just return a simple histogram
        hist, bins = np.histogram(samples, bins=max(3, len(samples)), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        return bin_centers, hist
        
    # Use scipy's KDE if available, otherwise use a simple histogram
    if SCIPY_AVAILABLE:
        try:
            # Use scipy's gaussian_kde for better estimation
            kde = stats.gaussian_kde(samples, bw_method='scott' if kernel_width is None else kernel_width)
            
            # Create evaluation grid
            min_val = np.min(samples)
            max_val = np.max(samples)
            x_grid = np.linspace(min_val - 0.1 * (max_val - min_val), 
                                max_val + 0.1 * (max_val - min_val), 
                                min(1000, max(100, len(samples) // 10)))
            
            # Evaluate density
            density = kde(x_grid)
            
            return x_grid, density
        except Exception as e:
            logger.debug(f"KDE estimation failed: {e}, falling back to histogram")
    
    # Fallback to histogram
    hist, bins = np.histogram(samples, bins=min(100, max(10, len(samples) // 20)), density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, hist


def detect_multimodal_distribution(samples: np.ndarray, min_peak_distance: float = 0.1) -> bool:
    """
    Detect if a distribution appears to be multimodal using density estimation.
    
    Args:
        samples: Numpy array of sampled values
        min_peak_distance: Minimum distance between peaks relative to range
        
    Returns:
        True if distribution appears multimodal
    """
    # For very small samples, use simple heuristics
    if len(samples) < 20:
        return False
    
    # For binary/discrete distributions, use simple analysis
    unique_vals = np.unique(samples)
    if len(unique_vals) < 10:
        # Check if values are split into distinct groups
        if len(unique_vals) >= 2:
            sorted_vals = np.sort(unique_vals)
            gaps = np.diff(sorted_vals)
            # If largest gap is significantly larger than average gap
            if np.max(gaps) > 2.0 * np.mean(gaps):
                return True
        return False
    
    # Compute sample statistics for quick check
    if SCIPY_AVAILABLE:
        kurtosis = stats.kurtosis(samples)
        # Negative kurtosis often indicates bimodality
        if kurtosis < -0.5:
            return True
    
    # Estimate density
    x_grid, density = estimate_density(samples)
    
    # Find peaks in the density
    peaks = []
    for i in range(1, len(density) - 1):
        if density[i] > density[i-1] and density[i] > density[i+1]:
            # Peak height must be at least 20% of max height
            if density[i] > 0.2 * np.max(density):
                peaks.append((x_grid[i], density[i]))
    
    # No significant peaks found
    if len(peaks) < 2:
        return False
    
    # Sort peaks by height (tallest first)
    peaks.sort(key=lambda x: x[1], reverse=True)
    peak_xs = [x for x, _ in peaks]
    
    # Check if peaks are sufficiently separated
    data_range = np.max(samples) - np.min(samples)
    if data_range < 1e-6:  # Avoid division by zero
        return False
        
    # Calculate peak distances
    peak_positions = np.array([x for x, _ in peaks])
    peak_distances = np.diff(np.sort(peak_positions))
    max_peak_distance = np.max(peak_distances) if len(peak_distances) > 0 else 0
    
    # Multimodal if peaks are significantly separated
    if max_peak_distance > min_peak_distance * data_range:
        # Ensure the peaks have significant valleys between them
        # Find the lowest density between the two tallest peaks
        if len(peaks) >= 2:
            p1_idx = np.argmin(np.abs(x_grid - peaks[0][0]))
            p2_idx = np.argmin(np.abs(x_grid - peaks[1][0]))
            start_idx, end_idx = min(p1_idx, p2_idx), max(p1_idx, p2_idx)
            
            if end_idx - start_idx > 1:  # Ensure we have points between peaks
                valley_density = np.min(density[start_idx:end_idx])
                peak_heights = [peaks[0][1], peaks[1][1]]
                min_peak = min(peak_heights)
                
                # If valley is significantly lower than the peaks
                if valley_density < 0.6 * min_peak:
                    return True
    
    return False


def detect_discrete_multimodal_tensor(arr: np.ndarray, sample_size: int = 10000) -> bool:
    """
    Detect if a tensor has a discrete or multimodal distribution.
    Uses advanced heuristics for more accurate detection.
    
    Args:
        arr: Numpy array to check
        sample_size: Maximum number of elements to sample for analysis
        
    Returns:
        True if tensor appears to have discrete or multimodal values
    """
    # Sample the array for efficiency with large tensors
    if arr.size > sample_size:
        flat_arr = arr.flatten()
        indices = np.random.choice(flat_arr.size, sample_size, replace=False)
        sample = flat_arr[indices]
    else:
        sample = arr.flatten()
    
    # Check number of unique values relative to tensor size
    unique_vals = np.unique(sample)
    
    # For clear discrete cases (very few unique values)
    if len(unique_vals) < min(30, max(5, sample.size * 0.05)):
        return True
    
    # Check for zero concentration (common in neural net weights)
    zero_ratio = np.sum(np.abs(sample) < 1e-6) / sample.size
    if zero_ratio > 0.4:
        return True
    
    # Compute basic statistics for quick checks
    if SCIPY_AVAILABLE and len(sample) >= 10:
        # Check for bimodality using Hartigan's dip test if available
        try:
            # Check kurtosis - negative values suggest potential bimodality
            kurtosis = stats.kurtosis(sample)
            if kurtosis < -0.6:  # More strict threshold
                return True
                
            # Use more advanced multimodality tests if sample size is sufficient
            if len(sample) >= 100:
                from scipy.stats import diptest
                dip, pval = diptest.diptest(sample)
                if pval < 0.05:  # Significant result suggests multimodality
                    return True
        except (ImportError, AttributeError):
            pass
            
    # For bias vectors, check for bimodality even with more unique values
    if arr.ndim == 1 or 'bias' in getattr(arr, 'name', ''):
        # Check if distribution appears multimodal
        if detect_multimodal_distribution(sample):
            return True
    
    return False


def compute_wasserstein_distance(arr1: np.ndarray, arr2: np.ndarray, 
                                sample_size: int = 10000) -> float:
    """
    Compute the 1D Wasserstein distance (Earth Mover's Distance) between two distributions.
    
    Args:
        arr1: First distribution array
        arr2: Second distribution array
        sample_size: Maximum number of elements to sample from each array
        
    Returns:
        Wasserstein distance between the distributions
    """
    # Sample arrays if they're large
    if arr1.size > sample_size:
        flat1 = arr1.flatten()
        indices = np.random.choice(flat1.size, sample_size, replace=False)
        samples1 = flat1[indices]
    else:
        samples1 = arr1.flatten()
        
    if arr2.size > sample_size:
        flat2 = arr2.flatten()
        indices = np.random.choice(flat2.size, sample_size, replace=False)
        samples2 = flat2[indices]
    else:
        samples2 = arr2.flatten()
    
    # Compute Wasserstein distance using scipy if available
    if SCIPY_AVAILABLE:
        try:
            distance = stats.wasserstein_distance(samples1, samples2)
            return float(distance)
        except Exception as e:
            logger.debug(f"Wasserstein distance calculation failed: {e}, using approximation")
    
    # Otherwise use a simple approximation based on sorted samples
    sorted1 = np.sort(samples1)
    sorted2 = np.sort(samples2)
    
    # Interpolate to same length if necessary
    if len(sorted1) != len(sorted2):
        # Linearly interpolate the shorter array to match the longer one
        if len(sorted1) < len(sorted2):
            indices = np.linspace(0, len(sorted1) - 1, len(sorted2))
            sorted1 = np.interp(indices, np.arange(len(sorted1)), sorted1)
        else:
            indices = np.linspace(0, len(sorted2) - 1, len(sorted1))
            sorted2 = np.interp(indices, np.arange(len(sorted2)), sorted2)
    
    # Compute average absolute difference between sorted arrays
    return float(np.mean(np.abs(sorted1 - sorted2)))


def compute_kl_divergence(arr1: np.ndarray, arr2: np.ndarray, 
                         sample_size: int = 10000, num_bins: int = 100) -> float:
    """
    Compute KL divergence between two distributions using histogram approximation.
    
    Args:
        arr1: First distribution array (reference)
        arr2: Second distribution array
        sample_size: Maximum number of elements to sample from each array
        num_bins: Number of bins for histogram approximation
        
    Returns:
        KL divergence from arr1 to arr2 (not symmetric)
    """
    # Sample arrays if they're large
    if arr1.size > sample_size:
        flat1 = arr1.flatten()
        indices = np.random.choice(flat1.size, sample_size, replace=False)
        samples1 = flat1[indices]
    else:
        samples1 = arr1.flatten()
        
    if arr2.size > sample_size:
        flat2 = arr2.flatten()
        indices = np.random.choice(flat2.size, sample_size, replace=False)
        samples2 = flat2[indices]
    else:
        samples2 = arr2.flatten()
    
    # Find common range for both distributions
    min_val = min(np.min(samples1), np.min(samples2))
    max_val = max(np.max(samples1), np.max(samples2))
    
    # Add small padding to range
    range_pad = 0.01 * (max_val - min_val)
    bin_range = (min_val - range_pad, max_val + range_pad)
    
    # Create histograms with same bins
    hist1, bin_edges = np.histogram(samples1, bins=num_bins, range=bin_range, density=True)
    hist2, _ = np.histogram(samples2, bins=bin_edges, density=True)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    hist1 = hist1 + epsilon
    hist2 = hist2 + epsilon
    
    # Normalize
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Compute KL divergence: sum(p_i * log(p_i / q_i))
    kl_div = np.sum(hist1 * np.log(hist1 / hist2))
    
    return float(kl_div)


def compute_distribution_metrics(arr_before: List[np.ndarray], arr_after: np.ndarray, 
                              max_sample_size: int = 50000) -> Dict[str, float]:
    """
    Compute comprehensive distribution metrics between input and merged distributions.
    
    Args:
        arr_before: List of arrays before merging
        arr_after: Array after merging
        max_sample_size: Maximum number of elements to sample for metrics
        
    Returns:
        Dictionary of distribution metrics
    """
    # Concatenate the before arrays to create a unified distribution
    # This assumes the distributions should be compared in aggregate
    before_sizes = [arr.size for arr in arr_before]
    total_before_size = sum(before_sizes)
    
    # Normalize to get proper weight for each input
    weights = [size / total_before_size for size in before_sizes]
    
    # For small tensors, use the entire distributions
    if total_before_size < max_sample_size and arr_after.size < max_sample_size:
        all_before = np.concatenate([arr.flatten() for arr in arr_before])
        after_flat = arr_after.flatten()
    else:
        # For larger tensors, use weighted sampling
        all_before = np.concatenate([
            np.random.choice(
                arr.flatten(), 
                size=min(arr.size, int(max_sample_size * weight)),
                replace=False
            ) for arr, weight in zip(arr_before, weights)
        ])
        
        after_flat = arr_after.flatten()
        if arr_after.size > max_sample_size:
            after_flat = np.random.choice(
                after_flat, 
                size=min(arr_after.size, max_sample_size),
                replace=False
            )
    
    # Basic statistics
    before_mean = float(np.mean(all_before))
    before_std = float(np.std(all_before))
    after_mean = float(np.mean(after_flat))
    after_std = float(np.std(after_flat))
    
    # Distribution distance metrics
    wasserstein_dist = compute_wasserstein_distance(all_before, after_flat)
    
    # For very discrete distributions, compute unique value metrics
    unique_before = len(np.unique(all_before))
    unique_after = len(np.unique(after_flat))
    unique_ratio = float(unique_after / max(1, unique_before))
    
    # Check for standard deviation contraction
    std_ratio = float(after_std / max(1e-10, before_std))
    
    # Compute KL divergence if distributions are non-trivial
    if unique_before > 10 and unique_after > 10:
        kl_div = compute_kl_divergence(all_before, after_flat)
    else:
        kl_div = None
    
    # Return comprehensive metrics
    metrics = {
        "mean_diff": float(abs(after_mean - before_mean)),
        "std_ratio": std_ratio,
        "wasserstein_dist": wasserstein_dist,
        "unique_ratio": unique_ratio,
        "unique_before": unique_before,
        "unique_after": unique_after
    }
    
    if kl_div is not None:
        metrics["kl_divergence"] = kl_div
        
    return metrics


def compute_stats(arr: np.ndarray, max_sample_size: int = 50000) -> Dict[str, float]:
    """
    Compute basic statistical metrics on an array with numeric stability.
    Uses sampling for large arrays to improve performance.
    
    Args:
        arr: Array to compute statistics for
        max_sample_size: Maximum number of elements to sample
        
    Returns:
        Dictionary of statistics
    """
    # Catch warnings to avoid exposing NumPy warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        # Use sampling for large arrays
        if arr.size > max_sample_size:
            flat = arr.flatten()
            indices = np.random.choice(flat.size, max_sample_size, replace=False)
            samples = flat[indices]
            
            # For unique values, use a better approximation
            try:
                # Take several random samples to estimate unique ratio
                n_samples = 5
                sample_size = min(max_sample_size // n_samples, 10000)
                unique_ratios = []
                
                for _ in range(n_samples):
                    idx = np.random.choice(flat.size, sample_size, replace=False)
                    sample = flat[idx]
                    unique_ratios.append(len(np.unique(sample)) / sample_size)
                
                # Use median ratio to avoid outliers
                unique_ratio = np.median(unique_ratios)
                # Scale to full size with dampening for large arrays
                unique_count = min(
                    int(unique_ratio * flat.size * np.power(flat.size / sample_size, -0.3)), 
                    flat.size
                )
            except:
                # Fallback if approximation fails
                unique_count = len(np.unique(samples))
                
            return {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "min": float(np.min(samples)),
                "max": float(np.max(samples)),
                "median": float(np.median(samples)),
                "unique_values": unique_count
            }
        else:
            # For smaller arrays, compute exact statistics
            flat = arr.flatten()
            return {
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "min": float(np.min(flat)),
                "max": float(np.max(flat)),
                "median": float(np.median(flat)),
                "unique_values": len(np.unique(flat))
            }

# ------------------------------------------------------------------------------
# Enhanced merge methods with distribution preservation
# ------------------------------------------------------------------------------

def average_weights(arrs: List[np.ndarray], ratios: List[float], **kwargs) -> np.ndarray:
    """
    Merge weights by computing the weighted average with distribution preservation.
    
    Args:
        arrs: List of numpy arrays to merge
        ratios: Weight ratios for each array
        
    Returns:
        Merged weights preserving original distribution characteristics
    """
    logger.debug("Computing weighted average with distribution preservation")
    
    # Normalize ratios to sum to 1
    ratios_np = np.array(ratios, dtype=np.float64)
    ratios_np = ratios_np / np.sum(ratios_np)
    
    # Convert to higher precision for calculation
    arrs_64 = [arr.astype(np.float64) for arr in arrs]
    
    # Initialize with zeros in same shape
    result = np.zeros_like(arrs_64[0])
    
    # Accumulate incrementally (in-place for better memory efficiency)
    for i, arr in enumerate(arrs_64):
        result += arr * ratios_np[i]
    
    # Convert back to original dtype
    return result.astype(arrs[0].dtype)


def trimmed_mean_weights(arrs: List[np.ndarray], trim_ratio: float, ratios: List[float], **kwargs) -> np.ndarray:
    """
    Merge weights using trimmed mean with distribution preservation.
    
    Args:
        arrs: List of numpy arrays to merge
        trim_ratio: Proportion of extreme values to trim
        ratios: Weight ratios for each array (used for small tensors)
        
    Returns:
        Merged weights using trimmed mean approach
    """
    logger.debug(f"Computing trimmed mean (trim_ratio={trim_ratio})")
    
    # Special case for small tensors
    if arrs[0].size < 100:
        logger.debug(f"Small tensor detected (size={arrs[0].size}), using average instead of trimmed mean")
        return average_weights(arrs, ratios)
    
    # Convert to higher precision
    X = np.stack([arr.astype(np.float64) for arr in arrs], axis=0)
    n = X.shape[0]
    k = int(np.floor(trim_ratio * n))
    
    if 2 * k >= n:
        logger.warning("trim_ratio too large for number of models, using average instead")
        return average_weights(arrs, ratios)
    
    # Sort along first axis (across models)
    X_sorted = np.sort(X, axis=0)
    trimmed = X_sorted[k: n - k]
    
    # Compute mean and convert back to original dtype
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        result = np.mean(trimmed, axis=0)
    
    return result.astype(arrs[0].dtype)


def geometric_median_weights(arrs: List[np.ndarray], eps: float = 1e-5, max_iters: int = 100, **kwargs) -> np.ndarray:
    """
    Compute the geometric median of weights with distribution preservation.
    
    Args:
        arrs: List of numpy arrays to merge
        eps: Convergence threshold
        max_iters: Maximum number of iterations
        
    Returns:
        Merged weights using geometric median approach
    """
    logger.debug(f"Computing geometric median (eps={eps}, max_iters={max_iters})")
    
    # Special case for small tensors or binary/discrete values
    if arrs[0].size < 100:
        logger.debug(f"Small tensor detected (size={arrs[0].size}), using element-wise median")
        # For small tensors, use element-wise median to preserve discrete values
        return element_median_weights(arrs)
    
    # Special case for binary or near-binary distributions (common in neural nets)
    # Check if values are mostly clustered around a few distinct values
    unique_vals = np.unique(arrs[0])
    if len(unique_vals) < 10 or (np.abs(unique_vals) < 0.01).sum() > 0.8 * len(unique_vals):
        logger.debug("Binary or discrete distribution detected, using element-wise median")
        return element_median_weights(arrs)
    
    # Standard case: implement Weiszfeld's algorithm for geometric median
    # Convert to higher precision
    X = np.stack([arr.astype(np.float64) for arr in arrs], axis=0)
    
    # Initialize with mean
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        median = np.mean(X, axis=0)
    
    # Use efficient implementation for large arrays
    if arrs[0].size > 1000000:  # 1M elements
        # For very large tensors, use a sampling-based approach
        sample_size = 100000
        orig_shape = X.shape[1:]
        
        # Sample indices once and reuse for all iterations
        indices = np.random.choice(np.prod(orig_shape), sample_size, replace=False)
        
        # Flatten input for efficient indexing
        X_flat = X.reshape(X.shape[0], -1)
        X_sampled = X_flat[:, indices]
        
        # Initialize median at sampled locations
        median_flat = median.reshape(-1)
        median_sampled = median_flat[indices]
        
        # Iterative update
        for i in range(max_iters):
            # Calculate distances from current median estimate
            diffs = X_sampled - median_sampled.reshape(1, -1)
            distances = np.sqrt((diffs ** 2).sum(axis=1) + eps)
            
            # Check for convergence
            if np.all(distances < eps):
                break
                
            # Update weights based on distances
            weights = 1.0 / np.maximum(distances, eps)
            w_sum = np.sum(weights)
            
            if w_sum < eps:
                break
                
            # Compute weighted sum of points
            weighted_points = weights.reshape(-1, 1) * X_sampled
            new_median_sampled = np.sum(weighted_points, axis=0) / w_sum
            
            # Check for convergence
            shift = np.sqrt(np.sum((new_median_sampled - median_sampled) ** 2))
            median_sampled = new_median_sampled
            
            if shift < eps:
                break
                
        # Back-propagate the sampled median to the full array
        # by applying relative changes
        rel_change = median_sampled / (median_flat[indices] + eps)
        # Clip extreme values to avoid numerical issues
        rel_change = np.clip(rel_change, 0.5, 2.0)
        # Apply smooth change to full median
        median_flat *= np.mean(rel_change)
        # Update the sampled indices directly
        median_flat[indices] = median_sampled
        
        # Reshape back to original shape
        result = median_flat.reshape(orig_shape)
    else:
        # Standard approach for manageable tensors
        # Flatten for easier distance calculation
        orig_shape = X.shape[1:]
        X_flat = X.reshape(X.shape[0], -1)
        median_flat = median.reshape(-1)
        
        for i in range(max_iters):
            # Calculate distances from current median estimate
            diffs = X_flat - median_flat.reshape(1, -1)
            distances = np.sqrt((diffs ** 2).sum(axis=1) + eps)
            
            # Check for convergence
            if np.all(distances < eps):
                break
                
            # Update weights based on distances
            weights = 1.0 / np.maximum(distances, eps)
            w_sum = np.sum(weights)
            
            if w_sum < eps:
                break
                
            # Compute weighted sum of points
            weighted_points = weights.reshape(-1, 1) * X_flat
            new_median_flat = np.sum(weighted_points, axis=0) / w_sum
            
            # Check for convergence
            shift = np.sqrt(np.sum((new_median_flat - median_flat) ** 2))
            median_flat = new_median_flat
            
            if shift < eps:
                break
        
        # Reshape back to original shape
        result = median_flat.reshape(orig_shape)
    
    return result.astype(arrs[0].dtype)


def mode_connectivity_weights(arrs: List[np.ndarray], ratios: List[float], **kwargs) -> np.ndarray:
    """
    Linear interpolation along the path connecting models with distribution preservation.
    
    Args:
        arrs: List of numpy arrays to merge
        ratios: Weight ratios for each array
        
    Returns:
        Merged weights using mode connectivity (linear interpolation)
    """
    logger.debug("Computing mode connectivity (linear interpolation)")
    
    # Special case for small tensors
    if arrs[0].size < 100:
        logger.debug(f"Small tensor detected (size={arrs[0].size}), using average instead")
        return average_weights(arrs, ratios)
    
    # Normalize ratios
    ratios_normalized = np.array(ratios, dtype=np.float64)
    ratios_normalized = ratios_normalized / np.sum(ratios_normalized)
    
    # Convert to higher precision
    arrs_64 = [arr.astype(np.float64) for arr in arrs]
    base = arrs_64[0].copy()
    
    # Linear interpolation
    for i, arr in enumerate(arrs_64[1:], start=1):
        # Compute interpolation factor
        contribution = ratios_normalized[i]
        
        # Blend models
        base = (1.0 - contribution) * base + contribution * arr
    
    return base.astype(arrs[0].dtype)


def element_median_weights(arrs: List[np.ndarray], **kwargs) -> np.ndarray:
    """
    Compute element-wise median of weights, preserving discrete values.
    
    Args:
        arrs: List of numpy arrays to merge
        
    Returns:
        Merged weights using element-wise median
    """
    logger.debug("Computing element-wise median")
    
    # Stack arrays along first dimension
    stacked = np.stack([arr.astype(np.float64) for arr in arrs], axis=0)
    
    # Compute median along first axis
    result = np.median(stacked, axis=0)
    
    return result.astype(arrs[0].dtype)


def kmeans_bin_merge(arrs: List[np.ndarray], ratios: List[float], n_bins: int = 8, **kwargs) -> np.ndarray:
    """
    Merge weights using K-means clustering to preserve multimodal distributions.
    
    Args:
        arrs: List of numpy arrays to merge
        ratios: Weight ratios for each array
        n_bins: Number of clusters to use
        
    Returns:
        Merged weights preserving original clusters
    """
    logger.debug(f"Computing K-means bin merge with {n_bins} bins")
    
    # Convert to higher precision for calculations
    arrs_np = [arr.astype(np.float64) for arr in arrs]
    ratios_np = np.array(ratios, dtype=np.float64) / np.sum(ratios)
    
    # Concatenate all arrays with proper weighting for clustering
    all_values = []
    for i, arr in enumerate(arrs_np):
        # Generate weighted samples by repeating values according to ratios
        weight = int(max(1, ratios_np[i] * 1000))
        flat = arr.flatten()
        
        # For efficiency, sample if the array is very large
        if flat.size > 100000:
            sample_size = min(100000, flat.size)
            flat = np.random.choice(flat, size=sample_size, replace=False)
            
        # Add weighted values to the pool
        all_values.extend([flat] * weight)
    
    all_values = np.concatenate(all_values)
    
    # For few unique values, use a simpler approach
    unique_vals = np.unique(all_values)
    if len(unique_vals) <= n_bins:
        logger.debug(f"Found only {len(unique_vals)} unique values, using discrete preservation")
        return discrete_preserving_merge(arrs, ratios)
    
    # Determine optimal number of bins using simple heuristic
    if SCIPY_AVAILABLE and len(all_values) > 1000:
        # Try to detect the optimal number of clusters
        from scipy.cluster.vq import kmeans
        from scipy.spatial.distance import cdist
        
        # Reshape for clustering
        values_2d = all_values.reshape(-1, 1)
        
        # Compute distortion for different k values
        n_bins_to_try = min(10, len(unique_vals) // 2) if len(unique_vals) > 20 else min(n_bins, len(unique_vals))
        if n_bins_to_try < 2:
            n_bins_to_try = 2
            
        distortions = []
        for k in range(2, n_bins_to_try + 1):
            try:
                centroids, _ = kmeans(values_2d, k)
                distortions.append(sum(np.min(cdist(values_2d, centroids, 'euclidean'), axis=1)) / values_2d.shape[0])
            except:
                break
                
        if distortions:
            # Find elbow point - where adding more clusters gives diminishing returns
            deltas = np.diff(distortions)
            if len(deltas) > 1:
                k_opt = np.argmax(deltas) + 2  # +2 because we started from k=2
                n_bins = min(n_bins, k_opt)  # Use detected k, but no more than original n_bins
    
    # Use KMeans clustering to find centroids
    try:
        if SCIPY_AVAILABLE:
            from scipy.cluster.vq import kmeans
            centroids, _ = kmeans(all_values.reshape(-1, 1), n_bins)
            centroids = centroids.flatten()
        else:
            # Simple KMeans implementation for when scipy is not available
            # Initialize centroids with quantiles
            centroids = np.quantile(all_values, np.linspace(0, 1, n_bins))
            
            for _ in range(10):  # 10 iterations should be enough for 1D
                # Assign points to nearest centroid
                distances = np.abs(all_values.reshape(-1, 1) - centroids.reshape(1, -1))
                labels = np.argmin(distances, axis=1)
                
                # Update centroids
                new_centroids = np.zeros_like(centroids)
                for i in range(n_bins):
                    mask = (labels == i)
                    if np.any(mask):
                        new_centroids[i] = np.mean(all_values[mask])
                
                # Check for convergence
                if np.allclose(centroids, new_centroids):
                    break
                    
                centroids = new_centroids
    except Exception as e:
        logger.warning(f"KMeans clustering failed: {e}, falling back to discrete_preserving_merge")
        return discrete_preserving_merge(arrs, ratios)
    
    # Sort centroids for consistent mapping
    centroids = np.sort(centroids)
    
    # Now compute standard weighted average
    result = np.zeros_like(arrs_np[0])
    for i, arr in enumerate(arrs_np):
        result += arr * ratios_np[i]
    
    # Map each value to the nearest centroid
    flat_result = result.flatten()
    mapped_result = np.zeros_like(flat_result)
    
    # Find nearest centroid for each value
    for i in range(len(flat_result)):
        idx = np.argmin(np.abs(centroids - flat_result[i]))
        mapped_result[i] = centroids[idx]
    
    # Reshape back to original shape
    return mapped_result.reshape(result.shape).astype(arrs[0].dtype)


def discrete_preserving_merge(arrs: List[np.ndarray], ratios: List[float], **kwargs) -> np.ndarray:
    """
    Merge weights with special handling for discrete-valued tensors like biases.
    Significantly enhanced to better preserve bimodal distributions.
    
    Args:
        arrs: List of numpy arrays to merge
        ratios: Weight ratios for each array
        
    Returns:
        Merged weights with preserved discrete values
    """
    logger.debug("Using discrete value preserving merge strategy")
    
    # Convert to higher precision for calculations
    arrs_np = [arr.astype(np.float64) for arr in arrs]
    ratios_np = np.array(ratios, dtype=np.float64) / np.sum(ratios)
    
    # Identify all unique values across all input arrays
    all_unique_values = np.unique(np.concatenate([arr.flatten() for arr in arrs_np]))
    n_unique = len(all_unique_values)
    
    # For truly discrete tensors with few unique values
    if n_unique < 20:
        logger.debug(f"Found strongly discrete tensor with {n_unique} unique values, preserving exact values")
        
        # Find the most common unique values
        flat_concatenated = np.concatenate([arr.flatten() for arr in arrs_np])
        value_counts = {}
        for val in all_unique_values:
            # Weight the counts by the model ratios
            weighted_count = 0
            for i, arr in enumerate(arrs_np):
                count = np.sum(arr == val)
                weighted_count += count * ratios_np[i]
            value_counts[val] = weighted_count
        
        # Sort values by weighted frequency
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Compute standard weighted average
        result = np.zeros_like(arrs_np[0])
        for i, arr in enumerate(arrs_np):
            result += arr * ratios_np[i]
        
        # For each element in the result, map to the nearest original discrete value
        original_shape = result.shape
        flat_result = result.flatten()
        
        # Map each value to the nearest discrete value from the original set
        for i in range(len(flat_result)):
            # Find nearest original value
            closest_idx = np.abs(all_unique_values - flat_result[i]).argmin()
            flat_result[i] = all_unique_values[closest_idx]
        
        # Reshape back to original shape
        result = flat_result.reshape(original_shape)
    else:
        # For tensors with more unique values but still discrete-like distribution
        # Check if distribution appears bimodal or multimodal
        flat_all = np.concatenate([arr.flatten() for arr in arrs_np])
        
        # Check for multimodality
        is_multimodal = detect_multimodal_distribution(flat_all)
        
        if is_multimodal:
            logger.debug("Detected multimodal distribution, using KMeans bin merge")
            # For multimodal distributions, use kmeans-based binning
            # to preserve the modes while still allowing for blending
            return kmeans_bin_merge(arrs, ratios, n_bins=8)
        
        # Standard weighted average for unimodal distributions
        result = np.zeros_like(arrs_np[0])
        for i, arr in enumerate(arrs_np):
            result += arr * ratios_np[i]
        
        # Check if we lost too many unique values
        result_unique = len(np.unique(result))
        if result_unique < 0.7 * n_unique and n_unique > 20:
            logger.debug(f"Adjusting distribution to preserve unique value count ({result_unique} → {n_unique})")
            
            # Use histogram matching to preserve distribution characteristics
            orig_flat = flat_all
            result_flat = result.flatten()
            
            # Simple histogram matching approach
            orig_sorted = np.sort(orig_flat)
            result_sorted = np.sort(result_flat)
            
            # Create mapping from result values to matched original distribution values
            matched_values = np.zeros_like(result_flat)
            for i, val in enumerate(result_flat):
                # Find percentile of this value in result distribution
                percentile = np.searchsorted(result_sorted, val) / len(result_sorted)
                # Map to same percentile in original distribution
                idx = int(percentile * len(orig_sorted))
                idx = max(0, min(idx, len(orig_sorted) - 1))  # Clip to valid range
                matched_values[i] = orig_sorted[idx]
            
            # Reshape back to original shape
            result = matched_values.reshape(result.shape)
    
    return result.astype(arrs[0].dtype)


def enhanced_calibration(merged: np.ndarray, arrs: List[np.ndarray], ratios: List[float], 
                        preserve_moments: int = 2, **kwargs) -> np.ndarray:
    """
    Advanced distribution calibration that preserves multiple statistical moments.
    
    Args:
        merged: Merged tensor to calibrate
        arrs: Original input tensors
        ratios: Weight ratios for each input tensor
        preserve_moments: Number of statistical moments to preserve (1=mean, 2=std, 3=skew, 4=kurtosis)
        
    Returns:
        Calibrated tensor with preserved distribution characteristics
    """
    # For small tensors, don't apply calibration
    if merged.size < 100:
        return merged
    
    # Convert to higher precision
    merged_64 = merged.astype(np.float64)
    arrs_64 = [arr.astype(np.float64) for arr in arrs]
    ratios_np = np.array(ratios, dtype=np.float64) / np.sum(ratios)
    
    # Reshape to 1D for processing
    orig_shape = merged.shape
    merged_flat = merged_64.flatten()
    
    # Compute target statistics from weighted inputs
    # Get samples from input arrays for efficiency with large tensors
    max_samples = 100000
    if sum(arr.size for arr in arrs) > max_samples:
        # Sample from each array proportional to its ratio
        all_samples = []
        for arr, ratio in zip(arrs_64, ratios_np):
            flat = arr.flatten()
            n_samples = max(1, int(max_samples * ratio))
            if flat.size > n_samples:
                samples = np.random.choice(flat, size=n_samples, replace=False)
            else:
                samples = flat
            all_samples.append(samples)
        target_samples = np.concatenate(all_samples)
    else:
        # Use all values if arrays are small enough
        target_samples = np.concatenate([arr.flatten() for arr in arrs_64])
    
    # Compute target moments
    target_mean = np.mean(target_samples)
    target_std = np.std(target_samples)
    
    # Compute current moments
    current_mean = np.mean(merged_flat)
    current_std = np.std(merged_flat)
    
    # Skip if either std is close to zero
    if target_std < 1e-6 or current_std < 1e-6:
        return merged
    
    # Apply calibration based on number of moments to preserve
    if preserve_moments >= 2:  # Preserve mean and std (Z-score transformation)
        calibrated = (merged_flat - current_mean) / current_std * target_std + target_mean
    else:  # Preserve only mean (shift)
        calibrated = merged_flat + (target_mean - current_mean)
    
    # For higher-order moments, use more sophisticated techniques
    if preserve_moments >= 3 and SCIPY_AVAILABLE and target_samples.size > 100:
        try:
            # Compute skewness and kurtosis
            from scipy import stats
            
            # If distribution is significantly non-normal, use full histogram matching
            target_skew = stats.skew(target_samples)
            target_kurt = stats.kurtosis(target_samples)
            current_skew = stats.skew(calibrated)
            current_kurt = stats.kurtosis(calibrated)
            
            # Check if higher moments are significantly different
            if (abs(target_skew - current_skew) > 0.2 or 
                abs(target_kurt - current_kurt) > 0.5):
                
                # Use rank-based matching (preserves all moments)
                # Sort both arrays
                target_sorted = np.sort(target_samples)
                current_sorted = np.sort(calibrated)
                
                # For each value in calibrated, find corresponding percentile in target
                result = np.zeros_like(calibrated)
                for i, val in enumerate(calibrated):
                    # Find percentile of this value in current distribution
                    percentile = np.searchsorted(current_sorted, val) / len(current_sorted)
                    # Map to same percentile in target distribution
                    idx = int(percentile * len(target_sorted))
                    idx = max(0, min(idx, len(target_sorted) - 1))
                    result[i] = target_sorted[idx]
                
                calibrated = result
        except Exception as e:
            logger.debug(f"Higher-moment calibration failed: {e}, using standard calibration")
    
    # For discrete distributions, snap to common values
    if len(np.unique(merged_flat)) < 100:
        unique_vals = np.unique(target_samples)
        if len(unique_vals) < 100:
            # Map to nearest discrete values
            for i in range(len(calibrated)):
                closest_idx = np.abs(unique_vals - calibrated[i]).argmin()
                calibrated[i] = unique_vals[closest_idx]
    
    # Reshape back to original shape
    return calibrated.reshape(orig_shape).astype(merged.dtype)


def adaptive_merge(arrs: List[np.ndarray], ratios: List[float], **kwargs) -> np.ndarray:
    """
    Smart merging strategy that automatically selects the best approach
    based on tensor characteristics.
    
    Args:
        arrs: List of numpy arrays to merge
        ratios: Weight ratios for each array
        
    Returns:
        Merged weights using the best strategy for this tensor
    """
    logger.debug("Using adaptive merge strategy")
    
    # Sample the tensor to analyze its distribution
    sample_size = min(10000, arrs[0].size)
    if arrs[0].size > sample_size:
        samples = [np.random.choice(arr.flatten(), sample_size, replace=False) for arr in arrs]
    else:
        samples = [arr.flatten() for arr in arrs]
    
    # Check for discreteness
    is_discrete = detect_discrete_multimodal_tensor(arrs[0])
    is_small = arrs[0].size < 100
    is_bias = 'bias' in getattr(arrs[0], 'name', '') or arrs[0].ndim == 1
    
    # Decide on merge strategy
    if is_discrete or is_small or is_bias:
        # For discrete or small tensors
        if detect_multimodal_distribution(np.concatenate(samples)):
            # For multimodal distributions, use K-means binning
            logger.debug("Detected multimodal discrete distribution, using kmeans bin merge")
            merged = kmeans_bin_merge(arrs, ratios)
        else:
            # For unimodal discrete distributions
            logger.debug("Detected unimodal discrete distribution, using discrete preserving merge")
            merged = discrete_preserving_merge(arrs, ratios)
    else:
        # For continuous tensors
        # Compute standard deviation of each input
        stds = [np.std(sample) for sample in samples]
        
        # If standard deviations vary widely, trimmed mean may be better
        if max(stds) > 2.0 * min(stds):
            logger.debug("Detected widely varying distributions, using trimmed mean")
            merged = trimmed_mean_weights(arrs, trim_ratio=0.2, ratios=ratios)
        else:
            # Otherwise, use standard weighted average
            logger.debug("Using standard weighted average for well-behaved distribution")
            merged = average_weights(arrs, ratios)
    
    # Apply calibration to ensure distribution characteristics are preserved
    result = enhanced_calibration(merged, arrs, ratios, preserve_moments=3)
    
    return result


def svd_sign_align(arrs: List[np.ndarray], **kwargs) -> List[np.ndarray]:
    """
    Align signs of weights using SVD for optimal matching.
    Handles both convolutional and linear weights.
    
    Args:
        arrs: List of weight arrays to align
        
    Returns:
        Sign-aligned weight arrays
    """
    # Only process arrays with ndim >= 2 (e.g., weights, not biases)
    if arrs[0].ndim < 2:
        return arrs.copy()
    
    # Use the first array as reference
    ref = arrs[0]
    aligned = [ref.copy()]  # Reference doesn't change
    
    for arr in arrs[1:]:
        if arr.ndim == 2:  # Linear layer (out_features, in_features)
            # For linear layers, we can flip signs of output neurons independently
            flipped = arr.copy()
            
            # Compute correlation for each output neuron
            for i in range(arr.shape[0]):
                ref_row = ref[i]
                arr_row = arr[i]
                corr = np.sum(ref_row * arr_row)
                if corr < 0:
                    flipped[i] = -arr_row
            
            aligned.append(flipped)
        elif arr.ndim == 4:  # Convolutional layer (out_channels, in_channels, height, width)
            # For convolutional layers, we can flip signs of output filters
            flipped = arr.copy()
            
            # Flatten the kernel dimensions for correlation
            ref_flat = ref.reshape(ref.shape[0], -1)
            arr_flat = arr.reshape(arr.shape[0], -1)
            
            # Compute correlation for each output filter
            for i in range(arr.shape[0]):
                ref_filter = ref_flat[i]
                arr_filter = arr_flat[i]
                corr = np.sum(ref_filter * arr_filter)
                if corr < 0:
                    flipped[i] = -arr[i]
            
            aligned.append(flipped)
        else:
            # For other shapes, don't modify
            aligned.append(arr.copy())
    
    return aligned


def permutation_align_weights(arrs: List[np.ndarray], **kwargs) -> List[np.ndarray]:
    """
    Align weights using permutation matching for layers with permutation symmetry.
    
    Args:
        arrs: List of weight arrays to align
        
    Returns:
        Permutation-aligned weight arrays
    """
    # Only applicable for specific layer types (fully connected, convolutions)
    if arrs[0].ndim < 2 or not SCIPY_AVAILABLE:
        return arrs
    
    # Use first array as reference
    ref = arrs[0]
    aligned = [ref.copy()]
    
    # Handle different layer types
    if ref.ndim == 2:  # Linear layers
        # Try to match neurons by similarity
        for arr in arrs[1:]:
            aligned_arr = arr.copy()
            
            # Compute pairwise similarities
            cost_matrix = cdist(ref, arr, 'cosine')
            
            # Use Hungarian algorithm to find optimal assignment
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Permute columns according to optimal assignment
                aligned_arr = arr[col_ind]
                
                # Apply sign flipping where needed
                for i in range(aligned_arr.shape[0]):
                    corr = np.sum(ref[i] * aligned_arr[i])
                    if corr < 0:
                        aligned_arr[i] = -aligned_arr[i]
            except Exception as e:
                logger.debug(f"Permutation alignment failed: {e}")
            
            aligned.append(aligned_arr)
    elif ref.ndim == 4:  # Conv layers
        # Similar approach but with flattened kernels
        for arr in arrs[1:]:
            aligned_arr = arr.copy()
            
            # Flatten kernels for similarity computation
            ref_flat = ref.reshape(ref.shape[0], -1)
            arr_flat = arr.reshape(arr.shape[0], -1)
            
            # Compute pairwise similarities
            cost_matrix = cdist(ref_flat, arr_flat, 'cosine')
            
            # Use Hungarian algorithm to find optimal assignment
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Permute according to optimal assignment
                for i, j in enumerate(col_ind):
                    aligned_arr[i] = arr[j]
                    
                    # Also check if sign flipping is needed
                    corr = np.sum(ref_flat[i] * arr_flat[j])
                    if corr < 0:
                        aligned_arr[i] = -aligned_arr[i]
            except Exception as e:
                logger.debug(f"Permutation alignment failed: {e}")
            
            aligned.append(aligned_arr)
    else:
        # For other shapes, don't modify
        for arr in arrs[1:]:
            aligned.append(arr.copy())
    
    return aligned


MERGE_METHODS: Dict[str, Callable[..., np.ndarray]] = {
    "average": average_weights,
    "trimmed_mean": trimmed_mean_weights,
    "geometric_median": geometric_median_weights,
    "mode_connectivity": mode_connectivity_weights,
    "discrete_preserving": discrete_preserving_merge,
    "element_median": element_median_weights,
    "kmeans_bin": kmeans_bin_merge,
    "adaptive": adaptive_merge
}

# ------------------------------------------------------------------------------
# ONNX helpers (with metadata extraction)
# ------------------------------------------------------------------------------

def load_onnx_weights(path: str) -> Tuple[Dict[str, np.ndarray], onnx.ModelProto, Dict[str, str]]:
    """
    Load weights, model, and metadata from an ONNX file.
    
    Args:
        path: Path to the ONNX model file
        
    Returns:
        Tuple of (weights, model, metadata)
    """
    logger.info(f"Loading ONNX model from {path}")
    model = onnx.load(path)
    
    # Get initial file size for reference
    model_size = os.path.getsize(path) / 1024.0
    logger.info(f"Input model size: {model_size:.2f} KB")
    
    weights = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    if metadata:
        logger.info(f"Found {len(metadata)} metadata entries in {os.path.basename(path)}")
    return weights, model, metadata


def save_onnx_weights(model: onnx.ModelProto, merged: Dict[str, np.ndarray], output_path: str):
    """
    Save merged weights to an ONNX file, preserving exact model structure and data types.
    
    Args:
        model: ONNX model to update
        merged: Dictionary of merged weights
        output_path: Path to save the merged model
    """
    logger.info(f"Saving merged ONNX model to {output_path}")
    
    # Analyze original data types
    original_dtypes = {}
    for init in model.graph.initializer:
        if hasattr(init, 'data_type'):
            original_dtypes[init.name] = init.data_type
    
    # Update weights in the model, preserving data types exactly
    for init in model.graph.initializer:
        if init.name in merged:
            try:
                # Get original tensor's data type
                original_tensor = init
                original_data_type = original_dtypes.get(init.name, None)
                
                # Get merged array
                new_array = merged[init.name]
                
                # Check for NaN/Inf
                if np.isnan(new_array).any() or np.isinf(new_array).any():
                    logger.warning(f"NaN or Inf values in merged tensor {init.name}, using original")
                    continue
                
                # Exactly match original tensor's data type
                if original_data_type is not None:
                    if original_data_type == onnx.TensorProto.FLOAT:
                        new_array = new_array.astype(np.float32)
                    elif original_data_type == onnx.TensorProto.DOUBLE:
                        new_array = new_array.astype(np.float64)
                    elif original_data_type == onnx.TensorProto.FLOAT16:
                        new_array = new_array.astype(np.float16)
                    elif original_data_type == onnx.TensorProto.INT32:
                        new_array = new_array.astype(np.int32)
                    elif original_data_type == onnx.TensorProto.INT64:
                        new_array = new_array.astype(np.int64)
                    elif original_data_type == onnx.TensorProto.INT16:
                        new_array = new_array.astype(np.int16)
                    elif original_data_type == onnx.TensorProto.INT8:
                        new_array = new_array.astype(np.int8)
                    elif original_data_type == onnx.TensorProto.UINT8:
                        new_array = new_array.astype(np.uint8)
                    elif original_data_type == onnx.TensorProto.BOOL:
                        new_array = new_array.astype(np.bool_)
                
                # Verify shapes match exactly
                if new_array.shape != tuple(d for d in init.dims):
                    logger.warning(f"Shape mismatch for {init.name}: {new_array.shape} vs {tuple(d for d in init.dims)}")
                    continue
                    
                # Create new tensor with the exact data type
                new_tensor = numpy_helper.from_array(new_array, name=init.name)
                
                # Replace the tensor in the model
                init.CopyFrom(new_tensor)
                logger.debug(f"Updated tensor {init.name} with shape {new_array.shape}")
                
            except Exception as e:
                logger.error(f"Failed to update tensor {init.name}: {e}")
                logger.warning(f"Keeping original weights for {init.name}")
    
    # Save the model with no modifications to structure or metadata
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.save(model, output_path)
    
    # Report final model size
    output_size = os.path.getsize(output_path) / 1024.0
    logger.info(f"Output model size: {output_size:.2f} KB")


def export_merge_metadata(
    input_metadata_list: List[Dict[str, str]],
    method: str,
    ratios: List[float],
    output_path: str
) -> None:
    """
    Export merge information to an external file without modifying the model's metadata.
    
    Args:
        input_metadata_list: List of metadata dicts loaded from each input model
        method: Name of the merge method used
        ratios: Merge ratios for each input model
        output_path: Path to save the metadata
    """
    import time
    import json
    
    # Create external merge info file
    merge_info = {
        "merge_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "merge_method": method,
        "merge_ratios": ratios,
        "input_metadata": input_metadata_list
    }
    
    # Save as JSON file alongside the model
    json_path = os.path.splitext(output_path)[0] + ".merge_info.json"
    with open(json_path, "w") as f:
        json.dump(merge_info, f, indent=2)
    
    logger.info(f"Exported merge information to {json_path}")

# ------------------------------------------------------------------------------
# Plugin system
# ------------------------------------------------------------------------------


def load_plugin(path: str) -> Callable[[List[np.ndarray]], np.ndarray]:
    """
    Load a plugin module that provides a custom merge function.
    
    Args:
        path: Path to Python file containing merge function
        
    Returns:
        Merge function from the plugin
    """
    spec = importlib.util.spec_from_file_location("plugin_module", path)
    if spec is None:
        raise ValueError(f"Could not load plugin from {path}")
        
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ValueError(f"Could not load plugin from {path} (no loader available)")
        
    spec.loader.exec_module(mod)
    
    if not hasattr(mod, "merge"):
        raise ValueError(f"Plugin {path} must define a 'merge(arrs: List[np.ndarray]) -> np.ndarray' function.")
    logger.info(f"Loaded plugin: {path}")
    return mod.merge

# ------------------------------------------------------------------------------
# Utility: distribution visualization & analysis
# ------------------------------------------------------------------------------


def visualize_distribution(before: List[np.ndarray], after: np.ndarray, name: str, outdir: str):
    """
    Create a histogram visualization of weight distributions before and after merging.
    
    Args:
        before: List of arrays before merging
        after: Array after merging
        name: Name of the layer
        outdir: Directory to save plots
    """
    os.makedirs(outdir, exist_ok=True)
    
    # For small tensors, use stem plot instead of histogram to better show discrete values
    if before[0].size < 100:
        plt.figure(figsize=(12, 6))
        
        # Split into two subplots for before and after
        plt.subplot(1, 2, 1)
        all_before = np.concatenate([arr.flatten() for arr in before])
        # Get unique values and their frequencies
        unique, counts = np.unique(all_before, return_counts=True)
        plt.stem(unique, counts, linefmt='C0-', markerfmt='C0o', basefmt='C0-', label="before")
        plt.title("Before merge (unique values)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        
        plt.subplot(1, 2, 2)
        after_flat = after.flatten()
        unique, counts = np.unique(after_flat, return_counts=True)
        plt.stem(unique, counts, linefmt='C1-', markerfmt='C1o', basefmt='C1-', label="after")
        plt.title("After merge (unique values)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
    else:
        # For larger tensors with distinct distributions, use dual plot
        plt.figure(figsize=(15, 6))
        
        # Create two subplots - standard histogram and density estimation
        plt.subplot(1, 2, 1)
        
        # Sample from arrays for efficiency
        max_samples = 50000
        sample_sizes = [min(arr.size, max_samples) for arr in before]
        total_samples = sum(sample_sizes)
        
        # Weight samples by array ratios for before plot
        all_before = np.concatenate([
            np.random.choice(arr.flatten(), size=size, replace=False)
            for arr, size in zip(before, sample_sizes)
        ])
        
        # Match sample size for after plot
        after_flat = after.flatten()
        if after_flat.size > total_samples:
            after_samples = np.random.choice(after_flat, size=total_samples, replace=False)
        else:
            after_samples = after_flat
        
        # Plot standard histogram
        bins = min(100, max(20, int(np.sqrt(total_samples))))
        plt.hist(all_before, bins=bins, alpha=0.5, label="before", density=True)
        plt.hist(after_samples, bins=bins, alpha=0.5, label="after", density=True)
        plt.title("Weight Distribution")
        plt.legend()
        
        # Second subplot for kernel density estimation
        plt.subplot(1, 2, 2)
        
        # Use KDE if scipy is available
        if SCIPY_AVAILABLE:
            try:
                from scipy import stats
                # Estimate densities
                kde_before = stats.gaussian_kde(all_before)
                kde_after = stats.gaussian_kde(after_samples)
                
                # Create evaluation points
                all_min = min(np.min(all_before), np.min(after_samples))
                all_max = max(np.max(all_before), np.max(after_samples))
                x = np.linspace(all_min, all_max, 500)
                
                # Plot density curves
                plt.plot(x, kde_before(x), label="before", linewidth=2)
                plt.plot(x, kde_after(x), label="after", linewidth=2)
                plt.title("Density Estimation")
                plt.legend()
            except Exception as e:
                # Fallback to histogram if KDE fails
                logger.debug(f"KDE plot failed: {e}, using histogram")
                plt.hist(all_before, bins=bins, alpha=0.5, label="before", density=True)
                plt.hist(after_samples, bins=bins, alpha=0.5, label="after", density=True)
                plt.title("Weight Distribution")
                plt.legend()
        else:
            # Fallback when scipy not available
            plt.hist(all_before, bins=bins, alpha=0.5, label="before", density=True)
            plt.hist(after_samples, bins=bins, alpha=0.5, label="after", density=True)
            plt.title("Weight Distribution")
            plt.legend()
    
    # Add metrics information to the title
    metrics = compute_distribution_metrics(before, after)
    plt.figtext(0.5, 0.01, 
                f"Wasserstein dist={metrics['wasserstein_dist']:.4f}, "
                f"Std ratio={metrics['std_ratio']:.4f}, "
                f"Unique ratio={metrics['unique_ratio']:.4f}",
                ha="center", fontsize=9, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    # Save plot with safe filename
    safe_name = name.replace('/', '_').replace('.', '_')
    plot_path = os.path.join(outdir, f"{safe_name}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    logger.debug(f"Saved distribution plot for {name} at {plot_path}")

# ------------------------------------------------------------------------------
# Core merging logic (restricted to reference model's keys)
# ------------------------------------------------------------------------------


def merge_state_dicts(
    dicts: List[Dict[str, np.ndarray]],
    method: str,
    ratios: List[float],
    trim_ratio: float,
    geo_eps: float,
    prefixes: Optional[List[str]],
    plugins: List[Callable[[List[np.ndarray]], np.ndarray]],
    plot_dir: Optional[str],
    tb_writer: Optional[Any],
    skip_mismatch: bool,
    force_merge: bool = False,
    align_weights: bool = True,
    calibration_level: int = 2,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, Dict[str, float], Dict[str, float], Dict[str, float], str]]]:
    """
    Enhanced merge function with comprehensive analysis and safeguards.
    
    Args:
        dicts: List of weight dictionaries to merge
        method: Merge method to use
        ratios: Weight ratios for each input model
        trim_ratio: Trim ratio for trimmed_mean method
        geo_eps: Epsilon for geometric_median method
        prefixes: Only merge layers with these prefixes
        plugins: Custom merge functions from plugins
        plot_dir: Directory to save distribution plots
        tb_writer: TensorBoard writer
        skip_mismatch: Skip layers with mismatched shapes
        force_merge: Force merge even when distribution shifts significantly
        align_weights: Perform weight alignment before merging
        calibration_level: Number of moments to preserve in calibration (0-3)
        
    Returns:
        Tuple of (merged_weights, summary_statistics)
    """
    # Suppress NumPy warnings during merge operation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        start = time.time()
        merged: Dict[str, np.ndarray] = {}
        # Add distribution metrics to the summary tuple
        summary: List[Tuple[str, Dict[str, float], Dict[str, float], Dict[str, float], str]] = []
        
        # Track what merge methods were actually used
        method_usage_counter = Counter()
        
        # Prepare hierarchical fallback strategies for problematic tensors
        fallback_hierarchy = {
            "average": ["discrete_preserving", "kmeans_bin", "element_median", "reference"],
            "trimmed_mean": ["average", "discrete_preserving", "element_median", "reference"],
            "geometric_median": ["element_median", "discrete_preserving", "average", "reference"],
            "mode_connectivity": ["average", "discrete_preserving", "reference"],
            "discrete_preserving": ["kmeans_bin", "element_median", "average", "reference"],
            "element_median": ["discrete_preserving", "kmeans_bin", "average", "reference"],
            "kmeans_bin": ["discrete_preserving", "element_median", "average", "reference"],
            "adaptive": ["discrete_preserving", "kmeans_bin", "average", "element_median", "reference"]
        }

        reference_keys = list(dicts[0].keys())
        extra = set().union(*(d.keys() for d in dicts[1:])) - set(reference_keys)
        if extra:
            logger.warning(f"Ignoring {len(extra)} extra layers not in reference model: {sorted(extra)[:5]}...")

        # Calculate total parameters for reporting
        total_params = 0
        
        # Group layers by type for more informed logging
        layer_types = {
            'bias': [],
            'weight': [],
            'other': []
        }
        
        for name in sorted(reference_keys):
            # Categorize layer by type
            if 'bias' in name.lower():
                layer_types['bias'].append(name)
            elif 'weight' in name.lower():
                layer_types['weight'].append(name)
            else:
                layer_types['other'].append(name)
        
        logger.info(f"Processing {len(layer_types['weight'])} weight layers, {len(layer_types['bias'])} bias layers")
        
        # Process all layers
        for name in sorted(reference_keys):
            arrs, missing = [], []
            for d in dicts:
                if name in d:
                    arrs.append(d[name])
                else:
                    missing.append(name)
            if missing:
                msg = f"Layer '{name}' missing in some models"
                if skip_mismatch:
                    logger.warning(msg + "; copying reference weights")
                    merged[name] = arrs[0].copy()  # Make a copy to avoid reference issues
                    total_params += np.prod(arrs[0].shape)
                    method_usage_counter["reference"] += 1
                    continue
                else:
                    raise ValueError(msg + "; use --skip-mismatch to bypass")

            if prefixes and not any(name.startswith(p) for p in prefixes):
                merged[name] = arrs[0].copy()  # Make a copy to avoid reference issues
                total_params += np.prod(arrs[0].shape)
                method_usage_counter["reference"] += 1
                logger.debug(f"Skipping layer '{name}' (prefix filter)")
                continue

            shapes = [a.shape for a in arrs]
            if len(set(str(s) for s in shapes)) != 1:  # Convert shapes to strings for comparison
                msg = f"Layer '{name}' has mismatched shapes {shapes}"
                if skip_mismatch:
                    logger.warning(msg + "; copying reference weights")
                    merged[name] = arrs[0].copy()  # Make a copy to avoid reference issues
                    total_params += np.prod(arrs[0].shape)
                    method_usage_counter["reference"] += 1
                    continue
                else:
                    raise ValueError(msg + "; use --skip-mismatch to bypass")

            # Check for NaN or Inf values in inputs
            has_nan = any(np.isnan(arr).any() for arr in arrs)
            has_inf = any(np.isinf(arr).any() for arr in arrs)
            
            if has_nan or has_inf:
                logger.warning(f"Layer '{name}' contains NaN or Inf values in input; copying reference weights")
                merged[name] = arrs[0].copy()  # Make a copy to avoid reference issues
                total_params += np.prod(arrs[0].shape)
                method_usage_counter["reference"] += 1
                continue
            
            # Special handling for layer types
            is_small_tensor = arrs[0].size < 100
            is_bias = 'bias' in name.lower() or arrs[0].ndim == 1
            is_discrete = any(detect_discrete_multimodal_tensor(arr) for arr in arrs)
            is_multimodal = detect_multimodal_distribution(np.concatenate([arr.flatten()[:min(10000, arr.size)] for arr in arrs]))
            
            # Choose best merging strategy based on tensor characteristics
            merge_strategy = method
            
            if is_bias and is_multimodal:
                # For multimodal bias tensors, kmeans binning works well
                logger.debug(f"Layer '{name}' is a multimodal bias tensor, using kmeans bin merge")
                merge_strategy = "kmeans_bin"
            elif is_bias or is_discrete:
                # For bias or other discrete tensors, use discrete preservation
                logger.debug(f"Layer '{name}' is a discrete tensor, using discrete preserving merge")
                merge_strategy = "discrete_preserving"
            elif is_small_tensor:
                if method == "trimmed_mean" or method == "geometric_median":
                    # For small tensors, these methods might not be appropriate
                    logger.debug(f"Layer '{name}' is small tensor, using average instead of {method}")
                    merge_strategy = "average"
            
            # Perform weight alignment if enabled and appropriate
            aligned_arrs = arrs
            if align_weights and not is_bias and arrs[0].ndim >= 2:
                if 'weight' in name.lower() and ('conv' in name.lower() or 'linear' in name.lower() or 'fc' in name.lower()):
                    logger.debug(f"Performing sign alignment for layer '{name}'")
                    aligned_arrs = svd_sign_align(arrs)
                    
                    # For problematic layers, also try permutation alignment
                    if is_multimodal and SCIPY_AVAILABLE:
                        logger.debug(f"Attempting permutation alignment for multimodal layer '{name}'")
                        aligned_arrs = permutation_align_weights(aligned_arrs)
            
            # Try merge with plugin first
            out = None
            merge_method_used = "custom_plugin"
            for plugin in plugins:
                try:
                    out = plugin(aligned_arrs)
                    logger.debug(f"Layer '{name}' merged via plugin")
                    break
                except Exception as e:
                    logger.debug(f"Plugin error on layer '{name}': {e}")

            # Fall back to built-in methods if plugin didn't work
            if out is None:
                try:
                    if merge_strategy in MERGE_METHODS:
                        # Try the selected strategy first
                        params = {
                            "ratios": ratios,
                            "trim_ratio": trim_ratio,
                            "eps": geo_eps,
                        }
                        
                        merge_func = MERGE_METHODS[merge_strategy]
                        out = merge_func(aligned_arrs, **params)
                        merge_method_used = merge_strategy
                    else:
                        logger.warning(f"Unknown merge strategy '{merge_strategy}', falling back to {method}")
                        # Fall back to original method
                        params = {
                            "ratios": ratios,
                            "trim_ratio": trim_ratio,
                            "eps": geo_eps,
                        }
                        
                        merge_func = MERGE_METHODS[method]
                        out = merge_func(aligned_arrs, **params)
                        merge_method_used = method
                except Exception as e:
                    logger.warning(f"Error in primary merge for '{name}': {e}")
                    
                    # Try fallback strategies in hierarchical order
                    fallbacks = fallback_hierarchy.get(merge_strategy, fallback_hierarchy["average"])
                    success = False
                    
                    for fallback in fallbacks:
                        if fallback == "reference":
                            # Final fallback is always just using reference weights
                            out = arrs[0].copy()
                            merge_method_used = "reference"
                            success = True
                            break
                            
                        try:
                            logger.debug(f"Trying fallback strategy '{fallback}' for layer '{name}'")
                            params = {
                                "ratios": ratios,
                                "trim_ratio": trim_ratio,
                                "eps": geo_eps,
                            }
                            
                            merge_func = MERGE_METHODS.get(fallback)
                            if merge_func:
                                out = merge_func(aligned_arrs, **params)
                                merge_method_used = f"{fallback}_fallback"
                                success = True
                                break
                        except Exception as fallback_err:
                            logger.debug(f"Fallback '{fallback}' also failed for '{name}': {fallback_err}")
                    
                    if not success:
                        # If all fallbacks failed, use reference weights
                        logger.warning(f"All merge strategies failed for layer '{name}', using reference weights")
                        out = arrs[0].copy()
                        merge_method_used = "reference"

            # Validate output for NaN or Inf
            if np.isnan(out).any() or np.isinf(out).any():
                logger.warning(f"Merge produced NaN or Inf values for layer '{name}'; using reference weights")
                out = arrs[0].copy()  # Make a copy to avoid reference issues
                merge_method_used = "reference"

            # Compute distribution metrics before calibration
            dist_metrics_before = compute_distribution_metrics(arrs, out)
            
            # Apply distribution calibration if needed
            if merge_method_used != "reference" and calibration_level > 0:
                # Check for significant distribution contraction
                std_ratio = dist_metrics_before["std_ratio"]
                
                if std_ratio < 0.2 and not force_merge:
                    logger.warning(
                        f"Layer '{name}' has significant distribution contraction (std ratio: {std_ratio:.3f}). "
                        f"Using reference weights. Use --force-merge to override."
                    )
                    out = arrs[0].copy()
                    merge_method_used = "reference"
                elif merge_method_used != "reference":
                    # Apply calibration based on the severity of the contraction
                    # Stronger calibration for more severe contraction
                    actual_level = calibration_level
                    if std_ratio < 0.5:
                        # For severe contraction, increase calibration level
                        actual_level = min(3, calibration_level + 1)
                    
                    logger.debug(f"Applying distribution calibration (level {actual_level}) for layer '{name}'")
                    out = enhanced_calibration(out, arrs, ratios, preserve_moments=actual_level)
                    
                    if "_fallback" in merge_method_used:
                        merge_method_used += "+calibrated"
                    else:
                        merge_method_used += "+calibrated"
            
            # Track which method was actually used
            method_usage_counter[merge_method_used] += 1

            # Preserve the original data type to avoid precision bloat
            orig_dtype = arrs[0].dtype
            if out.dtype != orig_dtype:
                logger.debug(f"Converting layer '{name}' from {out.dtype} back to original {orig_dtype}")
                out = out.astype(orig_dtype)
                
            total_params += np.prod(out.shape)
            
            # Compute stats safely
            try:
                stats_before = compute_stats(np.concatenate([a.flatten() for a in arrs]))
                stats_after = compute_stats(out)
                
                # Calculate comprehensive distribution metrics
                dist_metrics = compute_distribution_metrics(arrs, out)
                
                # Check for excessive unique value loss
                if dist_metrics["unique_ratio"] < 0.4 and arrs[0].size < 1000:
                    logger.warning(
                        f"Layer '{name}' lost many unique values (ratio: {dist_metrics['unique_ratio']:.3f}). "
                        f"This may indicate poor distribution preservation."
                    )
                
                summary.append((name, stats_before, stats_after, dist_metrics, merge_method_used))
                
                # For important layers, log detailed metrics
                if is_bias or is_small_tensor or is_discrete:
                    logger.info(
                        f"Layer '{name}' ({merge_method_used}): "
                        f"unique={stats_before['unique_values']}->{stats_after['unique_values']}, "
                        f"std={stats_before['std']:.4f}->{stats_after['std']:.4f}, "
                        f"W-dist={dist_metrics['wasserstein_dist']:.4f}"
                    )
                else:
                    logger.debug(
                        f"Layer '{name}' ({merge_method_used}): "
                        f"std_ratio={dist_metrics['std_ratio']:.3f}, "
                        f"unique_ratio={dist_metrics['unique_ratio']:.3f}, "
                        f"W-dist={dist_metrics['wasserstein_dist']:.4f}"
                    )
                
            except Exception as e:
                logger.warning(f"Error computing stats for layer '{name}': {e}")
                # Still include the layer in merged output even if stats computation fails

            # Generate visualizations if requested
            if plot_dir:
                try:
                    visualize_distribution(arrs, out, name, plot_dir)
                except Exception as e:
                    logger.warning(f"Failed to create distribution plot for layer '{name}': {e}")
                    
            # Add TensorBoard logging if enabled
            if tb_writer:
                try:
                    tb_writer.add_histogram(f"{name}/before", np.concatenate([a.flatten() for a in arrs]), 0)
                    tb_writer.add_histogram(f"{name}/after", out.flatten(), 0)
                    tb_writer.add_scalar(f"{name}/wasserstein_dist", dist_metrics["wasserstein_dist"], 0)
                    tb_writer.add_scalar(f"{name}/std_ratio", dist_metrics["std_ratio"], 0)
                    tb_writer.add_scalar(f"{name}/unique_ratio", dist_metrics["unique_ratio"], 0)
                except Exception as e:
                    logger.warning(f"TensorBoard logging failed for layer '{name}': {e}")

            merged[name] = out

        elapsed = time.time() - start
        logger.info(f"Merged {len(summary)} layers in {elapsed:.2f}s (skipped {len(reference_keys) - len(summary)})")
        logger.info(f"Total parameters in merged model: {total_params:,}")
        
        # Print a table of merge method usage
        logger.info("Merge methods used:")
        for method_name, count in method_usage_counter.most_common():
            logger.info(f"  {method_name}: {count} layers")
        
        return merged, summary

# ------------------------------------------------------------------------------
# I/O helpers (unified for PyTorch & ONNX, returning metadata)
# ------------------------------------------------------------------------------


def load_model(path: str) -> Tuple[Dict[str, np.ndarray], Any, Dict[str, str]]:
    """
    Load a model from file, supporting PyTorch (.pth, .pt) and ONNX (.onnx) formats.
    
    Args:
        path: Path to model file
        
    Returns:
        Tuple of (weights, model_proto, metadata)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pth", ".pt"):
        logger.info(f"Loading PyTorch model from {path}")
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and not any(isinstance(v, torch.Tensor) for v in sd.values()):
            sd = sd.get("state_dict", sd)
        np_dict = {k: v.cpu().numpy() for k, v in sd.items()}
        return np_dict, None, {}
    elif ext == ".onnx":
        w, model, meta = load_onnx_weights(path)
        return w, model, meta
    else:
        raise ValueError(f"Unsupported model format: {ext}")


def save_model(merged: Dict[str, np.ndarray], reference_proto: Any, output_path: str):
    """
    Save a merged model to file, supporting PyTorch and ONNX formats.
    
    Args:
        merged: Dictionary of merged weights
        reference_proto: Reference model (ONNX ModelProto for ONNX models)
        output_path: Path to save merged model
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".pth", ".pt"):
        logger.info(f"Saving merged PyTorch model to {output_path}")
        torch.save({k: torch.from_numpy(v) for k, v in merged.items()}, output_path)
    elif ext == ".onnx":
        save_onnx_weights(reference_proto, merged, output_path)
    else:
        raise ValueError(f"Unsupported output format: {ext}")

# ------------------------------------------------------------------------------
# Main entrypoint
# ------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Merge PyTorch or ONNX models with advanced distribution preservation.")
    parser.add_argument("models", nargs="+", help="Paths to model files (.pth, .pt, .onnx)")
    parser.add_argument("--method", required=True, 
                        choices=list(MERGE_METHODS) + ["plugin"],
                        help="Merge method to use")
    parser.add_argument("--ratios", type=float, nargs="+",
                        help="Weights for each model (must correspond to number of models)")
    parser.add_argument("--trim_ratio", type=float, default=0.1,
                        help="Trim ratio for trimmed_mean")
    parser.add_argument("--geo_eps", type=float, default=1e-5,
                        help="Epsilon for geometric_median convergence")
    parser.add_argument("--prefixes", nargs="+",
                        help="Only merge layers starting with these prefixes")
    parser.add_argument("--plugins", nargs="+", default=[],
                        help="Paths to plugin Python files providing custom merge()")
    parser.add_argument("--plot-dir", help="Directory to save distribution plots")
    parser.add_argument("--tensorboard-logdir", help="Directory for TensorBoard logs")
    parser.add_argument("--skip-mismatch", action="store_true",
                        help="Skip layers missing or mismatched in shape")
    parser.add_argument("--force-merge", action="store_true",
                        help="Force merge even when distribution shifts significantly")
    parser.add_argument("--no-align", action="store_true",
                        help="Disable automatic weight alignment")
    parser.add_argument("--calibration", type=int, choices=[0, 1, 2, 3], default=2,
                        help="Distribution calibration level (0=none, 1=mean, 2=std, 3=full)")
    parser.add_argument("--summary-csv", help="Path to write summary CSV of layer stats")
    parser.add_argument("--export-metadata-json", help="Path to dump merged metadata JSON")
    parser.add_argument("--output", required=True, help="Path to write merged model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)

    n = len(args.models)
    if args.ratios:
        if len(args.ratios) != n:
            parser.error(f"--ratios length ({len(args.ratios)}) != number of models ({n})")
        ratios = args.ratios
    else:
        ratios = [1.0] * n

    plugins: List[Callable[[List[np.ndarray]], np.ndarray]] = []
    for p in args.plugins:
        plugins.append(load_plugin(p))

    tb_writer = None
    if args.tensorboard_logdir:
        if SummaryWriter is None:
            logger.warning("TensorBoard not available, skipping TB logging")
        else:
            tb_writer = SummaryWriter(log_dir=args.tensorboard_logdir)

    # Print some system information for diagnostics
    logger.info(f"Python version: {sys.version}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"ONNX version: {onnx.__version__}")
    logger.info(f"SciPy available: {SCIPY_AVAILABLE}")

    # Suppress NumPy warnings globally
    np.seterr(all='ignore')

    # load models + capture metadata
    loaded = [load_model(p) for p in args.models]
    dicts, protos, metas = zip(*loaded)

    # merge weights
# merge weights with enhanced alignment and calibration
    merged, summary = merge_state_dicts(
        dicts=list(dicts),
        method=args.method,
        ratios=ratios,
        trim_ratio=args.trim_ratio,
        geo_eps=args.geo_eps,
        prefixes=args.prefixes,
        plugins=plugins,
        plot_dir=args.plot_dir,
        tb_writer=tb_writer,
        skip_mismatch=args.skip_mismatch,
        force_merge=args.force_merge,
        align_weights=not args.no_align,
        calibration_level=args.calibration,
    )

    # save merged model - model structure and metadata will be preserved exactly
    save_model(merged, protos[0], args.output)

    # Export merge info externally (never modify original metadata)
    if args.export_metadata_json:
        export_merge_metadata(list(metas), args.method, ratios, args.export_metadata_json)
    else:
        # Always export merge info for reference
        export_merge_metadata(list(metas), args.method, ratios, args.output)

    # write enhanced summary CSV with distribution metrics
    if args.summary_csv:
        os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
        with open(args.summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "layer",
                "mean_before", "std_before", "min_before", "max_before", "unique_before",
                "mean_after", "std_after", "min_after", "max_after", "unique_after",
                "wasserstein_dist", "std_ratio", "unique_ratio", "kl_divergence",
                "merge_method"
            ])
            for name, before, after, dist_metrics, method_used in summary:
                writer.writerow([
                    name,
                    before["mean"], before["std"], before["min"], before["max"], before.get("unique_values", "N/A"),
                    after["mean"], after["std"], after["min"], after["max"], after.get("unique_values", "N/A"),
                    dist_metrics.get("wasserstein_dist", "N/A"), 
                    dist_metrics.get("std_ratio", "N/A"),
                    dist_metrics.get("unique_ratio", "N/A"),
                    dist_metrics.get("kl_divergence", "N/A"),
                    method_used
                ])
        logger.info(f"Summary CSV written to {args.summary_csv}")

    # Analyze distribution preservation and find the worst layers
    logger.info("Analyzing distribution preservation quality...")
    
    # Sort layers by distribution shift severity
    distribution_issues = []
    for name, before, after, dist_metrics, method_used in summary:
        std_ratio = dist_metrics.get("std_ratio", 1.0)
        unique_ratio = dist_metrics.get("unique_ratio", 1.0)
        w_dist = dist_metrics.get("wasserstein_dist", 0.0)
        
        # Compute a severity score (lower is worse)
        if 'bias' in name.lower() or before.get("unique_values", 1000) < 100:
            # For bias vectors and small tensors, focus on unique value preservation
            severity = unique_ratio * 0.6 + std_ratio * 0.4
        else:
            # For larger tensors, focus on distribution shape
            severity = std_ratio * 0.6 + unique_ratio * 0.4
        
        # Adjust severity by Wasserstein distance normalized by std
        norm_w_dist = w_dist / (before["std"] + 1e-8)
        severity -= min(0.2, norm_w_dist * 0.1)  # Penalize high Wasserstein distance
        
        if severity < 0.85 or norm_w_dist > 0.5:
            distribution_issues.append((name, severity, std_ratio, unique_ratio, norm_w_dist, method_used))
    
    # Sort by severity (worst first)
    distribution_issues.sort(key=lambda x: x[1])
    
    if distribution_issues:
        logger.warning(f"Found {len(distribution_issues)} layers with potential distribution preservation issues.")
        logger.warning("Top 10 most problematic layers:")
        for i, (name, severity, std_ratio, unique_ratio, norm_w_dist, method_used) in enumerate(distribution_issues[:10]):
            logger.warning(
                f"  {i+1}. {name}: severity={severity:.3f}, std_ratio={std_ratio:.3f}, "
                f"unique_ratio={unique_ratio:.3f}, norm_w_dist={norm_w_dist:.3f}, method={method_used}"
            )
            
            # Recommend specific improvements for each problematic layer
            if std_ratio < 0.5 and 'calibrated' not in method_used:
                logger.warning(f"     → Consider using stronger calibration for this layer")
            elif unique_ratio < 0.5 and 'discrete' not in method_used:
                logger.warning(f"     → Consider using discrete_preserving or kmeans_bin merge for this layer")
            elif norm_w_dist > 0.5 and 'bias' in name.lower():
                logger.warning(f"     → This bias layer may have permutation issues; try adaptive merge")
        
        logger.info("Check distribution plots for more details.")
    else:
        logger.info("No significant distribution preservation issues detected.")
    
    # Final report
    outsize = os.path.getsize(args.output) / 1024.0
    logger.info(f"Merge complete. Final model size: {outsize:.2f} KB")


if __name__ == "__main__":
    main()