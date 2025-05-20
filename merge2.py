#!/usr/bin/env python3
"""
model_merger.py

Production-ready script to merge two or more driving/vision models (PyTorch or ONNX)
using a variety of aggregation strategies, with prefix-based layer selection,
plugin hooks, distribution visualizations, automated analysis, and verbose logging.

Usage example:
    python model_merger.py m1.pth m2.onnx --method average \
        --ratios 0.6 0.4 --prefixes backbone. classifier. \
        --plot-dir ./plots --tensorboard-logdir ./tb_logs \
        --plugins my_plugin.py another_plugin.py \
        --skip-mismatch --summary-csv summary.csv \
        --export-metadata-json meta.json \
        --output merged.onnx
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
from typing import List, Dict, Any, Callable, Optional, Tuple

import numpy as np
import torch
import onnx
from onnx import numpy_helper, helper
import matplotlib.pyplot as plt

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
# Merge methods with numeric stability improvements
# ------------------------------------------------------------------------------


def average_weights(arrs: List[np.ndarray], ratios: List[float]) -> np.ndarray:
    """
    Merge weights by computing the weighted average with improved numeric stability.
    Uses incremental approach and higher precision to avoid overflow.
    """
    logger.debug("Computing weighted average with numeric stability")
    
    # Normalize ratios to sum to 1
    ratios_np = np.array(ratios, dtype=np.float64) / np.sum(ratios, dtype=np.float64)
    
    # Initialize with zeros in same shape as inputs but higher precision
    result_shape = arrs[0].shape
    result_dtype = np.float64  # Use high precision for calculation
    result = np.zeros(result_shape, dtype=result_dtype)
    
    # Accumulate incrementally to avoid overflow
    for i, arr in enumerate(arrs):
        # Convert to higher precision, scale by ratio, and add to result
        contribution = arr.astype(result_dtype) * ratios_np[i]
        result += contribution
    
    # Convert back to original dtype for consistency
    return result.astype(arrs[0].dtype)


def trimmed_mean_weights(arrs: List[np.ndarray], trim_ratio: float, ratios: List[float]) -> np.ndarray:
    """
    Merge weights using trimmed mean with improved numeric stability.
    Discards highest and lowest values before computing mean.
    """
    logger.debug(f"Computing trimmed mean (trim_ratio={trim_ratio}) with numeric stability")
    
    # Ensure we're working with high precision during calculation
    X = np.stack([arr.astype(np.float64) for arr in arrs], axis=0)
    n = X.shape[0]
    k = int(np.floor(trim_ratio * n))
    
    if 2 * k >= n:
        raise ValueError("trim_ratio too large for number of models")
    
    # Sort along first axis (across models)
    X_sorted = np.sort(X, axis=0)
    trimmed = X_sorted[k: n - k]
    
    # Compute mean and convert back to original dtype
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        result = trimmed.mean(axis=0)
    
    return result.astype(arrs[0].dtype)


def geometric_median_weights(arrs: List[np.ndarray], eps: float = 1e-5, max_iters: int = 1000) -> np.ndarray:
    """
    Compute the geometric median of weights using Weiszfeld's algorithm with numeric stability.
    Uses higher precision and careful normalization to avoid overflow.
    """
    logger.debug(f"Computing geometric median (eps={eps}, max_iters={max_iters}) with numeric stability")
    
    # Convert to higher precision for calculation
    X = np.stack([arr.astype(np.float64) for arr in arrs], axis=0)
    
    # Initialize with mean
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        median = X.mean(axis=0)
    
    for i in range(max_iters):
        diffs = X - median
        
        # Reshape for distance calculation
        flat = diffs.reshape((X.shape[0], -1))
        
        # Compute distances with stability
        distances = np.sqrt(np.sum(flat * flat, axis=1) + eps)
        
        if np.all(distances < eps):
            break
            
        # Compute weights with numeric safeguards
        weights = 1.0 / np.maximum(distances, eps)
        w_sum = np.sum(weights)
        
        if w_sum < eps:
            break
            
        # Reshape weights for broadcasting
        w_shaped = weights.reshape((-1,) + (1,) * (X.ndim - 1))
        
        # Compute new median with careful normalization
        new_median = np.sum(w_shaped * X, axis=0) / w_sum
        
        # Check convergence
        shift = np.sqrt(np.sum((new_median - median).flatten() ** 2))
        median = new_median
        
        if shift < eps:
            break
    
    return median.astype(arrs[0].dtype)


def mode_connectivity_weights(arrs: List[np.ndarray], ratios: List[float]) -> np.ndarray:
    """
    Linear interpolation along the path connecting models with numeric stability.
    Uses higher precision during calculation to avoid overflow.
    """
    logger.debug("Computing mode connectivity (linear interpolation) with numeric stability")
    
    # Normalize ratios
    ratios_normalized = np.array(ratios, dtype=np.float64)
    ratios_normalized = ratios_normalized / np.sum(ratios_normalized)
    
    # Convert to higher precision
    base = arrs[0].astype(np.float64)
    
    for i, arr in enumerate(arrs[1:], start=1):
        # Compute interpolation carefully
        alpha = ratios_normalized[i] / (ratios_normalized[0] + np.sum(ratios_normalized[1:i+1]))
        arr_higher = arr.astype(np.float64)
        base = (1.0 - alpha) * base + alpha * arr_higher
    
    return base.astype(arrs[0].dtype)


MERGE_METHODS: Dict[str, Callable[..., np.ndarray]] = {
    "average": average_weights,
    "trimmed_mean": trimmed_mean_weights,
    "geometric_median": geometric_median_weights,
    "mode_connectivity": mode_connectivity_weights,
}

# ------------------------------------------------------------------------------
# ONNX helpers (with metadata extraction)
# ------------------------------------------------------------------------------


def load_onnx_weights(path: str) -> Tuple[Dict[str, np.ndarray], onnx.ModelProto, Dict[str, str]]:
    """Load weights, model, and metadata from an ONNX file."""
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
    Handles numeric precision carefully to maintain compatibility.
    """
    logger.info(f"Saving merged ONNX model to {output_path}")
    
    # Update weights in the model, preserving data types
    for init in model.graph.initializer:
        if init.name in merged:
            # Get original tensor's data type
            original_tensor = init
            original_data_type = original_tensor.data_type
            
            # Create new tensor with same name and data type as original
            new_array = merged[init.name]
            
            try:
                # Try to create a tensor with original's name and the merged data
                new_tensor = numpy_helper.from_array(new_array, init.name)
                
                # If the data type doesn't match, fix it explicitly
                if new_tensor.data_type != original_data_type:
                    logger.debug(f"Converting tensor {init.name} to match original data type")
                    
                    # Convert to the appropriate numpy type
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
                    
                    # Create a new tensor with the correct type
                    new_tensor = numpy_helper.from_array(new_array, init.name)
                
                # Update the tensor in the model
                init.CopyFrom(new_tensor)
                
            except Exception as e:
                logger.warning(f"Error updating tensor {init.name}: {e}")
                logger.warning(f"Using original tensor to maintain compatibility")
    
    # Save the model - no modifications to structure or metadata
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
        input_metadata_list: List of metadata dicts loaded from each input model.
        method: Name of the merge method used.
        ratios: Merge ratios for each input model.
        output_path: Base path for the output file.
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
    """Load a plugin module that provides a custom merge function."""
    spec = importlib.util.spec_from_file_location("plugin_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "merge"):
        raise ValueError(f"Plugin {path} must define a 'merge(arrs: List[np.ndarray]) -> np.ndarray' function.")
    logger.info(f"Loaded plugin: {path}")
    return mod.merge  # type: ignore

# ------------------------------------------------------------------------------
# Utility: distribution visualization & analysis
# ------------------------------------------------------------------------------


def visualize_distribution(before: List[np.ndarray], after: np.ndarray, name: str, outdir: str):
    """Create a histogram visualization of weight distributions before and after merging."""
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    
    # Handle potentially large arrays by sampling
    max_samples = 100000  # Limit samples for efficiency
    
    all_before = np.concatenate([arr.flatten() for arr in before])
    if len(all_before) > max_samples:
        indices = np.random.choice(len(all_before), max_samples, replace=False)
        all_before = all_before[indices]
    
    after_flat = after.flatten()
    if len(after_flat) > max_samples:
        indices = np.random.choice(len(after_flat), max_samples, replace=False)
        after_flat = after_flat[indices]
    
    plt.hist(all_before, bins=100, alpha=0.5, label="before")
    plt.hist(after_flat, bins=100, alpha=0.5, label="after")
    plt.title(f"Weight distribution for {name}")
    plt.legend()
    
    plot_path = os.path.join(outdir, f"{name.replace('/', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.debug(f"Saved distribution plot for {name} at {plot_path}")


def compute_stats(arr: np.ndarray) -> Dict[str, float]:
    """
    Compute basic statistical metrics on an array with numeric stability.
    """
    # Catch warnings to avoid exposing NumPy warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        flat = arr.flatten()
        return {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
        }

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
) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, Dict[str, float], Dict[str, float], float]]]:
    """
    Merge state dictionaries (weight mappings) from multiple models with numeric stability.
    
    Returns:
        Tuple of (merged_weights, summary_statistics)
    """
    # Suppress NumPy warnings during merge operation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        start = time.time()
        merged: Dict[str, np.ndarray] = {}
        summary: List[Tuple[str, Dict[str, float], Dict[str, float], float]] = []

        reference_keys = list(dicts[0].keys())
        extra = set().union(*(d.keys() for d in dicts[1:])) - set(reference_keys)
        if extra:
            logger.warning(f"Ignoring {len(extra)} extra layers not in reference model: {sorted(extra)}")

        # Calculate total parameters for reporting
        total_params = 0
        
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
                    merged[name] = arrs[0]
                    total_params += np.prod(arrs[0].shape)
                    continue
                else:
                    raise ValueError(msg + "; use --skip-mismatch to bypass")

            if prefixes and not any(name.startswith(p) for p in prefixes):
                merged[name] = arrs[0]
                total_params += np.prod(arrs[0].shape)
                logger.debug(f"Skipping layer '{name}' (prefix filter)")
                continue

            shapes = [a.shape for a in arrs]
            if len(set(str(s) for s in shapes)) != 1:  # Convert shapes to strings for comparison
                msg = f"Layer '{name}' has mismatched shapes {shapes}"
                if skip_mismatch:
                    logger.warning(msg + "; copying reference weights")
                    merged[name] = arrs[0]
                    total_params += np.prod(arrs[0].shape)
                    continue
                else:
                    raise ValueError(msg + "; use --skip-mismatch to bypass")

            # Check for NaN or Inf values in inputs
            has_nan = any(np.isnan(arr).any() for arr in arrs)
            has_inf = any(np.isinf(arr).any() for arr in arrs)
            
            if has_nan or has_inf:
                logger.warning(f"Layer '{name}' contains NaN or Inf values in input; copying reference weights")
                merged[name] = arrs[0]
                total_params += np.prod(arrs[0].shape)
                continue

            # Try merge with plugin first
            out = None
            for plugin in plugins:
                try:
                    out = plugin(arrs)
                    logger.debug(f"Layer '{name}' merged via plugin")
                    break
                except Exception as e:
                    logger.debug(f"Plugin error on layer '{name}': {e}")

            # Fall back to built-in methods if plugin didn't work
            if out is None:
                if method not in MERGE_METHODS:
                    raise ValueError(f"Unknown merge method '{method}'")
                
                try:
                    if method == "trimmed_mean":
                        out = MERGE_METHODS[method](arrs, trim_ratio, ratios)
                    elif method in ("average", "mode_connectivity"):
                        out = MERGE_METHODS[method](arrs, ratios)
                    else:
                        out = MERGE_METHODS[method](arrs, eps=geo_eps)
                except Exception as e:
                    logger.warning(f"Error merging layer '{name}': {e}")
                    logger.warning(f"Using reference weights for layer '{name}'")
                    out = arrs[0]

            # Validate output for NaN or Inf
            if np.isnan(out).any() or np.isinf(out).any():
                logger.warning(f"Merge produced NaN or Inf values for layer '{name}'; using reference weights")
                out = arrs[0]

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
                delta_norm = float(np.linalg.norm((out - arrs[0]).flatten()))
                summary.append((name, stats_before, stats_after, delta_norm))
                logger.info(f"Layer '{name}': before={stats_before}, after={stats_after}, ||Δ||={delta_norm:.4f}")
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
                except Exception as e:
                    logger.warning(f"TensorBoard logging failed for layer '{name}': {e}")

            merged[name] = out

        elapsed = time.time() - start
        logger.info(f"Merged {len(summary)} layers in {elapsed:.2f}s (skipped {len(reference_keys) - len(summary)})")
        logger.info(f"Total parameters in merged model: {total_params:,}")
        return merged, summary

# ------------------------------------------------------------------------------
# I/O helpers (unified for PyTorch & ONNX, returning metadata)
# ------------------------------------------------------------------------------


def load_model(path: str) -> Tuple[Dict[str, np.ndarray], Any, Dict[str, str]]:
    """
    Load a model from file, supporting PyTorch (.pth, .pt) and ONNX (.onnx) formats.
    
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
    Preserves exact structure for compatibility.
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
    parser = argparse.ArgumentParser(description="Merge PyTorch or ONNX models.")
    parser.add_argument("models", nargs="+", help="Paths to model files (.pth, .pt, .onnx)")
    parser.add_argument("--method", required=True, choices=list(MERGE_METHODS) + ["plugin"],
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

    # Suppress NumPy warnings globally
    np.seterr(all='ignore')

    # load models + capture metadata
    loaded = [load_model(p) for p in args.models]
    dicts, protos, metas = zip(*loaded)

    # merge weights
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
    )

    # save merged model - model structure and metadata will be preserved exactly
    save_model(merged, protos[0], args.output)

    # Export merge info externally (never modify original metadata)
    if args.export_metadata_json:
        export_merge_metadata(list(metas), args.method, ratios, args.export_metadata_json)
    else:
        # Always export merge info for reference
        export_merge_metadata(list(metas), args.method, ratios, args.output)

    # write summary CSV
    if args.summary_csv:
        os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
        with open(args.summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "layer",
                "mean_before", "std_before", "min_before", "max_before",
                "mean_after", "std_after", "min_after", "max_after",
                "delta_norm"
            ])
            for name, before, after, delta in summary:
                writer.writerow([
                    name,
                    before["mean"], before["std"], before["min"], before["max"],
                    after["mean"], after["std"], after["min"], after["max"],
                    delta
                ])
        logger.info(f"Summary CSV written to {args.summary_csv}")

    # Final report
    outsize = os.path.getsize(args.output) / 1024.0
    logger.info(f"Merge complete. Final model size: {outsize:.2f} KB")


if __name__ == "__main__":
    main()