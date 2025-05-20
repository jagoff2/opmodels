
"""
Model Optimization via SVD-Based Transformation

This script optimizes neural network models by adjusting their weight matrices
using Singular Value Decomposition (SVD) to achieve ideal power-law distributions
according to WeightWatcher quality metrics. Supports models with float16 precision.

Usage:
    python optimize_model.py --input driving_policy.onnx --output optimized_policy.onnx
"""

import os
import sys
import argparse
import logging
import traceback
import numpy as np
import pandas as pd
from scipy import stats
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import json
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_optimization.log")
    ]
)
logger = logging.getLogger(__name__)

# Set numpy print options
np.set_printoptions(precision=4, suppress=True)

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain NumPy types
        
    Returns:
        Object with NumPy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

class ModelOptimizer:
    """Optimizes neural network models using SVD-based transformations."""
    
    # Constants for ideal alpha ranges
    ALPHA_MIN_IDEAL = 2.0
    ALPHA_MAX_IDEAL = 6.0
    ALPHA_TARGET_UNDERFIT = 4.5  # Target for underfit layers (alpha > ALPHA_MAX_IDEAL)
    ALPHA_TARGET_OVERFIT = 2.5   # Target for overfit layers (alpha < ALPHA_MIN_IDEAL)
    
    def __init__(self, 
                 input_model_path: str, 
                 output_model_path: str,
                 target_layers: Optional[List[str]] = None,
                 optimization_strength: float = 0.5,
                 visualization_dir: Optional[str] = "optimization_plots",
                 preserve_scale: bool = True,
                 detailed_logging: bool = True):
        """
        Initialize the model optimizer.
        
        Args:
            input_model_path: Path to input ONNX model
            output_model_path: Path to save optimized ONNX model
            target_layers: List of layer names to optimize (None for all eligible layers)
            optimization_strength: Controls magnitude of changes (0.0-1.0)
            visualization_dir: Directory to save plots (None to disable)
            preserve_scale: Whether to preserve spectral norm of modified matrices
            detailed_logging: Enable detailed logging of transformations
        """
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.target_layers = target_layers
        self.optimization_strength = max(0.0, min(1.0, optimization_strength))
        self.visualization_dir = visualization_dir
        self.preserve_scale = preserve_scale
        self.detailed_logging = detailed_logging
        
        # State variables
        self.model = None
        self.weights = {}
        self.weight_info = {}
        self.initializers_map = {}
        self.modified_weights = {}
        self.optimization_stats = defaultdict(dict)
        
        # Create visualization directory if needed
        if self.visualization_dir:
            os.makedirs(self.visualization_dir, exist_ok=True)
    
    def load_model(self) -> None:
        """Load the ONNX model and extract its weights."""
        try:
            logger.info(f"Loading model from {self.input_model_path}")
            self.model = onnx.load(self.input_model_path)
            self._extract_weights()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _extract_weights(self) -> None:
        """Extract weight matrices and their metadata from the model."""
        initializer_count = 0
        weight_count = 0
        
        for initializer in self.model.graph.initializer:
            initializer_count += 1
            name = initializer.name
            self.initializers_map[name] = initializer
            
            # Extract array data
            try:
                tensor = numpy_helper.to_array(initializer)
                
                # Only consider matrices (2D or 4D tensors)
                if len(tensor.shape) in [2, 4]:
                    # For 4D convolution tensors, reshape to 2D
                    if len(tensor.shape) == 4:
                        # Conv filters: [out_channels, in_channels, height, width]
                        out_channels = tensor.shape[0]
                        tensor_2d = tensor.reshape(out_channels, -1)
                    else:
                        tensor_2d = tensor
                    
                    # Only consider matrices with sufficient dimensions for SVD
                    if min(tensor_2d.shape) > 3:  # Reduced threshold to catch more layers
                        weight_count += 1
                        self.weights[name] = tensor
                        self.weight_info[name] = {
                            'original_shape': tensor.shape,
                            'is_conv': len(tensor.shape) == 4,
                            'dtype': tensor.dtype,
                        }
            except Exception as e:
                logger.warning(f"Failed to extract weights for {name}: {str(e)}")
        
        logger.info(f"Extracted {weight_count} weight matrices from {initializer_count} initializers")
    
    def analyze_weights(self) -> pd.DataFrame:
        """
        Analyze weight matrices to determine their spectral properties.
        
        Returns:
            DataFrame containing analysis of each weight matrix
        """
        results = []
        
        for name, weight in self.weights.items():
            try:
                # Handle float16 by converting to float32 for analysis
                original_dtype = weight.dtype
                if original_dtype == np.float16:
                    # Convert to float32 for SVD
                    weight_f32 = weight.astype(np.float32)
                    logger.info(f"Converting {name} from float16 to float32 for analysis")
                else:
                    weight_f32 = weight
                
                # Ensure the weight is a 2D matrix for analysis
                if len(weight_f32.shape) == 4:  # Conv weights
                    out_channels = weight_f32.shape[0]
                    matrix = weight_f32.reshape(out_channels, -1)
                else:
                    matrix = weight_f32
                
                # Minimum dimension affects how many singular values we get
                min_dim = min(matrix.shape)
                
                # Compute SVD on float32 matrix
                try:
                    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
                    
                    # Compute robust power law fit for alpha metrics
                    alpha, alpha_weighted, xmin, spectral_norm = self._fit_powerlaw_robust(S)
                    
                    # Determine if layer needs optimization
                    status = self._determine_layer_status(alpha)
                    
                    # Compute additional metrics
                    stable_rank = np.sum(S**2) / (S[0]**2)
                    
                    results.append({
                        'layer_name': name,
                        'shape': str(weight.shape),
                        'min_dim': min_dim,
                        'alpha': float(alpha),  # Convert numpy.float32 to Python float
                        'alpha_weighted': float(alpha_weighted),
                        'spectral_norm': float(spectral_norm),
                        'stable_rank': float(stable_rank),
                        'status': status,
                        'xmin': float(xmin),
                        'xmax': float(S[0]),
                        'dtype': str(original_dtype)
                    })
                    
                    # Store SVD components for later use
                    self.weight_info[name].update({
                        'U': U,
                        'S': S,
                        'Vh': Vh,
                        'alpha': float(alpha),
                        'alpha_weighted': float(alpha_weighted),
                        'spectral_norm': float(spectral_norm),
                        'status': status
                    })
                    
                except np.linalg.LinAlgError as e:
                    logger.warning(f"SVD failed for layer {name}: {str(e)}")
                    continue
                
            except Exception as e:
                logger.error(f"Error analyzing layer {name}: {str(e)}")
                continue
        
        # Create and sort DataFrame
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by=['status', 'alpha'])
            
            # Log summary of layer statuses
            underfit_count = sum(df['status'] == 'underfit')
            overfit_count = sum(df['status'] == 'overfit')
            optimal_count = sum(df['status'] == 'optimal')
            logger.info(f"Layer classification summary: {underfit_count} underfit, {overfit_count} overfit, {optimal_count} optimal")
            
            # Print details of underfit layers
            if underfit_count > 0:
                underfit_layers = df[df['status'] == 'underfit']
                logger.info("Underfit layers detected:")
                for _, row in underfit_layers.iterrows():
                    logger.info(f"  - {row['layer_name']}: alpha={row['alpha']}")
            
            # Print details of overfit layers
            if overfit_count > 0:
                overfit_layers = df[df['status'] == 'overfit']
                logger.info("Overfit layers detected:")
                for _, row in overfit_layers.iterrows():
                    logger.info(f"  - {row['layer_name']}: alpha={row['alpha']}")
        
        return df
    
    def _fit_powerlaw_robust(self, S: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Fit a power law to the singular values using multiple methods 
        for robust alpha determination.
        
        Args:
            S: Array of singular values (in descending order)
            
        Returns:
            Tuple of (alpha, alpha_weighted, xmin, spectral_norm)
        """
        # Remove zeros and very small values to avoid log issues
        S_nonzero = S[S > S[0] * 1e-10]
        if len(S_nonzero) < 4:
            # Not enough points for reliable fitting
            return 2.5, 2.5, S_nonzero[-1] if len(S_nonzero) else 0, S[0]
        
        # Get spectral norm (largest singular value)
        spectral_norm = S[0]
        
        # Method 1: Full distribution fit
        try:
            # Log-log linear regression for power law fit on full distribution
            log_S = np.log(S_nonzero)
            log_rank = np.log(np.arange(1, len(S_nonzero) + 1))
            
            slope_full, _, r_value_full, _, _ = stats.linregress(log_rank, log_S)
            alpha_full = -slope_full + 1  # Convert to power law exponent
            r_squared_full = r_value_full**2
        except Exception:
            alpha_full = 2.5
            r_squared_full = 0
        
        # Method 2: Tail-only fit (using last 30% of values)
        try:
            n = len(S_nonzero)
            tail_start = max(int(n * 0.7), 3)  # At least 3 points
            S_tail = S_nonzero[tail_start:]
            
            if len(S_tail) >= 3:
                log_S_tail = np.log(S_tail)
                log_rank_tail = np.log(np.arange(tail_start + 1, n + 1))
                
                slope_tail, _, r_value_tail, _, _ = stats.linregress(log_rank_tail, log_S_tail)
                alpha_tail = -slope_tail + 1
                r_squared_tail = r_value_tail**2
            else:
                alpha_tail = alpha_full
                r_squared_tail = 0
        except Exception:
            alpha_tail = alpha_full
            r_squared_tail = 0
        
        # Method 3: Direct estimate from ratio of top to median singular values
        try:
            median_idx = len(S_nonzero) // 2
            ratio = S_nonzero[0] / S_nonzero[median_idx]
            # Map ratio to approximate alpha using power law
            alpha_ratio = 1.0 + np.log(ratio) / np.log(median_idx + 1)
        except Exception:
            alpha_ratio = 2.5
        
        # Method 4: Slope of first few points (can detect high alpha values better)
        try:
            # Use first 25% of points (but at least 3, at most 10)
            num_points = min(max(3, int(len(S_nonzero) * 0.25)), 10)
            if num_points < len(S_nonzero):
                log_S_head = np.log(S_nonzero[:num_points])
                log_rank_head = np.log(np.arange(1, num_points + 1))
                
                slope_head, _, r_value_head, _, _ = stats.linregress(log_rank_head, log_S_head)
                alpha_head = -slope_head + 1
                r_squared_head = r_value_head**2
            else:
                alpha_head = alpha_full
                r_squared_head = 0
        except Exception:
            alpha_head = alpha_full
            r_squared_head = 0
        
        # Choose best alpha based on R² values and range checks
        alphas = [(alpha_full, r_squared_full), 
                  (alpha_tail, r_squared_tail),
                  (alpha_head, r_squared_head)]
        
        # Filter valid and in-range alphas
        valid_alphas = [(a, r) for a, r in alphas if 0.5 < a < 15 and r > 0.5]
        
        if valid_alphas:
            # Choose alpha with best R² value
            alpha = max(valid_alphas, key=lambda x: x[1])[0]
        else:
            # Fall back to ratio method if no good fits
            alpha = max(0.5, min(15.0, alpha_ratio))
        
        # Compute weighted alpha using the most reliable method
        try:
            # Weighted linear regression emphasizing larger singular values
            weights = S_nonzero / np.sum(S_nonzero)
            log_S_all = np.log(S_nonzero)
            log_rank_all = np.log(np.arange(1, len(S_nonzero) + 1))
            
            weighted_slope = np.sum(weights * log_S_all * log_rank_all) / np.sum(weights * log_rank_all**2)
            alpha_weighted = -weighted_slope + 1
            
            # Clamp to reasonable range
            alpha_weighted = max(0.5, min(15.0, alpha_weighted))
        except Exception:
            alpha_weighted = alpha
        
        # Determine xmin (cutoff for power law region)
        xmin = S_nonzero[-1]  # Default to smallest value
        
        # Log diagnostic information for alpha determination
        logger.debug(f"Alpha values - full: {alpha_full:.4f}, tail: {alpha_tail:.4f}, " 
                     f"head: {alpha_head:.4f}, ratio: {alpha_ratio:.4f}, final: {alpha:.4f}")
        
        return alpha, alpha_weighted, xmin, spectral_norm
    
    def _determine_layer_status(self, alpha: float) -> str:
        """
        Determine the status of a layer based on its alpha value.
        
        Args:
            alpha: Power law exponent of the layer
            
        Returns:
            Status string: 'underfit', 'overfit', or 'optimal'
        """
        if alpha < self.ALPHA_MIN_IDEAL:
            return 'overfit'
        elif alpha > self.ALPHA_MAX_IDEAL:
            return 'underfit'
        else:
            return 'optimal'
    
    def _generate_target_distribution(self, S: np.ndarray, target_alpha: float) -> np.ndarray:
        """
        Generate a new distribution of singular values with the target power law exponent.
        
        Args:
            S: Original singular values
            target_alpha: Target power law exponent
            
        Returns:
            New singular values following target distribution
        """
        n = len(S)
        if n <= 2:
            return S.copy()
        
        # Use ranks as the x-values for the power law
        ranks = np.arange(1, n + 1)
        
        # Original max singular value
        s_max = S[0]
        
        # Generate new values following power law: s ∝ r^(-β)
        # Where β = α - 1
        beta = target_alpha - 1
        new_S = s_max * ranks**(-beta)
        
        return new_S
    
    def optimize_weights(self, analysis_df: pd.DataFrame) -> None:
        """
        Optimize weight matrices by adjusting their singular values to follow ideal
        power-law distributions.
        
        Args:
            analysis_df: DataFrame with analysis results
        """
        # Filter layers for optimization
        layers_to_optimize = []
        
        if self.target_layers is not None:
            # Use specified target layers
            for name in self.target_layers:
                if name in self.weights:
                    layer_status = self.weight_info[name].get('status')
                    if layer_status in ['overfit', 'underfit']:
                        layers_to_optimize.append(name)
                    else:
                        logger.info(f"Layer {name} is already optimal, skipping")
                else:
                    logger.warning(f"Specified target layer {name} not found in model")
        else:
            # Auto-detect layers to optimize
            for name, info in self.weight_info.items():
                if 'status' in info and info['status'] in ['overfit', 'underfit']:
                    layers_to_optimize.append(name)
        
        # Log optimization plan
        logger.info(f"Optimizing {len(layers_to_optimize)} layers:")
        for name in layers_to_optimize:
            status = self.weight_info[name]['status']
            alpha = self.weight_info[name]['alpha']
            logger.info(f"  - {name}: {status} (alpha={alpha:.3f})")
        
        # Apply transformations
        for name in layers_to_optimize:
            try:
                self._optimize_layer(name)
            except Exception as e:
                logger.error(f"Failed to optimize layer {name}: {str(e)}")
                logger.error(traceback.format_exc())
    
    def _optimize_layer(self, name: str) -> None:
        """
        Optimize a single layer's weight matrix.
        
        Args:
            name: Name of the layer to optimize
        """
        info = self.weight_info[name]
        original_shape = info['original_shape']
        is_conv = info['is_conv']
        status = info['status']
        original_alpha = info['alpha']
        U = info['U']
        S = info['S']
        Vh = info['Vh']
        original_dtype = self.weights[name].dtype
        
        # Determine target alpha
        if status == 'underfit':
            target_alpha = self.ALPHA_TARGET_UNDERFIT
            logger.info(f"Targeting underfit layer {name}: alpha={original_alpha:.4f} -> target={target_alpha:.4f}")
        elif status == 'overfit':
            target_alpha = self.ALPHA_TARGET_OVERFIT
            logger.info(f"Targeting overfit layer {name}: alpha={original_alpha:.4f} -> target={target_alpha:.4f}")
        else:
            # Should not happen, but just in case
            target_alpha = 4.0
            logger.warning(f"Unexpected optimization of optimal layer {name}")
        
        # Interpolate between original and target alpha based on optimization_strength
        effective_alpha = original_alpha * (1 - self.optimization_strength) + target_alpha * self.optimization_strength
        
        # Generate new singular values
        new_S = self._generate_target_distribution(S, effective_alpha)
        
        # Preserve spectral norm (scale) if required
        if self.preserve_scale:
            new_S = new_S * (S[0] / new_S[0])
        
        # Reconstruct matrix with new singular values
        if is_conv:
            # For convolutional layers
            out_channels = original_shape[0]
            rest_shape = original_shape[1:]
            rest_size = np.prod(rest_shape)
            
            # Reconstruct 2D matrix
            matrix_2d = U @ np.diag(new_S) @ Vh
            
            # Reshape back to 4D
            new_weight = matrix_2d.reshape(original_shape)
        else:
            # For linear layers
            new_weight = U @ np.diag(new_S) @ Vh
        
        # Convert back to original dtype if needed
        if original_dtype == np.float16:
            new_weight = new_weight.astype(np.float16)
            logger.info(f"Converting optimized weights for {name} back to float16")
        
        # Store modified weight
        self.modified_weights[name] = new_weight
        
        # Calculate new alpha for reporting (using float32 for calculation)
        new_weight_f32 = new_weight.astype(np.float32)
        _, new_S_actual, _ = np.linalg.svd(new_weight_f32.reshape(U.shape[0], -1), full_matrices=False)
        new_alpha, new_alpha_weighted, new_xmin, new_spectral_norm = self._fit_powerlaw_robust(new_S_actual)
        
        # Store optimization stats - convert all numpy types to Python native types
        self.optimization_stats[name] = {
            'original_alpha': float(original_alpha),
            'target_alpha': float(target_alpha),
            'achieved_alpha': float(new_alpha),
            'original_spectral_norm': float(S[0]),
            'new_spectral_norm': float(new_S_actual[0]),
            'status': status,
            'shape': str(original_shape),
            'dtype': str(original_dtype)
        }
        
        # Detailed logging
        if self.detailed_logging:
            logger.info(f"Optimized {name}:")
            logger.info(f"  Status: {status}")
            logger.info(f"  Shape: {original_shape}")
            logger.info(f"  Dtype: {original_dtype}")
            logger.info(f"  Alpha: {original_alpha:.4f} -> {new_alpha:.4f}")
            logger.info(f"  Alpha Weighted: {info['alpha_weighted']:.4f} -> {new_alpha_weighted:.4f}")
            logger.info(f"  Spectral Norm: {S[0]:.4f} -> {new_S_actual[0]:.4f}")
        
        # Visualization
        if self.visualization_dir:
            self._visualize_transformation(name, S, new_S_actual, original_alpha, new_alpha)
    
    def _visualize_transformation(self, name: str, 
                                 original_S: np.ndarray, 
                                 new_S: np.ndarray, 
                                 original_alpha: float, 
                                 new_alpha: float) -> None:
        """
        Create visualization for the singular value transformation.
        
        Args:
            name: Layer name
            original_S: Original singular values
            new_S: New singular values
            original_alpha: Original alpha
            new_alpha: New alpha
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Clean up name for filename
        safe_name = name.replace('/', '_').replace(':', '_')
        
        # Log-log plot of singular values
        ranks = np.arange(1, len(original_S) + 1)
        
        # Original distribution
        ax1.loglog(ranks, original_S, 'b-', label=f'Original (α={original_alpha:.2f})')
        
        # New distribution
        ax1.loglog(ranks, new_S, 'r-', label=f'Optimized (α={new_alpha:.2f})')
        
        # Ideal reference line
        if len(original_S) > 10:
            x_sample = np.logspace(0, np.log10(len(original_S)), 100)
            ideal_alpha = 4.0
            y_sample = original_S[0] * (x_sample)**(-ideal_alpha + 1)
            ax1.loglog(x_sample, y_sample, 'g--', label=f'Ideal (α={ideal_alpha:.1f})')
        
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('Singular Value')
        ax1.set_title('Singular Value Distribution (Log-Log)')
        ax1.legend()
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        
        # Linear plot showing largest singular values
        show_n = min(20, len(original_S))
        x = np.arange(show_n)
        
        ax2.plot(x, original_S[:show_n], 'bo-', label='Original')
        ax2.plot(x, new_S[:show_n], 'ro-', label='Optimized')
        
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Singular Value')
        ax2.set_title(f'Top {show_n} Singular Values')
        ax2.legend()
        ax2.grid(True, alpha=0.2)
        
        plt.suptitle(f'Layer: {name}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.visualization_dir, f"transform_{safe_name}.png"))
        plt.close(fig)
    
    def update_model(self) -> None:
        """
        Update the model with optimized weights.
        """
        # Create a copy of the model
        optimized_model = onnx.ModelProto()
        optimized_model.CopyFrom(self.model)
        
        # Update initializers with modified weights
        for i, initializer in enumerate(optimized_model.graph.initializer):
            name = initializer.name
            if name in self.modified_weights:
                # Replace with modified weights
                new_weight = self.modified_weights[name]
                original_dtype = self.weight_info[name]['dtype']
                
                # Ensure dtype matches original
                if new_weight.dtype != original_dtype:
                    new_weight = new_weight.astype(original_dtype)
                
                # Create new tensor with the same name
                new_tensor = numpy_helper.from_array(new_weight, name=name)
                optimized_model.graph.initializer[i].CopyFrom(new_tensor)
        
        # Save model
        try:
            onnx.save(optimized_model, self.output_model_path)
            logger.info(f"Saved optimized model to {self.output_model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def save_optimization_report(self, report_path: Optional[str] = None) -> None:
        """
        Save a detailed report of the optimization process.
        
        Args:
            report_path: Path to save the report (default: next to output model)
        """
        if not report_path:
            # Create default path based on output model path
            output_dir = os.path.dirname(self.output_model_path) or '.'
            output_name = os.path.splitext(os.path.basename(self.output_model_path))[0]
            report_path = os.path.join(output_dir, f"{output_name}_optimization_report.json")
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_model': self.input_model_path,
            'output_model': self.output_model_path,
            'optimization_strength': float(self.optimization_strength),
            'layers_optimized': len(self.modified_weights),
            'total_layers_analyzed': len(self.weight_info),
            'layer_details': {}
        }
        
        # Add details for each optimized layer
        for name, stats in self.optimization_stats.items():
            report['layer_details'][name] = stats
        
        # Convert all NumPy types to Python native types for JSON serialization
        report = convert_numpy_types(report)
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved optimization report to {report_path}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full optimization pipeline.
        
        Returns:
            Dictionary with optimization summary
        """
        start_time = time.time()
        
        try:
            # Step 1: Load model
            self.load_model()
            
            # Step 2: Analyze weights
            analysis_df = self.analyze_weights()
            
            # Step 3: Optimize weights
            self.optimize_weights(analysis_df)
            
            # Step 4: Update model
            self.update_model()
            
            # Step 5: Save report
            self.save_optimization_report()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
            
            # Return summary - convert NumPy types to native Python types
            return convert_numpy_types({
                'status': 'success',
                'layers_optimized': len(self.modified_weights),
                'total_layers': len(self.weight_info),
                'elapsed_time': elapsed_time,
                'optimized_model_path': self.output_model_path
            })
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Optimization failed after {elapsed_time:.2f} seconds: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'status': 'error',
                'error': str(e),
                'elapsed_time': float(elapsed_time)
            }


def main():
    """Parse command line arguments and run the optimizer."""
    parser = argparse.ArgumentParser(description='Optimize neural network models using SVD transformations')
    
    parser.add_argument('--input', '-i', required=True, 
                        help='Input ONNX model path')
    parser.add_argument('--output', '-o', required=True, 
                        help='Output ONNX model path')
    parser.add_argument('--layers', '-l', nargs='+', default=None,
                        help='Specific layers to optimize (default: auto-detect)')
    parser.add_argument('--strength', '-s', type=float, default=0.5,
                        help='Optimization strength (0.0-1.0, default: 0.5)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable visualization plots')
    parser.add_argument('--no-preserve-scale', action='store_true',
                        help='Do not preserve spectral norm of weight matrices')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce logging verbosity')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"Input model not found: {args.input}")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize and run optimizer
    optimizer = ModelOptimizer(
        input_model_path=args.input,
        output_model_path=args.output,
        target_layers=args.layers,
        optimization_strength=args.strength,
        visualization_dir=None if args.no_plots else "optimization_plots",
        preserve_scale=not args.no_preserve_scale,
        detailed_logging=not args.quiet
    )
    
    result = optimizer.run()
    
    if result['status'] == 'success':
        print(f"\nOptimization completed successfully!")
        print(f"Optimized {result['layers_optimized']} out of {result['total_layers']} layers")
        print(f"Model saved to: {result['optimized_model_path']}")
        return 0
    else:
        print(f"\nOptimization failed: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())