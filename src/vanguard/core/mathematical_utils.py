"""
VANGUARD: Mathematical Utilities Module

This module provides numerically stable implementations of mathematical functions
commonly used in Bayesian deep learning and medical image processing.

Key Features:
- Numerically stable implementations of common functions
- GPU-optimized tensor operations
- Gradient-safe computations for automatic differentiation
- Medical imaging specific utilities
"""

import math
from typing import Tuple, Optional, Union, List, Dict
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "stable_softplus",
    "stable_log_sum_exp", 
    "stable_kl_divergence_gaussian",
    "safe_log",
    "safe_sqrt",
    "robust_normalize",
    "compute_grad_norm",
    "spectral_norm_conv",
    "adaptive_gradient_clipping",
    "numerical_gradient_check",
    "safe_inverse",
    "stable_cholesky",
    "medical_image_stats",
    "intensity_normalization",
]


def stable_softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """
    Numerically stable softplus implementation.
    
    For large x, softplus(x) ≈ x to avoid overflow.
    For small x, uses standard softplus formula.
    
    Mathematical basis:
    softplus(x) = (1/β) * log(1 + exp(βx))
    For βx > threshold: softplus(x) ≈ x + (1/β) * log(β)
    
    Args:
        x: Input tensor
        beta: Scaling parameter (higher beta = steeper transition)
        threshold: Threshold for linear approximation
        
    Returns:
        Numerically stable softplus output
    """
    # Scale input
    scaled_x = beta * x
    
    # Use linear approximation for large values
    linear_approx = x + (1.0 / beta) * math.log(beta)
    
    # Standard softplus for moderate values
    standard_softplus = (1.0 / beta) * F.softplus(scaled_x)
    
    # Choose based on threshold
    return torch.where(scaled_x > threshold, linear_approx, standard_softplus)


def stable_log_sum_exp(x: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """
    Numerically stable log-sum-exp computation.
    
    Computes log(Σᵢ exp(xᵢ)) without numerical overflow/underflow.
    Uses the identity: LSE(x) = max(x) + log(Σᵢ exp(xᵢ - max(x)))
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute LSE
        keepdim: Whether to keep reduced dimension
        
    Returns:
        Log-sum-exp result
    """
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    
    # Handle case where x_max is -inf (all elements are -inf)
    x_max = torch.clamp(x_max, min=-1e10)
    
    lse = x_max + torch.log(torch.sum(torch.exp(x - x_max), dim=dim, keepdim=True))
    
    if not keepdim:
        lse = lse.squeeze(dim)
    
    return lse


def stable_kl_divergence_gaussian(
    mean1: Tensor, 
    std1: Tensor, 
    mean2: Tensor, 
    std2: Tensor
) -> Tensor:
    """
    Numerically stable KL divergence between two Gaussian distributions.
    
    KL[N(μ₁,σ₁²)||N(μ₂,σ₂²)] = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
    
    Args:
        mean1, std1: Parameters of first Gaussian
        mean2, std2: Parameters of second Gaussian
        
    Returns:
        KL divergence tensor
    """
    # Ensure numerical stability
    std1 = torch.clamp(std1, min=1e-8)
    std2 = torch.clamp(std2, min=1e-8)
    
    var1 = std1 ** 2
    var2 = std2 ** 2
    
    # Compute KL divergence components
    log_ratio = torch.log(std2) - torch.log(std1)
    var_term = var1 / var2
    mean_term = ((mean1 - mean2) ** 2) / var2
    
    kl_div = log_ratio + 0.5 * (var_term + mean_term - 1.0)
    
    return kl_div


def safe_log(x: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Safe logarithm that avoids log(0) by adding small epsilon.
    
    Args:
        x: Input tensor
        eps: Small positive constant to add
        
    Returns:
        Safe logarithm
    """
    return torch.log(torch.clamp(x, min=eps))


def safe_sqrt(x: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Safe square root that handles negative values.
    
    Args:
        x: Input tensor  
        eps: Small positive constant
        
    Returns:
        Safe square root
    """
    return torch.sqrt(torch.clamp(x, min=eps))


def robust_normalize(
    x: Tensor, 
    dim: Optional[int] = None, 
    eps: float = 1e-8,
    method: str = "l2"
) -> Tensor:
    """
    Robust tensor normalization with numerical stability.
    
    Args:
        x: Input tensor
        dim: Normalization dimension (None for global)
        eps: Numerical stability constant
        method: Normalization method ("l2", "l1", "max")
        
    Returns:
        Normalized tensor
    """
    if method == "l2":
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    elif method == "l1": 
        norm = torch.norm(x, p=1, dim=dim, keepdim=True)
    elif method == "max":
        norm = torch.max(torch.abs(x), dim=dim, keepdim=True)[0]
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Add epsilon for numerical stability
    norm = torch.clamp(norm, min=eps)
    
    return x / norm


def compute_grad_norm(parameters, norm_type: float = 2.0) -> float:
    """
    Compute gradient norm across model parameters.
    
    Args:
        parameters: Model parameters iterator
        norm_type: Type of norm (2.0 for L2, float('inf') for max)
        
    Returns:
        Total gradient norm
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return 0.0
    
    device = parameters[0].grad.device
    
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type
        )
    
    return total_norm.item()


def spectral_norm_conv(
    weight: Tensor, 
    u: Tensor, 
    num_power_iterations: int = 1
) -> Tuple[Tensor, Tensor]:
    """
    Compute spectral normalization for convolutional layers.
    
    Uses power iteration method to approximate largest singular value
    of the weight tensor when viewed as a matrix.
    
    Args:
        weight: Convolution weight tensor [out_ch, in_ch, ...]
        u: Left singular vector estimate
        num_power_iterations: Number of power iterations
        
    Returns:
        Tuple of (spectral_norm, updated_u)
    """
    # Reshape weight to matrix form
    weight_mat = weight.view(weight.size(0), -1)
    
    with torch.no_grad():
        for _ in range(num_power_iterations):
            # Power iteration: u^T W W^T u
            v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=1e-12)
            u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=1e-12)
    
    # Compute spectral norm
    sigma = torch.sum(u * torch.mv(weight_mat, v))
    
    return sigma, u


def adaptive_gradient_clipping(
    parameters, 
    clip_factor: float = 0.01,
    eps: float = 1e-3
) -> float:
    """
    Adaptive gradient clipping based on parameter norms.
    
    Clips gradients based on ratio of parameter norm to gradient norm,
    providing more stable training for different layer sizes.
    
    Args:
        parameters: Model parameters
        clip_factor: Clipping factor (smaller = more aggressive)
        eps: Small constant for numerical stability
        
    Returns:
        Applied clipping factor
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return 1.0
    
    # Compute parameter and gradient norms
    param_norm = torch.norm(torch.stack([torch.norm(p.detach()) for p in parameters]))
    grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters]))
    
    # Compute adaptive clipping value
    max_norm = clip_factor * param_norm / (grad_norm + eps)
    clip_coef = min(max_norm / (grad_norm + eps), 1.0)
    
    # Apply clipping
    for p in parameters:
        p.grad.detach().mul_(clip_coef)
    
    return clip_coef


def numerical_gradient_check(
    func,
    inputs: Tensor,
    eps: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> bool:
    """
    Verify gradients using finite differences.
    
    Args:
        func: Function to check (should return scalar)
        inputs: Input tensor requiring gradients
        eps: Finite difference step size
        rtol, atol: Relative and absolute tolerance
        
    Returns:
        True if gradients match finite differences
    """
    # Compute analytical gradients
    inputs.requires_grad_(True)
    output = func(inputs)
    analytical_grad = torch.autograd.grad(output, inputs)[0]
    
    # Compute numerical gradients
    numerical_grad = torch.zeros_like(inputs)
    
    for idx in torch.ndindex(inputs.shape):
        # Forward difference
        inputs_plus = inputs.clone()
        inputs_plus[idx] += eps
        output_plus = func(inputs_plus)
        
        # Backward difference  
        inputs_minus = inputs.clone()
        inputs_minus[idx] -= eps
        output_minus = func(inputs_minus)
        
        # Central difference
        numerical_grad[idx] = (output_plus - output_minus) / (2 * eps)
    
    # Check if gradients match
    return torch.allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)


def safe_inverse(matrix: Tensor, regularization: float = 1e-6) -> Tensor:
    """
    Compute matrix inverse with regularization for numerical stability.
    
    Adds regularization to diagonal elements to ensure invertibility.
    Uses Cholesky decomposition when possible for efficiency.
    
    Args:
        matrix: Input matrix [..., N, N]
        regularization: Diagonal regularization strength
        
    Returns:
        Regularized matrix inverse
    """
    # Add regularization to diagonal
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    regularized_matrix = matrix + regularization * identity
    
    try:
        # Try Cholesky decomposition for positive definite matrices
        if matrix.shape[-1] == matrix.shape[-2]:  # Square matrix
            chol = torch.linalg.cholesky(regularized_matrix)
            inverse = torch.cholesky_inverse(chol)
            return inverse
    except RuntimeError:
        pass  # Fall back to general inverse
    
    # General matrix inverse
    try:
        return torch.linalg.inv(regularized_matrix)
    except RuntimeError:
        # Final fallback: pseudo-inverse
        warnings.warn("Matrix inversion failed, using pseudo-inverse")
        return torch.linalg.pinv(regularized_matrix)


def stable_cholesky(matrix: Tensor, max_tries: int = 3) -> Tensor:
    """
    Numerically stable Cholesky decomposition with adaptive regularization.
    
    Attempts Cholesky decomposition with increasing regularization
    until successful or max attempts reached.
    
    Args:
        matrix: Positive definite matrix [..., N, N]
        max_tries: Maximum regularization attempts
        
    Returns:
        Cholesky factor L such that L @ L.T = matrix
    """
    regularization = 1e-6
    
    for attempt in range(max_tries):
        try:
            # Add regularization to diagonal
            identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
            regularized = matrix + regularization * identity
            
            # Attempt Cholesky decomposition
            return torch.linalg.cholesky(regularized)
            
        except RuntimeError:
            if attempt < max_tries - 1:
                regularization *= 10  # Increase regularization
                continue
            else:
                # Final attempt: use eigendecomposition
                eigenvals, eigenvecs = torch.linalg.eigh(matrix)
                eigenvals = torch.clamp(eigenvals, min=1e-8)  # Ensure positive
                
                # Reconstruct with clamped eigenvalues
                sqrt_eigenvals = torch.sqrt(eigenvals)
                return eigenvecs * sqrt_eigenvals.unsqueeze(-2)


def medical_image_stats(
    image: Tensor, 
    mask: Optional[Tensor] = None,
    percentiles: Tuple[float, float] = (1.0, 99.0)
) -> dict:
    """
    Compute comprehensive statistics for medical images.
    
    Calculates robust statistics commonly used in medical image processing,
    including percentile-based measures that are less sensitive to outliers.
    
    Args:
        image: Medical image tensor [..., D, H, W] or [..., H, W]
        mask: Optional binary mask for region of interest
        percentiles: Percentile range for robust statistics
        
    Returns:
        Dictionary of image statistics
    """
    if mask is not None:
        # Apply mask
        masked_values = image[mask > 0]
    else:
        # Use all values
        masked_values = image.flatten()
    
    # Remove NaN and infinite values
    valid_mask = torch.isfinite(masked_values)
    if valid_mask.sum() == 0:
        warnings.warn("No valid values found in image")
        return {}
    
    valid_values = masked_values[valid_mask]
    
    # Basic statistics
    stats = {
        'mean': torch.mean(valid_values).item(),
        'std': torch.std(valid_values, unbiased=True).item(),
        'min': torch.min(valid_values).item(),
        'max': torch.max(valid_values).item(),
        'median': torch.median(valid_values).item(),
        'num_voxels': len(valid_values),
    }
    
    # Percentile-based statistics
    p_low, p_high = percentiles
    stats[f'p{p_low:g}'] = torch.quantile(valid_values, p_low/100.0).item()
    stats[f'p{p_high:g}'] = torch.quantile(valid_values, p_high/100.0).item()
    
    # Robust statistics
    stats['iqr'] = stats[f'p75'] - stats[f'p25'] if 'p75' in stats and 'p25' in stats else 0.0
    stats['robust_mean'] = torch.mean(
        valid_values[(valid_values >= stats[f'p{p_low:g}']) & 
                    (valid_values <= stats[f'p{p_high:g}'])]
    ).item()
    
    # Signal-to-noise ratio approximation
    if stats['std'] > 1e-8:
        stats['snr_approx'] = stats['mean'] / stats['std']
    else:
        stats['snr_approx'] = float('inf')
    
    # Histogram statistics
    hist = torch.histc(valid_values, bins=100)
    hist_normalized = hist / hist.sum()
    
    # Entropy approximation
    hist_nonzero = hist_normalized[hist_normalized > 1e-12]
    if len(hist_nonzero) > 0:
        stats['entropy'] = -torch.sum(hist_nonzero * torch.log(hist_nonzero)).item()
    else:
        stats['entropy'] = 0.0
    
    return stats


def intensity_normalization(
    image: Tensor,
    method: str = "z_score",
    mask: Optional[Tensor] = None,
    percentiles: Tuple[float, float] = (1.0, 99.0),
    target_mean: float = 0.0,
    target_std: float = 1.0
) -> Tensor:
    """
    Apply intensity normalization to medical images.
    
    Supports multiple normalization methods commonly used in medical imaging:
    - Z-score normalization
    - Min-max scaling  
    - Percentile-based normalization
    - Histogram matching (simplified)
    
    Args:
        image: Input image tensor
        method: Normalization method
        mask: Optional mask for computing statistics
        percentiles: Percentile range for robust normalization
        target_mean, target_std: Target statistics for z-score
        
    Returns:
        Normalized image tensor
    """
    # Compute image statistics
    stats = medical_image_stats(image, mask, percentiles)
    
    if method == "z_score":
        # Standard z-score normalization
        if mask is not None:
            mean = torch.mean(image[mask > 0])
            std = torch.std(image[mask > 0], unbiased=True)
        else:
            mean = torch.mean(image)
            std = torch.std(image, unbiased=True)
        
        std = torch.clamp(std, min=1e-8)  # Avoid division by zero
        normalized = (image - mean) / std
        
        # Scale to target statistics
        normalized = normalized * target_std + target_mean
    
    elif method == "min_max":
        # Min-max scaling to [0, 1]
        min_val = stats['min']
        max_val = stats['max']
        
        if abs(max_val - min_val) < 1e-8:
            normalized = torch.zeros_like(image)
        else:
            normalized = (image - min_val) / (max_val - min_val)
    
    elif method == "percentile":
        # Percentile-based normalization
        p_low, p_high = percentiles
        low_val = stats[f'p{p_low:g}']
        high_val = stats[f'p{p_high:g}']
        
        if abs(high_val - low_val) < 1e-8:
            normalized = torch.zeros_like(image)
        else:
            normalized = (image - low_val) / (high_val - low_val)
            normalized = torch.clamp(normalized, 0, 1)  # Clip outliers
    
    elif method == "robust_z_score":
        # Z-score using robust statistics (median, MAD)
        if mask is not None:
            values = image[mask > 0]
        else:
            values = image.flatten()
        
        median = torch.median(values)
        mad = torch.median(torch.abs(values - median))  # Median Absolute Deviation
        mad = torch.clamp(mad, min=1e-8)
        
        # Scale MAD to approximate standard deviation
        mad_scaled = mad * 1.4826  # For Gaussian distribution
        
        normalized = (image - median) / mad_scaled
        normalized = normalized * target_std + target_mean
    
    elif method == "nyul":
        # Simplified Nyul histogram matching
        # Uses percentile landmarks for normalization
        landmarks = [1, 10, 25, 50, 75, 90, 99]
        
        # Compute current landmarks
        current_landmarks = []
        for p in landmarks:
            if mask is not None:
                values = image[mask > 0]
            else:
                values = image.flatten()
            landmark = torch.quantile(values, p/100.0)
            current_landmarks.append(landmark)
        
        # Define standard landmarks (example values)
        standard_landmarks = torch.tensor([0, 10, 25, 50, 75, 90, 100], dtype=image.dtype, device=image.device)
        
        # Linear interpolation between landmarks
        normalized = torch.zeros_like(image)
        for i in range(len(landmarks) - 1):
            mask_range = (image >= current_landmarks[i]) & (image <= current_landmarks[i + 1])
            if mask_range.sum() > 0:
                # Linear interpolation
                alpha = (image[mask_range] - current_landmarks[i]) / (current_landmarks[i + 1] - current_landmarks[i] + 1e-8)
                normalized[mask_range] = standard_landmarks[i] + alpha * (standard_landmarks[i + 1] - standard_landmarks[i])
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def gradient_magnitude_3d(image: Tensor, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tensor:
    """
    Compute 3D gradient magnitude for edge detection in medical images.
    
    Uses central differences with proper boundary handling and
    accounts for anisotropic voxel spacing.
    
    Args:
        image: 3D image tensor [B, C, D, H, W] or [D, H, W]
        spacing: Voxel spacing in (depth, height, width) order
        
    Returns:
        Gradient magnitude tensor
    """
    if image.dim() == 3:
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif image.dim() == 4:
        image = image.unsqueeze(0)  # Add batch dim
    
    # Sobel operators for 3D
    sobel_z = torch.tensor([[[[-1, -2, -1],
                             [-2, -4, -2], 
                             [-1, -2, -1]]],
                           [[[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]]],
                           [[[1, 2, 1],
                             [2, 4, 2],
                             [1, 2, 1]]]], dtype=image.dtype, device=image.device).unsqueeze(1) / (spacing[0] * 16)
    
    sobel_y = torch.tensor([[[[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]]],
                           [[[-2, -4, -2],
                             [0, 0, 0], 
                             [2, 4, 2]]],
                           [[[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]]]], dtype=image.dtype, device=image.device).unsqueeze(1) / (spacing[1] * 16)
    
    sobel_x = torch.tensor([[[[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]],
                           [[[-2, 0, 2],
                             [-4, 0, 4],
                             [-2, 0, 2]]],
                           [[[-1, 0, 1], 
                             [-2, 0, 2],
                             [-1, 0, 1]]]], dtype=image.dtype, device=image.device).unsqueeze(1) / (spacing[2] * 16)
    
    # Apply convolutions with proper padding
    grad_z = F.conv3d(image, sobel_z, padding=1)
    grad_y = F.conv3d(image, sobel_y, padding=1)  
    grad_x = F.conv3d(image, sobel_x, padding=1)
    
    # Compute gradient magnitude
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-12)
    
    return gradient_magnitude.squeeze()  # Remove added dimensions


def tissue_contrast_measure(
    image: Tensor,
    tissue_seg: Tensor,
    tissue_pairs: List[Tuple[int, int]] = [(1, 2), (1, 3), (2, 3)]
) -> Dict[str, float]:
    """
    Compute tissue contrast measures for medical image quality assessment.
    
    Calculates contrast-to-noise ratio (CNR) and other contrast measures
    between different tissue types based on segmentation masks.
    
    Args:
        image: Medical image tensor
        tissue_seg: Tissue segmentation mask
        tissue_pairs: Pairs of tissue labels to compare
        
    Returns:
        Dictionary of contrast measures
    """
    contrast_measures = {}
    
    for tissue1, tissue2 in tissue_pairs:
        # Extract tissue regions
        mask1 = (tissue_seg == tissue1)
        mask2 = (tissue_seg == tissue2)
        
        if mask1.sum() == 0 or mask2.sum() == 0:
            continue
        
        values1 = image[mask1]
        values2 = image[mask2]
        
        # Compute statistics
        mean1, std1 = torch.mean(values1), torch.std(values1, unbiased=True)
        mean2, std2 = torch.mean(values2), torch.std(values2, unbiased=True)
        
        # Contrast measures
        pair_name = f"tissue_{tissue1}_vs_{tissue2}"
        
        # Michelson contrast
        if abs(mean1 + mean2) > 1e-8:
            contrast_measures[f"{pair_name}_michelson"] = abs(mean1 - mean2) / (mean1 + mean2)
        
        # Weber contrast  
        if abs(mean2) > 1e-8:
            contrast_measures[f"{pair_name}_weber"] = (mean1 - mean2) / mean2
        
        # Contrast-to-noise ratio
        pooled_std = torch.sqrt((std1**2 + std2**2) / 2)
        if pooled_std > 1e-8:
            contrast_measures[f"{pair_name}_cnr"] = abs(mean1 - mean2) / pooled_std
        
        # Signal difference to noise ratio
        noise_std = torch.sqrt(std1**2 + std2**2)
        if noise_std > 1e-8:
            contrast_measures[f"{pair_name}_sdnr"] = abs(mean1 - mean2) / noise_std
    
    return {k: v.item() if torch.is_tensor(v) else v for k, v in contrast_measures.items()}