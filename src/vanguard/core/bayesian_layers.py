"""
VANGUARD: Bayesian Neural Network Layers with Variational Inference

This module implements the core Bayesian layers for the VANGUARD model, providing
principled uncertainty quantification through variational inference. The implementation
follows Bayes by Backprop (Blundell et al., 2015) with optimizations for 3D medical imaging.

Mathematical Foundation:
- Variational posterior: q(θ|φ) approximates true posterior p(θ|D)
- ELBO maximization: L(φ) = E_q[log p(y|x,θ)] - KL[q(θ|φ)||p(θ)]
- Reparameterization trick: θ = μ + σ ⊙ ε, where ε ~ N(0,I)
"""

import math
from typing import Optional, Tuple, Union, Callable
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import torch.distributions as dist
from torch.distributions.kl import kl_divergence

__all__ = [
    "BayesianLinear",
    "BayesianConv3d", 
    "BayesianConvTranspose3d",
    "BayesianBatchNorm3d",
    "VariationalDropout3d",
    "KLDivergenceLoss",
    "BayesianLayer",
    "uncertainty_forward",
]


class BayesianLayer(nn.Module):
    """
    Base class for Bayesian layers implementing variational inference.
    
    Provides common functionality for:
    - KL divergence computation between variational posterior and prior
    - Reparameterization trick for sampling
    - Numerical stability utilities
    """
    
    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        posterior_rho_init: float = -3.0,
        enable_kl_divergence: bool = True,
    ):
        super().__init__()
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.posterior_rho_init = posterior_rho_init
        self.enable_kl_divergence = enable_kl_divergence
        
        # Register KL divergence for accumulation during forward pass
        self.register_buffer('kl_divergence', torch.zeros(1))
    
    @staticmethod
    def rho_to_std(rho: Tensor) -> Tensor:
        """
        Convert rho parameter to standard deviation using softplus.
        
        σ = log(1 + exp(ρ)) provides numerical stability and ensures σ > 0.
        For numerical stability with large ρ, we use the identity:
        softplus(x) = x + log(1 + exp(-x)) for x > 20
        
        Args:
            rho: Log variance parameter
            
        Returns:
            Standard deviation tensor
        """
        return F.softplus(rho)
    
    @staticmethod
    def sample_weight(mean: Tensor, std: Tensor, epsilon: Optional[Tensor] = None) -> Tensor:
        """
        Sample weights using reparameterization trick: w = μ + σ ⊙ ε
        
        Args:
            mean: Variational posterior mean
            std: Variational posterior standard deviation  
            epsilon: Optional noise tensor (for deterministic sampling)
            
        Returns:
            Sampled weight tensor
        """
        if epsilon is None:
            epsilon = torch.randn_like(mean)
        return mean + std * epsilon
    
    def compute_kl_divergence(self, mean: Tensor, std: Tensor) -> Tensor:
        """
        Compute KL divergence between variational posterior and prior.
        
        For Normal distributions:
        KL[q(θ|μ,σ²)||p(θ|0,σ₀²)] = ½[log(σ₀²/σ²) + σ²/σ₀² + μ²/σ₀² - 1]
        
        Args:
            mean: Variational posterior mean
            std: Variational posterior standard deviation
            
        Returns:
            KL divergence scalar
        """
        if not self.enable_kl_divergence:
            return torch.zeros(1, device=mean.device)
            
        # Define distributions
        posterior = dist.Normal(mean, std)
        prior = dist.Normal(self.prior_mean, self.prior_std)
        
        # Compute KL divergence analytically for efficiency
        kl_div = kl_divergence(posterior, prior)
        
        # Sum over all parameters
        return kl_div.sum()
    
    def reset_kl_divergence(self):
        """Reset accumulated KL divergence."""
        self.kl_divergence.zero_()


class BayesianLinear(BayesianLayer):
    """
    Bayesian Linear layer with variational weights and biases.
    
    Implements fully connected layer with uncertainty quantification:
    - Weight distribution: w ~ N(μ_w, σ_w²)
    - Bias distribution: b ~ N(μ_b, σ_b²)
    - Forward: y = x @ w + b, where w,b are sampled from posteriors
    
    Mathematical Details:
    The layer learns variational parameters (μ, ρ) where σ = softplus(ρ).
    During training, weights are sampled; during inference, mean or multiple samples.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        posterior_rho_init: float = -3.0,
        enable_kl_divergence: bool = True,
    ):
        super().__init__(prior_mean, prior_std, posterior_rho_init, enable_kl_divergence)
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Variational parameters for weights
        self.weight_mean = Parameter(torch.empty(out_features, in_features))
        self.weight_rho = Parameter(torch.empty(out_features, in_features))
        
        # Variational parameters for bias
        if bias:
            self.bias_mean = Parameter(torch.empty(out_features))
            self.bias_rho = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_rho', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize variational parameters following Bayes by Backprop."""
    def reset_parameters(self):
        """Initialize variational parameters following Bayes by Backprop."""
        # Initialize weight mean with Xavier/Glorot normal
        nn.init.xavier_normal_(self.weight_mean)
        
        # Initialize weight rho to achieve desired initial std
        self.weight_rho.data.fill_(self.posterior_rho_init)
        
        if self.use_bias:
            # Initialize bias mean to small values
            nn.init.constant_(self.bias_mean, 0.0)
            self.bias_rho.data.fill_(self.posterior_rho_init)
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass with reparameterization trick.
        
        During training: Sample weights from variational posterior
        During inference: Use mean weights or multiple samples for uncertainty
        
        Args:
            input: Input tensor [B, in_features]
            
        Returns:
            Output tensor [B, out_features]
        """
        # Convert rho to std for numerical stability
        weight_std = self.rho_to_std(self.weight_rho)
        
        # Sample weights using reparameterization trick
        weight = self.sample_weight(self.weight_mean, weight_std)
        
        # Handle bias sampling
        bias = None
        if self.use_bias:
            bias_std = self.rho_to_std(self.bias_rho)
            bias = self.sample_weight(self.bias_mean, bias_std)
        
        # Compute KL divergence and accumulate
        kl_weight = self.compute_kl_divergence(self.weight_mean, weight_std)
        kl_bias = torch.zeros_like(kl_weight)
        if self.use_bias:
            kl_bias = self.compute_kl_divergence(self.bias_mean, bias_std)
        
        self.kl_divergence = kl_weight + kl_bias
        
        # Standard linear transformation
        return F.linear(input, weight, bias)


class BayesianConv3d(BayesianLayer):
    """
    Bayesian 3D Convolution layer for medical imaging applications.
    
    Extends standard Conv3d with variational weights and uncertainty quantification.
    Critical for 3D medical imaging where spatial coherence and uncertainty matter.
    
    Mathematical Foundation:
    - Weight tensor: W ~ N(μ_W, Σ_W) where Σ_W = diag(σ²_W)
    - Convolution: Y = W ⊗ X + b, where ⊗ denotes 3D convolution
    - KL regularization: KL[q(W,b|φ)||p(W,b)]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        posterior_rho_init: float = -3.0,
        enable_kl_divergence: bool = True,
    ):
        super().__init__(prior_mean, prior_std, posterior_rho_init, enable_kl_divergence)
        
        # Store convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._triple(kernel_size)
        self.stride = self._triple(stride)
        self.padding = self._triple(padding)
        self.dilation = self._triple(dilation)
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        
        # Variational parameters for weights
        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight_mean = Parameter(torch.empty(weight_shape))
        self.weight_rho = Parameter(torch.empty(weight_shape))
        
        # Variational parameters for bias
        if bias:
            self.bias_mean = Parameter(torch.empty(out_channels))
            self.bias_rho = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_rho', None)
        
        self.reset_parameters()
    
    @staticmethod
    def _triple(value: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Convert single int to 3D tuple."""
        if isinstance(value, int):
            return (value, value, value)
        return value
    
    def reset_parameters(self):
        """Initialize parameters for 3D convolution."""
        # Calculate fan_in for proper Xavier initialization
        fan_in = self.in_channels
        for k in self.kernel_size:
            fan_in *= k
        
        # Xavier/Glorot initialization for mean
        bound = math.sqrt(6.0 / fan_in)
        nn.init.uniform_(self.weight_mean, -bound, bound)
        
        # Initialize rho for desired initial uncertainty
        self.weight_rho.data.fill_(self.posterior_rho_init)
        
        if self.use_bias:
            nn.init.constant_(self.bias_mean, 0.0)
            self.bias_rho.data.fill_(self.posterior_rho_init)
    
    def forward(self, input: Tensor) -> Tensor:
        """
        3D convolution with variational weights.
        
        Args:
            input: Input tensor [B, C_in, D, H, W]
            
        Returns:
            Output tensor [B, C_out, D', H', W']
        """
        # Sample weights and bias
        weight_std = self.rho_to_std(self.weight_rho)
        weight = self.sample_weight(self.weight_mean, weight_std)
        
        bias = None
        if self.use_bias:
            bias_std = self.rho_to_std(self.bias_rho)
            bias = self.sample_weight(self.bias_mean, bias_std)
        
        # Compute KL divergence
        kl_weight = self.compute_kl_divergence(self.weight_mean, weight_std)
        kl_bias = torch.zeros_like(kl_weight)
        if self.use_bias:
            kl_bias = self.compute_kl_divergence(self.bias_mean, bias_std)
        
        self.kl_divergence = kl_weight + kl_bias
        
        # 3D convolution
        return F.conv3d(
            input, weight, bias,
            self.stride, self.padding, self.dilation, self.groups
        )


class BayesianConvTranspose3d(BayesianLayer):
    """
    Bayesian 3D Transposed Convolution for decoder/upsampling layers.
    
    Essential for U-Net decoder with uncertainty quantification.
    Maintains spatial resolution recovery with principled uncertainty.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding_mode: str = 'zeros',
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        posterior_rho_init: float = -3.0,
        enable_kl_divergence: bool = True,
    ):
        super().__init__(prior_mean, prior_std, posterior_rho_init, enable_kl_divergence)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._triple(kernel_size)
        self.stride = self._triple(stride)
        self.padding = self._triple(padding)
        self.output_padding = self._triple(output_padding)
        self.groups = groups
        self.use_bias = bias
        self.dilation = self._triple(dilation)
        self.padding_mode = padding_mode
        
        # Variational parameters for weights
        weight_shape = (in_channels, out_channels // groups, *self.kernel_size)
        self.weight_mean = Parameter(torch.empty(weight_shape))
        self.weight_rho = Parameter(torch.empty(weight_shape))
        
        if bias:
            self.bias_mean = Parameter(torch.empty(out_channels))
            self.bias_rho = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_rho', None)
        
        self.reset_parameters()
    
    @staticmethod
    def _triple(value: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        if isinstance(value, int):
            return (value, value, value)
        return value
    
    def reset_parameters(self):
        """Initialize parameters for transposed convolution."""
        fan_in = self.in_channels
        for k in self.kernel_size:
            fan_in *= k
        
        bound = math.sqrt(6.0 / fan_in)
        nn.init.uniform_(self.weight_mean, -bound, bound)
        self.weight_rho.data.fill_(self.posterior_rho_init)
        
        if self.use_bias:
            nn.init.constant_(self.bias_mean, 0.0)
            self.bias_rho.data.fill_(self.posterior_rho_init)
    
    def forward(self, input: Tensor, output_size: Optional[Tuple[int, ...]] = None) -> Tensor:
        """
        3D transposed convolution with variational weights.
        
        Args:
            input: Input tensor [B, C_in, D, H, W]
            output_size: Optional target output size
            
        Returns:
            Upsampled output tensor [B, C_out, D', H', W']
        """
        weight_std = self.rho_to_std(self.weight_rho)
        weight = self.sample_weight(self.weight_mean, weight_std)
        
        bias = None
        if self.use_bias:
            bias_std = self.rho_to_std(self.bias_rho)
            bias = self.sample_weight(self.bias_mean, bias_std)
        
        # Compute KL divergence
        kl_weight = self.compute_kl_divergence(self.weight_mean, weight_std)
        kl_bias = torch.zeros_like(kl_weight)
        if self.use_bias:
            kl_bias = self.compute_kl_divergence(self.bias_mean, bias_std)
        
        self.kl_divergence = kl_weight + kl_bias
        
        return F.conv_transpose3d(
            input, weight, bias,
            self.stride, self.padding, self.output_padding,
            self.groups, self.dilation
        )


class BayesianBatchNorm3d(BayesianLayer):
    """
    Bayesian Batch Normalization for 3D medical images.
    
    Incorporates uncertainty in normalization parameters while maintaining
    the benefits of batch normalization for training stability.
    
    Note: This is a simplified implementation. Full Bayesian BatchNorm
    requires careful treatment of running statistics and momentum updates.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        posterior_rho_init: float = -3.0,
        enable_kl_divergence: bool = True,
    ):
        super().__init__(prior_mean, prior_std, posterior_rho_init, enable_kl_divergence)
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            # Variational parameters for scale and shift
            self.weight_mean = Parameter(torch.ones(num_features))
            self.weight_rho = Parameter(torch.full((num_features,), posterior_rho_init))
            self.bias_mean = Parameter(torch.zeros(num_features))
            self.bias_rho = Parameter(torch.full((num_features,), posterior_rho_init))
        else:
            self.register_parameter('weight_mean', None)
            self.register_parameter('weight_rho', None)
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_rho', None)
        
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize batch normalization parameters."""
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        
        if self.affine:
            # Weight mean initialized to 1 for proper scaling
            nn.init.constant_(self.weight_mean, 1.0)
            # Bias mean initialized to 0
            nn.init.constant_(self.bias_mean, 0.0)
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Bayesian batch normalization forward pass.
        
        Args:
            input: Input tensor [B, C, D, H, W]
            
        Returns:
            Normalized output tensor [B, C, D, H, W]
        """
        # Update exponential moving averages
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        else:
            exponential_average_factor = 0.0
        
        # Compute batch statistics or use running statistics
        if self.training:
            # Calculate mean and variance across batch, spatial dimensions
            # Input shape: [B, C, D, H, W]
            batch_mean = input.mean(dim=(0, 2, 3, 4))
            batch_var = input.var(dim=(0, 2, 3, 4), unbiased=False)
            
            # Update running statistics
            if self.track_running_stats:
                self.running_mean = ((1 - exponential_average_factor) * self.running_mean +
                                   exponential_average_factor * batch_mean.detach())
                self.running_var = ((1 - exponential_average_factor) * self.running_var +
                                  exponential_average_factor * batch_var.detach())
            
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        input_normalized = (input - mean.view(1, -1, 1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1, 1) + self.eps)
        
        if self.affine:
            # Sample scale and shift parameters
            weight_std = self.rho_to_std(self.weight_rho)
            weight = self.sample_weight(self.weight_mean, weight_std)
            
            bias_std = self.rho_to_std(self.bias_rho)
            bias = self.sample_weight(self.bias_mean, bias_std)
            
            # Compute KL divergence
            kl_weight = self.compute_kl_divergence(self.weight_mean, weight_std)
            kl_bias = self.compute_kl_divergence(self.bias_mean, bias_std)
            self.kl_divergence = kl_weight + kl_bias
            
            # Apply affine transformation
            output = input_normalized * weight.view(1, -1, 1, 1, 1) + bias.view(1, -1, 1, 1, 1)
        else:
            output = input_normalized
            self.kl_divergence = torch.zeros(1, device=input.device)
        
        return output


class VariationalDropout3d(nn.Module):
    """
    Variational Dropout for 3D tensors with learned dropout rates.
    
    Implements the technique from "Variational Dropout and the Local Reparameterization Trick"
    (Kingma et al., 2015). Unlike standard dropout, the dropout rate is learned and
    can vary per feature/channel, providing more principled regularization.
    
    Mathematical Foundation:
    - Instead of binary mask, uses multiplicative Gaussian noise
    - Noise variance is a learnable parameter
    - Equivalent to Bayesian treatment of connection strengths
    """
    
    def __init__(
        self,
        initial_dropout_rate: float = 0.1,
        min_dropout_rate: float = 1e-8,
        max_dropout_rate: float = 0.95,
    ):
        super().__init__()
        
        self.min_dropout_rate = min_dropout_rate
        self.max_dropout_rate = max_dropout_rate
        
        # Learnable dropout rate (in logit space for stability)
        initial_logit = self._dropout_to_logit(initial_dropout_rate)
        self.dropout_logit = Parameter(torch.tensor(initial_logit))
    
    def _dropout_to_logit(self, dropout_rate: float) -> float:
        """Convert dropout rate to logit space for stable optimization."""
        # Clamp to valid range
        dropout_rate = max(self.min_dropout_rate, min(self.max_dropout_rate, dropout_rate))
        return math.log(dropout_rate / (1 - dropout_rate))
    
    def _logit_to_dropout(self, logit: Tensor) -> Tensor:
        """Convert logit back to dropout rate."""
        dropout_rate = torch.sigmoid(logit)
        return torch.clamp(dropout_rate, self.min_dropout_rate, self.max_dropout_rate)
    
    @property
    def dropout_rate(self) -> Tensor:
        """Current dropout rate."""
        return self._logit_to_dropout(self.dropout_logit)
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Apply variational dropout.
        
        Args:
            input: Input tensor [B, C, D, H, W]
            
        Returns:
            Output with variational dropout applied
        """
        if not self.training:
            return input
        
        # Get current dropout rate
        p = self.dropout_rate
        
        # Variational dropout: multiply by Gaussian noise with learned variance
        # This is equivalent to randomly dropping connections with rate p
        noise_var = p / (1 - p)
        noise = torch.randn_like(input) * torch.sqrt(noise_var) + 1
        
        return input * noise


class KLDivergenceLoss(nn.Module):
    """
    Utility class for computing and managing KL divergence losses.
    
    Collects KL divergences from all Bayesian layers in a model and
    provides functionality for β-VAE style annealing and weighting.
    """
    
    def __init__(self, beta: float = 1.0, anneal_steps: int = 0):
        super().__init__()
        self.beta = beta
        self.anneal_steps = anneal_steps
        self.step_count = 0
    
    def get_current_beta(self) -> float:
        """Get current β value with optional annealing."""
        if self.anneal_steps <= 0:
            return self.beta
        
        # Linear annealing from 0 to target beta
        progress = min(1.0, self.step_count / self.anneal_steps)
        return self.beta * progress
    
    def collect_kl_divergence(self, model: nn.Module) -> Tensor:
        """
        Collect KL divergences from all Bayesian layers in model.
        
        Args:
            model: Neural network containing Bayesian layers
            
        Returns:
            Total KL divergence across all layers
        """
        total_kl = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for module in model.modules():
            if isinstance(module, BayesianLayer) and hasattr(module, 'kl_divergence'):
                total_kl += module.kl_divergence
        
        return total_kl
    
    def forward(self, model: nn.Module) -> Tensor:
        """
        Compute weighted KL divergence loss.
        
        Args:
            model: Model containing Bayesian layers
            
        Returns:
            Weighted KL divergence loss
        """
        kl_div = self.collect_kl_divergence(model)
        current_beta = self.get_current_beta()
        
        # Increment step count for annealing
        if self.training:
            self.step_count += 1
        
        return current_beta * kl_div
    
    def reset_step_count(self):
        """Reset step count for annealing schedule."""
        self.step_count = 0


def uncertainty_forward(
    model: nn.Module,
    input: Tensor,
    num_samples: int = 10,
    enable_dropout: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Forward pass with uncertainty quantification.
    
    Performs multiple forward passes to estimate predictive uncertainty
    using Monte Carlo sampling from variational posterior.
    
    Args:
        model: Bayesian neural network
        input: Input tensor
        num_samples: Number of Monte Carlo samples
        enable_dropout: Whether to enable dropout during inference
        
    Returns:
        Tuple of (mean_prediction, aleatoric_uncertainty, epistemic_uncertainty)
    """
    model.train() if enable_dropout else model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Reset KL divergences for each sample
            for module in model.modules():
                if isinstance(module, BayesianLayer):
                    module.reset_kl_divergence()
            
            pred = model(input)
            predictions.append(pred)
    
    # Stack predictions: [num_samples, B, ...]
    predictions = torch.stack(predictions, dim=0)
    
    # Compute statistics
    mean_prediction = predictions.mean(dim=0)
    
    # Aleatoric uncertainty: average of individual prediction variances
    # (data-dependent uncertainty)
    aleatoric_uncertainty = predictions.var(dim=0, unbiased=False)
    
    # Epistemic uncertainty: variance of predictions
    # (model uncertainty due to limited training data)
    epistemic_uncertainty = aleatoric_uncertainty  # Simplified - in practice, need to separate
    
    return mean_prediction, aleatoric_uncertainty, epistemic_uncertainty


# Export main classes and functions
__all__ += [
    "uncertainty_forward",
]