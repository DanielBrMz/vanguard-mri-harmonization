"""
VANGUARD: Uncertainty Quantification Module

This module implements comprehensive uncertainty quantification methods for 
medical image harmonization, including calibration, decomposition of uncertainties,
and validation metrics.

Mathematical Foundation:
- Epistemic uncertainty: Model uncertainty due to limited training data
- Aleatoric uncertainty: Data-dependent uncertainty (measurement noise, etc.)
- Predictive uncertainty: Total uncertainty = Epistemic + Aleatoric
- Calibration: Alignment between confidence and accuracy

Author: Daniel Barreras
License: MIT
"""

import math
from typing import Tuple, List, Optional, Dict, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

__all__ = [
    "TemperatureScaling",
    "ExpectedCalibrationError", 
    "UncertaintyDecomposer",
    "ReliabilityDiagram",
    "BayesianModelEnsemble",
    "MonteCarloDropoutWrapper",
    "PredictiveEntropyCalculator",
    "UncertaintyCalibrator",
]


class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration using temperature scaling.
    
    Implements the method from "On Calibration of Modern Neural Networks" 
    (Guo et al., 2017). Learns a single temperature parameter T to rescale
    logits: p_calibrated = softmax(z/T), where z are logits.
    
    For regression tasks, applies temperature scaling to uncertainty estimates.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
    
    def forward(self, logits: Tensor) -> Tensor:
        """
        Apply temperature scaling to logits or uncertainty estimates.
        
        Args:
            logits: Raw model outputs [B, ...] 
            
        Returns:
            Temperature-scaled outputs
        """
        return logits / self.temperature
    
    def calibrate(
        self, 
        logits: Tensor, 
        targets: Tensor, 
        lr: float = 1e-3,
        max_iter: int = 1000,
        tolerance: float = 1e-6
    ):
        """
        Calibrate temperature parameter using validation data.
        
        Args:
            logits: Model outputs on validation set
            targets: Ground truth targets
            lr: Learning rate for temperature optimization
            max_iter: Maximum optimization iterations
            tolerance: Convergence tolerance
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            # For regression, use Gaussian NLL with scaled variance
            scaled_logits = self.forward(logits)
            loss = F.mse_loss(scaled_logits, targets)  # Simplified for regression
            loss.backward()
            return loss
        
        old_loss = float('inf')
        for _ in range(max_iter):
            loss = optimizer.step(eval_loss)
            if abs(old_loss - loss.item()) < tolerance:
                break
            old_loss = loss.item()


class ExpectedCalibrationError(nn.Module):
    """
    Compute Expected Calibration Error (ECE) for uncertainty quantification.
    
    ECE measures the difference between prediction confidence and accuracy:
    ECE = Σ_m (n_m/n) |acc(B_m) - conf(B_m)|
    
    where B_m are bins of predictions with similar confidence levels.
    """
    
    def __init__(self, n_bins: int = 15, bin_boundaries: Optional[List[float]] = None):
        super().__init__()
        self.n_bins = n_bins
        self.bin_boundaries = bin_boundaries or torch.linspace(0, 1, n_bins + 1)
    
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor, 
        confidences: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute ECE and reliability statistics.
        
        Args:
            predictions: Model predictions [B, ...]
            targets: Ground truth targets [B, ...]
            confidences: Confidence scores [B]. If None, uses prediction magnitude.
            
        Returns:
            Tuple of (ECE value, detailed statistics dictionary)
        """
        if confidences is None:
            # For regression, use prediction magnitude as confidence proxy
            confidences = torch.abs(predictions.flatten())
        
        predictions = predictions.flatten()
        targets = targets.flatten() 
        confidences = confidences.flatten()
        
        # Compute accuracies (for regression, use negative squared error as proxy)
        accuracies = -((predictions - targets) ** 2)  # Higher is better
        
        bin_boundaries = torch.tensor(self.bin_boundaries, device=predictions.device)
        
        ece = torch.tensor(0.0, device=predictions.device)