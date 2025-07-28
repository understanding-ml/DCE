"""
No Gradient Mixin for Strategy Classes - Pure Random Sampling

This module provides pure random sampling functionality without any gradient guidance.
This is the original implementation that does not use gradient information.
"""

import torch
import numpy as np


class NoGradientMixin:
    """
    Mixin class for pure random sampling without gradient guidance.
    
    This is the original implementation that provides traditional random sampling
    without using any gradient information from the differentiable terms.
    """
    
    def __init__(self, explainer, random_state=None, **kwargs):
        """
        Initialize pure random sampling parameters.
        
        Args:
            explainer: The DCE explainer instance
            random_state: Random state for reproducibility
        """
        # Don't call super().__init__() as this is a mixin
        self.explainer = explainer
        
        # Initialize random state components
        if hasattr(self, '_rng') and hasattr(self, '_torch_rng'):
            pass  # Already initialized by the concrete strategy class
        else:
            self._rng = np.random.RandomState(random_state)
            self._torch_rng = torch.Generator(device=explainer.device).manual_seed(random_state or 0)
    
    def _apply_random_sampling_to_feature(self, cand, eta, ref_idx, idx, idx_feat):
        """Apply pure random sampling to a single continuous feature"""
        min_val = self.explainer.X_prime[:, idx_feat].min()
        max_val = self.explainer.X_prime[:, idx_feat].max()
        rand_val = torch.rand(1, generator=self._torch_rng, device=self.explainer.device)
        sampled_val = min_val + rand_val * (max_val - min_val)
        cand[idx, idx_feat] = (1 - eta) * self.explainer.X_prime[ref_idx, idx_feat] + eta * sampled_val