"""
Gradient Guidance Mixin for Strategy Classes

This module provides common functionality for gradient-guided sampling
that can be used across all strategy classes.
"""

import torch
import numpy as np
import math


class GradientGuidanceMixin:
    """
    Mixin class that provides gradient guidance functionality for optimization strategies.
    
    This mixin adds the ability to use gradient information from the differentiable term1
    (SWD part) to guide the sampling direction, while maintaining random exploration
    for the non-differentiable term2.
    """
    
    def __init__(self, explainer, use_gradient_guidance=False, cone_angle=math.pi/4, random_state=None, **kwargs):
        """
        Initialize gradient guidance parameters.
        
        Args:
            explainer: The DCE explainer instance
            use_gradient_guidance (bool): Whether to use gradient guidance (default: False)
            cone_angle (float): Cone angle in radians for guided sampling (default: π/4 = 45°)
            random_state: Random state for reproducibility
        """
        # Don't call super().__init__() as this is a mixin
        self.explainer = explainer
        self.use_gradient_guidance = use_gradient_guidance
        self.cone_angle = cone_angle
        
        # Initialize random state components that might be needed for gradient guidance
        if hasattr(self, '_rng') and hasattr(self, '_torch_rng'):
            pass  # Already initialized by the concrete strategy class
        else:
            self._rng = np.random.RandomState(random_state)
            self._torch_rng = torch.Generator(device=explainer.device).manual_seed(random_state or 0)
        
    def _compute_term1_gradient(self, X, eta):
        """Compute gradient of term1 with respect to X"""
        with torch.enable_grad():
            X_temp = X.clone().detach().requires_grad_(True)
            
            # Recompute term1 (SWD part)
            X_s = X_temp[:, self.explainer.explain_indices] * self.explainer.costs_vector_reshaped
            X_t = self.explainer.X_prime[:, self.explainer.explain_indices] * self.explainer.costs_vector_reshaped
            
            n, m = X_s.shape[0], X_t.shape[0]
            num_features = X_s.shape[1]
            
            # Get thetas from SWD object and ensure correct dimension
            thetas = []
            for theta in self.explainer.swd.thetas:
                theta_tensor = torch.from_numpy(theta).float().to(self.explainer.device)
                # Ensure theta has the correct dimension for the features
                if theta_tensor.shape[0] != num_features:
                    # Pad or truncate theta to match feature dimension
                    if theta_tensor.shape[0] < num_features:
                        # Pad with zeros
                        padding = torch.zeros(num_features - theta_tensor.shape[0], device=self.explainer.device)
                        theta_tensor = torch.cat([theta_tensor, padding])
                    else:
                        # Truncate
                        theta_tensor = theta_tensor[:num_features]
                thetas.append(theta_tensor)
            
            # Convert to tensor stack
            thetas_stack = torch.stack(thetas)  # [num_thetas, num_features]
            
            X_proj = torch.matmul(X_s, thetas_stack.T)  # [n, num_thetas]
            X_prime_proj = torch.matmul(X_t, thetas_stack.T)  # [m, num_thetas]
            
            # Compute SWD
            term1 = 0
            for k in range(len(thetas)):
                X_proj_k_sorted = torch.sort(X_proj[:, k])[0]
                X_prime_proj_k_sorted = torch.sort(X_prime_proj[:, k])[0]
                term1 += torch.sum((X_proj_k_sorted - X_prime_proj_k_sorted) ** 2)
            
            term1 = term1 / len(thetas)
            
            # Compute gradient
            grad = torch.autograd.grad(term1, X_temp, create_graph=False, retain_graph=False)[0]
            return grad[:, self.explainer.explain_indices]
    
    def _sample_in_cone(self, guide_direction, cone_angle, num_features):
        """Sample directions within a cone around the guide direction"""
        # Normalize guide direction
        guide_direction = guide_direction / (torch.norm(guide_direction, dim=1, keepdim=True) + 1e-8)
        
        batch_size = guide_direction.shape[0]
        directions = []
        
        for i in range(batch_size):
            guide = guide_direction[i].cpu().numpy()
            
            # Generate random direction
            random_direction = self._rng.randn(num_features)
            random_direction = random_direction / np.linalg.norm(random_direction)
            
            # Calculate angle with guide direction
            cos_angle = np.dot(guide, random_direction)
            current_angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            # If angle exceeds cone range, project to cone boundary
            if current_angle > cone_angle:
                # Calculate projection direction
                # Use spherical linear interpolation to adjust direction to cone boundary
                target_cos = np.cos(cone_angle)
                
                # Find component perpendicular to guide_direction
                parallel_component = np.dot(random_direction, guide) * guide
                perpendicular_component = random_direction - parallel_component
                
                if np.linalg.norm(perpendicular_component) > 1e-8:
                    perpendicular_component = perpendicular_component / np.linalg.norm(perpendicular_component)
                    
                    # Reconstruct direction on cone boundary
                    random_direction = target_cos * guide + np.sin(cone_angle) * perpendicular_component
            
            directions.append(torch.from_numpy(random_direction).float().to(self.explainer.device))
        
        return torch.stack(directions)
    
    def _apply_gradient_guidance_to_continuous_features(self, cand, eta, ref_idx, idx):
        """
        Apply gradient guidance to continuous features.
        
        Args:
            cand: Candidate solution tensor
            eta: Current eta value
            ref_idx: Reference index for X_prime
            idx: Current candidate index
            
        Returns:
            bool: True if gradient guidance was successfully applied, False otherwise
        """
        if not self.use_gradient_guidance:
            return False
            
        try:
            grad_term1 = self._compute_term1_gradient(cand, eta)
            guide_direction = -grad_term1  # Negative gradient direction
            
            # Sample directions within cone region
            # Only use gradient direction for continuous features
            continuous_guide_direction = guide_direction[:, self.explainer.continuous_indices]
            cone_directions = self._sample_in_cone(continuous_guide_direction, self.cone_angle, len(self.explainer.continuous_indices))
            
            # Apply guided sampling
            if cone_directions.shape[0] > idx:
                for feat_idx, idx_feat in enumerate(self.explainer.continuous_indices):
                    if feat_idx < cone_directions.shape[1]:
                        min_val = self.explainer.X_prime[:, idx_feat].min()
                        max_val = self.explainer.X_prime[:, idx_feat].max()
                        
                        # Use cone sampling direction
                        direction = cone_directions[idx, feat_idx]
                        step_size = torch.rand(1, generator=self._torch_rng, device=self.explainer.device) * 0.1
                        
                        # Calculate new value
                        current_val = self.explainer.X_prime[ref_idx, idx_feat]
                        perturbation = direction * step_size * (max_val - min_val)
                        sampled_val = torch.clamp(current_val + perturbation, min_val, max_val)
                        
                        cand[idx, idx_feat] = sampled_val
                        # cand[idx, idx_feat] = (1 - eta) * self.explainer.X_prime[ref_idx, idx_feat] + eta * sampled_val
                    else:
                        # Fallback to random sampling for extra features
                        self._apply_random_sampling_to_feature(cand, eta, ref_idx, idx, idx_feat)
            else:
                # Fallback to random sampling if cone directions not available
                for idx_feat in self.explainer.continuous_indices:
                    self._apply_random_sampling_to_feature(cand, eta, ref_idx, idx, idx_feat)
            
            # Print success message when gradient guidance is used
            # print(f"[{self.__class__.__name__}] Successfully used gradient guidance with cone angle {self.cone_angle:.3f} rad ({math.degrees(self.cone_angle):.1f}°)")
            return True
            
        except Exception as e:
            # Fallback to random sampling if gradient computation fails
            print(f"[{self.__class__.__name__}] Gradient computation failed, falling back to random sampling: {str(e)}")
            return False
    
    def _apply_random_sampling_to_feature(self, cand, eta, ref_idx, idx, idx_feat):
        """Apply random sampling to a single continuous feature"""
        min_val = self.explainer.X_prime[:, idx_feat].min()
        max_val = self.explainer.X_prime[:, idx_feat].max()
        rand_val = torch.rand(1, generator=self._torch_rng, device=self.explainer.device)
        sampled_val = min_val + rand_val * (max_val - min_val)
        cand[idx, idx_feat] = (1 - eta) * self.explainer.X_prime[ref_idx, idx_feat] + eta * sampled_val
    
    def _apply_gradient_guidance_to_categorical_features(self, cand, eta, ref_idx, idx):
        """
        Apply gradient guidance to categorical features using gradient magnitude as sampling weights.
        """
        if not self.use_gradient_guidance:
            return False
            
        try:
            grad_term1 = self._compute_term1_gradient(cand, eta)
            
            for feat_idx_position, idx_feat in enumerate(self.explainer.categorical_indices):
                if idx_feat in self.explainer.explain_indices:
                    explain_position = self.explainer.explain_indices.index(idx_feat)
                    grad_magnitude = torch.abs(grad_term1[idx, explain_position])
                    
                    unique_vals = torch.unique(self.explainer.X_prime[:, idx_feat])
                    
                    if len(unique_vals) > 1:
                        # Use gradient magnitude to create sampling probabilities
                        probs = torch.ones(len(unique_vals), device=self.explainer.device)
                        probs = probs * (1.0 + float(grad_magnitude))  # Higher gradient = more exploration
                        probs = probs / probs.sum()
                        
                        sampled_idx = torch.multinomial(probs, 1, generator=self._torch_rng).item()
                        sampled_val = unique_vals[sampled_idx]
                    else:
                        sampled_val = unique_vals[0]
                    

                    cand[idx, idx_feat] = sampled_val
                    # cand[idx, idx_feat] = (1 - eta) * self.explainer.X_prime[ref_idx, idx_feat] + eta * sampled_val
            
            return True
            
        except Exception as e:
            print(f"[{self.__class__.__name__}] Categorical gradient guidance failed: {str(e)}")
            return False