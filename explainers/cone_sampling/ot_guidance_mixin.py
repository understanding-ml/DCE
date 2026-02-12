import torch
import numpy as np
import math

class OTGuidanceMixin:
    def __init__(self, explainer, cone_angle=math.pi/4, random_state=None, use_cone_sampling_categorical=True, use_cone_sampling_continuous=True):
        self.explainer = explainer
        self.cone_angle = cone_angle
        self.use_cone_sampling_categorical = use_cone_sampling_categorical
        self.use_cone_sampling_continuous = use_cone_sampling_continuous
        self._rng = np.random.RandomState(random_state)
        self._torch_rng = torch.Generator(device=explainer.device).manual_seed(random_state or 0)

    def _compute_term1_ot_direction(self, X, eta):
        """Compute gradient of term1 (SWD) with respect to input X."""
        with torch.enable_grad():
            X_temp = X.clone().detach().requires_grad_(True)

            X_s = X_temp[:, self.explainer.explain_indices] * self.explainer.costs_vector_reshaped
            X_t = self.explainer.X_prime[:, self.explainer.explain_indices] * self.explainer.costs_vector_reshaped

            num_features = X_s.shape[1]
            thetas = []
            for theta in self.explainer.swd.thetas:
                theta_tensor = torch.from_numpy(theta).float().to(self.explainer.device)
                if theta_tensor.shape[0] != num_features:
                    # Pad or truncate to match feature count
                    if theta_tensor.shape[0] < num_features:
                        pad = torch.zeros(num_features - theta_tensor.shape[0], device=self.explainer.device)
                        theta_tensor = torch.cat([theta_tensor, pad])
                    else:
                        theta_tensor = theta_tensor[:num_features]
                thetas.append(theta_tensor)

            thetas_stack = torch.stack(thetas)  # [n_proj, num_features]

            X_proj = X_s @ thetas_stack.T  # [n, n_proj]
            X_prime_proj = X_t @ thetas_stack.T  # [m, n_proj]

            term1 = 0
            for k in range(len(thetas)):
                X_proj_k = torch.sort(X_proj[:, k])[0]
                X_prime_proj_k = torch.sort(X_prime_proj[:, k])[0]
                term1 += torch.sum((X_proj_k - X_prime_proj_k) ** 2)
            term1 = term1 / len(thetas)

            grad = torch.autograd.grad(term1, X_temp, create_graph=False, retain_graph=False)[0]
            return grad  # shape [n, d]

    def _sample_in_cone(self, guide_direction, cone_angle, num_features):
        """Sample direction vectors inside a cone around given guide directions."""
        guide_direction = guide_direction / (torch.norm(guide_direction, dim=1, keepdim=True) + 1e-8)
        batch_size = guide_direction.shape[0]
        directions = []

        for i in range(batch_size):
            guide = guide_direction[i].cpu().numpy()
            rand_dir = self._rng.randn(num_features)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)

            cos_angle = np.dot(guide, rand_dir)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            if angle > cone_angle:
                target_cos = np.cos(cone_angle)
                parallel = np.dot(rand_dir, guide) * guide
                perp = rand_dir - parallel
                if np.linalg.norm(perp) > 1e-8:
                    perp = perp / np.linalg.norm(perp)
                    rand_dir = target_cos * guide + np.sin(cone_angle) * perp

            directions.append(torch.from_numpy(rand_dir).float().to(self.explainer.device))

        return torch.stack(directions)  # shape [batch_size, num_features]


class CategoricalEmbedding:
    """Simple categorical embedding for ot-guided sampling"""
    
    def __init__(self, unique_values, embed_dim=None, random_state=None):
        self.unique_values = unique_values
        num_categories = len(unique_values)
        
        # Auto-calculate embedding dimension if not provided
        if embed_dim is None:
            self.embed_dim = self._calculate_optimal_dim(num_categories)
        else:
            self.embed_dim = embed_dim
            
        self.value_to_idx = {val.item(): i for i, val in enumerate(unique_values)}
        
        # Use random_state for reproducible embedding initialization
        if random_state is not None:
            torch.manual_seed(random_state)
        
        # Simple random initialization of embeddings
        self.embeddings = torch.randn(len(unique_values), self.embed_dim)
        
        # Reset random state after embedding initialization
        if random_state is not None:
            torch.manual_seed(random_state)
    
    def _calculate_optimal_dim(self, num_categories):
        """Calculate optimal embedding dimension based on number of categories"""
        if num_categories <= 2:
            return 4
        elif num_categories <= 5:
            return 6
        elif num_categories <= 10:
            return 8
        elif num_categories <= 20:
            return 12
        else:
            return 16
    
    def encode(self, categorical_val):
        """Convert categorical value to embedding vector"""
        val = categorical_val.item()
        
        # If exact value exists, use it
        if val in self.value_to_idx:
            idx = self.value_to_idx[val]
        else:
            # Find closest value in unique_vals
            distances = torch.abs(self.unique_values - val)
            closest_idx = torch.argmin(distances).item()
            idx = closest_idx
            
        return self.embeddings[idx]
    
    def decode(self, embedding_vec, temperature=1.0, random_generator=None):
        """Find categorical value using temperature-based sampling"""
        distances = torch.norm(self.embeddings - embedding_vec, dim=1)
        
        if temperature > 0 and random_generator is not None:
            # Convert distances to similarities (negative distances)
            similarities = -distances / temperature
            # Use softmax to create probability distribution
            probs = torch.softmax(similarities, dim=0)
            # Sample according to probabilities
            sampled_idx = torch.multinomial(probs, 1, generator=random_generator)
            return self.unique_values[sampled_idx]
        else:
            closest_idx = torch.argmin(distances)
            return self.unique_values[closest_idx]
    
    def map_to_nearest_category(self, continuous_val):
        """Map a continuous value to the nearest valid categorical value"""
        distances = torch.abs(self.unique_values - continuous_val)
        closest_idx = torch.argmin(distances)
        return self.unique_values[closest_idx]