import torch
import numpy as np
import math
from .gradient_guidance_mixin import GradientGuidanceMixin

class ParticleSwarmOptimizationStrategy(GradientGuidanceMixin):
    def __init__(
        self,
        explainer,
        swarm_size: int = 10,
        w: float = 0.5,
        c1: float = 1.0,
        c2: float = 1.0,
        use_gradient_guidance=False,
        cone_angle=math.pi/4,
        random_state: int = None
    ):
        # Initialize base attributes first
        self.explainer = explainer
        
        # Initialize gradient guidance mixin
        GradientGuidanceMixin.__init__(self, explainer, use_gradient_guidance=use_gradient_guidance, cone_angle=cone_angle, random_state=random_state)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state

        # 设置随机种子
        self._rng = np.random.RandomState(random_state)
        self._torch_rng = torch.Generator(device=explainer.device)
        if random_state is not None:
            self._torch_rng.manual_seed(random_state)
            np.random.seed(random_state)

    def generate_new_X(self, eta, num_trials, top_k=1):
        explainer = self.explainer
        best_X = explainer.X.clone()
        best_overall_Q = float("inf")

        with torch.no_grad():
            X_s = explainer.X[:, explainer.explain_indices] * explainer.costs_vector_reshaped
            X_t = explainer.X_prime[:, explainer.explain_indices] * explainer.costs_vector_reshaped
            y = explainer.y.view(-1)
            y_prime = explainer.y_prime.view(-1)
            mu_list = explainer.swd.mu_list
            nu = explainer.wd.nu
            candidate_indices = explainer._get_topk_Q_indices(X_s, X_t, y, y_prime, mu_list, nu, eta, top_k)

            for idx in candidate_indices:
                x0 = explainer.X[idx, explainer.explain_indices].clone()

                # 初始化粒子和速度
                swarm = [
                    x0 + torch.randn(x0.shape, generator=self._torch_rng, device=explainer.device) * 0.01
                    for _ in range(self.swarm_size)
                ]
                velocities = [torch.zeros_like(x0) for _ in range(self.swarm_size)]
                personal_best = [p.clone() for p in swarm]
                personal_best_scores = []

                # 初始评价
                for p in swarm:
                    candidate_row = explainer.X[idx].clone()
                    for j, feat in enumerate(explainer.explain_indices):
                        candidate_row[feat] = p[j]
                    X_temp = explainer._update_row(explainer.X.clone(), idx, candidate_row)
                    y_temp = explainer.model(X_temp)
                    Q = explainer.evaluate_Q(X_temp, y_temp, eta)[0]
                    personal_best_scores.append(Q)

                global_best = personal_best[np.argmin(personal_best_scores)].clone()
                global_best_score = min(personal_best_scores)

                for _ in range(num_trials):
                    for i in range(self.swarm_size):
                        r1 = torch.rand(1, generator=self._torch_rng).item()
                        r2 = torch.rand(1, generator=self._torch_rng).item()

                        # 更新速度和位置
                        velocities[i] = (
                            self.w * velocities[i]
                            + self.c1 * r1 * (personal_best[i] - swarm[i])
                            + self.c2 * r2 * (global_best - swarm[i])
                        )
                        swarm[i] = swarm[i] + velocities[i]
                        
                        # Apply gradient guidance if enabled
                        if self.use_gradient_guidance:
                            candidate_row = explainer.X[idx].clone()
                            for j, feat in enumerate(explainer.explain_indices):
                                candidate_row[feat] = swarm[i][j]
                            X_candidate = explainer._update_row(explainer.X.clone(), idx, candidate_row)
                            ref_idx = 0  # Use first reference point
                            if self._apply_gradient_guidance_to_continuous_features(X_candidate, eta, ref_idx, idx):
                                # Update swarm particle with guided values
                                for j, feat in enumerate(explainer.explain_indices):
                                    if feat in explainer.continuous_indices:
                                        swarm[i][j] = X_candidate[idx, feat]

                        # 计算新位置 Q 值
                        candidate_row = explainer.X[idx].clone()
                        for j, feat in enumerate(explainer.explain_indices):
                            candidate_row[feat] = swarm[i][j]
                        X_temp = explainer._update_row(explainer.X.clone(), idx, candidate_row)
                        y_temp = explainer.model(X_temp)
                        Q = explainer.evaluate_Q(X_temp, y_temp, eta)[0]

                        # 更新个体最优和全局最优
                        if Q < personal_best_scores[i]:
                            personal_best[i] = swarm[i].clone()
                            personal_best_scores[i] = Q
                            if Q < global_best_score:
                                global_best = swarm[i].clone()
                                global_best_score = Q

                if global_best_score < best_overall_Q:
                    best_overall_Q = global_best_score
                    candidate_row = explainer.X[idx].clone()
                    for j, feat in enumerate(explainer.explain_indices):
                        candidate_row[feat] = global_best[j]
                    best_X = explainer._update_row(explainer.X.clone(), idx, candidate_row)

        return best_X
