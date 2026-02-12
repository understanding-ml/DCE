import os
import math
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 100


class CallbackVisualizer:
    def __init__(self, mode="final_only", model=None, data=None, explain_columns=None, y_target=None, max_iter=None, save_dir=None):
        self.mode = mode
        self.model = model
        self.data = data
        self.explain_columns = explain_columns
        self.y_target = y_target
        self.max_iter = max_iter
        self.output_dir = None
        self.parent_save_dir = save_dir  
        
        self.target_name = getattr(data, 'target_name', 'Risk') if data else 'Risk'

    def set_mode(self, mode):
        self.mode = mode

    def __call__(self, explainer, iteration):
        if self.mode == "off":
            return
        elif self.mode == "final_only":
            if self.max_iter is None or iteration not in [0, self.max_iter - 1]:
                return

        if self.output_dir is None:
            self._init_output_dir()

        df = explainer.data.df
        std = explainer.data.std[self.explain_columns]
        mean = explainer.data.mean[self.explain_columns]
        category = explainer.data.categorical_columns
        dtype_dict = df.dtypes.apply(lambda x: x.name).to_dict()

        def recover_types(df_):
            for col in df_.columns:
                if col in dtype_dict and dtype_dict[col].startswith("int"):
                    df_[col] = df_[col].round().astype("int")
            return df_

        X_fact = explainer.X_prime.detach().cpu().numpy() * std.values + mean.values
        X_cfact = explainer.X.detach().cpu().numpy() * std.values + mean.values

        df_factual = recover_types(pd.DataFrame(X_fact, columns=self.explain_columns))
        df_counter = recover_types(pd.DataFrame(X_cfact, columns=self.explain_columns))

        y_factual = explainer.model(explainer.X_prime).detach().cpu().numpy()
        y_counter = explainer.model(explainer.X).detach().cpu().numpy()
        y_target = self.y_target.detach().cpu().numpy() if self.y_target is not None else explainer.y_prime.detach().cpu().numpy()

        df_factual[self.target_name] = y_factual
        df_counter[self.target_name] = y_counter

        all_features = self.explain_columns + [self.target_name]
        num_cols = 3
        num_rows = math.ceil(len(all_features) / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
        axes = axes.flatten()

        for i, feature in enumerate(all_features):
            ax = axes[i]
            fx, cx = df_factual[feature], df_counter[feature]

            if feature in category:
                bins = np.arange(fx.min(), fx.max() + 2)
                ax.bar(bins[:-1] - 0.175, np.histogram(fx, bins=bins)[0], width=0.35, label="Fact.", color="blue")
                ax.bar(bins[:-1] + 0.175, np.histogram(cx, bins=bins)[0], width=0.35, label="C.Fact.", color="red")
            elif feature == self.target_name:
                ax.plot(np.sort(fx), np.linspace(0, 1, len(fx)), label="Factual", color="blue")
                ax.plot(np.sort(cx), np.linspace(0, 1, len(cx)), label="Counterfactual", color="red")
                if y_target is not None and len(y_target) > 0:
                    target_flat = y_target.flatten() if y_target.ndim > 1 else y_target
                    if len(target_flat) == len(fx):
                        ax.plot(np.sort(target_flat), np.linspace(0, 1, len(target_flat)), 
                               label="Target", color="black", linestyle="--")
                    else:
                        ax.plot(np.sort(target_flat), np.linspace(0, 1, len(target_flat)), 
                               label="Target", color="black", linestyle="--")
            else:
                ax.plot(np.sort(fx), np.linspace(0, 1, len(fx)), label="Factual", color="blue")
                ax.plot(np.sort(cx), np.linspace(0, 1, len(cx)), label="Counterfactual", color="red")

            ax.set_title(feature)
            ax.set_xlabel(feature)
            ax.set_ylabel("Quantiles")
            ax.legend()

        for j in range(len(all_features), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        fig.suptitle(f"Iteration = {iteration}", fontsize=20)

        fig_path = os.path.join(self.output_dir, f"iteration_{iteration}.png")
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        if iteration == self.max_iter - 1:
            time.sleep(0.2)
            self._combine_images()

    def _init_output_dir(self):
        if self.parent_save_dir:
            self.output_dir = os.path.join(self.parent_save_dir, "visualizations")
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            dataset_name = getattr(self.data, "name", "default")
            self.output_dir = os.path.join("Results", "visualizations", f"{dataset_name}_Result_{timestamp}")
            os.makedirs(self.output_dir, exist_ok=True)

    def _combine_images(self):
        img0_path = os.path.join(self.output_dir, "iteration_0.png")
        imgN_path = os.path.join(self.output_dir, f"iteration_{self.max_iter - 1}.png")

        for _ in range(10):
            if os.path.exists(img0_path) and os.path.exists(imgN_path):
                break
            time.sleep(0.2)
        else:
            print("❌ Combined image failed: iteration_0 or final image not found.")
            return

        img0 = Image.open(img0_path)
        imgN = Image.open(imgN_path)
        combined = Image.new("RGB", (img0.width + imgN.width, max(img0.height, imgN.height)))
        combined.paste(img0, (0, 0))
        combined.paste(imgN, (img0.width, 0))

        combined_path = os.path.join(self.output_dir, "combined_final.png")
        combined.save(combined_path)

        plt.figure(figsize=(combined.width / 100, combined.height / 100))
        plt.imshow(np.asarray(combined))
        plt.axis("off")
        plt.title("Iteration 0 (Left) vs Final (Right)", fontsize=16)
        plt.close()
