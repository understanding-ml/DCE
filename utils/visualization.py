import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_quantile(factual, counterfactual, column_name):
    quantiles_factual = factual[column_name].quantile(np.linspace(0, 1, 100))
    quantiles_counterfactual = counterfactual[column_name].quantile(
        np.linspace(0, 1, 100)
    )

    # Plot quantiles
    plt.figure(figsize=(8, 6))
    plt.plot(quantiles_factual.values, np.linspace(0, 1, 100), label="Factual")
    plt.plot(
        quantiles_counterfactual.values, np.linspace(0, 1, 100), label="Counterfactual"
    )
    plt.xlabel("Feature Values")
    plt.ylabel("Quantiles")
    plt.title(f"{column_name}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_box_whisker(factual, counterfactual, column_name):
    # Combine the data into a single DataFrame for plotting
    data_to_plot = pd.DataFrame(
        {"Factual": factual[column_name], "Counterfactual": counterfactual[column_name]}
    )

    # Creating the boxplot
    plt.figure(figsize=(10, 6))
    data_to_plot.boxplot(column=["Factual", "Counterfactual"])
    plt.title("Box-and-Whisker Plot for Factual and Counterfactual")
    plt.ylabel("Values")
    plt.grid(True)
    plt.show()


def category_box_plot(df, x, y, hue, title):
    plt.figure(figsize=(14, 12))

    plt.subplot(212)
    g2 = sns.boxplot(x=x, y=y, data=df.sort_values(by=x), palette="hls", hue=hue)
    g2.set_xlabel(x, fontsize=12)
    g2.set_ylabel(y, fontsize=12)
    g2.set_title(title, fontsize=15)

    plt.subplots_adjust(hspace=0.6, top=0.8)

    plt.show()


def hist_plot(df, x, hue, title):
    plt.figure(figsize=(14, 12))

    plt.subplot(221)
    g = sns.countplot(x=x, data=df.sort_values(by=x), palette="hls", hue=hue)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set_xlabel(x, fontsize=12)
    g.set_ylabel("Count", fontsize=12)
    g.set_title(title, fontsize=20)
