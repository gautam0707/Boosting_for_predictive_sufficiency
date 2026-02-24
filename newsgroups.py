"""
Hidden Confounding Experiment — 20 Newsgroups (NLP)
Updated plotting to follow the provided reference layout:
- Distribution Shift figure (density, ECDF, Q-Q)
- PCA visualizations colored by sampled confounder values (train/test)
- Performance / quantile analysis & common-support metrics

Saves plots under `results/`.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ks_2samp, probplot
from statsmodels.distributions.empirical_distribution import ECDF
import xgboost as xgb

warnings.filterwarnings("ignore")
os.makedirs("newsgroups", exist_ok=True)
np.random.seed(42)

print("1. Loading 20 Newsgroups dataset...")
newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
X_text = pd.Series(newsgroups.data).fillna("")   # ensure no NaNs
y = pd.Series(newsgroups.target)
print(f"Dataset size: {len(X_text)} samples, {len(np.unique(y))} classes")


# ---------------- confounder definitions (text-based proxies) ----------------
def comb_textlen_topicword(X, y):
    has_topic = X.str.contains(r"\b(computer|software|windows|file|program)\b", case=False, regex=True).astype(int)
    text_len = X.str.len().fillna(0)
    return np.log1p(text_len) * has_topic + 1.5 * (y.astype(float))


def comb_avg_wordlen_links(X, y):
    words = X.fillna("").str.split()
    avg_wlen = words.map(lambda w: np.mean([len(tok) for tok in w]) if len(w) > 0 else 0).fillna(0)
    has_link = X.str.contains(r"https?://|www\.", case=False, regex=True).astype(int)
    return avg_wlen * (1 + has_link) + 1.2 * (y.astype(float))


combinations = {
    "Comb1_TextLen_TopicWord": {
        "function": comb_textlen_topicword,
        "description": "log(text_length) * topic_word_presence + 1.5 * label_index",
    },
    "Comb2_AvgWordLen_LinkPresence": {
        "function": comb_avg_wordlen_links,
        "description": "avg_word_length * (1+link_presence) + 1.2 * label_index",
    },
}


# ---------------- splitting helpers (common support + frequency shift) ----------------
def create_frequency_shift(combination_values, train_ratio=0.7, n_quantiles=5):
    quantiles = pd.qcut(combination_values, q=n_quantiles, labels=False, duplicates="drop")
    train_indices = []
    test_indices = []
    for q in range(n_quantiles):
        q_idx = np.where(quantiles == q)[0]
        if len(q_idx) == 0:
            continue
        np.random.shuffle(q_idx)
        # balanced-ish train
        n_train_per_q = max(1, int(len(q_idx) * train_ratio / n_quantiles * 2))
        train_indices.extend(q_idx[:n_train_per_q])
        # imbalanced test: more from higher quantiles
        n_test_per_q = int(len(q_idx) * (1 - train_ratio) * (q + 1) / n_quantiles)
        test_indices.extend(q_idx[n_train_per_q : n_train_per_q + n_test_per_q])
    train_mask = np.zeros(len(combination_values), dtype=bool)
    test_mask = np.zeros(len(combination_values), dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    return train_mask, test_mask


# ---------------- plotting helper modeled on reference ----------------
def create_common_support_figures(comb_name, description, comb_train, comb_test,
                                 pca_train, pca_test, tsne_train, tsne_test, acc, f1,
                                 comb_train_sampled, comb_test_sampled,
                                 common_support, ks_stat, ks_p, y_test, y_pred):
    """Create figures emphasizing common support assumption (follows provided reference layout)"""

    # Figure 1: Distribution Shift with Common Support Evidence
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(comb_train, bins=30, alpha=0.5, density=True, label="Train", color="blue", edgecolor="black")
    plt.hist(comb_test, bins=30, alpha=0.5, density=True, label="Test", color="red", edgecolor="black")
    support_min = max(comb_train.min(), comb_test.min())
    support_max = min(comb_train.max(), comb_test.max())
    if support_min <= support_max:
        plt.axvspan(support_min, support_max, alpha=0.2, color="green", label="Common Support")
    plt.title(f"Distribution Shift with Common Support\n{description}", fontsize=12, fontweight="bold")
    plt.xlabel("Combination Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    ecdf_train = ECDF(comb_train)
    ecdf_test = ECDF(comb_test)
    x_vals = np.linspace(min(comb_train.min(), comb_test.min()), max(comb_train.max(), comb_test.max()), 200)
    plt.plot(x_vals, ecdf_train(x_vals), label="Train ECDF", linewidth=2)
    plt.plot(x_vals, ecdf_test(x_vals), label="Test ECDF", linewidth=2)
    plt.title(f"Empirical CDF Comparison\n(KS statistic: {ks_stat:.3f})", fontsize=12, fontweight="bold")
    plt.xlabel("Combination Value")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    qq_train = probplot(comb_train, dist="norm")
    qq_test = probplot(comb_test, dist="norm")
    plt.scatter(qq_train[0][0], qq_train[0][1], alpha=0.5, label="Train", s=10)
    plt.scatter(qq_test[0][0], qq_test[0][1], alpha=0.5, label="Test", s=10)
    # QQ line from train
    plt.plot(qq_train[0][0], qq_train[1][0] + qq_train[1][1] * qq_train[0][0], "r--", alpha=0.8)
    plt.title("Q-Q Plot: Distribution Comparison", fontsize=12, fontweight="bold")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig(f"results/{comb_name}_01_common_support.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2: PCA Visualizations with Common Support Focus
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    scatter = plt.scatter(pca_train[:, 0], pca_train[:, 1], c=comb_train_sampled,
                          cmap="BuGn", alpha=0.7, s=30)
    plt.title("Train Set PCA", fontsize=12, fontweight="bold")
    plt.xlabel("PC1", fontweight="bold")
    plt.ylabel("PC2", fontweight="bold")
    plt.colorbar(scatter, label="Hidden Confounder Value")

    plt.subplot(1, 3, 2)
    scatter = plt.scatter(pca_test[:, 0], pca_test[:, 1], c=comb_test_sampled,
                          cmap="BuGn", alpha=0.7, s=30)
    plt.title("Test Set PCA", fontsize=12, fontweight="bold")
    plt.xlabel("PC1", fontweight="bold")
    plt.ylabel("PC2", fontweight="bold")
    plt.colorbar(scatter, label="Hidden Confounder Value")

    plt.subplot(1, 3, 3)
    plt.scatter(pca_train[:, 0], pca_train[:, 1], alpha=0.7, s=30, label="Train", color="red", marker="^")
    plt.scatter(pca_test[:, 0], pca_test[:, 1], alpha=0.7, s=30, label="Test", color="blue", marker="v")
    plt.title("Distribution Shift", fontsize=12, fontweight="bold")
    plt.xlabel("PC1", fontweight="bold")
    plt.ylabel("PC2", fontweight="bold")
    plt.legend(prop={"size": 12})
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/{comb_name}_newsgroups.pdf", bbox_inches="tight")
    plt.close()

    # Figure 3: Performance Analysis with Common Support
    plt.figure(figsize=(12, 10))

    # Combine comb_train and comb_test and compute quantiles
    comb_all = np.concatenate([np.array(comb_train), np.array(comb_test)])
    n_quantiles = 5
    comb_quantiles = pd.qcut(comb_all, q=n_quantiles, labels=False, duplicates="drop")
    train_quantiles = comb_quantiles[: len(comb_train)]
    test_quantiles = comb_quantiles[len(comb_train) :]

    # Compute per-quantile accuracy on test set
    test_idx = np.arange(len(comb_test))
    pred_by_quantile = []
    for q in range(n_quantiles):
        q_mask = test_quantiles == q
        idx_q = test_idx[q_mask]
        if len(idx_q) > 0:
            acc_q = accuracy_score(y_test.iloc[idx_q], y_pred[idx_q])
            pred_by_quantile.append({"quantile": q, "test_samples": len(idx_q), "test_acc": acc_q})
        else:
            pred_by_quantile.append({"quantile": q, "test_samples": 0, "test_acc": np.nan})

    # Distribution by quantile
    plt.subplot(2, 2, 1)
    quantile_counts_train = [sum(train_quantiles == q) for q in range(n_quantiles)]
    quantile_counts_test = [sum(test_quantiles == q) for q in range(n_quantiles)]
    x = np.arange(n_quantiles)
    width = 0.35
    plt.bar(x - width / 2, quantile_counts_train, width, label="Train", alpha=0.7)
    plt.bar(x + width / 2, quantile_counts_test, width, label="Test", alpha=0.7)
    plt.title("Sample Distribution by Quantile", fontsize=12, fontweight="bold")
    plt.xlabel("Quantile")
    plt.ylabel("Number of Samples")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Mean values by quantile (combination value)
    plt.subplot(2, 2, 2)
    train_means = []
    test_means = []
    for q in range(n_quantiles):
        tmask = train_quantiles == q
        temask = test_quantiles == q
        train_means.append(np.array(comb_train)[tmask].mean() if tmask.sum() > 0 else np.nan)
        test_means.append(np.array(comb_test)[temask].mean() if temask.sum() > 0 else np.nan)
    plt.bar(x - width / 2, train_means, width, label="Train Mean", alpha=0.7)
    plt.bar(x + width / 2, test_means, width, label="Test Mean", alpha=0.7)
    plt.title("Mean Combination Value by Quantile", fontsize=12, fontweight="bold")
    plt.xlabel("Quantile")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Common support verification & metrics
    plt.subplot(2, 2, 3)
    support_metrics = {
        "Common Support": common_support,
        "KS Statistic": ks_stat,
        "Train Range": comb_train.max() - comb_train.min(),
        "Test Range": comb_test.max() - comb_test.min(),
        "Overlap %": (min(comb_train.max(), comb_test.max()) - max(comb_train.min(), comb_test.min()))
        / (max(comb_train.max(), comb_test.max()) - min(comb_train.min(), comb_test.min()))
        * 100,
    }
    metrics_names = list(support_metrics.keys())
    metrics_values = list(support_metrics.values())
    colors = ["green" if x is True else "red" if x is False else "blue" for x in metrics_values[:1]] + ["blue"] * (len(metrics_values) - 1)
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    plt.title("Common Support Metrics", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    for bar, value in zip(bars, metrics_values):
        if isinstance(value, bool):
            text_val = "Yes" if value else "No"
        else:
            text_val = f"{value:.2f}"
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, text_val, ha="center", va="bottom", fontsize=9)

    # Summary text (bottom-right)
    plt.subplot(2, 2, 4)
    plt.axis("off")
    summary_text = f"""
    COMMON SUPPORT EXPERIMENT
    Combination: {description}

    Key Assumption:
    - Test values were observed during training
    - Realistic generalization scenario

    Results:
    - Accuracy (test): {acc:.3f}
    - Macro F1 (test): {f1:.3f}
    - Common Support: {'SATISFIED' if common_support else 'VIOLATED'}
    - Distribution Difference (KS): {ks_stat:.3f}

    Training Details:
    - Train samples: {len(comb_train):,}
    - Test samples: {len(comb_test):,}
    - Frequency shift applied
    """
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    # plt.savefig(f"results/{comb_name}_03_common_support_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Created common support figures for {comb_name}")


# ---------------- experiment runner ----------------
def run_experiment(comb_name, comb_func, description, X_text, y):
    print("\n" + "=" * 60)
    print(f"Running experiment for: {comb_name}")
    print(f"Combination: {description}")
    print("=" * 60)

    # Reset indices
    X_reset = X_text.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)

    # Create combination values and standardize
    combination_values = comb_func(X_reset, y_reset)
    combination_values = (combination_values - np.mean(combination_values)) / (np.std(combination_values) + 1e-9)

    # create distribution shift with common support
    print("Creating distribution shift with common support...")
    train_mask, test_mask = create_frequency_shift(combination_values, train_ratio=0.7)

    X_train = X_reset[train_mask].reset_index(drop=True)
    y_train = y_reset[train_mask].reset_index(drop=True)
    X_test = X_reset[test_mask].reset_index(drop=True)
    y_test = y_reset[test_mask].reset_index(drop=True)
    comb_train = pd.Series(combination_values[train_mask]).reset_index(drop=True)
    comb_test = pd.Series(combination_values[test_mask]).reset_index(drop=True)

    # verify common support
    train_range = (comb_train.min(), comb_train.max())
    test_range = (comb_test.min(), comb_test.max())
    common_support = (test_range[0] >= train_range[0]) and (test_range[1] <= train_range[1])

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train combination range: [{comb_train.min():.3f}, {comb_train.max():.3f}]")
    print(f"Test combination range: [{comb_test.min():.3f}, {comb_test.max():.3f}]")
    print(f"Common support: {common_support}")
    print(f"Train mean: {comb_train.mean():.3f}, Test mean: {comb_test.mean():.3f}")

    ks_stat, ks_p = ks_2samp(comb_train, comb_test)
    print(f"KS test: statistic={ks_stat:.3f}, p-value={ks_p:.3f}")

    # Vectorize text and reduce with SVD for tree models
    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
    X_all_tfidf = vectorizer.fit_transform(pd.concat([X_train, X_test]))
    n_train = len(X_train)
    X_train_tfidf = X_all_tfidf[:n_train]
    X_test_tfidf = X_all_tfidf[n_train:]

    svd = TruncatedSVD(n_components=min(200, X_train_tfidf.shape[1] - 1 or 1), random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_svd)
    X_test_scaled = scaler.transform(X_test_svd)

    # Train classifier (XGBoost)
    model = xgb.XGBClassifier(n_estimators=150, max_depth=6, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy: {acc:.3f}, Macro F1: {f1:.3f}")

    # Leaf embeddings (convert leaves per tree to numeric embedding)
    leaf_idx_test = model.apply(X_test_scaled)
    leaf_idx_train = model.apply(X_train_scaled)
    if leaf_idx_test.ndim == 1:
        leaf_idx_test = leaf_idx_test.reshape(-1, 1)
    if leaf_idx_train.ndim == 1:
        leaf_idx_train = leaf_idx_train.reshape(-1, 1)

    n_test_samples, n_trees = leaf_idx_test.shape
    n_train_samples = leaf_idx_train.shape[0]

    leaf_embeddings_test = np.zeros((n_test_samples, n_trees))
    leaf_embeddings_train = np.zeros((n_train_samples, n_trees))
    for i in range(n_trees):
        t_leaves = leaf_idx_test[:, i]
        if len(np.unique(t_leaves)) > 1:
            leaf_embeddings_test[:, i] = (t_leaves - t_leaves.min()) / (t_leaves.max() - t_leaves.min())
        tr_leaves = leaf_idx_train[:, i]
        if len(np.unique(tr_leaves)) > 1:
            leaf_embeddings_train[:, i] = (tr_leaves - tr_leaves.min()) / (tr_leaves.max() - tr_leaves.min())

    # PCA on leaf embeddings: fit on test then transform train (as in reference)
    # guard for degenerate shapes (fallback to SVD features)
    try:
        if leaf_embeddings_test.size and leaf_embeddings_test.shape[1] > 1:
            pca = PCA(n_components=2, random_state=42)
            pca_test = pca.fit_transform(leaf_embeddings_test)
            pca_train = pca.transform(leaf_embeddings_train)
        else:
            pca = PCA(n_components=2, random_state=42)
            pca_test = pca.fit_transform(X_test_svd)
            pca_train = pca.transform(X_train_svd)
    except Exception:
        pca = PCA(n_components=2, random_state=42)
        pca_test = pca.fit_transform(X_test_svd)
        pca_train = pca.transform(X_train_svd)

    # TSNE sampling for visualization (if needed)
    if n_test_samples > 1000:
        sample_idx_test = np.random.choice(n_test_samples, 1000, replace=False)
        tsne_test = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(leaf_embeddings_test[sample_idx_test])
        comb_test_sampled = comb_test.iloc[sample_idx_test].values
        pca_test_sampled = pca_test[sample_idx_test]
    else:
        tsne_test = TSNE(n_components=2, random_state=42, perplexity=max(5, min(30, max(1, n_test_samples - 1)))).fit_transform(leaf_embeddings_test)
        comb_test_sampled = comb_test.values
        pca_test_sampled = pca_test

    if n_train_samples > 1000:
        sample_idx_train = np.random.choice(n_train_samples, 1000, replace=False)
        tsne_train = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(leaf_embeddings_train[sample_idx_train])
        comb_train_sampled = comb_train.iloc[sample_idx_train].values
        pca_train_sampled = pca_train[sample_idx_train]
    else:
        tsne_train = TSNE(n_components=2, random_state=42, perplexity=max(5, min(30, max(1, n_train_samples - 1)))).fit_transform(leaf_embeddings_train)
        comb_train_sampled = comb_train.values
        pca_train_sampled = pca_train

    # Create the three figures (Distribution, PCA, Performance) following reference layout
    create_common_support_figures(comb_name, description, comb_train, comb_test,
                                 pca_train_sampled, pca_test_sampled, tsne_train, tsne_test, acc, f1,
                                 comb_train_sampled, comb_test_sampled,
                                 common_support, ks_stat, ks_p, y_test.reset_index(drop=True), y_pred)

    return acc, f1, common_support, ks_stat


# ---------------- run for all combinations ----------------
results = {}
for comb_name, comb_info in combinations.items():
    try:
        acc_score, f1_score_, common_support, ks_stat = run_experiment(
            comb_name,
            comb_info["function"],
            comb_info["description"],
            X_text,
            y,
        )
        results[comb_name] = {
            "accuracy": acc_score,
            "macro_f1": f1_score_,
            "common_support": common_support,
            "ks_statistic": ks_stat,
            "description": comb_info["description"],
        }
    except Exception as e:
        import traceback
        print(f"Error in {comb_name}: {e}")
        traceback.print_exc()
        continue

# Comprehensive summary (similar to reference)
if results:
    print("\n" + "=" * 60)
    print("CREATING COMPREHENSIVE SUMMARY")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    comb_keys = list(results.keys())
    accs = [results[k]["accuracy"] for k in comb_keys]
    supports = [results[k]["common_support"] for k in comb_keys]
    names_short = [k.replace("Comb", "").replace("_", "\n") for k in comb_keys]

    colors = ["green" if s else "red" for s in supports]
    bars = axes[0, 0].bar(names_short, accs, color=colors, alpha=0.7)
    axes[0, 0].set_title("Model Accuracy with Common Support Indicator", fontsize=14, fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].grid(True, alpha=0.3)
    for bar, score, sup in zip(bars, accs, supports):
        h = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{score:.3f}", ha="center", va="bottom", fontweight="bold")
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, -0.05, "✓" if sup else "✗", ha="center", va="top", fontsize=14)

    ks_stats = [results[k]["ks_statistic"] for k in comb_keys]
    axes[0, 1].bar(names_short, ks_stats, color="skyblue", alpha=0.7)
    axes[0, 1].set_title("Distribution Differences (KS Statistics)", fontsize=14, fontweight="bold")
    axes[0, 1].set_ylabel("KS Statistic")
    axes[0, 1].grid(True, alpha=0.3)

    support_rate = sum(supports) / len(supports) * 100
    axes[1, 0].pie([support_rate, 100 - support_rate], labels=["Satisfied", "Violated"], autopct="%1.1f%%", colors=["lightgreen", "lightcoral"])
    axes[1, 0].set_title("Common Support Assumption Success Rate", fontsize=14, fontweight="bold")

    axes[1, 1].axis("off")
    summary_text = f"""
    COMMON SUPPORT EXPERIMENT SUMMARY

    - Total combinations: {len(results)}
    - Common support satisfied: {sum(supports)}/{len(supports)}
    - Average accuracy: {np.mean(accs):.3f}
    - Average KS: {np.mean(ks_stats):.3f}
    """
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, fontsize=11, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    plt.suptitle("Hidden Confounding (20 Newsgroups) — Common Support Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    # plt.savefig("results/00_common_support_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Experiment complete; summary saved to results/00_common_support_summary.png")
else:
    print("No experiments completed successfully.")
