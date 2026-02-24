import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, adjusted_rand_score, normalized_mutual_info_score
import scipy.stats as stats
import matplotlib
import matplotlib as mpl
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# CatBoost import
from catboost import CatBoostRegressor

np.random.seed(0)
cmap = 'viridis'
# ----------------------------
# 1. Generate In-Distribution Data (10 domains, each corresponding to a Gaussian)
# ----------------------------

num_samples_per_domain = 500
num_domains = 10
total_samples = num_samples_per_domain * num_domains

# Create domain labels
domains = np.repeat([str(i) for i in range(num_domains)], num_samples_per_domain)

# Randomly scatter means instead of using linspace
means_U = np.random.uniform(-50, 50, num_domains)
means_U.sort()

covs_U = np.random.uniform(0.5, 0.6, num_domains)

U = np.zeros(total_samples)
for i in range(total_samples):
    domain_idx = int(domains[i])
    U[i] = np.random.normal(loc=means_U[domain_idx], scale=covs_U[domain_idx])

X1 = U + np.random.normal(loc=0, scale=0.5, size=total_samples)
Y  = X1 - 3 * U + np.random.normal(loc=0, scale=0.5, size=total_samples)
S_U = 0.6 * U + np.random.normal(loc=0, scale=0.3, size=total_samples)

df = pd.DataFrame({
    'Domain': domains,
    'U':       U,
    'X':       X1,
    'S':       S_U,
    'Y':       Y
})

# ----------------------------
# 2. Generate Test Data
# ----------------------------
shift_factor = 0.2
means_U_test = means_U + shift_factor

U_test = np.zeros(total_samples)
for i in range(total_samples):
    domain_idx = int(domains[i])
    U_test[i] = np.random.normal(loc=means_U_test[domain_idx], scale=covs_U[domain_idx])

X_test_vals = U_test + np.random.normal(loc=0, scale=0.5, size=total_samples)
Y_test = X_test_vals - 3 * U_test + np.random.normal(loc=0, scale=0.5, size=total_samples)
S_U_test = 0.6 * U_test + np.random.normal(loc=0, scale=0.3, size=total_samples)

df_test = pd.DataFrame({
    'Domain': domains,
    'U':       U_test,
    'X':       X_test_vals,
    'S':       S_U_test,
    'Y':       Y_test
})
# ----------------------------
# Prepare train/test matrices
# ----------------------------
X_train = df[['X']].values
X_test  = df_test[['X']].values

def get_leaf_indices_catboost(model, X):
    return np.array(model.calc_leaf_indexes(X), dtype=int)

# ----------------------------
# Full CatBoost evaluation (with ARI & NMI like original)
# ----------------------------
def plot_all_embeddings_catboost(estimators, max_depths, 
                                 X_train, y_train, U_train, 
                                 X_test, y_test, U_test, save_prefix="results/"):

    tsne_results = {}
    pca_results = {}
    U_joint_dict = {}
    mse_dict = {}
    ari_dict = {}
    nmi_dict = {}

    le = LabelEncoder()
    true_labels = le.fit_transform(df['Domain'])

    for n_est in estimators:
        for depth in max_depths:
            model = CatBoostRegressor(iterations=n_est,
                                      depth=depth,
                                      random_seed=42,
                                      verbose=0)
            model.fit(X_train, y_train)

            pred_train = model.predict(X_train)
            pred_test  = model.predict(X_test)
            mse_id = mean_squared_error(y_train, pred_train)
            mse_ood = mean_squared_error(y_test, pred_test)

            leaf_indices_train = get_leaf_indices_catboost(model, X_train)
            leaf_indices_test  = get_leaf_indices_catboost(model, X_test)

            # --- One-hot leaf embedding ---
            n_train, n_trees = leaf_indices_train.shape
            max_leaf_idx = int(max(leaf_indices_train.max(), leaf_indices_test.max()) + 1)

            binary_embedding_train = np.zeros((n_train, max_leaf_idx * n_trees), dtype=np.float32)
            for i in range(n_train):
                for j in range(n_trees):
                    leaf_idx = int(leaf_indices_train[i, j])
                    binary_embedding_train[i, j * max_leaf_idx + leaf_idx] = 1

            # PCA → KMeans clustering
            pca = PCA(n_components=min(10, binary_embedding_train.shape[1]))
            dense_embeddings_train = pca.fit_transform(binary_embedding_train)

            kmeans = KMeans(n_clusters=num_domains, random_state=42)
            cluster_labels = kmeans.fit_predict(dense_embeddings_train)

            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)

            key = f"n{n_est}_d{depth}"
            ari_dict[key] = ari_score
            nmi_dict[key] = nmi_score

            leaf_joint = np.vstack([leaf_indices_train, leaf_indices_test])
            U_joint = np.concatenate([U_train, U_test])
            n_train_total = leaf_indices_train.shape[0]

            tsne_emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(leaf_joint)
            pca_emb  = PCA(n_components=2).fit_transform(leaf_joint)

            tsne_results[key] = (tsne_emb, n_train_total)
            pca_results[key]  = (pca_emb, n_train_total)
            U_joint_dict[key] = (U_train, U_test, U_joint)
            mse_dict[key] = (mse_id, mse_ood)

    # --- Print metrics ---
    print("ARI Scores (CatBoost):")
    for k, v in ari_dict.items():
        print(f"{k}: {v:.3f}")
    print("\nNMI Scores (CatBoost):")
    for k, v in nmi_dict.items():
        print(f"{k}: {v:.3f}")

    # --- t-SNE plots ---
    fig, axes = plt.subplots(len(estimators), len(max_depths), figsize=(4*len(max_depths), 4*len(estimators)))
    if len(estimators) == 1 and len(max_depths) == 1:
        axes = np.array([[axes]])

    for i, n_est in enumerate(estimators):
        for j, depth in enumerate(max_depths):
            ax = axes[i, j]
            key = f"n{n_est}_d{depth}"
            tsne_emb, n_train = tsne_results[key]
            U_train_local, U_test_local, U_joint = U_joint_dict[key]
            mse_id, mse_ood = mse_dict[key]
            ari_score = ari_dict[key]; nmi_score = nmi_dict[key]

            vmin, vmax = U_joint.min(), U_joint.max()
            ax.scatter(tsne_emb[:n_train,0], tsne_emb[:n_train,1],
                       c=U_train_local, cmap=cmap, alpha=0.7, marker='^', s=15,
                       vmin=vmin, vmax=vmax)
            ax.scatter(tsne_emb[n_train:,0], tsne_emb[n_train:,1],
                       c=U_test_local, cmap=cmap, alpha=0.7, marker='v', s=15,
                       vmin=vmin, vmax=vmax)
            ax.set_title(f"CatBoost (#estimators={n_est}, max depth={depth})\nID MSE={mse_id:.3f}, OOD MSE={mse_ood:.3f}\nARI"+r"(U,$\phi(X)$)"+f"={ari_score:.3f}, NMI"+r"(U,$\phi(X)$)"+f"={nmi_score:.3f}",
                         fontsize=9, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=U_joint.min(), vmax=U_joint.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02, label='U value (not used for training)')
    fig.savefig(f"{save_prefix}catboost_tsne.pdf",bbox_inches="tight")
    plt.close(fig)

    # --- PCA plots ---
    fig, axes = plt.subplots(len(estimators), len(max_depths), figsize=(4*len(max_depths), 4*len(estimators)))
    if len(estimators) == 1 and len(max_depths) == 1:
        axes = np.array([[axes]])

    for i, n_est in enumerate(estimators):
        for j, depth in enumerate(max_depths):
            ax = axes[i, j]
            key = f"n{n_est}_d{depth}"
            pca_emb, n_train = pca_results[key]
            U_train_local, U_test_local, U_joint = U_joint_dict[key]
            mse_id, mse_ood = mse_dict[key]
            ari_score = ari_dict[key]; nmi_score = nmi_dict[key]

            vmin, vmax = U_joint.min(), U_joint.max()
            ax.scatter(pca_emb[:n_train,0], pca_emb[:n_train,1],
                       c=U_train_local, cmap=cmap, alpha=0.7, marker='^', s=35,
                       vmin=vmin, vmax=vmax)
            ax.scatter(pca_emb[n_train:,0], pca_emb[n_train:,1],
                       c=U_test_local, cmap=cmap, alpha=0.7, marker='v', s=35,
                       vmin=vmin, vmax=vmax)
            ax.set_title(f"CatBoost (#estimators={n_est}, max depth={depth})\nID MSE={mse_id:.3f}, OOD MSE={mse_ood:.3f}\nARI"+r"(U,$\phi(X)$)"+f"={ari_score:.3f}, NMI"+r"(U,$\phi(X)$)"+f"={nmi_score:.3f}",
                         fontsize=9, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=U_joint.min(), vmax=U_joint.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02, label='U value (hidden to model)')
    fig.savefig(f"{save_prefix}catboost_pca.pdf",bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Run everything
# ----------------------------
estimators = [10,20,30,40]
max_depths = [1,2,3,4,5]

plot_all_embeddings_catboost(estimators, max_depths,
                             X_train, df['Y'].values, df['U'].values,
                             X_test, df_test['Y'].values, df_test['U'].values)
