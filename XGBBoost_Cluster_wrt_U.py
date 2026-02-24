import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, adjusted_rand_score, normalized_mutual_info_score
from xgboost import XGBRegressor
import scipy.stats as stats
import matplotlib
import matplotlib as mpl
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

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
# Sort the means to ensure they're in order for better visualization
means_U.sort()

# Use different covariances for each Gaussian
covs_U = np.random.uniform(0.5, 1, num_domains)

U = np.zeros(total_samples)
for i in range(total_samples):
    domain_idx = int(domains[i])
    U[i] = np.random.normal(loc=means_U[domain_idx], scale=covs_U[domain_idx])

# Structural equations
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
# 2. Generate Test Data with slightly shifted means
# ----------------------------
shift_factor = 0.2  # Amount to shift the means for test data
means_U_test = means_U + shift_factor

U_test = np.zeros(total_samples)
for i in range(total_samples):
    domain_idx = int(domains[i])
    U_test[i] = np.random.normal(loc=means_U_test[domain_idx], scale=covs_U[domain_idx])

# Structural equations remain the same
X_test = U_test + np.random.normal(loc=0, scale=0.5, size=total_samples)
Y_test = X_test - 3 * U_test + np.random.normal(loc=0, scale=0.5, size=total_samples)
S_U_test = 0.6 * U_test + np.random.normal(loc=0, scale=0.3, size=total_samples)

df_test = pd.DataFrame({
    'Domain': domains,
    'U':       U_test,
    'X':       X_test,
    'S':       S_U_test,
    'Y':       Y_test
})

def design_matrix_no(X):
    return np.column_stack((np.ones(len(X)), X)).astype(np.float64)

def design_matrix_with(X, D):
    return np.column_stack((np.ones(len(X)), X, D)).astype(np.float64)


X_train = design_matrix_no(df['X'].values)
X_test  = design_matrix_no(df_test['X'].values)

# ----------------------------
# 7. Training & Prediction with ARI/NMI metrics
# ----------------------------

def plot_all_embeddings(estimators, max_depths, 
                        X_train, y_train, U_train, 
                        X_test, y_test, U_test, save_prefix="results/"):

    # Prepare storage for embeddings + errors
    tsne_results = {}
    pca_results = {}
    U_joint_dict = {}
    mse_dict = {}
    ari_dict = {}  # Store ARI scores
    nmi_dict = {}  # Store NMI scores

    for n_est in estimators:
        for depth in max_depths:
            model = XGBRegressor(random_state=42, n_estimators=n_est, max_depth=depth)
            model.fit(X_train, y_train)

            # Predictions for MSE
            pred_train = model.predict(X_train)
            pred_test  = model.predict(X_test)
            mse_id = mean_squared_error(y_train, pred_train)
            mse_ood = mean_squared_error(y_test, pred_test)

            leaf_indices_train = model.apply(X_train)
            leaf_indices_test = model.apply(X_test)

            # Create binary embedding from leaf indices
            n_train = int(leaf_indices_train.shape[0])  # Convert to int
            n_test = int(leaf_indices_test.shape[0])    # Convert to int
            n_trees = int(leaf_indices_train.shape[1])  # Convert to int
            
            # Find the maximum leaf index across all trees
            max_leaf_idx = int(max(leaf_indices_train.max(), leaf_indices_test.max()) + 1)  # Convert to int
            
            # Create binary embedding matrix
            binary_embedding_train = np.zeros((n_train, max_leaf_idx * n_trees))
            for i in range(n_train):
                for j in range(n_trees):
                    leaf_idx = int(leaf_indices_train[i, j])  # Convert to int
                    binary_embedding_train[i, j * max_leaf_idx + leaf_idx] = 1
            
            # Apply PCA to reduce dimensionality
            pca = PCA(n_components=min(10, binary_embedding_train.shape[1]))  # Ensure n_components <= n_features
            dense_embeddings_train = pca.fit_transform(binary_embedding_train)
            
            # Cluster the embeddings
            kmeans = KMeans(n_clusters=num_domains, random_state=42)
            cluster_labels = kmeans.fit_predict(dense_embeddings_train)
            
            # Encode the true domain labels to integers
            le = LabelEncoder()
            true_labels = le.fit_transform(df['Domain'])
            
            # Calculate ARI and NMI
            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
            
            # Store the metrics
            key = f"n{n_est}_d{depth}"
            ari_dict[key] = ari_score
            nmi_dict[key] = nmi_score

            leaf_joint = np.vstack([leaf_indices_train, leaf_indices_test])
            U_joint = np.concatenate([U_train, U_test])
            n_train_total = leaf_indices_train.shape[0]

            # t-SNE and PCA embeddings
            tsne_emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(leaf_joint)
            pca_emb  = PCA(n_components=2).fit_transform(leaf_joint)

            tsne_results[key] = (tsne_emb, n_train_total)
            pca_results[key]  = (pca_emb, n_train_total)
            U_joint_dict[key] = (U_train, U_test, U_joint)
            mse_dict[key] = (mse_id, mse_ood)

    # Print ARI and NMI results
    print("ARI Scores:")
    for key, score in ari_dict.items():
        print(f"{key}: {score:.3f}")
    
    print("\nNMI Scores:")
    for key, score in nmi_dict.items():
        print(f"{key}: {score:.3f}")

    # ----------------------------
    # Plot all t-SNE results in one figure
    # ----------------------------
    
    fig, axes = plt.subplots(len(estimators), len(max_depths), figsize=(4*len(max_depths), 4*len(estimators)))
    if len(estimators) == 1 and len(max_depths) == 1:
        axes = np.array([[axes]])  # ensure 2D grid

    for i, n_est in enumerate(estimators):
        for j, depth in enumerate(max_depths):
            ax = axes[i, j]
            key = f"n{n_est}_d{depth}"
            tsne_emb, n_train = tsne_results[key]
            U_train, U_test, U_joint = U_joint_dict[key]
            mse_id, mse_ood = mse_dict[key]
            ari_score = ari_dict[key]
            nmi_score = nmi_dict[key]

            vmin, vmax = U_joint.min(), U_joint.max()
            ax.scatter(tsne_emb[:n_train,0], tsne_emb[:n_train,1],
                       c=U_train, cmap=cmap, alpha=0.7, marker='^', s=15,
                       vmin=vmin, vmax=vmax, label="Train")
            ax.scatter(tsne_emb[n_train:,0], tsne_emb[n_train:,1],
                       c=U_test, cmap=cmap, alpha=0.7, marker='v', s=15,
                       vmin=vmin, vmax=vmax, label="Test")
            ax.set_title(f"t-SNE (n={n_est}, d={depth})\nID MSE={mse_id:.3f}, OOD MSE={mse_ood:.3f}\nARI={ari_score:.3f}, NMI={nmi_score:.3f}", 
                         fontsize=10, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Add colorbar legend for U values
    norm = mpl.colors.Normalize(vmin=U_joint.min(), vmax=U_joint.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('U value', fontsize=20)
    fig.savefig(f"{save_prefix}xgb_tsne.pdf")
    plt.close(fig)

    # ----------------------------
    # Plot all PCA results in one figure
    # ----------------------------
    fig, axes = plt.subplots(len(estimators), len(max_depths), figsize=(4*len(max_depths), 4*len(estimators)))
    if len(estimators) == 1 and len(max_depths) == 1:
        axes = np.array([[axes]])

    for i, n_est in enumerate(estimators):
        for j, depth in enumerate(max_depths):
            ax = axes[i, j]
            key = f"n{n_est}_d{depth}"
            pca_emb, n_train = pca_results[key]
            U_train, U_test, U_joint = U_joint_dict[key]
            mse_id, mse_ood = mse_dict[key]
            ari_score = ari_dict[key]
            nmi_score = nmi_dict[key]

            vmin, vmax = U_joint.min(), U_joint.max()
            ax.scatter(pca_emb[:n_train,0], pca_emb[:n_train,1],
                       c=U_train, cmap=cmap, alpha=0.7, marker='^', s=35,
                       vmin=vmin, vmax=vmax, label="Train")
            ax.scatter(pca_emb[n_train:,0], pca_emb[n_train:,1],
                       c=U_test, cmap=cmap, alpha=0.7, marker='v', s=35,
                       vmin=vmin, vmax=vmax, label="Test")
            ax.set_title(f"PCA (n={n_est}, d={depth})\n ID MSE ={mse_id:.3f}, OOD MSE={mse_ood:.3f}\nARI={ari_score:.3f}, NMI={nmi_score:.3f}", 
                         fontsize=10, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    norm = mpl.colors.Normalize(vmin=U_joint.min(), vmax=U_joint.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('U value', fontsize=20)
    fig.savefig(f"{save_prefix}xgb_pca.pdf")

    plt.close(fig)

# ----------------------------
# Run everything
# ----------------------------
estimators = [50]
max_depths = [10]

plot_all_embeddings(estimators, max_depths,
                    df[['X']], df['Y'], df['U'],
                    df_test[['X']], df_test['Y'], df_test['U'])