import numpy as np
import pandas as pd
import math, random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from xgboost import XGBRegressor

# --------------------------
# Settings (tune for speed/accuracy)
# --------------------------
np.random.seed(0); random.seed(0)
num_domains = 10
n_estimators = 30
max_depth = 3
n_list = [50]*10
embedding_methods = ['tSNE', 'PCA']
models = ['X','X,S']
TSNE_MAX = 1000
TSNE_PRE_PCA_DIM = 30
TSNE_ITERS = 300
xgb_params = dict(n_estimators=n_estimators, max_depth=max_depth, random_state=0, verbosity=0)

# --------------------------
# Helpers
# --------------------------
def generate_data(n_per_domain, num_domains=10, shift_w=3, shift_factor=0.2, seed=0):
    rng = np.random.RandomState(seed)
    total = n_per_domain * num_domains
    domain_U = np.repeat(np.arange(num_domains), n_per_domain)
    domain_W = (domain_U + shift_w) % num_domains
    means_U = rng.uniform(-50, 50, num_domains)
    means_W = rng.uniform(-50, 50, num_domains)
    covs_U = rng.uniform(0.5, 0.6, num_domains)
    covs_W = rng.uniform(0.5, 0.6, num_domains)
    U = np.array([rng.normal(loc=means_U[domain_U[i]], scale=covs_U[domain_U[i]]) for i in range(total)])
    W = np.array([rng.normal(loc=means_W[domain_W[i]], scale=covs_W[domain_W[i]]) for i in range(total)])
    X = U + rng.normal(0, 0.5, size=total)
    S = W + rng.normal(0, 0.5, size=total)
    Y = X - 3.0 * U + 2.0 * W + rng.normal(0, 0.5, size=total)
    means_U_test = means_U + shift_factor
    means_W_test = means_W + shift_factor
    U_test = np.array([rng.normal(loc=means_U_test[domain_U[i]], scale=covs_U[domain_U[i]]) for i in range(total)])
    W_test = np.array([rng.normal(loc=means_W_test[domain_W[i]], scale=covs_W[domain_W[i]]) for i in range(total)])
    X_test = U_test + rng.normal(0, 0.5, size=total)
    S_test = W_test + rng.normal(0, 0.5, size=total)
    Y_test = X_test - 3.0 * U_test + 2.0 * W_test + rng.normal(0, 0.5, size=total)
    df = pd.DataFrame({'domain_U': domain_U, 'domain_W': domain_W, 'U': U, 'W': W, 'X': X, 'S': S, 'Y': Y})
    df_test = pd.DataFrame({'domain_U': domain_U, 'domain_W': domain_W, 'U': U_test, 'W': W_test, 'X': X_test, 'S': S_test, 'Y': Y_test})
    return df, df_test

def design_matrix_X(x_array):
    return np.column_stack((np.ones(len(x_array)), x_array)).astype(np.float64)

def design_matrix_XS(x_array, s_array):
    return np.column_stack((np.ones(len(x_array)), x_array, s_array)).astype(np.float64)

def build_binary_leaf_embedding(leaf_train, leaf_test):
    n_train, n_trees = leaf_train.shape
    enc_maps = []
    col_sizes = []
    for t in range(n_trees):
        uniq = np.unique(np.concatenate([leaf_train[:, t], leaf_test[:, t]])).astype(int)
        uniq_sorted = np.sort(uniq)
        mapping = {v: idx for idx, v in enumerate(uniq_sorted)}
        enc_maps.append(mapping)
        col_sizes.append(len(mapping))
    total_cols = sum(col_sizes)
    emb = np.zeros((n_train, total_cols), dtype=np.float32)
    for i in range(n_train):
        off = 0
        for t in range(n_trees):
            v = int(leaf_train[i, t])
            emb[i, off + enc_maps[t][v]] = 1.0
            off += col_sizes[t]
    return emb

# --------------------------
# Run experiment (or load precomputed CSV)
# --------------------------
# If you've already run the experiment and saved a CSV, you can load it:
# results_df = pd.read_csv("clustering_vs_rmse_results.csv")
# Otherwise compute:
records = []
for n_per_domain in n_list:
    print("n_per_domain =", n_per_domain)
    seed = np.random.randint(0, 10000)
    df, df_test = generate_data(n_per_domain, num_domains=num_domains, seed=seed)
    labels_U = df['domain_U'].values
    labels_W = df['domain_W'].values
    labels_both = pd.factorize(list(zip(labels_U, labels_W)))[0]
    X_train_X = design_matrix_X(df['X'].values); X_test_X = design_matrix_X(df_test['X'].values)
    X_train_XS = design_matrix_XS(df['X'].values, df['S'].values); X_test_XS = design_matrix_XS(df_test['X'].values, df_test['S'].values)

    for model_name in models:
        if model_name == 'X':
            X_tr, X_te = X_train_X, X_test_X
        else:
            X_tr, X_te = X_train_XS, X_test_XS
        y_tr = df['Y'].values; y_te = df_test['Y'].values

        xgb = XGBRegressor(**xgb_params)
        xgb.fit(X_tr, y_tr)
        preds_te = xgb.predict(X_te)
        rmse = mean_squared_error(y_te, preds_te)

        leaf_train = xgb.apply(X_tr).astype(int)
        leaf_test = xgb.apply(X_te).astype(int)
        binary_emb = build_binary_leaf_embedding(leaf_train, leaf_test)
        n_train = binary_emb.shape[0]

        for embed in embedding_methods:
            if embed == 'pca':
                pca2 = PCA(n_components=2)
                emb2d = pca2.fit_transform(binary_emb)
                clusters = KMeans(n_clusters=num_domains, random_state=0, n_init=10).fit_predict(emb2d)
                ari_u = adjusted_rand_score(labels_U, clusters)
                nmi_u = normalized_mutual_info_score(labels_U, clusters)
                ari_w = adjusted_rand_score(labels_W, clusters)
                nmi_w = normalized_mutual_info_score(labels_W, clusters)
                ari_b = adjusted_rand_score(labels_both, clusters)
                nmi_b = normalized_mutual_info_score(labels_both, clusters)
                records.append({'n': n_per_domain, 'model': model_name, 'embed': embed,
                                'label_type': 'U', 'ARI': ari_u, 'NMI': nmi_u, 'RMSE': rmse})
                records.append({'n': n_per_domain, 'model': model_name, 'embed': embed,
                                'label_type': 'W', 'ARI': ari_w, 'NMI': nmi_w, 'RMSE': rmse})
                records.append({'n': n_per_domain, 'model': model_name, 'embed': embed,
                                'label_type': 'U+W', 'ARI': ari_b, 'NMI': nmi_b, 'RMSE': rmse})
            else:
                if n_train > TSNE_MAX:
                    idxs = np.sort(np.random.choice(n_train, TSNE_MAX, replace=False))
                    binary_sub = binary_emb[idxs]; labels_U_sub = labels_U[idxs]; labels_W_sub = labels_W[idxs]; labels_both_sub = labels_both[idxs]
                else:
                    idxs = np.arange(n_train)
                    binary_sub = binary_emb; labels_U_sub = labels_U; labels_W_sub = labels_W; labels_both_sub = labels_both
                pre = PCA(n_components=min(TSNE_PRE_PCA_DIM, binary_sub.shape[1]))
                reduced = pre.fit_transform(binary_sub) if binary_sub.shape[1] > min(TSNE_PRE_PCA_DIM, binary_sub.shape[1]) else binary_sub
                tsne = TSNE(n_components=2, perplexity=30, random_state=0, init='pca', max_iter=TSNE_ITERS, learning_rate='auto')
                emb2d = tsne.fit_transform(reduced)
                clusters = KMeans(n_clusters=num_domains, random_state=0, n_init=10).fit_predict(emb2d)
                ari_u = adjusted_rand_score(labels_U_sub, clusters)
                nmi_u = normalized_mutual_info_score(labels_U_sub, clusters)
                ari_w = adjusted_rand_score(labels_W_sub, clusters)
                nmi_w = normalized_mutual_info_score(labels_W_sub, clusters)
                ari_b = adjusted_rand_score(labels_both_sub, clusters)
                nmi_b = normalized_mutual_info_score(labels_both_sub, clusters)
                records.append({'n': n_per_domain, 'model': model_name, 'embed': embed,
                                'label_type': 'U', 'ARI': ari_u, 'NMI': nmi_u, 'RMSE': rmse, 'sub': len(idxs)})
                records.append({'n': n_per_domain, 'model': model_name, 'embed': embed,
                                'label_type': 'W', 'ARI': ari_w, 'NMI': nmi_w, 'RMSE': rmse, 'sub': len(idxs)})
                records.append({'n': n_per_domain, 'model': model_name, 'embed': embed,
                                'label_type': 'U+W', 'ARI': ari_b, 'NMI': nmi_b, 'RMSE': rmse, 'sub': len(idxs)})

results_df = pd.DataFrame(records)
results_df.to_csv("clustering_vs_rmse_results.csv", index=False)
print("Saved results to clustering_vs_rmse_results.csv")

# --------------------------
# Combined plotting: overlay X and X+S for direct comparison, same axis scales
# --------------------------
label_types = ['U+W']

# Determine global axis limits for consistent scales
# For ARI plots:
ari_min = results_df['ARI'].min(); ari_max = results_df['ARI'].max()
nmi_min = results_df['NMI'].min(); nmi_max = results_df['NMI'].max()
rmse_min = results_df['RMSE'].min(); rmse_max = results_df['RMSE'].max()
# add small margins
ari_pad = (ari_max - ari_min) * 0.05 if ari_max > ari_min else 0.05
nmi_pad = (nmi_max - nmi_min) * 0.05 if nmi_max > nmi_min else 0.05
rmse_pad = (rmse_max - rmse_min) * 0.05 if rmse_max > rmse_min else 0.05

# Colors / markers for models
model_props = {'X': {'color': 'C0', 'marker': 'v', 'markersize': 70}, 'X,S': {'color': 'C1', 'marker': '^', 'markersize': 70}}

# ARI figure: rows = label_types, cols = embedding methods
fig_ari, axes_ari = plt.subplots(len(label_types), len(embedding_methods), figsize=(10,4), sharex=False, sharey=False)
for i, label_type in enumerate(label_types):
    for j, embed in enumerate(embedding_methods):
        ax = axes_ari[i, j] if axes_ari.ndim == 2 else axes_ari[max(i,j)]
        for model_name in models:
            sel = results_df[(results_df['model'] == model_name) & (results_df['embed'] == embed) & (results_df['label_type'] == label_type)].sort_values('n')
            # plot connected lines for trend and points
            ax.scatter(sel['ARI'].values, sel['RMSE'].values, label=f"Using {model_name}", color=model_props[model_name]['color'], marker=model_props[model_name]['marker'],s=model_props[model_name]['markersize'])
            # for idx, (a, r, nval) in enumerate(zip(sel['ARI'].values, sel['RMSE'].values, sel['n'].values)):
            #     ax.annotate(str(nval), (a, r), textcoords="offset points", xytext=(4,4), fontsize=7, color=model_props[model_name]['color'])
        ax.set_title(f"{embed}", fontweight='bold', fontsize=16)
        ax.set_xlabel("ARI", fontweight='bold', fontsize=16); ax.set_ylabel("MSE", fontweight='bold', fontsize=16)
        ax.set_xlim(ari_min - ari_pad, ari_max + ari_pad)
        ax.set_ylim(rmse_min - rmse_pad, rmse_max + rmse_pad)
        ax.grid(True)
        ax.legend()

plt.tight_layout()
plt.savefig("results/rmse_vs_ari.pdf", dpi=200)

# NMI figure: rows = label_types, cols = embedding methods
fig_nmi, axes_nmi = plt.subplots(len(label_types), len(embedding_methods),  figsize=(10,4), sharex=False, sharey=False)
for i, label_type in enumerate(label_types):
    for j, embed in enumerate(embedding_methods):
        ax = axes_nmi[i,j] if axes_nmi.ndim == 2 else axes_nmi[max(i,j)]
        for model_name in models:
            sel = results_df[(results_df['model'] == model_name) & (results_df['embed'] == embed) & (results_df['label_type'] == label_type)].sort_values('n')
            ax.scatter(sel['NMI'].values, sel['RMSE'].values, label=f"Using {model_name}", color=model_props[model_name]['color'], marker=model_props[model_name]['marker'],s=model_props[model_name]['markersize'])
            # for idx, (m, r, nval) in enumerate(zip(sel['NMI'].values, sel['RMSE'].values, sel['n'].values)):
            #     ax.annotate(str(nval), (m, r), textcoords="offset points", xytext=(4,4), fontsize=7, color=model_props[model_name]['color'])
        ax.set_title(f"{embed}",fontweight='bold', fontsize=16)
        ax.set_xlabel("NMI", fontweight='bold', fontsize=16); ax.set_ylabel("RMSE", fontweight='bold', fontsize=16)
        ax.set_xlim(nmi_min - nmi_pad, nmi_max + nmi_pad)
        ax.set_ylim(rmse_min - rmse_pad, rmse_max + rmse_pad)
        ax.grid(True)
        ax.legend()

plt.tight_layout()
plt.savefig("results/rmse_vs_nmi.pdf", dpi=200)
