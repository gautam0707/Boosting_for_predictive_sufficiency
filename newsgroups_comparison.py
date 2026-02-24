#!/usr/bin/env python3
"""
20 Newsgroups — XGBoost & CatBoost (CPU-only)
Compute:
  metric1 = I(Y; phi(X) | E) - I(Y; phi(X) | Yhat)
  metric2 = I(Y; phi(X) | Yhat) + I(Y; E | phi(X)) - I(Y; phi(X) | E, Yhat)

After hyperparameter search, the best model is re-trained with N_SEEDS_EVAL random seeds
and the mean/std of metrics across seeds are reported.

Saves CSV per combination with: model, accuracy_mean, accuracy_std, macro_f1_mean, macro_f1_std,
train_mse_mean, train_mse_std, train_r2_mean, train_r2_std, metric1_mean, metric1_std, metric2_mean, metric2_std
and MI components means.
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from scipy.stats import ks_2samp
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
os.makedirs('results', exist_ok=True)
np.random.seed(42)

# ---------------- Config ----------------
K_MI = 5                # k for k-NN MI estimator (npeet)
MI_SAMPLE_SIZE = 1000   # subsample size for MI (<= len(test))
N_MI_PCA = 10           # PCA dims for phi before MI
N_ITER_SEARCH = 100     # randomized search iterations per model (reduced for speed)
CV_FOLDS = 3
N_CLUSTERS = 5          # clusters for E
USE_ONEHOT_E = True     # one-hot encode E for conditioning
MAX_TFIDF_FEATURES = 20000
MAX_SVD_COMPONENTS = 200
N_SEEDS_EVAL = 5        # number of random seeds to evaluate best model

# ---------------- NPEET wrappers ----------------
try:
    from npeet import entropy_estimators as ee
except Exception as exc:
    raise ImportError("npeet is required. Install with `pip install npeet` or `pip install npeet-plus`. "
                      f"Original error: {exc}")

def ensure_2d_for_npeet(arr):
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a.astype(np.float64)

def mi_npeet(x, y, k=K_MI):
    x2 = ensure_2d_for_npeet(x)
    y2 = ensure_2d_for_npeet(y)
    return float(ee.mi(x2, y2, k=k))

def conditional_mi(a, b, cond, k=K_MI):
    """
    I(a; b | cond) = I(a; [b,cond]) - I(a; cond)
    Clip small negatives to 0.
    """
    a2 = ensure_2d_for_npeet(a)
    b2 = ensure_2d_for_npeet(b)
    c2 = ensure_2d_for_npeet(cond)
    joint = np.hstack([b2, c2])
    I_a_joint = mi_npeet(a2, joint, k=k)
    I_a_c = mi_npeet(a2, c2, k=k)
    val = I_a_joint - I_a_c
    return float(max(0.0, val))

# ---------------- leaf embeddings ----------------
def leaf_embeddings_from_leaf_indices(leaf_indices):
    leaf_indices = np.asarray(leaf_indices)
    if leaf_indices.ndim == 1:
        leaf_indices = leaf_indices.reshape(-1, 1)
    n_samples, n_trees = leaf_indices.shape
    embeddings = np.zeros((n_samples, n_trees), dtype=np.float64)
    for i in range(n_trees):
        leaves = leaf_indices[:, i]
        uniq = np.unique(leaves)
        if len(uniq) > 1:
            minv, maxv = leaves.min(), leaves.max()
            denom = (maxv - minv) if (maxv != minv) else 1.0
            embeddings[:, i] = (leaves - minv) / denom
        else:
            embeddings[:, i] = 0.0
    return embeddings

# ---------------- helpers: reduce / subsample / encoding ----------------
def reduce_phi_for_mi(phi_train, phi_test, n_components=N_MI_PCA):
    phi_train = np.asarray(phi_train)
    phi_test = np.asarray(phi_test)
    fit_data = phi_train if (phi_train.size and phi_train.shape[1] > 1) else phi_test
    max_possible = min(n_components, fit_data.shape[1], max(1, fit_data.shape[0]-1))
    max_possible = max(1, int(max_possible))
    try:
        pca = PCA(n_components=max_possible, random_state=42)
        pca.fit(fit_data)
        phi_train_red = pca.transform(phi_train) if phi_train.size else phi_train
        phi_test_red = pca.transform(phi_test)
    except Exception:
        phi_train_red = phi_train
        phi_test_red = phi_test
    return phi_train_red, phi_test_red

def subsample_indices(n, size=MI_SAMPLE_SIZE, seed=42):
    if n <= size:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=size, replace=False)

def one_hot_encode_labels(labels):
    labels = np.asarray(labels).astype(int)
    unique = np.unique(labels)
    mapping = {v:i for i,v in enumerate(unique)}
    mapped = np.array([mapping[v] for v in labels], dtype=int)
    n_mapped = len(unique)
    oh = np.zeros((len(labels), n_mapped), dtype=float)
    oh[np.arange(len(labels)), mapped] = 1.0
    return oh

# ---------------- frequency shift helper ----------------
def create_frequency_shift(combination_values, train_ratio=0.7, n_quantiles=5):
    quantiles = pd.qcut(combination_values, q=n_quantiles, labels=False, duplicates='drop')
    train_indices = []
    test_indices = []
    for q in range(n_quantiles):
        q_idx = np.where(quantiles == q)[0]
        if len(q_idx) == 0:
            continue
        np.random.shuffle(q_idx)
        n_train_per_q = max(1, int(len(q_idx) * train_ratio / n_quantiles * 2))
        train_indices.extend(q_idx[:n_train_per_q])
        n_test_per_q = int(len(q_idx) * (1 - train_ratio) * (q + 1) / n_quantiles)
        test_indices.extend(q_idx[n_train_per_q : n_train_per_q + n_test_per_q])
    train_mask = np.zeros(len(combination_values), dtype=bool)
    test_mask = np.zeros(len(combination_values), dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    leftover = ~(train_mask | test_mask)
    train_mask[leftover] = True
    return train_mask, test_mask

# ---------------- Load 20 Newsgroups ----------------
print("Loading 20 Newsgroups dataset...")
newsgroups = fetch_20newsgroups(subset="all", remove=("headers","footers","quotes"))
X_text = pd.Series(newsgroups.data).fillna("")
y = pd.Series(newsgroups.target)
print(f"Samples: {len(X_text)}, Classes: {len(np.unique(y))}")

# ---------------- confounder functions ----------------
def comb_textlen_topicword(X, y):
    has_topic = X.str.contains(r"\b(computer|software|windows|file|program)\b", case=False, regex=True).astype(int)
    text_len = X.str.len().fillna(0)
    return np.log1p(text_len) * has_topic + 1.5 * (y.astype(float))

combinations = {
    "Comb1_TextLen_TopicWord": {"function": comb_textlen_topicword, "description": "log(length) * topic_presence + 1.5*label"},
}

# ---------------- CPU-only model constructors ----------------
def get_xgb_classifier(**kwargs):
    clf = xgb.XGBClassifier(
        tree_method="hist",
        predictor="cpu_predictor",
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
        n_jobs=-1,
        **kwargs
    )
    return clf

def get_catboost_classifier(**kwargs):
    clf = CatBoostClassifier(task_type="CPU", verbose=False, random_seed=42, **kwargs)
    return clf

# ---------------- main runner ----------------
def run_experiment(comb_name, comb_func, description, X_text, y,
                   k_mi=K_MI, mi_sample_size=MI_SAMPLE_SIZE, n_mi_pca=N_MI_PCA,
                   n_clusters=N_CLUSTERS, use_onehot_e=USE_ONEHOT_E):
    t0 = time.time()
    print(f"\n=== Combination: {comb_name} - {description} ===")

    X_reset = X_text.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)

    comb_vals = comb_func(X_reset, y_reset)
    comb_vals = (comb_vals - comb_vals.mean()) / (comb_vals.std() + 1e-12)

    # create frequency shift with common support
    train_mask, test_mask = create_frequency_shift(comb_vals, train_ratio=0.7)
    X_train = X_reset[train_mask].reset_index(drop=True)
    y_train = y_reset[train_mask].reset_index(drop=True)
    X_test = X_reset[test_mask].reset_index(drop=True)
    y_test = y_reset[test_mask].reset_index(drop=True)
    comb_train = pd.Series(comb_vals[train_mask]).reset_index(drop=True)
    comb_test = pd.Series(comb_vals[test_mask]).reset_index(drop=True)

    ks_stat, ks_p = ks_2samp(comb_train, comb_test)
    train_range = (comb_train.min(), comb_train.max())
    test_range = (comb_test.min(), comb_test.max())
    common_support = (test_range[0] >= train_range[0]) and (test_range[1] <= train_range[1])
    print(f"Train size {len(X_train)}, Test size {len(X_test)}, common_support={common_support}, KS={ks_stat:.4f}")

    # Vectorize + SVD
    vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, stop_words="english")
    X_all_tfidf = vectorizer.fit_transform(pd.concat([X_train, X_test]))
    n_train = len(X_train)
    X_train_tfidf = X_all_tfidf[:n_train]
    X_test_tfidf = X_all_tfidf[n_train:]

    svd = TruncatedSVD(n_components=min(MAX_SVD_COMPONENTS, max(1, X_train_tfidf.shape[1]-1)), random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_svd)
    X_test_scaled = scaler.transform(X_test_svd)

    results_rows = []

    # prepare evaluation seeds (fixed RNG so repeatable)
    rng = np.random.default_rng(42)
    eval_seeds = rng.integers(0, 1_000_000, size=N_SEEDS_EVAL)

    # ---------- XGBoost classifier tuning ----------
    print("Tuning XGBoost (RandomizedSearchCV)...")
    xgb_clf = get_xgb_classifier()
    xgb_param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    xgb_search = RandomizedSearchCV(xgb_clf, xgb_param_dist, n_iter=N_ITER_SEARCH, scoring='accuracy',
                                    cv=CV_FOLDS, random_state=42, n_jobs=-1, verbose=0)
    t1 = time.time()
    xgb_search.fit(X_train_scaled, y_train)
    t2 = time.time()
    best_xgb_params = xgb_search.best_params_
    print(f"XGB tuned in {t2-t1:.1f}s; best_params={best_xgb_params}")

    # Evaluate best XGB over multiple seeds
    xgb_accs = []; xgb_f1s = []; xgb_train_mses = []; xgb_train_r2s = []
    xgb_m1 = []; xgb_m2 = []
    xgb_I_Y_phi_given_E_list = []; xgb_I_Y_phi_given_Yhat_list = []; xgb_I_Y_E_given_phi_list = []; xgb_I_Y_phi_given_EYhat_list = []

    for seed in eval_seeds:
        # build model with best params but new seed
        model_params = dict(best_xgb_params)
        model = get_xgb_classifier(random_state=int(seed), **model_params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_pred_train = model.predict(X_train_scaled)
        acc_val = accuracy_score(y_test, y_pred)
        f1_val = f1_score(y_test, y_pred, average='macro')
        mse_train = mean_squared_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)

        # leaf embeddings -> phi
        leaf_idx_test = model.apply(X_test_scaled)
        leaf_idx_train = model.apply(X_train_scaled)
        if leaf_idx_test.ndim == 1: leaf_idx_test = leaf_idx_test.reshape(-1,1)
        if leaf_idx_train.ndim == 1: leaf_idx_train = leaf_idx_train.reshape(-1,1)
        phi_test = leaf_embeddings_from_leaf_indices(leaf_idx_test)
        phi_train = leaf_embeddings_from_leaf_indices(leaf_idx_train)

        # clustering for E (per-run)
        try:
            pca_clust = PCA(n_components=min(20, phi_train.shape[1], max(1, phi_train.shape[0]-1)), random_state=int(seed))
            phi_train_for_clust = pca_clust.fit_transform(phi_train)
            phi_test_for_clust = pca_clust.transform(phi_test)
        except Exception:
            phi_train_for_clust = phi_train
            phi_test_for_clust = phi_test

        kmeans = KMeans(n_clusters=n_clusters, random_state=int(seed))
        kmeans.fit(phi_train_for_clust)
        E_test = kmeans.predict(phi_test_for_clust)
        E_test_cond = one_hot_encode_labels(E_test) if use_onehot_e else ensure_2d_for_npeet(E_test.astype(float))

        # reduce phi for MI and subsample (fixed subsample idx for fair comparison)
        phi_train_red, phi_test_red = reduce_phi_for_mi(phi_train, phi_test, n_components=n_mi_pca)
        idx = subsample_indices(phi_test_red.shape[0], size=mi_sample_size, seed=42)
        phi_test_red_mi = phi_test_red[idx]
        y_test_mi = y_test.values[idx].astype(float)
        yhat_mi = np.asarray(y_pred)[idx].astype(float)
        E_test_cond_mi = E_test_cond[idx]

        # compute MI components
        I_Y_phi_given_E = conditional_mi(y_test_mi, phi_test_red_mi, E_test_cond_mi, k=k_mi)
        I_Y_phi_given_Yhat = conditional_mi(y_test_mi, phi_test_red_mi, ensure_2d_for_npeet(yhat_mi), k=k_mi)
        I_Y_E_given_phi = conditional_mi(y_test_mi, E_test_cond_mi, phi_test_red_mi, k=k_mi)
        cond_E_Yhat = np.hstack([E_test_cond_mi, ensure_2d_for_npeet(yhat_mi)])
        I_Y_phi_given_EYhat = conditional_mi(y_test_mi, phi_test_red_mi, cond_E_Yhat, k=k_mi)

        metric1_val = I_Y_phi_given_E - I_Y_phi_given_Yhat
        metric2_val = I_Y_phi_given_Yhat + I_Y_E_given_phi - I_Y_phi_given_EYhat

        # collect
        xgb_accs.append(acc_val)
        xgb_f1s.append(f1_val)
        xgb_train_mses.append(mse_train)
        xgb_train_r2s.append(r2_train)
        xgb_m1.append(metric1_val)
        xgb_m2.append(metric2_val)
        xgb_I_Y_phi_given_E_list.append(I_Y_phi_given_E)
        xgb_I_Y_phi_given_Yhat_list.append(I_Y_phi_given_Yhat)
        xgb_I_Y_E_given_phi_list.append(I_Y_E_given_phi)
        xgb_I_Y_phi_given_EYhat_list.append(I_Y_phi_given_EYhat)

    # aggregate XGB stats
    results_rows.append({
        'model': 'xgboost',
        'accuracy_mean': float(np.mean(xgb_accs)), 'accuracy_std': float(np.std(xgb_accs, ddof=0)),
        'macro_f1_mean': float(np.mean(xgb_f1s)), 'macro_f1_std': float(np.std(xgb_f1s, ddof=0)),
        'train_mse_mean': float(np.mean(xgb_train_mses)), 'train_mse_std': float(np.std(xgb_train_mses, ddof=0)),
        'train_r2_mean': float(np.mean(xgb_train_r2s)), 'train_r2_std': float(np.std(xgb_train_r2s, ddof=0)),
        'metric1_mean': float(np.mean(xgb_m1)), 'metric1_std': float(np.std(xgb_m1, ddof=0)),
        'metric2_mean': float(np.mean(xgb_m2)), 'metric2_std': float(np.std(xgb_m2, ddof=0)),
        'I_Y_phi_given_E_mean': float(np.mean(xgb_I_Y_phi_given_E_list)),
        'I_Y_phi_given_Yhat_mean': float(np.mean(xgb_I_Y_phi_given_Yhat_list)),
        'I_Y_E_given_phi_mean': float(np.mean(xgb_I_Y_E_given_phi_list)),
        'I_Y_phi_given_EYhat_mean': float(np.mean(xgb_I_Y_phi_given_EYhat_list)),
        'n_clusters': n_clusters,
        'best_params': best_xgb_params,
        'eval_seeds': list(map(int, eval_seeds))
    })

    # ---------- CatBoost classifier tuning ----------
    print("Tuning CatBoost (RandomizedSearchCV)...")
    cb = get_catboost_classifier()
    cb_param_dist = {
        'iterations': [400,500,600],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
    }
    cb_search = RandomizedSearchCV(cb, cb_param_dist, n_iter=N_ITER_SEARCH, scoring='accuracy',
                                   cv=CV_FOLDS, random_state=42, n_jobs=-1, verbose=0)
    t3 = time.time()
    cb_search.fit(X_train_scaled, y_train)
    t4 = time.time()
    best_cb_params = cb_search.best_params_
    print(f"CatBoost tuned in {t4-t3:.1f}s; best_params={best_cb_params}")

    # Evaluate best CatBoost over multiple seeds
    cb_accs = []; cb_f1s = []; cb_train_mses = []; cb_train_r2s = []
    cb_m1 = []; cb_m2 = []
    cb_I_Y_phi_given_E_list = []; cb_I_Y_phi_given_Yhat_list = []; cb_I_Y_E_given_phi_list = []; cb_I_Y_phi_given_EYhat_list = []

    for seed in eval_seeds:
        model_params = dict(best_cb_params)
        # ensure reproducible random seed and CPU task_type
        model_params.update({'random_seed': int(seed), 'task_type': 'CPU', 'verbose': False})
        model = CatBoostClassifier(**model_params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_pred_train = model.predict(X_train_scaled)
        acc_val = accuracy_score(y_test, y_pred)
        f1_val = f1_score(y_test, y_pred, average='macro')
        mse_train = mean_squared_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)

        # leaf embeddings (CatBoost: calc_leaf_indexes)
        leaf_idx_test_cb = np.array(model.calc_leaf_indexes(X_test_scaled))
        leaf_idx_train_cb = np.array(model.calc_leaf_indexes(X_train_scaled))
        if leaf_idx_test_cb.ndim == 1: leaf_idx_test_cb = leaf_idx_test_cb.reshape(-1,1)
        if leaf_idx_train_cb.ndim == 1: leaf_idx_train_cb = leaf_idx_train_cb.reshape(-1,1)
        phi_test = leaf_embeddings_from_leaf_indices(leaf_idx_test_cb)
        phi_train = leaf_embeddings_from_leaf_indices(leaf_idx_train_cb)

        # clustering for E (per-run)
        try:
            pca_clust_cb = PCA(n_components=min(20, phi_train.shape[1], max(1, phi_train.shape[0]-1)), random_state=int(seed))
            phi_train_for_clust_cb = pca_clust_cb.fit_transform(phi_train)
            phi_test_for_clust_cb = pca_clust_cb.transform(phi_test)
        except Exception:
            phi_train_for_clust_cb = phi_train
            phi_test_for_clust_cb = phi_test

        kmeans_cb = KMeans(n_clusters=n_clusters, random_state=int(seed))
        kmeans_cb.fit(phi_train_for_clust_cb)
        E_test_cb = kmeans_cb.predict(phi_test_for_clust_cb)
        E_test_cb_cond = one_hot_encode_labels(E_test_cb) if use_onehot_e else ensure_2d_for_npeet(E_test_cb.astype(float))

        # reduce phi and subsample (fixed idx)
        phi_train_cb_red, phi_test_cb_red = reduce_phi_for_mi(phi_train, phi_test, n_components=n_mi_pca)
        idx_cb = subsample_indices(phi_test_cb_red.shape[0], size=mi_sample_size, seed=42)
        phi_test_red_cb = phi_test_cb_red[idx_cb]
        y_test_cb_mi = y_test.values[idx_cb].astype(float)
        yhat_cb_mi = np.asarray(y_pred)[idx_cb].astype(float)
        E_test_cb_cond_mi = E_test_cb_cond[idx_cb]

        # compute MI components
        I_Y_phi_given_E_cb = conditional_mi(y_test_cb_mi, phi_test_red_cb, E_test_cb_cond_mi, k=k_mi)
        I_Y_phi_given_Yhat_cb = conditional_mi(y_test_cb_mi, phi_test_red_cb, ensure_2d_for_npeet(yhat_cb_mi), k=k_mi)
        I_Y_E_given_phi_cb = conditional_mi(y_test_cb_mi, E_test_cb_cond_mi, phi_test_red_cb, k=k_mi)
        cond_E_Yhat_cb = np.hstack([E_test_cb_cond_mi, ensure_2d_for_npeet(yhat_cb_mi)])
        I_Y_phi_given_EYhat_cb = conditional_mi(y_test_cb_mi, phi_test_red_cb, cond_E_Yhat_cb, k=k_mi)

        metric1_val = I_Y_phi_given_E_cb - I_Y_phi_given_Yhat_cb
        metric2_val = I_Y_phi_given_Yhat_cb + I_Y_E_given_phi_cb - I_Y_phi_given_EYhat_cb

        # collect
        cb_accs.append(acc_val)
        cb_f1s.append(f1_val)
        cb_train_mses.append(mse_train)
        cb_train_r2s.append(r2_train)
        cb_m1.append(metric1_val)
        cb_m2.append(metric2_val)
        cb_I_Y_phi_given_E_list.append(I_Y_phi_given_E_cb)
        cb_I_Y_phi_given_Yhat_list.append(I_Y_phi_given_Yhat_cb)
        cb_I_Y_E_given_phi_list.append(I_Y_E_given_phi_cb)
        cb_I_Y_phi_given_EYhat_list.append(I_Y_phi_given_EYhat_cb)

    # aggregate CatBoost stats
    results_rows.append({
        'model': 'catboost',
        'accuracy_mean': float(np.mean(cb_accs)), 'accuracy_std': float(np.std(cb_accs, ddof=0)),
        'macro_f1_mean': float(np.mean(cb_f1s)), 'macro_f1_std': float(np.std(cb_f1s, ddof=0)),
        'train_mse_mean': float(np.mean(cb_train_mses)), 'train_mse_std': float(np.std(cb_train_mses, ddof=0)),
        'train_r2_mean': float(np.mean(cb_train_r2s)), 'train_r2_std': float(np.std(cb_train_r2s, ddof=0)),
        'metric1_mean': float(np.mean(cb_m1)), 'metric1_std': float(np.std(cb_m1, ddof=0)),
        'metric2_mean': float(np.mean(cb_m2)), 'metric2_std': float(np.std(cb_m2, ddof=0)),
        'I_Y_phi_given_E_mean': float(np.mean(cb_I_Y_phi_given_E_list)),
        'I_Y_phi_given_Yhat_mean': float(np.mean(cb_I_Y_phi_given_Yhat_list)),
        'I_Y_E_given_phi_mean': float(np.mean(cb_I_Y_E_given_phi_list)),
        'I_Y_phi_given_EYhat_mean': float(np.mean(cb_I_Y_phi_given_EYhat_list)),
        'n_clusters': n_clusters,
        'best_params': best_cb_params,
        'eval_seeds': list(map(int, eval_seeds))
    })

    results_df = pd.DataFrame(results_rows).set_index('model')
    csv_path = f"results/{comb_name}_newsgroups_metric1_metric2.csv"
    results_df.to_csv(csv_path)
    print(f"Saved results CSV: {csv_path}")
    print(results_df[['accuracy_mean','accuracy_std','macro_f1_mean','macro_f1_std','train_mse_mean','train_mse_std','train_r2_mean','train_r2_std','metric1_mean','metric1_std','metric2_mean','metric2_std']])
    print("Elapsed time: {:.1f}s".format(time.time() - t0))

    return results_df, common_support, ks_stat

# ---------------- run all combinations ----------------
if __name__ == "__main__":
    all_tables = []
    for comb_name, comb_info in combinations.items():
        try:
            df_res, cs_flag, ks = run_experiment(comb_name, comb_info['function'], comb_info['description'], X_text, y,
                                                 k_mi=K_MI, mi_sample_size=MI_SAMPLE_SIZE, n_mi_pca=N_MI_PCA,
                                                 n_clusters=N_CLUSTERS, use_onehot_e=USE_ONEHOT_E)
            tbl = df_res.reset_index().assign(combination=comb_name)
            all_tables.append(tbl)
        except Exception as e:
            import traceback
            print(f"Error in {comb_name}: {e}")
            traceback.print_exc()
            continue

    if all_tables:
        big_df = pd.concat(all_tables, ignore_index=True)
        big_csv = "results/all_newsgroups_metric1_metric2.csv"
        big_df.to_csv(big_csv, index=False)
        print(f"\nSaved aggregated CSV: {big_csv}")
        print(big_df[['combination','model','accuracy_mean','accuracy_std','macro_f1_mean','macro_f1_std','train_mse_mean','train_mse_std','train_r2_mean','train_r2_std','metric1_mean','metric1_std','metric2_mean','metric2_std']])
    else:
        print("No successful runs.")
    print("Done.")
