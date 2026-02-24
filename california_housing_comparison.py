#!/usr/bin/env python3
"""
California Housing — XGBoost & CatBoost
Compute only:
  metric1 = I(Y; phi(X) | E) - I(Y; phi(X) | Yhat)
  metric2 = I(Y; phi(X) | Yhat) + I(Y; E | phi(X)) - I(Y; phi(X) | E, Yhat)
Saves CSV with: model, r2_mean, r2_std, mse_mean, mse_std, metric1_mean, metric1_std, metric2_mean, metric2_std
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ks_2samp
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
os.makedirs('results', exist_ok=True)
np.random.seed(42)

# ---------------- Config ----------------
K_MI = 5               # k for k-NN MI estimator
MI_SAMPLE_SIZE = 2000  # subsample size for MI (<= len(test))
N_MI_PCA = 50          # PCA dims for phi before MI
N_ITER_SEARCH = 100     # randomized search iterations per model
CV_FOLDS = 3
N_CLUSTERS = 5         # clusters for E
USE_ONEHOT_E = True    # one-hot encode E for conditioning (recommended)
N_SEEDS_EVAL = 5       # number of random seeds to evaluate best model

# ---------------- NPEET k-NN MI wrappers ----------------
try:
    from npeet import entropy_estimators as ee
except Exception as exc:
    raise ImportError("npeet is required. Install with `pip install npeet`. Original error: " + str(exc))

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
    a, b, cond: arrays (n,) or (n, d)
    Returns non-negative float (clip small negatives to 0).
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

# ---------------- helpers: reduce phi & subsample ----------------
def reduce_phi_for_mi(phi_train, phi_test, n_components=N_MI_PCA):
    phi_train = np.asarray(phi_train)
    phi_test = np.asarray(phi_test)
    # pick fit data
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

# ---------------- frequency shift helper (same as earlier) ----------------
def create_frequency_shift(combination_values, train_ratio=0.7):
    n_quantiles = 5
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
        n_test_per_q = max(0, int(len(q_idx) * (1 - train_ratio) * (q + 1) / n_quantiles))
        test_indices.extend(q_idx[n_train_per_q: n_train_per_q + n_test_per_q])
    train_mask = np.zeros(len(combination_values), dtype=bool)
    test_mask = np.zeros(len(combination_values), dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    leftover = ~(train_mask | test_mask)
    train_mask[leftover] = True
    return train_mask, test_mask

# ---------------- Load data ----------------
print("Loading California Housing dataset...")
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
print("Dataset shape:", X.shape)

# ---------------- combination(s) ----------------
combinations = {
    'Comb1_MedInc_HouseAge': {
        'function': lambda X, y: np.log1p(X['MedInc']) * (X['HouseAge'] / 5) + 5*y,
        'description': 'log(MedInc) × HouseAge/5'
    }
}

# ---------------- main experiment ----------------
def run_experiment(cal_name, comb_func, description, X, y,
                   k_mi=K_MI, mi_sample_size=MI_SAMPLE_SIZE, n_mi_pca=N_MI_PCA,
                   n_clusters=N_CLUSTERS, use_onehot_e=USE_ONEHOT_E):
    t0 = time.time()
    print(f"\n=== Experiment: {cal_name} ===")
    # prepare
    X_reset = X.reset_index(drop=True); y_reset = y.reset_index(drop=True)
    comb_vals = comb_func(X_reset, y_reset)
    comb_vals = (comb_vals - comb_vals.mean()) / (comb_vals.std() + 1e-12)
    train_mask, test_mask = create_frequency_shift(comb_vals, train_ratio=0.7)

    X_train = X_reset[train_mask].reset_index(drop=True); y_train = y_reset[train_mask].reset_index(drop=True)
    X_test = X_reset[test_mask].reset_index(drop=True); y_test = y_reset[test_mask].reset_index(drop=True)
    comb_train = pd.Series(comb_vals[train_mask]).reset_index(drop=True); comb_test = pd.Series(comb_vals[test_mask]).reset_index(drop=True)

    ks_stat, ks_p = ks_2samp(comb_train, comb_test)
    train_range = (comb_train.min(), comb_train.max()); test_range = (comb_test.min(), comb_test.max())
    common_support = (test_range[0] >= train_range[0]) and (test_range[1] <= train_range[1])
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, common_support={common_support}, KS={ks_stat:.4f}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

    results_rows = []

    rng = np.random.default_rng(42)
    eval_seeds = rng.integers(0, 1_000_000, size=N_SEEDS_EVAL)

    # ---------- XGBoost ----------
    print("Tuning XGBoost (CPU)...")
    xgb_clf = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0, n_jobs=-1)
    xgb_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    xgb_search = RandomizedSearchCV(xgb_clf, xgb_param_dist, n_iter=N_ITER_SEARCH, scoring='neg_mean_squared_error',
                                    cv=CV_FOLDS, random_state=42, n_jobs=-1, verbose=0)
    t1 = time.time(); xgb_search.fit(X_train_scaled, y_train); t2 = time.time()
    best_xgb_params = xgb_search.best_params_
    print(f"XGB tuned in {t2-t1:.1f}s; best_params={best_xgb_params}")

    # Evaluate best XGB over multiple seeds
    xgb_r2s = []; xgb_mses = []; xgb_m1 = []; xgb_m2 = []
    xgb_I_Y_phi_given_E_list = []; xgb_I_Y_phi_given_Yhat_list = []; xgb_I_Y_E_given_phi_list = []; xgb_I_Y_phi_given_EYhat_list = []

    for seed in eval_seeds:
        model_params = dict(best_xgb_params)  # copy
        # enforce CPU settings and set seed
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=int(seed), verbosity=0, n_jobs=-1, **model_params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        r2_val = r2_score(y_test, y_pred); mse_val = mean_squared_error(y_test, y_pred)

        # leaf embeddings
        leaf_idx_test = model.apply(X_test_scaled)
        leaf_idx_train = model.apply(X_train_scaled)
        if leaf_idx_test.ndim == 1: leaf_idx_test = leaf_idx_test.reshape(-1,1)
        if leaf_idx_train.ndim == 1: leaf_idx_train = leaf_idx_train.reshape(-1,1)
        phi_test = leaf_embeddings_from_leaf_indices(leaf_idx_test)
        phi_train = leaf_embeddings_from_leaf_indices(leaf_idx_train)

        # clustering for E (per-run)
        try:
            pca_c = PCA(n_components=min(20, phi_train.shape[1], max(1, phi_train.shape[0]-1)), random_state=seed)
            phi_train_clust = pca_c.fit_transform(phi_train) if phi_train.size and phi_train.shape[1]>1 else phi_train
            phi_test_clust = pca_c.transform(phi_test) if phi_test.size and phi_test.shape[1]>1 else phi_test
        except Exception:
            phi_train_clust = phi_train; phi_test_clust = phi_test

        kmeans = KMeans(n_clusters=n_clusters, random_state=int(seed))
        kmeans.fit(phi_train_clust)
        E_test = kmeans.predict(phi_test_clust)
        E_test_cond = one_hot_encode_labels(E_test) if use_onehot_e else ensure_2d_for_npeet(E_test.astype(float))

        # reduce phi & subsample (use fixed subsample indices for fair comparisons)
        phi_train_red, phi_test_red = reduce_phi_for_mi(phi_train, phi_test, n_components=n_mi_pca)
        idx = subsample_indices(phi_test_red.shape[0], size=mi_sample_size, seed=42)
        phi_test_red_mi = phi_test_red[idx]
        y_test_mi = y_test.values[idx].astype(float)
        yhat_mi = np.asarray(y_pred)[idx].astype(float)
        E_test_cond_mi = E_test_cond[idx] if use_onehot_e else E_test_cond[idx].reshape(-1,1)

        # compute MI components
        I_Y_phi_given_E = conditional_mi(y_test_mi, phi_test_red_mi, E_test_cond_mi, k=k_mi)
        I_Y_phi_given_Yhat = conditional_mi(y_test_mi, phi_test_red_mi, ensure_2d_for_npeet(yhat_mi), k=k_mi)
        I_Y_E_given_phi = conditional_mi(y_test_mi, E_test_cond_mi, phi_test_red_mi, k=k_mi)
        cond_E_Yhat = np.hstack([E_test_cond_mi if use_onehot_e else E_test_cond_mi, ensure_2d_for_npeet(yhat_mi)])
        I_Y_phi_given_EYhat = conditional_mi(y_test_mi, phi_test_red_mi, cond_E_Yhat, k=k_mi)

        metric1_val = I_Y_phi_given_E - I_Y_phi_given_Yhat
        metric2_val = I_Y_phi_given_Yhat + I_Y_E_given_phi - I_Y_phi_given_EYhat

        # store
        xgb_r2s.append(r2_val); xgb_mses.append(mse_val)
        xgb_m1.append(metric1_val); xgb_m2.append(metric2_val)
        xgb_I_Y_phi_given_E_list.append(I_Y_phi_given_E)
        xgb_I_Y_phi_given_Yhat_list.append(I_Y_phi_given_Yhat)
        xgb_I_Y_E_given_phi_list.append(I_Y_E_given_phi)
        xgb_I_Y_phi_given_EYhat_list.append(I_Y_phi_given_EYhat)

    # aggregate stats
    results_rows.append({
        'model': 'xgboost',
        'r2_mean': float(np.mean(xgb_r2s)), 'r2_std': float(np.std(xgb_r2s, ddof=0)),
        'mse_mean': float(np.mean(xgb_mses)), 'mse_std': float(np.std(xgb_mses, ddof=0)),
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

    # ---------- CatBoost ----------
    print("Tuning CatBoost (CPU)...")
    cb = CatBoostRegressor(verbose=False, random_seed=42, task_type="CPU")
    cb_param_dist = {
        'iterations': [50,100,150,175],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5]
    }
    cb_search = RandomizedSearchCV(cb, cb_param_dist, n_iter=N_ITER_SEARCH, scoring='neg_mean_squared_error',
                                   cv=CV_FOLDS, random_state=42, n_jobs=-1, verbose=0)
    t3 = time.time(); cb_search.fit(X_train_scaled, y_train); t4 = time.time()
    best_cb_params = cb_search.best_params_
    print(f"CatBoost tuned in {t4-t3:.1f}s; best_params={best_cb_params}")

    # Evaluate best CatBoost over multiple seeds
    cb_r2s = []; cb_mses = []; cb_m1 = []; cb_m2 = []
    cb_I_Y_phi_given_E_list = []; cb_I_Y_phi_given_Yhat_list = []; cb_I_Y_E_given_phi_list = []; cb_I_Y_phi_given_EYhat_list = []

    for seed in eval_seeds:
        model_params = dict(best_cb_params)  # copy
        # ensure reproducible seed and CPU task_type
        model_params.update({'random_seed': int(seed), 'verbose': False, 'task_type': 'CPU'})
        model = CatBoostRegressor(**model_params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        r2_val = r2_score(y_test, y_pred); mse_val = mean_squared_error(y_test, y_pred)

        # leaf embeddings
        leaf_idx_test = np.array(model.calc_leaf_indexes(X_test_scaled))
        leaf_idx_train = np.array(model.calc_leaf_indexes(X_train_scaled))
        if leaf_idx_test.ndim == 1: leaf_idx_test = leaf_idx_test.reshape(-1,1)
        if leaf_idx_train.ndim == 1: leaf_idx_train = leaf_idx_train.reshape(-1,1)
        phi_test = leaf_embeddings_from_leaf_indices(leaf_idx_test)
        phi_train = leaf_embeddings_from_leaf_indices(leaf_idx_train)

        # clustering for E (per-run)
        try:
            pca_c = PCA(n_components=min(20, phi_train.shape[1], max(1, phi_train.shape[0]-1)), random_state=seed)
            phi_train_clust = pca_c.fit_transform(phi_train) if phi_train.size and phi_train.shape[1]>1 else phi_train
            phi_test_clust = pca_c.transform(phi_test) if phi_test.size and phi_test.shape[1]>1 else phi_test
        except Exception:
            phi_train_clust = phi_train; phi_test_clust = phi_test

        kmeans = KMeans(n_clusters=n_clusters, random_state=int(seed))
        kmeans.fit(phi_train_clust)
        E_test = kmeans.predict(phi_test_clust)
        E_test_cond = one_hot_encode_labels(E_test) if use_onehot_e else ensure_2d_for_npeet(E_test.astype(float))

        # reduce phi & subsample
        phi_train_red, phi_test_red = reduce_phi_for_mi(phi_train, phi_test, n_components=n_mi_pca)
        idx = subsample_indices(phi_test_red.shape[0], size=mi_sample_size, seed=42)
        phi_test_red_mi = phi_test_red[idx]
        y_test_mi = y_test.values[idx].astype(float)
        yhat_mi = np.asarray(y_pred)[idx].astype(float)
        E_test_cond_mi = E_test_cond[idx] if use_onehot_e else E_test_cond[idx].reshape(-1,1)

        # compute MI components
        I_Y_phi_given_E = conditional_mi(y_test_mi, phi_test_red_mi, E_test_cond_mi, k=k_mi)
        I_Y_phi_given_Yhat = conditional_mi(y_test_mi, phi_test_red_mi, ensure_2d_for_npeet(yhat_mi), k=k_mi)
        I_Y_E_given_phi = conditional_mi(y_test_mi, E_test_cond_mi, phi_test_red_mi, k=k_mi)
        cond_E_Yhat = np.hstack([E_test_cond_mi if use_onehot_e else E_test_cond_mi, ensure_2d_for_npeet(yhat_mi)])
        I_Y_phi_given_EYhat = conditional_mi(y_test_mi, phi_test_red_mi, cond_E_Yhat, k=k_mi)

        metric1_val = I_Y_phi_given_E - I_Y_phi_given_Yhat
        metric2_val = I_Y_phi_given_Yhat + I_Y_E_given_phi - I_Y_phi_given_EYhat

        # store
        cb_r2s.append(r2_val); cb_mses.append(mse_val)
        cb_m1.append(metric1_val); cb_m2.append(metric2_val)
        cb_I_Y_phi_given_E_list.append(I_Y_phi_given_E)
        cb_I_Y_phi_given_Yhat_list.append(I_Y_phi_given_Yhat)
        cb_I_Y_E_given_phi_list.append(I_Y_E_given_phi)
        cb_I_Y_phi_given_EYhat_list.append(I_Y_phi_given_EYhat)

    # aggregate stats
    results_rows.append({
        'model': 'catboost',
        'r2_mean': float(np.mean(cb_r2s)), 'r2_std': float(np.std(cb_r2s, ddof=0)),
        'mse_mean': float(np.mean(cb_mses)), 'mse_std': float(np.std(cb_mses, ddof=0)),
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

    # finalize
    results_df = pd.DataFrame(results_rows).set_index('model')
    csv_path = f'results/{cal_name}_xgb_cb_metric1_metric2.csv'
    results_df.to_csv(csv_path)
    print(f"Saved results CSV: {csv_path}")
    print(results_df[['r2_mean','r2_std','mse_mean','mse_std','metric1_mean','metric1_std','metric2_mean','metric2_std']])
    print("Elapsed time: {:.1f}s".format(time.time() - t0))
    return results_df, common_support, ks_stat

# ---------------- run ----------------
all_results = {}
for comb_name, comb_info in combinations.items():
    try:
        df_res, common_support_flag, ks_stat = run_experiment(comb_name, comb_info['function'], comb_info['description'], X, y,
                                                              k_mi=K_MI, mi_sample_size=MI_SAMPLE_SIZE, n_mi_pca=N_MI_PCA,
                                                              n_clusters=N_CLUSTERS, use_onehot_e=USE_ONEHOT_E)
        all_results[comb_name] = {'table': df_res, 'common_support': common_support_flag, 'ks_stat': ks_stat, 'description': comb_info['description']}
    except Exception as e:
        import traceback
        print(f"Error for {comb_name}: {e}")
        traceback.print_exc()
        continue

# aggregated
if all_results:
    all_tables = []
    for comb_name, info in all_results.items():
        t = info['table'].reset_index().assign(combination=comb_name)
        all_tables.append(t)
    big_df = pd.concat(all_tables, ignore_index=True)
    big_csv = 'results/all_california_xgb_cb_metric1_metric2.csv'
    big_df.to_csv(big_csv, index=False)
    print(f"\nSaved aggregated results to {big_csv}")
    print(big_df[['model','r2_mean','r2_std','mse_mean','mse_std','metric1_mean','metric1_std','metric2_mean','metric2_std']])
else:
    print("No successful experiments.")
print("Done.")
