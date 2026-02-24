"""
Experimental Study: Implicit Hidden Confounding with Common Support Assumption
Test values of the hidden confounder are already observed during training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

# Create directory for saving plots
os.makedirs('confounding_plots', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# 1. Download and prepare real-world data
print("1. Loading California Housing dataset...")
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
# import pdb; pdb.set_trace()
print(f"Dataset shape: {X.shape}")

# 2. Define multiple combinations for implicit confounding
combinations = {
    'Comb1_MedInc_HouseAge': {
        'function': lambda X, y: np.log1p(X['MedInc']) * (X['HouseAge'] / 5)+5*y,
        'description': 'log(MedInc) × HouseAge/5'
    },
    'Comb2_Location_Based': {
        'function': lambda X, y: X['AveRooms']+5*y,
        'description': 'AveRooms + 5 * y'
    }
}

def ensure_common_support(combination_values, train_ratio=0.7):
    """
    Ensure common support: test values are already observed in training
    by using a stratified split based on combination quantiles
    """
    # Create quantile-based strata
    n_strata = 10
    strata = pd.qcut(combination_values, q=n_strata, labels=False, duplicates='drop')
    
    # Initialize masks
    train_mask = np.zeros(len(combination_values), dtype=bool)
    test_mask = np.zeros(len(combination_values), dtype=bool)
    
    # For each stratum, split into train/test
    for stratum in np.unique(strata):
        stratum_indices = np.where(strata == stratum)[0]
        n_stratum_samples = len(stratum_indices)
        n_train = int(n_stratum_samples * train_ratio)
        
        # Shuffle and split within stratum
        np.random.shuffle(stratum_indices)
        train_mask[stratum_indices[:n_train]] = True
        test_mask[stratum_indices[n_train:]] = True
    
    return train_mask, test_mask

def create_frequency_shift(combination_values, train_ratio=0.7):
    """
    Create frequency shift while maintaining common support
    Training set gets balanced distribution, test set gets imbalanced distribution
    """
    # Create balanced training set (equal representation across quantiles)
    n_quantiles = 5
    quantiles = pd.qcut(combination_values, q=n_quantiles, labels=False, duplicates='drop')
    
    train_indices = []
    test_indices = []
    
    for q in range(n_quantiles):
        q_indices = np.where(quantiles == q)[0]
        np.random.shuffle(q_indices)
        
        # Training: balanced (equal samples from each quantile)
        n_train_per_q = int(len(q_indices) * train_ratio / n_quantiles * 2)  # More balanced
        train_indices.extend(q_indices[:n_train_per_q])
        
        # Test: imbalanced (varies by quantile)
        n_test_per_q = int(len(q_indices) * (1 - train_ratio) * (q + 1) / n_quantiles)  # More from higher quantiles
        test_indices.extend(q_indices[n_train_per_q:n_train_per_q + n_test_per_q])
    
    train_mask = np.zeros(len(combination_values), dtype=bool)
    test_mask = np.zeros(len(combination_values), dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, test_mask

# 3. Experiment function for each combination
def run_experiment(comb_name, comb_func, description, X, y):
    print(f"\n{'='*60}")
    print(f"Running experiment for: {comb_name}")
    print(f"Combination: {description}")
    print(f"{'='*60}")
    import pdb; pdb.set_trace()
    # Reset index to avoid indexing issues
    X_reset = X.reset_index(drop=True)
    y_reset = pd.Series(y).reset_index(drop=True)
    
    # Create combination values
    combination_values = comb_func(X_reset, y_reset)
    combination_values = (combination_values - combination_values.mean()) / combination_values.std()
    
    # Create distribution shift with COMMON SUPPORT
    print("Creating distribution shift with common support...")
    train_mask, test_mask = create_frequency_shift(combination_values, train_ratio=0.7)
    
    X_train = X_reset[train_mask].copy().reset_index(drop=True)
    y_train = y_reset[train_mask].reset_index(drop=True)
    X_test = X_reset[test_mask].copy().reset_index(drop=True)
    y_test = y_reset[test_mask].reset_index(drop=True)
    comb_train = combination_values[train_mask]
    comb_test = combination_values[test_mask]
    
    # Verify common support
    train_range = (comb_train.min(), comb_train.max())
    test_range = (comb_test.min(), comb_test.max())
    common_support = (test_range[0] >= train_range[0]) and (test_range[1] <= train_range[1])
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train combination range: [{comb_train.min():.3f}, {comb_train.max():.3f}]")
    print(f"Test combination range: [{comb_test.min():.3f}, {comb_test.max():.3f}]")
    print(f"Common support: {common_support}")
    print(f"Train mean: {comb_train.mean():.3f}, Test mean: {comb_test.mean():.3f}")
    
    if not common_support:
        print("WARNING: Common support assumption violated!")
    
    # Analyze distribution differences
    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(comb_train, comb_test)
    print(f"KS test: statistic={ks_stat:.3f}, p-value={ks_p:.3f}")
    
    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    import pdb; pdb.set_trace()

    # Leaf embeddings
    leaf_indices_test = model.apply(X_test_scaled)
    n_test_samples, n_trees = leaf_indices_test.shape
    
    leaf_embeddings_test = np.zeros((n_test_samples, n_trees))
    for i in range(n_trees):
        tree_leaves = leaf_indices_test[:, i]
        if len(np.unique(tree_leaves)) > 1:
            leaf_embeddings_test[:, i] = (tree_leaves - tree_leaves.min()) / (tree_leaves.max() - tree_leaves.min())
    
    leaf_indices_train = model.apply(X_train_scaled)
    n_train_samples, _ = leaf_indices_train.shape
    
    leaf_embeddings_train = np.zeros((n_train_samples, n_trees))
    for i in range(n_trees):
        tree_leaves = leaf_indices_train[:, i]
        if len(np.unique(tree_leaves)) > 1:
            leaf_embeddings_train[:, i] = (tree_leaves - tree_leaves.min()) / (tree_leaves.max() - tree_leaves.min())
    
    # Dimensionality reduction
    pca = PCA(n_components=2, random_state=42)
    pca_test = pca.fit_transform(leaf_embeddings_test)
    pca_train = pca.transform(leaf_embeddings_train)
    
    # Sample for TSNE if needed
    if n_test_samples > 1000:
        sample_idx_test = np.random.choice(n_test_samples, 1000, replace=False)
        tsne_test = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(leaf_embeddings_test[sample_idx_test])
        comb_test_sampled = comb_test.iloc[sample_idx_test].values if hasattr(comb_test, 'iloc') else comb_test[sample_idx_test]
        pca_test_sampled = pca_test[sample_idx_test]
    else:
        tsne_test = TSNE(n_components=2, random_state=42, perplexity=min(30, n_test_samples-1)).fit_transform(leaf_embeddings_test)
        comb_test_sampled = comb_test.values if hasattr(comb_test, 'values') else comb_test
        pca_test_sampled = pca_test
    
    if n_train_samples > 1000:
        sample_idx_train = np.random.choice(n_train_samples, 1000, replace=False)
        tsne_train = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(leaf_embeddings_train[sample_idx_train])
        comb_train_sampled = comb_train.iloc[sample_idx_train].values if hasattr(comb_train, 'iloc') else comb_train[sample_idx_train]
        pca_train_sampled = pca_train[sample_idx_train]
    else:
        tsne_train = TSNE(n_components=2, random_state=42, perplexity=min(30, n_train_samples-1)).fit_transform(leaf_embeddings_train)
        comb_train_sampled = comb_train.values if hasattr(comb_train, 'values') else comb_train
        pca_train_sampled = pca_train
    
    # Create comprehensive plots with common support focus
    create_common_support_figures(comb_name, description, comb_train, comb_test, 
                                 pca_train_sampled, pca_test_sampled, 
                                 tsne_train, tsne_test, r2, mse,
                                 comb_train_sampled, comb_test_sampled,
                                 common_support, ks_stat, ks_p)
    
    return r2, mse, common_support, ks_stat

def create_common_support_figures(comb_name, description, comb_train, comb_test, 
                                 pca_train, pca_test, tsne_train, tsne_test, r2, mse,
                                 comb_train_sampled, comb_test_sampled,
                                 common_support, ks_stat, ks_p):
    """Create figures emphasizing common support assumption"""
    
    # Figure 1: Distribution Shift with Common Support Evidence
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    # Density plot with common support visualization
    plt.hist(comb_train, bins=30, alpha=0.5, density=True, label='Train', color='blue', edgecolor='black')
    plt.hist(comb_test, bins=30, alpha=0.5, density=True, label='Test', color='red', edgecolor='black')
    
    # Highlight common support region
    support_min = max(comb_train.min(), comb_test.min())
    support_max = min(comb_train.max(), comb_test.max())
    if support_min <= support_max:
        plt.axvspan(support_min, support_max, alpha=0.2, color='green', label='Common Support')
    
    plt.title(f'Distribution Shift with Common Support\n{description}', fontsize=12, fontweight='bold')
    plt.xlabel('Combination Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    # ECDF plot to show distribution differences
    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf_train = ECDF(comb_train)
    ecdf_test = ECDF(comb_test)
    x_vals = np.linspace(min(comb_train.min(), comb_test.min()), 
                         max(comb_train.max(), comb_test.max()), 100)
    plt.plot(x_vals, ecdf_train(x_vals), label='Train ECDF', linewidth=2)
    plt.plot(x_vals, ecdf_test(x_vals), label='Test ECDF', linewidth=2)
    plt.title('Empirical CDF Comparison\n(KS statistic: {:.3f})'.format(ks_stat), 
              fontsize=12, fontweight='bold')
    plt.xlabel('Combination Value')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Quantile-quantile plot
    from scipy.stats import probplot
    qq_train = probplot(comb_train, dist="norm")
    qq_test = probplot(comb_test, dist="norm")
    
    plt.scatter(qq_train[0][0], qq_train[0][1], alpha=0.5, label='Train', s=10)
    plt.scatter(qq_test[0][0], qq_test[0][1], alpha=0.5, label='Test', s=10)
    plt.plot(qq_train[0][0], qq_train[1][0] + qq_train[1][1] * qq_train[0][0], 'r--', alpha=0.8)
    plt.title('Q-Q Plot: Distribution Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.savefig(f'confounding_plots/{comb_name}_01_common_support.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: PCA Visualizations with Common Support Focus
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(pca_train[:, 0], pca_train[:, 1], c=comb_train_sampled, 
                         cmap='BuGn', alpha=0.7, s=30)
    plt.title(f'Train Set PCA', fontsize=16, fontweight='bold')
    plt.xlabel('PC1', fontweight='bold', fontsize=16)
    plt.ylabel('PC2',fontweight='bold', fontsize=16)
    plt.colorbar(scatter, label='Hidden Confounder Value')
    
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(pca_test[:, 0], pca_test[:, 1], c=comb_test_sampled, 
                         cmap='BuGn', alpha=0.7, s=30)
    plt.title(f'Test Set PCA', fontsize=16, fontweight='bold')
    plt.xlabel('PC1', fontweight='bold', fontsize=16)
    plt.ylabel('PC2',fontweight='bold', fontsize=16)
    plt.colorbar(scatter, label='Hidden Confounder Value')
    
    plt.subplot(1, 3, 3)
    # Highlight common support in combined plot
    plt.scatter(pca_train[:, 0], pca_train[:, 1], alpha=0.7, s=30, 
                label='Train', color='red', marker='^')
    plt.scatter(pca_test[:, 0], pca_test[:, 1], alpha=0.7, s=30, 
                label='Test', color='blue', marker='v')
    plt.title('Distribution Shift', 
              fontsize=12, fontweight='bold')
    plt.xlabel('PC1', fontweight='bold')
    plt.ylabel('PC2',fontweight='bold')
    plt.legend(prop={'size': 16})
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{comb_name}_california.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Performance Analysis with Common Support
    plt.figure(figsize=(12, 10))
    
    # Analyze performance across combination quantiles
    n_quantiles = 5
    comb_quantiles = pd.qcut(np.concatenate([comb_train, comb_test]), q=n_quantiles, labels=False)
    train_quantiles = comb_quantiles[:len(comb_train)]
    test_quantiles = comb_quantiles[len(comb_train):]
    
    # Calculate performance by quantile
    quantile_performance = []
    for q in range(n_quantiles):
        train_q_mask = train_quantiles == q
        test_q_mask = test_quantiles == q
        
        if train_q_mask.sum() > 10 and test_q_mask.sum() > 10:  # Ensure enough samples
            # For this demo, we'll use placeholder values
            # In practice, you'd need to store predictions by quantile
            quantile_performance.append({
                'quantile': q,
                'train_samples': train_q_mask.sum(),
                'test_samples': test_q_mask.sum(),
                'train_mean': comb_train[train_q_mask].mean() if train_q_mask.any() else 0,
                'test_mean': comb_test[test_q_mask].mean() if test_q_mask.any() else 0
            })
    
    plt.subplot(2, 2, 1)
    # Distribution by quantile
    quantile_counts_train = [sum(train_quantiles == q) for q in range(n_quantiles)]
    quantile_counts_test = [sum(test_quantiles == q) for q in range(n_quantiles)]
    
    x = np.arange(n_quantiles)
    width = 0.35
    plt.bar(x - width/2, quantile_counts_train, width, label='Train', alpha=0.7)
    plt.bar(x + width/2, quantile_counts_test, width, label='Test', alpha=0.7)
    plt.title('Sample Distribution by Quantile', fontsize=12, fontweight='bold')
    plt.xlabel('Quantile')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    # Mean values by quantile
    if quantile_performance:
        train_means = [q['train_mean'] for q in quantile_performance]
        test_means = [q['test_mean'] for q in quantile_performance]
        plt.bar(x - width/2, train_means, width, label='Train Mean', alpha=0.7)
        plt.bar(x + width/2, test_means, width, label='Test Mean', alpha=0.7)
        plt.title('Mean Combination Value by Quantile', fontsize=12, fontweight='bold')
        plt.xlabel('Quantile')
        plt.ylabel('Mean Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Common support verification
    support_metrics = {
        'Common Support': common_support,
        'KS Statistic': ks_stat,
        'Train Range': comb_train.max() - comb_train.min(),
        'Test Range': comb_test.max() - comb_test.min(),
        'Overlap %': (min(comb_train.max(), comb_test.max()) - max(comb_train.min(), comb_test.min())) / 
                    (max(comb_train.max(), comb_test.max()) - min(comb_train.min(), comb_test.min())) * 100
    }
    
    metrics_names = list(support_metrics.keys())
    metrics_values = list(support_metrics.values())
    
    colors = ['green' if x is True else 'red' if x is False else 'blue' for x in metrics_values[:1]] + ['blue'] * (len(metrics_values)-1)
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    plt.title('Common Support Metrics', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, metrics_values):
        if isinstance(value, bool):
            text_val = 'Yes' if value else 'No'
        else:
            text_val = f'{value:.2f}'
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                text_val, ha='center', va='bottom', fontsize=9)
    
    plt.subplot(2, 2, 4)
    # Summary text
    summary_text = f"""
    COMMON SUPPORT EXPERIMENT
    Combination: {description}
    
    Key Assumption:
    - Test values were observed during training
    - Realistic generalization scenario
    
    Results:
    - R² Score: {r2:.3f}
    - MSE: {mse:.3f}
    - Common Support: {'SATISFIED' if common_support else 'VIOLATED'}
    - Distribution Difference (KS): {ks_stat:.3f}
    
    Training Details:
    - Train samples: {len(comb_train):,}
    - Test samples: {len(comb_test):,}
    - Frequency shift applied
    - Common support maintained
    """
    
    plt.axis('off')
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    # plt.savefig(f'confounding_plots/{comb_name}_03_common_support_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created common support figures for {comb_name}")

# 4. Run experiments for all combinations
results = {}

for comb_name, comb_info in combinations.items():
    try:
        r2_score_, mse_score, common_support, ks_stat = run_experiment(
            comb_name, 
            comb_info['function'], 
            comb_info['description'],
            X, y
        )
        results[comb_name] = {
            'r2': r2_score_,
            'mse': mse_score,
            'common_support': common_support,
            'ks_statistic': ks_stat,
            'description': comb_info['description']
        }
    except Exception as e:
        import traceback
        print(f"Error in {comb_name}: {e}")
        traceback.print_exc()
        continue

# 5. Create comprehensive summary
if results:
    print(f"\n{'='*60}")
    print("CREATING COMPREHENSIVE SUMMARY")
    
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: R² scores with common support indicator
    r2_scores = [results[comb]['r2'] for comb in results.keys()]
    common_supports = [results[comb]['common_support'] for comb in results.keys()]
    comb_names_short = [comb.replace('Comb', '').replace('_', '\n') for comb in results.keys()]
    
    colors = ['green' if sup else 'red' for sup in common_supports]
    bars = axes[0,0].bar(comb_names_short, r2_scores, color=colors, alpha=0.7)
    axes[0,0].set_title('Model Performance with Common Support Indicator', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('R² Score')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add value labels and support indicators
    for bar, score, support in zip(bars, r2_scores, common_supports):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                      f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        support_text = '✓' if support else '✗'
        axes[0,0].text(bar.get_x() + bar.get_width()/2, -0.05, 
                      support_text, ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Plot 2: KS statistics (distribution differences)
    ks_stats = [results[comb]['ks_statistic'] for comb in results.keys()]
    axes[0,1].bar(comb_names_short, ks_stats, color='skyblue', alpha=0.7)
    axes[0,1].set_title('Distribution Differences (KS Statistics)', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('KS Statistic')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Common support success rate
    support_rate = sum(common_supports) / len(common_supports) * 100
    axes[1,0].pie([support_rate, 100-support_rate], labels=['Satisfied', 'Violated'], 
                  autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    axes[1,0].set_title('Common Support Assumption Success Rate', fontsize=14, fontweight='bold')
    
    # Plot 4: Summary statistics
    axes[1,1].axis('off')
    summary_text = f"""
    COMMON SUPPORT EXPERIMENT SUMMARY
    
    Key Finding: Test values of hidden confounder
    were observed during training (common support)
    
    Results Summary:
    - Total combinations tested: {len(results)}
    - Common support satisfied: {sum(common_supports)}/{len(common_supports)}
    - Average R²: {np.mean(r2_scores):.3f}
    - Average KS statistic: {np.mean(ks_stats):.3f}
    
    Research Implications:
    • Realistic hidden confounding scenario
    • Models generalize to seen confounding patterns
    • Frequency shifts rather than support shifts
    • More realistic evaluation of robustness
    
    Best Combination:
    {max(results.items(), key=lambda x: x[1]['r2'])[0]}
    R² = {max(r2_scores):.3f}
    """
    
    axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Hidden Confounding with Common Support Assumption', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # plt.savefig('confounding_plots/00_common_support_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE - COMMON SUPPORT ASSUMPTION")
    print("="*60)
    print("\nResults Summary:")
    for comb_name, result in results.items():
        support_status = "SATISFIED" if result['common_support'] else "VIOLATED"
        print(f"  {comb_name}: R² = {result['r2']:.3f}, Common Support = {support_status}")
    
    print(f"\nCommon support rate: {sum(common_supports)}/{len(common_supports)} combinations")
    print("This represents a more realistic generalization scenario!")
    
else:
    print("No experiments completed successfully.")