#!/usr/bin/env python3
import os
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Define folder to save plots
    out_folder = "/mnt/iridia/sehlalou/thesis/examples/dl/clustering/clustering-ml/EDA"
    ensure_dir(out_folder)
    
    # Load the preprocessed AF data file
    data_file = '/mnt/iridia/sehlalou/thesis/examples/dl/clustering/clustering-ml/af_windows_cleaned.csv'
    df = pd.read_csv(data_file)
    
    # Select feature columns (excluding identifier columns)
    feature_cols = [col for col in df.columns if col not in ['label', 'patient', 'record']]
    X = df[feature_cols]

    # Standardize features for visualization and UMAP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- Descriptive Statistics ---
    desc_stats = pd.DataFrame(X_scaled, columns=feature_cols).describe()
    print("Descriptive Statistics of Standardized Features:")
    print(desc_stats)
    
    # --- Combined Histograms for All Features ---
    n_features = len(feature_cols)
    n_cols = math.ceil(math.sqrt(n_features))
    n_rows = math.ceil(n_features / n_cols)

    fig_hist, axs_hist = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axs_hist = axs_hist.flatten()

    for i, col in enumerate(feature_cols):
        sns.histplot(X_scaled[:, i], kde=True, ax=axs_hist[i])
        axs_hist[i].set_title(f"{col}")
        axs_hist[i].set_xlabel(col)
    # Hide unused subplots if any
    for j in range(i+1, len(axs_hist)):
        axs_hist[j].axis('off')
    fig_hist.tight_layout()
    hist_path = os.path.join(out_folder, "combined_feature_histograms.png")
    plt.savefig(hist_path)
    plt.close(fig_hist)
    print(f"Combined histograms saved to: {hist_path}")

    # --- Correlation Heatmap (without cell values) ---
    corr = pd.DataFrame(X_scaled, columns=feature_cols).corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", square=True, cbar=True)
    plt.title("Feature Correlation Heatmap")
    corr_path = os.path.join(out_folder, "correlation_heatmap.png")
    plt.savefig(corr_path)
    plt.close()
    print(f"Correlation heatmap saved to: {corr_path}")

    # --- UMAP Visualization ---
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x="UMAP1", y="UMAP2", data=umap_df, alpha=0.7)
    plt.title("UMAP Projection of AF Windows")
    umap_path = os.path.join(out_folder, "umap_af_windows.png")
    plt.savefig(umap_path)
    plt.close()
    print(f"UMAP projection saved to: {umap_path}")

    # --- PCA Visualization ---  
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x="PC1", y="PC2", data=pca_df, alpha=0.7)
    plt.title("PCA Projection of AF Windows")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    pca_path = os.path.join(out_folder, "pca_af_windows.png")
    plt.savefig(pca_path)
    plt.close()
    print(f"PCA projection saved to: {pca_path}")

    # --- Combined Boxplots for All Features ---
    fig_box, axs_box = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axs_box = axs_box.flatten()
    
    for i, col in enumerate(feature_cols):
        sns.boxplot(y=X_scaled[:, i], ax=axs_box[i], color='skyblue')
        axs_box[i].set_title(f"{col}")
        axs_box[i].set_ylabel(col)
    # Hide any empty subplots
    for j in range(i+1, len(axs_box)):
        axs_box[j].axis('off')
    fig_box.tight_layout()
    boxplot_path = os.path.join(out_folder, "combined_feature_boxplots.png")
    plt.savefig(boxplot_path)
    plt.close(fig_box)
    print(f"Combined boxplots saved to: {boxplot_path}")

if __name__ == "__main__":
    main()
