import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from sklearn.preprocessing import StandardScaler


sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")

import hyperparameters as hp
from record import Record


def load_dataset():
    dataset_path = Path(hp.DATASET_PATH, f"dataset_hrv_{cfg.WINDOW_SIZE}_{cfg.TRAINING_STEP}.csv")
    df = pd.read_csv(dataset_path)  
    df.drop(columns=["label", "patient", "record"], inplace=True)
    return df



def main():
    df = load_dataset()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)


    pca = PCA(n_components=2)  # 2D projection
    pca_components = pca.fit_transform(df_scaled)


    pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=df['label'], cmap='coolwarm')
    plt.title('PCA: HRV data projection in 2D')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(label='Label (AF or NSR)')
    plt.show()

if __name__ == "__main__":
    main()
