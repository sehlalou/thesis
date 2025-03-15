#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import config as cfg
import numpy as np
import time
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")

import hyperparameters as hp
from record import Record



def load_and_preprocess_data(dataset_path):
    """Load dataset and preprocess (drop columns and standardize)."""
    
    dataset = pd.read_csv(dataset_path)
    #if not af_nsr:
    #    dataset = dataset[dataset['label'] == 1] 
  
    X = dataset.drop(columns=["patient"])
    
    #X = dataset.drop(columns=["patient", "record", "label"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled


def build_autoencoder(input_dim, latent_dim):
    """Build and compile the autoencoder model."""
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(latent_dim, activation='relu')(encoded)

    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Models
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder, encoder

def train_autoencoder(autoencoder, data, epochs=50, batch_size=32):
    """Train the autoencoder."""
    autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, verbose=1)

def apply_clustering(latent_data, n_clusters):
    """Apply KMeans clustering on latent space data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(latent_data)
    silhouette = silhouette_score(latent_data, clusters)
    print(f"Silhouette Score: {silhouette}")
    return clusters, silhouette


def save_results(clusters, silhouette, output_dir):
    """Save clustering results to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame({
        "Cluster": clusters,
        "Silhouette_Score": [silhouette] * len(clusters)
    })
    results_path = Path(output_dir, "clustering_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")



def main():
    dataset_path = "data/datasets/aggregated_af_episodes.csv"
    output_dir = "plots/af/af_episodes/autoencoder/results"
    latent_dim = 10
    n_clusters = 2
    epochs = 50
    batch_size = 32

    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(dataset_path)

    print("Building autoencoder...")
    autoencoder, encoder = build_autoencoder(input_dim=data.shape[1], latent_dim=latent_dim)

    print("Training autoencoder...")
    train_autoencoder(autoencoder, data, epochs=epochs, batch_size=batch_size)

    print("Encoding data to latent space...")
    latent_data = encoder.predict(data)

    print("Applying clustering...")
    clusters, silhouette = apply_clustering(latent_data, n_clusters=n_clusters)

    print("Saving results...")
    save_results(clusters, silhouette, output_dir)
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()