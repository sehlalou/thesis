
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import config as cfg
import numpy as np
import time
import multiprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score, accuracy_score



sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")

import hyperparameters as hp
from record import Record


dataset_path = Path(hp.DATASET_PATH, f"dataset_hrv_{cfg.WINDOW_SIZE}_{cfg.TRAINING_STEP}.csv")
dataset = pd.read_csv(dataset_path)


# Verify categorical columns
categorical_columns = dataset.select_dtypes(include=['category', 'object']).columns
if len(categorical_columns) > 0:
    print("The following categorical columns were found:")
    print(categorical_columns)
else:
    print("No categorical columns found.")

true_labels = dataset['label']
X = dataset.drop(columns=["label", "patient", "record"])



# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# Build the Autoencoder
input_dim = scaled_data.shape[1]  # Number of HRV features
encoding_dim = 2  # Dimensionality of the latent space (compressed representation)


input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)  # Latent space layer

decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)


autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


history = autoencoder.fit(scaled_data, scaled_data, 
                          epochs=100, 
                          batch_size=32, 
                          shuffle=True, 
                          validation_split=0.2)


# Plot the training loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("plots/autoencoder_trainingloss")


# Extract the Encoded (Latent Space) Representation
encoder = Model(inputs= input_layer, outputs = encoded)
encoded_data = encoder.predict(scaled_data)

kmeans = KMeans(n_clusters=2, random_state=42)
predicted_labels = kmeans.fit_predict(encoded_data)

accuracy_mapping_1 = accuracy_score(true_labels, predicted_labels)
accuracy_mapping_2 = accuracy_score(true_labels, 1 - predicted_labels)  # Flip labels
best_accuracy = max(accuracy_mapping_1, accuracy_mapping_2)


# Calculate silhouette score
if len(set(predicted_labels)) > 1:  # Ensure there are more than one cluster
    silhouette_avg = silhouette_score(encoded_data, predicted_labels)
else:
    silhouette_avg = -1  # Assign a default value if there's only one cluster
    
    
results_csv_path = Path("clustering_af_nsr.csv")
if results_csv_path.exists():
    results_df = pd.read_csv(results_csv_path)
else:
    # If the file doesn't exist, create a new DataFrame
    results_df = pd.DataFrame(columns=["Algorithm", "Accuracy", "Silhouette Score"])



# Append the new result
new_result = {
    "Algorithm": "Autoencoder + K-means",
    "Accuracy": best_accuracy,
    "Silhouette Score": silhouette_avg
}

results_df.loc[len(results_df.index)] = ["Autoencoder + K-means", best_accuracy, silhouette_avg]

results_df.to_csv(results_csv_path, index=False)
print(f"\nUpdated results saved to {results_csv_path}")

if encoding_dim == 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=predicted_labels, cmap='viridis')
    plt.title("Clustering HRV Features in Latent Space (2D)")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.colorbar(label="Cluster Label")
    plt.savefig("plots/autoencoder_af_nsr_2D")
elif encoding_dim == 3:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2], c=predicted_labels, cmap='viridis')
    ax.set_title("Clustering HRV Features in Latent Space (3D)")
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.set_zlabel("Latent Dimension 3")
    plt.savefig("plots/autoencoder_af_nsr_3D")