import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,Dataset
import os
import math

ECG_FOLDER_PATH = "/mnt/iridia/sehlalou/thesis/ECGs"
SAMPLING_RATE = 200
NUM_EPOCHS = 50
HEARTBEATS_FILE_PATH = "/mnt/iridia/sehlalou/thesis/heartbeats"

# Fonction de filtrage du signal ECG
def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=200, order=2):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Fonction pour afficher les signaux ECG
def plot(ecg_data):
    plt.plot(ecg_data)
    plt.title("ECG Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

# Calcul de la durée du signal ECG
def duration(ecg_data):
    return ecg_data.shape[0] / SAMPLING_RATE

# Autoencodeur LSTM pour l'ECG
class ECGLSTMAutoencoder(nn.Module):
    def __init__(self, input_channels=2):
        super(ECGLSTMAutoencoder, self).__init__()
        
        # Encodeur
        self.encoder_lstm1 = nn.LSTM(input_size=input_channels, hidden_size=128, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        
        # Décodeur
        self.decoder_lstm1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, input_channels)

    def forward(self, x):
        x, _ = self.encoder_lstm1(x)
        encoded, _ = self.encoder_lstm2(x)
        
        x, _ = self.decoder_lstm1(encoded)
        x, _ = self.decoder_lstm2(x)
        reconstructed = self.fc(x)
        
        return reconstructed, encoded



def extract_heartbeats(epochs):
    heartbeats = []
    for key in epochs.keys():
        heartbeat = epochs[key]["Signal"].to_numpy()
        if heartbeat.size > 0:  # Only add non-empty heartbeats
            heartbeats.append(heartbeat)
    
    heartbeats_array = np.array(heartbeats)
    return heartbeats_array


# Assuming necessary constants and functions like bandpass_filter, SAMPLING_RATE, etc. are defined.

# Process all ECG files and write to a CSV incrementally
def process_all_ecg_files(ecg_folder_path, heartbeats_file_path):
    first_file = True  # Flag to determine if header should be written
    total_files = len([f for f in os.listdir(ecg_folder_path) if f.endswith(".h5")])  # Total number of .h5 files
    processed_files = 0  # Counter for processed files
    row_counter = 0  # To keep track of the row count

    # Iterate through all the files in the folder
    for file_name in os.listdir(ecg_folder_path):
        if file_name.endswith(".h5"):  # Check if it's an ECG file
            file_path = os.path.join(ecg_folder_path, file_name)
            print(f"Processing {file_path}...")

            # Process the ECG file
            with h5py.File(file_path, "r") as f:
                ecg_data = f["ecg"][:]  # Load ECG data

            # Filtering and processing
            ecg_lead1 = bandpass_filter(ecg_data[6000:20000, 0], fs=SAMPLING_RATE)
            ecg_signals, info = nk.ecg_process(ecg_lead1, sampling_rate=SAMPLING_RATE)
            r_peaks = info["ECG_R_Peaks"]
            cleaned_ecg = ecg_signals["ECG_Clean"]

            # Segment the heartbeats
            epochs = nk.ecg_segment(cleaned_ecg, rpeaks=r_peaks, sampling_rate=SAMPLING_RATE)
            heartbeats = extract_heartbeats(epochs)

            print(heartbeats.shape)
            # Convert to DataFrame and save incrementally
            df = pd.DataFrame(heartbeats)
           

            df = df.drop(df.index[-1])
            df = df.dropna()
            # If it's the first file, write with header, otherwise append
            if first_file:
                df.to_csv(heartbeats_file_path, index=False, mode="w", header=True)  # First file with header
                first_file = False
            else:
                df.to_csv(heartbeats_file_path, index=False, mode="a", header=False)  # Append subsequent files without header


            # Update the row_counter based on the number of rows in this DataFrame
            row_counter += len(df)

            # Increment the processed file counter
            processed_files += 1
            if processed_files == 1:
                break
            print(f"Saved {file_name} heartbeats to {heartbeats_file_path}")
            print(f"Processed {processed_files}/{total_files} files.")  # Show progress
            print(f"Row count after {file_name}: {row_counter}")  # Display row count after each file

    
    print("Processing complete.")




class ECGDataset(Dataset):
    def __init__(self, heartbeat_file, chunk_size=1):
        self.csv_path = heartbeat_file
        self.chunk_size = chunk_size
        self.data_generator = pd.read_csv(heartbeat_file, chunksize=chunk_size)
        self.total_rows = sum(1 for _ in open(heartbeat_file)) - 1  # Subtract header
        self.num_chunks = math.ceil(self.total_rows / self.chunk_size)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.data_generator)
            # Convert the DataFrame values to float32
            chunk = chunk.astype(np.float32)
            return torch.tensor(chunk.values, dtype=torch.float32).unsqueeze(-1)
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, index):
        # Read only the chunk corresponding to this index.
        chunk = pd.read_csv(self.csv_path, skiprows=index * self.chunk_size + 1,
                            nrows=self.chunk_size)

        # Convert the data to float32
        chunk = chunk.astype(np.float32)
        return torch.tensor(chunk.values, dtype=torch.float32).unsqueeze(-1)



def main():
    # Process ECG files to create the heartbeat CSV file.
    process_all_ecg_files(ECG_FOLDER_PATH, HEARTBEATS_FILE_PATH)
    
    # Create the dataset and dataloader.
    dataset = ECGDataset(HEARTBEATS_FILE_PATH)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Instantiate the autoencoder.
    autoencoder = ECGLSTMAutoencoder(input_channels=1)
    
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)
    
    # Define the loss function and optimizer.
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    
    # Training loop.
    autoencoder.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch in dataloader:
            if batch is None:
                continue
            # Original batch shape: [batch_size, chunk_size, heartbeat_length, 1]
            # Flatten batch and chunk dimensions to get shape: [batch_size * chunk_size, heartbeat_length, 1]
            B, chunk_size, seq_length, _ = batch.shape
            batch = batch.view(B * chunk_size, seq_length, 1).to(device)
            
            optimizer.zero_grad()
            reconstructed, encoded = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss / len(dataloader):.6f}")
    
    # Extract latent representations for clustering.
    autoencoder.eval()
    latent_features = []
    with torch.no_grad():
        for batch in dataloader:
            B, chunk_size, seq_length, _ = batch.shape
            batch = batch.view(B * chunk_size, seq_length, 1).to(device)
            _, encoded = autoencoder(batch)
            # Extract the representation from the last time step.
            latent = encoded[:, -1, :]  # shape: [batch_size * chunk_size, latent_dim]
            latent_features.append(latent.cpu().numpy())
    latent_features = np.concatenate(latent_features, axis=0)
    
    # Clustering using KMeans.
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_features)
    
    print("Cluster labels:", cluster_labels)

if __name__ == "__main__":
    main()


