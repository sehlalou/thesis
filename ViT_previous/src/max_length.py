import pandas as pd

# Path to the large CSV file
CSV_FILE_PATH = "/mnt/iridia/sehlalou/thesis/rrr_segments_all_RR.csv"

CHUNK_SIZE = 10**6  # 1 million rows per iteration

# Variables to track the maximum segment
max_length = 0
max_record = None  # Store details of the longest segment

# Read CSV in chunks
for chunk in pd.read_csv(CSV_FILE_PATH, chunksize=CHUNK_SIZE):
    # Compute segment length
    chunk["segment_length"] = chunk["global_end_index"] - chunk["global_start_index"]
    
    # Get the maximum in the current chunk
    chunk_max_idx = chunk["segment_length"].idxmax()
    chunk_max_length = chunk.loc[chunk_max_idx, "segment_length"]

    # Update global max if needed
    if chunk_max_length > max_length:
        max_length = chunk_max_length
        max_record = chunk.loc[chunk_max_idx]

# Display the longest segment
print(f"Maximum segment length: {max_length / 200} seconds")
print(f"Details:\n{max_record}")
