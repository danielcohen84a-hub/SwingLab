import numpy as np
import pandas as pd
import os
import random

def check_dataset():
    x_path = "Data/X_train.npy"
    y_path = "Data/y_train.npy"
    names_path = "Data/feature_names.txt"
    output_path = "TuningResults/Dataset_Snapshot.csv"
    
    if not os.path.exists(x_path):
        print("ERROR: X_train.npy not found. Run Step [4] first!")
        return

    # 1. Load the binary data and the feature names
    X = np.load(x_path)
    y = np.load(y_path)
    
    feature_names = []
    if os.path.exists(names_path):
        with open(names_path, "r") as f:
            feature_names = [line.strip() for line in f.readlines()]
    else:
        # Fallback if names file is missing
        feature_names = [f"Feature_{i}" for i in range(X.shape[2])]

    # 2. Pick 20 random sample indices
    num_samples = X.shape[0]
    sample_indices = random.sample(range(num_samples), min(20, num_samples))
    
    rows = []
    for idx in sample_indices:
        # Each sample is a (5, Features) matrix
        sample_matrix = X[idx] 
        target = y[idx]
        
        # We create a "Long Format" view: 
        # For each sample, we show all 5 swings in its history
        for swing_idx in range(X.shape[1]):
            row_data = {
                "Sample_ID": idx,
                "Lookback_History": f"{swing_idx + 1} of 5 (Oldest to Newest)",
                "Future_Target_To_Predict": target
            }
            # Add all features using their REAL names from the map
            for f_idx, col_name in enumerate(feature_names):
                row_data[col_name] = sample_matrix[swing_idx, f_idx]
            
            rows.append(row_data)

    # 3. Create DataFrame and Save
    snapshot_df = pd.DataFrame(rows)
    
    os.makedirs("TuningResults", exist_ok=True)
    snapshot_df.to_csv(output_path, index=False)
    
    print("=" * 60)
    print("SwingLab: Transparent Dataset Snapshot")
    print("=" * 60)
    print(f"Total Samples in .npy: {num_samples:,}")
    print(f"Randomly Sampled: {len(sample_indices)} Historie(s)")
    print(f"Features Labeled: {len(feature_names)}")
    print(f"Saved to: {output_path}")
    print("-" * 60)
    print("OPEN THE CSV TO SEE:")
    print("1. All columns now have REAL NAMES (RSI, Volume, etc.)")
    print("2. Lookback_History: 1 is the oldest memory, 5 is the most recent.")
    print("3. Future_Target: This is the return of the 6th swing (The Future).")
    print("=" * 60)

if __name__ == "__main__":
    check_dataset()
