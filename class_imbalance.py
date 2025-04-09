import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_class_imbalance(file_path, target_column):
    """
    Analyzes class imbalance in a dataset.

    Parameters:
    file_path (str): Path to the dataset (CSV file).
    target_column (str): The name of the target variable.

    Returns:
    None (displays plots and prints statistics).
    """

    # Load dataset
    df = pd.read_csv(file_path)

    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in dataset.")

    # Count class distribution
    class_counts = df[target_column].value_counts()
    total_samples = len(df)
    
    # Compute imbalance ratio (majority/minority class size)
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = round(max_class / min_class, 2) if min_class > 0 else "Undefined (single class present)"

    # Print class distribution
    print("Class Distribution (ECG of size 4096) :\n", class_counts)
    print(f"\nTotal Samples: {total_samples}")
    print(f"Imbalance Ratio (Majority/Minority): {imbalance_ratio}")

    # Plot class distribution
    plt.figure(figsize=(8, 5))
    sns.barplot(x=["NSR", "AF"], y=class_counts.values, palette="coolwarm")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.savefig("class_distribution_640.png")

# Example Usage
analyze_class_imbalance("/mnt/iridia/sehlalou/thesis/data/datasets/dataset_detection_ecg_640.csv", "label")




"""Output: Class Distribution (ECG of size 4096) :
 label
0    1827606
1     524097
Name: count, dtype: int64

Total Samples: 2351703
Imbalance Ratio (Majority/Minority): 3.49
"""