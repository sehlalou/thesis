import pandas as pd

def main():
    input_file = '/mnt/iridia/sehlalou/thesis/data/datasets/dataset_hrv_300_100.csv'
    df = pd.read_csv(input_file)
    print(f"Total windows loaded: {len(df)}")

    # Check for missing values and basic info
    missing = df.isna().sum()
    print("Missing values per column:\n", missing)
    print("\nData Summary:\n", df.describe())

    # Filter for AF-labeled windows (assuming label==1 corresponds to paroxysmal AF)
    af_df = df[df['label'] == 1]
    print(f"\nAF windows: {len(af_df)}")
    print("Unique labels in filtered data:", af_df['label'].unique())

    # Save the cleaned data for subsequent analysis
    cleaned_output = '/mnt/iridia/sehlalou/thesis/examples/dl/clustering/clustering-ml/af_windows_cleaned.csv'
    af_df.to_csv(cleaned_output, index=False)
    print(f"Cleaned AF data saved to '{cleaned_output}'")

if __name__ == "__main__":
    main()
