import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

def main():
    # Load preprocessed AF data (or use the standardized dataset if available)
    data_file = '/mnt/iridia/sehlalou/thesis/examples/dl/clustering/clustering-ml/af_windows_cleaned.csv'
    df = pd.read_csv(data_file)
    
    # Exclude identifier columns
    feature_cols = [col for col in df.columns if col not in ['label', 'patient', 'record']]
    X = df[feature_cols]
    
    # Compute variances on the original (non-standardized) data
    # (We assume higher variance means more informative in this context.)
    feature_variances = X.var()
    
    # Standardize features for further analysis/selection (but use original X for variance computation)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Correlation-based filtering with variance-based removal
    corr_matrix = pd.DataFrame(X_scaled, columns=feature_cols).corr().abs()
    # Create an upper triangular mask (excluding self-correlations)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = set()
    
    for col in upper.columns:
        for row in upper.index:
            if pd.isna(upper.loc[row, col]):
                continue
            # If the correlation is above threshold and neither feature has been already dropped
            if upper.loc[row, col] > 0.9 and row not in drop_cols and col not in drop_cols:
                # Choose to drop the feature with lower variance
                if feature_variances[row] >= feature_variances[col]:
                    drop_cols.add(col)
                else:
                    drop_cols.add(row)
    
    print("Dropping due to high correlation (keeping feature with higher variance):", drop_cols)
    
    # Retain features not marked for drop
    selected_features = [col for col in feature_cols if col not in drop_cols]
    
 
    
    # Save the selected feature dataset (using standardized data)
    selected_df = pd.DataFrame(X_scaled, columns=feature_cols)[selected_features]
    output_file = '/mnt/iridia/sehlalou/thesis/examples/dl/clustering/clustering-ml/af_windows_selected_features.csv'
    selected_df.to_csv(output_file, index=False)
    print(f"Selected features saved to {output_file}")

if __name__ == "__main__":
    main()
