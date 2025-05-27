import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# 1. Load all datasets from 'data' folder
data_dir = 'data'
datasets = {}
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        datasets[file] = pd.read_csv(os.path.join(data_dir, file))

for name, df in datasets.items():
    print(f"\n--- Dataset: {name} ---\n")
    print("Shape:", df.shape)
    print("First rows:\n", df.head())

    # 1a. Statistical description
    print("\nStatistical Description:")
    print(df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T)

    # 1b. Class distribution (assuming last column is target)
    target_col = df.columns[-1]
    if df[target_col].dtype == 'object' or len(df[target_col].unique()) < 20:
        print("\nClass Distribution:")
        print(df[target_col].value_counts(normalize=True))
        sns.countplot(x=target_col, data=df)
        plt.title(f"Class Distribution in {name}")
        plt.show()

    # 1c. Outlier detection (IQR method)
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_info[col] = outliers
    print("\nOutliers per feature (IQR method):")
    print(outlier_info)

    # 2a. Missing values analysis
    missing = df.isnull().sum()
    print("\nMissing values per feature:")
    print(missing[missing > 0])
    plt.figure(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title(f"Missing Values Heatmap in {name}")
    plt.show()

    # 2b. Imputation (KNN for numeric, mode for categorical)
    imputed_df = df.copy()
    if missing.sum() > 0:
        # Numeric
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=3)
            imputed_df[numeric_cols] = imputer.fit_transform(imputed_df[numeric_cols])
        # Categorical
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            mode = imputed_df[col].mode()[0]
            imputed_df[col].fillna(mode, inplace=True)
        print("Imputation applied (KNN for numeric, mode for categorical).")

    # 2c. Scaling
    scalers = {
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler(),
        'Robust': RobustScaler(),
        'Log': PowerTransformer(method='yeo-johnson')
    }
    scaled_data = {}
    for scaler_name, scaler in scalers.items():
        try:
            scaled = scaler.fit_transform(imputed_df[numeric_cols])
            scaled_data[scaler_name] = pd.DataFrame(scaled, columns=numeric_cols)
            # Plot before/after scaling for first 2 features
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(imputed_df[numeric_cols[0]], kde=True)
            plt.title(f"{numeric_cols[0]} Original")
            plt.subplot(1, 2, 2)
            sns.histplot(scaled_data[scaler_name][numeric_cols[0]], kde=True)
            plt.title(f"{numeric_cols[0]} {scaler_name} Scaled")
            plt.suptitle(f"Scaling Comparison: {scaler_name}")
            plt.show()
        except Exception as e:
            print(f"Scaling {scaler_name} failed: {e}")

    # 2d. Feature selection (remove low variance)
    selector = VarianceThreshold(threshold=0.01)
    selected = selector.fit_transform(imputed_df[numeric_cols])
    print(f"Features kept after low-variance removal: {np.array(numeric_cols)[selector.get_support()]}")

    # 2e. Train/val/test split
    X = imputed_df.drop(target_col, axis=1)
    y = imputed_df[target_col]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    print(f"Train/Val/Test sizes: {X_train.shape}, {X_val.shape}, {X_test.shape}")

    # 3. Dimensionality Reduction
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train.select_dtypes(include=np.number))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train.astype('category').cat.codes, cmap='viridis', alpha=0.5)
    plt.title("PCA Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    # LDA (if more than 1 class)
    if len(np.unique(y_train)) > 1:
        lda = LDA(n_components=2)
        try:
            X_lda = lda.fit_transform(X_train.select_dtypes(include=np.number), y_train)
            plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_train.astype('category').cat.codes, cmap='coolwarm', alpha=0.5)
            plt.title("LDA Projection")
            plt.xlabel("LD1")
            plt.ylabel("LD2")
            plt.show()
        except Exception as e:
            print("LDA failed:", e)

    # t-SNE (on a sample for speed)
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_sample = X_train.select_dtypes(include=np.number).sample(n=min(500, X_train.shape[0]), random_state=42)
        y_sample = y_train.loc[X_sample.index]
        X_tsne = tsne.fit_transform(X_sample)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample.astype('category').cat.codes, cmap='Spectral', alpha=0.5)
        plt.title("t-SNE Projection")
        plt.show()
    except Exception as e:
        print("t-SNE failed:", e)

    print("\n--- End of analysis for", name, "---\n")