# main.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

print("Starting Customer Segmentation Pipeline...")

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv("data/raw/bank_marketing.csv")

# Handle delimiter automatically
if df.shape[1] == 1:
    df = pd.read_csv("data/raw/bank_marketing.csv", sep=';')

print("Dataset loaded successfully.")

# 2. Basic Preprocessing
print("Preprocessing data...")

# Drop missing values (if any)
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category').cat.codes

# Separate features
X = df.drop(columns=['deposit'], errors='ignore')

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Preprocessing completed.")

# 3. Apply KMeans Clustering
print("Applying K-Means clustering...")

kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df['cluster'] = labels

# 4. Evaluate Model
sil_score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {sil_score:.4f}")

# 5. PCA Visualization Data
print("Applying PCA for visualization...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['cluster'] = labels

# 6. Save Outputs
df.to_csv("data/processed/final_clustered_output.csv", index=False)
pca_df.to_csv("results/pca_outputs/pca_visualization_data.csv", index=False)

print("Pipeline executed successfully.")
print("Final clustered dataset saved.")