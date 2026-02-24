
## Project Overview

This project implements an end-to-end customer segmentation system using 
unsupervised machine learning techniques to identify meaningful customer groups 
and generate actionable business insights.

The system helps businesses:
- Identify high-value customers
- Detect churn-prone segments
- Optimize marketing strategies
- Enable data-driven decision-making


## Problem Statement

Organizations often possess large volumes of customer data without labeled outcomes. 
This project applies clustering algorithms to segment customers based on behavioral, 
financial, and demographic features.


## Dataset Information

- Dataset: Bank Marketing Dataset
- Source: Kaggle
- Records: 5000+ customers
- Features: Demographic, financial, behavioral attributes



## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- VS Code



## Project Structure
customer-segmentation-unsupervised-shravani/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_eda.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_clustering_models.ipynb
│ ├── 05_model_comparison.ipynb
│ └── 06_visualization.ipynb
│
├── reports/
│ ├── final_report.pdf
│ └── business_insights.md
│
├── results/
│ ├── cluster_plots/
│ └── pca_outputs/
│
├── main.py
├── requirements.txt
└── README.md


## Clustering Algorithms Implemented

- K-Means
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Model (GMM)

Cluster optimization was performed using:
- Elbow Method
- Silhouette Score
- Davies–Bouldin Index


## Dimensionality Reduction

- Principal Component Analysis (PCA)
- t-SNE (for non-linear visualization)


## Business Insights

The model identified four distinct customer segments:

1. High-Value Engaged Customers
2. Low-Engagement At-Risk Customers
3. High Contact Low Conversion Customers
4. Stable Moderate-Value Customers

Each segment includes strategic recommendations for marketing and retention.


## How to Run the Project

1. Clone the repository
2. Create virtual environment
3. Install dependencies
4. pip install -r requirements.txt
5. Run notebooks sequentially OR execute



## Conclusion

This project demonstrates how unsupervised machine learning can transform raw 
customer data into strategic business intelligence.

