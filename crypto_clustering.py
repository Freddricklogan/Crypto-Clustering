# Cryptocurrency Clustering Analysis
import pandas as pd
import hvplot.pandas
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path

print("Cryptocurrency Clustering Analysis")

# Generate sample data for demonstration
# In a real scenario, you would load from a CSV file
def generate_sample_data():
    # Create sample cryptocurrency data
    np.random.seed(42)
    n_samples = 40
    
    # Generate sample cryptos
    cryptos = [f"Crypto_{i}" for i in range(1, n_samples+1)]
    
    # Generate random metrics
    price_change_24h = np.random.uniform(-15, 15, n_samples)
    price_change_7d = np.random.uniform(-25, 25, n_samples)
    price_change_30d = np.random.uniform(-50, 50, n_samples)
    price_change_60d = np.random.uniform(-70, 70, n_samples)
    price_change_200d = np.random.uniform(-150, 150, n_samples)
    price_change_1y = np.random.uniform(-200, 200, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'coin_id': cryptos,
        'price_change_percentage_24h': price_change_24h,
        'price_change_percentage_7d': price_change_7d,
        'price_change_percentage_30d': price_change_30d,
        'price_change_percentage_60d': price_change_60d,
        'price_change_percentage_200d': price_change_200d,
        'price_change_percentage_1y': price_change_1y
    })
    
    return df.set_index('coin_id')

# Load or generate data
print("Loading cryptocurrency data...")
try:
    # Try to load from CSV file if available
    file_path = Path("Resources/crypto_market_data.csv")
    df_market_data = pd.read_csv(file_path, index_col="coin_id")
    print("Data loaded from CSV file")
except FileNotFoundError:
    # Generate sample data if file not found
    df_market_data = generate_sample_data()
    print("Sample data generated")

# Display the data
print("\nCryptocurrency Market Data (First 5 rows):")
print(df_market_data.head())

# Normalize the data using StandardScaler
print("\nNormalizing data...")
scaler = StandardScaler()
df_market_data_scaled = scaler.fit_transform(df_market_data)

# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    df_market_data_scaled,
    columns=df_market_data.columns,
    index=df_market_data.index
)

print("\nScaled Data (First 5 rows):")
print(df_market_data_scaled.head())

# Find the best value for k using original scaled data
print("\nFinding the best value for k...")
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_market_data_scaled)
    inertia.append(kmeans.inertia_)

# Create elbow curve plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Curve for Original Scaled Data')
plt.grid(True)
plt.tight_layout()
plt.savefig('images/elbow_curve_original.png')
print("Saved elbow curve plot to 'images/elbow_curve_original.png'")

# Determine best k from elbow curve
# For demonstration, we'll use k=4
k = 4
print(f"\nSelected k = {k} for clustering")

# Cluster cryptocurrencies with K-means using original scaled data
kmeans = KMeans(n_clusters=k, random_state=42)
df_market_data['Cluster'] = kmeans.fit_predict(df_market_data_scaled)

# Create scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    df_market_data['price_change_percentage_24h'],
    df_market_data['price_change_percentage_7d'],
    c=df_market_data['Cluster'],
    cmap='viridis',
    s=100,
    alpha=0.8
)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Price Change Percentage (24h)')
plt.ylabel('Price Change Percentage (7d)')
plt.title('Cryptocurrency Clusters')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/crypto_clusters_original.png')
print("Saved cluster scatter plot to 'images/crypto_clusters_original.png'")

# Optimize clusters with PCA
print("\nReducing dimensions with PCA...")
pca = PCA(n_components=3)
market_data_pca = pca.fit_transform(df_market_data_scaled)

# Create a DataFrame with PCA data
df_market_data_pca = pd.DataFrame(
    market_data_pca,
    columns=['PC1', 'PC2', 'PC3'],
    index=df_market_data.index
)

print("\nPCA Data (First 5 rows):")
print(df_market_data_pca.head())

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_
print(f"\nExplained variance ratio: {explained_variance}")
print(f"Total explained variance: {sum(explained_variance)*100:.2f}%")

# Find the best value for k using PCA data
print("\nFinding the best value for k using PCA data...")
inertia_pca = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_market_data_pca)
    inertia_pca.append(kmeans.inertia_)

# Create elbow curve plot for PCA data
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_pca, 'o-', color='orange')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Curve for PCA Data')
plt.grid(True)
plt.tight_layout()
plt.savefig('images/elbow_curve_pca.png')
print("Saved PCA elbow curve plot to 'images/elbow_curve_pca.png'")

# Cluster cryptocurrencies with K-means using PCA data
k_pca = 4  # For demonstration
print(f"\nSelected k = {k_pca} for PCA clustering")

kmeans_pca = KMeans(n_clusters=k_pca, random_state=42)
df_market_data_pca['Cluster'] = kmeans_pca.fit_predict(df_market_data_pca)

# Create scatter plot for PCA clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    df_market_data_pca['PC1'],
    df_market_data_pca['PC2'],
    c=df_market_data_pca['Cluster'],
    cmap='viridis',
    s=100,
    alpha=0.8
)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Cryptocurrency Clusters with PCA')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/crypto_clusters_pca.png')
print("Saved PCA cluster scatter plot to 'images/crypto_clusters_pca.png'")

print("\nAnalysis complete!")
