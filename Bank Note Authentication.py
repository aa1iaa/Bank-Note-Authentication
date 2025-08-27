import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = 'D:/projects/k means/banknote_authentication.csv'
df = pd.read_csv(file_path)
print(df.head())

# Summary statistics
mean_values = df.mean()
std_values = df.std()
print("Mean values:\n", mean_values)
print("\nStandard Deviation values:\n", std_values)

# Visualization
sns.scatterplot(x=df['V1'], y=df['V2'])
plt.title("Scatter plot of V1 vs V2")
plt.xlabel("Feature V1")
plt.ylabel("Feature V2")
plt.show()

# Scale features
scaler = StandardScaler()
X = df.drop('Class', axis=1) if 'Class' in df.columns else df
scaled_data = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans.fit(scaled_data)

# Add cluster labels
df['Cluster'] = kmeans.labels_

# Cluster visualization
sns.scatterplot(x=df['V1'], y=df['V2'], hue=df['Cluster'], palette='Set2')
plt.title("K-Means Clusters")
plt.xlabel("Feature V1")
plt.ylabel("Feature V2")
plt.show()
