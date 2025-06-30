import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def run_clustering_with_plot(filenames, data_path="data/", save_path="static/plot.png"):
    if len(filenames) != 3:
        return {"error": "Pilih tepat 3 file untuk diproses."}

    dataframes = []
    for name in filenames:
        try:
            df = pd.read_csv(f"{data_path}/{name}")
            df = df[['Date', 'Close', 'Volume']]
            df['Saham'] = name.split('.')[0]
            dataframes.append(df)
        except Exception as e:
            return {"error": f"Gagal membaca {name}: {str(e)}"}

    df_all = pd.concat(dataframes)
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    df_pivot = df_all.pivot(index='Date', columns='Saham', values='Close').dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_pivot)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(pca_result)
    inertia = kmeans.inertia_

    # Mapping nama cluster
    cluster_labels = {
    0: "Cluster Stabil",
    1: "Cluster Sedang",
    2: "Cluster Volatil"
    }


    # Plot hasil clustering
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis')
    plt.title('Hasil Clustering PCA')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    # Hitung proporsi cluster
     # Proporsi ukuran cluster
    df_result = pd.DataFrame(df_pivot)
    df_result['Cluster ID'] = labels
    df_result['Cluster'] = df_result['Cluster ID'].map(cluster_labels)
    df_result['Saham Dominan'] = df_pivot.idxmax(axis=1)
    cluster_counts = df_result['Cluster'].value_counts()

    total = cluster_counts.sum()
    proportions = {
        cluster: f"{(count/total)*100:.2f}%" for cluster, count in cluster_counts.items()
        }

        # Alokasi saham per cluster
    cluster_saham = pd.crosstab(df_result['Cluster'], df_result['Saham Dominan'], normalize='index') * 100
    saham_allocation = cluster_saham.round(2).to_dict(orient='index')

    return {
        "success": True,
        "inertia": inertia,
        "proportions": proportions,
        "allocation": saham_allocation
    }