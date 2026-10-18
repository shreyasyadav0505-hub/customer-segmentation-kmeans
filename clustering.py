
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from preprocessing import load_data


def find_optimal_clusters(scaled_features):
    """Use the Elbow Method to help choose the number of clusters."""
    inertias = []
    cluster_range = range(1, 11)

    for k in cluster_range:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        model.fit(scaled_features)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(cluster_range), inertias, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(list(cluster_range))
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("outputs/elbow_method.png", dpi=300)
    print("\nElbow Method graph saved.")

    plt.show()

    return inertias


def train_kmeans(scaled_features, n_clusters=5):
    """Train the K-Means model and assign a cluster to each customer."""
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(scaled_features)

    print(f"\nK-Means trained with K = {n_clusters}")
    print(f"Number of customers: {len(labels)}")

    return model, labels


def save_results(df, labels):
    """Save every customer along with their cluster assignment."""
    results = df.copy()
    results["Cluster"] = labels

    output_path = "outputs/segmented_customers.csv"
    results.to_csv(output_path, index=False)

    print(f"\nCustomer segmentation saved to: {output_path}")

    print("\nCustomers in each cluster:")
    print(results["Cluster"].value_counts().sort_index())

    return results


def analyze_clusters(results):
    """Calculate statistics to understand each customer segment."""
    summary = results.groupby("Cluster").agg(
        Customer_Count=("CustomerID", "count"),
        Average_Age=("Age", "mean"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending=("Spending Score (1-100)", "mean")
    ).round(2)

    print("\n========== CUSTOMER SEGMENT ANALYSIS ==========")
    print(summary)

    output_path = "outputs/cluster_summary.csv"
    summary.to_csv(output_path)

    print(f"\nCluster analysis saved to: {output_path}")

    return summary


if __name__ == "__main__":
    # Step 1: Load and scale the dataset
    df, features, scaled_features = load_data()

    # Step 2: Generate the Elbow Method graph
    print("\nStarting Elbow Method...")
    find_optimal_clusters(scaled_features)

    # Step 3: Train K-Means with five clusters
    print("\nStarting K-Means training...")
    model, labels = train_kmeans(
        scaled_features,
        n_clusters=5
    )

    # Step 4: Save the customer cluster assignments
    results = save_results(df, labels)

    # Step 5: Analyze the customer segments
    analyze_clusters(results)

    print("\nAll clustering steps completed successfully!")
