
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_clusters():
    # Load the segmented customer data
    df = pd.read_csv("outputs/segmented_customers.csv")

    print("Segmented customer data loaded successfully!")

    # Display the number of customers in each cluster
    print("\nCustomers in each cluster:")
    print(df["Cluster"].value_counts().sort_index())

    # Create the scatter plot
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="viridis",
        s=100
    )

    plt.title("Customer Segmentation Using K-Means")
    plt.xlabel("Annual Income (thousands of dollars)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend(title="Customer Cluster")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the graph
    plt.savefig("outputs/customer_clusters.png", dpi=300)

    print("\nGraph saved to outputs/customer_clusters.png")

    # Display the graph
    plt.show()


if __name__ == "__main__":
    visualize_clusters()
