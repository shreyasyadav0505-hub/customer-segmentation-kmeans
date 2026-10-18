
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation Dashboard")
st.write(
    "Explore customer groups using Machine Learning "
    "and the K-Means clustering algorithm."
)

# --------------------------------------------------
# 2. LOAD CUSTOMER DATA
# --------------------------------------------------
data_path = Path("outputs/segmented_customers.csv")

if not data_path.exists():
    st.error(
        "Segmented customer data not found. "
        "Run python clustering.py first."
    )
    st.stop()

df = pd.read_csv(data_path)

# --------------------------------------------------
# 3. CREATE DESCRIPTIVE SEGMENT NAMES
# --------------------------------------------------
cluster_stats = df.groupby("Cluster").agg(
    avg_income=("Annual Income (k$)", "mean"),
    avg_spending=("Spending Score (1-100)", "mean")
)

income_mid = df["Annual Income (k$)"].median()
spending_mid = df["Spending Score (1-100)"].median()


def describe_segment(row):
    income = row["avg_income"]
    spending = row["avg_spending"]

    if income >= income_mid and spending >= spending_mid:
        return "High Income - High Spending"
    elif income >= income_mid and spending < spending_mid:
        return "High Income - Low Spending"
    elif income < income_mid and spending >= spending_mid:
        return "Low Income - High Spending"
    else:
        return "Low Income - Low Spending"


cluster_stats["Segment Name"] = cluster_stats.apply(
    describe_segment,
    axis=1
)

segment_names = cluster_stats["Segment Name"].to_dict()
df["Segment Name"] = df["Cluster"].map(segment_names)

# --------------------------------------------------
# 4. SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

clusters = sorted(df["Cluster"].unique().tolist())

selected_clusters = st.sidebar.multiselect(
    "Select customer clusters",
    options=clusters,
    default=clusters,
    format_func=lambda cluster: (
        f"Cluster {cluster}: {segment_names[cluster]}"
    )
)

filtered_df = df[df["Cluster"].isin(selected_clusters)]

# --------------------------------------------------
# 5. CUSTOMER OVERVIEW
# --------------------------------------------------
st.subheader("Customer Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(filtered_df))

col2.metric(
    "Average Annual Income",
    f"${filtered_df['Annual Income (k$)'].mean():.1f}k"
    if not filtered_df.empty else "N/A"
)

col3.metric(
    "Average Spending Score",
    f"{filtered_df['Spending Score (1-100)'].mean():.1f}"
    if not filtered_df.empty else "N/A"
)

# --------------------------------------------------
# 6. CLUSTER DISTRIBUTION
# --------------------------------------------------
st.subheader("Customers in Each Cluster")

if not filtered_df.empty:

    cluster_counts = (
        filtered_df.groupby(["Cluster", "Segment Name"])
        .size()
        .reset_index(name="Customers")
    )

    cluster_counts["Display Name"] = (
        "Cluster "
        + cluster_counts["Cluster"].astype(str)
        + ": "
        + cluster_counts["Segment Name"]
    )

    st.bar_chart(
        cluster_counts,
        x="Display Name",
        y="Customers"
    )

    # ----------------------------------------------
    # 7. CUSTOMER SCATTER PLOT
    # ----------------------------------------------
    st.subheader("Customer Segmentation Visualization")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=filtered_df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Segment Name",
        palette="viridis",
        s=100,
        ax=ax
    )

    ax.set_title("Customer Groups Based on Income and Spending")
    ax.set_xlabel("Annual Income (thousands of dollars)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.grid(alpha=0.3)
    ax.legend(
        title="Customer Segment",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

else:
    st.warning("Select at least one cluster to display charts.")

# --------------------------------------------------
# 8. CUSTOMER SEGMENT SUMMARY
# --------------------------------------------------
st.subheader("Customer Segment Summary")

summary = (
    filtered_df.groupby(["Cluster", "Segment Name"])
    .agg(
        Customer_Count=("CustomerID", "count"),
        Average_Age=("Age", "mean"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending=("Spending Score (1-100)", "mean")
    )
    .round(2)
    .reset_index()
)

st.dataframe(summary, use_container_width=True, hide_index=True)

# --------------------------------------------------
# 9. CUSTOMER RECORDS
# --------------------------------------------------
st.subheader("Customer Records")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 10. DOWNLOAD FILTERED CUSTOMER DATA
# --------------------------------------------------
csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Customer Data as CSV",
    data=csv_data,
    file_name="segmented_customers.csv",
    mime="text/csv"
)

# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------
st.caption(
    "Built with Python, Pandas, Scikit-learn, "
    "Matplotlib, Seaborn and Streamlit."
)
