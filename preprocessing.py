
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data():
    # Load the customer dataset
    df = pd.read_csv("data/customers.csv")

    print("Dataset loaded successfully!")
    print(f"Number of customers: {df.shape[0]}")

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Select features for clustering
    features = df[
        ["Annual Income (k$)", "Spending Score (1-100)"]
    ].copy()

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Convert scaled data back to a DataFrame
    scaled_df = pd.DataFrame(
        scaled_features,
        columns=features.columns
    )

    print("\nScaled features:")
    print(scaled_df.head())

    print("\nScaled feature means:")
    print(scaled_df.mean().round(2))

    return df, features, scaled_df


if __name__ == "__main__":
    df, features, scaled_df = load_data()
