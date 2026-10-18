# Customer Segmentation Using K-Means

A machine learning project that groups customers based on their annual income and spending scores using the K-Means clustering algorithm. It includes data preprocessing, cluster analysis, visualizations, and an interactive Streamlit dashboard.

## Project Overview

Understanding customer behavior helps businesses identify different customer groups and make better marketing decisions.

This project uses the Mall Customers dataset to group 200 customers based on two features:
- Annual Income
- Spending Score

The results are presented through charts, summary statistics, and an interactive web dashboard.

## Features

- Customer dataset loading and preprocessing
- Feature scaling using StandardScaler
- Elbow Method for exploring the number of clusters
- K-Means clustering with five clusters
- Customer cluster assignments saved as CSV
- Cluster summary with average age, income, and spending score
- Customer segmentation scatter plot
- Interactive Streamlit dashboard
- Cluster filters and descriptive segment labels
- Downloadable customer data

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit

## Project Structure

```text
CustomerSegmentationAI/
├── data/
│   └── customers.csv
├── outputs/
│   ├── elbow_method.png
│   ├── segmented_customers.csv
│   ├── customer_clusters.png
│   └── cluster_summary.csv
├── app.py
├── main.py
├── preprocessing.py
├── clustering.py
├── visualization.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `main.py` file is included in the structure only if it exists in your project folder.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CustomerSegmentationAI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run customer segmentation

```bash
python clustering.py
```

This generates the Elbow Method graph, segmented customer CSV, and cluster summary. Close the graph window if it opens so the script can finish.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Output Files

| File | Purpose |
|---|---|
| `elbow_method.png` | Helps explore a suitable number of clusters |
| `segmented_customers.csv` | Customer records with cluster assignments |
| `customer_clusters.png` | Customer segmentation scatter plot |
| `cluster_summary.csv` | Statistics for each cluster |

## How It Works

1. Load the customer dataset.
2. Select annual income and spending score as clustering features.
3. Standardize the features.
4. Explore possible cluster counts using the Elbow Method.
5. Train K-Means with five clusters.
6. Save cluster assignments and summary statistics.
7. Visualize customer groups and explore the results through Streamlit.

## Limitations

- The model uses only annual income and spending score.
- The current model uses five clusters; this is a chosen configuration and should be evaluated against the Elbow Method and other clustering quality measures.
- Descriptive segment names are simplified interpretations of average income and spending, not verified customer behavior or predictions.
- The dataset is small and intended for educational demonstration.

## Future Improvements

- Compare different values of K using the Silhouette Score.
- Add more customer features.
- Improve cluster descriptions and business recommendations.
- Deploy the dashboard online.
- Add interactive options to explore clustering configurations.

## Author

**Shreyas Yadav**

AIML Engineering Student | Interested in Machine Learning, Data Science, and AI Projects

---

*Educational machine learning project built with Python and Streamlit.*