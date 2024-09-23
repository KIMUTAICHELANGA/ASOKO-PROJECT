import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Helper function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Sidebar to select datasets for comparison
st.sidebar.title("Select Datasets to Compare")
data_folder = 'data'  # Folder containing datasets
available_datasets = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# Dynamically create checkboxes for each dataset
selected_datasets = [dataset for dataset in available_datasets if st.sidebar.checkbox(dataset)]

# Submit button for comparison
if st.sidebar.button('Submit'):
    if len(selected_datasets) < 2:
        st.error("Please select at least two datasets to compare.")
    else:
        # Load the selected datasets into a dictionary
        datasets = {dataset: load_data(os.path.join(data_folder, dataset)) for dataset in selected_datasets}
        
        # Display the selected datasets
        st.title("Dataset Comparison")
        
        # Show sample data for each dataset
        for dataset_name, dataset in datasets.items():
            st.subheader(f"Sample Data: {dataset_name}")
            st.dataframe(dataset.head())

        # Compare the datasets
        st.subheader("Comparison of Basic Statistics")
        
        stats = {}
        for dataset_name, dataset in datasets.items():
            stats[dataset_name] = dataset.describe()

        # Display statistical summaries side by side
        for dataset_name, summary_stats in stats.items():
            st.write(f"### {dataset_name} Statistics")
            st.dataframe(summary_stats)
        
        # Visualize correlations between the features in each dataset
        st.subheader("Correlation Heatmaps for Each Dataset")
        for dataset_name, dataset in datasets.items():
            st.write(f"### Correlation Heatmap: {dataset_name}")
            features = ['Political Stability', 'Economic Environment', 'Regulatory Framework', 'Infrastructure', 'Human Capital', 'Deal Value (USD Millions)']
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(dataset[features].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)

        # Compare feature distributions
        st.subheader("Feature Distributions")
        selected_feature = st.selectbox("Select a feature to compare", ['Political Stability', 'Economic Environment', 'Regulatory Framework', 'Infrastructure', 'Human Capital', 'Deal Value (USD Millions)'])

        fig, ax = plt.subplots(figsize=(10, 6))
        for dataset_name, dataset in datasets.items():
            sns.kdeplot(dataset[selected_feature], label=dataset_name, ax=ax)
        
        plt.title(f"Distribution of {selected_feature} across Datasets")
        plt.legend()
        st.pyplot(fig)

# Main page content displaying baseline features
st.title("Baseline Features for Deal-Making")

# Load the dataset to display features
baseline_file_path = os.path.join(data_folder, 'modeldata.csv')
if os.path.exists(baseline_file_path):
    baseline_data = load_data(baseline_file_path)
else:
    st.error("Baseline dataset 'africa_deal_data.csv' not found in the 'data' folder.")

# Display a brief description of the baseline features
st.subheader("Baseline Features and Target Variable")
st.write("""
The dataset `modeldata.csv` includes the following baseline features used to predict deal value:

- **CPI**: Measures the perception of corruption in a country. Higher CPI values indicate less corruption, which is often associated with a more favorable investment environment.
- **Exports**: Represents the total value of goods and services sold abroad. A higher export value indicates a strong international trade presence, which can boost deal-making activities.
- **GDP**: The total value of goods and services produced in a country. A higher GDP reflects a larger economy, which can influence the volume and value of deals.
- **Per_Capita**: The GDP divided by the population. This measure helps understand the economic conditions on a per-person basis, influencing individual and business investment decisions.
- **FDI_net**: Net foreign direct investment. High levels of FDI suggest that foreign investors are interested in the country’s economic prospects, which can correlate with higher deal-making activities.
- **Inflation**: The rate at which the general level of prices for goods and services is rising. Moderate inflation rates are generally preferred as high inflation can create uncertainty in investment decisions.
- **Political_Stability**: Indicates the likelihood of political unrest. Greater political stability can lead to a more predictable investment environment, encouraging deal-making.

These features have been selected because they offer a comprehensive view of the economic and political environment in a country, which are critical factors in determining deal-making activities.
""")

# Optionally display a sample of the baseline dataset
st.subheader("Sample Data from `africa_deal_data.csv`")
st.dataframe(baseline_data.head())
