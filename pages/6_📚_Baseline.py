import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Helper function to get available datasets
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
