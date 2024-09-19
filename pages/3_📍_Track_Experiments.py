import streamlit as st
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import os

# Set up MLflow tracking URI to point to the local UI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Initialize MLflowClient
client = mlflow.tracking.MlflowClient()

# Fetch all experiments using search_experiments
try:
    experiments = client.search_experiments()
    if not experiments:
        st.warning("No experiments found.")
        experiment_df = pd.DataFrame(columns=["Experiment ID", "Name"])
    else:
        # Create a DataFrame for easier manipulation
        experiment_df = pd.DataFrame([{
            "Experiment ID": exp.experiment_id,
            "Name": exp.name
        } for exp in experiments])
except Exception as e:
    st.error(f"Error fetching experiments: {e}")
    experiment_df = pd.DataFrame(columns=["Experiment ID", "Name"])

# Sidebar options
st.sidebar.title("Options")
compare_algorithms = st.sidebar.checkbox("Compare Algorithms")
selected_algorithms = []

if compare_algorithms:
    algorithms = ['Linear Regression', 'Gradient Boosting', 'ARIMA']
    selected_algorithms = st.sidebar.multiselect("Select Algorithms", algorithms)
    submit_button = st.sidebar.button("Submit")
    
if not experiment_df.empty:
    # Display experiment names for selection
    selected_experiment_name = st.selectbox("Select an Experiment", experiment_df["Name"])

    if selected_experiment_name:
        # Get the experiment ID for the selected name
        selected_experiment_id = experiment_df[experiment_df["Name"] == selected_experiment_name]["Experiment ID"].values[0]

        # Fetch runs for the selected experiment
        try:
            runs = client.search_runs(experiment_ids=[selected_experiment_id])
            if not runs:
                st.warning("No runs found for the selected experiment.")
            else:
                # Convert runs to DataFrame
                run_df = pd.DataFrame([{
                    "Run ID": run.info.run_id,
                    "Params": run.data.params,
                    "Metrics": run.data.metrics,
                    "Start Time": run.info.start_time
                } for run in runs])

                # Display run details
                st.write("## Runs")
                st.dataframe(run_df)

                # Display parameters of the runs
                st.write("### Parameters of the selected runs:")
                st.write(run_df['Params'].apply(pd.Series))

                # Extract metrics into a DataFrame
                metrics = pd.json_normalize(run_df['Metrics'].dropna())
                metrics['Start Time'] = run_df['Start Time']

                # Check available columns
                st.write("Available metrics columns:", metrics.columns)

                # Line chart for accuracy over time
                if 'accuracy' in metrics.columns:
                    fig = px.line(metrics, x='Start Time', y='accuracy', title='Accuracy Over Time')
                    st.plotly_chart(fig)

                # Bar chart for metrics comparison
                metrics_summary = metrics.mean()  # Average metrics for simplicity
                if not metrics_summary.empty:
                    st.bar_chart(metrics_summary)

                # Scatter plot of accuracy vs. training time
                if 'accuracy' in metrics.columns and 'training_time' in metrics.columns:
                    fig, ax = plt.subplots()
                    ax.scatter(metrics['training_time'], metrics['accuracy'])
                    ax.set_xlabel('Training Time')
                    ax.set_ylabel('Accuracy')
                    ax.set_title('Accuracy vs. Training Time')
                    st.pyplot(fig)

                # Histogram of accuracy distribution
                if 'accuracy' in metrics.columns:
                    fig, ax = plt.subplots()
                    ax.hist(metrics['accuracy'], bins=20, edgecolor='black')
                    ax.set_xlabel('Accuracy')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Accuracy')
                    st.pyplot(fig)

                # Pie chart of categorical metrics
                if 'category' in metrics.columns:
                    category_counts = metrics['category'].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
                    ax.set_title('Category Distribution')
                    st.pyplot(fig)

                # Algorithm Comparison
                if compare_algorithms and submit_button:
                    if selected_algorithms:
                        comparison_df = pd.DataFrame()
                        for algo in selected_algorithms:
                            algo_df = metrics[metrics['algorithm'] == algo]
                            algo_df['Algorithm'] = algo
                            comparison_df = pd.concat([comparison_df, algo_df], ignore_index=True)

                        if not comparison_df.empty:
                            comparison_summary = comparison_df.groupby('Algorithm').mean().reset_index()
                            st.write("### Algorithm Comparison")
                            st.dataframe(comparison_summary)

                            # Line chart for algorithm performance
                            fig = px.line(comparison_summary, x='Algorithm', y='accuracy', title='Algorithm Performance')
                            st.plotly_chart(fig)
                    else:
                        st.warning("Please select at least one algorithm to compare.")

                # Example of tracking data drift
                if 'feature' in metrics.columns and 'value' in metrics.columns:
                    drift_summary = metrics.groupby('feature').mean().reset_index()
                    fig = px.bar(drift_summary, x='feature', y='value', title='Feature Value Distribution')
                    st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error fetching runs: {e}")
else:
    st.warning("No experiments available.")
