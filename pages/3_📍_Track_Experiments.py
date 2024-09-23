import streamlit as st
import mlflow
import pandas as pd
import pickle
import plotly.express as px
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import os

# Set up MLflow tracking URI to point to the local UI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create or get experiments for each model
experiment_names = ['Arima Experiment', 'Gradient Boosting Experiment', 'Linear Regression Experiment']
experiment_ids = []

for name in experiment_names:
    try:
        experiment = mlflow.get_experiment_by_name(name)
        if experiment:
            experiment_ids.append(experiment.experiment_id)
        else:
            experiment_id = mlflow.create_experiment(name)
            experiment_ids.append(experiment_id)
    except Exception as e:
        st.error(f"Error creating/getting experiment {name}: {e}")

# Load models from the models folder
model_files = {
    'Arima': 'Models/arima.pkl',
    'Gradient Boosting': 'Models/gradient_boosting.pkl',
    'Linear Regression': 'Models/linear_regression.pkl'
}

models = {}
for model_name, file_path in model_files.items():
    try:
        with open(file_path, 'rb') as f:
            models[model_name] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model {model_name} from {file_path}: {e}")

# Main page options
st.title("Model Experiment Dashboard")

selected_experiment_name = st.selectbox("Select an Experiment", experiment_names)

# Fetch runs for the selected experiment
if selected_experiment_name:
    selected_experiment_id = experiment_ids[experiment_names.index(selected_experiment_name)]

    try:
        runs = mlflow.search_runs(experiment_ids=[selected_experiment_id])
        if not runs.empty:
            run_df = pd.DataFrame(runs)
            st.write("## Runs")
            st.dataframe(run_df)

            # Display metrics and visualizations for each selected model
            selected_algorithm = st.selectbox("Select Model", list(models.keys()), key='model_select')

            if selected_algorithm:
                # Fetch run details for the selected model
                algo_runs = run_df[run_df['tags.mlflow.runName'].str.contains(selected_algorithm, na=False)]
                if not algo_runs.empty:
                    st.write(f"### {selected_algorithm} Model Details")
                    st.dataframe(algo_runs)

                    # Display parameters and metrics
                    st.write("#### Parameters:")
                    st.json(algo_runs['params'].apply(lambda x: eval(x)).tolist())

                    st.write("#### Metrics:")
                    st.json(algo_runs['metrics'].apply(lambda x: eval(x)).tolist())

                    # Visualization examples
                    if 'accuracy' in algo_runs.columns:
                        fig = px.line(algo_runs, x='start_time', y='accuracy', title=f'{selected_algorithm} Accuracy Over Time')
                        st.plotly_chart(fig)

                    if 'training_time' in algo_runs.columns and 'accuracy' in algo_runs.columns:
                        fig, ax = plt.subplots()
                        ax.scatter(algo_runs['training_time'], algo_runs['accuracy'])
                        ax.set_xlabel('Training Time')
                        ax.set_ylabel('Accuracy')
                        ax.set_title(f'Accuracy vs. Training Time for {selected_algorithm}')
                        st.pyplot(fig)

        else:
            st.warning("No runs found for the selected experiment.")
    except Exception as e:
        st.error(f"Error fetching runs: {e}")
else:
    st.warning("No experiments available.")
