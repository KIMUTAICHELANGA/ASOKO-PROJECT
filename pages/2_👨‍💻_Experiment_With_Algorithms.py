import streamlit as st
import mlflow
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pmdarima import auto_arima  # For ARIMA
from sklearn.neural_network import MLPRegressor  # Neural Network
import subprocess
import os

# Define a custom selectbox with an initial "Select one option" placeholder
def selectbox_without_default(label, options):
    options = [''] + options
    format_func = lambda x: 'Select one option' if x == '' else x
    return st.selectbox(label, options, format_func=format_func)

# Load the deal-making dataset
@st.cache_data
def load_data():
    return pd.read_csv('data/africa_deal_data.csv')

# Available models for regression and ARIMA for time series
MODELS = {
    "Linear Regression": "LinearRegression",
    "Gradient Boosting": "GradientBoosting",
    "ARIMA": "ARIMA",  # Special handling for ARIMA
    "Neural Network": "NeuralNetwork",
    "Random Forest": "RandomForest",
    "Support Vector Regressor": "SVR"
}

# Function to open MLflow UI
def open_mlflow_ui():
    try:
        subprocess.Popen(["mlflow", "ui"])
    except Exception as e:
        st.sidebar.error(f"Error launching MLflow UI: {str(e)}")

# Load pre-trained model from file
def load_model_from_file(model_name):
    file_path = f'Models/{model_name}.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        return None

# Main app logic
def main():
    st.title("Experiment with Algorithms")

    # Load the dataset
    df = load_data()
    st.write("Dataset", df)

    # Select target column (deal volume)
    target_column = 'Number'

    # Model selection
    model_options = list(MODELS.keys())
    model_choice = selectbox_without_default("Choose a model", model_options)
    if not model_choice:
        st.stop()

    model_name = MODELS[model_choice]
    model = load_model_from_file(model_name)

    if model is None:
        st.warning(f"{model_choice} model is not yet available. Please check back later.")
        st.stop()

    # MLflow tracking option
    track_with_mlflow = st.checkbox("Track with MLflow?")

    # Model training or prediction
    start_training = st.button("Run model")
    if not start_training:
        st.stop()

    if track_with_mlflow:
        experiment_name = "Deal Volume Prediction"
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.log_param('model', model_choice)

    # Prepare the data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run the model and predict
    if model_choice == "ARIMA":
        # ARIMA models are time series, so we need to handle this separately
        model = auto_arima(y_train, seasonal=False)
        preds_train = model.predict_in_sample()
        preds_test = model.predict(n_periods=len(y_test))
    else:
        model.fit(X_train, y_train)
        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)

    # Model evaluation
    metric_train = r2_score(y_train, preds_train)
    metric_test = r2_score(y_test, preds_test)
    st.write(f"R² score (train): {round(metric_train, 3)}")
    st.write(f"R² score (test): {round(metric_test, 3)}")

    if track_with_mlflow:
        mlflow.log_metric("r2_score_train", metric_train)
        mlflow.log_metric("r2_score_test", metric_test)
        mlflow.end_run()

# Sidebar for MLflow Tracking
def sidebar_menu():
    st.sidebar.title("MLflow Tracking 🔎")
    if st.sidebar.button("Launch 🚀"):
        open_mlflow_ui()
        st.sidebar.success("MLflow Server is Live! http://localhost:5000")
        st.sidebar.markdown("[Open MLflow](http://localhost:5000)", unsafe_allow_html=True)

if __name__ == '__main__':
    sidebar_menu()
    main()
