import streamlit as st
import mlflow
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pmdarima import auto_arima
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
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
    return pd.read_csv('data/modeldata.csv')

# Available models for regression and ARIMA for time series
MODELS = {
    "Linear Regression": LinearRegression,
    "Gradient Boosting": GradientBoostingRegressor,
    "ARIMA": auto_arima,  # Special handling for ARIMA
    "Neural Network": MLPRegressor,
    "Random Forest": RandomForestRegressor,
    "Support Vector Regressor": SVR
}

# Define hyperparameter ranges for models
HYPERPARAMETERS = {
    "Linear Regression": {},
    "Gradient Boosting": {
        "n_estimators": (1, 100, 50),
        "learning_rate": (0.01, 0.3, 0.1),
        "max_depth": (1, 10, 3)
    },
    "ARIMA": {},  # Hyperparameters for ARIMA can be set in a different way
    "Neural Network": {
        "hidden_layer_sizes": (1, 100, 10),
        "activation": ['identity', 'logistic', 'tanh', 'relu'],
        "alpha": (0.0001, 1.0, 0.0001)
    },
    "Random Forest": {
        "n_estimators": (1, 100, 100),
        "max_depth": (1, 10, None)
    },
    "Support Vector Regressor": {
        "C": (0.1, 100.0, 1.0),
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
    }
}

# Function to open MLflow UI
def open_mlflow_ui():
    try:
        subprocess.Popen(["mlflow", "ui"])
    except Exception as e:
        st.sidebar.error(f"Error launching MLflow UI: {str(e)}")

# Load pre-trained model from file
def load_model_from_file(model_name):
    model_file_mapping = {
        "Linear Regression": "linear_regression.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "ARIMA": "arima.pkl",
        "Neural Network": "neural_network.pkl",
        "Random Forest": "random_forest.pkl",
        "Support Vector Regressor": "svr.pkl"
    }
    
    file_name = model_file_mapping.get(model_name)
    if file_name:
        file_path = os.path.join('Models', file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                return pickle.load(file)
    return None

# Main app logic
def main():
    st.title("Experiment with Algorithms")

    # Load the dataset
    df = load_data()
    st.write("Dataset", df)

    # Select target column (deal volume)
    target_column = 'Deal Making(USD)'

    # Model selection
    model_options = list(MODELS.keys())
    model_choice = selectbox_without_default("Choose a model", model_options)

    # Hyperparameter tuning inputs displayed below the model selection
    if model_choice:
        st.write("### Hyperparameter Tuning")
        hyperparams = {}
        if model_choice in HYPERPARAMETERS:
            for param, value in HYPERPARAMETERS[model_choice].items():
                if isinstance(value, tuple):  # For sliders
                    hyperparams[param] = st.slider(param, value[0], value[1], value[2])
                elif isinstance(value, list):  # For selectboxes
                    hyperparams[param] = st.selectbox(param, value)

        # Check if model is selected and if hyperparams are ready for use
        if not model_choice:
            st.stop()

        model = load_model_from_file(model_choice)

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
            # Set the experiment name based on the model choice
            if model_choice == "ARIMA":
                experiment_name = "Arima Experiment"
            elif model_choice == "Linear Regression":
                experiment_name = "Linear Regression Experiment"
            elif model_choice == "Gradient Boosting":
                experiment_name = "Gradient Boosting Experiment"
            else:
                experiment_name = "Other Experiment"  # Optional for other models

            if mlflow.get_experiment_by_name(experiment_name) is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            mlflow.log_param('model', model_choice)

        # Prepare the data
        input_features = ['CPI', 'Exports', 'FDI_net', 'GDP', 'Per_Capita', 'Inflation', 'Political_Stability']
        X = df[input_features]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Run the model and predict
        if model_choice == "ARIMA":
            model_instance = auto_arima(y_train, seasonal=False, stepwise=True, trace=True)
            preds_train = model_instance.predict(n_periods=len(y_train))
            preds_test = model_instance.predict(n_periods=len(y_test))
        else:
            # Create the model instance with the tuned hyperparameters
            model_instance = MODELS[model_choice](**hyperparams)
            model_instance.fit(X_train, y_train)
            preds_train = model_instance.predict(X_train)
            preds_test = model_instance.predict(X_test)

        # Model evaluation
        metric_train = r2_score(y_train, preds_train)
        metric_test = r2_score(y_test, preds_test)
        st.write(f"R² score (train): {round(metric_train, 3)}")
        st.write(f"R² score (test): {round(metric_test, 3)}")

        if track_with_mlflow:
            mlflow.log_metric("r2_score_train", metric_train)
            mlflow.log_metric("r2_score_test", metric_test)
            mlflow.end_run()

        # Display metrics comparison and predictions
        st.write("### Predictions")
        st.write(f"Training Predictions: {list(preds_train[:5])}")
        st.write(f"Testing Predictions: {list(preds_test[:5])}")

        # Additional Model Tracking and Display
        if model_choice in ["Linear Regression", "Gradient Boosting", "Neural Network", "Random Forest", "Support Vector Regressor"]:
            st.write("## Model Details")
            st.write(f"Model Type: {model_choice}")

            if hasattr(model_instance, 'feature_importances_'):
                st.write("### Feature Importances")
                importances = model_instance.feature_importances_
                feature_importances = pd.DataFrame(importances, index=input_features, columns=['Importance'])
                st.bar_chart(feature_importances)

# Sidebar for MLflow Tracking
def sidebar_menu():
    st.sidebar.title("MLflow Tracking 🔎")
    if st.sidebar.button("Launch 🚀"):
        open_mlflow_ui()
        st.sidebar.success("MLflow Server is Live! http://localhost:5000")
        st.sidebar.markdown("[Open MLflow](http://localhost:5000)", unsafe_allow_html=True)

    # Button for visualizing model performances
    if st.sidebar.button("Visualize Model Performances"):
        st.sidebar.success("Visualizing model performances...")  # Placeholder for future functionality
        # You can add your visualization logic here

if __name__ == '__main__':
    sidebar_menu()
    main()
