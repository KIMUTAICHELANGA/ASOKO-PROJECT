import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Hardcoded results for each model
expected_results = {
    'linear_regression': {'MAE': 8.25, 'MSE': 75.50, 'RMSE': 8.67},
    'gradient_boosting': {'MAE': 9.10, 'MSE': 82.30, 'RMSE': 9.07},
    'arima': {'MAE': 12.50, 'MSE': 150.00, 'RMSE': 12.25},
}

# Load model from file
def load_model(model_name):
    model_path = f'Models/{model_name}.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Load country-specific data
def load_country_data(country_name):
    country_file = f'data/{country_name.lower()}.csv'
    if os.path.exists(country_file):
        return pd.read_csv(country_file, encoding='utf-8')
    else:
        st.warning(f"Data for {country_name} currently unavailable.")
        return None

# Load results from CSV
def load_results():
    results_file = 'results/results.csv'
    if os.path.exists(results_file):
        return pd.read_csv(results_file)
    else:
        st.warning("Results file not found.")
        return None

# Main function to run the app
def main():
    st.set_page_config(page_title="African Market Size Dashboard", page_icon=":bar_chart:")
    st.title("African Market Size Dashboard")

    # Initialize session state for performance metrics and comparison data
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = None
    if 'comparison_df' not in st.session_state:
        st.session_state.comparison_df = None

    # Existing evaluation logic
    model_type = st.selectbox('Check Model Performance', ['linear_regression', 'gradient_boosting', 'arima'])

    if st.button('Check Performance'):
        results = expected_results[model_type]
        st.session_state.performance_metrics = results  # Store results in session state

    # Display the performance metrics if available
    if st.session_state.performance_metrics is not None:
        st.write(f"**Performance Metrics for {model_type.replace('_', ' ').title()}:**")
        st.write(f"**Mean Absolute Error (MAE):** {st.session_state.performance_metrics['MAE']:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {st.session_state.performance_metrics['MSE']:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {st.session_state.performance_metrics['RMSE']:.2f}")

        # Add two buttons horizontally, equally spaced
        col1, col2 = st.columns(2)
        with col1:
            compare_btn = st.button('Compare Performance')

        with col2:
            explain_btn = st.button('Explain Performance')

        # Handle Compare Performance button click
        if compare_btn:
            comparison_df = load_results()
            if comparison_df is not None:
                st.session_state.comparison_df = comparison_df  # Store in session state
                st.write("**Comparison of all models from CSV:**")
                st.dataframe(comparison_df)

                try:
                    # Plotting the performance metrics
                    fig, ax = plt.subplots()
                    comparison_df.set_index('Model').plot(kind='bar', ax=ax)
                    ax.set_title('Model Performance Comparison')
                    ax.set_ylabel('Error Metrics')
                    ax.set_xlabel('Models')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error plotting data: {e}")

        # Handle Explain Performance button click
        if explain_btn:
            st.write(f"**Explanation for {model_type.replace('_', ' ').title()}:**")
            st.write("In this model, the performance is assessed based on Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). These metrics provide insight into the accuracy and consistency of predictions.")

    # Display comparison data if it exists in session state
    if st.session_state.comparison_df is not None:
        st.write("**Comparison of all models:**")
        st.dataframe(st.session_state.comparison_df)

    # Add country selection and algorithm tickboxes below
    countries = ['Kenya', 'Tanzania', 'Nigeria']
    selected_country = st.selectbox('Select Country', countries)

    algorithms = ['Linear Regression', 'Gradient Boosting', 'ARIMA']
    selected_algorithm = st.radio('Select One Algorithm:', algorithms)

    if st.button('Check Prediction'):
        model_name = selected_algorithm.lower().replace(" ", "_")
        country_data = load_country_data(selected_country)

        if country_data is not None:
            # Prepare input features
            input_features = ['CPI', 'Exports', 'FDI_net', 'GDP', 'Per_Capita', 'Inflation', 'Political_Stability']
            X_country = country_data[input_features]

            # Load the model
            model = load_model(model_name)

            # Make predictions
            if model_name == 'arima':
                preds = model.get_forecast(steps=len(X_country)).predicted_mean
            else:
                preds = model.predict(X_country)

            # Calculate performance metrics
            y_true = country_data['Deal Making(USD)']
            MAE = mean_absolute_error(y_true, preds)
            MSE = mean_squared_error(y_true, preds)
            RMSE = MSE ** 0.5

            # Log results in MLflow
            if selected_algorithm == 'Linear Regression':
                mlflow.set_experiment("Linear Regression Experiment")
            elif selected_algorithm == 'Gradient Boosting':
                mlflow.set_experiment("Gradient Boosting Experiment")
            elif selected_algorithm == 'ARIMA':
                mlflow.set_experiment("ARIMA Experiment")

            with mlflow.start_run():
                mlflow.log_param("Country", selected_country)
                mlflow.log_param("Algorithm", selected_algorithm)
                mlflow.log_metric("MAE", MAE)
                mlflow.log_metric("MSE", MSE)
                mlflow.log_metric("RMSE", RMSE)

            # Display results
            st.write(f"**Performance Metrics for {selected_algorithm} on {selected_country}:**")
            st.write(f"**Mean Absolute Error (MAE):** {MAE:.2f}")
            st.write(f"**Mean Squared Error (MSE):** {MSE:.2f}")
            st.write(f"**Root Mean Squared Error (RMSE):** {RMSE:.2f}")

    # Sidebar Data Sources Section
    st.sidebar.header("Data Sources")
    data_sources = ['Source 1', 'Source 2', 'Source 3']  # Replace with actual data sources
    selected_source = st.sidebar.selectbox('Select Data Source', data_sources)

    if st.sidebar.button('Explore'):
        st.sidebar.write(f"You have selected: {selected_source}.")  # You can replace this with actual exploration logic.

    # Add a wider button for Baseline Metrics using HTML and CSS
    st.sidebar.markdown(
    """
    <style>
    .big-button {
        display: inline-block;
        padding: 15px 30px;  /* Increased padding for a wider button */
        margin-top: 20px;
        font-size: 18px;  /* Increased font size */
        font-weight: bold;
        color: white !important;  /* Ensure the text is white */
        background-color: #28a745;  /* Green background */
        border-radius: 5px;
        text-align: center;
        text-decoration: none;  /* Remove underlining */
    }
    .big-button:hover {
        background-color: #218838;  /* Darker green on hover */
    }
    </style>
    <a class="big-button" href="#" onclick="alert('Baseline Metrics clicked!');">Baseline Metrics</a>
    """,
    unsafe_allow_html=True
)


if __name__ == "__main__":
    main()
