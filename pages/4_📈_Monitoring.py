import os
import streamlit as st
import pandas as pd
import time
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, ColumnSummaryMetric, RegressionQualityMetric  # Removed DatasetQualityMetric

class MarketSizeMonitoringController:
    def __init__(self):
        self.view = MarketSizeMonitoringView()
        self.reference_data = None
        self.current_data = None

    def run_monitoring(self):
        st.title("Data & Model Monitoring App")
        st.write("You are in the Data & Model Monitoring App. Select the Year range from the sidebar and click 'Submit' to start model training and monitoring.")

        # Dropdown for selecting the data source
        st.subheader("Select Data Source")
        data_folder = "data"
        data_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        selected_file = st.selectbox("Choose a CSV file:", data_files)

        # Sliders for selecting year range
        start_year = st.sidebar.slider("Start Year", min_value=2018, max_value=2023, value=2018)
        end_year = st.sidebar.slider("End Year", min_value=2018, max_value=2023, value=2023)

        st.subheader("Select Reports to Generate")
        generate_column_drift = st.checkbox("Generate Column Drift Report")
        generate_column_summary = st.checkbox("Generate Column Summary Report")  # New checkbox for column summary
        generate_regression_quality = st.checkbox("Generate Regression Quality Report")  # New checkbox for regression quality

        # Submit button for fetching data and filtering by year
        if st.button("Submit"):
            st.write("Fetching your current batch data...")
            data_start = time.time()

            # Load the selected data source
            df = pd.read_csv(f"{data_folder}/{selected_file}")

            data_end = time.time()
            time_taken = data_end - data_start
            st.write(f"Fetched the data within {time_taken:.2f} seconds")

            # Ensure target and prediction columns exist
            df['target'] = df['Deal Making(USD)']  # Corrected target column
            df['prediction'] = df['target']  # Example placeholder for predictions

            # Filter data based on year range
            date_range = (df['Year'] >= start_year) & (df['Year'] <= end_year)
            self.reference_data = df[~date_range]
            self.current_data = df[date_range]

            if self.reference_data.empty or self.current_data.empty:
                st.write("No data available for the selected year range.")
                return

            # Generate reports based on selections
            if generate_column_drift:
                st.write("### Column Drift Report")
                st.write("Generating Column Drift Report...")
                try:
                    column_drift_report = self.generate_report(self.reference_data, self.current_data, ColumnDriftMetric())
                    st.components.v1.html(column_drift_report, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error generating Column Drift Report: {e}")

            if generate_column_summary:
                st.write("### Column Summary Report")
                st.write("Generating Column Summary Report...")
                try:
                    column_summary_report = self.generate_report(self.reference_data, self.current_data, ColumnSummaryMetric())
                    st.components.v1.html(column_summary_report, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error generating Column Summary Report: {e}")

            if generate_regression_quality:  # New regression quality report generation
                st.write("### Regression Quality Report")
                st.write("Generating Regression Quality Report...")
                try:
                    regression_quality_report = self.generate_report(self.reference_data, self.current_data, RegressionQualityMetric())
                    st.components.v1.html(regression_quality_report, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error generating Regression Quality Report: {e}")

        # Dropdown to choose a model
        st.subheader("Choose Model for Performance Report")
        model_choices = ["Linear Regression", "Gradient Boosting", "ARIMA"]
        selected_model = st.selectbox("Select Model:", model_choices)

        # Submit button for generating model performance report
        if st.button("Submit Model Report"):
            # Load results from the CSV file
            results_path = "results/results.csv"  # Adjust the path if necessary

            try:
                results_df = pd.read_csv(results_path)
            except Exception as e:
                st.error(f"Error loading results from {results_path}: {e}")
                return

            # Filter the results for the selected model
            model_results = results_df[results_df['Model'] == selected_model]

            if model_results.empty:
                st.error(f"No results found for the model: {selected_model}")
                return

            # Display model results
            st.write(f"### {selected_model} Performance Results")
            st.dataframe(model_results)

            # Display evaluation metrics directly from the results
            st.subheader("Model Evaluation Metrics")
            mae = model_results['Mean Absolute Error (MAE)'].values[0]
            mse = model_results['Mean Squared Error (MSE)'].values[0]
            rmse = model_results['Root Mean Squared Error (RMSE)'].values[0]

            st.write(f"- Mean Absolute Error (MAE): {mae:.2f}")
            st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")

            # After showing results, display the model explanation
            st.subheader("Model Explanation")
            st.write(f"Fetching explanation for {selected_model} model...")

            # Example explanation logic for different models
            if selected_model == "Linear Regression":
                st.write("Linear Regression is a simple regression technique used to model the relationship between a dependent variable and one or more independent variables.")
                st.write("Feature importance is determined by the coefficients of the linear equation.")

            elif selected_model == "Gradient Boosting":
                st.write("Gradient Boosting is an ensemble technique that builds multiple decision trees to improve model performance.")
                st.write("Feature importance can be evaluated based on the contribution of each feature in reducing the overall error.")

            elif selected_model == "ARIMA":
                st.write("ARIMA is a time-series forecasting method that combines autoregressive and moving average components.")
                st.write("Residual analysis is performed to check for patterns in the forecast errors.")

    def generate_report(self, reference_data, current_data, metric):
        # Ensure the data contains 'target' and 'prediction' columns
        if 'target' not in reference_data.columns or 'prediction' not in reference_data.columns:
            raise ValueError("The reference_data must contain 'target' and 'prediction' columns.")
        
        if 'target' not in current_data.columns or 'prediction' not in current_data.columns:
            raise ValueError("The current_data must contain 'target' and 'prediction' columns.")
        
        # Check for empty data
        if reference_data.empty or current_data.empty:
            raise ValueError("One or both datasets are empty.")

        # Create a report with the specified metric
        report = Report(metrics=[metric])
        report.run(reference_data=reference_data, current_data=current_data)

        # Save the report to an HTML file
        report_path = "report.html"
        report.save(report_path)

        # Read the HTML content
        try:
            with open(report_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            raise RuntimeError(f"Error reading the report HTML file: {e}")

class MarketSizeMonitoringView:
    def display_monitoring(self, reference_data, current_data):
        st.write("## Monitoring Overview")
        st.write("### Reference Data")
        st.dataframe(reference_data)
        st.write("### Current Data")
        st.dataframe(current_data)

if __name__ == "__main__":
    controller = MarketSizeMonitoringController()
    controller.run_monitoring()
