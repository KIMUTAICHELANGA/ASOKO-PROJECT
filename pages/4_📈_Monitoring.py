import os
import streamlit as st
import pandas as pd
import time
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, RegressionPerformanceTab, DataQualityTab

class MarketSizeMonitoringController:
    def __init__(self):
        self.view = MarketSizeMonitoringView()

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
        generate_target_drift = st.checkbox("Data Quality")
        generate_target_drift = st.checkbox("Generate Target Drift Report")
        generate_data_drift = st.checkbox("Generate Data Drift Report")
        generate_data_quality = st.checkbox("Generate Data Quality Report")

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
            df['target'] = df['Deal Value (USD Millions)']  # Example target column
            df['prediction'] = df['target']  # Example placeholder for predictions

            # Filter data based on year range
            date_range = (df['Year'] >= start_year) & (df['Year'] <= end_year)
            reference_data = df[~date_range]
            current_data = df[date_range]

            # Check for missing data and empty DataFrames
            st.write("Checking for missing values...")
            st.write(df.isnull().sum())
            
            if reference_data.empty or current_data.empty:
                st.write("No data available for the selected year range.")
                return
            
            st.write("Data Sample:")
            st.write(df[['Year', 'Deal Value (USD Millions)', 'target', 'prediction']].head())

            if generate_target_drift:
                st.write("### Target Drift Report")
                st.write("Generating Target Drift Report...")
                try:
                    target_drift_report = self.generate_report(reference_data, current_data, DataDriftTab())
                    st.components.v1.html(target_drift_report, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error generating Target Drift Report: {e}")

            if generate_data_drift:
                st.write("### Data Drift Report")
                st.write("Generating Data Drift Report...")
                try:
                    data_drift_report = self.generate_report(reference_data, current_data, DataDriftTab())
                    st.components.v1.html(data_drift_report, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error generating Data Drift Report: {e}")

            if generate_data_quality:
                st.write("### Data Quality Report")
                st.write("Generating Data Quality Report...")
                try:
                    data_quality_report = self.generate_report(reference_data, current_data, DataQualityTab())
                    st.components.v1.html(data_quality_report, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error generating Data Quality Report: {e}")

        # Dropdown to choose a model
        st.subheader("Choose Model for Performance Report")
        model_choices = ["Linear Regression", "Gradient Boosting", "ARIMA"]
        selected_model = st.selectbox("Select Model:", model_choices)
        
        # Generate Model Performance Report checkbox and submit button
        generate_model_report = st.checkbox("Generate Model Performance Report")
        
        if st.button("Submit Model Report"):
            if generate_model_report:
                st.write(f"### {selected_model} Performance Report")
                st.write(f"Generating {selected_model} Model Performance Report...")
                try:
                    performance_report = self.generate_report(reference_data, current_data, RegressionPerformanceTab())
                    st.components.v1.html(performance_report, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error generating {selected_model} Model Performance Report: {e}")

        

    def generate_report(self, reference_data, current_data, tab):
        # Ensure the data contains 'target' and 'prediction' columns
        if 'target' not in reference_data.columns or 'prediction' not in reference_data.columns:
            raise ValueError("The reference_data must contain 'target' and 'prediction' columns.")
        
        if 'target' not in current_data.columns or 'prediction' not in current_data.columns:
            raise ValueError("The current_data must contain 'target' and 'prediction' columns.")
        
        # Check for empty data
        if reference_data.empty or current_data.empty:
            raise ValueError("One or both datasets are empty.")

        column_mapping = {}  # Define your column mapping if needed
        dashboard = Dashboard(tabs=[tab])
        dashboard.calculate(reference_data, current_data, column_mapping=column_mapping)
        
        # Save the report to an HTML file with UTF-8 encoding
        report_path = "report.html"
        dashboard.save(report_path)
        
        # Read the HTML content with UTF-8 encoding
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
