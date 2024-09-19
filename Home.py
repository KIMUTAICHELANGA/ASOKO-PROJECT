import streamlit as st
import pandas as pd
import json
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import shap
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pickle
import os

# Page configuration
APP_TITLE = 'African Market Size Dashboard'
APP_SUB_TITLE = 'Source: African Market Data'

# Load Data
df_africa = pd.read_csv('data/africa_deal_data.csv', encoding='utf-8')

# Load GeoJSON for African country boundaries
with open('data/africa.geo.json', encoding='utf-8') as f:
    africa_geojson = json.load(f)

# Load model from file
def load_model(model_name):
    model_path = f'Models/{model_name}.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Load models and make predictions
def load_and_predict(X_test, y_test):
    # Load models
    lin_reg = load_model('Linear_Regression')
    gbr = load_model('Gradient_Boosting')
    arima_result = load_model('ARIMA')

    # Make predictions
    lin_reg_pred = lin_reg.predict(X_test)
    gbr_pred = gbr.predict(X_test)
    
    # ARIMA prediction (assume ARIMA model is pre-trained for forecasting)
    arima_pred = arima_result.forecast(steps=len(y_test))

    # Calculate metrics
    lin_reg_r2 = r2_score(y_test, lin_reg_pred)
    gbr_r2 = r2_score(y_test, gbr_pred)
    arima_r2 = r2_score(y_test, arima_pred)

    lin_reg_mae = mean_absolute_error(y_test, lin_reg_pred)
    gbr_mae = mean_absolute_error(y_test, gbr_pred)
    arima_mae = mean_absolute_error(y_test, arima_pred)

    return (
        lin_reg_pred, gbr_pred, arima_pred,
        lin_reg, gbr, arima_result,
        lin_reg_r2, gbr_r2, arima_r2,
        lin_reg_mae, gbr_mae, arima_mae
    )

# Display Filters and Year Selector
def display_filters():
    country_list = [''] + list(df_africa['Country'].unique())
    country_list.sort()
    selected_country = st.selectbox('Select Country', country_list)

    if st.button('Check Prediction', key='check_prediction_button'):
        st.session_state.show_prediction = True
        st.session_state.selected_country = selected_country

    return selected_country

# Display Deal-Making Metrics
def display_deal_facts(df, country_name, field, title, string_format='${:,}'):
    df = df[df['Country'] == country_name] if country_name else df
    df.drop_duplicates(inplace=True)
    total = df[field].sum() if len(df) else 0
    st.metric(title, string_format.format(round(total)))

# Display Deal Volume and Category
def display_deal_volume_category(df, country_name):
    if country_name:
        country_data = df[df['Country'] == country_name]
        total_volume = country_data['Deal Value (USD Millions)'].sum()
        
        # Determine deal volume category
        if total_volume < 100:
            category = 'Low'
        elif 100 <= total_volume < 500:
            category = 'Medium'
        else:
            category = 'High'

        st.write(f"**Total Deal Value: {total_volume} USD Millions**")
        st.write(f"Category: {category}")

# Display Algorithm Performance Table with actual metrics
def display_algorithm_performance(lin_reg_r2, gbr_r2, arima_r2, lin_reg_mae, gbr_mae, arima_mae):
    # Real performance metrics
    algorithm_results = {
        'Algorithm': ['Linear Regression', 'Gradient Boosting', 'ARIMA'],
        'Rank': [1, 2, 3],  # This can be dynamically ranked based on metrics if needed
        'R² Score': [lin_reg_r2, gbr_r2, arima_r2],
        'Mean Absolute Error': [lin_reg_mae, gbr_mae, arima_mae]
    }

    # Convert to DataFrame
    results_df = pd.DataFrame(algorithm_results)
    st.write("## Algorithm Performance Ranking")
    st.dataframe(results_df)

    # Display notice about prediction algorithm
    st.write("**Note:** The prediction is based on the rank 1 algorithm.")

# Display Linear Regression Explainability
def display_linear_regression_explainability(model, X_train):
    st.write("## Linear Regression Feature Coefficients")
    coefficients = model.coef_
    feature_names = X_train.columns
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=False)
    st.dataframe(coef_df)
    st.write("The coefficients represent the impact of each feature on the target variable. Higher coefficients mean a greater impact.")

# Display ARIMA Explainability
def display_arima_explainability(model, y_test):
    st.write("## ARIMA Model Diagnostics")

    # Calculate residuals
    residuals = y_test - model.predict(start=len(y_test) - len(y_test), end=len(y_test) - 1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    # Plot residuals
    ax[0].plot(residuals)
    ax[0].set_title('Residuals')

    # Plot ACF and PACF
    plot_acf(residuals, lags=20, ax=ax[1])
    plot_pacf(residuals, lags=20, ax=ax[1])

    st.pyplot(fig)
    st.write("Residuals represent the difference between the observed values and the values predicted by the model.")

# Sidebar for Data Sources
def sidebar_data_sources():
    st.sidebar.write("## Data Sources")

    # Get list of files in the 'data' folder
    data_folder = 'data'
    data_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

    # Dropdown to select a file
    selected_file = st.sidebar.selectbox("Select a Data File", data_files)

    # Brief description for each dataset
    descriptions = {
        'africa_deal_data.csv': "This dataset contains deal data from various African countries, including features such as Political Stability, Economic Environment, Regulatory Framework, Infrastructure, Human Capital, and Deal Value (USD Millions).",
        # Add more descriptions here if there are other datasets
    }

    # Show description for the selected file
    st.sidebar.write("### Description:")
    st.sidebar.write(descriptions.get(selected_file, "No description available for this file."))

    # Provide a preview of the selected dataset
    st.sidebar.write("### Dataset Preview:")
    file_path = os.path.join(data_folder, selected_file)
    
    # Load and display the preview
    try:
        preview_data = pd.read_csv(file_path).head()  # Preview the first few rows
        st.sidebar.dataframe(preview_data)
    except Exception as e:
        st.sidebar.write(f"Error loading file: {e}")

# Main function to run the app
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=":bar_chart:")
    st.title(APP_TITLE)
    st.subheader(APP_SUB_TITLE)

    # Add the sidebar data sources section
    sidebar_data_sources()

    selected_country = display_filters()

    if selected_country and st.session_state.get('show_prediction', False):
        col1, col2 = st.columns(2)

        with col1:
            display_deal_facts(df_africa, selected_country, 'Deal Value (USD Millions)', 'Total Deal Value')
        with col2:
            display_deal_volume_category(df_africa, selected_country)

        X = df_africa[['Political Stability', 'Economic Environment', 'Regulatory Framework', 'Infrastructure', 'Human Capital']]
        y = df_africa['Deal Value (USD Millions)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lin_reg_pred, gbr_pred, arima_pred, lin_reg, gbr, arima_result, lin_reg_r2, gbr_r2, arima_r2, lin_reg_mae, gbr_mae, arima_mae = load_and_predict(X_test, y_test)
        display_algorithm_performance(lin_reg_r2, gbr_r2, arima_r2, lin_reg_mae, gbr_mae, arima_mae)

        # Show 'Explain Prediction' button only after prediction
        if st.button('Explain Prediction', key='explain_prediction_button'):
            display_linear_regression_explainability(lin_reg, X_train)
            display_arima_explainability(arima_result, y_test)

if __name__ == "__main__":
    main()
