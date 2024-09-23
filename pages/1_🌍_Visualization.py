#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import os

#######################
# Page configuration
st.set_page_config(
    page_title="Africa Deal Making Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"],
[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

#######################
# Load data
data_folder = 'data'
csv_files = [file for file in os.listdir(data_folder) if file.endswith('.csv')]

# Load the first CSV file by default
df_reshaped = pd.read_csv(os.path.join(data_folder, csv_files[0]))

#######################
# Sidebar for selecting color theme
st.sidebar.header('Settings')
color_theme_list = ['blues', 'greens', 'reds', 'rainbow', 'turbo', 'viridis']
selected_color_theme = st.sidebar.selectbox('Select a color theme', color_theme_list)

#######################
# Dashboard Main Panel
st.title('🏂 AFRICA DEAL MAKING DASHBOARD')

# Dropdown to select a country
country_list = df_reshaped['Country'].unique()
selected_country = st.selectbox('Select a country', country_list)

# Filter data based on selected country
country_data = df_reshaped[df_reshaped['Country'] == selected_country]

# Year filter
year_list = list(df_reshaped['Year'].unique())[::-1]
selected_year = st.selectbox('Select a year', year_list)

# Filter data based on selected year
country_data_year_filtered = country_data[country_data['Year'] == selected_year]

if not country_data_year_filtered.empty:
    country_value = country_data_year_filtered['Deal Making(USD)'].sum()
    country_deals = country_data_year_filtered['GDP'].sum()  # Replace 'Number' with GDP
else:
    country_value = 0
    country_deals = 0

# Display results
st.metric(label=f"{selected_country} (GDP: {country_deals})", 
          value=f"${country_value} M")

#######################
# Plots

# Choropleth map function
def make_choropleth(input_df, input_id, input_column, input_color_theme):
    choropleth = px.choropleth(input_df, locations=input_id, color=input_column,
                               color_continuous_scale=input_color_theme,
                               scope="africa",
                               labels={'Deal Making(USD)': 'Deal Making(USD)'}
                              )
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return choropleth

with st.container():
    st.markdown('#### Total Deal Value')

    choropleth = make_choropleth(df_reshaped, 'Country', 'Deal Making(USD)', selected_color_theme)
    st.plotly_chart(choropleth, use_container_width=True)

    # Heatmap function
    def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
        heatmap = alt.Chart(input_df).mark_rect().encode(
                y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
                x=alt.X(f'{input_x}:O', axis=alt.Axis(title="Country", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
                color=alt.Color(f'{input_color}:Q',
                                 legend=None,
                                 scale=alt.Scale(scheme=input_color_theme)),
                stroke=alt.value('black'),
                strokeWidth=alt.value(0.25),
            ).properties(width=900).configure_axis(
            labelFontSize=12,
            titleFontSize=12
            ) 
        return heatmap

    heatmap = make_heatmap(df_reshaped, 'Year', 'Country', 'Deal Making(USD)', selected_color_theme)
    st.altair_chart(heatmap, use_container_width=True)

with st.container():
    st.markdown('#### Top Sectors')

    # Group by 'Country' or any other feature relevant to your data
    country_group = df_reshaped.groupby('Country').sum().reset_index().sort_values('Deal Making(USD)', ascending=False)
    st.dataframe(country_group[['Country', 'Deal Making(USD)', 'GDP']])  
