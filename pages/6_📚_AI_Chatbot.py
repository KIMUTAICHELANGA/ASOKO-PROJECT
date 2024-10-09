import streamlit as st
import pandas as pd
import os

# Helper function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to get AI response (mockup)
def get_ai_response(user_input):
    # Here you can implement your AI logic or model inference.
    # For now, we'll return a mock response.
    return f"You said: {user_input}. This is a mock response."

# Chatbot interface
st.title("AI Chatbot")
st.write("Ask questions relating to deal making in Africa")

# Input field for user query (increased size)
user_input = st.text_area("Type your question or data request:", height=150)

# Replace the submit button with a search icon
if st.button("🔍"):
    if user_input:
        # Get response from AI
        response = get_ai_response(user_input)
        st.subheader("Chatbot Response")
        st.write(response)
    else:
        st.error("Please enter a question or data request.")

# Sidebar for selecting data sources
st.sidebar.title("Select Data Sources")
data_source_option = st.sidebar.radio("Choose an option:", ("General", "Choose Data Sources"))

if data_source_option == "Choose Data Sources":
    year = st.sidebar.selectbox("Select Year:", [2018, 2019, 2020, 2021, 2022, 2023])
    if st.sidebar.button("Choose"):
        st.sidebar.success(f"You selected the year: {year}")
        # Here, you can add any functionality you want after selecting the year.
    else:
        st.sidebar.warning("Select a year and click 'Choose'.")
