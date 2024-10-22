import streamlit as st
import pandas as pd

# Function to return predefined deal-making data for all years
def get_general_deal_data():
    return {
        2019: {
            "KENYA": {
                "amount": "$1.3 - $1.7 billion USD",
                "summary": "Kenya saw continued investment in the technology sector (especially fintech) and infrastructure development, contributing to deal-making."
            },
            "EGYPT": {
                "amount": "$3.5 - $4.5 billion USD",
                "summary": "Strong growth, driven by mega-projects such as the New Administrative Capital and large energy deals (especially gas and renewables)."
            },
            "RWANDA": {
                "amount": "$400 - $600 million USD",
                "summary": "Rwanda continued to attract investment in tourism, technology (especially smart cities), and renewable energy. Kigali’s development was a key focus."
            },
            "DJIBOUTI": {
                "amount": "$200 - $300 million USD",
                "summary": "Deal-making centered around port expansions and logistics infrastructure due to Djibouti’s strategic position as a gateway to East Africa."
            },
        },
        2020: {
            "KENYA": {
                "amount": "$1 - $1.3 billion USD",
                "summary": "Despite the pandemic, Kenya's energy sector (especially natural gas) remained strong, with investments continuing in real estate and infrastructure."
            },
            "EGYPT": {
                "amount": "$3.5 - $4 billion USD",
                "summary": "Due to the COVID-19 pandemic, economic activities slowed, though investments in technology and healthcare helped cushion the fall. Government fiscal policies also supported sectors like agriculture."
            },
            "RWANDA": {
                "amount": "$200 - $400 million USD",
                "summary": "The pandemic caused disruptions, especially in tourism, but green energy and technology sectors still saw modest growth."
            },
            "DJIBOUTI": {
                "amount": "$120 - $200 million USD",
                "summary": "COVID-19 slowed global trade, but Djibouti’s ports remained operational, keeping deal-making relatively stable."
            },
        },
        2021: {
            "KENYA": {
                "amount": "$1.5 - $2 billion USD",
                "summary": "The economy began recovering with more tech deals, expansion in renewable energy, and ongoing investments in agriculture and logistics."
            },
            "EGYPT": {
                "amount": "$4.5 - $5.5 billion USD",
                "summary": "Egypt showed resilience, with a strong rebound in infrastructure projects and renewable energy. Private equity activity also picked up."
            },
            "RWANDA": {
                "amount": "$400 - $600 million USD",
                "summary": "Rwanda’s economy rebounded as tourism gradually returned and new tech initiatives and sustainability projects gained traction."
            },
            "DJIBOUTI": {
                "amount": "$200 - $300 million USD",
                "summary": "As global trade began recovering, Djibouti benefitted from increased activity in its ports and logistics, along with some interest in renewable energy."
            },
        },
        2022: {
            "KENYA": {
                "amount": "$2 - $2.5 billion USD",
                "summary": "Kenya experienced a strong recovery, driven by tech startups, infrastructure projects (e.g., the Nairobi Expressway), and increasing investor confidence."
            },
            "EGYPT": {
                "amount": "$5 - $6 billion USD",
                "summary": "Investments continued to rise, focusing on technology, regional logistics, and green energy, with Egypt positioned as an innovation hub in East Africa."
            },
            "RWANDA": {
                "amount": "$500 - $700 million USD",
                "summary": "Investments continued to rise, focusing on technology, regional logistics, and green energy, with Rwanda positioned as an innovation hub in East Africa."
            },
            "DJIBOUTI": {
                "amount": "$250 - $350 million USD",
                "summary": "Steady investments in logistics and energy, including port expansion projects and interest in renewable energy."
            },
        }
    }

# Function to get AI response (mockup)
def get_ai_response(user_input):
    # Mockup AI response logic
    return 

# Chatbot interface
st.title("AI Chatbot")
st.write("Ask questions relating to deal making in Africa")

# Input field for user query
user_input = st.text_area("Type your question or data request:", height=150)

# Sidebar for selecting data sources
st.sidebar.title("Select Data Sources")
data_source_option = st.sidebar.radio("Choose an option:", ("General", "Choose Data Sources"))

# Replace the submit button with a search icon
if st.button("🔍"):
    if user_input:
        # Get response from AI
        response = get_ai_response(user_input)
        st.subheader("Chatbot Response")
        st.write(response)

        # Show general deal-making data if "General" is selected in the sidebar
        if data_source_option == "General":
            st.subheader("General Deal-Making Data:")
            # Get all years' data
            general_data = get_general_deal_data()
            # Loop through the data and display it
            for year, details in general_data.items():
                st.write(f"**{year}:**")
                for country, info in details.items():
                    st.write(f"**{country}:** {info['amount']}")
                    st.write(f"**Deal Summary:** {info['summary']}")
                st.write("")  # Add a line break for better readability
    else:
        st.error("Please enter a question or data request.")

# When "Choose Data Sources" is selected, allow year selection
if data_source_option == "Choose Data Sources":
    year = st.sidebar.selectbox("Select Year:", [2019, 2020, 2021, 2022])
    if st.sidebar.button("Choose"):
        st.sidebar.success(f"You selected the year: {year}")
        # Show the data for the selected year
        general_data = get_general_deal_data()
        st.subheader(f"Deal-Making Data for {year}:")
        for country, info in general_data[year].items():
            st.write(f"**{country}:** {info['amount']}")
            st.write(f"**Deal Summary:** {info['summary']}")
        st.write("")  # Add a line break for better readability
    else:
        st.sidebar.warning("Select a year and click 'Choose'.")
