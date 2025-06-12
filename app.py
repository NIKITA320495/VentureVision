import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import Tool
import os
import asyncio
from agents import extract, run_market_analysis_agent, run_competitive_analysis_agent, financial_analysis, combined_agent

# --- Streamlit Application Layout ---
st.set_page_config(page_title="VentureVision: Your AI Business Analyzer", layout="wide")

st.title("💡 VentureVision: Your AI Business Analyzer")
st.markdown(
    """
    Enter your business idea and let our AI agents provide comprehensive analysis
    on market trends, financial viability, location suitability, and personalized mentorship.
    """
)

# Sidebar for analysis type selection
st.sidebar.title("Choose Analysis Type")
analysis_type = st.sidebar.radio(
    "Select the type of analysis you need:",
    ("Overall Business Analysis", "Market Research Analysis", "Competitive Analysis", "Financial Analysis"),
    key="analysis_type_selector"
)

user_input = st.text_input(
    "Give a small description of your startup idea.",
    key="user_query_input"
)

if st.button("Analyze Business Idea", key="analyze_button"):
    if user_input:
        st.subheader("🚀 Analysis in Progress...")

        # 1. User Intent Agent - Always run this first to get core info
        with st.spinner("Extracting your business intent..."):
            parsed_user_intent = extract(user_input)
            business_type = parsed_user_intent.get("business", "unknown")
            location_location = parsed_user_intent.get("location", "any")
            project_description = parsed_user_intent.get("description", "N/A")
            
            st.success("Intent extracted!")
            st.write(f"**Business Type:** {business_type}")
            st.write(f"**Location:** {location_location}")
            st.write(f"**Description:** {project_description}")

        if business_type == "unknown":
            st.warning("Please provide a clearer business type in your query to proceed with analysis.")
        else:
            # Conditional rendering based on sidebar selection
            if analysis_type == "Overall Business Analysis":
                st.markdown("---")
                st.subheader("📊 Overall Business Analysis (Market, Financial, Location & Strategy)")

                with st.spinner("Performing in-depth analysis... This may take a few moments."):
                    try:
                        analysis_result =  asyncio.run(combined_agent(user_input))
                        st.success(" Analysis complete!")

                        # Display extracted info
                        if isinstance(analysis_result, dict):
                            st.markdown(analysis_result)
                            st.markdown("#### 🌍 Market Trends")
                            st.markdown(analysis_result.get("market_analysis", "No data available."))
                            st.markdown("#### 💰 Competitors")
                            st.markdown(analysis_result.get("competitive_analysis", "No data available."))
                            st.markdown("#### 📈 Financial Overview")
                            st.markdown(analysis_result.get("financial_analysis", "No data available."))
                            st.markdown("#### 📝 Executive Summary")
                            st.markdown(analysis_result.get("executive_summary", "No data available."))

                    except Exception as e:
                        st.error(f"❌ Error occurred during analysis: {e}")
                

            elif analysis_type == "Market Research Analysis":
                st.markdown("---")
                st.subheader("📊 Market Research Analysis")
                with st.spinner("Performing market analysis... This might take a moment."):
                    market_analysis_result = asyncio.run(run_market_analysis_agent(business_type, location_location, project_description))
                    st.success("Market analysis complete!")

                    # ---- BEAUTIFY MARKET ANALYSIS REPORT ----
                    if isinstance(market_analysis_result, dict):
                        st.markdown("#### 🏢 Market Overview")
                        st.markdown(market_analysis_result.get("market_overview", "No data available."))

                        st.markdown("#### 🏆 Competitive Landscape")
                        st.markdown(market_analysis_result.get("competitive_landscape", "No data available."))

                        st.markdown("#### 🎯 Target Customers")
                        st.markdown(market_analysis_result.get("target_customers", "No data available."))

                        st.markdown("#### ⚖️ Regulatory Environment")
                        st.markdown(market_analysis_result.get("regulatory_environment", "No data available."))

                        st.markdown("#### 💡 SWOT Analysis")
                        st.markdown(market_analysis_result.get("swot_analysis", "No data available."))
                        

                        st.markdown("#### 🚀 Emerging Trends")
                        st.markdown(market_analysis_result.get("emerging_trends", "No data available."))

                        st.markdown("#### 📌 Key Recommendations")
                        st.markdown(market_analysis_result.get("key_recommendations", "No data available."))

                       
                    else:
                        st.markdown(market_analysis_result)

            elif analysis_type == "Competitive Analysis":

                st.markdown("---")
                st.subheader("🏁 Competitive Analysis")
                with st.spinner("Performing competitive analysis... This might take a moment."):
                    competitive_analysis_result = asyncio.run(run_competitive_analysis_agent(business_type, location_location, project_description))
                    st.success("Competitive analysis complete!")

                    if isinstance(competitive_analysis_result, dict):
                        st.markdown("#### 🏢 Key Competitors")
                        st.markdown(competitive_analysis_result.get("competitors", "No data available."))

                        st.markdown("#### 🧑‍💼 Competitor Profiles")
                        st.markdown(competitive_analysis_result.get("competitor_profiles", "No data available."))

                        st.markdown("#### 📊 Market Positioning")
                        st.markdown(competitive_analysis_result.get("market_positioning", "No data available."))

                        st.markdown("#### 💪 Strengths & Weaknesses")
                        st.markdown(competitive_analysis_result.get("strengths_weaknesses", "No data available."))

                        st.markdown("#### ⚡ Opportunities & Threats")
                        st.markdown(competitive_analysis_result.get("opportunities_threats", "No data available."))

                        st.markdown("#### 🧭 Strategic Recommendations")
                        st.markdown(competitive_analysis_result.get("strategic_recommendations", "No data available."))
                    else:
                        st.markdown(competitive_analysis_result)






            elif analysis_type == "Financial Analysis":
                st.markdown("---")
                st.subheader("💰 Financial Analysis")
                with st.spinner("Performing financial analysis..."):
                    # Assuming a default budget for now, as it's not extracted by 'extract'
                    financial_analysis_result = asyncio.run(financial_analysis(
                        business_type,
                        location_location,
                        project_description,
                    ))
                    st.success("Financial analysis complete!")

                    if isinstance(financial_analysis_result, dict):
                        st.markdown("#### 🏗️ Startup Costs")
                        st.markdown(financial_analysis_result.get("startup_costs", "No data available."))

                        st.markdown("#### 💵 Revenue Potential")
                        st.markdown(financial_analysis_result.get("revenue_potential", "No data available."))

                        st.markdown("#### 🏦 Funding Options")
                        st.markdown(financial_analysis_result.get("funding_options", "No data available."))

                        st.markdown("#### 📈 Profit Margins")
                        st.markdown(financial_analysis_result.get("profit_margins", "No data available."))

                        st.markdown("#### ⚠️ Financial Risks")
                        st.markdown(financial_analysis_result.get("financial_risks", "No data available."))

                        st.markdown("#### 🧭 Strategic Recommendations")
                        st.markdown(financial_analysis_result.get("strategic_recommendations", "No data available."))

                        
                    else:
                        st.markdown(financial_analysis_result)

    else:
        st.error("Please enter a business idea to get started!")

st.markdown("---")
st.info("Disclaimer: This tool provides AI-generated insights and should be used for informational purposes only. Consult with professionals for critical business decisions.")
