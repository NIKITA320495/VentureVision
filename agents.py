import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import Tool
from pydantic import BaseModel
import os

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

google_api_key_gemini = os.getenv("GOOGLE_API_KEY_GEMINI")
google_api_key_cse = os.getenv("GOOGLE_API_KEY_CSE")
google_cse_id_value = os.getenv("GOOGLE_CSE_ID")

class MarketAnalysisResponse(BaseModel):
    """
    Represents a comprehensive market analysis report.
    Each field corresponds to a specific section of the analysis.
    """
    market_overview: str
    competitive_landscape: str
    target_customers: str
    regulatory_environment: str
    swot_analysis: str
    emerging_trends: str
    key_recommendations: str

class CompetitiveAnalysisResponse(BaseModel):
    """
    Represents a detailed competitive analysis report.
    Each field corresponds to a specific aspect of the competitive landscape.
    """
    competitors: str
    competitor_profiles: str
    market_positioning: str
    strengths_weaknesses: str
    opportunities_threats: str
    strategic_recommendations: str


class FinancialAnalysisResponse(BaseModel):
    startup_costs: str 
    revenue_potential: str 
    funding_options: str 
    profit_margins: str 
    financial_risks: str 
    strategic_recommendations: str 

########################################################
############ User Intent Agent ##################
########################################################

def extract(text: str) -> dict:
    """
    Extracts information from a given text query, identifying business type,
    location, and generates a brief description in a structured JSON format.

    Args:
        text (str): The user's original query.

    Returns:
        dict: A dictionary containing the extracted 'business', 'location', and 'description'.
              Returns 'unknown' or 'any' for missing/unclear info, and includes an 'error' key if parsing fails.
    """
    tool_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        max_tokens=200,
        api_key=google_api_key_gemini
    )

    extraction_prompt = PromptTemplate(
        template="""
        From the user's input, perform the following two tasks:
        1. **Extract Information**:
           - **Startup**: Identify the type of business or startup the user wants to start.
           - **Location**: Identify any specific location mentioned.

        2. **Generate Description**:
           - **Description**: Based on the startup idea, generate a concise, brief description of the business idea. This should be a new generation if the user's input is very short.

        Always respond ONLY in pure JSON format, structured as follows:
        {{
            "business": "Bakery",
            "location": "Mumbai",
            "description": "A cozy bakery offering a variety of breads and pastries."
        }}

        If a piece of extracted information (business, location) is missing or unclear, use "unknown" for business and "any" for location. If no clear description can be generated, use "N/A" for description.

        User input: "{query}"
        """,
        input_variables=["query"],
    )

    try:
        extraction_chain = extraction_prompt | tool_llm
        response_message = extraction_chain.invoke({"query": text})
        
        llm_output_str = response_message.content.strip()

        # Clean up Markdown code block formatting if present
        if llm_output_str.startswith("```json"):
            llm_output_str = llm_output_str.replace("```json", "").replace("```", "").strip()
        elif llm_output_str.startswith("```") and llm_output_str.endswith("```"):
            llm_output_str = llm_output_str[3:-3].strip()

        extracted_data = json.loads(llm_output_str)
        return extracted_data
    except json.JSONDecodeError as e:
        st.error(f"Error in extract tool JSON decoding: {e}")
        return {"business": "unknown", "location": "any", "description": "N/A", "error": str(e), "raw_llm_output": llm_output_str}
    except Exception as e:
        st.error(f"An unexpected error occurred in extract tool: {e}")
        return {"business": "unknown", "location": "any", "description": "N/A", "error": str(e)}

#####################################################
######### Market Research Analysis Agent ############
#####################################################


# Define the search tool
search = GoogleSearchAPIWrapper(google_api_key=google_api_key_cse, google_cse_id=google_cse_id_value)
search_tool = Tool(
    name="search",
    func=search.run,
    description="Tool for performing Google searches to find information on the web."
)

def run_market_analysis_agent(business: str, location: str, description: str) -> dict:
    """
    Runs an AI agent to perform deep market research analysis based on the provided business type, location, and description.
    The agent uses a search tool to gather relevant market information and then synthesizes it.

    Args:
        business (str): The type of business (e.g., "AI marketing firm").
        location (str): The location for the business (e.g., "Mumbai").
        description (str): A brief description of the business idea.

    Returns:
        str: A comprehensive market analysis generated by the AI agent.
    """

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_output_tokens=1500,
        api_key=google_api_key_gemini
    )

    # Create the ReAct agent
    agent = create_react_agent(
        model=llm,
        tools=[search_tool],
        response_format=MarketAnalysisResponse,
        prompt="""
You are an elite market research analyst, renowned for delivering in-depth, data-driven, and actionable reports for new business ventures. Your mission is to provide a comprehensive market analysis for a proposed startup, using the latest available data and strategic insight.

Your responsibilities:
Analyze the provided business type, location, and business description to uncover critical market dynamics.
Synthesize findings into a clear, structured, and actionable report that empowers the entrepreneur to make informed decisions.


Use the 'search' tool to collect the most relevant, up-to-date, and credible information for each aspect of the market.
-   **Market Size & Growth Trends:** Current size, projected growth, and key drivers of the market.Include statistics, reports, and forecasts.
-   **Target Customer Demographics & Behavior:** Ideal customer profile, their needs, preferences, pain points, and purchasing habits .
-   **Regulatory Environment & Legal Considerations:** Relevant laws, licenses, permits, and industry-specific regulations impacting the business.
-   **SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats):** A detailed assessment specific to the startup within the identified market.
-   **Emerging Trends & Disruptors:** Any new technologies, consumer shifts, or innovations that could impact the market.

Your analysis report must be:
-   **Structured:** Organized with clear headings for each section (e.g., "Market Overview," "Competitive Analysis," "Target Audience," "Regulatory Environment," "SWOT Analysis," "Key Recommendations").
-  **Numerical:** Include relevant statistics, figures, and data points to support your analysis.
-   **Factual:** Supported by data and evidence found through your searches.
-   **Insightful:** Go beyond mere data aggregation to provide meaningful interpretations and implications for the startup.
-   **Detailed:** Cover all aspects of the market comprehensively, ensuring no critical information is overlooked.
-   **Actionable:** Conclude with clear, strategic recommendations that the entrepreneur can use to make informed decisions and navigate the market successfully.


You have access to the `search` tool to find all necessary information.Do not use any other tools or resources.Do not use your own knowledge or assumptions outside of the search results.

Your task is to perform the following steps:
**Steps:**
1.  **Understand & Deconstruct:** Fully grasp the startup's business type, specific location (if any), and the unique aspects of its description.
2.  **Formulate Precise Queries:** Generate highly targeted and efficient search queries to extract the most relevant data for each market aspect using the `search` tool.
3.  **Execute & Validate:** Perform the searches and critically evaluate the reliability and currency of the retrieved information.
4.  **Synthesize & Analyze:** Consolidate all gathered data, identifying patterns, opportunities, challenges, and key competitive factors.
5.  **Construct Report:** Write the comprehensive market analysis report following the structured format outlined below. Ensure your report directly addresses the entrepreneurial implications of your findings.

...
Your final output MUST be a JSON object that strictly conforms to the following schema:
```json
{
  "market_overview": "...",
  "target_customers": "...",
  "regulatory_environment": "...",
  "swot_analysis": "...",
  "emerging_trends": "...",
  "key_recommendations": "...",
}

**Instructions:**
-   Maintain a professional, objective, and data-driven tone.
-   For any market aspect where specific, reliable information cannot be found after thorough searching, clearly state "Information not available for this section" or similar, rather than omitting the section.
-   Prioritize insights that directly contribute to the entrepreneur's strategic decision-making.
-   Ensure the final output is only the market analysis report, without including your internal thought process, search queries, or intermediate steps.

        """
    )

    # Combine business info into a single input string
    agent_input = {
        "messages": [{"role": "user", "content": f"do an in-depth market analysis for the business {business} and location :{location} having description {description}."}],
    }

    try:
        result = agent.invoke(agent_input)
        structured = result.get('structured_response', result)
        if isinstance(structured, MarketAnalysisResponse):
            return structured.dict()
        elif isinstance(structured, dict):
            return MarketAnalysisResponse(**structured).dict()
        else:
            return {
                "market_overview": "Error: Unexpected response structure.",
                "competitive_landscape": "",
                "target_customers": "",
                "regulatory_environment": "",
                "swot_analysis": "",
                "emerging_trends": "",
                "key_recommendations": "",
               
            }
    except Exception as e:
        return {
            "market_overview": f"Error: {e}",
            "competitive_landscape": "",
            "target_customers": "",
            "regulatory_environment": "",
            "swot_analysis": "",
            "emerging_trends": "",
            "key_recommendations": "",
           
        }

################################################
############Competitive Analysis Agent ##########
################################################

COMPETITIVE_ANALYSIS_PROMPT = """
You are an elite competitive intelligence analyst, renowned for delivering in-depth, data-driven, and actionable competitive analysis reports for new business ventures. Your mission is to provide a comprehensive competitive analysis for a proposed startup, using only the latest available data and your strategic expertise.

Your responsibilities:
- Carefully analyze the provided business type, location, and business description to uncover the competitive landscape and unique opportunities or threats.
- Synthesize your findings into a clear, structured, and actionable report that empowers the entrepreneur to make informed, confident decisions.
- Use the `search` tool to gather the most relevant, up-to-date, and credible information for each aspect of the competitive landscape.

Your competitive analysis must thoroughly address the following aspects:
- Key Competitors: Identify and briefly describe the most significant direct and indirect competitors in the relevant market and location.
- Competitor Profiles: For each key competitor, provide details such as business model, product/service offerings, pricing, target customers, market share (if available), and unique selling propositions.
- Market Positioning: Analyze how the startup and its competitors are positioned in the market, including brand perception, customer segments, and competitive advantages/disadvantages.
- Strengths & Weaknesses: Summarize the main strengths and weaknesses of each competitor as they relate to the startup’s offering.
- Competitive Strategies: Outline the primary strategies employed by leading competitors (e.g., pricing, marketing, innovation, partnerships, customer service).
- Barriers to Entry: Identify major barriers to entry in the market, such as capital requirements, regulatory hurdles, established brand loyalty, or technological advantages.
- Opportunities & Threats: Highlight significant opportunities for differentiation as well as potential threats posed by competitors.
- Key Recommendations: Provide actionable, prioritized recommendations for how the startup can effectively compete and carve out a sustainable position in the market.
- Resources: List the most relevant and credible sources, links, or references used in your research. For each, include the title, URL, and a one-sentence summary of its relevance.

Your analysis report must be:
- Structured: Organized with clear, descriptive headings for each section.
- Factual: Supported by recent data, credible statistics, and direct evidence from your searches.
- Numerical: Include relevant statistics, figures, and data points to support your analysis wherever possible.
- Insightful: Go beyond summarizing facts—interpret what the data means for this specific business idea and location, highlighting key implications and opportunities.
- Actionable: Conclude with clear, strategic recommendations that the entrepreneur can use to make well-informed decisions and successfully compete in the market.

You have access to the `search` tool to find all necessary information. Do not use any other tools or resources. Do not use your own knowledge or assumptions outside of the search results.

Your task is to perform the following steps:
1. Understand & Deconstruct: Fully comprehend the startup's business type, specific location (if any), and the unique aspects of its description.
2. Formulate Precise Queries: Generate highly targeted and efficient search queries to extract the most relevant data for each competitive aspect using the `search` tool.
3. Execute & Validate: Perform the searches and critically evaluate the reliability, authority, and currency of the retrieved information.
4. Synthesize & Analyze: Integrate all gathered data, identifying patterns, opportunities, challenges, and key competitive factors.
5. Construct Report: Write the comprehensive competitive analysis report following the structured format outlined below. Ensure your report directly addresses the entrepreneurial implications of your findings.

Your final output MUST be a JSON object that strictly conforms to the following schema:
{
  "competitors": "...",
  "competitor_profiles": "...",
  "market_positioning": "...",
  "strengths_weaknesses": "...",
  "opportunities_threats": "...",
  "strategic_recommendations": "...",
}

Instructions:
- Maintain a professional, objective, and data-driven tone throughout your analysis.
- For any aspect where specific, reliable information cannot be found after thorough searching, clearly state "Information not available for this section" or a similar message, rather than omitting the section.
- Prioritize insights and recommendations that directly contribute to the entrepreneur's strategic decision-making.
- Ensure the final output is only the competitive analysis report in the specified JSON format, without including your internal thought process, search queries, or intermediate steps.
"""

def run_competitive_analysis_agent(business: str, location: str, description: str) -> dict:
    """
    Runs an AI agent to perform a comprehensive competitive analysis for a startup.
    Returns a structured dictionary matching the CompetitiveAnalysisResponse schema.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_output_tokens=1800,
        api_key=google_api_key_gemini
    )

    agent = create_react_agent(
        model=llm,
        tools=[search_tool],
        response_format=CompetitiveAnalysisResponse,
        prompt=COMPETITIVE_ANALYSIS_PROMPT
    )

    agent_input = {
        "messages": [{
            "role": "user",
            "content": f"Perform an in-depth competitive analysis for the business '{business}' in '{location}' with the following description: {description}"
        }],
    }

    try:
        result = agent.invoke(agent_input)
        structured = result.get('structured_response', result)
        if isinstance(structured, CompetitiveAnalysisResponse):
            return structured.dict()
        elif isinstance(structured, dict):
            return CompetitiveAnalysisResponse(**structured).dict()
        else:
            return {
                "competitors": "Error: Unexpected response structure.",
                "competitor_profiles": "",
                "market_positioning": "",
                "strengths_weaknesses": "",
                "opportunities_threats": "",
                "strategic_recommendations": "",
            }
    except Exception as e:
        return {
            "competitors": f"Error: {e}",
            "competitor_profiles": "",
            "market_positioning": "",
            "strengths_weaknesses": "",
            "opportunities_threats": "",
            "strategic_recommendations": "",
        }



############################################
######## Financial Analysis Agent ##########
############################################


FINANCIAL_ANALYSIS_PROMPT = """
You are a top-tier financial analyst for startups. Your job is to deliver a clear, data-driven, and actionable financial analysis for a new business idea.

Responsibilities:
- Analyze the business type, location, and description to assess financial viability.
- Use the `search` tool to gather the most relevant, up-to-date, and credible information for each aspect of the financial analysis.

Your report must cover:
- Startup Costs: Estimated costs to launch the business, with a breakdown if possible.
- Revenue Potential: Main revenue streams, market size, and realistic revenue estimates.
- Funding Options: Possible funding sources (e.g., loans, grants, investors) and strategies.
- Profit Margins: Expected profit margins and what affects them in this industry.
- Financial Risks: Major risks and how to mitigate them.
- Strategic Recommendations: Clear, prioritized financial actions for the entrepreneur.

Instructions:
- Structure your report with clear section headings as above.
- Use recent data, statistics, and evidence from your searches.
- If information for any section is unavailable, state "Information not available for this section."
- Output ONLY a JSON object matching this schema:
{
  "startup_costs": "...",
  "revenue_potential": "...",
  "funding_options": "...",
  "profit_margins": "...",
  "financial_risks": "...",
  "strategic_recommendations": "..."
}
Do not include your thought process or any content outside this JSON object.
"""

def financial_analysis(business: str, location: str, description: str, budget: str = "N/A") -> dict:
    # Example search queries for financial analysis
    queries = [
        f"startup costs for {business} in {location}",
        f"average revenue for {business} in {location}",
        f"funding options for {business} startup",
        f"profit margins for {business} industry"
    ]
    
    financial_results = []
    for query_str in queries:
        try:
            search_results = search_tool.run(query_str)
            financial_results.append(f"--- Search Results for '{query_str}' ---\n{search_results}\n")
        except Exception as e:
            st.warning(f"Error performing search for '{query_str}': {e}\n")

    combined_results = "\n".join(financial_results)

    financial_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        max_output_tokens=1500,
        api_key=google_api_key_gemini
    )

    finance_agent = create_react_agent(
        model=financial_llm,
        tools=[search_tool],
        response_format=FinancialAnalysisResponse,
        prompt=FINANCIAL_ANALYSIS_PROMPT
    )

    agent_input = {
        "messages": [{
            "role": "user",
            "content": f"Perform an in-depth financial analysis for the business '{business}' in '{location}' with the following description: {description}"
        }],
    }

    try:
        result = finance_agent.invoke(agent_input)
        structured = result.get('structured_response', result)
        print (f"Structured response: {structured}")
        if isinstance(structured, FinancialAnalysisResponse):
            return structured.dict()
        elif isinstance(structured, dict):
            return FinancialAnalysisResponse(**structured).dict()
        else:
            return {
                "startup_costs": "Error: Unexpected response structure.",
                "revenue_potential": "",
                "funding_options": "",
                "profit_margins": "",
                "financial_risks": "",
                "strategic_recommendations": "",
            }
    except Exception as e:
        return {
            "startup_costs": f"Error: {e}",
            "revenue_potential": "",
            "funding_options": "",
            "profit_margins": "",
            "financial_risks": "",
            "strategic_recommendations": "",
        }