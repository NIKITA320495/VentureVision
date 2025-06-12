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
from typing import Dict, Any
from datetime import datetime
import asyncio 

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

class CombinedResponse(BaseModel):
    """
    Represents the combined response from the orchestrator agent.
    Contains the market, competitive, and financial analysis results.
    """
    market_analysis: str
    competitive_analysis: str
    financial_analysis: str
    executive_summary: str


def get_current_date() -> str:
    """Returns the current date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")

date_tool = Tool(
    name="current_date",
    func=get_current_date,
    description="Returns the current date. Useful for determining the recency of information."
)

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

def google_trends_insight(term: str, timeframe: str = 'today 12-m') -> str:
    """
    Provides search trend analysis for a given term using Google Trends data.

    Args:
        term (str): The keyword or business term to analyze.
        timeframe (str): Timeframe to fetch data (e.g., 'today 12-m' for last 12 months).

    Returns:
        str: A summary of the trend over the specified time.
    """
    try:
        pytrends = TrendReq(hl='en-US', tz=330)
        pytrends.build_payload([term], cat=0, timeframe=timeframe, geo='', gprop='')

        # Interest over time
        data = pytrends.interest_over_time()
        if data.empty:
            return f"No trend data found for '{term}'."

        latest = data[term].iloc[-1]
        peak = data[term].max()
        mean_trend = data[term].mean()

        trend_summary = f"""
 Google Trends Summary for **{term}** ({timeframe}):
- Peak Interest: {peak}
- Latest Interest Level: {latest}
- Average Interest: {mean_trend:.2f}
-  Timeframe: {timeframe}

The term '{term}' has shown a {'rising' if latest > mean_trend else 'declining'} trend in search interest recently.
        """.strip()

        return trend_summary

    except Exception as e:
        return f"Error fetching trend data for '{term}': {e}"
trendtool = Tool(
    name="googletrends",
    func=google_trends_insight,
    description="Tool for performing Google Trends analysis to find search interest trends for a given term."
)

async def run_market_analysis_agent(business: str, location: str, description: str) -> dict:
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
        tools=[search_tool,trendtool],
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
Use the 'googletrends' tool to analyze search interest trends for the business type and location.
-   **Emerging Trends & Disruptors:** Any new technologies, consumer shifts, or innovations that could impact the market.

Your analysis report must be:
-   **Structured:** Organized with clear headings for each section (e.g., "Market Overview," "Competitive Analysis," "Target Audience," "Regulatory Environment," "SWOT Analysis," "Key Recommendations").
-  **Numerical:** Include relevant statistics, figures, and data points to support your analysis.
-   **Factual:** Supported by data and evidence found through your searches.
-   **Insightful:** Go beyond mere data aggregation to provide meaningful interpretations and implications for the startup.
-   **Detailed:** Cover all aspects of the market comprehensively, ensuring no critical information is overlooked.
-   **Actionable:** Conclude with clear, strategic recommendations that the entrepreneur can use to make informed decisions and navigate the market successfully.


You have access to the `search` tool to find all necessary information and the `googletrends` tool to analyze search interest trends for the business type and location. Do not use your own knowledge or assumptions outside of the search results.

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
        result = await agent.ainvoke(agent_input)
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
- Strengths & Weaknesses: Summarize the main strengths and weaknesses of each competitor as they relate to the startups offering.
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

You have access to the `search` tool to find all necessary information. Do not use your own knowledge or assumptions outside of the search results.

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
}"

Instructions:
- Maintain a professional, objective, and data-driven tone throughout your analysis.
- For any aspect where specific, reliable information cannot be found after thorough searching, clearly state "Information not available for this section" or a similar message, rather than omitting the section.
- Prioritize insights and recommendations that directly contribute to the entrepreneur's strategic decision-making.
- Ensure the final output is only the competitive analysis report in the specified JSON format, without including your internal thought process, search queries, or intermediate steps.
"""

async def run_competitive_analysis_agent(business: str, location: str, description: str) -> dict:
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
        result = await agent.ainvoke(agent_input)
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
financesearch_tool = Tool(
    name="financesearch",
    func=search.run,
    description="Tool for performing financial searches to find information on startup costs, revenue potential, funding options, and profit margins."
)

FINANCIAL_ANALYSIS_PROMPT = """
You are a top-tier financial analyst for startups. Your job is to deliver a clear, data-driven, and actionable financial analysis for a new business idea.

Responsibilities:
- Analyze the business type, location, and description to assess financial viability.
- Use the `financesearch` tool to gather the most relevant, up-to-date, and credible information for each aspect of the financial analysis.
- **Crucially: Synthesize all information found in the search results. If precise, consolidated figures are not available, state what trends, ranges, or qualitative insights *are* present. Do not simply state "Information not available" if search results contain relevant details, even if indirect.**

Your report must cover:
- **Startup Costs:** Estimated costs to launch the business, with a breakdown. Estimated costs should be in numbers with a breakdown. 
- **Revenue Potential:** Main revenue streams, market size, and realistic revenue estimates all should be in numbers.
- **Funding Options:** List the specific funding sources available depending upon the business type and location, such as venture capital, angel investors, crowdfunding, or loans. Include specific funding amounts or ranges mentioned in the search results.
- **Profit Margins:** Expected profit margins and what affects them in this industry. Report any specific profit margin percentages, ranges, or qualitative descriptions found, and discuss factors like price volatility or feed costs if mentioned. Acknowledge any conflicting data and present the different perspectives found.
- **Financial Risks:** Major risks and how to mitigate them.
- **Strategic Recommendations:** Clear, prioritized financial actions for the entrepreneur, derived directly from your analysis of the search results. Write 3-4 actionable recommendations based on the financial analysis.

Instructions:
- Structure your report with clear section headings as above.
- **Use recent data, statistics, and direct evidence extracted from your searches.** Refer to specific findings from the search results.
- **If *no* relevant information can be found for a specific section *after thorough searching*, then and only then, state "Information not available for this section."**
- Output ONLY a JSON object matching this schema:
{
  "startup_costs": "...",
  "revenue_potential": "...",
  "funding_options": "...",
  "profit_margins": "...",
  "financial_risks": "...",
  "strategic_recommendations": "..."
}
Do not include your internal thought process, search queries, or any content outside this JSON object.
"""


async def financial_analysis(business: str, location: str, description: str) -> dict:
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
            print(f"Search results for '{query_str}': {search_results}")

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
        tools=[financesearch_tool],
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
        result = await finance_agent.ainvoke(agent_input)
        structured = result.get('structured_response', result)
        print (f"Structured response: {structured}")
        return structured.dict()
    except Exception as e:
        return {
            "startup_costs": f"Error: {e}",
            "revenue_potential": "",
            "funding_options": "",
            "profit_margins": "",
            "financial_risks": "",
            "strategic_recommendations": "",
        }

############################################
######## Combined Agent ##################
############################################


import asyncio


async def combined_agent(user_query: str) -> dict:
    """
    Orchestrator agent that:
    1. Extracts the business info.
    2. Runs market, competitive, and financial agents.
    3. Synthesizes all responses into a final executive summary.
    """
    # Step 1: Extract user intent
    extracted = extract(user_query)
    business = extracted.get("business")
    location = extracted.get("location")
    description = extracted.get("description", "N/A")

    if not business or not location or business == "unknown":
        return {
            "error": "Business type or location could not be extracted.",
            "extracted_info": extracted
        }

    # Step 2: Run sub-agents
    market = await run_market_analysis_agent(business, location, description)
    competition = await run_competitive_analysis_agent(business, location, description)
    financial = await financial_analysis(business, location, description)

    # Step 3: Define synthesizer agent
    synthesis_prompt = """
You are an expert business consultant. Your task is to write a 5-point executive summary for a proposed startup, based on its market, competitive, and financial analyses.

Business: {business}
Location: {location}
Description: {description}

Market:
{market}

Competition:
{competition}

Financial:
{financial}

**Output Task & Structure:**

Your final output MUST be a valid JSON object with four top-level keys: `market_analysis`, `competitive_analysis`, `financial_analysis`, and `executive_summary`.

1.  **`market_analysis` (JSON Key Content):**
    * Identify and elaborate on the top 2-3 most significant market opportunities derived from the market analysis.
    * Discuss the target audience, their unmet needs, and the potential for a viable customer base.
    * Cite specific statistics or market size estimates (e.g., "a projected market size of X billion by Y year").
    * This summary should be concise, approximately **50-60 words**, focusing on quantitative data and key insights.

2.  **`competitive_analysis` (JSON Key Content):**
    * Summarize the competitive landscape based on the competitive analysis .
    * Identify the top 2-3 primary competitors, describe their business models, and explain how they differentiate themselves.
    * Discuss their strengths and weaknesses.
    * Strategically position the startup against these competitors, highlighting its unique selling propositions or identifying underserved market gaps.

3.  **`financial_analysis` (JSON Key Content):**
    * Provide a concise overview of the startup's financial health and prospects.
    * Include details on initial costs, revenue potential, and specific funding requirements.
    * Summarize key financial projections (e.g., initial budget, projected revenue, breakeven point).
    * Mention any identified funding sources or requirements.
    * Briefly outline potential financial risks and proposed mitigation strategies.

4.  **`executive_summary` (JSON Key Content - This will be a single string containing the entire 5-point summary):**

    Structure this executive summary as follows, ensuring a logical flow and direct integration of insights from *all* provided analyses. Each point should be a distinct paragraph or clearly delineated section within the single string.

    * **1. Business Overview**: Provide a brief, compelling introduction to the startup, clearly stating its core offering, unique value proposition, and the problem it aims to solve.

    * **2. Market Opportunity**: Elaborate on the identified market opportunities. Discuss the target audience and their unmet needs, leveraging specific data and statistics directly from the `market_analysis` section to quantify the potential.

    * **3. Competitive Positioning**: Analyze the competitive landscape, discussing key competitors, their business models, and how the startup will strategically differentiate itself or exploit market gaps, drawing directly from the `competitive_analysis` section.

    * **4. Financial Viability & Outlook**: Detail the startup's financial health, including initial costs, revenue potential, and funding needs, based on the `financial_analysis` section. Include key financial projections (e.g., initial budget, projected revenue, breakeven point) and any identified financial risks with proposed mitigation.

    * **5. Strategic Recommendations**: Offer actionable and practical strategic recommendations for the founder. These recommendations should synthesize insights from all analyses, covering differentiation strategies, initial steps for market penetration, and guidance on product/service development (e.g., MVP).

---
Your final output MUST be a JSON object that strictly conforms to the following schema:

    {
    "market_analysis": "...",
    "competitive_analysis": "...",
    "financial_analysis": "...",
    "executive_summary": "..."
    }
"""

    summary_prompt = PromptTemplate(
        template=synthesis_prompt,
        input_variables=["business", "location", "description", "market", "competition", "financial"]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_output_tokens=1200,
        api_key=google_api_key_gemini
    )

    final_agent = create_react_agent(
        model=llm,
        tools=[],
        prompt=summary_prompt
    )

    summary_response = await final_agent.ainvoke({
            "business": business,
            "location": location,
            "description": description,
            "market": json.dumps(market, indent=2),
            "competition": json.dumps(competition, indent=2),
            "financial": json.dumps(financial, indent=2),
        })
    response= summary_response
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(response)
    return response