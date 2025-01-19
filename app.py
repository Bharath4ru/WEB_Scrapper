import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from bs4 import BeautifulSoup
import requests
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

import os

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="Web URL Summarizer Agent",
    page_icon="🌐",
    layout="wide"
)

st.title("Phidata Web URL Summarizer Agent 🌐📝")
st.header("Powered by Gemini 2.0 Flash Exp")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Web Content Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

def fetch_webpage_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        return f"Error fetching content: {str(e)}"

# Initialize the agent
multimodal_Agent = initialize_agent()

# URL input
url = st.text_input(
    "Enter Website URL",
    placeholder="https://example.com",
    help="Enter the URL of the webpage you want to analyze"
)

user_query = st.text_area(
    "What insights are you seeking from the webpage?",
    placeholder="Ask anything about the webpage content. The AI agent will analyze and gather additional context if needed.",
    help="Provide specific questions or insights you want from the content."
)

if st.button("🔍 Analyze Content", key="analyze_content_button"):
    if not url:
        st.warning("Please enter a URL to analyze.")
    elif not user_query:
        st.warning("Please enter a question or insight to analyze the content.")
    else:
        try:
            with st.spinner("Fetching webpage content and gathering insights..."):
                # Fetch webpage content
                webpage_content = fetch_webpage_content(url)
                
                # Prompt generation for analysis
                analysis_prompt = f"""
                Analyze the following webpage content and respond to the user's query:
                
                Webpage URL: {url}
                User Query: {user_query}
                
                Content:
                {webpage_content[:8000]}  # Limiting content length to avoid token limits
                
                Provide a detailed, well-structured, and actionable response that directly addresses the user's query.
                Include relevant quotes or examples from the webpage when appropriate.
                """
                
                # AI agent processing
                response = multimodal_Agent.run(analysis_prompt)
                
                # Display the result
                st.subheader("Analysis Result")
                st.markdown(response.content)
                
        except Exception as error:
            st.error(f"An error occurred during analysis: {error}")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
