import os
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field
#from openai import OpenAI
import logging
import json
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI as PandasAIOpenAI
from openai import OpenAI

class StreamlitCallback(BaseCallback):
    def __init__(self, container) -> None:
        """Initialize callback handler."""
        self.container = container

    def on_code(self, response: str):
        self.container.code(response)


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return



# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client for expert insights
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Data Models
class ChatMessage(BaseModel):
    """Model for storing chat messages"""
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class SupplyChainInsight(BaseModel):
    """Model for storing supply chain expert insights"""
    observation: str
    recommendation: str
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    supporting_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'expert_insights' not in st.session_state:
    st.session_state.expert_insights = []
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'smart_df' not in st.session_state:
    st.session_state.smart_df = None

def analyze_data(df: pd.DataFrame, query: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    try:
        # Initialize PandasAI with OpenAI
        llm = PandasAIOpenAI(api_token=os.environ["OPENAI_API_KEY"])
        
        # Create or get SmartDataframe
        if st.session_state.smart_df is None:
            st.session_state.smart_df = SmartDataframe(df, 
                                                       config={
                                                           "llm": llm,
                                                           "response_parser": StreamlitResponse
                                                       })
        
        # Get response from SmartDataframe
        response = st.session_state.smart_df.chat(query)
        
        # Check for generated chart
        additional_data = {}
        try:
            if os.path.exists("temp._chart.png"):
                with open("temp._chart.png", "rb") as img_file:
                    additional_data['chart'] = img_file.read()
        except Exception as e:
            logger.warning(f"Could not read chart file: {e}")
        
        return str(response), additional_data
    except Exception as e:
        logger.error(f"Error in data analysis: {e}")
        raise

def get_supply_chain_expert_insight(df: pd.DataFrame, query: str, chat_history: List[ChatMessage]) -> SupplyChainInsight:
    """Get expert insights using OpenAI"""
    try:
        # Prepare context from chat history
        chat_context = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history[-5:]])  # Last 5 messages
        
        # Create expert prompt
        expert_prompt = f"""
        As a supply chain expert, analyze the following data and provide insights about this query:
        
        Query: {query}
        
        Data Summary:
        {df.describe().to_string()}
        
        Recent Conversation Context:
        {chat_context}
        
        Please provide:
        1. Key observations about the supply chain performance related to the query
        2. Specific recommendations for improvement
        3. A confidence score (between 0 and 1) for your recommendations
        4. Supporting data points
        5. Follow-up questions to explore further
        
        Format your response in a clear, structured way.
        """
        
        # Get expert analysis from OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert supply chain analyst with deep knowledge of operations, logistics, and optimization. Provide detailed, actionable insights and maintain context from previous analyses."},
                {"role": "user", "content": expert_prompt}
            ]
        )
        
        insight_text = response.choices[0].message.content
        
        # Parse the response into structured format
        # You might want to add more sophisticated parsing here
        return SupplyChainInsight(
            observation=insight_text.split("Recommendations:")[0] if "Recommendations:" in insight_text else insight_text,
            recommendation=insight_text.split("Recommendations:")[1] if "Recommendations:" in insight_text else "",
            confidence_score=0.8,
            supporting_data={"raw_insight": insight_text}
        )
    except Exception as e:
        logger.error(f"Error in expert advisor: {e}")
        raise

def process_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Process uploaded file and return DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format")
        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="Supply Chain Analysis", layout="wide")
st.title("Supply Chain Analysis Dashboard 📊")

# File upload in sidebar
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload your supply chain data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        if st.session_state.current_df is None:
            st.session_state.current_df = process_uploaded_file(uploaded_file)
            if st.session_state.current_df is not None:
                st.success("File uploaded successfully!")
                st.dataframe(st.session_state.current_df.head())
                # Reset smart_df when new file is uploaded
                st.session_state.smart_df = None
    else:
        st.info("Please upload a file to begin analysis")

# Main content area
if st.session_state.current_df is not None:
    # Create tabs
    tab1, tab2 = st.tabs(["Interactive Chat", "Analysis History"])

    with tab1:
        st.header("Interactive Chat with Supply Chain Expert")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message.role):
                st.write(message.content)
        
        # Chat input
        if prompt := st.chat_input("Ask questions about your supply chain data"):
            # Add user message to chat history
            st.session_state.chat_history.append(ChatMessage(role="user", content=prompt))
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your data..."):
                    try:
                        # Use the analyze_data function
                        analysis_result, additional_data = analyze_data(st.session_state.current_df, prompt)
                        
                        # Display response
                        st.write("Analysis Result:")
                        st.write(analysis_result)
                        
                        # Display chart if present
                        if additional_data.get('chart'):
                            st.write("Generated Visualization:")
                            st.image(additional_data['chart'])
                        
                        # Get expert insights
                        expert_insight = get_supply_chain_expert_insight(
                            st.session_state.current_df,
                            prompt,
                            st.session_state.chat_history
                        )
                        st.session_state.expert_insights.append(expert_insight)
                        
                        st.write("\nExpert Insights:")
                        st.write(expert_insight.supporting_data["raw_insight"])
                        
                        # Add assistant's response to chat history
                        st.session_state.chat_history.append(
                            ChatMessage(role="assistant", content=analysis_result)
                        )
                        
                        # Clean up the temporary chart file
                        try:
                            if os.path.exists("temp._chart.png"):
                                os.remove("temp._chart.png")
                        except Exception as e:
                            logger.warning(f"Could not remove temporary chart file: {e}")
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        logger.error(f"Error in analysis: {e}")

    with tab2:
        st.header("Analysis History")
        
        # Display expert insights
        st.subheader("Expert Insights History")
        for insight in st.session_state.expert_insights:
            with st.expander(f"Insight from {insight.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"):
                st.write(f"**Observation:**\n{insight.observation}")
                st.write(f"**Recommendation:**\n{insight.recommendation}")
                st.write(f"**Confidence Score:** {insight.confidence_score:.2f}")
                if insight.supporting_data:
                    st.write("**Detailed Analysis:**")
                    st.write(insight.supporting_data["raw_insight"])
else:
    st.info("Please upload a file to begin analysis")