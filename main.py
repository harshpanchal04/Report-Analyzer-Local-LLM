import os
import logging
from dotenv import load_dotenv
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# Constants for API keys and settings
SERPER_API_KEY = os.getenv("Serper-API-Key")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the local LLM
def initialize_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm

def initialize_agents(llm):
    """Initialize and return the agents used in the application."""
    medical_analyst = Agent(
        role='Medical Analyst',
        goal='Analyze the blood test report and provide a summary in simple terms.',
        backstory="An expert in interpreting medical data and explaining it to non-medical people.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    health_researcher = Agent(
        role='Health Researcher',
        goal='Search for articles based on the blood test analysis.',
        backstory="Skilled at finding accurate and relevant health information online.",
        verbose=True,
        allow_delegation=False,
        tools=[SerperDevTool(api_key=SERPER_API_KEY)],
        llm=llm
    )

    health_advisor = Agent(
        role='Health Advisor',
        goal='Provide health recommendations based on the articles and blood test summary.',
        backstory="Experienced in providing personalized health advice.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    return medical_analyst, health_researcher, health_advisor

def initialize_tasks(medical_analyst, health_researcher, health_advisor):
    """Initialize and return the tasks used in the application."""
    analyze_blood_test = Task(
        description='Analyze the extracted text from the blood test report and provide a summary.',
        expected_output='A concise summary of the blood test results in simple terms.',
        agent=medical_analyst
    )

    search_for_articles = Task(
        description='Search for articles relevant to the health issues identified in the blood test summary.',
        expected_output='A list of relevant articles with key insights.',
        agent=health_researcher
    )

    provide_recommendations = Task(
        description='Provide health recommendations based on the articles and blood test summary.',
        expected_output='Personalized health advice based on the analysis and research.',
        agent=health_advisor
    )

    return [analyze_blood_test, search_for_articles, provide_recommendations]


def main():
    st.title("Blood Test Report Analysis (Using Local LLM)")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from PDF
        text = ""
        try:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading the PDF file: {e}")
            logger.error(f"Error reading the PDF file: {e}")
            return

        if st.button("Analyze Health Report"):
            st.write("Analyzing the report... This may take a few minutes.")
            logger.info("Report analysis started")

            # Initialize LLM
            llm = initialize_llm()

            # Initialize agents and tasks
            medical_analyst, health_researcher, health_advisor = initialize_agents(llm)
            tasks = initialize_tasks(medical_analyst, health_researcher, health_advisor)

            # Form the crew and define the process
            crew = Crew(
                agents=[medical_analyst, health_researcher, health_advisor],
                tasks=tasks,
                process=Process.sequential
            )

            # Kick off the crew process with the extracted text
            with st.spinner("Processing..."):
                try:
                    result = crew.kickoff(inputs={"text": text})
                    logger.info("Report analysis completed successfully")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    logger.error(f"An error occurred during analysis: {e}")
                    return

            # Display results
            st.subheader("Analysis Results")
            st.markdown(result)
    else:
        st.write("Please upload a PDF file to begin analysis.")

if __name__ == "__main__":
    main()