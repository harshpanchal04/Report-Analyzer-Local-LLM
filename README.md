# Blood Test Report Analysis (Using Local LLM)

This project is a Streamlit-based application that leverages a local instance of Llama 2 along with multi-agent AI architecture to analyze blood test reports from PDFs, retrieve relevant health articles, and provide personalized health recommendations.

## Project Overview

The application reads a PDF file containing a blood test report, extracts the text, and runs an analysis to provide a user-friendly summary. Additionally, it searches for relevant health articles and offers personalized health recommendations based on the report. The entire analysis is powered by a local Large Language Model (LLM) using `LlamaCpp` and a multi-agent AI system configured with CrewAI.

## Features

- **PDF Text Extraction**: Extracts text from uploaded blood test reports in PDF format.
- **Medical Analysis**: Summarizes complex medical information into simple terms.
- **Health Research**: Searches for relevant articles based on the analysis.
- **Personalized Health Recommendations**: Provides advice tailored to the health report.
- **Local LLM Integration**: Runs a locally hosted LLM using Llama 2 (7B model).

## Setup Instructions

### Prerequisites

- Python 3.8 or later
- `pip` for managing Python packages

### Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harshpanchal04/Report-Analyzer-Local-LLM.git
    cd Report-Analyzer-Local-LLM
    ```

2. **Set Up a Virtual Environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the Llama 2 Model**:
   - Place the `llama-2-7b-chat.Q4_K_M.gguf` model in the `models/` directory.

5. **Set Up Environment Variables**:
   - Create a `.env` file and add your Serper API key:
     ```plaintext
     Serper-API-Key=your_serper_api_key
     ```

## How to Use

1. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

2. **Upload a PDF**: Once the app is running, upload a PDF file containing the blood test report.

3. **Analyze the Report**: Click the "Analyze Health Report" button and wait for the analysis to complete.

4. **View Results**: The application will display a summary of the blood test, relevant articles, and health recommendations.

## Technologies Used

- **Python**: Core language for the project.
- **Streamlit**: Web framework for building the interactive user interface.
- **LlamaCpp**: For running the Llama 2 model locally.
- **CrewAI**: Agent-based architecture for handling tasks.
- **Serper API**: Tool for searching online articles.
- **PyPDF**: Library for PDF text extraction.
- **dotenv**: For managing environment variables.

## Project Structure

├── main.py # Main application script
├── models/ # Directory to store Llama model files
├── .env # Environment variables
├── requirements.txt # Python dependencies
└── README.md # Project documentation
