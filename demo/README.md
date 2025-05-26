# Demo User Guide

This project is the CAIA Agent Demo. It enables you to quickly use a simple Agent framework with web search to run questions from the dataset and generate data for evaluation. Below is a quick start guide:

## 1. Environment Setup

- **Python Version**: Python 3.12 or above is recommended.
- **Install Dependencies**  
  In the project root directory, run:
  ```
  pip install -r requirements.txt
  ```

## 2. Configure API Key

To use OpenAI or other LLM services, set the corresponding API Key in your environment variables. For example, when using OpenAI:

## 3. Configure Your LLM
Edit the **run_agent.py** file to specify which model to use.
Edit **llm.py** to configure how your model is used.

## 4. Run the Agent
