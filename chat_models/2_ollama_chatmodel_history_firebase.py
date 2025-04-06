"""
File: 2_ollama_chatmodel_history_firebase.py
Description: This script demonstrates how to integrate a local AI language model with Firebase Firestore
Date: 2025-04-01
Author: Jesus Grillet
Version: 1.0

Features:
- Integration with Firebase Firestore for chat history persistence.
- Interaction with a local AI language model using LangChain.
- Environment variable configuration for flexible deployment.

Dependencies:
- Python 3.8+
- `google-cloud-firestore` for Firestore integration.
- `langchain` for AI model interaction.
- `python-dotenv` for environment variable management.
- Google Cloud CLI for Firestore setup and authentication.

Usage:
1. Configure the `.env` file with the required environment variables:
    - `PROJECT_ID`: Your Firebase project ID.
    - `SESSION_ID`: A unique identifier for the chat session.
    - `COLLECTION_NAME`: The Firestore collection name for storing chat messages.
    - `MODEL_NAME`: The name of the local AI model to use.
    - `BASE_URL`: The base URL for the local AI model API.
2. Run the script and start interacting with the AI model.

Note:
Ensure that the Google Cloud CLI is authenticated and configured with the correct project 
before running this script.


Description:
This script demonstrates how to integrate a local AI language model with Firebase Firestore 
to maintain a chat history. It uses the LangChain framework for AI model interaction and 
Google Firestore for persistent chat message storage.

Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project
    - Copy the project ID
3. Create a Firestore database in the Firebase project
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""
import os
import getpass
from typing import List
from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    if "LANGSMITH_API_KEY" not in os.environ:
        os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
            prompt="Enter your LangSmith API key (optional): "
        )
    if "LANGSMITH_PROJECT" not in os.environ:
        os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
            prompt='Enter your LangSmith Project Name (default = "default"): '
        )
        if not os.environ.get("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "default"
    pass

# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=os.getenv("PROJECT_ID"))

# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=os.getenv("SESSION_ID"),
    collection=os.getenv("COLLECTION_NAME"),
    client=client,
)
print("Chat History Initialized.")
#print("Current Chat History:", chat_history.messages)

#Instance local AI LLM
model = ChatOllama(
    model= os.getenv("MODEL_NAME"),
    temperature=0,
    verbose=True,
    base_url=os.getenv("BASE_URL")
)

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
