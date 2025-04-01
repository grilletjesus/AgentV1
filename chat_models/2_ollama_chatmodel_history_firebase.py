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
print("Current Chat History:", chat_history.messages)

#Instance local AI LLM
model = ChatOllama(
    model= os.getenv("MODEL_NAME"),
    temperature=0,
    verbose=True,
    base_url=os.getenv("BASE_URL")
)


