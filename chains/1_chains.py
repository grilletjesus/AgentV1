import os
import getpass
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser

#   Load environment variables from .env file (requires `python-dotenv`)
try:
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

#   Initialize ChatOllama model
model = ChatOllama(
    model=os.getenv("MODEL_NAME"),
    temperature=0,
    verbose=True,
    base_url=os.getenv("BASE_URL"),
)

#   Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You can answer questions precisely to the best of your ability."),
        ("human", "Tell me about {topic}. Please provide a detailed answer but no more than 500 words."),
    ]
)

#   Create the combined chain using LCEL
chain = prompt_template | model | StrOutputParser()

#   Run the chain
result = chain.invoke({"topic": "Python programming"})

#   Output
print(result)