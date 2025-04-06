"""
FILE: 3_chains_parallel.py
AUTHOR: Jesus Grillet

DESCRIPTION:
This script demonstrates the use of LangChain's Runnable components to create a 
parallelized chain for analyzing the features of a product and generating a review. 
The chain is designed to extract the main features of a product, analyze the pros 
and cons of those features, and combine the results into a final review.

The script utilizes the following components:
- ChatOllama: A language model for generating responses based on prompts.
- ChatPromptTemplate: A template for defining structured prompts.
- RunnableLambda: A wrapper for custom Python functions to be used in the chain.
- RunnableParallel: A component for running multiple branches of a chain in parallel.
- StrOutputParser: A parser for converting model outputs into strings.

The chain is structured as follows:
1. Extract the main features of a product using a prompt.
2. Analyze the pros and cons of the features in parallel branches.
3. Combine the results into a final review.

ENVIRONMENT VARIABLES:
- MODEL_NAME: The name of the ChatOllama model to use.
- BASE_URL: The base URL for the ChatOllama API.
- LANGSMITH_API_KEY: (Optional) API key for LangSmith.
- LANGSMITH_PROJECT: (Optional) Project name for LangSmith (default: "default").

USAGE:
Run the script to generate a review for a specified product. The product name 
is passed as input to the chain, and the final review is printed to the console.

DEPENDENCIES:
- python-dotenv: For loading environment variables from a .env file.
- langchain: For building and running the chain components.
- langchain_ollama: For interacting with the ChatOllama model.

EXAMPLE:
To generate a review for the "Oura ring", simply run the script. The output will 
include the pros and cons of the product based on its features.
"""
import os
import getpass
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

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

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert traveler and reviewer of cities. You are very good at finding the best spots and making itineraries."),
        ("human", "Can you give me the review and best to visit in {city_name}."),
    ]
)


# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human","Given these features: {features}, list the pros of these features."),
        ]
    )
    return pros_template.format_prompt(features=features)


# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human","Given these features: {features}, list the cons of these features."),
        ]
    )
    return cons_template.format_prompt(features=features)


# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


# Simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# Run the chain
#result = chain.invoke({"product_name": "Oura ring"})
result = chain.invoke({"city_name": "Porto, Portugal"})

# Output
print(result)