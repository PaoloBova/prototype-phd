import os

from dotenv import load_dotenv
# Load API key as environment variable from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAI LLM with logprobs enabled
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini").bind(logprobs=True)


json_schema = {
    "title": "FinalResponse",
    "description": "The final response from the agent.", 
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "decision": {"type": "integer"}
    },
    "required": ["reasoning", "decision"]
}
output_schema = json_schema["properties"]
# llm = llm.with_structured_output(output_schema)
prompt = f"You can choose between either Cooperate (1) or Defect (0) in the Prisoner's Dilemma. Think about what to do and make a choice. Respond only according to the following schema: {output_schema}"
msg = llm.invoke(prompt)

# print(msg)
# print(type(msg))

import json
import regex
from typing import Any, Dict, List

# msg.response_metadata["logprobs"]["content"][:5]


# # <!-- ruff: noqa: F821 -->
# from langchain_core.globals import set_llm_cache
# from langchain_community.cache import SQLiteCache

# set_llm_cache(SQLiteCache(database_path=".langchain.db"))


from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(temperature=0)
model= model.bind(logprobs=True, top_logprobs=2)

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model

result = chain.invoke({"query": joke_query})

print(f"Result: {result}")

usage_data = result.usage_metadata

# print(f"Parsed result: {parser.parse(result.content)}")
# print(f"Parser instructions: {parser.get_format_instructions()}")

logprobs_data = msg.response_metadata["logprobs"]
logprobs_content = logprobs_data["content"]
logprobs_refusal = logprobs_data["refusal"]

import prototype_phd.langchain_utils as langchain_utils


chat_completion_response = result
# add_logprobs expects a class instance with a choices attribute that is a
# list of choices. Each choice is essentially a response from langchain.
# So we need to wrap our response in a class that has a choices attribute.

from pydantic import BaseModel
from typing import Optional, Dict

class ChatCompletionTokenLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: Optional[Dict[str, float]] = None
    bytes: Optional[int] = None

class Message:
    def __init__(self, content: str):
        self.content = content

class Choice:
    def __init__(self, content: str, additional_kwargs: dict = None, response_metadata: dict = None):
        self.message = Message(content)
        self.additional_kwargs = additional_kwargs or {}
        self.response_metadata = response_metadata or {}

    @property
    def logprobs(self):
        # Return the logprobs from response_metadata if available.
        return self.response_metadata.get("logprobs", None)

class ChatResponse(BaseModel):
    choices: list

response_metadata = chat_completion_response.response_metadata
content = chat_completion_response.content
choice = Choice(content=content, response_metadata=response_metadata)
chat_completion_response = ChatResponse(choices=[choice])


# Convert the logprobs in response_metadata to a list of ChatCompletionTokenLogprob
# instances and add them to the response.
chat_completion
chat_completion_response = 
langchain_utils.add_logprobs(chat_completion_response)

# print(f"Logprobs refusal: {logprobs_refusal}")
# print(f"Logprobs content: {logprobs_content}")

# logprobs_content is a list of dictionaries that contain the token for that
# position,the log probability of that token, the bytes of the token, and the
# top_logprobs for that token. The top_logprobs is a list of dictionaries 
# containing the log probabilities of alternative tokens in this position. The
# structure of these dictionaries is the same as at the top level except that
# the top_logprobs field is missing.


# # Example dataset of questions
# questions = [
#     "What is the capital of France?",
#     "Who wrote '1984'?",
#     "What is the boiling point of water in Celsius?"
# ]

# for question in questions:
#     # Use generate to get a detailed result including the raw API response
#     result = llm.generate([question])
#     answer = result.generations[0][0].text.strip()
#     raw_output = result.llm_output  # this dictionary holds the raw API response

#     print("Question:", question)
#     print("Answer:", answer)
    
#     # Check if the raw output contains logprobs and print them
#     if raw_output and "choices" in raw_output:
#         # Typically, the first choice will include the logprobs if available.
#         logprobs = raw_output["choices"][0].get("logprobs")
#         if logprobs:
#             print("Logprobs:", logprobs)
#         else:
#             print("Logprobs not returned in the response.")
#     else:
#         print("No raw output available.")
#     print("-" * 50)


