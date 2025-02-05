from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="gpt-2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    max_new_tokens=500,
    temperature=0.7,
)

prompt = "Can You please explain me this codebase https://github.com/GeorgeLifinRell/D-Lottery.git"
response = llm.invoke(prompt)
print(response)

# System prompt engineering for better context
system_prompt = """You are an expert software engineer and code explainer.
You will be given a query and the codebase of a Github repository.
Your task is to provide a detailed and accurate explanation of the codebase
related to the query. Be specific and avoid generic answers.  Prioritize
explaining the functionality and architecture of the relevant parts of the
code. Explain any potential issues or improvements.  Reference specific
files and lines when possible.
"""

prompt = f"""{system_prompt}

Query: {query}
Codebase: https://github.com/GeorgeLifinRell/D-Lottery.git
"""
response = llm.invoke(prompt)
print(response)
