import os
from dotenv import load_dotenv
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint

load_dotenv()

def get_llm(model_name):
    llm = HuggingFaceEndpoint(
        model=model_name,
        max_new_tokens=300,
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        task="text-generation"
    )
    return llm

def query_llm(llm, prompt):
    result = llm.invoke(prompt)
    return result
