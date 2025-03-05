import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

def get_llm(model_name):
    """
    Create a language model using the newer HuggingFace InferenceClient approach
    """
    try:
        llm = HuggingFaceEndpoint(
            endpoint_url=f"https://api-inference.huggingface.co/models/{model_name}",
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            task="text-generation",
            max_new_tokens=300,
            temperature=0.3,
            top_p=0.95
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

def query_llm(llm, prompt):
    """
    Query the language model with proper error handling
    """
    try:
        result = llm.invoke(prompt)
        return result
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return f"Error: {str(e)}"

def query_with_client(prompt, model_name="gpt2"):
    """
    Query using the recommended InferenceClient directly
    """
    try:
        client = InferenceClient(
            model=model_name,
            token=os.getenv("HF_TOKEN")
        )
        
        # Use the specific task method as recommended
        response = client.text_generation(
            prompt,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.95
        )
        return response
    except Exception as e:
        print(f"Error with InferenceClient: {e}")
        return f"Error: {str(e)}"

if __name__ == '__main__':
    llm = get_llm("gpt2")
    if llm is None:
        print("Failed to get LLM")
        exit(1)
    print("LLM initialized successfully")
    prompt = "What is the capital of France?"
    result = query_llm(llm, prompt)
    print(result)
