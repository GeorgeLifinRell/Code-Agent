import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint as HFE
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import AgentType
from langchain.agents.initialize import initialize_agent
from langchain.agents import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

def get_llm(model_name="google/gemma-3-27b-it"):
    """
    Create a language model using the newer HuggingFace InferenceClient approach
    """
    try:
        llm = HuggingFaceEndpoint(
            endpoint_url=f"https://api-inference.huggingface.co/models/{model_name}",
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            task="text-generation",
            max_new_tokens=256,
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
    
def get_tavily_search_tool():
    """
    Create a Tavily search tool
    """
    try:
        search_tool = TavilySearchResults(
            api_key=os.getenv("TAVILY_API_KEY"),
            description="Search for any topic using Tavily API"
        )
        return search_tool
    except Exception as e:
        print(f"Error initializing Tavily search tool: {e}")
        return None
    
def get_agent_with_llm_and_tools(llm, tools):
    """
    Create an agent with the provided LLM and tools
    """
    try:
        # Create tools
        tools = [Tool(name=tool['name'], func=tool['func'], description=tool['description']) for tool in tools]

        # Initialize agent
        agent = initialize_agent(
            tools,
            llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return None

if __name__ == '__main__':
    llm = get_llm("gpt2")
    if llm is None:
        print("Failed to get LLM")
        exit(1)
    print("LLM initialized successfully")
    prompt = "What is the most capable model of google?"
    tavily_search_tool = get_tavily_search_tool()
    if tavily_search_tool is None:
        print("Failed to get Tavily search tool")
        exit(1)
    print("Tavily search tool initialized successfully")
    agent = get_agent_with_llm_and_tools(
        llm,
        tools=[
            {
            'name': 'tavily_search',
            'func': tavily_search_tool.run,
            'description': 'Search for any topic using Tavily API'
            },
        ]
    )
    result = agent.invoke(prompt)
    if result is None:
        print("Failed to get agent result")
        exit(1)
    # result = query_llm(llm, prompt)
    print(result)
