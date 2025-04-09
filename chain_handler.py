import warnings
import asyncio
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from agent_handler import get_llm

# Suppress all warnings
warnings.filterwarnings("ignore")

def get_chain_with_sources(model_name):
    try:
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=get_llm(model_name=model_name),
        )
    except:
        pass

async def get_llm_chain(model_name):
    """
    Get the LLM chain
    """
    try:
        prompt = ChatPromptTemplate.from_template(
            template=
            """
            You are a digital assistant. You receive the following message: '{query}'.
            Please respond.
            """,
            query_placeholder="{query}"
        )
        llm = get_llm("gpt2")
        llm_chain = prompt | llm | StrOutputParser()
        return llm_chain
    except Exception as e:
        print(f"Error in get_llm_chain: {e}")
        return None



if __name__ == '__main__':
    async def main():
        chain = await get_llm_chain("gpt2")
        query = "AVL Trees"
        async for response in chain.stream(query):
            print(response)
    asyncio.run(main())