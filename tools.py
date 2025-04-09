from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.tools import Tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.tools.brave_search.tool import BraveSearch

ddg_tool = DuckDuckGoSearchResults()

results = ddg_tool.invoke("AVL Trees")
print(results)

# Document retrieval tool
retriever_tool = create_retriever_tool(
    retriever,
    name="document_search",
    description="Searches internal documents for relevant information."
)

# Web search tool
search = DuckDuckGoSearchResults()
search_tool = Tool(
    name="web_search",
    func=search.run,
    description="Useful for answering questions about current events or unknown topics."
)

tools = [retriever_tool, search_tool]