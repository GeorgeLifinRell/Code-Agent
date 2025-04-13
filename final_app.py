import os
import dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from repository_handler import load_repository_documents
from vectorization_handler import get_vector_store
from agent_handler import get_llm

dotenv.load_dotenv()
processed_docs = load_repository_documents(
    repo_url="https://github.com/cyclotruc/gitingest"
)
hf = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("SRIRAM_HF_TOKEN")
)
vector_store = get_vector_store(processed_docs, hf_embeddings=hf)
if vector_store:
    print("vector store created")
    retriever = vector_store.as_retriever()
llm = get_llm()
if not llm:
    print("LLM init failed!")
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever
)
result = chain.invoke({"question": "What is this project about?"})
print(result)
