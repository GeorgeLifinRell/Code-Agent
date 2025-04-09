import os
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from repository_handler import load_repository_documents
from vectorization_handler import create_and_save_faiss_vectorstore
from agent_handler import get_llm

processed_docs = load_repository_documents(
    repo_url="https://github.com/GeorgeLifinRell/helper-app.git"
)

hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
                    api_key=os.getenv("HF_TOKEN"),
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

vector_store = create_and_save_faiss_vectorstore(documents=processed_docs, hf_embeddings=hf_embeddings, index_path="index")
retriever = vector_store.as_retriever()

llm = get_llm()

chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
result = chain.invoke({"question": "What is this project about?"})
print(result)
