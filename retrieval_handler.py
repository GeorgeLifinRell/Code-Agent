from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

def get_retrieval_qa_chain(llm_model, vector_store):
    """
    Get the retrieval QA chain
    """
    try:
        # Create the retrieval QA chain
        retrieval_qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm_model,  # The LLM model
            chain_type="stuff",  # The chain type
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        return retrieval_qa_chain
        
    except Exception as e:
        print(f"Error in get_retrieval_qa_chain: {e}")
        return None

if __name__ == '__main__':
    from llm_handler import get_llm
    from vectorization_handler import load_faiss_vector_store
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    from dotenv import load_dotenv
    import os

    load_dotenv()
    # Load the LLM model
    llm = get_llm("gpt2")

    hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_TOKEN"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load the vector store
    vector_store = load_faiss_vector_store(
        index_path="vector_store",
        hf_embeddings=hf_embeddings
    )
    
    # Get the retrieval QA chain
    retrieval_qa_chain = get_retrieval_qa_chain(llm, vector_store)
    
    # Query the retrieval QA chain
    query = "Hospital"
    response = retrieval_qa_chain.invoke(query)
    print(response)

    prompt_template = PromptTemplate(
        template="You are a digital assistant. You receive the following question: '{query}'. Please provide an answer with a source.",
        query_placeholder="{query}"
    )
    prompt = prompt_template.input_variables(query=query)
    chain = prompt | llm | retrieval_qa_chain
    response = chain.run()
    print(response)
