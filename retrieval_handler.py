from langchain.chains import RetrievalQA

def get_retrieval_qa_chain(llm_model, vector_store):
    """
    Get the retrieval QA chain
    """
    try:
        # Create the retrieval QA chain
        retrieval_qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,  # The LLM model
            chain_type="stuff",  # The chain type
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        return retrieval_qa_chain
        
    except Exception as e:
        print(f"Error in get_retrieval_qa_chain: {e}")
        return None
