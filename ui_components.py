import json
import os
import dotenv
import numpy as np
import streamlit as st
from constants import REPOSITORY_PATH
from anytree.exporter import DictExporter
from repository_handler import build_file_tree
from repository_handler import (
    load_repository_documents,
)
from vectorization_handler import (
    get_vector_store
)
from agent_handler import (
    get_llm,
    query_llm
)
from constants import (
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
    AVAILABLE_LLMS
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

dotenv.load_dotenv()

hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HF_TOKEN"),
            # model_name=EMBEDDING_MODEL_NAME
            )

def sidebar_content():
    """Render sidebar content"""
    st.sidebar.header("Repository Settings")
    model_name = st.sidebar.selectbox(
        "Select Embedding Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2", 
            "microsoft/codebert-base",
            "microsoft/codebert-base-mlm",
        ]
    )
    
    chunk_size = st.sidebar.slider(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )
    
    return {
        "model_name": model_name,
        "chunk_size": chunk_size
    }

def print_file_tree(repo_path=REPOSITORY_PATH, repo_url=None):
    st.write(f"Repository path: {repo_path}")
    if os.listdir(repo_path) == []:
        try:
            load_repository_documents(
                repo_url=repo_url
            )
        except Exception as e:
            st.error(f"Error fetching repository: {e}")
            return
    file_tree = build_file_tree(repo_path)
    exporter = DictExporter()
    print(exporter.export(file_tree))
    st.json(json.dumps(exporter.export(file_tree), indent=2))

def similarity_search_content():
    """Render the content for similarity search functionality"""
    st.subheader("Use the URL of the repository to do vector similarity search")
    # repo_url = st.text_input("Enter the repository URL")
    # branch = st.text_input("Enter the branch name", value="main")
    
    # if repo_url:
    #     # Process repository
    #     if st.button("Process Repository"):
    #         with st.spinner("Processing repository..."):
    #             try:
    #                 repo_documents = load_repository_documents(repo_url, branch=branch)
    #                 if repo_documents:
    #                     # Show document preview
    #                     st.success(f"Found {len(repo_documents)} documents in repository")
    #                     with st.expander("Preview First 5 Documents"):
    #                         for i, doc in enumerate(repo_documents[:5]):
    #                             st.markdown(f"**Document {i+1}**")
    #                             st.markdown(f"*Path:* `{doc.metadata.get('source', 'Unknown')}`")
    #                             st.markdown("*Content Preview:*")
    #                             st.code(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    #                             st.markdown("---")
                        
    #                     embeddings = generate_and_save_document_embeddings(
    #                         processed_documents=repo_documents,
    #                         hf_embeddings=hf_embeddings
    #                     )
    #                     if not embeddings:
    #                         st.error("Failed to generate embeddings")
    #                         return
    #                     st.success("Embeddings generated successfully!")
    #                     vector_store = create_and_save_faiss_vectorstore(
    #                         documents=repo_documents,
    #                         hf_embeddings=hf_embeddings,
    #                         index_path=VECTOR_STORE_DIR
    #                     )
    #                     if not vector_store:
    #                         st.error("Failed to create vector store")
    #                         return
    #                     st.success("Vector store created successfully!")
    #                     st.write(f"Vector store saved at: {VECTOR_STORE_DIR}")
    #                     st.write(f"Number of embeddings: {len(embeddings)}")
    #                     st.write(f"Number of documents: {len(repo_documents)}")
    #             except Exception as e:
    #                 st.error(f"Error: {e}")
        
    #     # Search functionality
    #     query = st.text_input("Enter your search query")
    #     if query and st.button("Search"):
    #         with st.spinner("Searching..."):
    #             try:
    #                 # Load index
    #                 vector_store = load_faiss_vector_store(
    #                     index_path=VECTOR_STORE_DIR,
    #                     hf_embeddings=hf_embeddings
    #                 )
    #                 if not vector_store:
    #                     st.error("Failed to load vector store")
    #                     return
    #                 # Perform search
    #                 results = vector_store.similarity_search(
    #                     query=query,
    #                     k=5
    #                 )
    #                 if not results:
    #                     st.error("No results found")
    #                     return
    #                 st.success(f"Found {len(results)} results")
    #                 for i, doc in enumerate(results):
    #                     st.markdown(f"**Result {i+1}**")
    #                     st.markdown(f"*Path:* `{doc.metadata.get('source', 'Unknown')}`")
    #                     st.markdown("*Content Preview:*")
    #                     st.code(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    #                     st.markdown("---")
    #             except Exception as e:
    #                 st.error(f"Search error: {str(e)}")

def llm_inference_content():
    """
    Render the content for LLM inference functionality
    """
    st.subheader("Chat with an LLM about the repository")
    
    # Initialize chat history in session state
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = []
    
    # # Initialize message for next run
    # if "message" not in st.session_state:
    #     st.session_state.message = ""

    # LLM selection
    # selected_llm = st.selectbox(
    #     label="💬 Select an LLM to chat with",
    #     options=AVAILABLE_LLMS,
    #     help="Choose the language model you want to interact with"
    # )

    # Initialize LLM
    # llm = get_llm(selected_llm)
    # if llm:
    #     st.success("LLM initialized successfully")
    #     # Chat interface
    #     with st.container():
    #         # Display chat history
    #         for speaker, text in st.session_state.chat_history:
    #             with st.chat_message(speaker.lower()):
    #                 st.markdown(text)
            
    #         # Input area with callback
    #         def on_input_change():
    #             if st.session_state.user_input:
    #                 with st.spinner("Generating response..."):
    #                     response_text = query_llm(llm, st.session_state.user_input)
    #                     st.session_state.chat_history.extend([
    #                         ("You", st.session_state.user_input),
    #                         ("Assistant", response_text)
    #                     ])
    #                     st.session_state.message = "Message sent!"
            
    #         user_input = st.text_input(
    #             "Ask about the repository:",
    #             key="user_input",
    #             help="Enter your question about the repository",
    #             on_change=on_input_change
    #         )
            
    #         if st.session_state.message:
    #             st.success(st.session_state.message)
    #             st.session_state.message = ""
    # else:
    #     st.error("Failed to initialize LLM. Please check your configuration.")

    # # Add clear chat button
    # if st.button("Clear Chat History"):
    #     st.session_state.chat_history = []
    #     st.session_state.message = "Chat history cleared!"

def rag_chain_inference_content():
    st.subheader("Chat with an LLM about the repository")
    repo_url = st.text_input("Enter the repository URL")
    branch = st.text_input("Enter the branch name", value="main")
    selected_llm = st.selectbox(
        label="💬 Select an LLM to chat with",
        options=AVAILABLE_LLMS,
        help="Choose the language model you want to interact with"
    )
    if repo_url and branch:
        with st.spinner("Processing repository..."):
            processed_docs = load_repository_documents(repo_url=repo_url, branch=branch)
            if len(processed_docs) > 0:
                st.success("Repository loaded successfully!", icon="✌️")
            else:
                st.error("Fetching repo failed", icon="😭")
                return
            if hf_embeddings:
                with st.spinner("Generating embeddings and FAISS vector store..."):
                    vector_store = get_vector_store(processed_docs, hf_embeddings=hf_embeddings)
                    if vector_store:
                        st.success("FAISS vector store created successfully!", icon="🗂️")
                    else:
                        st.error("FAISS creation failed", icon="🥶")
                        return
                    with st.spinner("Creating retriever..."):
                        retriever = vector_store.as_retriever()
                        if retriever:
                            st.success("Retriever created successfully", icon="👨‍⚕️")
                        else:
                            st.error("Retriever creation failed", icon="🚨")
                            return
                        with st.spinner("Creating an llm instance..."):
                            llm = get_llm(model_name=selected_llm)
                            if llm:
                                st.success("LLM instance loaded", icon="⭐️")
                            else:
                                st.error("LLM inference failed")
                                return
                        chain = RetrievalQAWithSourcesChain.from_chain_type(
                            llm=llm, chain_type="stuff", retriever=retriever
                        )
                        if chain:
                            user_question = st.chat_input(placeholder="Your question")
                            if user_question:
                                result = chain.invoke({"question": user_question})
                                st.write(result)
