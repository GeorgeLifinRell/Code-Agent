import json
import os
import streamlit as st
from constants import REPOSITORY_PATH
from anytree.exporter import DictExporter
from repository_handler import build_file_tree
from repository_handler import (
    fetch_repository,
    split_repository_documents
)
from vectorization_handler import (
    generate_and_store_embeddings,
    load_faiss_vector_store
)
from llm_handler import (
    get_llm,
    query_llm
)
from constants import (
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
    AVAILABLE_LLMS
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

def sidebar_content():
    """Render sidebar content"""
    st.sidebar.header("Repository Settings")
    model_name = st.sidebar.selectbox(
        "Select Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2", "other-model-option"]
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
            fetch_repository(
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
    """
    Render the content for similarity search functionality
    """
    st.subheader("Use the URL of the repository to do vector similarity search")
    repo_url = st.text_input("Enter the repository URL")
    
    if repo_url:
        st.write(f"Repository URL: {repo_url}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Get Repository Tree"):
                print_file_tree(repo_url=repo_url)
                
        with col2:
            if st.button("Process Repository"):
                with st.spinner("Fetching repository..."):
                    try:
                        repo_documents = fetch_repository(repo_url)
                        if not repo_documents:
                            st.error("No documents found in repository")
                            return
                            
                        parsed_repo = split_repository_documents(repo_documents)
                        if parsed_repo:
                            st.success(f"Parsed repository contains {len(parsed_repo)} documents.")
                            
                            with st.spinner(f"Generating embeddings using {EMBEDDING_MODEL_NAME}..."):
                                vector_store = generate_and_store_embeddings(parsed_repo, EMBEDDING_MODEL_NAME)
                                if vector_store:
                                    vector_store.save_local(VECTOR_STORE_DIR)
                                    st.success("Embeddings generated and stored successfully!")
                                else:
                                    st.error("Failed to generate embeddings")
                        else:
                            st.error("Failed to parse the repository or repository is empty.")
                    except Exception as e:
                        st.error(f"Error processing repository: {str(e)}")

    # Query section
    if os.path.exists(VECTOR_STORE_DIR):
        query = st.text_input("Enter your query")
        if query:
            with st.spinner("Processing query..."):
                try:
                    embeddings = HuggingFaceInferenceAPIEmbeddings(
                        api_key=os.getenv("HF_TOKEN"),
                        model_name=EMBEDDING_MODEL_NAME
                    )
                    
                    # Verify embeddings are working
                    test_embedding = embeddings.embed_query("test")
                    if not test_embedding:
                        st.error("Failed to generate embeddings. Check your HF_TOKEN.")
                        return
                        
                    vector_store = FAISS.load_local(
                        VECTOR_STORE_DIR, 
                        embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    
                    similar_documents = vector_store.similarity_search(
                        query=query,
                        k=5  # Limit results to top 5
                    )
                    
                    if similar_documents:
                        st.subheader("Results by relevance:")
                        for i, document in enumerate(similar_documents, 1):
                            with st.expander(f"Result {i}"):
                                st.markdown(document.page_content)
                                st.json(document.metadata)
                    else:
                        st.info("No relevant documents found for your query.")
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

def llm_inference_content():
    """
    Render the content for LLM inference functionality
    """
    st.subheader("Chat with an LLM about the repository")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # LLM selection
    selected_llm = st.selectbox(
        label="💬 Select an LLM to chat with",
        options=AVAILABLE_LLMS,
        help="Choose the language model you want to interact with"
    )

    # Initialize LLM
    llm = get_llm(selected_llm)
    
    if llm:
        # Chat interface
        with st.container():
            # Display chat history
            for speaker, text in st.session_state.chat_history:
                message_placeholder = st.empty()
                with message_placeholder.container():
                    st.markdown(f"**{speaker}:** {text}")
            
            # Input area
            user_input = st.text_input(
                "Ask about the repository:",
                key="user_input",
                help="Enter your question about the repository"
            )
            
            # Send button with loading state
            if st.button("Send", type="primary"):
                if user_input:
                    with st.spinner("Generating response..."):
                        response_text = query_llm(llm, user_input)
                        st.session_state.chat_history.extend([
                            ("You", user_input),
                            ("Assistant", response_text)
                        ])
                        # Clear input after sending
                        st.session_state.user_input = ""
                else:
                    st.warning("Please enter a question first")
    else:
        st.error("Failed to initialize LLM. Please check your configuration.")

    # Add clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()