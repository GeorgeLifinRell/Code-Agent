import dotenv
import os
import shutil
import streamlit as st
from ui_components import (
    sidebar_content, 
    similarity_search_content, 
    llm_inference_content
)

dotenv.load_dotenv()
if os.path.exists("./repository"):
    shutil.rmtree("./repository")
os.makedirs("./repository")

# Streamlit UI
st.title("Repository Query Interface")

# Sidebar
st.sidebar.title("Settings")
sidebar_content()

# Create tabs
similarity_search_tab, llm_inference_tab = st.tabs([
    "🔎 Similarity search",
    "🤖 LLM Inference"
])

# Similarity Search tab content
with similarity_search_tab:
    similarity_search_content()
    # st.subheader("Use the URL of the repository to do vector similarity search")
    # repo_url = st.text_input("Repository URL")
    # if st.button("Search"):
    #     if repo_url:
    #         st.info("Processing repository...")
    #         # Add your similarity search logic here
    #     else:
    #         st.error("Please enter a repository URL")

# LLM Inference tab content
with llm_inference_tab:
    llm_inference_content()
    # st.subheader("LLM Inference")
    # query = st.text_area("Enter your query")
    # if st.button("Generate Response"):
    #     if query:
    #         st.info("Generating response...")
    #         # Add your LLM inference logic here
    #     else:
    #         st.error("Please enter a query")