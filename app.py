import dotenv
import os
import shutil
import streamlit as st
from ui_components import (
    sidebar_content, 
    similarity_search_content, 
    llm_inference_content,
    rag_chain_inference_content
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
similarity_search_tab, llm_inference_tab, rag_chain_inference_tab = st.tabs([
    "🔎 Similarity search",
    "🤖 LLM Inference",
    "🧵 RAG Chain Inference"
])

# Similarity Search tab content
with similarity_search_tab:
    similarity_search_content()

# LLM Inference tab content
with llm_inference_tab:
    llm_inference_content()

# RAG Chain Inference tab content
with rag_chain_inference_tab:
    st.write("RAG Chain Inference content goes here...")
    rag_chain_inference_content()
