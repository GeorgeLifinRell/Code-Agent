import dotenv
import os
import shutil
import streamlit as st
from ui_components import (
    sidebar,
    print_file_tree
)


dotenv.load_dotenv()
if os.path.exists("./repository"):
    shutil.rmtree("./repository")
os.makedirs("./repository")

# Streamlit UI
st.title("Repository Query Interface")

sidebar()

similarity_search_tab, llm_inference_tab = st.tabs([
    "🔎 Similarity search",
    "🤖 LLM Inference"
])
similarity_search_tab.subheader("Use the URL of the repository to do vector similarity search")
llm_inference_tab.subheader("LLM Inference")

with similarity_search_tab:
    similarity_search_tab()

with llm_inference_tab:
    llm_inference_tab()