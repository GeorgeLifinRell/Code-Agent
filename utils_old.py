import os
import logging
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_parser(language):
    """
    Initialize a Tree-sitter parser for the given language.
    For .js and .ts files, simply chunk the code.
    """
    try:
        if language in ["javascript", "typescript"]:
            return None  # No parser needed, just chunk the code
        lang = get_language(language)
        if lang is None:
            raise ValueError(f"Language {language} is not supported by tree-sitter")
        parser = Parser()
        parser.set_language(lang)
        return parser
    except Exception as e:
        logging.error(f"Exception thrown: {e}")
        return None

def preprocess_repository(repo_path):
    """
    Convert every source code file in the repository to manageable text for chunking.
    """
    supported_files = ('.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.php', '.swift', '.rs')
    all_texts = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(supported_files):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    all_texts.append(text)

    return all_texts

def chunk_text(text, max_chunk_size):
    """
    Split the text into smaller chunks of a specified maximum size.
    """
    chunks = []
    for i in range(0, len(text), max_chunk_size):
        chunks.append(text[i:i + max_chunk_size])
    return chunks

def embed_and_store(chunks):
    """
    Embed the chunks and store them in a FAISS vector store.
    """
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Generate embeddings for the chunks
    embeddings = []
    for chunk in chunks:
        try:
            embedding = embeddings_model.embed(chunk)
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Error generating embedding for chunk '{chunk}': {e}")

    # Check if embeddings are created successfully
    if not embeddings or not embeddings[0]:
        logging.error('Error: Embeddings list is empty or invalid.')
        return None

    # Create FAISS vector store
    try:
        vector_store = FAISS.from_texts(chunks, embeddings)
    except IndexError as e:
        logging.error(f"IndexError: {e}")
        logging.error(f"Embeddings: {embeddings}")
        return None

    return vector_store

def save_faiss_index(vector_store, index_path):
    """
    Save the FAISS index to the specified path.
    """
    if vector_store is None:
        logging.error("Error: Vector store is None.")
        return

    vector_store.save_local(index_path)

def load_faiss_index(index_path, embeddings):
    """
    Load the FAISS index from the specified path.
    """
    vector_store = FAISS.load_local(index_path, embeddings)
    return vector_store

# Example usage
if __name__ == "__main__":
    repo_path = "./repository"
    max_chunk_size = 500
    index_path = "./index"

    # Preprocess repository
    all_texts = preprocess_repository(repo_path)
    logging.info("Preprocessed texts:")
    for idx, text in enumerate(all_texts):
        logging.info(f"Text {idx + 1}: {text[:100]}...")  # Print the first 100 characters of each text

    # Chunk texts
    all_chunks = []
    for text in all_texts:
        chunks = chunk_text(text, max_chunk_size)
        all_chunks.extend(chunks)

    print(f"All chunks: {all_chunks}")

    # Embed and store chunks
    vector_store = embed_and_store(all_chunks)
    if vector_store:
        save_faiss_index(vector_store, index_path)
        logging.info("FAISS index saved successfully!")

        # Load the FAISS index
        embedding_model = "all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        logging.info(f"Embeddings: {embeddings}")
        vector_store = load_faiss_index(index_path, embeddings)

        # Query the FAISS index
        query = "List all the functions in the repository."
        results = vector_store.similarity_search(query, k=5)

        logging.info("Top 5 results:")
        for result in results:
            logging.info(result)
    else:
        logging.error("Failed to create vector store.")