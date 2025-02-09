import os
import dotenv
import logging
import networkx as nx
import shutil
from anytree import Node
from constants import REPOSITORY_PATH
from langchain.schema import Document
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.git import GitLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
dotenv.load_dotenv()

# Define file extensions for supported languages
supported_extensions = [
    "*.py", "*.js",
    "*.ts", "*.java", 
    "*.c", "*.cpp", 
    "*.go", "*.rb", 
    "*.php", "*.rs"
]

LANGUAGE_MAPPING = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".rs": "rust",
    ".go": "go"
}

def fetch_repository(
        repo_url: str, 
        branch: str = "main", 
        file_filter: Optional[callable] = None) -> List[Document]:
    """
    Load a GitHub repository.
    
    Args:
        repo_url (str): URL of the repository to clone
        branch (str, optional): Branch to clone. Defaults to "main"
        file_filter (callable, optional): Function to filter files. Defaults to None
    
    Returns:
        List[Document]: List of documents from the repository
    """
    try:
        # Check if repository directory exists
        repo_path = Path(REPOSITORY_PATH)
        if repo_path.exists():
            # Remove directory if it exists but is empty
            if not any(repo_path.iterdir()):
                repo_path.rmdir()
                logging.info(f"Removed empty directory: {REPOSITORY_PATH}")
            else:
                # Remove existing repository to clone fresh
                shutil.rmtree(REPOSITORY_PATH)
                logging.info(f"Removed existing repository: {REPOSITORY_PATH}")
        
        # Create repository directory
        repo_path.mkdir(exist_ok=True)
        logging.info(f"Created repository directory: {REPOSITORY_PATH}")

        # Initialize GitLoader and clone repository
        loader = GitLoader(
            clone_url=repo_url,
            repo_path=REPOSITORY_PATH,
            branch=branch,
            file_filter=file_filter
        )
        documents = loader.load()
        logging.info(f"Successfully loaded {len(documents)} documents from repository")
        return documents
    except Exception as e:
        logging.error(f"Error loading repository: {e}")
        # Cleanup on error
        if Path(REPOSITORY_PATH).exists():
            shutil.rmtree(REPOSITORY_PATH)
            logging.info(f"Cleaned up repository directory after error: {REPOSITORY_PATH}")
        return []

def split_repository_documents(documents: List[Document]) -> List[Document]:
    """
    Load and parse a GitHub repository.
    """
    # Load the GitHub repository
    
    try:
        # Create Document objects with metadata
        parsed_documents = []
        for doc in documents:
            parsed_documents.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        "source": doc.metadata.get("source", ""),
                        "file_path": doc.metadata.get("file_path", ""),
                    }
                )
            )

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_documents = text_splitter.split_documents(parsed_documents)
        
        return split_documents
        
    except Exception as e:
        print(f"Error processing repository: {e}")
        return []

def build_file_tree(directory, parent=None):
    """ Recursively builds a file tree structure. """
    node = Node(os.path.basename(directory), parent=parent)
    
    if os.path.isdir(directory):
        for entry in sorted(os.listdir(directory)):
            build_file_tree(os.path.join(directory, entry), node)
    
    return node

def build_knowledge_graph(documents):
    """ Builds a knowledge graph from parsed documents. """
    # Initialize the knowledge graph
    graph = nx.DiGraph()
    
    # Add nodes and edges to the graph
    for doc in documents:
        if 'variables' in doc.metadata:
            for var in doc.metadata['variables']:
                graph.add_node(var, type='variable')
        
        if 'functions' in doc.metadata:
            for func in doc.metadata['functions']:
                graph.add_node(func, type='function')
        
        if 'classes' in doc.metadata:
            for cls in doc.metadata['classes']:
                graph.add_node(cls, type='class')
    
    return graph

def detect_language(file_path):
    """Detect the programming language based on file extension."""
    _, ext = os.path.splitext(file_path)
    return LANGUAGE_MAPPING.get(ext, None)

def parse_repository(directory: str) -> List[Document]:
    """
    Parses all source code files in a repository using LanguageParser.
    
    Args:
        directory (str): Directory containing the repository
    
    Returns:
        List[Document]: List of parsed documents
    """
    repo_documents = []
    repo_path = Path(directory)

    if not repo_path.exists():
        logging.error(f"Directory {directory} does not exist")
        return []

    for file_path in repo_path.rglob('*'):
        if file_path.is_file():
            language = detect_language(str(file_path))
            if language:
                try:
                    code = file_path.read_text(encoding="utf-8")
                    parser = LanguageParser()
                    parsed_code = parser.parse(code)

                    doc = Document(
                        page_content=str(parsed_code),
                        metadata={
                            "file_path": str(file_path),
                            "language": language
                        }
                    )
                    repo_documents.append(doc)
                    logging.debug(f"Successfully parsed {file_path}")
                except Exception as e:
                    logging.error(f"Error parsing {file_path}: {e}")

    return repo_documents

# if __name__ == "__main__":
#     repository_docs = fetch_repository(
#         "https://github.com/ajkulkarni/online-banking-application.git",
#         branch="master"
#     )
#     print(f"Repository contains {len(repository_docs)} documents.")
#     tree = build_file_tree(REPOSITORY_PATH)
#     print(tree)
    # parse_repository("./repository")
    # print("Repository parsing complete.")