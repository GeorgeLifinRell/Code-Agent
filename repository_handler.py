import os
import dotenv
import logging
import sys
from anytree import Node
from langchain.schema import Document
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.git import GitLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
dotenv.load_dotenv()
REPOSITORY_PATH = './repository'

# Define file extensions for supported languages
supported_extensions = [
    "*.py", "*.js",
    "*.ts", "*.java", 
    "*.c", "*.cpp", 
    "*.go", "*.rb", 
    "*.php", "*.rs"
]

def fetch_repository(
        # repo_path: str,
        repo_url: str, 
        branch: str ="main", 
        file_filter=None) -> list[Document]:
    """
    Load a GitHub repository.
    """
    # Load the GitHub repository
    loader = GitLoader(
        # repo_path=repo_path,
        clone_url=repo_url,
        repo_path=REPOSITORY_PATH,
        file_filter=file_filter,
        branch=branch
    )
    
    try:
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading repository: {e}")
        return []

def split_repository_documents(documents: list[Document]) -> list[Document]:
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

def parse_and_print_repository(repo_path):
    loader = GenericLoader.from_filesystem(
                repo_path, 
                glob="**/[!.]*",
                suffixes=supported_extensions,
                show_progress=True
            )
    # loader = DirectoryLoader(
    #     repo_path,
    #     glob=supported_extensions,
    #     show_progress=True,
    #     recursive=True,
    #     use_multithreading=True
    # )
    documents = loader.load()
    print(f"Length Documents: {documents}")

    # Process and display parsed information
    for doc in documents:
        print(f"File: {doc.metadata['source']}")
        print(f"Language: {doc.metadata.get('language', 'Unknown')}")
        
        # Extract and display variables, functions, and classes
        if 'variables' in doc.metadata:
            print("Variables:")
            for var in doc.metadata['variables']:
                print(f"  - {var}")
        
        if 'functions' in doc.metadata:
            print("Functions:")
            for func in doc.metadata['functions']:
                print(f"  - {func}")
        
        if 'classes' in doc.metadata:
            print("Classes:")
            for cls in doc.metadata['classes']:
                print(f"  - {cls}")
        
        print("\n")

if __name__ == "__main__":
    print("Starting repository parsing...")
    if not os.path.exists("./repository"):
        print("Error: Directory './repository' does not exist")
        sys.exit(1)
    fetch_repository(
        "./repository",
        "https://github.com/ajkulkarni/online-banking-application.git"
        )
    parse_and_print_repository("./repository")
    print("Repository parsing complete.")