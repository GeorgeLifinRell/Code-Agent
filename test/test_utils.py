import sys
import os

# Add the directory containing utils.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils_old import embed_and_store

def test_embed_and_store():
    chunks = ["This is a test chunk.", "Another test chunk."]
    vector_store = embed_and_store(chunks)
    
    assert vector_store is not None, "Vector store should not be None"
    print("Test passed: Vector store created successfully.")

if __name__ == "__main__":
    test_embed_and_store()