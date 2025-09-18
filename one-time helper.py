# one-time helper to delete existing ChromaDB collection
from chromadb import PersistentClient
c = PersistentClient(path="./chroma_store")
c.delete_collection("guideline_chunks")