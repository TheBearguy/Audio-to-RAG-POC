import chromadb
from chromadb.config import Settings
import numpy as np

chroma_client = chromadb.PersistentClient(path="./chroma", settings=Settings())
collection = chroma_client.create_collection(name="Voice Memory")

chunks = list(np.load("data/chunks.npy", allow_pickle=True))
embs = np.load("data/embs.npy")
ids = [f"rec_{i}" for i in range(len(chunks))]
metas = [{"source": "sample.wav", "index": i} for i in range(len(chunks))]

collection.add(ids=ids, documents=chunks, embeddings=embs.tolist())

print("Stored to chromadb")
