from chromadb.config import Settings
import openai
import os
from sentence_transformers import SentenceTransformers
import chromadb

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "sentence-transformers/intfloat/e5-small"

client = chromadb.PersistentClient(path="./chroma", settings=Settings())

collection = client.get_collection("voice_memory")
embedder = SentenceTransformers(EMBED_MODEL)


def retrieve(query, k=5):
    q_emb = embedder.encode([query])[0].tolist()
    res = collection.query(query_images=[q_emb], n_results=k)
    return res


def ask_rag(query):
    res = retrieve(query)
    docs = res["documents"]
    metas = res["metadatas"]
    context = "\n\n".join([f"[{m['index']}] {d}" for m, d in zip(metas, docs)])
    prompt = f"Use the notes below to answer the user. If answer unknown, say 'I don't know'.\n\nNotes:\n{
        context
    }\n\nUser: {query}\nAssistant:"
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )
    return resp["choices"][0]["message"]["content"]


if __name__ == "__main__":
    q = "What smell is there?"
    print(ask_rag(q))
