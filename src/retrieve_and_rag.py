from chromadb.config import Settings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb

MODEL = "Qwen/Qwen2.5-3B-Instruct"
EMBED_MODEL = "intfloat/e5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)
embedder = SentenceTransformer(EMBED_MODEL)

chroma_client = chromadb.PersistentClient(path="./chroma", settings=Settings())

collection = chroma_client.get_collection("voice_memory")


def retrieve(query, k: int = 5):
    q_emb = embedder.encode(query)
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    return res


def generate(prompt):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    response = model.generate(
        **tokens, max_new_tokens=256, do_sample=False, temperature=0.0
    )

    return tokenizer.decode(response[0], skip_special_tokens=True)


def build_prompt(query, docs, metas):
    context = ""
    for m, d in zip(metas, docs):
        context += f"[{m['index']}] {d} \n\n"
    return (
        "You are a retrieval-grounded assistant. "
        "Use only the notes provided. Answer concisely. "
        "If information is missing, respond: I don't know.\n\n"
        f"Notes:\n{context}\n"
        f"User: {query}\n"
        "Assistant:"
    )


def ask(query):
    res = retrieve(query)
    docs = res["documents"]
    metas = res["metadatas"]
    prompt = build_prompt(query, docs, metas)
    response = generate(prompt)
    print(f"Response received :: {response}")
    return response


if __name__ == "__main__":
    ask("What is the smell of?")
