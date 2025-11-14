import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

EMBED_MODEL = "intfloat/e5-small"
TOKENIZER = "sentence-transformer/all-mpnet-base-v2"

def load_transcript(path: str):
    return json.loads(path)

# Tokenize first, then chunk
def chunk_transcript(transcript_text, max_tokens=512, overlap=64): 
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER) 
    tokens = tokenizer.encode(transcript_text)
    #now making chunks
    chunks = []
    i = 0
    chunk_tokens = ""
    for token in tokens: 
        if i <= 512: 
            chunk_tokens += token
        else: 
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            chunk_tokens = ""
        i += max_tokens - overlap
    return chunks

def embed(chunks): 
    embed_model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    return embeddings


if __name__ == "__main__": 
    transcript_path = "data/transcript.json"
    text = load_transcript(transcript_path)["text"]
    chunks = chunk_transcript(text)
    embeddings = embed(chunks)
    np.save("data/chunks.npy", chunks)
    np.save("data/embs.npy", embeddings)
    print(f"Saved chunks and embeddings")