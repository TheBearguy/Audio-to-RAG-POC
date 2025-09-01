import voyageai
from pymongo import MongoClient
from transcribe import transcribe_with_speakers
import os

def create_and_store_embeddings(
    audio_path: str, 
    db_name: str  = "voyage",
    collection_name: str = "voyage_embeddings",
    mongo_uri: str = os.getenv("MONGO_URI")
): 
    """
    Create and store embeddings for a given audio path.
    Generates embeddings for each utterance,
    and stores the result in a MongoDB collection.

    Args:
        audio_path (str): The path to the audio file.
        db_name (str): The name of the MongoDB database.
        collection_name (str): The name of the MongoDB collection.
        mongo_uri (str): The URI of the MongoDB server.

    Returns:
        int: The number of documents successfully inserted into the database.
    """
    # 1. transcribe the audio to get the speaker  separated utterances (text)
    speaker_utterances = transcribe_with_speakers(audio_path)
    
    if not speaker_utterances:
        raise ValueError("No utterances found in the audio file")

    texts_to_embed = [utterance["text"] for utterance in speaker_utterances]
    print(f"Found {len(texts_to_embed)} utterances to embed")

    # generate voyageai client
    vo = voyageai.Client()
    # 2. Generate embeddings for each utterance using Voyage AI
    embeddings = vo.embed(
        texts=texts_to_embed,
        model="voyage-context-3",
        input_type="document",
    ).embeddings

    # 3. Store the embeddings in a MongoDB collection
    # combine the text and embedding into a single document
    docs = []
    for (utterance, embedding) in zip(speaker_utterances, embeddings):
        docs.append(
            {
                "text": utterance["text"],
                "speaker": utterance["speaker"],
                "embedding": embedding
            }
        )
    # 4. Connect to MongoDB and store the documents in the collection
    client = MongoClient(mongo_uri)
    collection = client[db_name][collection_name]
    collection.insert_many(docs)
    print(f"Inserted {len(docs)} documents into the collection")

    return len(docs)