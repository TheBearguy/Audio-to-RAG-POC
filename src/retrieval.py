from re import search
from typing import List
from pymongo.operations import SearchIndexModel
from pymongo import MongoClient
from pymongo.collection import Collection
import os
import voyageai

def create_mongo_vector_index(
    collection: Collection, 
    index_name: str = "vector_search_index", 
    num_dimensions: int = 1536 
): 
    """
    Creates a vector search index in a mongodb collection. 
    This is a one-time setup operation. 

    Args: 
        collection (Collection): The Pymongo Collection object. 
        index_name (str): The name of the search index. 
        num_dimensions (int): The number of dimensions for the embeddings. (eg: voyage-context-3 has 1024) 
    """
    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector", 
                    "path": "embedding", 
                    "numDimensions": num_dimensions, 
                    "similarity": "cosine" 
                }
            ]
        },
        name=index_name, 
        type="vectorSearch"
    )
    
    try:
        print(f"Creating Index '{index_name}'... (This may take a few minutes)")
        collection.create_search_index(model=search_index_model)
        print("Index created successfully")
    except Exception as e: 
        # This can happen if the index already exists. 
        print(f"An error occured in creating the index.. Or The index may already exist: {e}")


def perform_vector_search(
    query: str, 
    collection: Collection, 
    index_name: str =   "vector_search_index", 
    top_k: int = 5
) -> List: 
    """
    Performs a vector search on mongodb collection to find relevant documents
    Args: 
        query (str): The user's query string
        collection (Collection): The Pymongo Collection object. 
        index_name (str): The name of the vector search index to use. 
        top_k (int): The number of top results to retrieve. 
    
    Returns: 
        list: A list of top_k relevant document texts. 
    """
    vo = voyageai.Client()

    # 1. Generate the embedding for the user's query: 
    query_embedding = vo.embed(
        texts=[query], 
        model="voyage-context-3", 
        input_type="query" # User "query" for better seaarch performance. 
    ).embeddings[0]

    # 2. Perform the vector search using an aggregation pipeline: 
    results = collection.aggregate(
        [
            {
                "$vectorSearch": {
                    "index": index_name, 
                    "queryVector": query_embedding, 
                    "path": "embedding", 
                    "numCandidates": 100, # Nummber of candidates to consider,
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "_id": 0, 
                    "speaker": 1, 
                    "text": 1, 
                    "score": {
                        "$meta": "vectorSearchScore"
                    }
                }
            }
        ]
    )

    return list(results)