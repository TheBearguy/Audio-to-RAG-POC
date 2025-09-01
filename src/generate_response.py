from typing import Dict, Generator, List
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole

def generate_response(
    query: str, 
    context: List[Dict],
    llm_model_name: str = "gpt-oss"
) -> Generator:
    """
    Generates a response from an LLM based on a query and retrieved context.

    Args:
        query (str): The user's original query.
        context (List[Dict]): A list of context documents retrieved from search.
                               Each dictionary is expected to have a 'text' key.
        llm_model_name (str): The name of the Ollama model to use.

    Returns:
        Generator: A generator that yields the streaming response tokens from the LLM.
    """
    
    # 1. Merge the context into a single string: 
    if not context: 
        merged_context = "No context information was provided"
    else: 
        merged_context = "\n\n------\n\n".join([doc['text'] for doc in context])
        
    # 2. Construct the prompt with the context and query
    prompt = (
        f"You are a helpful AI assistant. Use the context information below to answer the user's query. "
        f"Provide a concise and direct answer based only on the given context.\n\n"
        f"--- CONTEXT ---\n"
        f"{merged_context}\n"
        f"--- END CONTEXT ---\n\n"
        f"Query: {query}\n"
        f"Answer: "
    )

    # 3. Set up the LLM
    llm = Ollama(model=llm_model_name, request_timeout=120.0)

    # 4. Generate and stream the response: 
    user_msg = ChatMessage(role=MessageRole.USER, content=prompt)

    response = llm.stream_complete(user_msg.content)
    return response
