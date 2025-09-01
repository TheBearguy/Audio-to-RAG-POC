import requests
import json
from typing import Dict, List, Iterator

def generate_response_api(
    query: str, 
    context: List[Dict],
    ollama_url: str = "http://localhost:11434",
    model_name: str = "gpt-oss"
) -> Iterator[str]:
    """
    Generates a response using Ollama API directly to bypass template issues.
    
    Args:
        query (str): The user's original query.
        context (List[Dict]): A list of context documents retrieved from search.
        ollama_url (str): The Ollama API URL.
        model_name (str): The name of the model to use.
    
    Yields:
        str: Response tokens from the LLM.
    """
    
    # 1. Merge the context into a single string
    if not context:
        merged_context = "No context information was provided"
    else:
        merged_context = "\n\n------\n\n".join([doc['text'] for doc in context])
    
    # 2. Create a simple prompt without complex templating
    prompt = f"""Context information:
{merged_context}

Question: {query}

Please provide a helpful answer based on the context provided above."""
    
    # 3. Make API call to Ollama
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            },
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            yield f"Error: API request failed with status {response.status_code}"
            return
        
        # 4. Stream the response
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        yield data['response']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
                    
    except requests.exceptions.RequestException as e:
        yield f"Error connecting to Ollama: {e}"
    except Exception as e:
        yield f"Unexpected error: {e}"

def test_ollama_api(
    ollama_url: str = "http://localhost:11434",
    model_name: str = "gpt-oss"
) -> bool:
    """
    Test if Ollama API is working with the specified model.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Test with a simple prompt
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": "Hello, respond with just 'Hi there!'",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'response' in data:
                print(f"✅ Ollama API test successful: {data['response'].strip()}")
                return True
        
        print(f"❌ Ollama API test failed: {response.status_code} - {response.text}")
        return False
        
    except Exception as e:
        print(f"❌ Ollama API test error: {e}")
        return False

if __name__ == "__main__":
    # Test the API function
    print("Testing Ollama API...")
    success = test_ollama_api()
    
    if success:
        print("\nTesting response generation...")
        test_context = [{"text": "The sky is blue because of light scattering."}]
        test_query = "Why is the sky blue?"
        
        print(f"Query: {test_query}")
        print("Response: ", end="")
        
        for token in generate_response_api(test_query, test_context):
            print(token, end="", flush=True)
        print("\n")
