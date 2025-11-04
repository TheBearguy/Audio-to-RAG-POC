#!/usr/bin/env python3
"""
Audio-to-RAG POC Main Entry Point

This script demonstrates the audio-to-RAG pipeline:
1. Transcribe audio with speaker identification
2. Generate embeddings and store in MongoDB
3. Perform vector search retrieval
4. Generate responses using retrieved context
"""

import os
import sys
from pymongo import MongoClient
from embed_and_store import create_and_store_embeddings
from retrieval import create_mongo_vector_index, perform_vector_search
from generate_response import generate_response

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables from .env file won't be loaded.")
    print("Install with: pip install python-dotenv")

def test_pipeline():
    """
    Test the audio-to-RAG pipeline with basic functionality checks.
    """
    print("Audio-to-RAG POC - Testing Pipeline")
    print("====================================\n")
    
    # Check environment variables
    print("1. Checking environment variables...")
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("Error: MONGO_URI environment variable not set")
        return False
    else:
        print("MONGO_URI is set")
    
    # Check VoyageAI API key
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_api_key:
        print("Warning: VOYAGE_API_KEY environment variable not set")
        print("   This may cause issues with embedding generation")
    else:
        print("VOYAGE_API_KEY is set")
    
    # Check AssemblyAI API key
    aai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not aai_api_key:
        print("Warning: ASSEMBLYAI_API_KEY environment variable not set")
        print("   This may cause issues with audio transcription")
    else:
        print("ASSEMBLYAI_API_KEY is set")
    
    print("\n2. Testing MongoDB connection...")
    try:
        # More detailed connection testing
        client = MongoClient(
            mongo_uri, 
            serverSelectionTimeoutMS=30000,  # 30 seconds
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
        
        # Test the connection
        print("   Attempting to connect to MongoDB Atlas...")
        server_info = client.server_info()
        print(f"MongoDB connection successful (version: {server_info.get('version', 'unknown')})")
        
        # Test collection access
        db = client["voyage"]
        collection = db["voyage_embeddings"]
        
        # Try to get collection stats (this will test authentication)
        try:
            stats = db.command("collStats", "voyage_embeddings")
            print(f"Database and collection accessible (documents: {stats.get('count', 0)})")
        except Exception:
            # Collection might not exist yet, which is fine
            print("Database accessible (collection will be created when needed)")
        
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        print(f"   Connection string (masked): {mongo_uri.split('@')[1] if '@' in mongo_uri else 'Invalid format'}")
        print("   Possible issues:")
        print("   - Network connectivity problem")
        print("   - Invalid credentials in connection string")
        print("   - Database cluster is paused or not accessible")
        print("   - IP address not whitelisted in MongoDB Atlas")
        return False
    
    print("\n3. Testing module imports...")
    try:
        import voyageai
        import assemblyai as aai
        from llama_index.llms.ollama import Ollama
        from llama_index.core.llms import ChatMessage, MessageRole
        print("✅ All required modules imported successfully")
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        return False
    
    print("\n4. Testing Ollama connection...")
    try:
        # Test with direct API call first to check service
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            print(f"✅ Ollama service is running with models: {model_names}")
            
            # Test if we can use the model through LlamaIndex
            try:
                llm = Ollama(model="gpt-oss", request_timeout=30.0)
                # Try a very simple test
                response = llm.complete("Hi")
                print("✅ Ollama model integration successful")
            except Exception as llm_error:
                print(f"⚠️ Warning: LlamaIndex integration has issues: {llm_error}")
                print("   Will try to use API directly in the pipeline")
        else:
            raise Exception("Ollama service not responding")
    except Exception as e:
        print(f"⚠️ Warning: Ollama connection failed: {e}")
        print("   Make sure Ollama is running and the 'gpt-oss' model is available")
        print("   The pipeline will work but response generation will fail without Ollama")
    
    print("\n✅ Pipeline basic checks completed successfully!")
    print("\nTo run the full pipeline, you would need:")
    print("- An audio file to process")
    print("- Proper API keys set in environment variables")
    print("- MongoDB running and accessible")
    print("- Ollama running with the required model")
    
    return True

def demo_usage():
    """
    Show example usage of the pipeline components.
    """
    print("\nExample usage:")
    print("=============\n")
    
    print("# 1. Create and store embeddings from audio")
    print("from embed_and_store import create_and_store_embeddings")
    print("num_docs = create_and_store_embeddings('path/to/audio.wav')")
    print()
    
    print("# 2. Set up vector search index (one-time setup)")
    print("from retrieval import create_mongo_vector_index")
    print("from pymongo import MongoClient")
    print("client = MongoClient(os.getenv('MONGO_URI'))")
    print("collection = client['voyage']['voyage_embeddings']")
    print("create_mongo_vector_index(collection)")
    print()
    
    print("# 3. Search and generate response")
    print("from retrieval import perform_vector_search")
    print("from generate_response import generate_response")
    print("results = perform_vector_search('your query', collection)")
    print("response = generate_response('your query', results)")
    print("for token in response:")
    print("    print(token.delta, end='')")

def main():
    """
    Main entry point for the application.
    """
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demo_usage()
        elif sys.argv[1] == "--audio":
            print("🎙️ Starting Audio RAG Interface...")
            try:
                from audio_rag_interface import main as audio_main
                audio_main()
            except ImportError as e:
                print(f"❌ Audio RAG not available: {e}")
                print("   Make sure audio dependencies are installed")
        elif sys.argv[1] == "--setup":
            print("🔧 Running setup check...")
            try:
                import sys
                sys.path.append('..')
                from setup_audio_rag import main as setup_main
                setup_main()
            except ImportError as e:
                print(f"❌ Setup script not available: {e}")
        else:
            print("Usage:")
            print("  python src/main.py           # Test pipeline")
            print("  python src/main.py --demo    # Show usage examples")
            print("  python src/main.py --audio   # Start Audio RAG interface")
            print("  python src/main.py --setup   # Run setup check")
    else:
        test_pipeline()

if __name__ == "__main__":
    main()
