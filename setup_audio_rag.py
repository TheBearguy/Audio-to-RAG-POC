#!/usr/bin/env python3
"""
Audio RAG Setup Script

This script helps set up the Audio RAG system by:
1. Checking system requirements
2. Testing audio components
3. Verifying API connections
4. Setting up the vector index
"""

import os
import sys
import subprocess
from typing import List, Tuple

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_dependencies() -> Tuple[List[str], List[str]]:
    """Check which dependencies are installed."""
    required_packages = [
        'voyageai', 'assemblyai', 'pymongo', 'pyaudio', 
        'pyttsx3', 'requests', 'python-dotenv'
    ]
    
    installed = []
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    return installed, missing

def test_audio_recording() -> bool:
    """Test if audio recording works."""
    try:
        import pyaudio
        
        # Try to initialize PyAudio
        audio = pyaudio.PyAudio()
        
        # Check if we have input devices
        input_devices = []
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append(device_info['name'])
        
        audio.terminate()
        
        if input_devices:
            print(f"Audio recording available ({len(input_devices)} input devices)")
            return True
        else:
            print("No audio input devices found")
            return False
            
    except Exception as e:
        print(f"Audio recording test failed: {e}")
        return False

def test_text_to_speech() -> bool:
    """Test if text-to-speech works."""
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        if voices:
            print(f"Text-to-speech available ({len(voices)} voices)")
            engine.stop()
            return True
        else:
            print("No TTS voices available")
            engine.stop()
            return False
            
    except Exception as e:
        print(f"Text-to-speech test failed: {e}")
        return False

def check_environment_variables() -> Tuple[List[str], List[str]]:
    """Check which environment variables are set."""
    required_vars = [
        'MONGO_URI',
        'VOYAGE_API_KEY', 
        'ASSEMBLYAI_API_KEY'
    ]
    
    set_vars = []
    missing_vars = []
    
    for var in required_vars:
        if os.getenv(var):
            set_vars.append(var)
        else:
            missing_vars.append(var)
    
    return set_vars, missing_vars

def test_mongodb_connection() -> bool:
    """Test MongoDB connection."""
    try:
        from pymongo import MongoClient
        
        mongo_uri = os.getenv('MONGO_URI')
        if not mongo_uri:
            print("MONGO_URI not set")
            return False
        
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        client.close()
        
        print("MongoDB connection successful")
        return True
        
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False

def test_ollama_connection() -> bool:
    """Test Ollama connection."""
    try:
        import requests
        
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if 'gpt-oss' in model_names:
                print("Ollama with gpt-oss model available")
                return True
            else:
                print(f"Ollama available but gpt-oss model not found. Available: {model_names}")
                return False
        else:
            print("Ollama not responding")
            return False
            
    except Exception as e:
        print(f"Ollama connection failed: {e}")
        return False

def setup_vector_index() -> bool:
    """Set up the MongoDB vector index."""
    try:
        from pymongo import MongoClient
        from src.retrieval import create_mongo_vector_index
        
        mongo_uri = os.getenv('MONGO_URI')
        if not mongo_uri:
            print("MONGO_URI not set")
            return False
        
        client = MongoClient(mongo_uri)
        collection = client['voyage']['voyage_embeddings']
        
        print("Setting up vector search index...")
        create_mongo_vector_index(collection)
        
        client.close()
        print("Vector index setup completed")
        return True
        
    except Exception as e:
        print(f"Vector index setup failed: {e}")
        return False

def create_env_template():
    """Create a .env template file."""
    template = """# Audio RAG Environment Variables
# Copy this file to .env and fill in your actual values

# MongoDB connection string
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/

# VoyageAI API key for embeddings
VOYAGE_API_KEY=your_voyage_api_key_here

# AssemblyAI API key for speech transcription
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
"""
    
    try:
        if not os.path.exists('.env'):
            with open('.env.template', 'w') as f:
                f.write(template)
            print("Created .env.template file")
            print("   Copy it to .env and fill in your API keys")
        else:
            print(".env file already exists")
        return True
    except Exception as e:
        print(f"Failed to create .env template: {e}")
        return False

def main():
    """Main setup function."""
    print("Audio RAG Setup")
    print("=" * 20)
    
    # Load environment variables if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    all_good = True
    
    # 1. Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        all_good = False
    
    # 2. Check dependencies
    print("\n2. Checking Python dependencies...")
    installed, missing = check_dependencies()
    
    if installed:
        print(f"Installed: {', '.join(installed)}")
    
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        all_good = False
    
    # 3. Test audio components
    print("\n3. Testing audio components...")
    
    print("   Audio Recording:")
    if not test_audio_recording():
        all_good = False
    
    print("   Text-to-Speech:")
    if not test_text_to_speech():
        all_good = False
    
    # 4. Check environment variables
    print("\n4. Checking environment variables...")
    set_vars, missing_vars = check_environment_variables()
    
    if set_vars:
        print(f"Set: {', '.join(set_vars)}")
    
    if missing_vars:
        print(f"Missing: {', '.join(missing_vars)}")
        create_env_template()
        all_good = False
    
    # 5. Test external services
    print("\n5. Testing external services...")
    
    print("   MongoDB:")
    if not test_mongodb_connection():
        all_good = False
    
    print("   Ollama:")
    if not test_ollama_connection():
        all_good = False
    
    # 6. Setup vector index (if everything else works)
    if all_good:
        print("\n6. Setting up vector index...")
        setup_vector_index()
    
    # Summary
    print("\n" + "=" * 40)
    if all_good:
        print("Setup completed successfully!")
        print("\nYou can now run:")
        print("  python demo_audio_rag.py")
        print("  python src/audio_rag_interface.py")
    else:
        print("Setup incomplete. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Set up .env file with your API keys")
        print("- Start Ollama: ollama serve")
        print("- Pull the model: ollama pull gpt-oss")

if __name__ == "__main__":
    main()