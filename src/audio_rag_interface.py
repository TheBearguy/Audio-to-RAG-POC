#!/usr/bin/env python3
"""
Complete Audio RAG Interface

This module provides a complete audio-based RAG system that:
1. Records audio questions from the user
2. Transcribes speech to text
3. Performs RAG-based question answering
4. Converts responses back to speech
"""

import os
import tempfile
from typing import Optional, Generator
from pymongo import MongoClient

# Import our existing components
from audio_recorder import AudioRecorder, record_question
from transcribe import transcribe_with_speakers
from embed_and_store import create_and_store_embeddings
from retrieval import create_mongo_vector_index, perform_vector_search
from generate_response_api import generate_response_api, test_ollama_api
from text_to_speech import TextToSpeech, StreamingTTS, speak_response

class AudioRAG:
    """
    Complete Audio RAG system for voice-based question answering.
    """
    
    def __init__(self, 
                 mongo_uri: str = None,
                 db_name: str = "voyage",
                 collection_name: str = "voyage_embeddings",
                 ollama_url: str = "http://localhost:11434",
                 model_name: str = "gpt-oss"):
        """
        Initialize the Audio RAG system.
        
        Args:
            mongo_uri (str): MongoDB connection URI
            db_name (str): MongoDB database name
            collection_name (str): MongoDB collection name
            ollama_url (str): Ollama API URL
            model_name (str): LLM model name
        """
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI")
        self.db_name = db_name
        self.collection_name = collection_name
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        # Initialize components
        self.tts = TextToSpeech()
        self.client = None
        self.collection = None
        
        # Connect to database
        self._connect_to_database()
    
    def _connect_to_database(self) -> bool:
        """
        Connect to MongoDB database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.mongo_uri:
                print("MongoDB URI not provided")
                return False
            
            self.client = MongoClient(self.mongo_uri)
            self.collection = self.client[self.db_name][self.collection_name]
            
            # Test connection
            self.client.server_info()
            print("Connected to MongoDB")
            return True
            
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            return False
    
    def setup_vector_index(self) -> bool:
        """
        Set up the vector search index (one-time setup).
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.collection:
            print("Database not connected")
            return False
        
        try:
            create_mongo_vector_index(self.collection)
            return True
        except Exception as e:
            print(f"Failed to create vector index: {e}")
            return False
    
    def add_audio_to_knowledge_base(self, audio_path: str) -> bool:
        """
        Add an audio file to the knowledge base.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Processing audio file: {audio_path}")
            
            # Create embeddings and store in database
            num_docs = create_and_store_embeddings(audio_path, self.db_name, self.collection_name, self.mongo_uri)
            
            print(f"Added {num_docs} documents to knowledge base")
            return True
            
        except Exception as e:
            print(f"Failed to add audio to knowledge base: {e}")
            return False
    
    def record_and_ask(self, max_duration: int = 30, speak_response: bool = True) -> Optional[str]:
        """
        Record a question and get an answer.
        
        Args:
            max_duration (int): Maximum recording duration in seconds
            speak_response (bool): Whether to speak the response aloud
            
        Returns:
            Optional[str]: The text response if successful, None otherwise
        """
        temp_audio = None
        
        try:
            # 1. Record the question
            print("\nAudio RAG - Voice Question")
            print("=" * 30)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_audio = f.name
            
            audio_file = record_question(temp_audio, max_duration)
            if not audio_file:
                print("Failed to record question")
                return None
            
            # 2. Transcribe the question
            print("Transcribing your question...")
            transcription = transcribe_with_speakers(audio_file)
            
            if not transcription:
                print("Failed to transcribe audio")
                return None
            
            # Extract the question text (combine all speakers)
            question = " ".join([utterance["text"] for utterance in transcription])
            print(f"Question: {question}")
            
            # 3. Get the answer
            response_text = self.ask_question(question, speak_response)
            return response_text
            
        except Exception as e:
            print(f"Error in record_and_ask: {e}")
            return None
        finally:
            # Clean up temporary file
            if temp_audio and os.path.exists(temp_audio):
                try:
                    os.unlink(temp_audio)
                except:
                    pass
    
    def ask_question(self, question: str, speak_response: bool = True) -> Optional[str]:
        """
        Ask a text question and get an answer.
        
        Args:
            question (str): The question to ask
            speak_response (bool): Whether to speak the response aloud
            
        Returns:
            Optional[str]: The text response if successful, None otherwise
        """
        try:
            if not self.collection:
                print("Database not connected")
                return None
            
            print(f"\nSearching for relevant information...")
            
            # 1. Perform vector search
            search_results = perform_vector_search(question, self.collection, top_k=5)
            
            if not search_results:
                response = "I couldn't find any relevant information to answer your question."
                print(f"Response: {response}")
                if speak_response:
                    self.tts.speak(response)
                return response
            
            print(f"Found {len(search_results)} relevant documents")
            
            # 2. Generate response using LLM
            print("Generating response...")
            
            if speak_response:
                # Use streaming TTS for real-time speech
                streaming_tts = StreamingTTS(chunk_size=40)
                streaming_tts.start_streaming()
                
                response_text = ""
                print("Response: ", end="", flush=True)
                
                try:
                    for token in generate_response_api(question, search_results, self.ollama_url, self.model_name):
                        print(token, end="", flush=True)
                        response_text += token
                        streaming_tts.add_text(token)
                    
                    print()  # New line after response
                    
                    # Wait a moment for speech to finish
                    import time
                    time.sleep(1)
                    streaming_tts.stop_streaming()
                    streaming_tts.cleanup()
                    
                except Exception as e:
                    streaming_tts.stop_streaming()
                    streaming_tts.cleanup()
                    raise e
            else:
                # Just collect the text response
                response_text = ""
                print("Response: ", end="", flush=True)
                
                for token in generate_response_api(question, search_results, self.ollama_url, self.model_name):
                    print(token, end="", flush=True)
                    response_text += token
                
                print()  # New line after response
            
            return response_text.strip()
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return None
    
    def interactive_session(self):
        """
        Start an interactive audio RAG session.
        """
        print("\nAudio RAG Interactive Session")
        print("=" * 35)
        print("Commands:")
        print("  'record' or 'r' - Record a voice question")
        print("  'text' or 't' - Type a text question")
        print("  'add' or 'a' - Add audio file to knowledge base")
        print("  'voices' or 'v' - List available voices")
        print("  'help' or 'h' - Show this help")
        print("  'quit' or 'q' - Exit")
        print()
        
        while True:
            try:
                command = input("Enter command (or question): ").strip().lower()
                
                if command in ['quit', 'q', 'exit']:
                    print("Goodbye!")
                    break
                
                elif command in ['record', 'r']:
                    self.record_and_ask()
                
                elif command in ['text', 't']:
                    question = input("Enter your question: ").strip()
                    if question:
                        self.ask_question(question)
                
                elif command in ['add', 'a']:
                    audio_path = input("Enter path to audio file: ").strip()
                    if audio_path and os.path.exists(audio_path):
                        self.add_audio_to_knowledge_base(audio_path)
                    else:
                        print("File not found")
                
                elif command in ['voices', 'v']:
                    voices = self.tts.get_available_voices()
                    print("\nAvailable voices:")
                    for voice in voices:
                        print(f"  {voice['index']}: {voice['name']}")
                    
                    try:
                        voice_idx = int(input("\nEnter voice index (or press Enter to skip): ").strip())
                        self.tts.set_voice(voice_idx)
                    except (ValueError, EOFError):
                        pass
                
                elif command in ['help', 'h']:
                    print("\nAudio RAG Interactive Session")
                    print("Commands:")
                    print("  'record' or 'r' - Record a voice question")
                    print("  'text' or 't' - Type a text question")
                    print("  'add' or 'a' - Add audio file to knowledge base")
                    print("  'voices' or 'v' - List available voices")
                    print("  'help' or 'h' - Show this help")
                    print("  'quit' or 'q' - Exit")
                
                elif command:
                    # Treat as a direct question
                    self.ask_question(command)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.tts:
            self.tts.cleanup()
        if self.client:
            self.client.close()

def main():
    """Main entry point for the Audio RAG system."""
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Check prerequisites
    print("Checking Audio RAG Prerequisites...")
    
    # Check MongoDB connection
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("MONGO_URI environment variable not set")
        return
    
    # Check Ollama
    if not test_ollama_api():
        print("Ollama API not available")
        return
    
    # Initialize Audio RAG system
    try:
        audio_rag = AudioRAG()
        
        print("Audio RAG system initialized successfully!")
        print("\nStarting interactive session...")
        
        audio_rag.interactive_session()
        
    except Exception as e:
        print(f"Failed to initialize Audio RAG: {e}")
    finally:
        try:
            audio_rag.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()