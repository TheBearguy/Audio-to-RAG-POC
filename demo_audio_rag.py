#!/usr/bin/env python3
"""
Audio RAG Demo Script

This script demonstrates the complete Audio RAG pipeline:
1. Record audio questions
2. Convert speech to text
3. Perform RAG-based question answering
4. Convert responses back to speech
"""

import os
import sys
from src.audio_rag_interface import AudioRAG

def demo_workflow():
    """Demonstrate the complete Audio RAG workflow."""
    
    print("Audio RAG System Demo")
    print("=" * 25)
    print()
    
    # Check environment setup
    print("1. Checking environment setup...")
    
    required_vars = ["MONGO_URI", "VOYAGE_API_KEY", "ASSEMBLYAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease set these in your .env file:")
        for var in missing_vars:
            print(f"  {var}=your_{var.lower()}_here")
        return False
    
    print("Environment variables configured")
    
    # Initialize Audio RAG
    print("\n2. Initializing Audio RAG system...")
    try:
        audio_rag = AudioRAG()
        print("Audio RAG system initialized")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return False
    
    # Demo workflow
    print("\n3. Audio RAG Workflow Demo")
    print("-" * 30)
    
    try:
        while True:
            print("\nChoose an option:")
            print("  1. Record voice question")
            print("  2. Type text question")
            print("  3. Add audio file to knowledge base")
            print("  4. Start interactive session")
            print("  5. Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                print("\nVoice Question Demo")
                print("You'll be able to record a question and get a spoken response")
                audio_rag.record_and_ask(max_duration=30, speak_response=True)
            
            elif choice == "2":
                print("\nText Question Demo")
                question = input("Enter your question: ").strip()
                if question:
                    audio_rag.ask_question(question, speak_response=True)
            
            elif choice == "3":
                print("\nAdd Audio File Demo")
                audio_path = input("Enter path to audio file: ").strip()
                if audio_path and os.path.exists(audio_path):
                    audio_rag.add_audio_to_knowledge_base(audio_path)
                else:
                    print("File not found")
            
            elif choice == "4":
                print("\nStarting Interactive Session")
                audio_rag.interactive_session()
            
            elif choice == "5":
                print("Exiting demo")
                break
            
            else:
                print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        audio_rag.cleanup()
    
    return True

def quick_test():
    """Quick test of the system components."""
    
    print("Quick System Test")
    print("=" * 20)
    
    # Test imports
    print("\n1. Testing imports...")
    try:
        from src.audio_recorder import AudioRecorder
        from src.text_to_speech import TextToSpeech
        from src.audio_rag_interface import AudioRAG
        print("All modules imported successfully")
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    # Test TTS
    print("\n2. Testing Text-to-Speech...")
    try:
        tts = TextToSpeech()
        print("TTS engine initialized")
        
        # Quick speech test
        test_text = "Audio RAG system test successful"
        print(f"Speaking: {test_text}")
        tts.speak(test_text)
        tts.cleanup()
        
    except Exception as e:
        print(f"TTS error: {e}")
    
    # Test audio recorder (without actually recording)
    print("\n3. Testing Audio Recorder...")
    try:
        recorder = AudioRecorder()
        print("Audio recorder initialized")
        recorder.cleanup()
    except Exception as e:
        print(f"Audio recorder error: {e}")
    
    print("\nQuick test completed")
    return True

def main():
    """Main entry point."""
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv not installed, .env file won't be loaded")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        quick_test()
    else:
        demo_workflow()

if __name__ == "__main__":
    main()