import pyttsx3
import os
import tempfile
from typing import Optional, Union
import threading
import queue

class TextToSpeech:
    """
    Text-to-speech engine for converting responses to audio.
    """
    
    def __init__(self, 
                 rate: int = 200,
                 volume: float = 0.9,
                 voice_index: int = 0):
        """
        Initialize the TTS engine.
        
        Args:
            rate (int): Speech rate (words per minute)
            volume (float): Volume level (0.0 to 1.0)
            voice_index (int): Voice index (0 for first available voice)
        """
        self.engine = pyttsx3.init()
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        
        self._configure_engine()
    
    def _configure_engine(self):
        """Configure the TTS engine with specified settings."""
        try:
            # Set rate
            self.engine.setProperty('rate', self.rate)
            
            # Set volume
            self.engine.setProperty('volume', self.volume)
            
            # Set voice
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > self.voice_index:
                self.engine.setProperty('voice', voices[self.voice_index].id)
                print(f"🔊 Using voice: {voices[self.voice_index].name}")
            else:
                print("Using default voice")
                
        except Exception as e:
            print(f"Warning: Could not configure TTS engine: {e}")
    
    def speak(self, text: str) -> bool:
        """
        Speak the given text.
        
        Args:
            text (str): Text to speak
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not text.strip():
                return False
            
            print(f"🔊 Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
            self.engine.say(text)
            self.engine.runAndWait()
            return True
            
        except Exception as e:
            print(f"Error speaking text: {e}")
            return False
    
    def speak_async(self, text: str) -> threading.Thread:
        """
        Speak text asynchronously in a separate thread.
        
        Args:
            text (str): Text to speak
            
        Returns:
            threading.Thread: The thread handling the speech
        """
        def _speak():
            self.speak(text)
        
        thread = threading.Thread(target=_speak)
        thread.start()
        return thread
    
    def save_to_file(self, text: str, filename: str) -> bool:
        """
        Save speech to an audio file.
        
        Args:
            text (str): Text to convert to speech
            filename (str): Path to save the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
            
            print(f"Speech saved to: {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving speech to file: {e}")
            return False
    
    def get_available_voices(self) -> list:
        """
        Get list of available voices.
        
        Returns:
            list: List of available voice information
        """
        try:
            voices = self.engine.getProperty('voices')
            voice_info = []
            
            for i, voice in enumerate(voices):
                voice_info.append({
                    'index': i,
                    'id': voice.id,
                    'name': voice.name,
                    'languages': getattr(voice, 'languages', []),
                    'gender': getattr(voice, 'gender', 'Unknown')
                })
            
            return voice_info
            
        except Exception as e:
            print(f"Error getting voices: {e}")
            return []
    
    def set_voice(self, voice_index: int) -> bool:
        """
        Change the voice used for speech.
        
        Args:
            voice_index (int): Index of the voice to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            voices = self.engine.getProperty('voices')
            if voices and 0 <= voice_index < len(voices):
                self.engine.setProperty('voice', voices[voice_index].id)
                self.voice_index = voice_index
                print(f"🔊 Voice changed to: {voices[voice_index].name}")
                return True
            else:
                print(f"Invalid voice index: {voice_index}")
                return False
                
        except Exception as e:
            print(f"Error setting voice: {e}")
            return False
    
    def set_rate(self, rate: int) -> bool:
        """
        Change the speech rate.
        
        Args:
            rate (int): New speech rate (words per minute)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.engine.setProperty('rate', rate)
            self.rate = rate
            print(f"Speech rate set to: {rate} WPM")
            return True
            
        except Exception as e:
            print(f"Error setting rate: {e}")
            return False
    
    def cleanup(self):
        """Clean up TTS engine resources."""
        try:
            self.engine.stop()
        except:
            pass

class StreamingTTS:
    """
    Streaming text-to-speech for real-time response playback.
    """
    
    def __init__(self, chunk_size: int = 50):
        """
        Initialize streaming TTS.
        
        Args:
            chunk_size (int): Number of characters to accumulate before speaking
        """
        self.tts = TextToSpeech()
        self.chunk_size = chunk_size
        self.text_queue = queue.Queue()
        self.is_speaking = False
        self.speaker_thread: Optional[threading.Thread] = None
    
    def start_streaming(self):
        """Start the streaming TTS worker thread."""
        self.is_speaking = True
        self.speaker_thread = threading.Thread(target=self._speaking_worker)
        self.speaker_thread.start()
    
    def add_text(self, text: str):
        """
        Add text to the speaking queue.
        
        Args:
            text (str): Text to add to the queue
        """
        if text.strip():
            self.text_queue.put(text)
    
    def _speaking_worker(self):
        """Worker thread that processes the text queue and speaks."""
        accumulated_text = ""
        
        while self.is_speaking:
            try:
                # Get text from queue with timeout
                text = self.text_queue.get(timeout=0.5)
                accumulated_text += text
                
                # Speak when we have enough text or hit sentence boundaries
                if (len(accumulated_text) >= self.chunk_size or 
                    any(punct in text for punct in '.!?')):
                    
                    if accumulated_text.strip():
                        self.tts.speak(accumulated_text.strip())
                        accumulated_text = ""
                
            except queue.Empty:
                # Speak any remaining text if we're done
                if accumulated_text.strip():
                    self.tts.speak(accumulated_text.strip())
                    accumulated_text = ""
                continue
            except Exception as e:
                print(f"Error in speaking worker: {e}")
    
    def stop_streaming(self):
        """Stop the streaming TTS."""
        self.is_speaking = False
        if self.speaker_thread:
            self.speaker_thread.join(timeout=2.0)
    
    def cleanup(self):
        """Clean up streaming TTS resources."""
        self.stop_streaming()
        self.tts.cleanup()

def speak_response(text: str, async_mode: bool = False) -> Union[bool, threading.Thread]:
    """
    Convenience function to speak a response.
    
    Args:
        text (str): Text to speak
        async_mode (bool): Whether to speak asynchronously
        
    Returns:
        Union[bool, threading.Thread]: Success status or thread if async
    """
    tts = TextToSpeech()
    
    if async_mode:
        return tts.speak_async(text)
    else:
        return tts.speak(text)

if __name__ == "__main__":
    # Test the TTS system
    print("Text-to-Speech Test")
    print("==================")
    
    tts = TextToSpeech()
    
    # Show available voices
    print("\nAvailable voices:")
    voices = tts.get_available_voices()
    for voice in voices[:3]:  # Show first 3 voices
        print(f"  {voice['index']}: {voice['name']}")
    
    # Test speaking
    test_text = "Hello! This is a test of the text-to-speech system. The audio RAG pipeline is working correctly."
    
    print(f"\nTesting speech with text: {test_text}")
    success = tts.speak(test_text)
    
    if success:
        print("TTS test successful")
    else:
        print("TTS test failed")
    
    # Test streaming TTS
    print("\nTesting streaming TTS...")
    streaming_tts = StreamingTTS(chunk_size=30)
    streaming_tts.start_streaming()
    
    # Simulate streaming response
    response_parts = [
        "This is a streaming ",
        "text-to-speech test. ",
        "Each part is spoken ",
        "as it becomes available. ",
        "This simulates real-time response generation."
    ]
    
    for part in response_parts:
        streaming_tts.add_text(part)
        import time
        time.sleep(0.5)  # Simulate delay between parts
    
    streaming_tts.stop_streaming()
    streaming_tts.cleanup()
    
    print("Streaming TTS test completed")
    
    tts.cleanup()