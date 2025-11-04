import pyaudio
import wave
import threading
import time
from typing import Optional
import os

class AudioRecorder:
    """
    Real-time audio recorder for capturing voice input.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 audio_format: int = pyaudio.paInt16):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate (int): Sample rate for recording (16kHz recommended for speech)
            channels (int): Number of audio channels (1 for mono)
            chunk_size (int): Size of audio chunks to read at once
            audio_format (int): Audio format (16-bit PCM recommended)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio_format = audio_format
        
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.frames = []
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
    
    def start_recording(self) -> bool:
        """
        Start recording audio.
        
        Returns:
            bool: True if recording started successfully, False otherwise
        """
        try:
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.frames = []
            self.is_recording = True
            
            # Start recording in a separate thread
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.start()
            
            print("Recording started... Press Enter to stop or call stop_recording()")
            return True
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False
    
    def _record_audio(self):
        """
        Internal method to record audio in a separate thread.
        """
        while self.is_recording and self.stream:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                print(f"Error during recording: {e}")
                break
    
    def stop_recording(self) -> bool:
        """
        Stop recording audio.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        try:
            self.is_recording = False
            
            if self.recording_thread:
                self.recording_thread.join(timeout=2.0)
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            print("Recording stopped")
            return True
            
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return False
    
    def save_recording(self, filename: str) -> bool:
        """
        Save the recorded audio to a WAV file.
        
        Args:
            filename (str): Path to save the audio file
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            if not self.frames:
                print("No audio data to save")
                return False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
            
            print(f"💾 Audio saved to: {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False
    
    def record_for_duration(self, duration_seconds: float, filename: str) -> bool:
        """
        Record audio for a specific duration and save it.
        
        Args:
            duration_seconds (float): How long to record in seconds
            filename (str): Path to save the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"🎙️ Recording for {duration_seconds} seconds...")
        
        if not self.start_recording():
            return False
        
        time.sleep(duration_seconds)
        
        if not self.stop_recording():
            return False
        
        return self.save_recording(filename)
    
    def cleanup(self):
        """
        Clean up audio resources.
        """
        if self.is_recording:
            self.stop_recording()
        
        if self.audio:
            self.audio.terminate()

def record_question(output_file: str = "temp_question.wav", 
                   max_duration: int = 30) -> Optional[str]:
    """
    Convenience function to record a question from the user.
    
    Args:
        output_file (str): Path to save the recorded question
        max_duration (int): Maximum recording duration in seconds
        
    Returns:
        Optional[str]: Path to the recorded file if successful, None otherwise
    """
    recorder = AudioRecorder()
    
    try:
        print(f"🎙️ Ready to record your question (max {max_duration}s)")
        print("Press Enter to start recording...")
        input()
        
        if not recorder.start_recording():
            return None
        
        print("Recording... Press Enter to stop")
        
        # Wait for user input or timeout
        start_time = time.time()
        while recorder.is_recording:
            if time.time() - start_time > max_duration:
                print(f"\nMaximum recording time ({max_duration}s) reached")
                break
            
            # Check if user pressed Enter (non-blocking)
            import select
            import sys
            
            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                input()  # Consume the Enter press
                break
            
            time.sleep(0.1)
        
        recorder.stop_recording()
        
        if recorder.save_recording(output_file):
            return output_file
        else:
            return None
            
    except KeyboardInterrupt:
        print("\nRecording cancelled by user")
        return None
    except Exception as e:
        print(f"Error during recording: {e}")
        return None
    finally:
        recorder.cleanup()

if __name__ == "__main__":
    # Test the audio recorder
    print("Audio Recorder Test")
    print("==================")
    
    # Test recording for 5 seconds
    test_file = "test_recording.wav"
    
    try:
        audio_file = record_question(test_file, max_duration=10)
        if audio_file:
            print(f"✅ Test recording successful: {audio_file}")
            
            # Check file size
            if os.path.exists(audio_file):
                size = os.path.getsize(audio_file)
                print(f"📊 File size: {size} bytes")
                
                # Clean up test file
                os.remove(audio_file)
                print("Test file cleaned up")
        else:
            print("Test recording failed")
            
    except Exception as e:
        print(f"Test error: {e}")