# Audio-to-RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline for audio content. It transcribes audio files with speaker separation, embeds the content using VoyageAI's embeddings, stores them in MongoDB, and provides a question-answering interface using local LLMs through Ollama.

## 🌟 Features

- 🎙️ **Audio Transcription** with speaker diarization using AssemblyAI
- 🧠 **High-quality Embeddings** using VoyageAI's `voyage-context-3` model
- 📚 **Vector Storage** in MongoDB with efficient vector search
- 🤖 **Local LLM Integration** through Ollama
- 🔄 **Streaming Responses** for real-time answer generation
- 🎯 **Accurate Context Retrieval** using cosine similarity search

## 🛠️ Requirements

- Python 3.8+
- MongoDB (local or Atlas)
- [Ollama](https://ollama.com) with `gpt-oss` model
- AssemblyAI API key
- VoyageAI API key

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TheBearguy/Audio-to-RAG-POC.git
   cd Audio-to-RAG-POC
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in `.env`:
   ```env
   MONGO_URI="your_mongodb_connection_string"
   VOYAGE_API_KEY="your_voyage_api_key"
   ASSEMBLYAI_API_KEY="your_assemblyai_api_key"
   ```

## 🏗️ Architecture

The pipeline consists of four main components:

1. **Audio Transcription** (`transcribe.py`)
   - Converts audio to text with speaker identification
   - Uses AssemblyAI's advanced transcription service

2. **Embedding Generation** (`embed_and_store.py`)
   - Generates embeddings using VoyageAI
   - Stores utterances and embeddings in MongoDB

3. **Vector Search** (`retrieval.py`)
   - Creates and manages vector search indices
   - Performs semantic similarity search

4. **Response Generation** (`generate_response.py`)
   - Generates contextual responses using local LLM
   - Provides streaming response interface

## ⚠️ Important Notes

- Ensure audio files are in a supported format (WAV, MP3, etc.)
- For large audio files, consider the AssemblyAI API costs
- The gpt-oss model requires significant RAM (13+ GB)
- MongoDB vector search requires MongoDB 7.0+


### Getting Help

- Check the error message in the diagnostic output (`python src/main.py`)
- Review the component logs for detailed error information
- Ensure all services (MongoDB, Ollama) are running
- Verify Python environment has all dependencies installed
