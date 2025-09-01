import assemblyai as aai

def transcribe_with_speakers(audio_path: str) -> list:
    """
    Transcribes an audio file and identifies different speakers.

    Args:
        audio_path (str): The path to the audio file to be transcribed.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              the speaker and their corresponding text.
              e.g., [{'speaker': 'Speaker A', 'text': 'Hello there.'}]
    """
    # Create a Transcriber object
    transcriber = aai.Transcriber()

    # Configure the transcription to enable speaker labels
    config = aai.TranscriptionConfig(speaker_labels=True)

    # Perform the transcription
    transcript = transcriber.transcribe(audio_path, config=config)

    if not transcript.utterances:
        print("No speaker utterances found.")
        return []

    # Process the transcript to create a clean list
    speaker_transcriptions = []
    for utterance in transcript.utterances:
        speaker_transcriptions.append({
            "speaker": f"Speaker {utterance.speaker}",
            "text": utterance.text
        })

    return speaker_transcriptions