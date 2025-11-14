import json
import sys
import whisper 

MODEL = "tiny"


def transcribe_audio_file(audio_path: str, output_path: str): 
    whisper_model = whisper.load_model(MODEL)
    res = whisper_model.transcribe(audio_path)
    print(f"TRANSCRIPTIONS :: {res}")
    with open(output_path, "w", encoding="utf-8") as f: 
        json.dump(res, f, indent=4)
    print(f"Wrote the transcripts to: {output_path}")
    


if __name__ == "__main__": 
    audio_path = sys.argv[0]
    output_path=sys.argv[1]
    transcribe_audio_file(audio_path, output_path)