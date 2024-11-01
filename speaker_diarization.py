import whisper
from pyannote.audio import Pipeline
import torch
import webvtt
from pydub import AudioSegment

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["segments"]

def perform_diarization(audio_file):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="YOUR_HF_TOKEN")
    diarization = pipeline(audio_file)
    return diarization

def align_transcription_with_diarization(transcription, diarization):
    aligned_results = []
    for segment in transcription:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        
        # Find the speaker(s) for this segment
        speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= start_time < turn.end or turn.start < end_time <= turn.end:
                speakers.add(speaker)
        
        aligned_results.append({
            'start': start_time,
            'end': end_time,
            'text': text,
            'speakers': list(speakers)
        })
    
    return aligned_results

def main(audio_file):
    # Perform transcription
    print("Transcribing audio...")
    transcription = transcribe_audio(audio_file)
    
    # Perform diarization
    print("Performing diarization...")
    diarization = perform_diarization(audio_file)
    
    # Align results
    print("Aligning transcription with diarization...")
    aligned_results = align_transcription_with_diarization(transcription, diarization)
    
    # Print results
    for result in aligned_results:
        print(f"[{result['start']:.2f} - {result['end']:.2f}] {' & '.join(result['speakers'])}: {result['text']}")

if __name__ == "__main__":
    audio_file = "path/to/your/audio/file.wav"
    main(audio_file)