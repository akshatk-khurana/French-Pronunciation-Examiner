import os
import whisper
from pydub import AudioSegment

model = whisper.load_model("base")

input_folder = "Student Responses"
output_folder = "Chunks"
os.makedirs(output_folder, exist_ok=True)


for filename in os.listdir(input_folder):
    if filename.endswith(".mp3"):
        input_path = os.path.join(input_folder, filename)
        
        result = model.transcribe(input_path, language="fr", word_timestamps=True)
        
        # Load the audio file using pydub
        audio = AudioSegment.from_file(input_path)
        
        # Iterate over each segment in the transcription
        for segment in result["segments"]:
            for word_info in segment["words"]:
                word = word_info["word"].strip().lower()
                start_time = word_info["start"]
                end_time = word_info["end"]
                
                # Convert start and end times to milliseconds
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                
                # Extract the word audio segment
                word_audio = audio[start_ms:end_ms]
                
                # Create a safe filename
                safe_word = "".join(c for c in word if c.isalnum() or c in ['_', '-'])
                output_filename = f"{safe_word}_{start_ms}.mp3"
                output_path = os.path.join(output_folder, output_filename)
                
                # Export the word audio segment
                word_audio.export(output_path, format="mp3")