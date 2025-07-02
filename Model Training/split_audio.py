"""
This script splits each .mp3 file in the input folder into word-level audio chunks
using OpenAI's Whisper transcription. 
"""

import os
import whisper
from pydub import AudioSegment

model = whisper.load_model("base")

input_folder = "Student Responses"
output_folder = "Chunks"
os.makedirs(output_folder, exist_ok=True)

model = WhisperModel("base", compute_type="int8")

saved_count = 0

for file in os.listdir(input_folder):
    if file.endswith((".mp3")):
        input_path = os.path.join(input_folder, file)
        audio = effects.normalize(AudioSegment.from_file(input_path))

        temp_path = "temp.wav"
        audio.export(temp_path, format="wav")

        segments, _ = model.transcribe(temp_path, language="fr", beam_size=5, word_timestamps=True)
        for segment in segments:
            for word_info in segment.words:
                word = word_info.word.strip().lower()

                if not word.isalpha():
                    continue

                start = int(word_info.start * 1000)
                end = int(word_info.end * 1000)
                word_audio = audio[start:end]

                base_name = word
                output_path = os.path.join(output_folder, f"{base_name}.mp3")
                count = 1
                while os.path.exists(output_path):
                    output_path = os.path.join(output_folder, f"{base_name}_{count}.mp3")
                    count += 1

                word_audio.export(output_path, format="mp3")
                saved_count += 1

                if saved_count % 10 == 0:
                  print(f"{saved_count} chunks have been saved so far.")

        os.remove(temp_path)