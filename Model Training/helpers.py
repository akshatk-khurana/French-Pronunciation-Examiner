import os
from pydub import AudioSegment

def rename_and_convert_files(folder):
    counter = 0
    for filename in os.listdir(folder):
        if filename == '.DS_Store':
            continue
        old_path = os.path.join(folder, filename)
        if os.path.isfile(old_path):
            name, ext = os.path.splitext(filename)
            new_filename = f"speaking_{counter}.mp3"
            new_path = os.path.join(folder, new_filename)
            if ext.lower() != ".mp3":
                try:
                    audio = AudioSegment.from_file(old_path)
                    audio.export(new_path, format="mp3")
                    os.remove(old_path)
                except Exception as e:
                    print(f"Error converting {filename}: {e}")
            else:
                os.rename(old_path, new_path)
            counter += 1

rename_and_convert_files("Student Responses")