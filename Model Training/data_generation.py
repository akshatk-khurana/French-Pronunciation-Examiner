import glob
import shutil
import os
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

qualities = {"g": 1.00,
            "b": 0.00,
            "m": 0.5}

named_folder_path = "Chunks"
spoken_destination_path = "Spoken Samples"
standard_destination_path = "Standard Samples"

named_mp3_files = glob.glob(f"{named_folder_path}/*.mp3")

def play_audio(path):
    """Play an audio file at the given path using pydub."""
    sound = AudioSegment.from_file(path)
    play(sound)

counter = None
with open("similarity_labels.txt", "r", encoding="utf-8") as readingfile:
    lines = readingfile.readlines()
    last_row = lines[-1].strip()
    counter = int(last_row.split(" ")[0]) + 1

# Populate the labels file with a record of each audio sample.
with open("similarity_labels.txt", "a") as datafile:
    for f in named_mp3_files:
        f = f.split("/")[1]

        word = None

        if "_" in f:
            word, identifier = f.split("_")
        else:
            word = f.split(".")[0]
        
        print()
        print(f"\033[92mWord being pronounced is {word}\033[0m")

        full_path = os.path.join(named_folder_path, f)
        play_audio(full_path)

        action = input("Include this file in training data? (y/n): ")

        if action == "y":
            play_audio(full_path)

            quality = input("Good/Bad/Middle (g/b/m): ")

            new_filename = f"human_{counter}.mp3"
            destination_file = f"{spoken_destination_path}/{new_filename}"

            shutil.copy(full_path, destination_file)

            destination_file = f"{standard_destination_path}/standard_{counter}.mp3"
            tts = gTTS(word, lang='fr', slow=False)
            tts.save(destination_file)
            
            datafile.write(f'{counter} {word.capitalize()} {qualities[quality]}\n')

            counter += 1

            os.system('clear')

            os.remove(full_path)
            print(f"Removed: {f}")
        else:
            os.remove(full_path)
            print(f"Removed: {f}")