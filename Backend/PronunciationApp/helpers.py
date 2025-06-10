import subprocess
import torch
import os
from .snn import SiameseNetwork

harmful_characters = ["<", ">", "?", "!"]

def sanitise(s):
    s = s.strip()
    for char in harmful_characters:
        s = s.replace(char, "")
    
    return s

def convert_to_mp3(input_path, output_path):   
    subprocess.run(
        [
            "ffmpeg", "-i", input_path, "-vn", "-ar", "44100", "-ac", "2",
            "-b:a", "192k", "-y", output_path
        ],
    )