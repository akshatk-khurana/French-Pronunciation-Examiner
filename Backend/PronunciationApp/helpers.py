import subprocess
import torch
import os
from .snn import SiameseNetwork

numbers = "0123456789"

def validate_password(password):
    error = ""
    valid = True

    if password.isalnum():
        has_number = False
        for i in password:
            if i in numbers:
                has_number = True
        
        if not has_number:
            error = "Please include at least one number in your password."
            valid = False
    else:
        error = "Password can only include numbers and letters."
        valid = False
        
    return (valid, error)

def validate_username(username):
    if username.isalpha():
        return (True, "")
    return (False, "Username can only contain letters.")

def convert_to_mp3(input_path, output_path):   
    subprocess.run(
        [
            "ffmpeg", "-i", input_path, "-vn", "-ar", "44100", "-ac", "2",
            "-b:a", "192k", "-y", output_path
        ],
    )