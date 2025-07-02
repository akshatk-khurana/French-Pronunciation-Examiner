"""This file contains any helper functions that are utilised by view functions in views.py.
"""

from django.conf import settings
from gtts import gTTS

import subprocess, os

numbers = "0123456789"

def validate_password(password):
    """Validate a password to ensure it contains at least one number and only letters/numbers.

    Args:
        password: The password string to validate.
    
    Returns:
        tuple: A tuple containing:
            valid: Whether the password is valid.
            error: An error message if the password is invalid, otherwise an empty string.
    
    """
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
    """Validate a username to ensure it contains only letters.

    Args:
        username (str): The username string to validate.
    
    Returns:
        tuple: A tuple containing:
            valid (bool): Whether the username is valid.
            error (str): An error message if the username is invalid, otherwise an empty string.
    """
    if username.isalpha():
        return (True, "")
    return (False, "Username can only contain letters.")

def convert_to_mp3(input_path, output_path):
    """Convert an audio file to .mp3 format using ffmpeg.

    Args:
        input_path: A string of the path to the input audio file.
        output_path: A string of the path to save the converted .mp3 file.
    """
    subprocess.run(
        [
            "ffmpeg", "-i", input_path, "-vn", "-ar", "44100", "-ac", "2",
            "-b:a", "192k", "-y", output_path
        ],
    )

def get_standard_pronunciation(word):
    """Generate a standard French pronunciation audio file for a given word using gTTS.

    Args:
        word: A string of the word or phrase to generate pronunciation for.
    """
    audio_dir = os.path.join(settings.BASE_DIR, "PronunciationApp", "audio")

    output_path = os.path.join(audio_dir, "spoken.mp3")

    tts = gTTS(word, lang='fr', slow=False)
    tts.save(output_path)