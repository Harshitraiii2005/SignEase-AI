# utils/tts.py
from gtts import gTTS
import os
from datetime import datetime

def speak_from_text(text):
    try:
        tts = gTTS(text)
        filename = f"static/tts_output_{datetime.now().strftime('%H%M%S')}.mp3"
        tts.save(filename)
        return filename  # return the path to use in frontend
    except Exception:
        return None
