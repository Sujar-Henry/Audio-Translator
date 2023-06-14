import pyttsx3
from language_change import translate_text


def texttospeech(translated_words):# Initialize the pyttsx3 engine
    engine = pyttsx3.init()

    # Set the properties for the speech
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.8)  # Volume (0.0 to 1.0)
    
    # Convert text to speech
    engine.say(translated_words)

    # Run the speech engine
    engine.runAndWait()

