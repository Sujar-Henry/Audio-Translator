from googletrans import Translator

def translate_text(text):
    translator = Translator()

    translated_text = translator.translate(text, dest="es")  

    print(translated_text.text)
