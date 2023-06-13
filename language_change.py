from libretranslatepy import LibreTranslateAPI

def translate_text(text,language):
    lt = LibreTranslateAPI("https://translate.argosopentech.com/")

    translator = lt.translate(text,"en", language)
    
    print(translator)
