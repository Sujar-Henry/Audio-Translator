from libretranslatepy import LibreTranslateAPI


lt = LibreTranslateAPI("https://translate.argosopentech.com/")

def translate_text(text, language):
    
    translator = lt.translate(text, "en", language)

    print(translator)
    
    return translator
    