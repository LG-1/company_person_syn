import speech_recognition as sr
asr = sr.Recognizer()

def extract_text_from_audio_sphinx(audio_pth):
    text = ''
    with sr.AudioFile(audio_pth) as source:
        audio_data = asr.record(source)
        text = asr.recognize_sphinx(audio_data)
        
    return text

def extract_text_from_audio_google(audio_pth):
    text = ''
    with sr.AudioFile(audio_pth) as source:
        audio_data = asr.record(source)
        text = asr.recognize_google(audio_data, show_all=True)
        
    return text


# https: // github.com/Uberi/speech_recognition  # readme

# https://cloud.google.com/speech-to-text


# Google api
"this is a download from BBC learning English to find out more visit our website and that's because we're not able to record in our normal Studios during the coronavirus outbreak and staying at home"


# CMU Sphinx (works offline)
"this is a download from BBC learning English to find out more visit our website and that's because we're not able to record in our normal Studios during the coronavirus outbreak and staying at home"
