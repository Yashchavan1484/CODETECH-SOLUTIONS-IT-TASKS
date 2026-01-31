import speech_recognition as sr

def speech_to_text(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        print("Listening to audio...")
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text

    except sr.UnknownValueError:
        return "Speech could not be understood."

    except sr.RequestError:
        return "Could not request results from speech recognition service."

# ------------------- DEMO -------------------

audio_path = "harvard.wav"

result = speech_to_text(audio_path)

print("TRANSCRIBED TEXT:\n")
print(result)

