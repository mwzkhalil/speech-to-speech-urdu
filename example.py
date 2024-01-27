from util.speech2text import Whisper, ToSpeech, Record

whisper = Whisper()
speech = ToSpeech()
recording = Record(16000)
audio = recording.output()[:,0]
transcription = whisper.process(audio)
print(transcription)
speech.process(transcription)