import streamlit as st
from util.speech2text import Whisper, ToSpeech, Record
from audio_recorder_streamlit import audio_recorder
import base64

st.title('Urdu Speech Translator')

whisper = Whisper()
speech = ToSpeech()


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/wav">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )




if st.button('Record'):
    st.session_state['chatbot-disabled'] = True
    recording = Record(16000)
    audio = recording.output()[:,0]
    transcription = whisper.process(audio)
    print(transcription)
    speech.process(transcription)
    st.write("## Audio Playback")
    autoplay_audio("techno.wav")
    st.write(transcription)
  
