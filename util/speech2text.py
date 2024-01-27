from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline,VitsTokenizer, VitsModel, set_seed
import numpy
import streamlit as st
import sounddevice as sd
import time
import scipy
import torch

class Record():
        def __init__(self,sampling_rate):
                self.freq = sampling_rate
                self.duration = 5
                self.record()
                
        def record(self):
                print('Recording')
                with st.spinner('Recording'):
                        self.recording = sd.rec(int(self.duration * self.freq), 
                                                        samplerate=self.freq, channels=1)
                        time.sleep(5)
                sd.wait()
                
                print('Recording Over')

        def output(self):
                return self.recording
        
      
class Whisper():
        def __init__(self):
                print('Loading Model')
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                model_id = "openai/whisper-large-v3"
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
                            )
                
                self.model.to(device)
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.pipe = pipeline(
                        "automatic-speech-recognition",
                        model=self.model,
                        tokenizer=self.processor.tokenizer,
                        feature_extractor=self.processor.feature_extractor,
                        max_new_tokens=128,
                        chunk_length_s=30,
                        batch_size=16,
                        return_timestamps=True,
                        torch_dtype=torch_dtype,
                        device=device,
                        )
                print('Model Loaded')

        def process(self,speech):
                print('Processing Voice')
                
                result = self.pipe(speech,generate_kwargs={"language": "urdu"})
                transcription = result["text"]
                print('Process Complete')
                print('Output')
                return transcription
        

class ToSpeech():
        def __init__(self):
              self.tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-urd-script_arabic")
              self.model = VitsModel.from_pretrained("facebook/mms-tts-urd-script_arabic")

        def process(self,input):
              inputs = self.tokenizer(text=input, return_tensors="pt")
              with torch.no_grad():
                outputs = self.model(**inputs)
              set_seed(100)
              waveform = outputs.waveform[0]
              scipy.io.wavfile.write("techno.wav", rate=self.model.config.sampling_rate, data=waveform.numpy())
