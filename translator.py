#=================================AUDIO INPUT FROM MIC=================================
import pyaudio
import wave


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

stream = p.open(
    format = FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    frames_per_buffer = CHUNK)

print("start recording")

frames = []
seconds = 10
for i in range(0,int(RATE/CHUNK * seconds)):
    data = stream.read(CHUNK)
    frames.append(data)

print("recording stopped")

stream.stop_stream()
stream.close()
p.terminate


wf = wave.open("output.wav","wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close
#==========================AUDIO INPUT FROM MIC========================

#=============IMPORTS TO CONFIRM PYTORCH WORKING=====================================#
import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=============================IMPORTING SPEECH FILE========================
import IPython
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset

SPEECH_FILE = download_asset(r"C:\Users\sking\Desktop\Word_Detection_and_Translator\output.wav")


#======================CREATE PIPELINE

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())

#=======================CREATE MODEL

model = bundle.get_model().to(device)

#===============LOADING DATA
IPython.display.Audio(SPEECH_FILE)


#=============CHANGING RATE IF NOT THE SAME AS AUDIO CLIP
waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)


#====================EXTRACT ACOUSTIC FEATURES
with torch.inference_mode():
    features, _ = model.extract_features(waveform)



#===========FEATURE EXTRACTION AND CLASSIFICATION
with torch.inference_mode():
    emission, _ = model(waveform)


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

#====CREATE DECODER OBJECT AND DECODE THE TRANSCRIPT
decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])

print(transcript)
IPython.display.Audio(SPEECH_FILE)

