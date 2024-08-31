import scipy
import numpy as np
import pandas as pd
import glob
import os
from collections import defaultdict

import torch
import torchaudio
import torchaudio.transforms as T
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, AutoModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration

"""
Using whisper model to get the real transcripts of the entire original audio
"""

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
whisper.config.forced_decoder_ids = None

sr = 16000

def run_whisper(input):

    input_features = processor(input, sampling_rate=sr, return_tensors="pt").input_features
    # generate token ids
    predicted_ids = whisper.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

## Read all the data 
df = pd.read_csv('/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Human_Experiment_Data/Copy of Voice_Identification_Experiment - Annesya.csv')
audio_files = glob.glob('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/*.wav')

real_transcript = []

resampler = torchaudio.transforms.Resample(44100, sr)

for i in range(len(audio_files)):
  audio_1_path = os.path.join('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/',audio_files[i]) ## faked speaker
  real_audio, sr_audio = torchaudio.load(audio_1_path)
  real_audio = resampler(real_audio)
  input_audio_1 = real_audio

  # decode token ids to text
  transcription = run_whisper(input_audio_1)
  real_transcript.append(transcription)

df = pd.DataFrame()
df['Audio_Name'] = audio_files
df['Whisper_Real_Full'] = real_transcript

df.to_csv('Whisper_Transcription_RealSpeech_FULL.csv')