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
audio_path = df['Unnamed: 1'][22:]
audio_path_corrected = []

for i in range(len(audio_path)):
    path = audio_path.iloc[i]
    wav_name = path.split('/')[-1]
    full_wav_path = os.path.join('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/FAKE_PARSED/', wav_name)
    audio_path_corrected.append(full_wav_path)

real_transcript = []
fake_transcript = []

resampler = torchaudio.transforms.Resample(44100, sr)

for i in range(len(audio_path)):
  audio_1_path = os.path.join('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/FAKE_PARSED/',audio_path.iloc[i]) ## faked speaker
  audio_2_path = os.path.join('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/',audio_path.iloc[i].split('-')[0]+'-original_'+audio_path.iloc[i].split('_')[1]) ## source speaker

  fake_audio, sr_audio = torchaudio.load(audio_1_path)
  fake_audio = resampler(fake_audio)
  input_audio_1 = fake_audio[0,:]
  fake_audio, sr_audio = torchaudio.load(audio_2_path)
  fake_audio = resampler(fake_audio)
  input_audio_2 = fake_audio[0,:]

  # decode token ids to text
  transcription_1 = run_whisper(input_audio_1)
  transcription_2 = run_whisper(input_audio_2)
  fake_transcript.append(transcription_1)
  real_transcript.append(transcription_2)

df = pd.DataFrame()
df['Audio_Name'] = audio_path_corrected
df['Whisper_on_Fake'] = fake_transcript
df['Whisper_on_Real'] = real_transcript

df.to_csv('Whisper_Transcription_Fake_and_Real_Speech.csv')