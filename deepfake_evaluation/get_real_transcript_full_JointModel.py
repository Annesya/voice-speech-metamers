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

sys.path.append('/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers')
from utils import *
from learner import Learner
from tokenizer import Tokenizer
from decoder import Speech_Decoder_Linear, Speaker_Decoder_Linear
from encoder import Speaker_Encoder, Speech_Encoder, Joint_Encoder


"""
Using joint model to get the real transcripts of the entire original audio
"""
torch.manual_seed(97)
sr = 16000 # target sampling rate

# load in joint 
config_path = "/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/config.yaml"

# Load config file
config = load_yaml_config(config_path)


#define a tokenizer for the vocabulary
tokenizer = Tokenizer(**config.text)

speaker_encoder = Speaker_Encoder(config.encoder.model_cache)
speech_encoder = Speech_Encoder(config.encoder.model_cache)

#define joint encoder
saganet = Joint_Encoder(config.saganet.d_model,
                        config.saganet.num_head,
                        config.saganet.dim_feedforward,
                        config.saganet.num_layers)

#define decoders
speech_decoder = Speech_Decoder_Linear()
speaker_decoder = Speaker_Decoder_Linear()


checkpoint = "/om2/user/annesyab/SLP_Project_2024/saganet/saganet_d-704_atthead-8/best42-val_loss0.64.ckpt"
saganet = Learner.load_from_checkpoint(checkpoint_path=checkpoint,
                                                config=config, 
                                                tokenizer=tokenizer,
                                                speech_encoder=speech_encoder,
                                                speaker_encoder=speaker_encoder,
                                                joint_encoder=saganet,
                                                speech_decoder=speech_decoder,
                                                speaker_decoder = speaker_decoder,)

print('Loaded in joint model')


## Read all the data 
audio_files = glob.glob('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/*.wav')

real_transcript = []

resampler = torchaudio.transforms.Resample(44100, sr)

for i in range(len(audio_files)):
  audio_1_path = os.path.join('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/',audio_files[i]) 
  real_audio, sr_audio = torchaudio.load(audio_1_path)
  real_audio = resampler(real_audio)
  input_audio_1 = real_audio[0,:]
  real_transcript.append(saganet({'input_values': input_audio_1.unsqueeze(0)})[1])

df = pd.DataFrame()
df['Audio_Name'] = audio_files
df['Joint_Real_Full'] = real_transcript

df.to_csv('JointModel_Transcription_RealSpeech_FULL_wParsing.csv')


## Read all the data 
audio_files = glob.glob('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/FAKE_PARSED/*.wav')

real_transcript = []

resampler = torchaudio.transforms.Resample(44100, sr)

for i in range(len(audio_files)):
  audio_1_path = os.path.join('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/FAKE_PARSED/',audio_files[i]) 
  real_audio, sr_audio = torchaudio.load(audio_1_path)
  real_audio = resampler(real_audio)
  input_audio_1 = real_audio[0,:]
  real_transcript.append(saganet({'input_values': input_audio_1.unsqueeze(0)})[1])

df = pd.DataFrame()
df['Audio_Name'] = audio_files
df['Joint_Fake_Full'] = real_transcript

df.to_csv('JointModel_Transcription_FakeSpeech_FULL_wParsing.csv')