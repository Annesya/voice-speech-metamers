import scipy
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
import tqdm
import matplotlib.pyplot as plt

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
from learner_whisper import Learner
from tokenizer import Tokenizer
from decoder import Speech_Decoder_Linear, Speaker_Decoder_Linear
from encoder import Speaker_Encoder, Speech_Encoder, Joint_Encoder

import editdistance as ed 
def cer(hypothesis, groundtruth):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        p = list(p)#p.split(' ')
        t = list(t)#t.split(' ')
        err += float(ed.eval(p, t))
        tot += len(t)

    return err / tot

sr = 16000
min_snr, max_snr, snr_step = 0, 25, 5
# 15 min for 500 samples
n_samples = 1000

#################################################  WHISPER  #################################################
# Load config file
config_path = '/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/config_whisper.yaml'
config = load_yaml_config(config_path)

#define a tokenizer for the vocabulary
tokenizer = Tokenizer(**config.text)

#load pre-trained encoder model
speech_encoder = Speech_Encoder(config.encoder.model_cache)

#define joint encoder
whisper = Joint_Encoder(config.saganet.d_model,
                        config.saganet.num_head,
                        config.saganet.dim_feedforward,
                        config.saganet.num_layers)

#define decoders
speech_decoder = Speech_Decoder_Linear(d_model=config.saganet.d_model)


# checkpoint = "/om2/user/gelbanna/saganet/whisper_asr_bs-8_layers-4_heads-8_e-59_lr-0.0001_rs-42/best17-val_loss0.51.ckpt"
# layers = 4

checkpoint = "/om2/user/gelbanna/saganet/whisper_asr_bs-8_e-59_lr-0.0001_rs-42/best14-val_loss0.53.ckpt"
# layers = 2
whisper = Learner.load_from_checkpoint(
                config=config, 
                checkpoint_path=checkpoint,
                tokenizer=tokenizer,
                joint_encoder=whisper,
                speech_encoder=speech_encoder,
                speech_decoder=speech_decoder)

print('Loaded in WhisperCER_ model')


## Read all the data 
audio_files = glob.glob('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/*.wav')

real_transcript = []

resampler = torchaudio.transforms.Resample(44100, sr)

for i in range(len(audio_files)):
  audio_1_path = os.path.join('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/',audio_files[i]) 
  real_audio, sr_audio = torchaudio.load(audio_1_path)
  real_audio = resampler(real_audio)
  input_audio_1 = real_audio[0,:]
  real_transcript.append(whisper({'input_values': input_audio_1.unsqueeze(0)})[1])

df = pd.DataFrame()
df['Audio_Name'] = audio_files
df['WhisperCER__Real_Full'] = real_transcript

df.to_csv('WhisperCER__Transcription_RealSpeech_FULL_wParsing.csv')


## Read all the data 
audio_files = glob.glob('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/FAKE_PARSED/*.wav')

real_transcript = []

resampler = torchaudio.transforms.Resample(44100, sr)

for i in range(len(audio_files)):
  audio_1_path = os.path.join('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/FAKE_PARSED/',audio_files[i]) 
  real_audio, sr_audio = torchaudio.load(audio_1_path)
  real_audio = resampler(real_audio)
  input_audio_1 = real_audio[0,:]
  real_transcript.append(whisper({'input_values': input_audio_1.unsqueeze(0)})[1])

df = pd.DataFrame()
df['Audio_Name'] = audio_files
df['WhisperCER_Fake_Full'] = real_transcript

df.to_csv('WhisperCER__Transcription_FakeSpeech_FULL_wParsing.csv')