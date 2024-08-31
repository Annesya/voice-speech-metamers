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

print('Whisper loaded')

## Read all the data 
df = pd.read_csv('/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Human_Experiment_Data/Copy of Voice_Identification_Experiment - Annesya.csv')
audio_path = df['Unnamed: 1'][22:]
audio_path_corrected = []

for i in range(len(audio_path)):
    path = audio_path.iloc[i]
    wav_name = path.split('/')[-1]
    full_wav_path = os.path.join('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/FAKE_PARSED/', wav_name)
    audio_path_corrected.append(full_wav_path)

## get the transcription from the whisper_cer
whisper_cer_transcription_fake = []
whisper_cer_transcription_real = []


target_length = 20 # number of seconds
sr = 16000
num_samples = int(target_length * sr)

resampler = torchaudio.transforms.Resample(44100, sr)

## Speech recognition score calculation

for i in range(len(audio_path)):
    audio_1_path = os.path.join('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/FAKE_PARSED/',audio_path.iloc[i]) ## faked speaker
    audio_2_path = os.path.join(
        '/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/',
        audio_path.iloc[i].split('-')[0]+'-original_'+audio_path.iloc[i].split('_')[1]) ## source speaker
    
    fake_audio, sr_audio = torchaudio.load(audio_1_path)
    fake_audio = resampler(fake_audio)
    input_audio_1 = fake_audio[0,:]
    fake_audio, sr_audio = torchaudio.load(audio_2_path)
    fake_audio = resampler(fake_audio)
    input_audio_2 = fake_audio[0,:]

    whisper_cer_transcription_fake.append(whisper({'input_values': input_audio_1.unsqueeze(0)})[1])
    whisper_cer_transcription_real.append(whisper({'input_values': input_audio_2.unsqueeze(0)})[1])
    
print('Finished embedding extraction')

df = pd.DataFrame()
df['Audio_Name'] = audio_path_corrected
df['WhisperCER_on_Fake'] = whisper_cer_transcription_fake
df['WhisperCER_on_Real'] = whisper_cer_transcription_real

df.to_csv('Saganet_Transcription_Fake_and_Real_Speech_v2.csv')

# Human Data Preparation:

sub_1 = pd.read_csv('/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Human_Experiment_Data/Voice_Identification_Experiment - Gasser.csv')
sub_2 = pd.read_csv('/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Human_Experiment_Data/Voice_Identification_Experiment - Sagarika.csv')
sub_3 = pd.read_csv('/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Human_Experiment_Data/Voice_Identification_Experiment - Richard.csv')
sub_4 = pd.read_csv('/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Human_Experiment_Data/Voice_Identification_Experiment - Annika.csv')
sub_5 = pd.read_csv('/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Human_Experiment_Data/Voice_Identification_Experiment - Ajani.csv')
sub_6 = pd.read_csv('/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Human_Experiment_Data/Voice_Identification_Experiment - AI agent.csv')

sub_1_rating = sub_1['Unnamed: 4'][24:]
sub_2_rating = sub_2['Unnamed: 4'][24:]
sub_3_rating = sub_3['Unnamed: 4'][22:]
sub_4_rating = sub_4['Unnamed: 4'][24:]
sub_5_rating = sub_5['Unnamed: 4'][24:]
sub_6_rating = sub_6['Unnamed: 4'][24:]

sub_1_rating = [sub_1_rating.iloc[i] for i in range(len(sub_1_rating))]
sub_2_rating = [sub_2_rating.iloc[i] for i in range(len(sub_2_rating))]
sub_3_rating = [sub_3_rating.iloc[i] for i in range(len(sub_3_rating))]
sub_4_rating = [sub_4_rating.iloc[i] for i in range(len(sub_4_rating))]
sub_5_rating = [sub_5_rating.iloc[i] for i in range(len(sub_5_rating))]
sub_6_rating = [sub_6_rating.iloc[i] for i in range(len(sub_6_rating))]

subject_response_mean = (np.array(sub_1_rating, dtype=float) + 
                         np.array(sub_2_rating, dtype=float) + 
                         np.array(sub_3_rating, dtype=float) + 
                         np.array(sub_4_rating, dtype=float) + 
                         np.array(sub_5_rating, dtype=float) + 
                         np.array(sub_6_rating, dtype=float)) / 6

plt.figure(figsize=[6,4])
plt.scatter(subject_response_mean, np.array(whisper_cer_transcription_fake))
plt.legend()
plt.ylim([0,1])
plt.xlabel('Mean Rating')
plt.ylabel('Joint Model Score')
plt.title('Rating vs Score')
plt.xlim([1,5])
plt.ylim([0,1])

plt.savefig('Whisper_CER_Speech.jpg')



