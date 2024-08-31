import scipy
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
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
from learner import Learner
from tokenizer import Tokenizer
from decoder import Speech_Decoder_Linear, Speaker_Decoder_Linear
from encoder import Speaker_Encoder, Speech_Encoder, Joint_Encoder

import editdistance as ed 
def cer(hypothesis, groundtruth):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        p = p.split(' ')
        t = t.split(' ')
        err += float(ed.eval(p, t))
        tot += len(t)
        breakpoint()

    return err / tot

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
df = pd.read_csv('/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Human_Experiment_Data/Copy of Voice_Identification_Experiment - Annesya.csv')
audio_path = df['Unnamed: 1'][22:]
audio_path_corrected = []

for i in range(len(audio_path)):
    path = audio_path.iloc[i]
    wav_name = path.split('/')[-1]
    full_wav_path = os.path.join('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/FAKE_PARSED/', wav_name)
    audio_path_corrected.append(full_wav_path)


## get the similarity between two combinations of fake and real speakers 
fake_speakers_score_rating_12 = []
fake_speakers_score_rating_13 = []

## get the transcription from the saganet
saganet_transcription_fake = []
saganet_transcription_real = []


target_length = 20 # number of seconds
sr = 16000
num_samples = int(target_length * sr)

resampler = torchaudio.transforms.Resample(44100, sr)
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

## Speaker Identification Score calculation

for i in range(len(audio_path)):
    audio_1_path = os.path.join('/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/FAKE_PARSED/',audio_path.iloc[i]) ## faked speaker
    audio_2_path = os.path.join(
        '/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/',
        audio_path.iloc[i].split('-')[0]+'-original_'+audio_path.iloc[i].split('_')[1]) ## source speaker
    audio_3_path = os.path.join(
        '/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/',
        audio_path.iloc[i].split('-')[2].split('_')[0].lower()+'-original_2.wav') ## target speaker original

    fake_audio, sr_audio = torchaudio.load(audio_1_path)
    fake_audio = resampler(fake_audio)
    input_audio_1 = fake_audio[0,:]
    fake_audio, sr_audio = torchaudio.load(audio_2_path)
    fake_audio = resampler(fake_audio)
    input_audio_2 = fake_audio[0,:]
    fake_audio, sr_audio = torchaudio.load(audio_3_path)
    fake_audio = resampler(fake_audio)
    input_audio_3 = fake_audio[0,:]

    decoder_output_audio_1 = saganet.speaker_decoder.linear_project_speaker(torch.mean(saganet({'input_values': input_audio_1.unsqueeze(0)})[0],dim=1))
    decoder_output_audio_2 = saganet.speaker_decoder.linear_project_speaker(torch.mean(saganet({'input_values': input_audio_2.unsqueeze(0)})[0],dim=1))
    decoder_output_audio_3 = saganet.speaker_decoder.linear_project_speaker(torch.mean(saganet({'input_values': input_audio_3.unsqueeze(0)})[0],dim=1))

    score12 = similarity(decoder_output_audio_1, decoder_output_audio_2)
    score13 = similarity(decoder_output_audio_1, decoder_output_audio_3)

    # print(score12)
    # print(score13)

    # print(saganet.speaker_decoder.linear_project_speaker.weight)

    saganet_transcription_fake.append(saganet({'input_values': input_audio_1.unsqueeze(0)})[1])
    saganet_transcription_real.append(saganet({'input_values': input_audio_2.unsqueeze(0)})[1])
    
    fake_speakers_score_rating_12.append(score12.cpu().detach().numpy()) # score between fake voice and source real voice -- high score means the fake voice retains a lot of info from the source voice, i.e., not good deepfake
    fake_speakers_score_rating_13.append(score13.cpu().detach().numpy()) # score between target voice real and fake -- high score means the deepfaking is good; should match with human rating

print('Finished embedding extraction')

df = pd.DataFrame()
df['Audio_Name'] = audio_path_corrected
df['Saganet_on_Fake'] = saganet_transcription_fake
df['Saganet_on_Real'] = saganet_transcription_real

df['Score_Fake_Real_Target_Speaker'] = fake_speakers_score_rating_13
df['Score_Fake_Target_Real_Source_Speaker'] = fake_speakers_score_rating_12

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
plt.scatter(subject_response_mean, np.array(fake_speakers_score_rating_13))
plt.legend()
plt.ylim([0,1])
plt.xlabel('Mean Rating')
plt.ylabel('Joint Model Score')
plt.title('Rating vs Score')
plt.xlim([1,5])
plt.ylim([0,1])

plt.savefig('JointModel_Speaker_v2.jpg')


