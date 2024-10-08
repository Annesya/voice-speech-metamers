# -*- coding: utf-8 -*-
"""Audio_Content_Parsing

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PfcUlDuSYLxJVNR3HW7UtO9R8PTGcuc8
"""

from collections import defaultdict

import torch
import torchaudio
import IPython
import os
import glob
import pandas as pd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration


## list all the audios in the fake and real dataset
real_files_list = glob.glob('/om2/scratch/Fri/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL/*.wav')
fake_files_list = glob.glob('/om2/scratch/Fri/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/FAKE/*.wav')

new_audio_real_path = '/om2/scratch/Fri/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/'
# new_audio_fake_path = '/content/drive/MyDrive/Deepfake_Analysis/AUDIO/FAKE_PARSED/'

for i in range(len(real_files_list)):
  real_audio, sr_audio = torchaudio.load(real_files_list[i])
  print(len(real_audio[0])/sr_audio)
  num_snippets = int((len(real_audio[0])/sr_audio)/10)
  for j in range(num_snippets):
    start_idx = j*10*sr_audio
    end_idx = (j+1)*10*sr_audio
    audio = real_audio[:,start_idx:end_idx]
    snippet_name = real_files_list[i].split('/')[-1].split('.')[0] + '_' + str(j) + '.wav'
    torchaudio.save(new_audio_real_path + snippet_name, audio, sr_audio)

new_audio_fake_path = '/om2/scratch/Fri/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/FAKE_PARSED/'

for i in range(len(fake_files_list)):
  real_audio, sr_audio = torchaudio.load(fake_files_list[i])
  print(len(fake_files_list[0])/sr_audio)
  num_snippets = int((len(real_audio[0])/sr_audio)/10)
  for j in range(num_snippets):
    start_idx = j*10*sr_audio
    end_idx = (j+1)*10*sr_audio
    audio = real_audio[:,start_idx:end_idx]
    snippet_name = fake_files_list[i].split('/')[-1].split('.')[0] + '_' + str(j) + '.wav'
    torchaudio.save(new_audio_fake_path + snippet_name, audio, sr_audio)
