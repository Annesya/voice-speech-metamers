import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jiwer import wer, cer

### ANALYSIS on Joint Model

csv_path = "/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/JointModel_Transcription_FakeSpeech_FULL_wParsing.csv"

df = pd.read_csv(csv_path)
df['Speaker'] = df['Audio_Name'].apply(lambda x: x.split('/')[-1].split('-')[0].lower())
df['Speaker_ID'] = df['Audio_Name'].apply(lambda x: x.split('/')[-1].split('_')[1].split('.')[0])

unique_speakers = df['Speaker'].unique()

# Whipser WER Reference
csv_path = "/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Whisper_Transcription_RealSpeech_FULL.csv"
df_ref = pd.read_csv(csv_path)

cer_dict_mean = {}
cer_dict_std = {}

for speaker in unique_speakers:
    print(speaker)
    df_hypothesis = df[df['Speaker']==speaker]
    hypothesis = df_hypothesis['Joint_Fake_Full']
    temp_list = []
    for i in range(len(hypothesis)):
        reference_path = '/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/'+df_hypothesis['Speaker'].iloc[i]+'-original_'+df_hypothesis['Speaker_ID'].iloc[i]+'.wav'
        df_ref_choose = df_ref[df_ref['Audio_Name']==reference_path]
        ref_transcription = str(df_ref_choose['Whisper_Real_Full'].iloc[0])
        hyp_transcription = str(hypothesis.iloc[i])
        temp_list.append(cer(ref_transcription, hyp_transcription))
    temp_mean = np.mean(np.array(temp_list))
    temp_std = np.std(np.array(temp_list))
    cer_dict_mean[speaker] = temp_mean
    cer_dict_std[speaker] = temp_std

np.save('Joint_OnFake_Mean.npy', cer_dict_mean)
np.save('Joint_OnFake_STD.npy', cer_dict_std)

plt.figure(figsize=[10,6])
plt.subplot(1,2,1)
x = np.arange(len(unique_speakers))
y = np.array([cer_dict_mean[speaker] for speaker in unique_speakers])
y_err = np.array([cer_dict_std[speaker] for speaker in unique_speakers])
plt.errorbar(x,y,y_err,label='Joint Model')
plt.xticks(x, unique_speakers, rotation=45)
plt.xlabel('Speakers')
plt.ylabel('CER')
plt.title('Joint Model Performance')
plt.ylim([0,7])
# plt.savefig('JointModel_OnFake_SpeakerWise.jpg')

### ANALYSIS on Whisper CER

csv_path = "/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/WhisperCER__Transcription_FakeSpeech_FULL_wParsing.csv"

df = pd.read_csv(csv_path)
df['Speaker'] = df['Audio_Name'].apply(lambda x: x.split('/')[-1].split('-')[0].lower())
df['Speaker_ID'] = df['Audio_Name'].apply(lambda x: x.split('/')[-1].split('_')[1].split('.')[0])

unique_speakers = df['Speaker'].unique()

# Whipser WER Reference
csv_path = "/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Whisper_Transcription_RealSpeech_FULL.csv"
df_ref = pd.read_csv(csv_path)

cer_dict_mean = {}
cer_dict_std = {}

for speaker in unique_speakers:
    print(speaker)
    df_hypothesis = df[df['Speaker']==speaker]
    hypothesis = df_hypothesis['WhisperCER_Fake_Full']
    temp_list = []
    for i in range(len(hypothesis)):
        reference_path = '/om2/scratch/Thu/annesyab/Deepfake_Datasets/archive/KAGGLE/AUDIO/REAL_PARSED/'+df_hypothesis['Speaker'].iloc[i]+'-original_'+df_hypothesis['Speaker_ID'].iloc[i]+'.wav'
        df_ref_choose = df_ref[df_ref['Audio_Name']==reference_path]
        ref_transcription = str(df_ref_choose['Whisper_Real_Full'].iloc[0])
        hyp_transcription = str(hypothesis.iloc[i])
        temp_list.append(cer(ref_transcription, hyp_transcription))
    temp_mean = np.mean(np.array(temp_list))
    temp_std = np.std(np.array(temp_list))
    cer_dict_mean[speaker] = temp_mean
    cer_dict_std[speaker] = temp_std

np.save('WhisperCER_OnFake_Mean.npy', cer_dict_mean)
np.save('WhisperCER_OnFake_STD.npy', cer_dict_std)

# plt.figure(figsize=[10,6])
plt.subplot(1,2,2)
x = np.arange(len(unique_speakers))
y = np.array([cer_dict_mean[speaker] for speaker in unique_speakers])
y_err = np.array([cer_dict_std[speaker] for speaker in unique_speakers])
plt.errorbar(x,y,y_err,label='Joint Model')
plt.xticks(x, unique_speakers, rotation=45)
plt.xlabel('Speakers')
plt.ylabel('CER')
plt.title('Whisper on Character Model Performance')
plt.ylim([0,7])
# plt.savefig('JointModel_OnFake_SpeakerWise.jpg')
plt.savefig('Comparison_Joint_WhisperCER_OnFake_Speakerwise.jpg')
