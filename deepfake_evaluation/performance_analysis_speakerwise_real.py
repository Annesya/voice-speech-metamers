import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jiwer import wer, cer

### ANALYSIS on Joint Model

csv_path = "/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/JointModel_Transcription_RealSpeech_FULL_wParsing.csv"

df = pd.read_csv(csv_path)
df['Speaker'] = df['Audio_Name'].apply(lambda x: x.split('/')[-1].split('-')[0])
unique_speakers = df['Speaker'].unique()

# Whipser WER Reference
csv_path = "/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Whisper_Transcription_RealSpeech_FULL.csv"

df_ref = pd.read_csv(csv_path)
df_ref['Speaker'] = df_ref['Audio_Name'].apply(lambda x: x.split('/')[-1].split('-')[0])

cer_dict_mean = {}
cer_dict_std = {}

for speaker in unique_speakers:
    df_hypothesis = df[df['Speaker']==speaker]
    hypothesis_joint_model_real = df_hypothesis['Joint_Real_Full']

    df_ref_byspeaker = df_ref[df_ref['Speaker']==speaker]
    reference = df_ref_byspeaker['Whisper_Real_Full']
    temp_list = []
    for i in range(len(reference)):
        # value = cer(hypothesis[i], reference[i])
        value = cer(reference.iloc[i], hypothesis_joint_model_real.iloc[i])
        temp_list.append(value)
    cer_mean = np.mean(np.array(temp_list))
    cer_std = np.std(np.array(temp_list))

    cer_dict_mean[speaker] = cer_mean
    cer_dict_std[speaker] = cer_std

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
# plt.savefig('JointModel_OnReal_SpeakerWise.jpg')

### ANALYSIS on Whisper CER

csv_path = "/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/WhisperCER__Transcription_RealSpeech_FULL_wParsing.csv"

df = pd.read_csv(csv_path)
df['Speaker'] = df['Audio_Name'].apply(lambda x: x.split('/')[-1].split('-')[0])
unique_speakers = df['Speaker'].unique()

# Whipser WER Reference
csv_path = "/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Whisper_Transcription_RealSpeech_FULL.csv"

df_ref = pd.read_csv(csv_path)
df_ref['Speaker'] = df_ref['Audio_Name'].apply(lambda x: x.split('/')[-1].split('-')[0])

cer_dict_mean = {}
cer_dict_std = {}

for speaker in unique_speakers:
    df_hypothesis = df[df['Speaker']==speaker]
    hypothesis_joint_model_real = df_hypothesis['WhisperCER__Real_Full']

    df_ref_byspeaker = df_ref[df_ref['Speaker']==speaker]
    reference = df_ref_byspeaker['Whisper_Real_Full']
    temp_list = []
    for i in range(len(reference)):
        # value = cer(hypothesis[i], reference[i])
        value = cer(reference.iloc[i], hypothesis_joint_model_real.iloc[i])
        temp_list.append(value)
    cer_mean = np.mean(np.array(temp_list))
    cer_std = np.std(np.array(temp_list))

    cer_dict_mean[speaker] = cer_mean
    cer_dict_std[speaker] = cer_std

# plt.figure()
plt.subplot(1,2,2)
x = np.arange(len(unique_speakers))
y = np.array([cer_dict_mean[speaker] for speaker in unique_speakers])
y_err = np.array([cer_dict_std[speaker] for speaker in unique_speakers])
plt.errorbar(x,y,y_err,label='Whisper Character Model')
plt.xticks(x, unique_speakers, rotation=45)
plt.xlabel('Speakers')
plt.ylabel('CER')
plt.title('Whisper on Character Model Performance')
plt.ylim([0,7])
# plt.savefig('WhisperCER_OnReal_SpeakerWise.jpg')
plt.savefig('Comparison_Joint_WhisperCER_OnReal_Speakerwise.jpg')