import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jiwer import wer

import editdistance as ed 
def cer(hypothesis, groundtruth):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        p = p.split(' ')
        t = t.split(' ')
        err += float(ed.eval(p, t))
        tot += len(t)
        # break()

    return err / tot


csv_path = "/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Saganet_Transcription_Fake_and_Real_Speech.csv"

df = pd.read_csv(csv_path)

hypothesis = df['Saganet_on_Fake']
# hypothesis = df['Saganet_on_Real']

csv_path = "/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Whisper_Transcription_Fake_and_Real_Speech.csv"

df = pd.read_csv(csv_path)

reference = df['Whisper_on_Real']
hypothesis_whisper = df['Whisper_on_Fake']

cer_list_whisper = []
cer_list = []

for i in range(len(hypothesis)):
    value = cer(hypothesis[i], reference[i])
    cer_list.append(value)
    cer_list_whisper.append(wer(str(hypothesis_whisper[i]), str(reference[i])))

# performance = cer(hypothesis, reference)

# print('Character Error Rate:', performance)

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
plt.scatter(subject_response_mean, np.array(cer_list))
plt.legend()
plt.xlabel('Mean Rating')
plt.ylabel('Joint Model CER')
plt.title('Rating vs Score')
# plt.xlim([1,5])
# plt.ylim([0,1])

plt.savefig('JointModel_Speech_vs_Rating.jpg')

plt.figure(figsize=[6,4])
plt.scatter(subject_response_mean, np.array(cer_list_whisper))
plt.legend()
plt.xlabel('Mean Rating')
plt.ylabel('Whisper WER')
plt.title('Rating vs Score')
# plt.xlim([1,5])
# plt.ylim([0,1])

plt.savefig('Whisper_Speech_vs_Rating.jpg')

