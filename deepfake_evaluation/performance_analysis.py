import pandas as pd
import numpy as np
from joint_model_test import cer


csv_path = "/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/deepfake_evaluation/Saganet_Transcription_Fake_and_Real_Speech.csv"

df = pd.read_csv(csv_path)

hypothesis = df['Saganet_on_Fake']
reference = df['Saganet_on_Real']

performance = cer(hypothesis, reference)

print('Character Error Rate:', performance)