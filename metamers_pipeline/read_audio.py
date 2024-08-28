import numpy as np
from scipy.io.wavfile import write
import torch
import torchaudio


audio = '/om2/user/amagaro/voice-speech-metamers/metamers_pipeline/kell2018/metamers/psychophysics_wsj400_jsintest_inversion_loss_layer_RS0_I3000_N8/0_SOUND_million/orig.wav'
sr = 16000
signal, fs = torchaudio.load(audio)

input_noise_init = torch.randn(signal.shape)
np.save('Ecapa_metamer_init.npy', input_noise_init)


