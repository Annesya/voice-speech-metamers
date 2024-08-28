import torch
import torchaudio
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
#from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, AutoModel

torch.manual_seed(100)

# Initialize input_noise_init here
audio = '/om2/user/amagaro/voice-speech-metamers/metamers_pipeline/kell2018/metamers/psychophysics_wsj400_jsintest_inversion_loss_layer_RS0_I3000_N8/0_SOUND_million/orig.wav'
sr = 16000
signal, fs = torchaudio.load(audio)
signal = signal.squeeze(0)

input_noise_init = torch.randn(signal.shape)
input_noise_init = input_noise_init.squeeze(0)
print(input_noise_init.shape)
input_noise_init = input_noise_init * torch.std(signal) / torch.std(input_noise_init)
input_noise_init = torch.nn.parameter.Parameter(input_noise_init, requires_grad=True)

mse_loss = torch.nn.MSELoss()  # Assuming CrossEntropyLoss is being used

print('Initializing Optimizer')
iterations_adam = 30000
log_loss_every_num = 50
starting_learning_rate_adam = 0.1
adam_exponential_decay = 0.95

INIT_LR = 0.01
MAX_LR = 0.1
step_size = 2 * log_loss_every_num

optimizer = optim.SGD([input_noise_init], lr=INIT_LR)
clr = optim.lr_scheduler.CyclicLR(optimizer, base_lr=INIT_LR, max_lr=MAX_LR)

# load in model 
feature_extract = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
model = AutoModel.from_pretrained("openai/whisper-base")
decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
print('Loaded in Whisper model')

# Get target embedding by running signal through model
target = model(feature_extract(signal, sampling_rate=16000, return_tensors="pt").input_features, decoder_input_ids=decoder_input_ids).encoder_last_hidden_state

def loss_fn():
        y_pred = model(feature_extract(input_noise_init.detach().numpy(), sampling_rate=16000, return_tensors="pt").input_features, decoder_input_ids=decoder_input_ids).encoder_last_hidden_state
        y_org = target
        loss_value = mse_loss(y_pred,y_org)
        return loss_value


print('Performing optimization')

for i in range(iterations_adam + 1):
    optimizer.zero_grad()
    # Assuming loss is calculated here
    loss = loss_fn()
    loss.backward()
    optimizer.step()
    clr.step()

    if i % log_loss_every_num == 0:
        input_noise_tensor_optimized = input_noise_init.detach().numpy()
        print('Saving Weights')
        np.save('Whisper_metamer.npy', input_noise_tensor_optimized)

    if i == iterations_adam - 1:
        print('Saving Final Weights')
        np.save('Whisper_metamer.npy', input_noise_tensor_optimized)

    if i % log_loss_every_num == 0:
        loss_temp = loss_fn()
        print('Loss Value: ', loss_temp.item())

import torch
import torchaudio
import numpy as np
import sys
import scipy
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from transformers import AutoFeatureExtractor, AutoModel

torch.manual_seed(100)

# Initialize input_noise_init here
audio = sys.argv[2]
sr = 16000
signal, fs = torchaudio.load(audio)

if len(signal.shape)>1:
    # Reshape signal as necessary
    signal = torch.squeeze(signal)

# initialize random noise 
input_noise_init = torch.randn(signal.shape)
input_noise_init = input_noise_init * torch.std(signal) / torch.std(input_noise_init)
input_noise_init = torch.nn.parameter.Parameter(input_noise_init, requires_grad=True)

# initialize loss and optimizer
mse_loss = torch.nn.MSELoss()  # Assuming CrossEntropyLoss is being used

print('Initializing Optimizer')
iterations_adam = 30000
log_loss_every_num = 50
starting_learning_rate_adam = 0.1
adam_exponential_decay = 0.95

INIT_LR = 0.01
MAX_LR = 0.1
step_size = 2 * log_loss_every_num

optimizer = optim.SGD([input_noise_init], lr=INIT_LR)
clr = optim.lr_scheduler.CyclicLR(optimizer, base_lr=INIT_LR, max_lr=MAX_LR)

# load in model 
whisper_feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
whisper_encoder = AutoModel.from_pretrained("openai/whisper-base")#, cache_dir=cache_dir)
decoder_input_ids = torch.tensor([[1, 1]]) * whisper_encoder.config.decoder_start_token_id
whisper_encoder.eval()

def run_model(input):
    """
    runs the whisper model when given audio input
    """
    input = whisper_feature_extractor(input.detach().cpu(), sampling_rate=sr, return_tensors="pt").input_features
    # UPDATE: Should I be pushing to cuda?
    output = whisper_encoder(input, decoder_input_ids=decoder_input_ids)
    return output.encoder_last_hidden_state

print('Loaded in Whisper model')

# Get target embedding by running signal through model
target = run_model(signal)
print(target)
print(target.shape)


def loss_fn():
        y_pred = run_model(input_noise_init)
        y_org = run_model(signal)
        loss_value = mse_loss(y_pred,y_org)
        return loss_value


print('Performing optimization')

for i in range(iterations_adam + 1):
    optimizer.zero_grad()
    # Assuming loss is calculated here
    loss = loss_fn()
    loss.backward()
    optimizer.step()
    clr.step()

    if i % log_loss_every_num == 0:
        input_noise_tensor_optimized = input_noise_init.detach().numpy()
        print(f'Saving Weights, {i/iterations_adam}%')
        np.save('whisper/whisper_metamer.npy', input_noise_tensor_optimized)

    if i == iterations_adam - 1:
        print('Saving Final Weights')
        np.save('whisper/whisper_metamer.npy', input_noise_tensor_optimized)
        scipy.io.wavfile.write('whisper/whisper_metamer.wav', sr, input_noise_tensor_optimized)

    if i % log_loss_every_num == 0:
        loss_temp = loss_fn()
        print('Loss Value: ', loss_temp.item())
