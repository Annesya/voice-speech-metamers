import torch
import torchaudio
import numpy as np
import sys
import scipy
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from transformers import AutoFeatureExtractor, AutoModel

# load in model imports
sys.path.append('/om2/user/salavill/misc/voice-speech-metamers/')
from utils import *
from learner import Learner
from tokenizer import Tokenizer
from decoder import Speech_Decoder_Linear, Speaker_Decoder_Linear
from encoder import Speaker_Encoder, Speech_Encoder, Joint_Encoder


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
config_path = "../config.yaml"
# Load config file
config = load_yaml_config(config_path)

#define a tokenizer for the vocabulary
tokenizer = Tokenizer(**config.text)

#load pre-trained encoder model
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

checkpoint = "/om2/user/gelbanna/saganet/saganet_d-704_atthead-8_ffd-512_num_layers-2_bs-8_e-59_lr-0.0001_rs-42/best57-val_loss0.65.ckpt"
model = Learner.load_from_checkpoint(checkpoint_path=checkpoint,
                                                config=config, 
                                                tokenizer=tokenizer,
                                                speech_encoder=speech_encoder,
                                                speaker_encoder=speaker_encoder,
                                                joint_encoder=saganet,
                                                speech_decoder=speech_decoder,
                                                speaker_decoder = speaker_decoder,)

print('Loaded in joint model')

# Get target embedding by running signal through model
target, _ = model(signal)
print(target)
print(target.shape)


def loss_fn():
        y_pred, _ = model(input_noise_init)
        y_org, _ = model(signal)
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
        np.save('joint_model/joint_metamer.npy', input_noise_tensor_optimized)

    if i == iterations_adam - 1:
        print('Saving Final Weights')
        np.save('joint_model/joint_metamer.npy', input_noise_tensor_optimized)
        scipy.io.wavfile.write('joint_model/joint_metamer.wav', sr, input_noise_tensor_optimized)

    if i % log_loss_every_num == 0:
        loss_temp = loss_fn()
        print('Loss Value: ', loss_temp.item())
