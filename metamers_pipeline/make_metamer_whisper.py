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
print('Loaded in Whisper model')

# Get target embedding by running signal through model
target = whisper_encoder(signal, decoder_input_ids=decoder_input_ids).encoder_last_hidden_state
print(target)


def loss_fn():
        y_pred = whisper_encoder(input_noise_init, decoder_input_ids=decoder_input_ids).encoder_last_hidden_state
        y_org = whisper_encoder(signal, decoder_input_ids=decoder_input_ids).encoder_last_hidden_state
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
        np.save('whisper/whisper_metamer.npy', input_noise_tensor_optimized)

    if i == iterations_adam - 1:
        print('Saving Final Weights')
        np.save('whisper/whisper_metamer.npy', input_noise_tensor_optimized)
        scipy.io.wavfile.write('whisper/whisper_metamer.wav', sr, input_noise_tensor_optimized)

    if i % log_loss_every_num == 0:
        loss_temp = loss_fn()
        print('Loss Value: ', loss_temp.item())
