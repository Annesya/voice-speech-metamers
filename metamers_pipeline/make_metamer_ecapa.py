import torch
import torchaudio
import numpy as np
import sys
import scipy
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from speechbrain.inference.speaker import EncoderClassifier

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
model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
print('Loaded in ECAPA model')

# Get target embedding by running signal through model
target = model.encode_batch(signal)[0]

def loss_fn():
        y_pred = model.encode_batch(input_noise_init)[0]
        y_org = model.encode_batch(signal)[0]
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
        # save out metamer every n iterations
        input_noise_tensor_optimized = input_noise_init.detach().numpy()
        print(f'Saving Weights, {i/iterations_adam}%')
        np.save('ecapa/Ecapa_metamer.npy', input_noise_tensor_optimized)

    if i == iterations_adam - 1:
        # save out final metamer
        print('Saving Final Weights')
        np.save('ecapa/Ecapa_metamer.npy', input_noise_tensor_optimized)
        scipy.io.wavfile.write('ecapa/Ecapa_metamer.wav', sr, input_noise_tensor_optimized)

    if i % log_loss_every_num == 0:
        # calculate loss and print
        loss_temp = loss_fn()
        print('Loss Value: ', loss_temp.item())
