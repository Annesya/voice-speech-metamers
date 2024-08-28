import torch
import torchaudio
import numpy as np
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from speechbrain.inference.speaker import EncoderClassifier
from typing import Dict, Iterable, Callable
from torchvision.models import resnet50
from torch import nn, Tensor
from speechbrain.inference.encoders import WaveformEncoder


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.activation = {str(layer): torch.empty(0) for layer in range(4)}

        for layer in range(4):
            getattr(self.model.mods.embedding_model.blocks, str(layer)).register_forward_hook(
                self.get_layer_embeddings(str(layer)))

    def get_layer_embeddings(self, name):
        def hook(model, input, output):
            self.activation[name] = output[0]
        return hook

    def forward(self, x: Tensor):
        y = self.model(x)
        return y, self.activation


# load in model 
model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
print('Loaded in ECAPA model')

model_features = FeatureExtractor(model)


torch.manual_seed(100)

# Initialize input_noise_init here
audio = sys.argv[2]
sr = 16000
signal, fs = torchaudio.load(audio)

# initialize random noise 
input_noise_init = torch.randn(signal.shape)
input_noise_init = input_noise_init * torch.std(signal) / torch.std(input_noise_init)

# # model metamer (whole model)
# input_noise_final = torch.nn.parameter.Parameter(input_noise_init, requires_grad=True)

# # initialize loss and optimizer
# mse_loss = torch.nn.MSELoss()  # Assuming CrossEntropyLoss is being used

# print('Initializing Optimizer')
# iterations_adam = 3000 # change to 30K
# log_loss_every_num = 50
# starting_learning_rate_adam = 0.1
# adam_exponential_decay = 0.95

# INIT_LR = 0.01
# MAX_LR = 0.1
# step_size = 2 * log_loss_every_num

# optimizer = optim.SGD([input_noise_final], lr=INIT_LR)
# clr = optim.lr_scheduler.CyclicLR(optimizer, base_lr=INIT_LR, max_lr=MAX_LR)


# # for name, module in model.named_modules():
# #     print(name)

# # model2 = resnet50()
# # for name, module in model2.named_modules():
# #     print(name)

# # model3 = WaveformEncoder.from_hparams(source="speechbrain/ssl-wav2vec2-base-librispeech")
# # for name, module in model3.named_modules():
# #     print(name)

# # Get target embedding by running signal through model
# target = model.encode_batch(signal)[0]

# def loss_fn():
#     y_pred = model.encode_batch(input_noise_final)[0]
#     y_org = model.encode_batch(signal)[0]
#     loss_value = mse_loss(y_pred,y_org)
#     return loss_value


# print('Performing optimization')

# for i in range(iterations_adam + 1):
#     optimizer.zero_grad()
#     # Assuming loss is calculated here
#     loss = loss_fn()
#     loss.backward()
#     optimizer.step()
#     clr.step()

#     if i % log_loss_every_num == 0:
#         input_noise_tensor_optimized = input_noise_final.detach().numpy()
#         print('Saving Weights')
#         np.save('ecapa_metamer_final.npy', input_noise_tensor_optimized)

#     if i == iterations_adam - 1:
#         print('Saving Final Weights')
#         np.save('ecapa_metamer_final.npy', input_noise_tensor_optimized)

#     if i % log_loss_every_num == 0:
#         loss_temp = loss_fn()
#         print('Loss Value: ', loss_temp.item())


# layer-level metamer
print(sys.argv)
layer = sys.argv[1]
input_noise_layer = torch.nn.parameter.Parameter(input_noise_init, requires_grad=True)

# initialize loss and optimizer
mse_loss = torch.nn.MSELoss()  # Assuming CrossEntropyLoss is being used

print(f'Initializing Optimizer (layer {layer})')
iterations_adam = 30000
log_loss_every_num = 50
starting_learning_rate_adam = 0.1
adam_exponential_decay = 0.95

INIT_LR = 0.01
MAX_LR = 0.1
step_size = 2 * log_loss_every_num

optimizer = optim.SGD([input_noise_layer], lr=INIT_LR)
clr = optim.lr_scheduler.CyclicLR(optimizer, base_lr=INIT_LR, max_lr=MAX_LR)

# Get target embedding by running signal through model
target = model_features(signal)[1][str(layer)]

print(f'Performing optimization for layer {layer}')

for i in range(iterations_adam + 1):
    optimizer.zero_grad()
    # Assuming loss is calculated here
    y_pred = model_features(input_noise_layer)[1][str(layer)]
    loss = mse_loss(y_pred, target)
    loss.backward()
    optimizer.step()
    clr.step()

    if i % log_loss_every_num == 0:
        input_noise_tensor_optimized = input_noise_layer.detach().numpy()
        print(f'Saving Weights (layer {layer})')
        np.save(f'ecapa_metamer_layer_{layer}.npy', input_noise_tensor_optimized)

    if i == iterations_adam - 1:
        print(f'Saving Final Weights (layer {layer})')
        np.save(f'ecapa_metamer_layer_{layer}.npy', input_noise_tensor_optimized)

    if i % log_loss_every_num == 0:
        loss_temp = loss
        print('Loss Value: ', loss_temp.item())
