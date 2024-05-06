from speechbrain.inference.speaker import EncoderClassifier
import torch
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoFeatureExtractor, WhisperModel, WhisperTokenizer
from datasets import load_dataset, DatasetDict, Audio, load_metric

## Whisper Encoding
model = WhisperModel.from_pretrained("openai/whisper-base")
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
input_features = inputs.input_features
decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).encoder_last_hidden_state
list(last_hidden_state.shape)