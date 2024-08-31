"""Training Learner definitions
"""
import numpy as np
from utils import cer
from collections import defaultdict

import torch
from torch import nn
import lightning.pytorch as pl
from torch.nn.utils.rnn import pad_sequence


class Learner(pl.LightningModule):
    def __init__(self, 
                config, 
                tokenizer,
                joint_encoder,
                speech_encoder,
                speech_decoder,
                global_step=None,
                ckpt_dir=None):
        
            super().__init__()
            self.config = config
            self.tokenizer = tokenizer
            
            self.joint_encoder = joint_encoder

            self.speech_encoder = speech_encoder
            for param in self.speech_encoder.parameters():
                param.requires_grad = False

            self.speech_decoder = speech_decoder

            self.ctc_objective = nn.CTCLoss(
                                blank=self.tokenizer.pad_idx,
                                zero_infinity=True,
                                )
            self.metric_speech = cer

            self.val_logs = {}
            self.completed_steps = 0
            if global_step is not None:
                self.completed_steps = global_step
            self.ckpt_dir = ckpt_dir
    
    def _shared_eval_step(self, batch, step=None):
        #extract encoder features
        features_speech = self.speech_encoder(batch['input_values'])

        features_speech = self.joint_encoder(features_speech)
        
        #get the length of the features and labels
        features_len = torch.IntTensor([feat.shape[0] for feat in features_speech])
        labels_speech_len = torch.IntTensor([len(label) for label in batch['speech_labels']])
        
        #print(features_len, labels_speech_len)

        #pad both features and labels
        features_speech = pad_sequence(features_speech, batch_first=True)
        speech_labels = pad_sequence(
                            batch['speech_labels'],
                            batch_first=True,
                            padding_value=self.tokenizer.pad_idx,
                            )

        #get the word and speaker logits
        logits_speech = self.speech_decoder(features_speech)
        
        #compute the prob for each class
        log_probs_speech = nn.functional.log_softmax(logits_speech, dim=-1)

        #compute CTC loss
        loss = self.ctc_objective(
            log_probs_speech.transpose(0, 1),  # (N, T, C) -> (T, N, C)
            speech_labels,
            features_len,
            labels_speech_len,
        )

        pred_tokens = log_probs_speech.argmax(dim=-1)
        filtered_tokens = []
        for pred_token in pred_tokens:
            pred_token = pred_token.unique_consecutive()
            filtered_token = [
                token
                for token in pred_token.tolist()
                if token != self.tokenizer.pad_idx and token != self.tokenizer.eos_idx
            ]
            filtered_tokens.append(filtered_token)
        hypothesis = [
            self.tokenizer.decode(h) for h in filtered_tokens
        ]
        
        groundtruth = [self.tokenizer.decode(g.tolist()) for g in batch['speech_labels']]
        
        if step == 'test':
            per_values = [self.metric([hyp], [gt]) for hyp, gt in zip(hypothesis, groundtruth)]
            text = batch['text']
            return loss, per_values, hypothesis, groundtruth, text

        cer_value = self.metric_speech(hypothesis, groundtruth)
        return loss, cer_value*100

    def training_step(self, batch):
        
        loss, cer_value = self._shared_eval_step(batch)
        
        self.log_dict({
                    "Total loss": loss,
                    "CER": cer_value,
                }, on_step=True, on_epoch=False, sync_dist=True)
        return loss
    
    def on_train_batch_end(self, out, batch, batch_idx):
         if batch_idx % self.config.trainer.gradient_accumulation_steps == 0:
            self.log("current_step", self.completed_steps, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.completed_steps += 1
    
    def on_validation_epoch_start(self):
         self.val_logs = {
            "val_CER": 0,
            "val_Loss": 0,
        }

    def validation_step(self, batch, _):
        
        loss, cer_value = self._shared_eval_step(batch)

        self.val_logs["val_Loss"] = loss
        self.val_logs["val_CER"] = cer_value

        self.val_logs = {k: v for k, v in self.val_logs.items()}
        for k, v in self.val_logs.items():
            self.log(k, float(v), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def on_test_start(self):
        self.predictions = defaultdict(list)

    def test_step(self, batch, _):
        loss, per_values, hypothesis, groundtruth, text = self._shared_eval_step(batch, step='test')

        self.predictions['text'].extend(text)
        self.predictions['groundtruth'].extend(groundtruth)
        self.predictions['hypothesis'].extend(hypothesis)
        self.predictions['per'].extend(per_values)
        
        self.log('test_loss', float(loss), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_per', float(np.mean(per_values)), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    
    def on_test_end(self):
        torch.save(self.predictions, f'{self.ckpt_dir}/predictions.pt')

    def on_predict_start(self):
        self.predictions = defaultdict(list)

    def predict_step(self, batch, _):
        self.predictions['hypothesis'].extend(self(batch)) 
    
    def forward(self, x):
        #extract encoder features
        features_speech = self.speech_encoder(x['input_values'])

        #get the word and speaker logits
        logits_speech = self.speech_decoder(features_speech)                  
        
        #compute the prob for each class
        log_probs_speech = nn.functional.log_softmax(logits_speech, dim=-1)

        pred_tokens = log_probs_speech.argmax(dim=-1)
        filtered_tokens = []
        for pred_token in pred_tokens:
            pred_token = pred_token.unique_consecutive()
            filtered_token = [
                token
                for token in pred_token.tolist()
                if token != self.tokenizer.pad_idx and token != self.tokenizer.eos_idx
            ]
            filtered_tokens.append(filtered_token)
        hypothesis = [
            self.tokenizer.decode(h) for h in filtered_tokens
        ]
        return features_speech, hypothesis

        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
             self.parameters(),
             lr=self.config.optimization.learning_rate,
             betas=[self.config.optimization.adam_beta1, self.config.optimization.adam_beta2],
             eps=float(self.config.optimization.adam_epsilon),
        )
        
        def _lr_lambda(_):
            # if self.completed_steps < self.config.optimization.num_warmup_steps:
            #     return float(self.completed_steps) / float(max(1, self.config.optimization.num_warmup_steps))
            # num_steps = self.config.max_train_steps/self.config.num_gpus
            # return max(0.0, float(num_steps - self.completed_steps) / float(max(1, num_steps - self.config.num_warmup_steps)))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }