# This script finetunes Wav2Vec 2.0 on phoneme recognition.
# The Pytorch Lightning code skeleton is adapted from: https://github.com/nttcslab/byol-a

import os
import fire
import math
import shutil

from utils import *
from learner import Learner
from data import DataCollator
from tokenizer import Tokenizer
from decoder import Speech_Decoder_Linear, Speaker_Decoder_Linear
from encoder import Speaker_Encoder, Speech_Encoder, Joint_Encoder

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets import Dataset, DatasetDict

torch.set_float32_matmul_precision('medium' or 'high')

def main(config_path='config.yaml', layer_num=None) -> None:
    
    # Load config file
    config = load_yaml_config(config_path)

    # Essentials
    logger = get_logger(__name__)
    logger.info(config)
    seed_everything(config.seed)

    # 1. Read data and split it

    #read csv data as dataframe
    df = pd.read_csv(config.data.data_path)

    #convert dataframes into huggingface dataset
    raw_datasets = DatasetDict()
    raw_datasets["train"] = Dataset.from_pandas(df.loc[df.split == "train"])
    raw_datasets["validation"] = Dataset.from_pandas(df.loc[df.split == "val"])
    raw_datasets["test"] = Dataset.from_pandas(df.loc[df.split == "test"])

    # 2. Define Tokenizer

    #define a tokenizer for the vocabulary
    tokenizer = Tokenizer(**config.text)
    
    # 3. Preprocess and tokenize data

    #define paths for cached files
    cache_file_names = None
    if config.data.cache_file_path is not None:
        cache_file_names = {"train": f"{config.data.cache_file_path}/train", 
                            "validation": f"{config.data.cache_file_path}/val",
                            "test": f"{config.data.cache_file_path}/test"}
    #load, resample and tokenize audio files
    remove_columns = ["wav_path", "split", "client_id", "Unnamed: 0", "sr", "gender", "total_file_duration_in_s", "__index_level_0__"]
    vectorized_datasets = raw_datasets.map(prepare_data,
                                        remove_columns=remove_columns,
                                        cache_file_names=cache_file_names,
                                        fn_kwargs={"target_sr": config.data.sampling_rate, "tokenizer": tokenizer})
    
    # 4. Define DataCollator and DataLoaders

    data_collator = DataCollator()
    train_dataloader = DataLoader(
            vectorized_datasets["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=config.dataloader.per_device_train_batch_size,
            num_workers=config.dataloader.num_workers,
        )
    eval_dataloader = DataLoader(
        vectorized_datasets["validation"], 
        collate_fn=data_collator, 
        batch_size=config.dataloader.per_device_eval_batch_size,
        num_workers=config.dataloader.num_workers,
    )
    
    # 5. Define Encoders (ECAPA and Whisper) and Joint model

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

    logger.info('Loading models and dataloaders is done!')

    # 7. Define training parameters and callbacks

    #calculate the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.trainer.gradient_accumulation_steps)
    total_batch_size = config.dataloader.per_device_train_batch_size * config.trainer.num_gpus * config.trainer.gradient_accumulation_steps

    #calculate number of training epochs
    num_train_epochs = math.ceil(config.trainer.max_train_steps / num_update_steps_per_epoch)

    #define name of the directory saving the checkpoints
    name = (f"saganet"
            f'_d-{config.saganet.d_model}'
            f'_atthead-{config.saganet.num_head}'
            f'_ffd-{config.saganet.dim_feedforward}'
            f'_num_layers-{config.saganet.num_layers}'
            f'_bs-{config.dataloader.per_device_train_batch_size}'
            f'_e-{num_train_epochs}'
            f'_lr-{config.optimization.learning_rate}'
            f'_rs-{config.seed}'
            )
    
   
    
    dir_name = f'{config.callbacks.checkpoint_folder}/{name}'
    os.makedirs(dir_name, exist_ok=True)
    shutil.copy(config_path, dir_name)

    #define callbacks and logger
    model_checkpoint = ModelCheckpoint(dirpath=dir_name,
                                       filename='best{epoch:02d}-val_loss{val_Loss:.2f}', 
                                       monitor='val_Loss',
                                       save_last=True,
                                       save_top_k=1, save_weights_only=False,
                                       auto_insert_metric_name=False)
    # lr_monitor = LearningRateMonitor(logging_interval='step')
    #wandb_logger = WandbLogger(save_dir=config.callbacks.checkpoint_folder, version=name, project=f"SAGANet")
    # early_stop_callback = EarlyStopping(monitor="val_PER", min_delta=0.005, patience=10, verbose=False, mode="min")
    callbacks = [model_checkpoint]
    # if config.callbacks.push_to_repo:
    #     local_dir = f'{dir_name}/huggingface_repo'
    #     os.makedirs(local_dir, exist_ok=True)
    #     hf_push_repo = HuggingFaceHubCallback(f'gelbanna/{name}', dir_name=dir_name, local_dir=local_dir, git_user='GasserElbanna')
    #     callbacks = [model_checkpoint, lr_monitor, hf_push_repo]

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(vectorized_datasets['train'])}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.dataloader.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.trainer.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.trainer.max_train_steps}")

    # 7. Define Trainer and Learner
    trainer = pl.Trainer(accelerator='gpu',
                        devices=config.trainer.num_gpus,
                        num_nodes=config.trainer.num_nodes,
                        strategy=DDPStrategy(find_unused_parameters=True),
                        max_epochs=num_train_epochs,
                        callbacks=callbacks,
                        accumulate_grad_batches=config.trainer.gradient_accumulation_steps,
                        #logger=wandb_logger,
                        logger=False,
                        gradient_clip_val=config.trainer.gradient_clip_val,
                        gradient_clip_algorithm='norm',
                        val_check_interval=config.trainer.check_val_steps,
                        fast_dev_run=False,
                        num_sanity_val_steps=0,
                        profiler="simple")
    
    #run training
    ckpt_path=None

    global_step_offset = None
    if config.trainer.resume or os.path.isfile(f'{dir_name}/last.ckpt'):
        ckpt_path = f'{dir_name}/last.ckpt'
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        global_step_offset = checkpoint["global_step"]
        logger.info(f'Global Step Restored:{global_step_offset}')
    
    learner = Learner(config=config, 
                    tokenizer=tokenizer,
                    speech_encoder=speech_encoder,
                    speaker_encoder=speaker_encoder,
                    joint_encoder=saganet,
                    speech_decoder=speech_decoder,
                    speaker_decoder = speaker_decoder,
                    global_step=global_step_offset)

    trainer.fit(learner, train_dataloader, eval_dataloader, ckpt_path=ckpt_path)

    logger.info(f'Training is finished.')
    
if __name__ == '__main__':
    fire.Fire(main(config_path='config.yaml'))