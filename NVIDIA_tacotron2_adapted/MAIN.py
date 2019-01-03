######################################################################################################
# The main script where the data preparation, training and evaluation happens.
######################################################################################################

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hyper_parameters import tacotron_params
from data_preparation import DataPreparation, DataCollate
from training import train

# ---------------------------------------- DEFINING INPUT ARGUMENTS ------------------------------------------------- #

training_files = 'filelists/ljs_audio_text_train_filelist.txt'
validation_files = 'filelists/ljs_audio_text_val_filelist.txt'

output_directory = '/homedtic/apeiro/tacotron2/output_directory'
log_directory = '/homedtic/apeiro/tacotron2/log_directory'
checkpoint_path = None
warm_start = False
n_gpus = 1  # NEED TO ASK TO TIC DEPARTMENT ABOUT CLUSTER
rank = 0

torch.backends.cudnn.enabled = tacotron_params['cudnn_enabled']
torch.backends.cudnn.benchmark = tacotron_params['cudnn_benchmark']

print("FP16 Run:", tacotron_params['fp16_run'])
print("Dynamic Loss Scaling:", tacotron_params['dynamic_loss_scaling'])
print("Distributed Run:", tacotron_params['distributed_run'])
print("CUDNN Enabled:", tacotron_params['cudnn_enabled'])
print("CUDNN Benchmark:", tacotron_params['cudnn_benchmark'])

# --------------------------------------------- PREPARING DATA ------------------------------------------------------ #

# Read the training files
with open(training_files, encoding='utf-8') as f:
    training_audiopaths_and_text = [line.strip().split("|") for line in f]
# if tacotron_params['sort_by_length']:
#    training_audiopaths_and_text.sort(key=lambda x: len(x[1]))

# Read the validation files
with open(validation_files, encoding='utf-8') as f:
    validation_audiopaths_and_text = [line.strip().split("|") for line in f]
# if tacotron_params['sort_by_length']:
#    validation_audiopaths_and_text.sort(key=lambda x: len(x[1]))

# prepare the data
train_data = DataPreparation(training_audiopaths_and_text, tacotron_params)
validation_data = DataPreparation(validation_audiopaths_and_text, tacotron_params)
collate_fn = DataCollate(tacotron_params['number_frames_step'])

# DataLoader prepares a loader for a set of data including a function that processes every
# batch as we wish (collate_fn). This creates an object with which we can list the batches created.
# DataLoader and Dataset (IMPORTANT FOR FURTHER DESIGNS WITH OTHER DATABASES)
# https://jdhao.github.io/2017/10/23/pytorch-load-data-and-make-batch/

train_sampler = DistributedSampler(train_data) if tacotron_params['distributed_run'] else None
val_sampler = DistributedSampler(valset) if tacotron_params['distributed_run'] else None

train_loader = DataLoader(train_data, num_workers=1, shuffle=False, sampler=train_sampler,
                          batch_size=tacotron_params['batch_size'], pin_memory=False, drop_last=True,
                          collate_fn=collate_fn)

validate_loader = DataLoader(validation_data, num_workers=1, shuffle=False, sampler=val_sampler,
                          batch_size=tacotron_params['batch_size'], pin_memory=False, drop_last=True,
                          collate_fn=collate_fn)

# ------------------------------------------------- TRAIN ----------------------------------------------------------- #

train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus, rank, hyper_params=tacotron_params,
      train_loader=train_loader, validate_loader=validate_loader)
