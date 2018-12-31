import torch
import librosa.display
import numpy as np
import torch.utils.data
from scipy.io.wavfile import read
from text import text_to_sequence
from librosa.filters import mel as librosa_mel_fn
from stft import STFT
import matplotlib.pyplot as plt


# Load text and audio paths
filename = '/home/alex/PyTorch_TACOTRON_2/pycharm-tacotron2/filelists/ljs_audio_text_' + which_data + '_filelist.txt'

with open(filename, encoding='utf-8') as f:
    audiopaths_and_text = [line.strip().split("|") for line in f]

# clean the text and convert it into a sequence of characters
text_cleaners = ['english_cleaners']
audiopath, plane_text = audiopaths_and_text[0][0], audiopaths_and_text[0][1]

"""print(plane_text)
print(audiopath)"""

# puts an integer number for each character, space and punctuation mark
sample_seq = text_to_sequence(plane_text, text_cleaners)

# same character length + 1 due to the End-of-Sentence tag appended at the end
text_norm = torch.IntTensor(sample_seq)


# read audio file from path
sr, data = read(audiopath)
assert sr == sampling_rate, "{} sampling rate doesn't match {} on path {}".format(sr, sampling_rate, audiopath)
# why float32?? I guess is for decimal precision, as it is normalized
audio_read = torch.FloatTensor(data.astype(np.float32))

# convert into mel-scale spectrogram
audio_norm = audio_read / max_wav_value
# adds a dimension (1 x len(audio_norm)) for a better format before the training)
audio_norm = audio_norm.unsqueeze(0)
# This will be the input variable which is not trainable, then does not require gradient computation
audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)

## MEL-SCALE SPECTROGRAM TRANSFORMATION ##

"""A class adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
stft_fn = STFT(filter_length, hop_length, win_length)
# an np.ndarray with shape (n_mels, 1 + nfft/2). Is the mel transform matrix
mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)

# convert the mel transform matrix into a torch float tensor
mel_basis = torch.from_numpy(mel_basis).float()

assert(torch.min(audio_norm.data) >= -1)
assert(torch.max(audio_norm.data) <= 1)
# transforms the audio sample into spectrogram representation (magnitude and phase)
magnitudes, phases = stft_fn.transform(audio_norm)
# because magnitudes is a Variable (a wrapper)
magnitudes = magnitudes.data
# a matrix-by-matrix multiplication
mel_output = torch.matmul(mel_basis, magnitudes)
# dynamic range compression after mel-scale representation
mel_output_compressed = torch.log(torch.clamp(mel_output, min=clip_val) * C)

######################################################################################################
# How to pass from a torch tensor to a numpy array and plot the computed spectrogram
# Check previously the type of the object before plotting. Also the size; for this representation the
# tensor must be 2-D
mel_output_np = mel_output_compressed.data.cpu().numpy()
mel_output_np = np.squeeze(mel_output_np)
plt.figure()
librosa.display.specshow(mel_output_np)
plt.ylabel('Mel filter')
plt.title('Mel filter bank')
plt.colorbar()
plt.tight_layout()
plt.show()
######################################################################################################

# Process all the training, validation and testing data and store it into files to load them afterwards
# files to check: stft, audio_processing, layers, model, utils, train, data_utils and hparams.py
# 18/10/2018



"""text, audio_data = train_data.__getitem__(0)"""

"""audio_data_np = audio_data.data.cpu().numpy()
audio_data_np = np.squeeze(audio_data_np)
plt.figure()
librosa.display.specshow(audio_data_np)
plt.ylabel('Mel filter')
plt.title('Mel filter bank')
plt.colorbar()
plt.tight_layout()
plt.show()"""

tacotron_model = tacotron_2(tacotron_params)

# print(type(train_loader))

# encoder_part = Encoder(tacotron_params)

embedding = torch.nn.Embedding(tacotron_params['n_symbols'], tacotron_params['symbols_embedding_length'])

for i, batch in enumerate(train_loader):
    text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
    # mel_padded = mel_padded.contiguous().cuda(async=True)
    # mel_padded = torch.autograd.Variable(mel_padded).float()
    # print(text_padded.shape)
    # embedded_inputs = embedding(text_padded)
    # print(mel_padded.shape)
    # embedded_inputs_transposed = embedded_inputs.transpose(1, 2)
    # output = encoder_part.forward(embedded_inputs_transposed, input_lengths)
    time_start = time.clock()
    outputs = tacotron_model.forward(batch)
    time_elapsed = (time.clock() - time_start)
    print(time_elapsed)
    break




