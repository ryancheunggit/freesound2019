from datetime import datetime
t0 = datetime.now()
print('-- started at time {}'.format(t0))
import gc
import os
import random
import torch
import librosa
import torch.nn.functional as F
from torch import nn
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from torchvision import models

NUM_WORKERS = os.cpu_count()
print(NUM_WORKERS)
trainable_labels = [
    'Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause',
    'Bark', 'Bass_drum', 'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bicycle_bell',
    'Burping_and_eructation', 'Bus', 'Buzz',
    'Car_passing_by', 'Cheering', 'Chewing_and_mastication', 'Child_speech_and_kid_speaking',
    'Chink_and_clink', 'Chirp_and_tweet', 'Church_bell', 'Clapping', 'Computer_keyboard',
    'Crackle', 'Cricket', 'Crowd', 'Cupboard_open_or_close', 'Cutlery_and_silverware',
    'Dishes_and_pots_and_pans', 'Drawer_open_or_close', 'Drip',
    'Electric_guitar',
    'Fart', 'Female_singing', 'Female_speech_and_woman_speaking', 'Fill_(with_liquid)',
    'Finger_snapping', 'Frying_(food)',
    'Gasp', 'Glockenspiel', 'Gong', 'Gurgling',
    'Harmonica', 'Hi-hat', 'Hiss',
    'Keys_jangling', 'Knock',
    'Male_singing', 'Male_speech_and_man_speaking', 'Marimba_and_xylophone', 'Mechanical_fan',
    'Meow', 'Microwave_oven', 'Motorcycle',
    'Printer', 'Purr', 'Race_car_and_auto_racing', 'Raindrop', 'Run',
    'Scissors', 'Screaming', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', 'Skateboard',
    'Slam', 'Sneeze', 'Squeak', 'Stream', 'Strum',
    'Tap', 'Tick-tock', 'Toilet_flush', 'Traffic_noise_and_roadway_noise', 'Trickle_and_dribble',
    'Walk_and_footsteps', 'Water_tap_and_faucet', 'Waves_and_surf', 'Whispering', 'Writing',
    'Yell',
    'Zipper_(clothing)'
]

def read_audio(filepath, target_sr=44100):
    try:
        sample, sr = librosa.load(path=filepath, sr=None)
        if sr != target_sr:
            sample = librosa.resample(y=sample, orig_sr=sr, target_sr=target_sr)
        sample = librosa.effects.trim(sample)[0]
    except:
        sample = np.ones((44100,))
    return sample

def extract_logmel_spectrogram(sample, sr=44100, n_fft=2560, hop_length=1024, n_mels=128, n_channels=1):
    try:
        mel_spectrogram = librosa.feature.melspectrogram(y=sample, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=20, fmax=int(sr/2))
        logmel_spectrogram = librosa.power_to_db(mel_spectrogram).T
        feature = np.expand_dims(logmel_spectrogram, 0)
    except:
        feature = np.zeros((n_channels, 128, n_mels))
    return feature

def extract_feature(row):
    filepath = row['filepath']
    sample = read_audio(filepath)
    feature = extract_logmel_spectrogram(sample)
    return (filepath, feature)

def cache_features(metadata):
    result = Parallel(n_jobs=NUM_WORKERS)(delayed(extract_feature)(row)for idx, row in metadata.iterrows())
    path_feature_mapping = {filepath: feature for (filepath, feature) in result}
    return path_feature_mapping

def inference(model, metadata, feature_cache, step_size=24, signal_k=1, top_k=20):
    model.eval()
    with torch.no_grad():
        model_predicts = []
        for idx, row in metadata.iterrows():
            filepath = row['filepath']
            x = feature_cache[filepath]
            length = x.shape[1]
            if length <= 128:
                left_pad = (128 - length) // 2
                right_pad = (128 - length) - left_pad
                sample_mean = x.mean()
                x = np.pad(array=x, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='constant', constant_values=(sample_mean, sample_mean))
                sample = torch.from_numpy(x).cuda().float().unsqueeze(0)
                final_predictions = model(sample).sigmoid().cpu().detach().numpy().flatten()
            else:
                results = []
                start = 0
                while start * step_size + 128 < length:
                    sample = torch.from_numpy(x[:, start * step_size: start * step_size + 128, :]).cuda().float().unsqueeze(0)
                    predictions = model(sample).sigmoid().cpu().detach().numpy().flatten()
                    preds = predictions.copy()
                    preds.sort()
                    signal = preds[-signal_k:].sum() / preds.sum()
                    results.append((predictions, signal))
                    start += 1
                results.sort(key=lambda x: x[1])
                final_predictions = [result[0] for result in results[-top_k:]]
                final_predictions = np.vstack(final_predictions).mean(0)
            model_predicts.append(final_predictions)
        model_predicts = np.vstack(model_predicts)
    return model_predicts


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_type='max', use_bn=True):
        super(ConvBlock, self).__init__()
        self.pool_type = pool_type
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weights()

    def _init_layer(self, layer, nonlinearity='leaky_relu'):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def _init_bn(self, bn):
        bn.bias.data.fill_(0.)
        bn.running_mean.data.fill_(0.)
        bn.weight.data.fill_(1.)
        bn.running_var.data.fill_(1.)

    def init_weights(self):
        self._init_layer(self.conv1)
        self._init_layer(self.conv2)
        self._init_bn(self.bn1)
        self._init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2)):
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(input)))
            x = F.relu(self.bn2(self.conv2(x)))
        else:
            x = F.relu(self.conv1(input))
            x = F.relu(self.conv2(x))
        if self.pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif self.pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class BNVgg8(nn.Module):
    def __init__(self, input_shape=(1, 128, 128)):
        super(BNVgg8, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=input_shape[0], out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.feature = MeanMaxPool()
        self.classifier = nn.Linear(in_features=512, out_features=80)

    def forward(self, x):
        x = self.conv_block1(x, pool_size=(2, 2))
        x = self.conv_block2(x, pool_size=(2, 2))
        x = self.conv_block3(x, pool_size=(2, 2))
        x = self.conv_block4(x, pool_size=(1, 1))
        x = self.feature(x)
        output = self.classifier(x)
        return output


class MeanMaxPool(nn.Module):
    def __init__(self):
        super(MeanMaxPool, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(None, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d(output_size=1)

    def forward(self, x):
        x = self.global_max_pool(self.global_avg_pool(x))
        x = x.view(x.shape[0], -1)
        return x

def densenet161(input_shape=(1, 128, 128)):
    model = models.densenet161(pretrained=False)
    model.features.conv0 = nn.Conv2d(
        in_channels=input_shape[0],
        out_channels=96,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False
    )
    model.classifier = nn.Linear(in_features=2208, out_features=80)
    return model

sub = pd.read_csv('../input/freesound-audio-tagging-2019/sample_submission.csv')
metadata = pd.read_csv('../input/freesound-audio-tagging-2019/sample_submission.csv')
metadata['filepath'] = metadata['fname'].map(lambda x: '../input/freesound-audio-tagging-2019/test/' + x)
print('-- csv loaded secs since start {}'.format((datetime.now() - t0).total_seconds()))

feature_cache = cache_features(metadata)
print('-- feature loaded to memory since start {}'.format((datetime.now() - t0).total_seconds()))

model = BNVgg8()
model.cuda()

print('-- model 1 start {}'.format((datetime.now() - t0).total_seconds()))
for fold in range(5):
    model_state_path = '../input/fsmodels/models/train_cv_v1_fold_{}_gpu_1.tar'.format(fold)
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state["state_dict"])
    predictions = inference(model, metadata, feature_cache)[:]
    if fold == 0:
        model_1_predictions = 1 / predictions
    else:
        model_1_predictions += 1 / predictions
    print('-- fold {} prediction done {}'.format(fold, (datetime.now() - t0).total_seconds()))
model_1_predictions = 1 / model_1_predictions

sub.iloc[:, 1:] = model_1_predictions
sub.to_csv('submission.csv', index=False)
print('-- submission updated {}'.format((datetime.now() - t0).total_seconds()))

print('-- model 2 start {}'.format((datetime.now() - t0).total_seconds()))
for fold in range(5):
    model_state_path = '../input/fsmodels/models/train_cv_v2_fold_{}_gpu_1.tar'.format(fold)
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state["state_dict"])
    predictions = inference(model, metadata, feature_cache)[:]
    if fold == 0:
        model_2_predictions = 1 / predictions
    else:
        model_2_predictions += 1 / predictions
    print('-- fold {} prediction done {}'.format(fold, (datetime.now() - t0).total_seconds()))
model_2_predictions = 1 / model_2_predictions

sub.iloc[:, 1:] += model_2_predictions
sub.to_csv('submission.csv', index=False)
print('-- submission updated {}'.format((datetime.now() - t0).total_seconds()))

print('-- model 3 start {}'.format((datetime.now() - t0).total_seconds()))
for fold in range(5):
    model_state_path = '../input/fsmodels/models/train_cv_v3_fold_{}_gpu_1.tar'.format(fold)
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state["state_dict"])
    predictions = inference(model, metadata, feature_cache)[:]
    if fold == 0:
        model_3_predictions = 1 / predictions
    else:
        model_3_predictions += 1 / predictions
    print('-- fold {} prediction done {}'.format(fold, (datetime.now() - t0).total_seconds()))
model_3_predictions = 1 / model_3_predictions

sub.iloc[:, 1:] += model_3_predictions
sub.to_csv('submission.csv', index=False)
print('-- submission updated {}'.format((datetime.now() - t0).total_seconds()))

print('-- model 4 start {}'.format((datetime.now() - t0).total_seconds()))
for fold in range(5):
    model_state_path = '../input/fsmodels/models/train_cv_v4_fold_{}_gpu_1.tar'.format(fold)
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state["state_dict"])
    predictions = inference(model, metadata, feature_cache)[:]
    if fold == 0:
        model_4_predictions = 1 / predictions
    else:
        model_4_predictions += 1 / predictions
    print('-- fold {} prediction done {}'.format(fold, (datetime.now() - t0).total_seconds()))
model_4_predictions = 1 / model_4_predictions

sub.iloc[:, 1:] += model_4_predictions
sub.to_csv('submission.csv', index=False)
print('-- submission updated {}'.format((datetime.now() - t0).total_seconds()))

del model
torch.cuda.empty_cache()
gc.collect()

model = densenet161()
model.cuda()

model_state_path = '../input/fsmodels/models/densenet161_noisy_0.tar'
model_state = torch.load(model_state_path)
model.load_state_dict(model_state["state_dict"])
model_5_predictions = inference(model, metadata, feature_cache, step_size=36, signal_k=2, top_k=18)[:]
print('-- densenet 1 prediction generated {}'.format((datetime.now() - t0).total_seconds()))

sub.iloc[:, 1:] += .15 * model_5_predictions
sub.to_csv('submission.csv', index=False)
print('-- submission updated {}'.format((datetime.now() - t0).total_seconds()))

model_state_path = '../input/fsmodels/models/dense161_no_noisy_0.tar'
model_state = torch.load(model_state_path)
model.load_state_dict(model_state["state_dict"])
model_6_predictions = inference(model, metadata, feature_cache, step_size=36, signal_k=2, top_k=18)[:]
print('-- densenet 2 prediction generated {}'.format((datetime.now() - t0).total_seconds()))

sub.iloc[:, 1:] += .15 * model_6_predictions
sub.to_csv('submission.csv', index=False)
print('-- submission updated {}'.format((datetime.now() - t0).total_seconds()))
