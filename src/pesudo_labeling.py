import os
import librosa
import pandas as pd
from model import bnvgg13
from joblib import Parallel, delayed
from config import trainable_labels
from config import label_cards
from tqdm import tqdm
import numpy as np
import gc
import torch
gc.enable()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


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
        mel_spectrogram = librosa.feature.melspectrogram(
            y=sample,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=20,
            fmax=int(sr/2)
        )
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
    result = Parallel(n_jobs=NUM_WORKERS)(delayed(extract_feature)(row)for idx, row in tqdm(metadata.iterrows()))
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


NUM_WORKERS = 12
metadata = pd.read_csv('/media/ren/crucial/Projects/freesound_audio_tagging/data/train_noisy.csv')
metadata['filepath'] = metadata['fname'].map(lambda x: '/media/ren/crucial/Projects/freesound_audio_tagging/data/train_noisy/' + x)
feature_cache = cache_features(metadata)


predictions = np.zeros((len(metadata), 80))
for fold in tqdm(range(5)):
    model = bnvgg13((1, 128, 128))
    model_state_path = '/media/ren/crucial/Projects/freesound_audio_tagging/models/bnvgg13_fold_{}.tar'.format(fold)
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state["state_dict"])
    model.cuda()
    predictions += 1 / inference(model, metadata, feature_cache)

predictions = 1 / predictions
predictions = pd.DataFrame(predictions)
predictions.columns = trainable_labels
pesudo_labeled = pd.concat([metadata, predictions], axis=1)


pesudo_labeled['top1'] = pesudo_labeled[trainable_labels].idxmax(1)
label_cards = pd.Series(label_cards).map(np.round).astype(int)
results = []
for idx, row in tqdm(metadata.iterrows()):
    top1_label = row['top1']
    card = label_cards[top1_label]
    label_scores = row[trainable_labels].astype(float)
    signal_noisy_ratio = label_scores.nlargest(card).sum() / label_scores.sum()
    pesudo_label = ','.join(label_scores.nlargest(card).index)
    results.append({'signal_noisy_ratio': signal_noisy_ratio, 'plabels': pesudo_label})
results = pd.DataFrame(results)

pesudo_labeled = pd.concat([pesudo_labeled, results], axis=1)
pesudo_labeled.to_csv('../data/train_noisy_pesudo.csv', index=False)
