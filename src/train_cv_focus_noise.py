"""Training the network."""
import argparse
import os
import torch
import librosa
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
from model import BNVgg8
from joblib import Parallel, delayed
from tqdm import tqdm
from functools import partial
from torch.utils.data import Dataset, DataLoader
from metric import LwlrapAccumulator
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util import EarlyStopping
from config import trainable_labels, Path, Setting
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


parser = argparse.ArgumentParser(description='train model')
parser.add_argument('--gpu', type=str, default='1', help='Which gpu to use')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
cudnn.enabled = True

NUM_WORKERS = 12
MAX_ITERATIONS = 200


def split_labels(row, label_column='labels', cols_to_keep=('fname')):
    new_row = pd.Series({label: 1 for label in row[label_column].split(',')}).astype(int)
    for col in cols_to_keep:
        new_row[col] = row[col]
    return new_row


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
        mel_spectrogram = librosa.feature.melspectrogram(y=sample, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                         n_mels=n_mels, fmin=20, fmax=int(sr/2))
        logmel_spectrogram = librosa.power_to_db(mel_spectrogram).T
        feature = np.expand_dims(logmel_spectrogram, 0)
    except:
        feature = np.zeros((n_channels, 128, n_mels))
    return feature


def extract_feature(row):
    filepath = row['filepath']
    sample = read_audio(filepath)
    if 'start' in row.index:
        start = int(row['start'])
        end = min(int(row['end']), start + 8828)
        sample = sample[int(start * 16): int(end * 16)]
    feature = extract_logmel_spectrogram(sample)
    return (filepath, feature)


def cache_features(metadata):
    result = Parallel(n_jobs=NUM_WORKERS)(delayed(extract_feature)(row)for idx, row in tqdm(metadata.iterrows()))
    path_feature_mapping = {filepath: feature for (filepath, feature) in result}
    return path_feature_mapping


def agument(spec, num_masks=2, t_mask=0.1, f_mask=0.1):
    spec = spec.copy()
    c, t, f = spec.shape
    f_mean, t_mean = spec.mean(2), spec.mean(1)
    for i in range(num_masks):
        f_width = int(round(np.random.uniform(0.0, f_mask) * f))
        start = int(np.random.uniform(low=0.0, high=f - f_width))
        spec[:, :, start: start + f_width] = np.expand_dims(f_mean, 2)

        t_width = int(round(np.random.uniform(0.0, t_mask) * t))
        start = int(np.random.uniform(low=0.0, high=t - t_width))
        spec[:, start: start + t_width, :] = np.expand_dims(t_mean, 1)
    return spec


class FreesoundDataset(Dataset):
    def __init__(self, metadata, task, cache=None, flip=False, mask=False):
        self.metadata = metadata
        self.task = task
        self.cache = cache
        self.filepath = np.array(metadata['filepath'])
        self.cache = cache
        if task != 'test':
            self.labels = metadata[trainable_labels].values
        else:
            self.labels = np.zeros((metadata.shape[0], len(trainable_labels)))
        self._n = metadata.shape[0]
        self.flip = flip
        self.mask = mask

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        feature = self.cache[self.filepath[idx]]
        if self.flip and np.random.random() >= .5:
            feature = np.flip(feature, 1).copy()
        target = self.labels[idx, :]
        n_time_step = feature.shape[1]
        if n_time_step < 128:
            left_pad = (128 - n_time_step) // 2
            right_pad = (128 - n_time_step) - left_pad
            sample_mean = feature.mean()
            feature = np.pad(array=feature, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='constant',
                             constant_values=(sample_mean, sample_mean))
        else:
            start = np.random.randint(low=0, high=n_time_step - 128 + 1)
            feature = feature[:, start: start + 128, :]
        if self.mask:
            feature = agument(feature)
        return torch.from_numpy(feature), target


def mixup_data(x1, y1, x2, y2, alpha):
    lambd = np.random.beta(alpha, alpha)
    lambd = max(lambd, 1-lambd)
    permutation = torch.randperm(x2.shape[0]).cuda()
    x_mixed = lambd * x1 + (1 - lambd) * x2[permutation, :]
    y_mixed = lambd * y1 + (1 - lambd) * y2[permutation]
    return x_mixed, y_mixed


def validate_random(model, valid_dl, n_tta=5):
    model.eval()
    with torch.no_grad():
        average_lwlrap = []
        for tta in range(n_tta):
            y_pred = []
            y_true = []
            lwlrap_scorer = LwlrapAccumulator()
            for idx, (x_batch, y_batch) in enumerate(valid_dl):
                x_batch, y_batch = x_batch.float().cuda(), y_batch.float().cuda()
                out = model(x_batch)
                y_true.append(y_batch.cpu().detach().numpy())
                y_pred.append(out.sigmoid().cpu().detach().numpy())
            y_true = np.vstack(y_true)
            y_pred = np.vstack(y_pred)
            lwlrap_scorer.accumulate_samples(y_true, y_pred)
            average_lwlrap.append(lwlrap_scorer.overall_lwlrap())
    return np.mean(average_lwlrap)


def validate_search(model, valid_meta, valid_cache, step_size=24, signal_k=1, top_k=20):
    model.eval()
    with torch.no_grad():
        true_labels = []
        model_predicts = []
        for idx, row in valid_meta.iterrows():
            filepath = row['filepath']
            y = np.array(row[trainable_labels].tolist())
            true_labels.append(y)
            x = valid_cache[filepath]
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
        true_labels = np.vstack(true_labels)
        model_predicts = np.vstack(model_predicts)

        scorer = LwlrapAccumulator()
        scorer.accumulate_samples(true_labels, model_predicts)
        score = scorer.overall_lwlrap()
    return score, model_predicts


focus_meta = pd.read_csv(Path.FOCUS_PESUDO_META)
focus_meta = focus_meta.apply(partial(split_labels, label_column='labels', cols_to_keep=['filepath', 'start', 'end']),
                              axis=1)
focus_meta.fillna(0, inplace=True)
focus_cache = cache_features(focus_meta)

train_meta = pd.read_csv(Path.TRAIN_CURATED_META)
train_meta = train_meta.loc[~train_meta['fname'].isin([
    'f76181c4.wav', '77b925c2.wav', '6a1f682a.wav', 'c7db12aa.wav', '7752cc8a.wav', '1d44b0bd.wav',
    '30965e0c.wav', 'bf80b29a.wav', '6340b6d0.wav', '6f77b5ae.wav', '686560d8.wav', 'b65311b2.wav',
    '99bd8571.wav', '56c78c5d.wav', 'cb0a2290.wav', 'a4bff6ea.wav', 'bf5f6f91.wav', 'a3001011.wav',
    '9378307a.wav', '7f409e1a.wav', '1588dfe0.wav', 'ed3ed0e0.wav', '63239644.wav', '6a30e390.wav',
    '95731027.wav', 'bc00bedc.wav'])
]
train_meta['filepath'] = train_meta['fname'].map(lambda x: os.path.join(Path.TRAIN_CURATED_AUDIO_DIR, x))
train_meta = train_meta.apply(partial(split_labels, label_column='labels', cols_to_keep=['filepath']), axis=1)
train_meta.fillna(0, inplace=True)
train_cache = cache_features(train_meta)


noisy_meta = pd.read_csv(Path.NOISY_PESUDO_META)
noisy_meta = noisy_meta.loc[noisy_meta['signal_noisy_ratio'] > .75]
noisy_meta['filepath'] = noisy_meta['fname'].map(lambda x: os.path.join(Path.TRAIN_NOISY_AUDIO_DIR, x))
noisy_meta = noisy_meta.apply(partial(split_labels, label_column='plabels', cols_to_keep=['filepath']), axis=1)
for label in trainable_labels:
    if label not in noisy_meta.columns:
        noisy_meta[label] = 0
noisy_meta.fillna(0, inplace=True)
noisy_cache = cache_features(noisy_meta)


mskf = MultilabelStratifiedKFold(n_splits=Setting.CV_FOLDS, shuffle=True, random_state=Setting.CV_SEED)
focus_splits = list(mskf.split(X=focus_meta, y=focus_meta[trainable_labels]))
train_splits = list(mskf.split(X=train_meta, y=train_meta[trainable_labels]))
train_splits = list(mskf.split(X=train_meta, y=train_meta[trainable_labels]))
noisy_splits = list(mskf.split(X=noisy_meta, y=noisy_meta[trainable_labels]))


sub = pd.read_csv(Path.SUB)

batch_size = 32
alpha = .5
w1 = .3 * np.ones(MAX_ITERATIONS)
w2 = .7 * np.ones(MAX_ITERATIONS)
w1[:30] = np.linspace(.7, .3, 30)
w2[:30] = np.linspace(.3, .7, 30)
w3 = .2

for fold in range(Setting.CV_FOLDS):
    if fold > 1:
        break
    focus_ds = FreesoundDataset(focus_meta.iloc[focus_splits[fold][0]], 'train', focus_cache, flip=True, mask=True)
    train_ds = FreesoundDataset(train_meta.iloc[train_splits[fold][0]], 'train', train_cache, flip=True, mask=True)
    pesudo_ds = FreesoundDataset(noisy_meta.iloc[noisy_splits[fold][1]], 'train', noisy_cache, flip=False, mask=False)
    valid_ds = FreesoundDataset(train_meta.iloc[train_splits[fold][1]], 'train', train_cache, flip=False, mask=False)

    focus_dl = DataLoader(dataset=focus_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    pesudo_dl = DataLoader(dataset=pesudo_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    model = BNVgg8((1, 128, 128))
    model.cuda()
    criterion = BCEWithLogitsLoss().cuda()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=7, verbose=True, cooldown=3, min_lr=1e-7)
    earlystopper = EarlyStopping(mode='max', min_delta=0.0001, patience=15, percentage=False)
    best_score = 0
    model_state_path = os.path.join(Path.MODEL_DIR, 'train_focus_noisy_fold_{}_gpu_{}.tar'.format(fold, args.gpu))

    for epoch in range(MAX_ITERATIONS):
        model.train()
        for idx, ((x1, y1), (x2, y2)) in enumerate(zip(focus_dl, train_dl, pesudo_dl)):
            optimizer.zero_grad()
            x1, y1 = x1.float().cuda(), y1.float().cuda()
            x2, y2 = x2.float().cuda(), y2.float().cuda()
            x3, y3 = x3.float().cuda(), y3.float().cuda()
            # x1, y1 = mixup_data(x1, y1, x1, y1, alpha=alpha)
            x2, y2 = mixup_data(x2, y2, x2, y2, alpha=alpha)
            x3, y3 = mixup_data(x3, y3, x3, y3, alpha=alpha)
            out3 = model(x3)
            out2 = model(x2)
            out1 = model(x1)
            loss3 = criterion(out3, y3)
            loss2 = criterion(out2, y2)
            loss1 = criterion(out1, y1)
            loss = w1[epoch] * loss1 + w2[epoch] * loss2 + w3 * loss3
            loss.backward()
            optimizer.step()

        score = validate_random(model, valid_dl, n_tta=10)
        if score > best_score:
            best_score = score
            model_state = {
                "model_name": 'freesound',
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "state_dict": model.state_dict()
            }
            torch.save(model_state, model_state_path)
        scheduler.step(score)
        print(epoch, score)
        earlystopper.step(score)
        if earlystopper.stop:
            break
    print(best_score)

    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state["state_dict"])
    score, predictions = validate_search(model, train_meta.iloc[train_splits[fold][1]], train_cache)
    print('score of loaded model: {}'.format(score))
    sub.iloc[:, 1:] = predictions
    sub.to_csv('train_focus_noisy_fold_{}_gpu_{}.csv'.format(fold, args.gpu), index=False)
    if fold == 0:
        full_predictions = 1 / predictions
    else:
        full_predictions += 1 / predictions

sub.iloc[:, 1:] = 1 / full_predictions
sub.to_csv(os.path.join(Path.SUB_DIR, 'train_focus_noisy_avg_gpu_{}.csv'.format(args.gpu)), index=False)
