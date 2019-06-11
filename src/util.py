"""Utilities."""
import numpy as np
from operator import lt, gt


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t)/60
        hr = t//60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError


class EarlyStopping(object):
    """Monitoring an metric, flag when to stop training."""

    def __init__(self, mode='min', min_delta=0, percentage=False, patience=10, initial_bad=0, initial_best=np.nan):
        assert patience > 0, 'patience must be positive integer'
        assert mode in ['min', 'max'], 'mode must be either min or max'
        self.mode = mode
        self.patience = patience
        self.best = initial_best
        self.num_bad_epochs = initial_bad
        self.is_better = self._init_is_better(mode, min_delta, percentage)
        self._stop = False

    @property
    def stop(self):
        return self._stop

    def step(self, metric):
        if self.is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
        else:
            self.num_bad_epochs += 1

        if np.isnan(self.best) and (not np.isnan(metric)):
            self.num_bad_epochs = 0
            self.best = metric

        self._stop = self.num_bad_epochs >= self.patience

    def _init_is_better(self, mode, min_delta, percentage):
        comparator = lt if mode == 'min' else gt
        if not percentage:
            def _is_better(new, best):
                target = best - min_delta if mode == 'min' else best + min_delta
                return comparator(new, target)
        else:
            def _is_better(new, best):
                target = best * (1 - (min_delta / 100)) if mode == 'min' else best * (1 + (min_delta / 100))
                return comparator(new, target)
        return _is_better

    def __repr__(self):
        return '<EarlyStopping object with: mode - {} - num_bad_epochs - {} - patience - {} - best - {}>'.format(
            self.mode, self.num_bad_epochs, self.patience, self.best)
