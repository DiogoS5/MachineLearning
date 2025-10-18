import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SEQ_COL = "Skeleton_Sequence"
NK = 33


class SeqStatsFeaturizer(BaseEstimator, TransformerMixin):
    """
    From sequences shaped (T, NK, 2), compute:
      - per-KP mean (NK*2)
      - per-KP std  (NK*2)
      - per-KP avg speed (NK)  [mean ||Δ||]
      - per-KP total travel (NK) [sum ||Δ||]
    => total = 6*NK
    """
    def __init__(self, seq_col=SEQ_COL, n_keypoints=NK, n_dims=2):
        self.seq_col = seq_col
        self.n_keypoints = n_keypoints
        self.n_dims = n_dims

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SeqStatsFeaturizer expects a pandas DataFrame input.")
        n_samples = len(X)

        means = np.zeros((n_samples, self.n_keypoints, self.n_dims))
        stds = np.zeros((n_samples, self.n_keypoints, self.n_dims))
        avg_speeds = np.zeros((n_samples, self.n_keypoints))
        travels = np.zeros((n_samples, self.n_keypoints))

        for i, seq in enumerate(X[self.seq_col].values):
            arr = np.asarray(seq, dtype=float)
            if arr.size == 0:
                continue
            arr = arr.reshape(-1, self.n_keypoints, self.n_dims)

            means[i] = np.nanmean(arr, axis=0)
            stds[i]  = np.nanstd(arr, axis=0)

            diffs  = np.diff(arr, axis=0)
            if diffs.size > 0:
                speeds = np.linalg.norm(diffs, axis=2)
                avg_speeds[i] = np.nanmean(speeds, axis=0)
                travels[i]    = np.nansum(speeds, axis=0)

        feats = np.concatenate(
            [
                means.reshape(n_samples, -1),
                stds.reshape(n_samples, -1),
                avg_speeds,
                travels,
            ],
            axis=1,
        )
        return feats


class TemporalExtras(BaseEstimator, TransformerMixin):
    """
    Speed/acceleration distribution stats per keypoint:
    For each of speed and accel: mean, std, p25, p75  (each NK long) => 4*NK per set => total 8*NK.
    """
    def __init__(self, seq_col=SEQ_COL, n_keypoints=NK):
        self.seq_col = seq_col
        self.n_keypoints = n_keypoints

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for seq in X[self.seq_col].values:
            a = np.asarray(seq, float).reshape(-1, self.n_keypoints, 2)
            if a.shape[0] < 3:
                feats.append(np.zeros(8 * self.n_keypoints, dtype=float))
                continue

            v = np.diff(a, axis=0)       
            acc = np.diff(v, axis=0)   
            spd = np.linalg.norm(v, axis=2) 
            accn = np.linalg.norm(acc, axis=2)

            def stat(z):
                return np.concatenate([
                    np.nanmean(z, axis=0),
                    np.nanstd(z, axis=0),
                    np.nanpercentile(z, 25, axis=0),
                    np.nanpercentile(z, 75, axis=0),
                ], axis=None)

            feats.append(np.concatenate([stat(spd), stat(accn)], axis=0))

        return np.asarray(feats)


class PostureFeaturizer(BaseEstimator, TransformerMixin):
    """
    Adds:
      - ranges: per-KP motion range across time: ||max - min|| (NK)
      - right_vs_left: scalar = sum(std on right arm) - sum(std on left arm)
      - posture_angle: scalar = atan2(torso_vec.x, torso_vec.y)   # angle vs vertical (y)
    """
    LEFT_ARM  = [14,16,18,20,22]
    RIGHT_ARM = [13,15,17,19,21]
    SHOULDERS = [11,12]
    HIPS      = [23,24]

    def __init__(self, seq_col=SEQ_COL, n_keypoints=NK):
        self.seq_col = seq_col
        self.n_keypoints = n_keypoints

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples = len(X)
        ranges = np.zeros((n_samples, self.n_keypoints))
        right_vs_left = np.zeros((n_samples, 1))
        posture_angle = np.zeros((n_samples, 1))

        for i, seq in enumerate(X[self.seq_col].values):
            arr = np.asarray(seq, dtype=float)
            if arr.size == 0:
                continue
            a = arr.reshape(-1, self.n_keypoints, 2)

            a_max = np.nanmax(a, axis=0)
            a_min = np.nanmin(a, axis=0)
            ranges[i] = np.linalg.norm(a_max - a_min, axis=1)

            means = np.nanmean(a, axis=0)
            stds  = np.nanstd(a, axis=0)

            left_std  = np.linalg.norm(stds[self.LEFT_ARM], axis=1)
            right_std = np.linalg.norm(stds[self.RIGHT_ARM], axis=1)
            right_vs_left[i, 0] = np.nansum(right_std) - np.nansum(left_std)

            torso_vec = np.nanmean(means[self.HIPS], axis=0) - np.nanmean(means[self.SHOULDERS], axis=0)
            posture_angle[i, 0] = np.arctan2(torso_vec[0], torso_vec[1])

        return np.concatenate([ranges, right_vs_left, posture_angle], axis=1)
