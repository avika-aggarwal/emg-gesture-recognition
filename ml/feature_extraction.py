import numpy as np

def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract 4 time-domain EMG features from a single window of ADC samples.
    Window should be mean-centered before calling this.
    """
    rms = np.sqrt(np.mean(window ** 2))
    mav = np.mean(np.abs(window))
    zcr = float(np.sum(np.diff(np.sign(window)) != 0))
    wl  = np.sum(np.abs(np.diff(window)))
    return np.array([rms, mav, zcr, wl], dtype=np.float32)


def extract_dataset(raw_data: np.ndarray):
    """
    Extract features from a full labelled dataset.
    raw_data shape: (N, window_size + 1), last column is gesture label.
    """
    X = np.array([extract_features(row[:-1]) for row in raw_data])
    y = raw_data[:, -1].astype(int)
    return X, y
