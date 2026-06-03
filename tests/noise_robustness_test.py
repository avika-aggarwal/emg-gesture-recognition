import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def add_gaussian_noise(X: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(X ** 2, axis=0)
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise        = np.random.randn(*X.shape) * np.sqrt(noise_power)
    return X + noise

def run_noise_sweep(clf, X_test, y_test, snr_range=None):
    if snr_range is None:
        snr_range = [5, 10, 15, 20, 25, 30, float('inf')]

    results = {}
    print(f"{'SNR (dB)':<12} {'Accuracy':>10}")
    print("-" * 24)

    for snr in snr_range:
        X_noisy = X_test.copy() if snr == float('inf') else add_gaussian_noise(X_test, snr)
        acc     = accuracy_score(y_test, clf.predict(X_noisy))
        results[snr] = acc
        label   = "clean" if snr == float('inf') else str(snr)
        print(f"{label:<12} {acc*100:>9.1f}%")

    return results

def plot_noise_sweep(results, output_path=None):
    snrs = [s for s in results if s != float('inf')]
    accs = [results[s] * 100 for s in snrs]

    plt.figure(figsize=(8, 4))
    plt.plot(snrs, accs, marker='o', linewidth=2)
    plt.axhline(y=results[float('inf')] * 100, linestyle='--',
                color='gray', label='Clean baseline')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Classifier Accuracy vs. Noise Level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()
