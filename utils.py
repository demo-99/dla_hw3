import io

import matplotlib.pyplot as plt


def plot_spectrogram_to_buf(reconstructed_wav, name=None):
    plt.figure(figsize=(20, 5))
    plt.plot(reconstructed_wav, alpha=.5)
    plt.title(name)
    plt.grid()
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf
