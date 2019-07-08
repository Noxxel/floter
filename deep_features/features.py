import librosa
import librosa.display as dsp
import numpy as np
import matplotlib.pyplot as plt

filename = "/home/flo/Lectures/floter/deep_features/relish_it.mp3"
duration = 30
n_fft = 2**11         # shortest human-disting. sound (music)
hop_length = 2**9    # => 75% overlap of frames
n_mels = 128

y, sr = librosa.load(filename, mono=True, duration=duration)

print("Sample rate:", sr)
print("Signal:", y.shape)
ticks = []
for i in range(duration+1):
    ticks.append(sr*i)

plt.figure(figsize=(24, 6))
plt.plot(list(range(y.shape[0])), y)
plt.xticks(ticks=ticks, labels=list(range(duration+1)))
#plt.show()
plt.tight_layout()
plt.savefig("signal.png")
plt.clf()

fft = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
print(fft.shape)
dsp.specshow(librosa.amplitude_to_db(np.abs(fft), ref=np.max), y_axis="log", x_axis="time", sr=sr)
plt.title("Power spectrogram")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
#plt.show()
plt.savefig("spectogram.png")
plt.clf()

mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels) #window of about 10ms 
print(mel.shape)
dsp.specshow(librosa.power_to_db(mel, ref=np.max), y_axis="mel", x_axis="time", sr=sr)
plt.title("Mel spectrogram")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
#plt.show()
plt.savefig("mel_spectogram.png")
plt.clf()
