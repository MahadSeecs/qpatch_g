import librosa, os, numpy as np

base_path = 'savee'
sr = 22050
n_mels = 128
max_duration = 5
max_len = sr * max_duration // 512 + 1

X, y = [], []

emotion_map = {
    'a':0, 'd':1, 'f':2, 'h':3, 'n':4, 'sa':5, 'su':6
}

for filename in os.listdir(base_path):
    if filename.endswith(".wav"):
        label = [k for k in emotion_map if k in filename][0]
        filepath = os.path.join(base_path, filename)
        signal, _ = librosa.load(filepath, sr=sr)
        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
        mel = librosa.power_to_db(mel)
        mel = mel.T
        if mel.shape[0] < max_len:
            pad_width = max_len - mel.shape[0]
            mel = np.pad(mel, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mel = mel[:max_len, :]
        X.append(mel)
        y.append(emotion_map[label])

np.save("savee_mel_features.npy", np.array(X))
np.save("savee_labels.npy", np.array(y))
