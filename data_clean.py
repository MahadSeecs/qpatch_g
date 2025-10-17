import librosa
import os
import numpy as np

base_path = 'ravdess'  # root folder containing actor folders
X, y = [], []

sr = 22050
n_mels = 128
max_duration = 5  # seconds
max_len = sr * max_duration // 512 + 1  # approximate number of frames with default hop_length=512

# emotion mapping if needed (RAVDESS has 8 emotions)
emotion_map = {
    '01': 0,  # neutral
    '02': 1,  # calm
    '03': 2,  # happy
    '04': 3,  # sad
    '05': 4,  # angry
    '06': 5,  # fearful
    '07': 6,  # disgust
    '08': 7   # surprised
}

for actor_folder in sorted(os.listdir(base_path)):
    folder_path = os.path.join(base_path, actor_folder)
    if not os.path.isdir(folder_path):
        continue

    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.wav'):
            path = os.path.join(folder_path, file)
            # load audio
            y_, _ = librosa.load(path, sr=sr, duration=max_duration)
            
            # extract mel
            mel = librosa.feature.melspectrogram(y=y_, sr=sr, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max).T  # shape: (time, n_mels)
            
            # pad or truncate to max_len
            if mel_db.shape[0] < max_len:
                pad_width = max_len - mel_db.shape[0]
                mel_db = np.pad(mel_db, ((0, pad_width), (0, 0)), mode='constant')
            else:
                mel_db = mel_db[:max_len, :]
            
            X.append(mel_db)

            # extract emotion from filename: e.g., "03-01-06-01-02-01-12.wav"
            emotion_code = file.split('-')[2]
            y.append(emotion_map[emotion_code])

# convert to numpy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("Final shapes:", X.shape, y.shape)

# save
np.save('ravdess_mel_features.npy', X)
np.save('ravdess_labels.npy', y)
