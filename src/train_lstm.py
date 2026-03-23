import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

df = pd.read_csv("data/clean_pitches.csv")

# Encode pitch type
le_pitch = LabelEncoder()
df["pitch_type_encoded"] = le_pitch.fit_transform(df["pitch_type"])

# Sort into proper at-bat order
df = df.sort_values(["game_date", "at_bat_number", "pitch_number"])

# Encode previous pitch
le_prev = LabelEncoder()
df["prev_pitch_encoded"] = le_prev.fit_transform(df["prev_pitch"])

# Features to use per pitch in sequence
sequence_features = ["prev_pitch_encoded", "balls", "strikes", "on_1b", "on_2b", "on_3b"]

# Build sequences — one per at-bat
MAX_SEQ_LEN = 10  # max pitches per at-bat

sequences = []
targets = []

for _, at_bat in df.groupby(["game_date", "at_bat_number", "pitcher_name"]):
    at_bat = at_bat.sort_values("pitch_number")
    
    for i in range(1, len(at_bat)):
        seq = at_bat[sequence_features].iloc[:i].values
        target = at_bat["pitch_type_encoded"].iloc[i]
        
        # Pad sequence to MAX_SEQ_LEN
        if len(seq) < MAX_SEQ_LEN:
            pad = np.zeros((MAX_SEQ_LEN - len(seq), len(sequence_features)))
            seq = np.vstack([pad, seq])
        else:
            seq = seq[-MAX_SEQ_LEN:]
        
        sequences.append(seq)
        targets.append(target)

X = np.array(sequences)
y = to_categorical(targets)

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    LSTM(64, input_shape=(MAX_SEQ_LEN, len(sequence_features)), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(len(le_pitch.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {round(accuracy * 100, 2)}%")

model.save("models/lstm_model.keras")
joblib.dump(le_pitch, "models/lstm_label_encoder.pkl")
joblib.dump(le_prev, "models/lstm_prev_encoder.pkl")

