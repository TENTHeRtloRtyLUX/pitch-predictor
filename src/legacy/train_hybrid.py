import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model

df = pd.read_csv("data/clean_pitches.csv")

le_prev = joblib.load("models/lstm_prev_encoder.pkl")
le_pitch = joblib.load("models/lstm_label_encoder.pkl")

lstm_model = load_model("models/lstm_model.keras")

df["prev_pitch_encoded"] = le_prev.transform(df["prev_pitch"])

df = df.sort_values(["game_date", "at_bat_number", "pitch_number"])

MAX_SEQ_LEN = 10
sequence_features = ["prev_pitch_encoded", "balls", "strikes", "on_1b", "on_2b", "on_3b"]

sequences = []
indices = []

for _, at_bat in df.groupby(["game_date", "at_bat_number", "pitcher_name"]):
    at_bat = at_bat.sort_values("pitch_number")
    
    for i in range(1, len(at_bat)):
        seq = at_bat[sequence_features].iloc[:i].values
        if len(seq) < MAX_SEQ_LEN:
            pad = np.zeros((MAX_SEQ_LEN - len(seq), len(sequence_features)))
            seq = np.vstack([pad, seq])
        else:
            seq = seq[-MAX_SEQ_LEN:]
        
        sequences.append(seq)
        indices.append(at_bat.index[i])

X_seq = np.array(sequences)


lstm_probs = lstm_model.predict(X_seq, verbose=0)

lstm_prob_cols = [f"lstm_prob_{c}" for c in le_pitch.classes_]
lstm_prob_df = pd.DataFrame(lstm_probs, columns=lstm_prob_cols, index=indices)

df = df.loc[indices].copy()
df = df.join(lstm_prob_df)

print("Combined dataframe shape:", df.shape)
print("Sample LSTM probability columns:\n", df[lstm_prob_cols].head())

df = pd.get_dummies(df, columns=["p_throws", "stand", "pitcher_name", "prev_pitch", "count"])

df["pitch_type"] = le_pitch.transform(df["pitch_type"])

df = df.drop(columns=["game_date", "pitch_number", "at_bat_number", "prev_pitch_encoded"])

X = df.drop(columns=["pitch_type"])
y = df["pitch_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.2,
    random_state=42,
    eval_metric="mlogloss"
)
model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = model.predict(X_test)
print("Hybrid Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le_pitch.classes_))

joblib.dump(model, "models/hybrid_model.pkl")
joblib.dump(list(X.columns), "models/hybrid_feature_columns.pkl")
print("Hybrid model saved.")