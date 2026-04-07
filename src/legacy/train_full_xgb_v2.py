import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

df = pd.read_csv("data/clean_full_pitches_with_tendencies.csv")

df = df.drop(columns=["game_id", "at_bat_number", "pitch_number", "batter_name"])

df = pd.get_dummies(df, columns=["p_throws", "stand", "pitcher_name", "prev_pitch", "count"])

le_pitch = LabelEncoder()
df["pitch_type"] = le_pitch.fit_transform(df["pitch_type"])

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

print("Training on", len(X_train), "pitches...")
model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le_pitch.classes_))

joblib.dump(model, "models/full_xgb_v2_model.pkl")
joblib.dump(le_pitch, "models/full_v2_label_encoder.pkl")
joblib.dump(list(X.columns), "models/full_v2_feature_columns.pkl")
print("\nModel v2 saved.")