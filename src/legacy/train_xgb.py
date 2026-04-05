import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

df = pd.read_csv("data/clean_pitches.csv")

df = pd.get_dummies(df, columns=["p_throws", "stand", "pitcher_name", "prev_pitch", "count"])

le_pitch = LabelEncoder()
df["pitch_type"] = le_pitch.fit_transform(df["pitch_type"])

# Features (X) and target (y)
X = df.drop(columns=["pitch_type"])
y = df["pitch_type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_pitch.classes_))

joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(le_pitch, "models/label_encoder.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")

print("Model saved")