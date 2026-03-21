import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("data/clean_pitches.csv")

df = pd.get_dummies(df, columns=["p_throws", "stand", "pitcher_name", "prev_pitch", "count"], drop_first=True)

le_pitch = LabelEncoder()
df["pitch_type"] = le_pitch.fit_transform(df["pitch_type"])

# Features (X) and target (y)
X = df.drop(columns=["pitch_type"])
y = df["pitch_type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_pitch.classes_))