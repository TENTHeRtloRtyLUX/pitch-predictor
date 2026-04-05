import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("data/clean_pitches.csv")

# Label encoding for categorical vars
# le_pitch = LabelEncoder()
# le_throws = LabelEncoder()
# le_stand = LabelEncoder()
# le_pitcher = LabelEncoder()

# df["pitch_type"] = le_pitch.fit_transform(df["pitch_type"])
# df["p_throws"] = le_throws.fit_transform(df["p_throws"])
# df["stand"] = le_stand.fit_transform(df["stand"])
# df["pitcher_name"] = le_pitcher.fit_transform(df["pitcher_name"])

# One-hot encoding for categorical vars
df = pd.get_dummies(df, columns=["p_throws", "stand", "pitcher_name"])

le_pitch = LabelEncoder()
df["pitch_type"] = le_pitch.fit_transform(df["pitch_type"])

# Features (X) and target (y)
X = df.drop(columns=["pitch_type"])
y = df["pitch_type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_pitch.classes_))