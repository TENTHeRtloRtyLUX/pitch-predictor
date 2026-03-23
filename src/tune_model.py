import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

df = pd.read_csv("data/clean_pitches.csv")

df = pd.get_dummies(df, columns=["p_throws", "stand", "pitcher_name", "prev_pitch", "count"])

le_pitch = LabelEncoder()
df["pitch_type"] = le_pitch.fit_transform(df["pitch_type"])

X = df.drop(columns=["pitch_type"])
y = df["pitch_type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
}

model = XGBClassifier(random_state = 42, eval_metric="mlogloss")

search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="f1_macro",
    verbose=1,
    n_jobs=-1,
)

search.fit(X_train, y_train, sample_weight=sample_weights)

print("\nBest parameters:", search.best_params_)
print("Best F1 Score:", search.best_score_)