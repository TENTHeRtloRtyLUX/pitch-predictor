import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

DROP_COLUMNS = ["game_id", "at_bat_number", "pitch_number", "batter_name", "id"]
CATEGORICAL_COLUMNS = ["p_throws", "stand", "pitcher_name", "prev_pitch", "count"]
TARGET_COLUMN = "pitch_type"

def prepare_tabular_features(df):
    df = df.copy()

    existing_drop_columns = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=existing_drop_columns)

    df = pd.get_dummies(df, columns=[col for col in CATEGORICAL_COLUMNS if col in df.columns])

    label_encoder = LabelEncoder()
    df[TARGET_COLUMN] = label_encoder.fit_transform(df[TARGET_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y, label_encoder

def split_training_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def compute_balanced_weights(y_train):
    return compute_sample_weight(class_weight="balanced", y=y_train)


