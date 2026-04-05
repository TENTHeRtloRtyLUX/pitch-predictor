import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

DROP_COLUMNS = ["game_id", "at_bat_number", "pitch_number", "batter_name", "id", "pitch_uid", "at_bat_index", "play_event_index"]
LOW_CARDINALITY_COLUMNS = ["p_throws", "stand", "prev_pitch", "count"]
HIGH_CARDINALITY_COLUMNS = ["pitcher_name"]
TARGET_COLUMN = "pitch_type"
HASH_FEATURES = 128


class SparsePitchPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, low_cardinality_columns=None, high_cardinality_columns=None, hash_features=HASH_FEATURES):
        self.low_cardinality_columns = low_cardinality_columns or LOW_CARDINALITY_COLUMNS
        self.high_cardinality_columns = high_cardinality_columns or HIGH_CARDINALITY_COLUMNS
        self.hash_features = hash_features
        try:
            self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
        self.feature_hasher = FeatureHasher(
            n_features=self.hash_features,
            input_type="string",
            alternate_sign=False,
        )

    def fit(self, df, y=None):
        df = df.copy()
        self.numeric_columns_ = [
            col for col in df.columns
            if col not in set(self.low_cardinality_columns + self.high_cardinality_columns)
        ]
        self.present_low_cardinality_columns_ = [
            col for col in self.low_cardinality_columns if col in df.columns
        ]
        self.present_high_cardinality_columns_ = [
            col for col in self.high_cardinality_columns if col in df.columns
        ]

        if self.present_low_cardinality_columns_:
            self.one_hot_encoder.fit(df[self.present_low_cardinality_columns_].fillna("__missing__").astype(str))

        return self

    def transform(self, df):
        if not hasattr(self, "numeric_columns_"):
            raise ValueError("SparsePitchPreprocessor must be fit before calling transform.")

        df = df.copy()
        matrices = []

        if self.numeric_columns_:
            numeric_matrix = sparse.csr_matrix(
                df.reindex(columns=self.numeric_columns_, fill_value=0).astype(float).to_numpy()
            )
            matrices.append(numeric_matrix)

        if self.present_low_cardinality_columns_:
            encoded_matrix = self.one_hot_encoder.transform(
                df[self.present_low_cardinality_columns_].fillna("__missing__").astype(str)
            )
            matrices.append(encoded_matrix)

        for col in self.present_high_cardinality_columns_:
            hashed_inputs = [[value] for value in df[col].fillna("__missing__").astype(str).tolist()]
            hashed_matrix = self.feature_hasher.transform(hashed_inputs)
            matrices.append(hashed_matrix.tocsr())

        if not matrices:
            raise ValueError("No features were produced by the tabular preprocessor.")

        return sparse.hstack(matrices, format="csr")

    def fit_transform(self, df, y=None):
        return self.fit(df, y=y).transform(df)


def validate_training_dataframe(df):
    if df is None or df.empty:
        raise ValueError("Training dataframe is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Training dataframe must include '{TARGET_COLUMN}'.")


def prepare_tabular_features(df):
    validate_training_dataframe(df)

    df = df.copy()
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Training dataframe has no rows after dropping missing targets.")

    existing_drop_columns = [col for col in DROP_COLUMNS if col in df.columns]
    feature_df = df.drop(columns=existing_drop_columns)

    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(feature_df[TARGET_COLUMN]), name=TARGET_COLUMN)

    X_df = feature_df.drop(columns=[TARGET_COLUMN])
    preprocessor = SparsePitchPreprocessor()
    X = preprocessor.fit_transform(X_df)

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Prepared feature matrix is empty.")

    return X, y, label_encoder, preprocessor


def split_training_data(X, y, test_size=0.2, random_state=42):
    if X.shape[0] == 0 or len(y) == 0:
        raise ValueError("Cannot split empty feature or target data.")

    class_counts = y.value_counts()
    if class_counts.empty or len(class_counts) < 2:
        raise ValueError("Need at least two target classes to train.")
    if (class_counts < 2).any():
        too_small = class_counts[class_counts < 2].index.tolist()
        raise ValueError(f"Cannot stratify because some classes have fewer than 2 rows: {too_small}")

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def compute_balanced_weights(y_train):
    if len(y_train) == 0:
        raise ValueError("Cannot compute sample weights for an empty training target.")
    return compute_sample_weight(class_weight="balanced", y=y_train)
