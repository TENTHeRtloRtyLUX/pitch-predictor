import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    LSTM,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from model_artifacts import save_bundle_metadata
from training_data_pipeline import (
    build_sequence_training_dataset,
    save_sequence_preprocessor,
    save_sequence_split_manifest,
)


MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")


def ensure_output_dirs(models_dir=MODELS_DIR, metrics_dir=METRICS_DIR):
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)


def build_lstm_model(max_sequence_length, vocab_size, sequence_feature_dim, static_feature_dim, num_classes):
    token_input = Input(shape=(max_sequence_length,), name="sequence_tokens")
    numeric_input = Input(shape=(max_sequence_length, sequence_feature_dim), name="sequence_numeric")
    static_input = Input(shape=(static_feature_dim,), name="static_features")

    token_embedding = Embedding(input_dim=vocab_size, output_dim=16, mask_zero=True)(token_input)
    sequence_input = Concatenate(axis=-1)([token_embedding, numeric_input])
    sequence_encoding = LSTM(64)(sequence_input)

    combined = Concatenate()([sequence_encoding, static_input])
    hidden = Dense(64, activation="relu")(combined)
    hidden = Dropout(0.3)(hidden)
    hidden = Dense(32, activation="relu")(hidden)
    hidden = Dropout(0.2)(hidden)
    output = Dense(num_classes, activation="softmax")(hidden)

    model = Model(inputs=[token_input, numeric_input, static_input], outputs=output)
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def transformer_block(inputs, num_heads=4, key_dim=32, ff_dim=64, dropout_rate=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)(
        inputs,
        inputs,
    )
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_residual = Add()([inputs, attention_output])
    attention_norm = LayerNormalization(epsilon=1e-6)(attention_residual)

    feed_forward = Dense(ff_dim, activation="relu")(attention_norm)
    feed_forward = Dense(inputs.shape[-1])(feed_forward)
    feed_forward = Dropout(dropout_rate)(feed_forward)
    feed_forward_residual = Add()([attention_norm, feed_forward])
    return LayerNormalization(epsilon=1e-6)(feed_forward_residual)


def build_transformer_model(max_sequence_length, vocab_size, sequence_feature_dim, static_feature_dim, num_classes):
    token_input = Input(shape=(max_sequence_length,), name="sequence_tokens")
    numeric_input = Input(shape=(max_sequence_length, sequence_feature_dim), name="sequence_numeric")
    static_input = Input(shape=(static_feature_dim,), name="static_features")

    token_embedding = Embedding(input_dim=vocab_size, output_dim=32, mask_zero=False)(token_input)
    sequence_input = Concatenate(axis=-1)([token_embedding, numeric_input])
    encoded = transformer_block(sequence_input, num_heads=4, key_dim=16, ff_dim=64, dropout_rate=0.1)
    encoded = transformer_block(encoded, num_heads=4, key_dim=16, ff_dim=64, dropout_rate=0.1)
    pooled = GlobalAveragePooling1D()(encoded)

    combined = Concatenate()([pooled, static_input])
    hidden = Dense(64, activation="relu")(combined)
    hidden = Dropout(0.3)(hidden)
    output = Dense(num_classes, activation="softmax")(hidden)

    model = Model(inputs=[token_input, numeric_input, static_input], outputs=output)
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_model(model_type, dataset):
    train_examples = dataset["train_examples"]
    vocab_size = len(dataset["preprocessor"]["pitch_to_id"])
    max_sequence_length = dataset["preprocessor"]["max_sequence_length"]
    sequence_feature_dim = train_examples["sequence_numeric"].shape[-1]
    static_feature_dim = train_examples["static_features"].shape[-1]
    num_classes = len(dataset["label_encoder"].classes_)

    if model_type == "lstm":
        return build_lstm_model(
            max_sequence_length,
            vocab_size,
            sequence_feature_dim,
            static_feature_dim,
            num_classes,
        )
    if model_type == "transformer":
        return build_transformer_model(
            max_sequence_length,
            vocab_size,
            sequence_feature_dim,
            static_feature_dim,
            num_classes,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def evaluate_sequence_model(model, examples, label_encoder):
    predictions = model.predict(
        [
            examples["sequence_tokens"],
            examples["sequence_numeric"],
            examples["static_features"],
        ],
        verbose=0,
    )
    y_pred = np.argmax(predictions, axis=1)
    y_true = examples["targets"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_sequence_bundle(
    model_name,
    model_type,
    model,
    metrics,
    label_encoder,
    preprocessor,
    split_manifest_path,
    models_dir=MODELS_DIR,
    metrics_dir=METRICS_DIR,
):
    model_filename = f"{model_name}.keras"
    label_encoder_filename = f"{model_name}_label_encoder.pkl"
    preprocessor_filename = f"{model_name}_preprocessor.pkl"
    metrics_filename = f"{model_name}_metrics.json"

    model.save(models_dir / model_filename)
    joblib.dump(label_encoder, models_dir / label_encoder_filename)
    save_sequence_preprocessor(preprocessor, models_dir / preprocessor_filename)

    with open(metrics_dir / metrics_filename, "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)

    save_bundle_metadata(
        model_name=model_name,
        model_type=model_type,
        model_filename=model_filename,
        label_encoder_filename=label_encoder_filename,
        preprocessor_filename=preprocessor_filename,
        metrics_filename=metrics_filename,
        output_dir=models_dir,
        extra_metadata={"split_manifest_filename": Path(split_manifest_path).name},
    )


def train_sequence_model(
    model_type,
    seasons=(2023, 2024, 2025),
    max_sequence_length=12,
    models_dir=MODELS_DIR,
    metrics_dir=METRICS_DIR,
    epochs=15,
    batch_size=64,
):
    ensure_output_dirs(models_dir=models_dir, metrics_dir=metrics_dir)
    dataset = build_sequence_training_dataset(
        seasons=list(seasons),
        max_sequence_length=max_sequence_length,
    )
    split_manifest_path = models_dir / f"{model_type}_sequence_split.json"
    save_sequence_split_manifest(dataset, split_manifest_path)

    model = build_model(model_type=model_type, dataset=dataset)
    callbacks = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    history = model.fit(
        [
            dataset["train_examples"]["sequence_tokens"],
            dataset["train_examples"]["sequence_numeric"],
            dataset["train_examples"]["static_features"],
        ],
        dataset["train_examples"]["targets"],
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    metrics = evaluate_sequence_model(model, dataset["test_examples"], dataset["label_encoder"])
    metrics["epochs_ran"] = len(history.history.get("loss", []))
    metrics["train_examples"] = int(len(dataset["train_examples"]["targets"]))
    metrics["test_examples"] = int(len(dataset["test_examples"]["targets"]))

    model_name = f"{model_type}_sequence"
    save_sequence_bundle(
        model_name=model_name,
        model_type=model_type,
        model=model,
        metrics=metrics,
        label_encoder=dataset["label_encoder"],
        preprocessor=dataset["preprocessor"],
        split_manifest_path=split_manifest_path,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
    )

    return {
        "success": True,
        "model_name": model_name,
        "model_type": model_type,
        "accuracy": metrics["accuracy"],
        "models_dir": str(models_dir.as_posix()),
        "metrics_dir": str(metrics_dir.as_posix()),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a sequence model for pitch prediction.")
    parser.add_argument("--model-type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--max-sequence-length", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    summary = train_sequence_model(
        model_type=args.model_type,
        max_sequence_length=args.max_sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
