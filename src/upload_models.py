from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import os

load_dotenv()

token = os.environ.get("HF_TOKEN")
login(token=token)

api = HfApi()

api.create_repo("rkhosla/pitch-predictor", repo_type="model", exist_ok=True)

files = [
    "models/full_xgb_v2_model.pkl",
    "models/full_v2_label_encoder.pkl",
    "models/full_v2_feature_columns.pkl",
    "models/lstm_model.keras",
    "models/lstm_label_encoder.pkl",
    "models/lstm_prev_encoder.pkl",
    "models/hybrid_model.pkl",
    "models/hybrid_feature_columns.pkl",
]

for f in files:
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f.split("/")[-1],
        repo_id="rkhosla/pitch-predictor",
    )
    print(f"Uploaded {f}")

print("All models uploaded.")
