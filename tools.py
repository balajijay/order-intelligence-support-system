from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
RETURN_MODEL_PATH = ROOT / "models" / "return_risk_model.pkl"
IMAGE_MODEL_PATH = ROOT / "models" / "product_classifier.pt"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def check_return_risk(order_features: dict) -> dict:
    """Load and call the saved Part 1 pipeline."""
    model = joblib.load(RETURN_MODEL_PATH)

    required_columns = list(model.feature_columns_)
    row = {
        column: order_features.get(column)
        for column in required_columns
    }

    probability = float(
        model.predict_proba(pd.DataFrame([row]))[0, 1]
    )
    threshold = float(model.t_rf_)

    if probability < threshold:
        bucket = "Low"
    elif probability >= threshold + 0.15:
        bucket = "High"
    else:
        bucket = "Medium"

    return {
        "return_probability": round(probability, 6),
        "risk_bucket": bucket,
        "t_rf": threshold,
        "cut_points": {
            "low_below": threshold,
            "high_at_or_above": threshold + 0.15,
        },
    }


def _load_image_model():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    state_dict = torch.load(IMAGE_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def classify_product_image(image_path: str) -> dict:
    """Load and call the saved Part 2 classifier on a real PNG file."""
    path = Path(image_path)
    if not path.is_absolute():
        path = ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert("L")
    tensor = _IMAGE_TRANSFORM(image).unsqueeze(0)

    model = _load_image_model()

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        class_index = int(probabilities.argmax())
        confidence = float(probabilities[class_index])

    return {
        "category": CLASS_NAMES[class_index],
        "confidence": round(confidence, 6),
        "image_path": str(path.relative_to(ROOT)),
    }