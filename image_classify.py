from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

RANDOM_STATE = 42
BATCH_SIZE = 128
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 2
IMAGE_SIZE = 96

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
CACHE_DIR = DATA_DIR / "feature_cache"
SAMPLE_DIR = DATA_DIR / "sample_images"
MODEL_PATH = MODEL_DIR / "product_classifier.pt"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def build_backbone():
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)
    feature_size = model.fc.in_features
    model.fc = nn.Identity()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, feature_size


def extract_features(backbone, dataset, indices, cache_name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_name}.pt"

    if cache_path.exists():
        print(f"Loading cached features: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    features = []
    labels = []

    backbone.eval()
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_features = backbone(
                batch_images.to(DEVICE)
            ).cpu()
            features.append(batch_features)
            labels.append(batch_labels.cpu())

    result = {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
    }
    torch.save(result, cache_path)
    print(f"Saved cached features: {cache_path}")
    return result


def train_head(features, labels, validation_features, validation_labels):
    head = nn.Linear(features.shape[1], 10).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        torch.utils.data.TensorDataset(features, labels),
        batch_size=256,
        shuffle=True,
    )

    for epoch in range(HEAD_EPOCHS):
        head.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            outputs = head(batch_features.to(DEVICE))
            loss = criterion(outputs, batch_labels.to(DEVICE))
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            validation_predictions = head(
                validation_features.to(DEVICE)
            ).argmax(dim=1).cpu()

        accuracy = (
            validation_predictions == validation_labels
        ).float().mean().item()

        print(
            f"Head epoch {epoch + 1}/{HEAD_EPOCHS} "
            f"validation accuracy: {accuracy:.4f}"
        )

    return head, accuracy


def predict_model(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    predictions = []
    labels = []

    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            outputs = model(images.to(DEVICE))
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy())

    return np.array(labels), np.array(predictions)


def main():
    print("Device:", DEVICE)
    print("Loading Fashion-MNIST full dataset...")

    train_dataset = torchvision.datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    all_train_indices = np.arange(len(train_dataset))
    train_indices, validation_indices = train_test_split(
        all_train_indices,
        test_size=5000,
        stratify=train_dataset.targets.numpy(),
        random_state=RANDOM_STATE,
    )

    print(
        f"Split sizes: train={len(train_indices)}, "
        f"validation={len(validation_indices)}, "
        f"test={len(test_dataset)}"
    )

    backbone, feature_size = build_backbone()
    backbone = backbone.to(DEVICE)

    train_cache = extract_features(
        backbone,
        train_dataset,
        train_indices,
        "train",
    )
    validation_cache = extract_features(
        backbone,
        train_dataset,
        validation_indices,
        "validation",
    )

    head, validation_accuracy = train_head(
        train_cache["features"],
        train_cache["labels"],
        validation_cache["features"],
        validation_cache["labels"],
    )

    print(
        "Feature-extraction validation accuracy:",
        round(validation_accuracy, 4),
    )

    # Build the final model from the frozen backbone and trained head.
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )
    model.fc = nn.Linear(feature_size, 10)
    model.load_state_dict({
        **{
            key: value
            for key, value in backbone.state_dict().items()
        },
        "fc.weight": head.weight.detach().cpu(),
        "fc.bias": head.bias.detach().cpu(),
    }, strict=False)
    model = model.to(DEVICE)

    if validation_accuracy < 0.80:
        print("Validation accuracy is below 80%; fine-tuning layer4.")

        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

        for parameter in model.fc.parameters():
            parameter.requires_grad = True

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.0001,
        )
        criterion = nn.CrossEntropyLoss()

        fine_tune_dataset = Subset(train_dataset, train_indices)
        loader = DataLoader(
            fine_tune_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        for epoch in range(FINE_TUNE_EPOCHS):
            model.train()
            for images, labels in loader:
                optimizer.zero_grad()
                outputs = model(images.to(DEVICE))
                loss = criterion(outputs, labels.to(DEVICE))
                loss.backward()
                optimizer.step()

            print(
                f"Fine-tuning epoch {epoch + 1}/{FINE_TUNE_EPOCHS} complete"
            )

        validation_labels, validation_predictions = predict_model(
            model,
            Subset(train_dataset, validation_indices),
        )
        validation_accuracy = (
            validation_labels == validation_predictions
        ).mean()

        print(
            "After fine-tuning validation accuracy:",
            round(float(validation_accuracy), 4),
        )
    else:
        print("Feature extraction alone was sufficient; fine-tuning skipped.")

    print("Evaluating once on untouched test split...")
    test_labels, test_predictions = predict_model(model, test_dataset)
    test_accuracy = (test_labels == test_predictions).mean()

    print("Final test accuracy:", round(float(test_accuracy), 4))
    print("\nConfusion matrix:")
    print(confusion_matrix(test_labels, test_predictions))

    print("\nPer-class precision and recall:")
    print(classification_report(
        test_labels,
        test_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    selected_classes = set()

    for index, label in enumerate(test_dataset.targets.tolist()):
        if label not in selected_classes:
            image = test_dataset.data[index].numpy()
            filename = f"{label:02d}_{CLASS_NAMES[label].lower().replace('/', '_').replace(' ', '_')}.png"
            Image.fromarray(image).save(SAMPLE_DIR / filename)
            selected_classes.add(label)

        if len(selected_classes) >= 5:
            break

    print(f"Exported {len(selected_classes)} real test images to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()