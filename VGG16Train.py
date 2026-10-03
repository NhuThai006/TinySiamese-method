"""
Train VGG16 4-class (Arch, LeftLoop, RightLoop, Whorl) and save
model/VGG16FVC.h5 with the same architecture as FeatureVectorsTrain.py.

Run (conda env tinysiamese, from this folder):
    python VGG16Train.py
"""
import os
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image


INPUT_PATH = "Database/Train"
OUTPUT_PATH = "model/VGG16FVC.h5"
NUM_CLASSES = 4
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4
REPEATS_PER_IMAGE = 2


def build_vgg16_fvc():
    try:
        from torchvision.models import VGG16_Weights

        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.vgg16(pretrained=True)

    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, NUM_CLASSES),
    )
    return model


def is_readable_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


class RepeatedFolder(Dataset):
    """ImageFolder wrapper that skips corrupt files and repeats with augmentation."""

    def __init__(self, folder, transform, repeats):
        self.base = datasets.ImageFolder(folder)
        skipped = []
        valid = []
        for path, label in self.base.samples:
            if is_readable_image(path):
                valid.append((path, label))
            else:
                skipped.append(path)
        if skipped:
            print("Skipped unreadable files (%d):" % len(skipped))
            for path in skipped:
                print(" -", path)
        self.samples = valid
        self.transform = transform
        self.repeats = max(1, repeats)
        if len(self.base.classes) != NUM_CLASSES:
            raise RuntimeError(
                "Can train VGG16FVC only with exactly 4 class folders under "
                "Database/Train (Arch, LeftLoop, RightLoop, Whorl). Found: "
                + str(self.base.classes)
            )
        if not self.samples:
            raise RuntimeError("No readable images found in Database/Train.")

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, index):
        real_index = index % len(self.samples)
        path, label = self.samples[real_index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def main():
    if not os.path.isdir(INPUT_PATH):
        raise FileNotFoundError(
            "Missing folder %s. Put fingerprint images into "
            "Database/Train/<class>/ first." % INPUT_PATH
        )

    os.makedirs("model", exist_ok=True)

    train_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = RepeatedFolder(INPUT_PATH, train_tf, REPEATS_PER_IMAGE)
    print("Classes:", dataset.base.classes, flush=True)
    print("Readable images:", len(dataset.samples), "| Augmented/epoch:", len(dataset), flush=True)
    if len(dataset.samples) < NUM_CLASSES:
        raise RuntimeError(
            "Need at least 1 image in each of the 4 class folders before training."
        )

    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    model = build_vgg16_fvc().to(device)
    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_acc = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch+1}/{EPOCHS}...", flush=True)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = 100.0 * correct / max(total, 1)
        print("loss: %.4f  acc: %.2f%%" % (epoch_loss, epoch_acc), flush=True)

        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            best_state = copy.deepcopy(model.state_dict())

    torch.save(best_state if best_state is not None else model.state_dict(), OUTPUT_PATH)
    print("Saved", OUTPUT_PATH, "best_train_acc=%.2f%%" % best_acc, flush=True)
    print(
        "Note: with only a few images this file exists so TinySiamese can run, "
        "but accuracy will be weak until you add more FVC images per class.",
        flush=True
    )


if __name__ == "__main__":
    main()
