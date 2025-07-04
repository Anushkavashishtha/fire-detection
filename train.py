# train.py

import os
import cv2
import torch
import numpy as np
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

# Set seed
torch.manual_seed(42)

# Custom Dataset class
class FireDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)

        if image is None:
            print(f"[Warning] Skipping unreadable image: {image_path}")
            return self.__getitem__((idx + 1) % len(self.image_paths))  # skip to next

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# Load image paths and labels
def load_data(data_dir):
    classes = sorted(os.listdir(data_dir))  # ensure consistent class order
    image_paths = []
    labels = []
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for cls in classes:
        cls_folder = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_folder):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(cls_folder, img_name)
                image_paths.append(img_path)
                labels.append(class_to_idx[cls])

    return image_paths, labels, class_to_idx

# CNN model
class FireCNN(nn.Module):
    def __init__(self):
        super(FireCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Main training logic
if __name__ == "__main__":
    data_dir = "dataset"
    image_paths, labels, class_to_idx = load_data(data_dir)

    print(f"Classes: {class_to_idx}")
    print(f"Total images: {len(image_paths)}")

    # Train-validation split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Image transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Datasets and loaders
    train_dataset = FireDataset(train_paths, train_labels, transform)
    val_dataset = FireDataset(val_paths, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FireCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Save the model
    torch.save(model.state_dict(), "fire_cnn.pth")
    print("✅ Model saved as fire_cnn.pth")

