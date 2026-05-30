# ============================================================
# 2.1 PyTorch CNN Fashion-MNIST Classification
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import random
import numpy as np


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# 2. Hyperparameters
# ============================================================

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4


# ============================================================
# 3. Dataset & DataLoader
# ============================================================

# 학습용 transform
# RandomCrop, RandomHorizontalFlip이 들어가서 학습 난이도를 높이고 일반화 성능을 올림
train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# 평가용 transform
# train accuracy / test accuracy 제출용 측정에는 증강 없이 평가
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# 실제 학습에 사용하는 train dataset
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

# 제출용 train accuracy 측정용 dataset
# 같은 train set이지만 transform은 eval_transform 사용
train_eval_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=eval_transform
)

# test dataset
test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=eval_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

train_eval_loader = DataLoader(
    train_eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# 4. CNN Model
# ============================================================

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()

        # Input: 1 x 28 x 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.10)
        )
        # Output: 32 x 14 x 14

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.15)
        )
        # Output: 64 x 7 x 7

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Dropout(0.20)
        )
        # Output: 128 x 7 x 7

        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


model = FashionCNN().to(device)
print(model)


# ============================================================
# 5. Loss, Optimizer, Scheduler
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# 6. Train / Evaluate Functions
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
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

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total

    return train_loss, train_acc


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    eval_loss = running_loss / total
    eval_acc = 100.0 * correct / total

    return eval_loss, eval_acc


# ============================================================
# 7. Training Loop
# ============================================================

train_losses = []
test_losses = []
train_accs = []
test_accs = []

best_test_acc = 0.0
best_epoch = 0

for epoch in range(1, EPOCHS + 1):

    # 실제 학습
    train_aug_loss, train_aug_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    # 제출용 train accuracy 측정
    # 증강 없는 train set + eval 모드
    train_eval_loss, train_eval_acc = evaluate(
        model,
        train_eval_loader,
        criterion,
        device
    )

    # test accuracy 측정
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    scheduler.step()

    # 그래프에는 제출용 train accuracy를 기록
    train_losses.append(train_eval_loss)
    test_losses.append(test_loss)
    train_accs.append(train_eval_acc)
    test_accs.append(test_acc)

    # 가장 좋은 test accuracy 모델 저장
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch
        torch.save(model.state_dict(), "best_fashion_mnist_cnn.pth")

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Train Acc(eval): {train_eval_acc:.2f}% | "
        f"Test Acc: {test_acc:.2f}% | "
        f"Train Loss(eval): {train_eval_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Train Acc(aug): {train_aug_acc:.2f}%"
    )


print("\n======================================")
print(f"Best Epoch: {best_epoch}")
print(f"Best Test Accuracy: {best_test_acc:.2f}%")
print("======================================")


# ============================================================
# 8. Load Best Model and Final Evaluation
# ============================================================

model.load_state_dict(torch.load("best_fashion_mnist_cnn.pth", map_location=device))

final_train_loss, final_train_acc = evaluate(
    model,
    train_eval_loader,
    criterion,
    device
)

final_test_loss, final_test_acc = evaluate(
    model,
    test_loader,
    criterion,
    device
)

print("\nFinal Evaluation with Best Model")
print(f"Final Train Loss: {final_train_loss:.4f}")
print(f"Final Train Accuracy: {final_train_acc:.2f}%")
print(f"Final Test Loss: {final_test_loss:.4f}")
print(f"Final Test Accuracy: {final_test_acc:.2f}%")


# ============================================================
# 9. Graph Visualization
# ============================================================

epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accs, label="Train Accuracy")
plt.plot(epochs, test_accs, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Fashion-MNIST CNN Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("fashion_mnist_accuracy_graph.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Fashion-MNIST CNN Loss")
plt.legend()
plt.grid(True)
plt.savefig("fashion_mnist_loss_graph.png", dpi=300)
plt.show()


# ============================================================
# 10. Save Final Result Text
# ============================================================

with open("result_summary.txt", "w", encoding="utf-8") as f:
    f.write("Fashion-MNIST CNN Classification Result\n")
    f.write("=====================================\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Final Train Accuracy: {final_train_acc:.2f}%\n")
    f.write(f"Final Test Accuracy: {final_test_acc:.2f}%\n")
    f.write(f"Final Train Loss: {final_train_loss:.4f}\n")
    f.write(f"Final Test Loss: {final_test_loss:.4f}\n")

print("\nSaved files:")
print("- best_fashion_mnist_cnn.pth")
print("- fashion_mnist_accuracy_graph.png")
print("- fashion_mnist_loss_graph.png")
print("- result_summary.txt")
