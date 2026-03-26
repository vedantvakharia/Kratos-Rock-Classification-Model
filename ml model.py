import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import os
import random

# --- CONFIGURATION ---
random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(0)
torch.manual_seed(0)

# Check for GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on GPU:", torch.cuda.get_device_name(0))
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
else:
    device = torch.device("cpu")
    print("Running on CPU...")

config = dict(
    epochs=50,                  # train longer
    classes=15,
    train_batch_size=32,
    val_batch_size=32,
    test_batch_size=32,
    lr=2e-4,  # a bit higher for fine-tuning
    dataset="15 Class Rock Data",
    architecture="EfficientNet-B4 + fine-tuning + mixup"
)

# --- HELPER FUNCTIONS ---
def make_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=2,
                      pin_memory=True if torch.cuda.is_available() else False)

def make(config):
    # Data augmentations
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(380),  # EfficientNet-B4 default input size = 380
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    
    # Load dataset
    dataset = torchvision.datasets.ImageFolder(
        root=r"C:\Users\Vedant\Downloads\Rock Classification Models\Dataset1",
        transform=train_transforms
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    
    # Apply different transforms for val/test
    val_data.dataset.transform = test_transforms
    test_data.dataset.transform = test_transforms

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Load pretrained EfficientNet-B4
    model = torchvision.models.efficientnet_b4(weights="EfficientNet_B4_Weights.IMAGENET1K_V1")
    
    # Unfreeze last 2 blocks for fine-tuning
    for param in model.parameters():
        param.requires_grad = False
    for param in list(model.features[-2].parameters()) + list(model.features[-1].parameters()):
        param.requires_grad = True

    # Adjust classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, config['classes'])
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    model.to(device)
    criterion.to(device)

    return model, train_data, val_data, test_data, criterion, optimizer, scheduler

# --- MIXUP FUNCTION ---
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- TRAINING & EVALUATION ---
def train_model(model, train_data, val_data, criterion, optimizer, scheduler, epochs):
    train_loader = make_loader(train_data, config['train_batch_size'])
    val_loader = make_loader(val_data, config['val_batch_size'], shuffle=False)

    best_val_acc = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # --- TRAIN ---
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (lam * preds.eq(targets_a).sum().item() +
                                 (1 - lam) * preds.eq(targets_b).sum().item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- VALIDATE ---
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model")

    print("\nTraining complete. Best Val Acc: {:.4f}".format(best_val_acc))

def test_model(model, test_data, criterion):
    test_loader = make_loader(test_data, config['test_batch_size'], shuffle=False)
    model.eval()
    test_loss, test_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data).item()

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_corrects / len(test_loader.dataset)
    print(f"\nTest Loss: {test_loss:.4f} Acc: {test_acc:.4f}")


# --- MAIN ---
if __name__ == '__main__':
    if os.path.exists("C:\\Users\\Vedant\\Downloads\\Rock Classification Models\\Dataset"):
        model, train_data, val_data, test_data, criterion, optimizer, scheduler = make(config)
        train_model(model, train_data, val_data, criterion, optimizer, scheduler, config['epochs'])
        
        # Load best model for testing
        model.load_state_dict(torch.load("best_model.pth"))
        test_model(model, test_data, criterion)
    else:
        print("Error: Data directory './Kratos_ML/Data/train' not found.")
