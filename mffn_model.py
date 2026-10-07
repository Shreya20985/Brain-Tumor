import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ====================================================================
# 1. Fractal Residual Learning Modules (Figure 4 in the Paper)
# ====================================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FractalResidualBlockM1(nn.Module):
    """
    Base Block M1: Two sequential Conv-BN-ReLU layers with a residual connection.
    """
    def __init__(self, channels):
        super(FractalResidualBlockM1, self).__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = ConvBNReLU(channels, channels)
        self.shortcut = ConvBNReLU(channels, channels, kernel_size=1, padding=0)

    def forward(self, z):
        res = self.conv2(self.conv1(z))
        shortcut = self.shortcut(z)
        return res + shortcut


class FractalResidualBlockM2(nn.Module):
    """
    Recursive Level M2: Evaluates M1 sub-paths and joins them with the input z.
    """
    def __init__(self, channels):
        super(FractalResidualBlockM2, self).__init__()
        self.m1_sub = FractalResidualBlockM1(channels)
        self.join_conv = ConvBNReLU(channels * 3, channels, kernel_size=1, padding=0)

    def forward(self, z):
        m1_out = self.m1_sub(z)
        # Multi-scale join of M1, recursive M1, and raw input z
        m2_stacked = torch.cat([m1_out, self.m1_sub(m1_out), z], dim=1)
        return self.join_conv(m2_stacked)


class FractalResidualBlockM3(nn.Module):
    """
    Level M3: Hierarchical combination of M2, M1, and residual input z.
    """
    def __init__(self, channels):
        super(FractalResidualBlockM3, self).__init__()
        self.m2 = FractalResidualBlockM2(channels)
        self.m1 = FractalResidualBlockM1(channels)
        self.join_conv = ConvBNReLU(channels * 3, channels, kernel_size=1, padding=0)

    def forward(self, z):
        out_m2 = self.m2(z)
        out_m1 = self.m1(z)
        combined = torch.cat([out_m2, out_m1, z], dim=1)
        return self.join_conv(combined)


# ====================================================================
# 2. Multi-Scale Fractal Feature Network (MFFN)
# ====================================================================

class MFFN(nn.Module):
    """
    Multi-Scale Fractal Feature Network (Jakhar et al., 2024)
    Takes 3-channel input (context slices or multi-scale channels), extracts
    fractal multi-depth features, and routes them to a classifier.
    """
    def __init__(self, in_channels=3, num_classes=3):
        super(MFFN, self).__init__()
        
        # Initial projection
        self.stem = ConvBNReLU(in_channels, 32, kernel_size=3, stride=1, padding=1)
        
        # Block 1: Channels = 32
        self.stage1 = FractalResidualBlockM3(channels=32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: Channels = 64
        self.trans1 = ConvBNReLU(32, 64, kernel_size=1, padding=0)
        self.stage2 = FractalResidualBlockM3(channels=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: Channels = 128
        self.trans2 = ConvBNReLU(64, 128, kernel_size=1, padding=0)
        self.stage3 = FractalResidualBlockM3(channels=128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Aggregation & Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def extract_features(self, x):
        """Extracts deep embeddings for the SVM stage."""
        x = self.stem(x)
        x = self.pool1(self.stage1(x))
        x = self.trans1(x)
        x = self.pool2(self.stage2(x))
        x = self.trans2(x)
        x = self.pool3(self.stage3(x))
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.fc(feats)
        return logits


# ====================================================================
# 3. Training & Evaluation Pipeline
# ====================================================================

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = corrects / total
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")


def evaluate_with_svm(model, dataloaders, device):
    """
    Extracts features using the trained MFFN backbone and fits an SVM classifier
    as documented in Section 3.6 of the paper.
    """
    print("\n--- Running SVM Classifier on Extracted MFFN Features ---")
    model.eval()
    
    def get_features_and_labels(loader):
        features_list, labels_list = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                feats = model.extract_features(inputs)
                features_list.append(feats.cpu().numpy())
                labels_list.append(labels.numpy())
        return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

    X_train, y_train = get_features_and_labels(dataloaders['train'])
    X_val, y_val = get_features_and_labels(dataloaders['val'])

    # Fit SVM with linear kernel matching the paper
    svm_clf = SVC(kernel='linear', C=1.0)
    svm_clf.fit(X_train, y_train)

    y_pred = svm_clf.predict(X_val)
    print("Classification Report (MFFN + SVM):")
    print(classification_report(y_val, y_pred, target_names=['No Tumor', 'LG Tumor', 'HG Tumor']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))


# ====================================================================
# 4. Main Entry Point
# ====================================================================

if __name__ == '__main__':
    data_dir = "./brain_tumor_dataset"  # Point to your local dataset path
    batch_size = 16
    epochs = 20
    learning_rate = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Multi-slice / image transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Dummy data fallback for testing without downloading immediately
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' not found. Initializing a simulated test run...")
        os.makedirs(f"{data_dir}/train/no_tumor", exist_ok=True)
        os.makedirs(f"{data_dir}/train/glioma_tumor", exist_ok=True)
        os.makedirs(f"{data_dir}/train/meningioma_tumor", exist_ok=True)
        os.makedirs(f"{data_dir}/val/no_tumor", exist_ok=True)
        os.makedirs(f"{data_dir}/val/glioma_tumor", exist_ok=True)
        os.makedirs(f"{data_dir}/val/meningioma_tumor", exist_ok=True)
        
        from PIL import Image
        for split in ['train', 'val']:
            for cls in ['no_tumor', 'glioma_tumor', 'meningioma_tumor']:
                for i in range(5):
                    dummy_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
                    dummy_img.save(f"{data_dir}/{split}/{cls}/sample_{i}.png")

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=2)
        for x in ['train', 'val']
    }

    model = MFFN(in_channels=3, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    print(f"Using compute device: {device}")
    train_model(model, dataloaders, criterion, optimizer, device, num_epochs=epochs)
    evaluate_with_svm(model, dataloaders, device)