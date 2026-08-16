import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix
import os

print("🔄 Setting up data pipeline...")
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
])

# 1. Load data and extract an ultra-lightweight training & test slice
full_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
full_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Using a subset ensures your MacBook finishes training in seconds rather than hours
train_dataset = Subset(full_train, list(range(600)))
test_dataset = Subset(full_test, list(range(150)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. Configure model architecture
print("🤖 Loading pre-trained ResNet-18 engine...")
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False  # Freeze old layers

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10 fashion product classes

# 3. Define Loss and Optimizer (Only optimizing the final layer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

# 4. Model Training Loop
print("\n🏋️ Training the final layer (Transfer Learning)...")
model.train()
for epoch in range(2):  # 2 quick iterations
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"   Epoch {epoch+1}/2 complete. Loss: {running_loss/len(train_loader):.4f}")

# 5. Evaluate and Generate Confusion Matrix
print("\n📊 Evaluating model on validation data...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Calculate custom accuracy & build evaluation matrix
cm = confusion_matrix(all_labels, all_preds)
print("\n=== CONFUSION MATRIX ANALYSIS ===")
print(cm)
print("\n💡 Read: Rows represent true items, columns represent model predictions.")

# 6. Export production vision asset
os.makedirs('artifacts', exist_ok=True)
artifact_path = 'artifacts/product_classifier.pt'
torch.save(model.state_dict(), artifact_path)
print(f"\n✅ SUCCESS: Vision model asset saved to -> {artifact_path}")
