import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms,models
print('===============')
print('image classifier training')
print('===============')
BATCH_SIZE=8
IMAGE_SIZE=128
EPOCHS=25
LEARNING_RATE=0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

print("\nLoading dataset...")
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(root="data/train", transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)

categories = train_data.classes
print(f"Categories found: {categories}")
print(f"Total training images: {len(train_data)}")

print("\nBuilding model (MobileNetV2)...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze early layers to save VRAM and train faster
for param in model.features[:14].parameters():
    param.requires_grad = False

# Adjust final output layer to match your categories count
model.classifier[1] = nn.Linear(model.last_channel, len(categories))
model = model.to(device)

print("\nTraining started...\n")
criterion = nn.CrossEntropyLoss()

# Using AdamW - The modern industry standard optimizer!
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%")

save_path = "my_ai_model.pth"
torch.save({
    'model_state': model.state_dict(),
    'categories': categories,
    'image_size': IMAGE_SIZE
}, save_path)

print("\nTraining complete!")
print(f"Model saved to: {save_path}")