import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

print("Loading AI model...")

checkpoint = torch.load("my_ai_model.pth", map_location="cpu", weights_only=False)
categories = checkpoint['categories']
image_size = checkpoint['image_size']

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(categories))
model.load_state_dict(checkpoint['model_state'])
model.eval()

print(f"Categories: {categories}")
print("Model is ready!\n")

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = probabilities.max(0)

    predicted = categories[top_idx.item()]
    confidence = top_prob.item() * 100

    print(f"Image: {image_path}")
    print(f"Prediction: {predicted.upper()}")
    print(f"Confidence: {confidence:.1f}%")

if len(sys.argv) > 1:
    predict(sys.argv[1])
else:
    print("Usage: py test_my_ai.py path/to/image.jpg")
    