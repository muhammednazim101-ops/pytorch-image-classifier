"""
Evaluate the trained cats-vs-dogs model on the held-out test set.

Prints overall accuracy, per-class accuracy, per-image predictions
and a confusion matrix, then writes model_info.json which the web
app (app.py) reads to show a "model card".

Run:  py evaluate_my_ai.py
"""
import json

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

MODEL_PATH = "my_ai_model.pth"
TEST_DIR = "data/test"
INFO_PATH = "model_info.json"

print("========================")
print(" model evaluation report")
print("========================")

# ---- load the trained model ----
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
categories = checkpoint["categories"]
image_size = checkpoint["image_size"]

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(categories))
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ---- same preprocessing the model was trained with (no augmentation!) ----
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_data = datasets.ImageFolder(root=TEST_DIR, transform=transform)
if test_data.classes != categories:
    raise SystemExit(
        f"Test classes {test_data.classes} don't match trained classes {categories}!"
    )

print(f"Test images: {len(test_data)} | Classes: {categories}\n")

# ---- run predictions, one image at a time (so we can report per image) ----
confusion = [[0] * len(categories) for _ in categories]
per_class = {c: {"correct": 0, "total": 0} for c in categories}
per_image = []

with torch.no_grad():
    for i in range(len(test_data)):
        path, true_idx = test_data.samples[i]
        image, _ = test_data[i]

        output = model(image.unsqueeze(0))
        probs = torch.softmax(output[0], dim=0)
        confidence, pred_idx = probs.max(0)

        pred_name = categories[pred_idx.item()]
        true_name = categories[true_idx]
        correct = pred_idx.item() == true_idx

        confusion[true_idx][pred_idx.item()] += 1
        per_class[true_name]["total"] += 1
        if correct:
            per_class[true_name]["correct"] += 1

        per_image.append({
            "path": path,
            "true": true_name,
            "predicted": pred_name,
            "confidence": round(confidence.item() * 100, 1),
            "correct": correct,
        })

        mark = "OK " if correct else "WRONG"
        print(f"[{mark}] {path}")
        print(f"       -> predicted {pred_name} ({confidence.item() * 100:.1f}%), "
              f"actual {true_name}")

# ---- overall results ----
total = sum(row["total"] for row in per_class.values())
correct_total = sum(row["correct"] for row in per_class.values())
accuracy = 100.0 * correct_total / total if total else 0.0

print("\n========================")
print(" results")
print("========================")
print(f"Overall test accuracy: {accuracy:.1f}% ({correct_total}/{total} images)\n")

print("Per-class accuracy:")
for name in categories:
    d = per_class[name]
    acc = 100.0 * d["correct"] / d["total"] if d["total"] else 0.0
    print(f"  {name}: {acc:.1f}% ({d['correct']}/{d['total']})")

print("\nConfusion matrix (rows = actual, columns = predicted):")
cell = max(len(c) for c in categories) + 2
header = " " * (cell + 5) + "".join(c.rjust(cell) for c in categories)
print(header)
for true_name, row in zip(categories, confusion):
    line = true_name.rjust(cell + 5) + "".join(str(n).rjust(cell) for n in row)
    print(line)

# ---- write model card for the web app ----
info = {
    "model": "MobileNetV2 (transfer learning)",
    "checkpoint": MODEL_PATH,
    "classes": categories,
    "image_size": image_size,
    "test_images": total,
    "test_accuracy": round(accuracy, 1),
    "per_class_accuracy": {
        name: round(100.0 * d["correct"] / d["total"], 1) if d["total"] else None
        for name, d in per_class.items()
    },
}
with open(INFO_PATH, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)

print(f"\nModel card saved to: {INFO_PATH}")
print("Run  py app.py  to launch the web app!")
