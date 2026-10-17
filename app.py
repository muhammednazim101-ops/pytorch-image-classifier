"""
Cats vs Dogs - AI web app
=========================
A Gradio web UI for the image classifier trained in train_my_ai.py.

Tabs:
  1. Upload image -> drag & drop any photo, get cat vs dog prediction
  2. Webcam       -> snap a photo with your camera
  3. Examples     -> one-click predictions on the built-in test images

Run:  py app.py   ->  open http://127.0.0.1:7860
"""
import glob
import json
import os
import sys

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "my_ai_model.pth"
INFO_PATH = "model_info.json"

# ---------------------------------------------------------------- model ----
if not os.path.exists(MODEL_PATH):
    sys.exit(
        f"Model file '{MODEL_PATH}' not found!\n"
        "Train it first:  py train_my_ai.py"
    )

print("Loading AI model...")
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
categories = checkpoint["categories"]
image_size = checkpoint["image_size"]

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(categories))
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"Ready! Categories: {categories}\n")

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def predict(image) -> dict:
    """Classify a PIL image, returning {label: confidence} for the UI."""
    if image is None:
        return {}
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output[0], dim=0)
    return {name: float(prob) for name, prob in zip(categories, probs)}


# ----------------------------------------------------------- model card ----
model_card_md = "*(model_info.json not found - run  py evaluate_my_ai.py  first)*"
if os.path.exists(INFO_PATH):
    with open(INFO_PATH, encoding="utf-8") as f:
        info = json.load(f)
    per_class = "\n".join(
        f"- **{name}**: {acc:.1f}%" if acc is not None else f"- **{name}**: no test images"
        for name, acc in info.get("per_class_accuracy", {}).items()
    )
    model_card_md = (
        f"**Architecture:** {info.get('model', 'MobileNetV2')}\n\n"
        f"**Classes:** {', '.join(info.get('classes', categories))}\n\n"
        f"**Test accuracy:** {info.get('test_accuracy', '?')}% "
        f"({info.get('test_images', '?')} held-out images)\n\n"
        f"{per_class}\n\n"
        f"**Input size:** {info.get('image_size', image_size)}x{info.get('image_size', image_size)} px"
    )

# ------------------------------------------------------------- examples ----
example_paths = sorted(glob.glob(os.path.join("data", "test", "*", "*")))

# ------------------------------------------------------------------- UI ----
with gr.Blocks(title="Cats vs Dogs AI") as demo:
    gr.Markdown(
        "# 🐱🐶 Cats vs Dogs AI\n"
        "Upload a photo, use your webcam, or click an example below - "
        "and watch the AI classify it in real time."
    )

    with gr.Tab("Upload image"):
        with gr.Row():
            upload_input = gr.Image(type="pil", label="Your image", height=320)
            upload_output = gr.Label(num_top_classes=len(categories),
                                     label="AI prediction")
        upload_button = gr.Button("Classify!", variant="primary")
        upload_button.click(predict, inputs=upload_input, outputs=upload_output)
        upload_input.change(predict, inputs=upload_input, outputs=upload_output)

    with gr.Tab("Webcam"):
        with gr.Row():
            webcam_input = gr.Image(sources=["webcam"], type="pil",
                                    label="Camera", height=320)
            webcam_output = gr.Label(num_top_classes=len(categories),
                                     label="AI prediction")
        webcam_input.change(predict, inputs=webcam_input, outputs=webcam_output)

    with gr.Tab("Examples"):
        with gr.Row():
            example_image = gr.Image(type="pil", label="Test image", height=320)
            example_output = gr.Label(num_top_classes=len(categories),
                                      label="AI prediction")
        # Prediction runs via the .change event when an example loads into
        # the image panel (works around Examples fn not firing in Gradio 6).
        example_image.change(predict, inputs=example_image,
                             outputs=example_output)
        gr.Examples(examples=example_paths, inputs=example_image,
                    outputs=example_output, fn=predict, cache_examples=False)

    with gr.Accordion("📊 Model card", open=False):
        gr.Markdown(model_card_md)

if __name__ == "__main__":
    demo.launch(show_error=True, server_port=7860)
