# 🐱🐶 Cats vs Dogs AI Web App

An image classifier **and** a working web app — built with PyTorch + Gradio.
This is step 3 of the ML journey: regression (`hello_ ai.py`) →
classifier (`train_my_ai.py`) → **a real, shareable demo** (`app.py`).

## What's inside

| File | What it does |
|---|---|
| `hello_ ai.py` | Tiny starter model: learns Celsius → Fahrenheit |
| `train_my_ai.py` | Fine-tunes MobileNetV2 on `data/train` (cats vs dogs) |
| `evaluate_my_ai.py` | Honest test-set report: accuracy, per-class stats, confusion matrix |
| `app.py` | Gradio web app: upload / webcam / examples + model card |
| `my_ai_model.pth` | The trained model weights |
| `model_info.json` | Auto-generated "model card" used by the web app |
| `data/` | `train/` and `test/` images for cats and dogs |

## Quick start

```bash
# 1. Install dependencies
py -m pip install -r requirements.txt

# 2. (Optional) retrain from scratch
py train_my_ai.py

# 3. Check the honest accuracy numbers
py evaluate_my_ai.py

# 4. Launch the web app
py app.py
```

Then open **http://127.0.0.1:7860** in your browser.

## Features

- **Upload tab** — drag & drop any photo, get instant cat-vs-dog prediction with confidence bars
- **Webcam tab** — snap a photo with your camera and classify it live
- **Examples tab** — one-click predictions on the built-in test images
- **Model card** — real test accuracy and per-class numbers from `evaluate_my_ai.py`

## Current results (test set, 8 images)

- Overall accuracy: **87.5%** (7/8)
- Cats: 75.0% · Dogs: 100.0%
- The one miss: a cat predicted as "dog" with 99.9% confidence — a good
  reminder that confident ≠ correct!
