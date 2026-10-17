# Run doc — Cats vs Dogs AI web app (Gradio)

## Reproduce the artifacts

Dependencies (Python 3.14, use the `py` launcher or the full python path):

```
py -m pip install -r requirements.txt
```

Model checkpoint (`my_ai_model.pth`) and model card (`model_info.json`):

- If `my_ai_model.pth` is missing, train it (CPU, ~15 epochs, MobileNetV2 fine-tune):
  ```
  py train_my_ai.py
  ```
- Regenerate the model card used by the web app:
  ```
  py evaluate_my_ai.py
  ```

## Run the server

```
py app.py
```

- Serves on http://127.0.0.1:7860 (fixed via `server_port=7860` in `app.py`).
- Torch import takes ~20-40s before the port opens — wait for it.
- Detached launch (PowerShell; stdout and stderr to DIFFERENT files):
  ```
  powershell -NoProfile -Command "(Start-Process -FilePath 'C:\Users\muham\AppData\Local\Python\pythoncore-3.14-64\python.exe' -ArgumentList 'app.py' -WorkingDirectory 'C:\Users\muham\my ai' -RedirectStandardOutput '<stdout log>' -RedirectStandardError '<stderr log>' -WindowStyle Hidden -PassThru).Id"
  ```
  Do NOT use `py` as the executable — Start-Process cannot resolve the `py` shim.
