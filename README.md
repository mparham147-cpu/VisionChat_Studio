# VisionChat Studio

An interactive studio for image editing, video object
detection, speech conversion, and AI chat with document context — all
in one Streamlit application. Every feature is designed to be read,
understood and extended by a single developer without any hidden
plumbing.

## Section overview

| Section    | What it does                                                        |
|------------|---------------------------------------------------------------------|
| Image      | Upload an image, apply one effect at a time, preview it live, save  |
| Video      | Preview a video, run YOLO object detection, export crops and CSVs   |
| Conversion | Text → speech (Edge TTS) and speech → text (Faster-Whisper)         |
| ChatBot    | Chat with an LLM (with attachments), plus two reflection workflows  |
| File       | Inspect documents, export conversations, download sample data       |
| Help       | Full in-app documentation: guide and function dictionary            |
| Settings   | Input limits, model registry, API keys                              |

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python -m streamlit run app.py
```

A browser opens at `http://localhost:8501`.

## API keys

Put them in a file named `.env` next to `app.py`:

```
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=...
```

Or type them at runtime under **Settings → API keys**.

## Local model files

Nothing is bundled. Provide paths in the UI:

| Purpose              | Suggested path                     | Where to get it                          |
|----------------------|------------------------------------|------------------------------------------|
| YOLO detection       | `models/yolov8n.pt`                | Ultralytics download                     |
| Faster-Whisper       | `models/faster-whisper-small`      | HuggingFace `Systran/faster-whisper-small` |
| Whisper (HF pipeline)| `models/whisper-persian-v4`        | Your fine-tuned model                    |

## How to add a new image effect

1. Open `image_tools/operations.py`.
2. Add a new function named after the effect, taking the image plus the
   parameters the user should control. Example:

   ```python
   def apply_posterize(image, levels):
       """Reduce each channel to `levels` distinct intensities."""
       factor = max(1, 256 // levels)
       return (image // factor) * factor
   ```

3. Open `image_tools/effects_menu.py`.
4. Add one entry to the `EFFECTS` dictionary (shown below). That is the
   only place the effect must be registered.

```python
EFFECTS = {
    "Posterize": {
        "group": "Color and values",
        "help": "Reduce the number of intensity levels.",
        "function": apply_posterize,
        "parameters": [
            {"type": "slider", "name": "levels", "min": 2, "max": 64, "default": 8,
             "help": "Number of discrete brightness levels per channel."},
        ],
    },
    # ... existing entries
}
```

The image screen picks the effect, draws one widget per parameter,
runs the function, and shows the result — automatically.

## How to add a whole new top-level section

1. Create `sections/my_section.py` with a single `render()` function.
2. Open `app.py` and add one line to the `SECTIONS` dictionary.
3. Done. The left menu picks it up on the next page load.

## How to add a new model

Open `models.py`, add one entry to `MODELS`:

```python
"Claude Sonnet 4.6": {
    "id": "claude-sonnet-4-6",
    "provider": "anthropic",
    "reasoning_efforts": [],
    "vision": True,
},
```

If the provider is not OpenAI or DeepSeek you may need to adjust
`build_chat_model()` in the same file — every provider-specific line
is at the top of that function.

## Notes on safety

- **SQL reflection** opens the target database read-only. A SQLite
  authorizer blocks every write attempt. Generated SQL correctness is
  not guaranteed.
- **Chart reflection** executes Python you have read and approved. It is
  **not sandboxed**. Only run it on your own machine.
- **Attachments** are decoded locally. No PDF, OCR, or legacy `.doc`
  parsing is included.
- **API keys** never leave the process except as part of the request to
  the provider you selected.


## PyCharm setup

Use Python 3.11 or 3.12 for broad compatibility with the speech and vision dependencies. Extract this ZIP, then open the **VisionChat_Studio** folder itself in PyCharm. Select a project virtual environment. Run these commands in the terminal whose working directory contains app.py:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For a PyCharm Run configuration, select **Python → Module name: streamlit**, use **Parameters: run app.py**, and set the working directory to VisionChat_Studio. An ordinary Python run of app.py will not launch the application correctly.

The full requirements install speech and detection dependencies. For an initial image/chat/document setup, install requirements-core.txt instead; install the full list before using speech or YOLO. FFmpeg may be needed for compressed audio; install it on Windows and add its bin folder to PATH before restarting PyCharm. Model weights, API credentials and generated chat histories are not bundled.

## Navigation and editing

Main sections use a collapsible sidebar. Submenus appear in ordered bordered boxes above the workspace and can be opened or closed. Parent image task/group menus start collapsed while editing. Application buttons share the same size and rounded styling. Hover over buttons for descriptions. Settings → Models controls all chatbot selectors. Input length defaults to no app limit and can be changed in Settings.

Each image effect uses a fixed starting snapshot. Parameter changes replace its preview rather than stacking. Selecting another effect uses the latest unsaved result. Start a new edit repeats the same effect on its current result. The common Reset button restores the original and clears effect selection. Undo restores the preceding image; Redo restores the undone result. Random seed controls repeatability. Mean and Gaussian blur are distinct filters.

SQLite saves full chat transcripts and extracted attachment text. The notebook's create_agent/InMemorySaver pattern rebuilds model memory from that archive on each request. Recent turns, full history and summarization change the context sent to the model, not the saved transcript. Summarization may make additional calls and is rebuilt as needed. Only provider-designated reasoning summaries can be displayed; raw internal thinking is not shown. These sessions have no user authentication.

## Supplied helpers and examples

utils/db_helpers.py includes your transaction database generator and read-only adapters for schema/query helpers. utils/llm_helpers.py retains your data preparation, schema, code-tag and image-encoding functions. Notebook-only IPython display helpers were omitted because Streamlit is the only UI. SOURCE_MAP.md maps the uploaded examples to application modules. NEW_CALLS.md explains added calls and behavior.

## Verification

```bash
python -m unittest discover -s tests -v
```

Unit tests do not contact providers or load speech/YOLO weights. TEST_REPORT.md records the checks performed for this package and the remaining external checks.

## Request review

REQUEST_REVIEW.md maps every earlier requirement to the current implementation and explicitly lists remaining limitations. Menu and action styling is centralized in ui.py. The browser preview could not access the local server here, so rendered appearance has not been visually verified.


## Noise and filter update

Image → Effects → Noise contains Gaussian, Salt, Pepper, Salt and pepper, Uniform, Speckle and Poisson. All expose their distribution settings and seed. IMAGE_COVERAGE.md identifies supplied implementations versus extensions. No new library is needed.

Detail and noise → Custom convolution accepts coefficients such as `0,-1,0; -1,5,-1; 0,-1,0`. Random pipeline exposes its step chances and filter parameters. Blur kernels support odd 1–31 values. Binary threshold and Background mask expose intensity/difference cutoffs. Highlight offers Blend and Lesson additive modes.

Descriptions of menu buttons are hover-only. Ranges and current numeric values stay visible; full explanations remain in Help and docstrings. Eighteen tests pass after this update; all 36 registered effects were exercised. Browser-rendered tooltip/CSS appearance remains unverified because the local browser preview was blocked previously.

## Explicit selection and history update

See IMAGE_HISTORY.md. No first menu item is chosen automatically. A shared Reset/Undo/Redo toolbar appears in every Image task. Undo means return to the state before the last applied effect; Redo restores an undone result. Twenty-one tests pass after this update.
