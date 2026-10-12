# Uploaded examples → project modules

| Source | Implementation |
|---|---|
| Pasted text.txt | app.py, ui.py, models.py, image_tools/, sections/, utils/, tests/; original directory structure retained |
| streamlit_app.py | Streamlit controls; menus centralized in ui.py |
| Session2_1.py | Image decoding, channel inspection, grayscale, rescale, normalize, blending and background subtraction |
| Session3_1.py | Resize, four blur filters, flips, rotation, sharpening, noise, reproducible random pipeline |
| Session4_1.py | Line, rectangle, highlight, circle, polyline and polygon; video sources |
| Session5_1.py | Text, markers and arrows |
| YOLO_01.py | Video/image object detection, annotated image, crops and CSV export |
| Session_8_streamlit_app_voice_to_text.py | Faster Whisper transcription and Edge TTS |
| Session8_streamlist_fine_tune_mic.py | Browser recording connected to local transcription |
| Session8_streamlit_app_voice_to_text_fine_tune.py | Local Transformers transcription |
| 04_01-Models_Messages_StructuredOutput.ipynb | Model creation and typed reflection outputs |
| 04_02_Memory.ipynb | create_agent, InMemorySaver, thread IDs, recent messages and summarization |
| 04_03_sqllite.ipynb | SQLite access and persistent chat archive |
| 04_04_LCEL_Chains_and_parsers.ipynb | LangChain model calls and structured parsing used by reflection |
| 04_05_Multimodal_Messages.ipynb | Chart review receives the rendered image |
| 05_01_reflection_visual_generation.ipynb | Generate chart → review/run → critique actual image → review/run refinement |
| 05_02_reflection_sql_generation.ipynb | Generate SQL → execute → review actual rows/errors → refine and execute |
| db_helpers.py | utils/db_helpers.py and database.py; notebook UI omitted, generated queries constrained read-only |
| llm_helpers.py | utils/llm_helpers.py; chart data preparation and schema functions used directly |
| coffee_sales.csv, products.db, sample_file.txt | Bundled unchanged in data/ |

Examples were integrated as reusable functions rather than imported wholesale: many original files execute demonstrations, open OpenCV windows or access devices on import. The face-detection notes have no runnable detector and are not exposed as a finished feature. PDF/OCR and legacy Word .doc parsing are not included.

## Noise/filter extension

image_tools/noise.py implements common noise options with NumPy. Gaussian is the supplied generator; impulse noise is mentioned by the median-filter lesson. Uniform, speckle and Poisson are clearly identified additions. Custom convolution exposes the existing filter2D call; Background mask implements the source threshold comment. IMAGE_COVERAGE.md records the full audit.
