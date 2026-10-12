"""
sections/conversion_section.py
==============================

Speech conversion: text to speech, and speech to text.

Two speech engines are supported:

* Faster Whisper (CTranslate2, CPU int8)
* Transformers (Hugging Face pipeline)

The model is cached with ``st.cache_resource`` so it is only loaded
once per session.
"""

from pathlib import Path
import tempfile

import streamlit as st

from ui import menu
from utils.file_io import read_text


@st.cache_resource
def load_speech_model(engine, model_path):
    """
    Load and cache a local speech model.

    ``engine`` must be "Faster Whisper" or "Transformers".
    """
    if not Path(model_path).is_dir():
        raise ValueError(f"Model directory not found: {model_path}")

    if engine == "Faster Whisper":
        from faster_whisper import WhisperModel
        return WhisperModel(model_path, device="cpu", compute_type="int8")

    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=model_path)


# ---------------------------------------------------------------------------
# Text to speech
# ---------------------------------------------------------------------------

def text_to_speech_task():
    """Text → downloadable MP3 using Edge TTS."""
    uploaded_file = st.file_uploader("Optional text or Word file",
                                     type=["txt", "docx"], help='Load text from a UTF-8 text file or DOCX document.')
    initial_text = read_text(uploaded_file) if uploaded_file else ""
    text = st.text_area("Text to read aloud", initial_text, height=200, help='Type or edit the text that will be sent to the speech service.')

    voice = st.selectbox(
        "Voice",
        ["fa-IR-DilaraNeural", "fa-IR-FaridNeural"],
        help="Dilara is female, Farid is male.",
    )
    rate = st.slider("Rate (%)", -50, 50, 0,
                     help="Speed relative to normal. Negative values slow down.")
    pitch = st.slider("Pitch (Hz)", -50, 50, 0,
                      help="Raise or lower the pitch relative to normal.")
    volume = st.slider("Volume (%)", -50, 50, 0,
                       help="Change loudness relative to normal.")

    st.caption("Edge TTS needs internet access. The text is sent to the speech service.")

    if st.button("Convert to speech", help='Generate an MP3 using the chosen voice, speed, pitch and volume.'):
        if not text.strip():
            st.error("Enter text first.")
            return
        from edge_tts.communicate import Communicate
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "speech.mp3"
            communicate = Communicate(
                text=text,
                voice=voice,
                rate=f"{rate:+d}%",
                pitch=f"{pitch:+d}Hz",
                volume=f"{volume:+d}%",
            )
            communicate.save_sync(str(output_path))
            st.session_state.speech_bytes = output_path.read_bytes()

    if "speech_bytes" in st.session_state:
        st.audio(st.session_state.speech_bytes, format="audio/mp3")
        st.download_button("Save MP3", st.session_state.speech_bytes, "speech.mp3", help='Download the last generated speech recording.')


# ---------------------------------------------------------------------------
# Speech to text
# ---------------------------------------------------------------------------

def speech_to_text_task(use_microphone):
    """Audio → transcript using a local speech model."""
    engine = menu(
        "Speech engine",
        {
            "Faster Whisper": "CPU int8 transcription (CTranslate2).",
            "Transformers":   "Hugging Face Whisper model, including Persian fine-tunes.",
        },
        "speech_engine",
    )
    if engine is None:
        return
    default_model_path = (
        "models/faster-whisper-small" if engine == "Faster Whisper"
        else "models/whisper-persian-v4"
    )
    model_path = st.text_input("Local model directory", default_model_path, key="speech_path_" + engine,
        help="Existing local model folder for the selected transcription engine.")
    language = st.text_input(
        "Language code", "fa",
        help="Example: fa for Persian, en for English. Leave empty for autodetect.",
    )

    if use_microphone:
        audio = st.audio_input(
            "Record from microphone",
            help="The browser asks for microphone permission the first time.",
        )
    else:
        audio = st.file_uploader("Audio file",
                                 type=["wav", "mp3", "m4a", "ogg"], help='Upload an audio recording for transcription with a local model.')

    if audio is not None:
        st.audio(audio.getvalue())

    if st.button("Transcribe", help='Run the chosen local speech model on the uploaded or recorded audio.'):
        if audio is None:
            st.error("Record or upload audio first.")
            return

        with st.spinner("Transcribing…"):
            model = load_speech_model(engine, model_path)
            if engine == "Faster Whisper":
                audio.seek(0)
                segments, _ = model.transcribe(audio, language=language.strip() or None)
                transcript = " ".join(segment.text for segment in segments)
            else:
                kwargs = {"language": language.strip()} if language.strip() else {}
                result = model(audio.getvalue(),
                               return_timestamps=True,
                               generate_kwargs=kwargs)
                transcript = result["text"]

        st.session_state.transcript = transcript

    if "transcript" in st.session_state:
        st.text_area("Transcript", st.session_state.transcript, height=200, help='Review or copy the last generated transcription.')
        st.download_button("Save transcript",
                           st.session_state.transcript, "transcript.txt", help='Download the last generated transcription as a text file.')


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def render():
    """Draw the Conversion section."""
    task = menu(
        "Conversion tasks",
        {
            "Text to speech":      "Read typed text or a document aloud; save MP3.",
            "Audio file to text":  "Transcribe an uploaded audio file.",
            "Microphone to text":  "Record in the browser, then transcribe.",
        },
        "conversion_task",
    )
    if task is None:
        return

    if task == "Text to speech":
        text_to_speech_task()
    elif task == "Audio file to text":
        speech_to_text_task(use_microphone=False)
    else:
        speech_to_text_task(use_microphone=True)
