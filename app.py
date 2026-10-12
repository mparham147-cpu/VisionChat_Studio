"""
app.py
======

VisionChat Studio — top-level entry point.

Run with:

    python -m streamlit run app.py

Each section module exposes a ``render()`` function. This file loads
``.env`` (for API keys), draws the top-level menu, and calls the chosen
section's ``render()``.

To add a new section:
    1. Create ``sections/my_section.py`` with a ``render()`` function.
    2. Add one entry to the SECTIONS dictionary below.
"""

import streamlit as st
from dotenv import load_dotenv

from ui import menu, apply_style
from sections import (
    image_section,
    video_section,
    conversion_section,
    chatbot_section,
    files_section,
    help_section,
    settings_section,
)


SECTIONS = {
    "Image":      (image_section,      "Edit images with OpenCV."),
    "Video":      (video_section,      "Preview videos and run YOLO detection."),
    "Conversion": (conversion_section, "Text to speech and speech to text."),
    "ChatBot":    (chatbot_section,    "Chat with attachments, SQL and chart reflection."),
    "File":       (files_section,      "Inspect documents and export data."),
    "Help":       (help_section,       "Documentation for the whole app."),
    "Settings":   (settings_section,   "Limits, models and API keys."),
}


def main():
    """Configure the page, draw the menu, and dispatch to a section."""
    load_dotenv()
    st.set_page_config(page_title="VisionChat Studio", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
    apply_style()
    st.session_state.setdefault("section", None)
    st.sidebar.title("VisionChat Studio")
    st.sidebar.caption("Choose a section. Its task menus appear above the workspace.")

    selected_section = menu(
        "Sections",
        {name: description for name, (_, description) in SECTIONS.items()},
        "section",
    )

    if selected_section is None:
        st.title("VisionChat Studio")
        st.info("Choose a section from the sidebar to begin.")
        return
    st.title(selected_section)
    section_module = SECTIONS[selected_section][0]
    try:
        section_module.render()
    except ImportError as error:
        st.error(f"Missing optional dependency: {error}. See README.md.")
    except Exception as error:
        st.error(f"The task could not complete: {error}")
        st.caption("Check inputs, settings and model files, then try again.")


if __name__ == "__main__":
    main()
