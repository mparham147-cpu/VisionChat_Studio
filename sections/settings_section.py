"""
sections/settings_section.py
============================

Settings section: conversation limits, model registry, API keys.
"""

import streamlit as st

from ui import menu, number_input
from models import MODELS


def render():
    """Draw the Settings section."""
    task = menu(
        "Settings tasks",
        {
            "Conversation limits": "Input length, history turns, attachments.",
            "Models":              "Which models appear in every chatbot menu.",
            "API keys":            "Provide credentials for this session only.",
        },
        "settings_task",
    )
    if task is None:
        return

    if task == "Conversation limits":
        # Each entry: (session key, label, min, max, default)
        settings_list = [
            ("input_limit",      "Input character limit (0 = no limit)",
             0,    1_000_000, 0),
            ("history_turns",    "Recent conversation turns sent",
             0,    100,      10),
            ("attachment_limit", "Attachment characters per message",
             1000, 1_000_000, 30000),
        ]
        descriptions = {
            "input_limit": "Maximum typed characters per message; 0 means no application limit.",
            "history_turns": "Number of recent user/assistant pairs sent in Recent turns mode. The archive remains complete.",
            "attachment_limit": "Maximum extracted document characters attached to one model request.",
        }
        for key, label, minimum, maximum, default in settings_list:
            st.session_state[key] = number_input(
                label, minimum, maximum,
                st.session_state.get(key, default),
                f"widget_{key}", help_text=descriptions[key],
            )
        st.info(
            "These are application limits, not the model context window. "
            "The provider still enforces its own token limit."
        )

    elif task == "Models":
        # `multiselect` returns a list of the chosen labels.
        currently_enabled = st.session_state.get("enabled_models", list(MODELS))
        chosen = st.multiselect(
            "Models available in the ChatBot menu",
            options=list(MODELS.keys()),
            default=currently_enabled,
            help="Only these appear in model selectors elsewhere in the app.",
        )
        if chosen:
            st.session_state.enabled_models = chosen
            st.success("Model list updated.")
        else:
            st.warning("Keep at least one model enabled.")

    else:
        for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
            value = st.text_input(
                name,
                value=st.session_state.get(name, ""),
                type="password",
                key=f"widget_{name}",
                help="Kept only in this session. A .env file also works.",
            )
            st.session_state[name] = value
        st.caption(
            "API keys are never included in exports. They are cleared "
            "when the server restarts."
        )

