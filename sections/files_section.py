"""
sections/files_section.py
=========================

File section: preview documents, export the current conversation,
download sample data, clear the session workspace.
"""

import json

import streamlit as st

from ui import menu
from utils.file_io import read_text, PROJECT_ROOT
from utils.chat_store import load_messages


def render():
    """Draw the File section."""
    task = menu(
        "File tasks",
        {
            "Preview document":    "Read a text or DOCX file locally.",
            "Export conversation": "Download the current chat as JSON or text.",
            "Sample data":         "Download the bundled CSV, DB and sample text.",
            "Clear workspace":     "Remove temporary session results.",
        },
        "file_task",
    )
    if task is None:
        return

    if task == "Preview document":
        uploaded = st.file_uploader("Document",
                                    type=["txt", "md", "csv", "json", "py", "docx"], help='Upload a supported text or DOCX document for local extraction.')
        if uploaded:
            text = read_text(uploaded)
            st.write({
                "filename": uploaded.name,
                "bytes": uploaded.size,
                "extracted_characters": len(text),
            })
            st.text_area("Extracted text", text, height=400, help='Preview the decoded document. This panel does not edit the original file.')
            st.download_button("Save extracted text", text, "extracted.txt", help='Download the extracted paragraphs and table text as UTF-8 text.')

    elif task == "Export conversation":
        thread_id = st.session_state.get("active_thread")
        if not thread_id:
            st.info("No conversation has been opened yet.")
            return
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in load_messages(thread_id)
        ]
        st.download_button(
            "Download JSON",
            json.dumps(messages, ensure_ascii=False, indent=2),
            "conversation.json",
            "application/json",
         help='Export user-visible messages from the selected conversation in JSON format.')
        st.download_button(
            "Download plain text",
            "\n\n".join(f"{m['role']}: {m['content']}" for m in messages),
            "conversation.txt",
         help='Export user-visible messages from the selected conversation as plain text.')

    elif task == "Sample data":
        for filename in ("coffee_sales.csv", "products.db", "sample_file.txt"):
            file_path = PROJECT_ROOT / "data" / filename
            if file_path.exists():
                st.download_button(
                    f"Download {filename}",
                    file_path.read_bytes(),
                    filename,
                 help='Download this result or sample file to your browser download folder.')

    else:
        st.caption(
            "Clears loaded images, chart/SQL results, audio, and other "
            "session results. Saved chats, settings and API keys remain."
        )
        if st.button("Clear session workspace", help='Clear temporary media and reflection results while keeping saved chats and settings.'):
            keys_to_keep = {
                "section", "file_task", "enabled_models", "active_thread",
                "input_limit", "history_turns", "attachment_limit",
                "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
            }
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.success("Session workspace cleared.")
