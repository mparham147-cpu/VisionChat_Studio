# Added calls explained

The project uses Streamlit as its only UI library. Your pasted structure already contains most calls below. Comments above ui.menu and the image widget builders explain their use.

| Call | Meaning and reason |
|---|---|
| st.sidebar.expander / st.expander | Main sections use a collapsible sidebar; submenus use bordered collapsible boxes above the workspace. Parent image menus collapse while editing. |
| st.rerun | Restarts rendering after a menu choice so selection highlighting updates immediately. |
| st.session_state | Keeps UI selections and temporary images across widget reruns. Chat transcripts are stored in SQLite. |
| st.button(type=...) | Highlights the selected menu item using primary styling; other buttons remain secondary and compact. |
| st.multiselect | Enables several model profiles in Settings; model selectors use this saved list. |
| st.container / st.chat_message | Groups chat output so the current response appears above the composer; styles messages by role. |
| st.audio_input | Records browser microphone audio for the existing transcription models. |
| st.cache_resource | Reuses loaded speech/YOLO models between reruns; it does not save them to disk. |
| st.download_button / st.image / st.video / st.audio | Downloads results and previews media inside Streamlit. |
| st.toggle / st.stop | Switches Live preview; ends rendering when validation needs correction. |
| python-docx.Document | Reads DOCX paragraphs and table cells; it cannot read legacy .doc files or OCR scans. |
| SQLite URI mode=ro / set_authorizer | Refuses database write operations in generated SQL; chat_store uses a separate writable archive. |
| SQLite set_progress_handler | Interrupts SQL exceeding the configured step budget. |
| NumPy copy | Takes an independent editing snapshot so repeated previews do not modify their input. |
| cv2.addWeighted | Blends a drawing overlay with the image according to opacity. |
| with_structured_output | Parses model replies into the Pydantic query/chart schemas from your reflection examples. |
| create_agent / InMemorySaver / before_model | Your memory pattern: separate thread IDs and middleware controlling model context. |
| SummarizationMiddleware | Condenses old model context; the complete archived transcript stays unchanged. |

Only provider-labeled summary text is eligible for the optional reasoning-summary display. Reasoning effort is model computation, not internet-search depth. No web-search connector is included.

Chart exec() follows your notebook workflow but requires review of every code version. It runs with normal local Python permissions; the prompt's restrictions are instructions, not a sandbox.


## This revision's menu shape

ui.menu places the main section list in the sidebar. Other menus draw buttons in rows of three equal Streamlit columns within an expander. Empty cells on the final row preserve alignment. Selected buttons use primary styling; all menus call the same renderer.

ui.apply_style uses st.markdown with unsafe_allow_html=True to supply a static CSS style block. The existing Help section already uses this Streamlit facility. This CSS contains no uploaded/user text, loads no external assets and runs no JavaScript. It targets app and download buttons: 180-pixel width, 64-pixel height, rounded corners, wrapped labels and visible keyboard focus. max-width permits narrowing on small screens. Native browser/file-picker controls are not targeted. CSS selectors depend on Streamlit's rendered markup, so a future Streamlit redesign may require adjusting this one helper.

Image parameter controls use two Streamlit columns for a shorter workspace. Reset sets result_image to the original, clears active_effect, changes image_task to Inspect, and calls st.rerun so the user sees the original immediately.


## Additional noise and kernel calls

NumPy RandomState.uniform(low, high, shape) samples bounded equal-probability values. For impulse noise these values decide which whole pixels become white/black. RandomState.poisson(expected_counts) samples nonnegative event counts for the photon simulation. Both use the existing NumPy library and a local seeded stream. normal() is already used by the supplied Gaussian example; speckle uses it multiplicatively instead of additively. None changes the global RNG.

np.asarray turns a parsed coefficient matrix into an array, and np.isfinite rejects NaN/infinity. cv2.filter2D is already in Session3; the new screen exposes its kernel and delta parameters. Custom matrix parsing uses standard string split, float conversion and np.array. NumPy comparisons produce threshold masks. All added functions have parameter/return/behavior docstrings in noise.py and operations.py.

The menu renderer no longer prints selected-option descriptions. Only st.button(help=...) holds those explanations; range/current-value captions are separate numerical feedback, not tooltips left on screen.

## Explicit selection and history update

See IMAGE_HISTORY.md. No first menu item is chosen automatically. A shared Reset/Undo/Redo toolbar appears in every Image task. Undo means return to the state before the last applied effect; Redo restores an undone result. Twenty-one tests pass after this update.
