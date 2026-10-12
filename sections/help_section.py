"""
sections/help_section.py
========================

Two help documents rendered as HTML:

* Project guide — what every section does, with a table of contents.
* Function dictionary — every public function, its file, and its purpose.
"""

import streamlit as st

from ui import menu
from image_tools.effects_menu import GROUPS, EFFECTS


STYLES = """
<style>
.help-doc h1 { margin-bottom: 0.3em; }
.help-doc h2 { margin-top: 1.2em; }
.help-doc table { border-collapse: collapse; width: 100%; font-size: 14px; }
.help-doc th, .help-doc td {
    border: 1px solid #e5e7eb; padding: 6px 8px;
    text-align: left; vertical-align: top;
}
.help-doc th { background: #f9fafb; }
.help-doc code { background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }
.help-doc .card {
    border: 1px solid #e5e7eb; border-radius: 10px;
    padding: 12px 16px; margin: 10px 0; background: #fbfbfd;
}
.help-doc a { text-decoration: none; color: #2563eb; }
.help-doc a:hover { text-decoration: underline; }
</style>
"""


def build_project_guide_html():
    """Build the full project guide with one bullet list per effect group."""
    effect_lists = []
    for group_name, group_help in GROUPS.items():
        effect_lists.append(f"<li><strong>{group_name}</strong> — {group_help}<ul>")
        for effect_name, definition in EFFECTS.items():
            if definition["group"] == group_name:
                effect_lists.append(
                    f"<li><code>{effect_name}</code> — {definition['help']}</li>"
                )
        effect_lists.append("</ul></li>")
    effects_section = "\n".join(effect_lists)

    return STYLES + f"""
<div class="help-doc">

<h1>VisionChat Studio — Project Guide</h1>

<p>
A single Streamlit app that gathers image editing, video detection,
speech conversion, and AI chat with document context.
</p>

<div class="card">
<strong>Contents</strong>
<ul>
  <li><a href="#layout">Directory layout</a></li>
  <li><a href="#run">Install and run</a></li>
  <li><a href="#image">Image section</a></li>
  <li><a href="#video">Video section</a></li>
  <li><a href="#conversion">Conversion section</a></li>
  <li><a href="#chatbot">ChatBot section</a></li>
  <li><a href="#files">File section</a></li>
  <li><a href="#settings">Settings section</a></li>
  <li><a href="#extending">Extending the project</a></li>
  <li><a href="#safety">Safety notes</a></li>
</ul>
</div>

<h2 id="layout">Directory layout</h2>
<table>
<tr><th>Path</th><th>What it does</th></tr>
<tr><td><code>app.py</code></td><td>Top menu and section dispatch.</td></tr>
<tr><td><code>ui.py</code></td><td>Shared Streamlit widgets.</td></tr>
<tr><td><code>models.py</code></td><td>Model registry and constructor.</td></tr>
<tr><td><code>image_tools/operations.py</code></td><td>Pure image functions.</td></tr>
<tr><td><code>image_tools/drawing.py</code></td><td>Pure drawing functions.</td></tr>
<tr><td><code>image_tools/effects_menu.py</code></td><td>Registry of every effect.</td></tr>
<tr><td><code>sections/*.py</code></td><td>One file per top-level section.</td></tr>
<tr><td><code>utils/file_io.py</code></td><td>Read and encode files.</td></tr>
<tr><td><code>utils/database.py</code></td><td>Read-only SQLite access.</td></tr>
<tr><td><code>utils/chat_store.py</code></td><td>Chat archive in SQLite.</td></tr>
<tr><td><code>utils/chat_agent.py</code></td><td>Chat agent with memory strategies.</td></tr>
</table>

<h2 id="run">Install and run</h2>
<pre><code>python -m venv .venv
.venv\\Scripts\\activate       # Windows
source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
python -m streamlit run app.py</code></pre>

<p>Main sections are in the sidebar. Task menus appear in bordered boxes above the workspace; click their headings to open or close them. Buttons use a consistent size. Reset restores the original. Undo restores the preceding edit; Redo restores an undone result.</p>
<h2 id="image">Image section</h2>
<p>The Noise group includes Gaussian, salt, pepper, salt-and-pepper, uniform, speckle and Poisson. Custom convolution accepts your coefficients; Random pipeline exposes each step’s chance and settings. Menu explanations are hover-only. Full operation descriptions are listed below.</p>
<p>
Every effect works on a fixed "base" snapshot. Changing parameters
recomputes from that snapshot, so effects never stack accidentally.
Selecting a new effect uses the latest result as the new base, which
lets you chain unsaved edits.
</p>
<table>
<tr><th>Task</th><th>Purpose</th></tr>
<tr><td>Upload</td><td>Load an image from disk.</td></tr>
<tr><td>Blank canvas</td><td>Create a black image for drawing practice.</td></tr>
<tr><td>Inspect</td><td>Show shape, dtype, range, memory size.</td></tr>
<tr><td>Effects</td><td>Groups below.</td></tr>
<tr><td>Save</td><td>Download the last result.</td></tr>
</table>
<h3>Effect groups</h3>
<ul>{effects_section}</ul>

<h2 id="video">Video section</h2>
<p>Preview a source, or run YOLO detection on video or on a still image.</p>
<table>
<tr><th>Task</th><th>Purpose</th></tr>
<tr><td>Preview</td><td>Frame-by-frame playback without detection.</td></tr>
<tr><td>Detect in video</td><td>YOLO boxes and labels on every frame.</td></tr>
<tr><td>Detect in image</td><td>YOLO on a still image plus CSV and crops.</td></tr>
</table>

<h2 id="conversion">Conversion section</h2>
<table>
<tr><th>Task</th><th>Purpose</th></tr>
<tr><td>Text to speech</td><td>Persian voice via Edge TTS.</td></tr>
<tr><td>Audio file to text</td><td>Transcribe a file with a local model.</td></tr>
<tr><td>Microphone to text</td><td>Record in the browser, then transcribe.</td></tr>
</table>

<h2 id="chatbot">ChatBot section</h2>
<table>
<tr><th>Task</th><th>Purpose</th></tr>
<tr><td>Conversation</td><td>Chat with attachments and persistent history.</td></tr>
<tr><td>SQL reflection</td><td>Generate, execute, reflect, refine.</td></tr>
<tr><td>Chart reflection</td><td>Generate chart code, review, run, reflect.</td></tr>
</table>

<h2 id="files">File section</h2>
<p>Preview documents, export conversations, download sample data,
clear session workspace.</p>

<h2 id="settings">Settings section</h2>
<p>Conversation limits, model registry, API keys.</p>

<h2 id="extending">Extending the project</h2>
<h3>Add a new image effect</h3>
<ol>
<li>Write the function in <code>image_tools/operations.py</code> or
    <code>image_tools/drawing.py</code>.</li>
<li>Add one entry to <code>image_tools/effects_menu.py</code>.</li>
</ol>
<h3>Add a whole new section</h3>
<ol>
<li>Create <code>sections/my_section.py</code> with a
    <code>render()</code> function.</li>
<li>Add one entry to <code>SECTIONS</code> in <code>app.py</code>.</li>
</ol>
<h3>Add a new model</h3>
<ol>
<li>Add one entry to <code>MODELS</code> in <code>models.py</code>.</li>
</ol>

<h2 id="safety">Safety notes</h2>
<ul>
<li>SQL reflection opens the database read-only.</li>
<li>Chart reflection runs Python you approved; it is not sandboxed.</li>
<li>No PDF or legacy .doc parsing is included.</li>
<li>API keys are never exported.</li>
</ul>

</div>
"""


FUNCTION_DICTIONARY_HTML = STYLES + """
<div class="help-doc">
<h1>Function dictionary</h1>
<p>Every public function, grouped by file.</p>

<h2>ui.py</h2>
<table>
<tr><th>Function</th><th>Purpose</th></tr>
<tr><td><code>menu</code></td><td>Sidebar section buttons and collapsible top submenu grids; remembers selection.</td></tr>
<tr><td><code>number_input</code></td><td>Bounded number input with error check.</td></tr>
<tr><td><code>slider_input</code></td><td>Bounded slider showing value and range.</td></tr>
<tr><td><code>show_image</code></td><td>Display BGR/BGRA/gray arrays correctly.</td></tr>
</table>

<h2>models.py</h2>
<table>
<tr><th>Function</th><th>Purpose</th></tr>
<tr><td><code>get_enabled_model_names</code></td><td>List models enabled in Settings.</td></tr>
<tr><td><code>model_selector</code></td><td>Draw model + effort pickers.</td></tr>
<tr><td><code>build_chat_model</code></td><td>Instantiate a LangChain chat model.</td></tr>
<tr><td><code>extract_answer_text</code></td><td>Get visible text from a response.</td></tr>
</table>

<h2>image_tools/operations.py</h2>
<table>
<tr><th>Function</th><th>Purpose</th></tr>
<tr><td><code>convert_to_grayscale</code></td><td>BGR → gray.</td></tr>
<tr><td><code>convert_to_reduced_grayscale</code></td><td>Gray and 8× smaller.</td></tr>
<tr><td><code>extract_channel</code></td><td>Return B, G, R or alpha.</td></tr>
<tr><td><code>rescale_to_unit_range</code></td><td>Divide by 255.</td></tr>
<tr><td><code>normalize_to_unit_range</code></td><td>Stretch range to 0–1.</td></tr>
<tr><td><code>denormalize_to_bytes</code></td><td>Normalize then return bytes.</td></tr>
<tr><td><code>resize_image</code></td><td>Change dimensions.</td></tr>
<tr><td><code>flip_image</code></td><td>Horizontal, vertical or both.</td></tr>
<tr><td><code>rotate_image</code></td><td>Rotate with optional canvas expansion.</td></tr>
<tr><td><code>apply_mean_blur</code></td><td>cv2.blur.</td></tr>
<tr><td><code>apply_gaussian_blur</code></td><td>cv2.GaussianBlur.</td></tr>
<tr><td><code>apply_median_blur</code></td><td>cv2.medianBlur.</td></tr>
<tr><td><code>apply_bilateral_blur</code></td><td>cv2.bilateralFilter.</td></tr>
<tr><td><code>apply_sharpen</code></td><td>3×3 sharpening filter.</td></tr>
<tr><td><code>apply_gaussian_noise</code></td><td>Reproducible Gaussian noise.</td></tr>
<tr><td><code>apply_random_pipeline</code></td><td>Random combination of the lesson filters.</td></tr>
<tr><td><code>blend_two_images</code></td><td>Weighted add of two images.</td></tr>
<tr><td><code>subtract_background</code></td><td>Subtract a background image.</td></tr>
</table>

<h2>image_tools/drawing.py</h2>
<table>
<tr><th>Function</th><th>Purpose</th></tr>
<tr><td><code>draw_line</code></td><td>Straight segment.</td></tr>
<tr><td><code>draw_rectangle</code></td><td>Rectangle outline or fill.</td></tr>
<tr><td><code>draw_filled_highlight</code></td><td>Translucent rectangle.</td></tr>
<tr><td><code>draw_circle</code></td><td>Circle outline or fill.</td></tr>
<tr><td><code>draw_polyline</code></td><td>Connect points.</td></tr>
<tr><td><code>draw_filled_polygon</code></td><td>Fill polygon.</td></tr>
<tr><td><code>draw_text</code></td><td>Latin text.</td></tr>
<tr><td><code>draw_marker</code></td><td>Marker shape.</td></tr>
<tr><td><code>draw_arrow</code></td><td>Arrow with adjustable tip.</td></tr>
</table>

<h2>utils/*</h2>
<table>
<tr><th>Function</th><th>Purpose</th></tr>
<tr><td><code>file_io.decode_image</code></td><td>Bytes → NumPy image.</td></tr>
<tr><td><code>file_io.encode_image</code></td><td>NumPy image → bytes.</td></tr>
<tr><td><code>file_io.read_text</code></td><td>Read UTF-8 or DOCX.</td></tr>
<tr><td><code>database.open_readonly_connection</code></td><td>Read-only SQLite.</td></tr>
<tr><td><code>database.get_schema_text</code></td><td>Return CREATE TABLE text.</td></tr>
<tr><td><code>database.run_readonly_query</code></td><td>Run one read-only statement.</td></tr>
<tr><td><code>chat_store.open_connection</code></td><td>Open chat database.</td></tr>
<tr><td><code>chat_store.create_conversation</code></td><td>New conversation id.</td></tr>
<tr><td><code>chat_store.list_conversations</code></td><td>List saved chats.</td></tr>
<tr><td><code>chat_store.load_messages</code></td><td>Load full transcript.</td></tr>
<tr><td><code>chat_store.save_turn</code></td><td>Save Q + A atomically.</td></tr>
<tr><td><code>chat_agent.build_agent</code></td><td>Build a LangChain agent.</td></tr>
<tr><td><code>chat_agent.send_message</code></td><td>Rebuild from history and invoke.</td></tr>
</table>

<h2>sections/*</h2>
<table>
<tr><th>Module</th><th>render() draws</th></tr>
<tr><td><code>image_section</code></td><td>Image editing screen.</td></tr>
<tr><td><code>video_section</code></td><td>Video preview and YOLO detection.</td></tr>
<tr><td><code>conversion_section</code></td><td>Speech conversion.</td></tr>
<tr><td><code>chatbot_section</code></td><td>Chat and reflection workflows.</td></tr>
<tr><td><code>sql_reflection_section</code></td><td>SQL workflow screen.</td></tr>
<tr><td><code>chart_reflection_section</code></td><td>Chart workflow screen.</td></tr>
<tr><td><code>files_section</code></td><td>File tools.</td></tr>
<tr><td><code>help_section</code></td><td>This help page.</td></tr>
<tr><td><code>settings_section</code></td><td>Limits, models, keys.</td></tr>
</table>
</div>
"""


def render():
    """Draw the Help section."""
    task = menu(
        "Help topics",
        {
            "Project guide":     "Overview and explanation of every section.",
            "Function lookup":   "Every public function with its purpose.",
        },
        "help_task",
    )
    if task is None:
        return
    if task == "Project guide":
        st.markdown(build_project_guide_html(), unsafe_allow_html=True)
    else:
        st.markdown(FUNCTION_DICTIONARY_HTML, unsafe_allow_html=True)
