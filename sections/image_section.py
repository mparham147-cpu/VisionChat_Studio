"""
sections/image_section.py
=========================

The Image section.

Five tasks, chosen from the left menu:

    Upload       - load a file from the user's computer
    Blank canvas - create a black image for drawing practice
    Inspect      - show image metadata
    Effects      - pick an effect, set parameters, preview live
    Save         - download the current result

The Effects task reads every available effect from
``image_tools/effects_menu.py``. It knows nothing about the effects
themselves: it just walks the parameters and draws widgets.

Adding a new effect means editing only
``image_tools/effects_menu.py``; this file stays untouched.
"""

import cv2
import numpy as np
import streamlit as st

from ui import menu, number_input, show_image
from utils.file_io import decode_image, encode_image
from image_tools.effects_menu import GROUPS, EFFECTS
from utils.image_history import initialize_history, record_edit, undo_image, redo_image, reset_image


# ---------------------------------------------------------------------------
# Widget builders (one per parameter type)
# ---------------------------------------------------------------------------

def build_number_widget(parameter, widget_key):
    """Draw a bounded number input and return its current value."""
    return number_input(
        label=parameter["name"].replace("_", " ").title(),
        minimum=parameter["min"],
        maximum=parameter["max"],
        default=parameter["default"],
        key=widget_key,
        help_text=parameter.get("help", ""),
    )


def build_slider_widget(parameter, widget_key):
    """Draw a slider and return its current value."""
    st.caption(f"Min {parameter['min']}, max {parameter['max']}")
    step = parameter.get("step", 1)
    value = st.slider(
        parameter["name"].replace("_", " ").title(),
        min_value=parameter["min"],
        max_value=parameter["max"],
        value=parameter["default"],
        step=step,
        key=widget_key,
        help=parameter.get("help", ""),
    )
    st.write(f"Current value: {value}")
    return value


def build_select_widget(parameter, widget_key):
    """Draw a selectbox and return the *value* (not the label)."""
    labels_to_values = parameter["options"]
    default_label = parameter["default"]
    chosen_label = st.selectbox(
        parameter["name"].replace("_", " ").title(),
        options=list(labels_to_values.keys()),
        index=list(labels_to_values.keys()).index(default_label),
        key=widget_key,
        help=parameter.get("help", ""),
    )
    return labels_to_values[chosen_label]


def build_checkbox_widget(parameter, widget_key):
    """Draw a checkbox and return its boolean value."""
    return st.checkbox(
        parameter["name"].replace("_", " ").title(),
        value=parameter["default"],
        key=widget_key,
        help=parameter.get("help", ""),
    )


def build_color_widget(parameter, widget_key):
    """Draw a color picker and return a (Blue, Green, Red) tuple."""
    hex_color = st.color_picker(
        parameter["name"].replace("_", " ").title(),
        value=parameter["default"],
        key=widget_key,
        help=parameter.get("help", ""),
    )
    # Hex is '#RRGGBB'. OpenCV expects BGR.
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    return (blue, green, red)


def build_file_widget(parameter, widget_key):
    """Draw a file uploader and return the decoded image (or None)."""
    uploaded = st.file_uploader(
        parameter["name"].replace("_", " ").title(),
        type=parameter["types"],
        key=widget_key,
        help=parameter.get("help", ""),
    )
    if uploaded is None:
        st.info("Upload a file to continue.")
        st.stop()
    return decode_image(uploaded.getvalue())[:, :, :3]


def build_text_widget(parameter, widget_key):
    """Draw a single-line text input and return its value."""
    return st.text_input(
        parameter["name"].replace("_", " ").title(),
        value=parameter["default"],
        key=widget_key,
        help=parameter.get("help", ""),
    )


def build_textarea_widget(parameter, widget_key):
    """Draw a multi-line text input and return its value."""
    return st.text_area(
        parameter["name"].replace("_", " ").title(),
        value=parameter["default"],
        key=widget_key,
        help=parameter.get("help", ""),
        height=80,
    )


WIDGET_BUILDERS = {
    "number": build_number_widget,
    "slider": build_slider_widget,
    "select": build_select_widget,
    "checkbox": build_checkbox_widget,
    "color": build_color_widget,
    "file": build_file_widget,
    "text": build_text_widget,
    "textarea": build_textarea_widget,
}


# ---------------------------------------------------------------------------
# Parameter collection
# ---------------------------------------------------------------------------

def collect_parameters(effect_definition, effect_name, image=None):
    """
    Draw the widget for every parameter of the chosen effect and return
    a dict of {parameter_name: value}.
    """
    values = {}
    cells = st.columns(2, gap="medium")
    for index, definition in enumerate(effect_definition["parameters"]):
        parameter = definition.copy()
        # Coordinate bounds follow the current unsaved image dimensions.
        if image is not None and parameter["name"] in ("x", "x1", "x2", "center_x", "y", "y1", "y2", "center_y"):
            axis = 1 if "x" in parameter["name"] else 0
            parameter["max"] = image.shape[axis] - 1
            parameter["default"] = min(parameter["default"], parameter["max"])
        widget_key = f"{effect_name}::{parameter['name']}"
        if parameter["type"] == "number" and widget_key in st.session_state:
            current = st.session_state[widget_key]
            if current > parameter["max"] or current < parameter["min"]:
                st.session_state.pop(widget_key, None)
        builder = WIDGET_BUILDERS[parameter["type"]]
        with cells[index % 2]:
            values[parameter["name"]] = builder(parameter, widget_key)
    return values


def parse_points_string(text):
    """Parse 'x,y; x,y; x,y' into a list of (x, y) integer tuples."""
    points = []
    for pair in text.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_text, y_text = pair.split(",")
        points.append((int(x_text.strip()), int(y_text.strip())))
    if len(points) < 3:
        raise ValueError("Enter at least three points as x,y; x,y; x,y.")
    return points


def parse_kernel(text):
    """Parse semicolon-separated rows into an odd, finite convolution matrix.

    Example: '0,-1,0; -1,5,-1; 0,-1,0'. Rows must have equal length;
    dimensions must be odd and at most 31. Signed and decimal entries are
    supported. Bad values raise ValueError for the existing error panel.
    """
    try:
        rows = [[float(value.strip()) for value in row.split(',')]
                for row in text.split(';')]
        kernel = np.array(rows, dtype=np.float32)
    except (ValueError, TypeError):
        raise ValueError('Use equal-length numeric rows: 0,-1,0; -1,5,-1; 0,-1,0.')
    if kernel.ndim != 2 or any(size < 1 or size > 31 or size % 2 == 0 for size in kernel.shape):
        raise ValueError('Kernel width and height must be odd, from 1 to 31.')
    if not np.isfinite(kernel).all():
        raise ValueError('Kernel entries must be finite numbers.')
    return kernel


def convert_parameters_for_function(effect_name, raw_parameters):
    """
    Translate the raw UI values into the argument shape the effect
    function expects. Only two effects need extra translation:
    polyline and polygon both take a parsed list of points.
    """
    if effect_name == "Custom convolution":
        raw_parameters["kernel"] = parse_kernel(raw_parameters.pop("kernel_text"))

    if effect_name in ("Draw polyline", "Draw filled polygon"):
        raw_parameters["points"] = parse_points_string(raw_parameters["points"])

    if effect_name == "Draw marker":
        raw_parameters["marker_kind"] = {
            "cross":    cv2.MARKER_CROSS,
            "star":     cv2.MARKER_STAR,
            "diamond":  cv2.MARKER_DIAMOND,
            "triangle": cv2.MARKER_TRIANGLE_UP,
            "square":   cv2.MARKER_SQUARE,
        }[raw_parameters["marker_kind"]]

    if effect_name in ("Draw line", "Draw arrow"):
        raw_parameters["point1"] = (raw_parameters.pop("x1"), raw_parameters.pop("y1"))
        raw_parameters["point2"] = (raw_parameters.pop("x2"), raw_parameters.pop("y2"))

    if effect_name == "Draw rectangle" or effect_name == "Draw translucent highlight":
        raw_parameters["point1"] = (raw_parameters.pop("x1"), raw_parameters.pop("y1"))
        raw_parameters["point2"] = (raw_parameters.pop("x2"), raw_parameters.pop("y2"))

    if effect_name == "Draw circle":
        raw_parameters["center"] = (raw_parameters.pop("center_x"),
                                    raw_parameters.pop("center_y"))

    if effect_name == "Draw text":
        raw_parameters["position"] = (raw_parameters.pop("x"), raw_parameters.pop("y"))

    if effect_name == "Draw marker":
        raw_parameters["position"] = (raw_parameters.pop("x"), raw_parameters.pop("y"))

    return raw_parameters


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def show_workspace():
    """Display the unchanged original and current result, even with no effect selected."""
    if 'source_image' not in st.session_state:
        return
    left, right = st.columns(2)
    with left:
        show_image(st.session_state.source_image, 'Original')
    with right:
        show_image(st.session_state.result_image, st.session_state.get('result_label', 'Result'))


def history_toolbar():
    """Show the same Reset/Undo/Redo actions above every Image task.

    Undo is the action described as 'redo' in the request: restore the image
    before the last edit. Redo also exists to restore an undone result. Menu
    browsing alone does not create a history entry or change the image.
    """
    loaded = 'source_image' in st.session_state
    left, middle, right = st.columns(3)
    if left.button('Reset to original', disabled=not loaded,
                   help='Restore the uploaded original and clear effect selection. Undo can restore the previous result.'):
        reset_image(st.session_state)
        st.rerun()
    if middle.button('Undo', disabled=not st.session_state.get('image_undo'),
                     help='Restore the image from before the last applied effect, including its parameter previews.'):
        undo_image(st.session_state)
        st.rerun()
    if right.button('Redo', disabled=not st.session_state.get('image_redo'),
                    help='Restore the most recently undone result without recalculating it.'):
        redo_image(st.session_state)
        st.rerun()


def render():
    """Draw the Image section; no task/effect is chosen automatically."""
    history_toolbar()
    task = menu(
        "Image tasks",
        {
            "Upload":       "Load a starting image from your computer.",
            "Blank canvas": "Create a black canvas of a given size.",
            "Inspect":      "Show the image's shape, dtype, range and byte size.",
            "Effects":      "Pick an effect, set parameters, preview the result.",
            "Save":         "Download the last result as PNG, JPG or BMP.",
        },
        "image_task",
    )
    if task is None:
        show_workspace()
        return

    # --- Upload ---------------------------------------------------------
    if task == "Upload":
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "bmp"],
            help="PNG preserves alpha; JPG and BMP do not.",
        )
        if uploaded_file and st.button("Load image", help='Decode this upload and replace the starting image and result.'):
            loaded = decode_image(uploaded_file.getvalue())
            st.session_state.source_image = loaded
            st.session_state.result_image = loaded.copy()
            st.session_state.result_label = "Original"
            initialize_history(st.session_state)
            st.rerun()
            st.success("Image loaded. Open the Effects task to edit it.")

    # --- Blank canvas ---------------------------------------------------
    elif task == "Blank canvas":
        canvas_width = number_input("Width", 1, 4096, 600, "blank_width")
        canvas_height = number_input("Height", 1, 4096, 600, "blank_height")
        if st.button("Create canvas", help='Create a black image at the chosen pixel dimensions.'):
            blank = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            st.session_state.source_image = blank
            st.session_state.result_image = blank.copy()
            st.session_state.result_label = "Blank canvas"
            initialize_history(st.session_state)
            st.rerun()

    # If nothing is loaded, every other task has to stop.
    if "source_image" not in st.session_state:
        st.info("Upload an image or create a blank canvas first.")
        return

    source = st.session_state.source_image

    # --- Inspect --------------------------------------------------------
    if task == "Inspect":
        st.write({
            "shape": source.shape,
            "dtype": str(source.dtype),
            "min": int(source.min()),
            "max": int(source.max()),
            "bytes_in_memory": source.nbytes,
        })
        st.caption(
            "Bytes in memory is not the file size on disk. "
            "For RGBA images, alpha 0 is transparent and 255 is opaque."
        )

    # --- Effects --------------------------------------------------------
    if task == "Effects":
        group_name = menu("Effect groups", GROUPS, "effect_group")
        if group_name is None:
            show_workspace()
            return

        # Build the effect menu from the registry, filtering by group.
        effect_options = {
            name: definition["help"]
            for name, definition in EFFECTS.items()
            if definition["group"] == group_name
        }
        effect_name = menu("Effects", effect_options, "effect_name")
        if effect_name is None:
            show_workspace()
            return
        effect_definition = EFFECTS[effect_name]

        # Reset the working base when the user switches effects.
        if st.session_state.get("active_effect") != effect_name:
            st.session_state.working_base = st.session_state.result_image.copy()
            st.session_state.active_effect = effect_name
            st.session_state.edit_pending = True

        if st.button("Start another edit", help="Apply this effect again using the latest result as the new starting image."):
            st.session_state.working_base = st.session_state.result_image.copy()
            st.session_state.edit_pending = True

        # Prepare a working copy that always has 3 channels (except for
        # the extract-channel effect, which needs the original).
        working = st.session_state.working_base.copy()
        if working.dtype != np.uint8:
            working = np.clip(working * 255, 0, 255).astype(np.uint8)
        if working.ndim == 2 and effect_name != "Extract channel":
            working = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
        elif working.ndim == 3 and effect_name != "Extract channel":
            if working.shape[2] == 4:
                st.caption("Alpha channel is dropped for this effect.")
            working = working[:, :, :3]

        # Draw one widget per parameter.
        st.subheader("Parameters")
        raw_values = collect_parameters(effect_definition, effect_name, working)
        try:
            values = convert_parameters_for_function(effect_name, raw_values)
            if values.get("thickness") == 0:
                raise ValueError("Thickness must be positive; -1 is available for filled shapes. Zero is not allowed.")
            if "points" in values:
                height, width = working.shape[:2]
                if any(not (0 <= x < width and 0 <= y < height) for x, y in values["points"]):
                    raise ValueError(f"Points must fit the image: X 0–{width-1}, Y 0–{height-1}.")
            if "label" in values and not values["label"].isascii():
                raise ValueError("OpenCV's lesson font supports Latin characters only.")
        except ValueError as error:
            st.error(str(error)); return

        # Live preview vs Apply.
        live_preview = st.toggle(
            "Live preview",
            value=True,
            help="Recompute on every change. Turn off to use the Apply button instead.",
        )
        apply_now = st.button("Apply", disabled=live_preview, help='Recalculate this effect from its fixed starting snapshot using current parameters.')

        if live_preview or apply_now:
            try:
                output = effect_definition["function"](working.copy(), **values)
            except Exception as error:
                st.error(f"Could not apply the effect: {error}")
                output = None

            if output is not None:
                # A few effects return (image, list_of_step_names).
                if isinstance(output, tuple):
                    output, applied_steps = output
                    st.write("Random steps applied:",
                             ", ".join(applied_steps) or "None for this seed")
                new_history_entry = st.session_state.get('edit_pending', False)
                record_edit(st.session_state)
                st.session_state.result_image = output
                st.session_state.result_label = effect_name
                if new_history_entry:
                    # Refresh common toolbar enabled states after the first
                    # successful calculation. Apply mode retains this result.
                    st.rerun()

        if not live_preview:
            st.info("Press Apply to recompute with the current settings.")

    # --- Save -----------------------------------------------------------
    if task == "Save":
        extension = st.selectbox("Format", ["png", "jpg", "bmp"],
                                 help="PNG keeps alpha; JPG is lossy.")
        st.download_button(
            "Download image",
            encode_image(st.session_state.result_image, extension),
            file_name=f"result.{extension}",
            mime="image/jpeg" if extension == "jpg" else f"image/{extension}",
         help='Save the latest successful result in the selected format.')

    show_workspace()
