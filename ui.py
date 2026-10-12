"""
ui.py
=====

Reusable Streamlit widgets for the whole project.

Nothing in this file knows about images, videos, chat, or effects. It
just draws small, consistent controls and returns what the user chose.

Every other file imports from here so the visual language stays the
same. If you want to change how a slider looks or how a menu behaves,
you only change it here.
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------------

def apply_style():
    """Apply a small shared style through Streamlit, without another UI library.

    st.markdown's unsafe_allow_html permits this static CSS style block.
    It contains no uploaded content. The selectors target native Streamlit
    buttons, including download buttons. One size keeps menus and actions
    consistent; max-width lets them fit on a small screen.
    """
    st.markdown("""
    <style>
    div.stButton button, div.stDownloadButton button {
        width: 180px !important;
        max-width: 100%;
        height: 64px;
        min-height: 64px;
        border-radius: 10px;
        padding: 6px 12px;
        font-weight: 600;
        white-space: normal;
    }
    div.stButton button p, div.stDownloadButton button p {
        font-size: 0.9rem;
        line-height: 1.15;
    }
    div.stButton button:focus-visible, div.stDownloadButton button:focus-visible {
        outline: 2px solid #3b82f6;
        outline-offset: 2px;
    }
    </style>
    """, unsafe_allow_html=True)


def menu(title, options, key):
    """Return the chosen item; sidebar for sections, top boxes for submenus.

    Each submenu is a bordered Streamlit container with equal grid cells.
    Only active levels render, so old task descriptions disappear on navigation.
    Add a dictionary entry to extend a menu without changing this renderer.
    """
    if st.session_state.get(key) not in options:
        st.session_state[key] = None
    selected = st.session_state[key]
    if key == "section":
        with st.sidebar.expander("Main menu", expanded=True):
            draw_menu_buttons(options, key, selected)
    else:
        # st.expander draws a bordered, collapsible box around the submenu.
        # st.columns puts its buttons in an ordered grid, instead of long
        # stacked menus or increasingly narrow nested columns.
        # Parent levels collapse to their selected label while editing an
        # image, keeping the current effect and workspace near the top.
        parent_level = (key == "image_task" and selected == "Effects") or (key == "effect_group" and selected is not None)
        with st.expander(title + " · " + (selected or "Choose an option"), expanded=not parent_level):
            entries = list(options.items())
            for start in range(0, len(entries), 3):
                cells = st.columns(3, gap="small")
                for cell, (label, tooltip) in zip(cells, entries[start:start + 3]):
                    with cell:
                        draw_menu_buttons({label: tooltip}, key, selected)
    return st.session_state[key]


def draw_menu_buttons(options, key, selected):
    """Draw uniform buttons with selected styling and hover descriptions."""
    for label, tooltip in options.items():
        if st.button(label, key=f"{key}:{label}", help=tooltip,
                     type="primary" if label == selected else "secondary"):
            st.session_state[key] = label
            if key == "effect_group":
                st.session_state.effect_name = None
            if key == "image_task" and label == "Effects" and selected != label:
                st.session_state.effect_group = None
                st.session_state.effect_name = None
            # Rerun refreshes selection highlighting immediately.
            st.rerun()



# ---------------------------------------------------------------------------
# Bounded numeric input
# ---------------------------------------------------------------------------

def number_input(label, minimum, maximum, default, key, help_text=""):
    """
    A number input that shows the allowed range and refuses out-of-range values.

    The bounds are printed above the widget so the user never has to guess
    what values are legal. If the user types something outside the range,
    a red error appears and the page stops rendering.
    """
    st.caption(f"{label}: min {minimum}, max {maximum}")
    value = st.number_input(label, value=default, key=key, help=help_text)
    if not minimum <= value <= maximum:
        st.error(f"{label} must be between {minimum} and {maximum}.")
        st.stop()
    st.caption(f"Current value: {value}")
    return value


def slider_input(label, minimum, maximum, default, key, step=1, help_text=""):
    """
    A slider that shows the current value and the legal range next to it.
    """
    value = st.slider(label, minimum, maximum, default, step=step, key=key, help=help_text)
    st.write(f"{label}: {value} (min {minimum}, max {maximum})")
    return value


# ---------------------------------------------------------------------------
# Image display
# ---------------------------------------------------------------------------

def show_image(image, caption="Result"):
    """
    Display a NumPy array as an image with the colors a human expects.

    OpenCV stores images in Blue-Green-Red order; browsers expect
    Red-Green-Blue. This function does the conversion only for display,
    never for the stored array.
    """
    import cv2
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2RGBA if image.shape[2] == 4 else cv2.COLOR_BGR2RGB
        image = cv2.cvtColor(image, code)
    st.image(image, caption=caption, clamp=True)
