"""Image snapshots for Undo/Redo; independent of Streamlit widgets.

One history entry represents one applied effect, including all its parameter
previews. Stored arrays are copies, so later edits cannot mutate old states.
History is session-only and bounded to ten entries / approximately 64 MiB
per stack, retaining at least one snapshot even for a larger image.
"""

def snapshot(state):
    """Copy the displayed image and label without modifying the original."""
    return (state['result_image'].copy(), state.get('result_label', 'Original'))


def append_snapshot(stack, entry):
    """Append an independent snapshot and remove oldest entries past the budget."""
    stack.append(entry)
    while len(stack) > 1 and (len(stack) > 10 or sum(image.nbytes for image, _ in stack) > 64 * 1024 * 1024):
        stack.pop(0)


def clear_effect_selection(state):
    """Stop automatic preview after history navigation; require a new effect click."""
    state['effect_name'] = None
    state['active_effect'] = None
    state['edit_pending'] = False
    state['working_base'] = state['result_image'].copy()


def initialize_history(state):
    """Clear history/selections when a different source image or canvas is loaded."""
    state['image_undo'] = []
    state['image_redo'] = []
    state['effect_group'] = None
    clear_effect_selection(state)


def record_edit(state):
    """Save the pre-effect image only once, when the pending edit first succeeds."""
    if state.get('edit_pending'):
        append_snapshot(state.setdefault('image_undo', []), snapshot(state))
        state['image_redo'] = []
        state['edit_pending'] = False


def undo_image(state):
    """Restore the image before the last applied effect; return whether it changed."""
    stack = state.setdefault('image_undo', [])
    if not stack:
        return False
    append_snapshot(state.setdefault('image_redo', []), snapshot(state))
    state['result_image'], state['result_label'] = stack.pop()
    clear_effect_selection(state)
    return True


def redo_image(state):
    """Restore an undone result as a snapshot, without recalculating its effect."""
    stack = state.setdefault('image_redo', [])
    if not stack:
        return False
    append_snapshot(state.setdefault('image_undo', []), snapshot(state))
    state['result_image'], state['result_label'] = stack.pop()
    clear_effect_selection(state)
    return True


def reset_image(state):
    """Restore the uploaded original, making Reset itself undoable if it changes it."""
    import numpy as np
    source = state['source_image']
    if source.dtype != state['result_image'].dtype or not np.array_equal(source, state['result_image']):
        append_snapshot(state.setdefault('image_undo', []), snapshot(state))
        state['image_redo'] = []
    state['result_image'] = source.copy()
    state['result_label'] = 'Original'
    state['effect_group'] = None
    clear_effect_selection(state)
