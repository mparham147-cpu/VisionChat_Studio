# Explicit selections and image history

No menu button is selected automatically on first use. Main section, task, effect group and effect start with None. Clicking Image → Effects does not apply grayscale. Clicking a group only reveals its choices. Live preview begins only after explicitly selecting an effect; Apply mode waits for Apply.

The common Image toolbar is always present across Upload, Blank canvas, Inspect, Effects and Save:

- Reset to original restores the uploaded color/alpha image exactly and clears effect selection. Reset can itself be undone.
- Undo restores the snapshot before the last applied effect. This is the back-to-previous-state action described as “redo” in the request.
- Redo restores an undone result exactly, without recalculating noise or filters.

Parameter previews and repeated Apply clicks remain one history entry per edit. Start another edit begins a new entry. Menu browsing alone creates no entry. A newly applied edit after Undo clears the redo branch. Loading another image/canvas clears history; it cannot restore an earlier upload.

Images/labels are copied before edits. Working snapshots preserve shape, dtype, channels and alpha. History is temporary session memory: each stack retains up to ten entries with an approximate 64 MiB budget, always keeping at least the last snapshot. Original and current images remain separate.

## Implementation

ui.menu stores None instead of defaulting to the first option. Section routers return early when no choice has been made. Image shows the current result while waiting for a selection. utils/image_history.py contains snapshot, initialize_history, record_edit, undo_image, redo_image and reset_image, with docstrings above every helper. history_toolbar() draws common actions; they are disabled when unavailable. A first successful effect refreshes the toolbar's enabled states without adding another history entry.

21 tests passed. Explicit-selection UI tests confirm that Effects and group navigation preserve the color image, grayscale requires a click, Undo returns the exact color array, Redo returns the exact grayscale array, and Reset restores the original. Toolbar availability is checked across all Image tasks. History tests verify one entry across previews, reset/undo and redo-branch clearing. All existing 36 effect tests continue to pass.
