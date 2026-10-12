# Verification report

- Ten unittest tests passed. The registry test exercises all 27 registered effects using their default parameters and confirms source images remain unchanged.
- Streamlit AppTest opened all seven main sections without exceptions or application error messages.
- Checked Settings model filtering, visible chat composer, chained unsaved effects, live-rerun stability, and repeated Apply clicks.
- Verified separate conversation persistence and recent-turn agent memory using a fake model.
- Verified generated SQL cannot delete/drop/attach/write and ordinary SELECT succeeds.
- Executed a local Matplotlib chart after approval, captured its PNG, then edited the code and verified approval was cleared.
- All Python modules compile; ZIP integrity and inclusion of the entry point checked.

Not exercised: live provider responses or summarization calls, speech inference, Edge TTS service, YOLO inference, camera, RTSP streams, browser microphone recording. Those require credentials, downloaded model weights, network/device access. Chart execution uses local Python permissions and is not sandboxed.

Commands: python -m unittest discover -s tests -v; python -m compileall -q .


## Menu review revision

Ten tests passed after moving submenus into the main-page grid. Navigation tests now use both sidebar and top menu buttons and assert that the sidebar contains only seven main sections. Reset was verified to restore the original exactly. Static inspection found no missing help arguments on section action/download buttons, uploads, text inputs/areas, selectboxes and checkboxes. The supplied style dimensions are not screenshot-verified: cloud browser access to the local app was blocked with ERR_BLOCKED_BY_CLIENT.

## Latest noise/filter revision

18 tests passed, including all 36 registry defaults and eight noise/kernel/UI regression tests. Covered impulse boundary cases, seeded repeatability, source preservation, grayscale shape, zero-strength identities, clipping, invalid parameters, custom kernel errors, pipeline chance boundaries, threshold output and hover-only menu descriptions. No new external inference was attempted.

## Explicit selection and history update

See IMAGE_HISTORY.md. No first menu item is chosen automatically. A shared Reset/Undo/Redo toolbar appears in every Image task. Undo means return to the state before the last applied effect; Redo restores an undone result. Twenty-one tests pass after this update.
