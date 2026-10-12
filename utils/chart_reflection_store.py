"""Chart experiment logs, CSV snapshots, and rendered-image artifacts."""

import hashlib
from pathlib import Path

from utils import sql_reflection_store as audit

ROOT = Path(__file__).resolve().parents[1] / "data" / "chart_reflection"


def new_session(root=ROOT):
    return audit.new_session(root)


def list_sessions(root=ROOT):
    return audit.list_sessions(root)


def events(session_id, root=ROOT):
    return audit.events(session_id, root)


def log_event(session_id, kind, payload, root=ROOT):
    return audit.log_event(session_id, kind, payload, root)


def snapshot_csv(raw, name, root=ROOT):
    digest = hashlib.sha256(raw).hexdigest()
    path = Path(root) / "sources" / f"{digest}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    return dict(name=name, sha256=digest, snapshot=str(path))


def save_image(raw, root=ROOT):
    path = Path(root) / "charts" / (hashlib.sha256(raw).hexdigest() + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    return str(path)
