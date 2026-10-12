"""Automatic SQL generation, execution repair, semantic review, and logging.

This module contains the non-visual pipeline used by
``sections/sql_reflection_section.py``. The section injects model construction,
model invocation, and Pydantic schemas into :func:`submit_request`, which keeps
this workflow independent of a particular provider or UI implementation.

A successful request produces request/data events, one or more initial syntax
attempts, exactly one semantic review, one or more post-semantic syntax checks,
and a complete result. The UI creates a final event only after user confirmation.
"""

import json
import re
import sqlite3
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pydantic import BaseModel

from utils import sql_reflection_store as store
from utils.database import open_readonly_connection
from utils.db_helpers import execute_sql, get_schema


class ComparisonReview(BaseModel):
    """Legacy/shared comparison response used by chart reflection.

    SQL reflection no longer offers answer comparison, but
    ``utils.chart_reflection_workflow`` imports this schema. ``feedback`` is the
    model's comparison text.
    """
    feedback: str


def numbered_answers(history):
    """Map sequential answer numbers to persisted result events.

    ``history`` is an ordered ``list[dict]`` from ``store.events``. Only
    ``kind == 'result'`` is included. The returned ``dict[int, dict]`` starts at
    one and is independent of gaps in database event IDs. Called by
    :func:`resolved_answers` and imported by chart reflection.
    """
    return {number: event for number, event in enumerate(
        (event for event in history if event["kind"] == "result"), start=1
    )}

def snapshot_database(path):
    """Create and persist an immutable snapshot of a SQLite database.

    Parameters
    ----------
    path : str | Path
        Original SQLite database path.

    Returns
    -------
    dict
        Source metadata from ``store.snapshot_data`` plus ``database_path``.

    Calls and call sites
    --------------------
    Uses ``open_readonly_connection`` from ``utils.database``, SQLite's backup
    API, and ``store.snapshot_data`` from ``utils.sql_reflection_store``.
    Called by the section when data is applied and by :func:`submit_request`
    when the first request uses a filesystem path.
    """
    with TemporaryDirectory() as directory:
        backup_path = Path(directory) / "snapshot.db"
        original = open_readonly_connection(path)
        backup = sqlite3.connect(backup_path)
        try:
            original.backup(backup)
        finally:
            backup.close()
            original.close()
        source = store.snapshot_data(backup_path.read_bytes(), Path(path).name + ".db")
    source["database_path"] = path
    return source


def current_source(history):
    """Return the newest source snapshot in ``history``, or ``None``.

    Searches backward through ``data`` and ``result`` events. Called by the SQL
    section's data/model panel and by chart-reflection utilities.
    """
    return next((event["source"] for event in reversed(history)
                 if event["kind"] in ("data", "result") and event.get("source")), None)


def join_source(base, addition, prefix):
    """Copy prefixed tables from ``addition`` into a new snapshot of ``base``.

    Parameters
    ----------
    base : dict
        Current immutable source metadata.
    addition : dict
        Source whose user tables will be copied.
    prefix : str
        SQL identifier prefix added to every copied table.

    Returns
    -------
    dict
        New persisted source metadata with a ``components`` list.

    Raises
    ------
    ValueError
        Prefix is unsafe, no tables exist, or a prefixed name already exists.

    Calls and call sites
    --------------------
    Uses ``re.fullmatch``, read-only connections, SQLite backup, Pandas
    ``read_sql_query``/``to_sql``, and ``store.snapshot_data``. Called by
    ``data_and_model_settings`` when the user applies **Add data for joins**.
    """
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", prefix):
        raise ValueError("Use a table prefix starting with a letter, containing letters, numbers or underscores.")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "joined.db"
        target = sqlite3.connect(path)
        original = open_readonly_connection(base["snapshot"])
        extra = open_readonly_connection(addition["snapshot"])
        try:
            original.backup(target)
            tables = extra.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
            if not tables:
                raise ValueError("The added database has no data tables.")
            existing = {row[0].casefold() for row in target.execute("SELECT name FROM sqlite_master")}
            for (name,) in tables:
                new_name = f"{prefix}_{name}"
                if new_name.casefold() in existing:
                    raise ValueError(f"Table {new_name} already exists. Choose a different prefix.")
                quoted = name.replace('"', '""')
                frame = pd.read_sql_query(f'SELECT * FROM "{quoted}"', extra)
                frame.to_sql(new_name, target, index=False, if_exists="fail")
                existing.add(new_name.casefold())
            target.commit()
        finally:
            extra.close()
            original.close()
            target.close()
        source = store.snapshot_data(path.read_bytes(), f"{base['name']} + {addition['name']}.db")
    source["components"] = base.get("components", [base]) + [dict(addition, table_prefix=prefix)]
    return source


def combine_uploaded_sources(base, uploaded_files, prefix="extra"):
    """Build one queryable snapshot from multiple uploaded data files.

    Parameters
    ----------
    base : dict | None
        Existing source to extend. When ``None``, the first uploaded file
        becomes the base source.
    uploaded_files : list[UploadedFile]
        Streamlit uploads exposing ``name`` and ``getvalue()``.
    prefix : str
        Prefix for one addition. Multiple additions receive numbered prefixes.

    Returns
    -------
    dict
        One immutable source containing every uploaded table.

    Operation and call sites
    ------------------------
    Each file is persisted by ``store.snapshot_data``. :func:`join_source`
    copies subsequent tables into the accumulating snapshot. Called by the
    section's **Apply data** action and by :func:`submit_request` when files are
    submitted before being explicitly applied.
    """
    files = list(uploaded_files or [])
    if not files:
        if base:
            return base
        raise ValueError("Upload at least one CSV or SQLite database file.")

    combined = base
    for index, uploaded in enumerate(files, start=1):
        addition = store.snapshot_data(uploaded.getvalue(), uploaded.name)
        if combined is None:
            combined = addition
            continue
        if len(files) == 1:
            table_prefix = prefix or "extra"
        else:
            clean_name = re.sub(r"[^A-Za-z0-9_]+", "_", Path(uploaded.name).stem).strip("_")
            if not clean_name or not clean_name[0].isalpha():
                clean_name = f"file_{clean_name}" if clean_name else "file"
            table_prefix = f"{clean_name}_{index}"
        combined = join_source(combined, addition, table_prefix)
    return combined


def resolved_answers(history):
    """Return display-ready numbered answers, including legacy executions.

    ``history`` is the complete event list. The function copies results from
    :func:`numbered_answers`, then merges later ``execution`` events by
    ``result_id``. Current automatic pipeline results already contain rows;
    merging remains for old saved sessions. Returns ``dict[int, dict]``.
    Called by the section, :func:`submit_request`, and legacy helpers.
    """
    answers = {number: dict(answer) for number, answer in numbered_answers(history).items()}
    by_id = {answer["id"]: answer for answer in answers.values()}
    for event in history:
        if event["kind"] == "execution" and event["result_id"] in by_id:
            by_id[event["result_id"]].update({key: event[key] for key in
                ("rows", "csv", "row_count", "truncated", "execution_error")})
            by_id[event["result_id"]]["executed_at"] = event["created_at"]
    return answers


def result_payload(frame):
    """Convert a DataFrame into JSON/CSV fields stored on a result event.

    ``frame`` is returned by ``execute_sql``. The output dictionary contains
    split-oriented JSON rows, complete CSV, row count, truncation state, and an
    error string when the database layer returned an ``error`` column. Called
    by :func:`submit_request` and legacy :func:`run_answer`.
    """
    return {
        "rows": json.loads(frame.to_json(orient="split", date_format="iso")),
        "csv": frame.to_csv(index=False), "row_count": len(frame),
        "truncated": bool(frame.attrs.get("truncated", False)),
        "execution_error": str(frame.iloc[0]["error"]) if list(frame.columns) == ["error"] and len(frame) else None,
    }


MAX_SYNTAX_ATTEMPTS = 5


def sql_error(frame):
    """Return an encoded SQLite error string, otherwise ``None``.

    ``utils.database.run_readonly_query`` represents caught database exceptions
    as a one-row DataFrame whose only column is ``error``. Called after every
    execution by :func:`repair_until_executable`.
    """
    if list(frame.columns) == ["error"] and len(frame):
        return str(frame.iloc[0]["error"])
    return None


def repair_until_executable(session_id, request_id, model, settings, source,
                            schema, original_request, semantic_comment, rules,
                            initial_sql, initial_explanation, phase, invoke,
                            query_schema, phase_update=None):
    """Execute and repair SQL until SQLite succeeds or five attempts fail.

    Parameters
    ----------
    session_id : str
        Experiment receiving attempt events.
    request_id : int
        Parent request-event ID.
    model
        Already constructed model reused throughout the pipeline.
    settings : dict
        Model/provider metadata used for invocation and logs.
    source : dict
        Immutable database snapshot metadata.
    schema : str
        Current SQLite CREATE TABLE definitions.
    original_request : str
        First request describing the experiment objective.
    semantic_comment : str
        Optional user clarification about intended meaning.
    rules : str
        Optional data/business definitions.
    initial_sql : str
        SQL executed without another model call on attempt one.
    initial_explanation : str
        Explanation associated with attempt one.
    phase : str
        ``initial_syntax`` or ``post_semantic_syntax`` for logs.
    invoke : callable
        Provider-independent callback defined as ``invoke_sql_response`` in the
        section module.
    query_schema
        Pydantic response class, normally ``QueryReview``.
    phase_update : callable | None
        Optional UI callback receiving human-readable phase labels, including
        the current attempt and the five-attempt maximum.

    Returns
    -------
    tuple[str, str, pandas.DataFrame]
        Executable SQL, latest explanation, and complete result frame.

    Raises
    ------
    ValueError
        All ``MAX_SYNTAX_ATTEMPTS`` executions failed.

    Operation and calls
    -------------------
    Attempt one executes ``initial_sql``. Later attempts call ``invoke`` with
    the exact previous SQLite error. ``execute_sql`` is defined in
    ``utils.db_helpers`` and is called with ``row_limit=None`` for the complete
    result. :func:`sql_error` interprets the frame and ``store.log_event`` logs
    every attempt. :func:`submit_request` calls this before and after semantic
    review; the semantic model call itself is never repeated here.
    """
    sql = initial_sql.strip()
    explanation = initial_explanation
    previous_error = None

    for attempt in range(1, MAX_SYNTAX_ATTEMPTS + 1):
        phase_name = ("Initial syntax" if phase == "initial_syntax"
                      else "Post-semantic syntax")
        if attempt > 1:
            if phase_update:
                phase_update(
                    f"{phase_name} repair by model — attempt "
                    f"{attempt}/{MAX_SYNTAX_ATTEMPTS}"
                )
            # Feed the concrete SQLite error back to the same selected model.
            prompt = (
                "Correct only the SQLite execution problem below.\n"
                "Preserve the intended meaning of the query.\n"
                "Return one complete read-only SQLite SELECT query.\n"
                "Use only tables and columns present in the schema.\n"
                "Do not return markdown, placeholders, or alternatives.\n\n"
                f"Original request:\n{original_request}\n\n"
                f"Semantic clarification:\n{semantic_comment or 'None'}\n\n"
                f"Database schema:\n{schema}\n\n"
                f"Data definitions:\n{rules or 'None'}\n\n"
                f"Previous SQL:\n{sql}\n\n"
                f"SQLite execution error:\n{previous_error}"
            )
            response = invoke(model, settings, query_schema, prompt)
            sql = response.refined_sql.strip()
            explanation = response.explanation

        if phase_update:
            phase_update(
                f"{phase_name} execution check — attempt "
                f"{attempt}/{MAX_SYNTAX_ATTEMPTS}"
            )
        # Execute against the immutable snapshot and retain the complete result.
        frame = execute_sql(sql, source["snapshot"], row_limit=None)
        previous_error = sql_error(frame)
        store.log_event(session_id, "syntax_attempt", {
            **settings, "request_id": request_id, "phase": phase,
            "attempt": attempt, "sql": sql, "model_explanation": explanation,
            "execution_error": previous_error,
            "successful": previous_error is None,
        })
        if previous_error is None:
            return sql, explanation, frame

    raise ValueError(
        f"SQL remained invalid after {MAX_SYNTAX_ATTEMPTS} attempts. "
        f"Last SQLite error: {previous_error}"
    )


def submit_request(session_id, history, message, semantic_comment, settings,
                   data, rules, build_model, invoke, query_schema,
                   semantic_schema, guidance_schema, phase_update=None):
    """Run the complete automatic pipeline and persist its audit trail.

    Parameters
    ----------
    session_id : str
        Active experiment ID.
    history : list[dict]
        Events loaded before this submission.
    message : str
        Current independent natural-language request.
    semantic_comment : str
        Optional explicit business/meaning clarification.
    settings : dict
        Selected model name, reasoning effort, provider, and model ID.
    data : dict
        Source/upload/reuse/path information from the section.
    rules : str
        Optional data definitions entered by the user.
    build_model : callable
        ``models.build_chat_model`` injected by the section.
    invoke : callable
        ``sections.sql_reflection_section.invoke_sql_response``.
    query_schema
        ``QueryReview`` Pydantic class.
    semantic_schema
        ``SemanticReview`` Pydantic class.
    guidance_schema
        ``QueryFailureGuidance`` Pydantic class used only when the request
        cannot be converted into executable SQL.
    phase_update : callable | None
        Optional callback used by the Streamlit section to display the current
        model/execution phase while this blocking workflow runs.

    Returns
    -------
    None
        Results are persisted as events and shown after Streamlit reruns.

    Operation and calls
    -------------------
    Logs the request, resolves/snapshots data, reads the schema with
    ``utils.db_helpers.get_schema``, builds one model, invokes initial SQL,
    calls :func:`repair_until_executable`, invokes exactly one semantic review,
    calls the repair loop again for execution safety, converts the final frame
    with :func:`result_payload`, and logs ``result``. When repair is exhausted,
    it invokes ``guidance_schema`` once and logs actionable ``guidance_result``
    text; provider-level failure becomes ``pipeline_failure``. Called only by
    ``sections.sql_reflection_section.render`` in the SQL feature.
    """
    started = time.perf_counter()
    request = dict(settings, user_request=message,
                   semantic_comment=semantic_comment, rules=rules,
                   response_type="sql",
                   request_type="question")
    # Persist the user's command before any model/database work begins.
    request_id = store.log_event(session_id, "request", request)
    model = None
    schema = ""
    try:
        # Resolve exactly one immutable database snapshot for this request.
        if data.get("source"):
            source = data["source"]
        elif data.get("uploaded") is not None:
            source = combine_uploaded_sources(None, data["uploaded"])
        elif data["reuse"] and data["previous_source"]:
            source = data["previous_source"]
        else:
            source = snapshot_database(data["path"])

        schema = get_schema(source["snapshot"])
        store.log_event(session_id, "data", {
            "request_id": request_id, "source": source, "schema": schema,
        })
        original_request = message
        # Construct once; syntax generation/repair and semantic review reuse it.
        model = build_model(settings["model"], settings["effort"])
        prompt = (
            "Create one complete read-only SQLite SELECT query.\n"
            "Use only tables and columns present in the schema.\n"
            "Make the query executable and do not return markdown or alternatives.\n\n"
            f"Original request:\n{original_request}\n\n"
            f"Current user message:\n{message}\n\n"
            f"Semantic clarification:\n{semantic_comment or 'None'}\n\n"
            f"Database schema:\n{schema}\n\n"
            f"Data definitions:\n{rules or 'None'}"
        )
        # Initial model response proposes SQL; actual correctness is decided by
        # SQLite execution in repair_until_executable.
        if phase_update:
            phase_update("Syntactic SQL generation by model")
        response = invoke(model, settings, query_schema, prompt)
        syntax_sql, syntax_explanation, syntax_frame = repair_until_executable(
            session_id, request_id, model, settings, source, schema,
            original_request, semantic_comment, rules, response.refined_sql,
            response.explanation, "initial_syntax", invoke, query_schema,
            phase_update,
        )
        semantic_prompt = (
            "Perform exactly one semantic investigation of this executable SQL.\n"
            "Determine whether it precisely answers the user request and clarification.\n"
            "Check for wrong or missing joins, duplicated rows, incorrect aggregation, "
            "grouping, filters, dates, null handling, selected columns, and business meaning.\n"
            "Correct every semantic problem in the same response.\n"
            "Preserve valid parts and use only tables and columns in the schema.\n"
            "Return one complete read-only SQLite SELECT query.\n"
            "If the query is already correct, return it unchanged.\n\n"
            f"Original request:\n{original_request}\n\n"
            f"Current user message:\n{message}\n\n"
            f"User semantic clarification:\n{semantic_comment or 'None'}\n\n"
            f"Database schema:\n{schema}\n\n"
            f"Data definitions:\n{rules or 'None'}\n\n"
            f"Successfully executed SQL:\n{syntax_sql}\n\n"
            f"Execution output (up to 50 rows):\n{syntax_frame.head(50).to_csv(index=False)}"
        )
        # This is the pipeline's single semantic model call.
        if phase_update:
            phase_update("Semantic investigation by model — one pass")
        semantic_response = invoke(model, settings, semantic_schema, semantic_prompt)
        semantic_sql = semantic_response.corrected_sql.strip()
        store.log_event(session_id, "semantic_review", {
            **settings, "request_id": request_id, "input_sql": syntax_sql,
            "output_sql": semantic_sql,
            "semantic_issues": semantic_response.semantic_issues,
            "explanation": semantic_response.explanation,
            "user_comment": semantic_comment,
        })
        # Semantic editing may introduce a syntax/name error. Repair only those
        # execution errors without repeating semantic investigation.
        final_sql, final_explanation, final_frame = repair_until_executable(
            session_id, request_id, model, settings, source, schema,
            original_request, semantic_comment, rules, semantic_sql,
            semantic_response.explanation, "post_semantic_syntax", invoke,
            query_schema, phase_update,
        )
        payload = dict(
            sql=final_sql, answer=semantic_response.explanation,
            semantic_issues=semantic_response.semantic_issues,
            semantic_comment=semantic_comment, syntax_sql=syntax_sql,
            syntax_explanation=syntax_explanation,
            final_syntax_explanation=final_explanation,
            source=source, schema=schema, query_author="pipeline",
            **result_payload(final_frame),
        )
        # A successful result stores the complete rows and downloadable CSV.
        if phase_update:
            phase_update("Saving the successful final result")
        store.log_event(session_id, "result", dict(
            request, **payload, request_id=request_id,
            prompt=prompt,
            duration_seconds=round(time.perf_counter() - started, 3),
        ))
    except Exception as exc:
        if model is not None:
            try:
                if phase_update:
                    phase_update("Generating guidance for an invalid request")
                guidance_prompt = (
                    "The request below could not be converted into executable SQLite SQL.\n"
                    "Explain the problem in plain language. Identify missing or mismatched "
                    "tables, columns, relationships, ambiguity, or unsupported expectations.\n"
                    "Tell the user exactly how to rewrite the request and provide one concrete "
                    "example request. Do not invent tables or columns.\n\n"
                    f"User request:\n{message}\n\n"
                    f"Semantic clarification:\n{semantic_comment or 'None'}\n\n"
                    f"Database schema:\n{schema or 'Schema could not be read.'}\n\n"
                    f"Final pipeline error:\n{exc}"
                )
                guidance = invoke(model, settings, guidance_schema, guidance_prompt)
                store.log_event(session_id, "guidance_result", {
                    **settings, "request_id": request_id, "user_request": message,
                    "explanation": guidance.explanation,
                    "request_advice": guidance.request_advice,
                    "pipeline_error": str(exc), "prompt": guidance_prompt,
                    "duration_seconds": round(time.perf_counter() - started, 3),
                })
                return
            except Exception as guidance_exc:
                error = f"{exc}; guidance generation also failed: {guidance_exc}"
        else:
            error = str(exc)
        store.log_event(session_id, "pipeline_failure", {
            "request_id": request_id, "error": error,
            "duration_seconds": round(time.perf_counter() - started, 3),
        })

def mark_final(session_id, history, number, note):
    """Legacy helper that manually logs one resolved answer as final.

    ``session_id`` selects the experiment, ``history`` supplies results,
    ``number`` selects the answer, and ``note`` records an optional reason.
    Returns ``None``. It calls :func:`resolved_answers` and
    ``store.log_event``. The SQL section no longer calls it because
    ``draw_conversation`` logs confirmation directly from the result checkbox.
    """
    answer = resolved_answers(history)[number]
    store.log_event(session_id, "final", {
        "result_id": answer["id"], "answer_number": number,
        "request_id": answer["request_id"], "note": note,
        "user_request": answer["user_request"], "sql": answer.get("sql"),
        "answer": answer["answer"], "rows": answer.get("rows"),
        "source": answer.get("source"),
        "request_count": sum(event["kind"] == "request" for event in history),
    })


def run_answer(session_id, history, number):
    """Legacy helper that executes a saved answer against its source snapshot.

    Parameters are the experiment ID, complete history, and answer number. It
    calls :func:`resolved_answers`, ``execute_sql``, :func:`result_payload`, and
    ``store.log_event`` and returns ``None``. The current SQL section no longer
    calls it because execution is automatic inside
    :func:`repair_until_executable`.
    """
    answer = resolved_answers(history)[number]
    started = time.perf_counter()
    try:
        frame = execute_sql(answer["sql"], answer["source"]["snapshot"])
        payload = result_payload(frame)
    except Exception as exc:
        payload = result_payload(pd.DataFrame({"error": [str(exc)]}))
    store.log_event(session_id, "execution", dict(
        result_id=answer["id"], answer_number=number, sql=answer["sql"],
        source=answer["source"], duration_seconds=round(time.perf_counter() - started, 3), **payload,
    ))
