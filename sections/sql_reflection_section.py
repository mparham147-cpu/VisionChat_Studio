"""Streamlit user interface for the automatic SQL-reflection pipeline.

Program entry
-------------
``sections.chatbot_section`` imports and calls :func:`render` when the user
selects ``SQL reflection``. Streamlit reruns :func:`render` after every widget
interaction, so persistent experiment state is stored by
``utils.sql_reflection_store`` rather than in local variables.

Pipeline
--------
The user selects one model and data source, submits a natural-language request,
and may add a semantic clarification. :func:`render` passes these values to
``utils.sql_reflection_workflow.submit_request``. That workflow:

1. asks the selected model for SQL;
2. executes it against a read-only SQLite snapshot;
3. automatically gives real SQLite errors back to the same model until the
   query executes or the attempt limit is reached;
4. asks the same model once to inspect semantic correctness;
5. executes and, if necessary, syntactically repairs the semantic query;
6. logs every attempt and waits for the user to confirm the successful result.

This module defines the model-response schemas, provider-independent response
validation, experiment/data/model controls, conversation and investigation-log
rendering, request form, JSON export, and top-level page controller. Query
execution and persistence are intentionally delegated to the utility modules.
"""

import json

import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field

from models import model_selector, build_chat_model, extract_answer_text
from utils.file_io import PROJECT_ROOT
from utils import sql_reflection_store as store
from utils.sql_reflection_workflow import (
    current_source,
    combine_uploaded_sources,
    join_source,
    resolved_answers,
    snapshot_database,
    submit_request,
)


class QueryReview(BaseModel):
    """Structured output used for SQL generation and syntax repair.

    ``explanation`` explains the generated or repaired query. ``refined_sql`` is
    the complete SQLite query executed by the workflow. Instances are produced
    by :func:`invoke_sql_response` and consumed by ``submit_request`` and its
    automatic repair loop in ``utils.sql_reflection_workflow``.
    """
    explanation: str = Field(description="One or two sentences explaining the generated SQL.")
    refined_sql: str = Field(description="The final SQLite query.")

class SemanticReview(BaseModel):
    """Structured output for the single semantic investigation.

    ``semantic_issues`` lists meaning or business-logic problems,
    ``explanation`` describes the conclusion and corrections, and
    ``corrected_sql`` contains the final read-only query. The workflow executes
    that query after this response, so a semantically edited query must still
    pass SQLite execution validation.
    """

    semantic_issues: list[str] = Field(
        description="Semantic problems found in the SQL. Return an empty list when none are found."
    )
    explanation: str = Field(
        description="Explain whether the SQL answers the original request and describe any correction."
    )
    corrected_sql: str = Field(
        description="The final corrected read-only SQLite SELECT query."
    )


class QueryFailureGuidance(BaseModel):
    """Helpful response returned when no executable query can be produced.

    ``explanation`` tells the user why the request could not become valid SQL.
    ``request_advice`` explains what database information is missing or how to
    rewrite the request. The workflow logs and displays this response instead
    of exposing only a low-level SQLite error.
    """

    explanation: str = Field(description="Clear explanation of why valid SQL could not be produced.")
    request_advice: str = Field(description="Specific instructions and an example of a better request.")

def invoke_sql_response(chat_model, profile, schema, prompt):
    """Invoke a model and return a response validated against ``schema``.

    Parameters
    ----------
    chat_model
        LangChain-compatible model created by ``models.build_chat_model``.
    profile : dict
        Selected model settings. ``profile['provider']`` determines whether
        native OpenAI structured output or JSON-text validation is used.
    schema : type[BaseModel]
        Expected Pydantic class, normally :class:`QueryReview` or
        :class:`SemanticReview`.
    prompt : str
        Complete instruction and database context sent to the model.

    Returns
    -------
    BaseModel
        A validated instance of the supplied schema.

    Raises
    ------
    ValueError
        A text-based provider returned JSON that does not match the schema.

    Calls and call sites
    --------------------
    OpenAI uses ``chat_model.with_structured_output(...).invoke(...)``. Other
    providers use ``chat_model.invoke``, ``models.extract_answer_text``, and
    Pydantic's ``model_validate_json``. :func:`render` passes this function to
    ``submit_request`` as its invocation callback; the workflow calls it for
    initial generation, execution-error repair, and semantic review.
    """
    # OpenAI can enforce the Pydantic schema at the provider/model layer.
    if profile["provider"] == "openai":
        return chat_model.with_structured_output(schema).invoke(prompt)

    # Other providers receive the JSON schema as an explicit instruction.
    instructions = (
        "Return only one JSON object matching the following schema. "
        "Do not include markdown or explanations outside the JSON.\n"
        + json.dumps(schema.model_json_schema())
    )
    response = extract_answer_text(chat_model.invoke(instructions + "\n\n" + prompt)).strip()
    # Local reasoning models may include a thinking block before the answer;
    # only the final visible response is parsed.
    if "</think>" in response:
        response = response.split("</think>", 1)[1].strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return schema.model_validate_json(response)
    except ValueError as exc:
        raise ValueError(
            "The selected model returned an invalid SQL response. ",
            "Retry or choose another model.",
            response
        ) from exc


def experiment_picker():
    """Render experiment navigation and return the active experiment ID.

    Parameters
    ----------
    None
        State is read from and written to ``st.session_state``.

    Returns
    -------
    str
        Active experiment/session identifier.

    Operation and calls
    -------------------
    Applies a pending experiment switch, renders saved sessions, creates a new
    session when requested or when none exist, and synchronizes the selectbox
    with ``sql_active_experiment``. It calls ``store.new_session`` and
    ``store.list_sessions``, defined in ``utils/sql_reflection_store.py``.
    Streamlit calls the nested ``select_experiment`` callback when the selectbox
    changes. :func:`render` calls this function first on every page run.
    """
    # A data replacement schedules a new experiment before Streamlit reruns.
    pending = st.session_state.pop("sql_pending_experiment", None)
    if pending:
        st.session_state.sql_active_experiment = pending
        st.session_state.sql_experiment_picker = pending
    with st.expander("Saved experiments"):
        if st.button("New experiment"):
            st.session_state.sql_active_experiment = store.new_session()
            st.session_state.sql_experiment_picker = st.session_state.sql_active_experiment
        sessions = store.list_sessions()
        if not sessions:
            st.session_state.sql_active_experiment = store.new_session()
            sessions = store.list_sessions()
        titles = {row[0]: f"{row[1]} ({row[2][:10]})" for row in sessions}
        active = st.session_state.get("sql_active_experiment")
        if active not in titles:
            active = next(iter(titles))
            st.session_state.sql_active_experiment = active
        st.session_state.sql_experiment_picker = active

        def select_experiment():
            """Copy the selectbox value into the active experiment state.

            Streamlit calls this zero-argument nested callback through the
            selectbox's ``on_change`` parameter. It returns ``None``.
            """
            st.session_state.sql_active_experiment = st.session_state.sql_experiment_picker

        st.selectbox("Saved SQL experiments", list(titles), index=None,
                     format_func=titles.get, key="sql_experiment_picker", on_change=select_experiment)
        return st.session_state.sql_active_experiment


def data_and_model_settings(session_id, history):
    """Render model/data controls and return workflow configuration.

    Parameters
    ----------
    session_id : str
        Active experiment identifier, used to create unique widget keys and to
        associate applied data with the correct event log.
    history : list[dict]
        Ordered events previously loaded with ``store.events``.

    Returns
    -------
    tuple[dict, dict, str]
        ``settings`` contains model name, reasoning effort, provider, and model
        ID. ``data`` describes the current/uploaded/path source. The final
        string contains optional user-supplied data definitions.

    Operation and calls
    -------------------
    Calls ``current_source`` from ``utils.sql_reflection_workflow`` and
    ``model_selector`` from ``models.py``. Applied uploads use
    ``store.snapshot_data``; path databases use ``snapshot_database``; added
    join data uses ``join_source``. ``utils.db_helpers.get_schema`` validates
    the candidate. Data changes are logged by ``store.log_event`` and followed
    by ``st.rerun``. :func:`render` calls this once per page run.
    """
    source = current_source(history)
    with st.expander("Model and data"):
        name, effort, profile = model_selector("sql_model", include_all_models=True)
        if source:
            st.write(f"This chat's data: {source['name']}")
            mode = st.radio("Data change", ["Replace data (new chat)", "Add data for joins"],
                            key=f"sql_data_mode_{session_id}")
            st.caption("Replacing data starts a separate chat. Adding tables keeps this chat and allows JOIN queries.")
        else:
            mode = "Initial data"
        uploaded = st.file_uploader("Browse for data", type=["csv", "db", "sqlite", "sqlite3"],
                                    accept_multiple_files=True,
                                    key=f"sql_upload_{session_id}",
                                    help=("Upload one or more CSV/SQLite files. Additional files are "
                                          "copied into the same snapshot with prefixed table names."))
        path = st.text_input("SQLite database path", str(PROJECT_ROOT / "data" / "products.db"),
                             key=f"sql_path_{session_id}", disabled=bool(uploaded))
        prefix = st.text_input("Prefix for added tables", "extra", key=f"sql_join_prefix_{session_id}") if mode == "Add data for joins" else ""
        if st.button("Apply data", key=f"sql_apply_data_{session_id}"):
            try:
                # Convert an upload or path database into an immutable snapshot.
                candidate = (combine_uploaded_sources(None, uploaded, prefix)
                             if uploaded else snapshot_database(path))
                from utils.db_helpers import get_schema
                # Opening the schema here rejects an invalid database before it
                # becomes the experiment's active source.
                get_schema(candidate["snapshot"])
                if source and mode == "Add data for joins":
                    candidate = (combine_uploaded_sources(source, uploaded, prefix)
                                 if uploaded else join_source(source, candidate, prefix))
                    destination = session_id
                    notice = "Added tables to this chat. Use their prefixed names for JOIN queries."
                elif source and source["sha256"] != candidate["sha256"]:
                    destination = store.new_session()
                    st.session_state.sql_pending_experiment = destination
                    notice = "Data changed: opened a new chat. The previous conversation is saved in Saved experiments."
                else:
                    destination = session_id
                    notice = "Using the selected data in this chat."
                store.log_event(destination, "data", {"source": candidate, "data_action": mode})
                st.session_state.sql_data_notice = notice
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        rules = st.text_area("Data definitions (optional)", key=f"sql_rules_{session_id}")
    source_name = source["name"] if source else "Choose data above or use the database path for your first request"
    st.caption(f"{name} | Data: {source_name}")
    if source:
        st.caption("Changing the model keeps this chat. New data is used only after Apply data.")
    return dict(model=name, effort=effort, provider=profile["provider"], model_id=profile["id"]), {
        "source": source, "uploaded": uploaded if uploaded and not source else None,
        "reuse": bool(source), "previous_source": source, "path": path,
    }, rules


def draw_conversation(session_id, history, answers):
    """Render persisted requests and their automatically executed answers.

    Parameters
    ----------
    session_id : str
        Active experiment identifier. Retained for page-level context; answer
        widget keys use event IDs.
    history : list[dict]
        Complete ordered event log.
    answers : dict[int, dict]
        Display-ready result events returned by ``resolved_answers``.

    Returns
    -------
    None
        The function writes directly to the Streamlit page.

    Operation and calls
    -------------------
    Renders each request and response without visible numbering, reconstructs
    result tables with ``pandas.DataFrame``, supplies CSV downloads, accepts a
    one-time correctness confirmation, and displays helpful guidance or pipeline
    failures. Confirmation writes a ``final`` event with ``store.log_event``.
    It does not execute SQL. :func:`render` calls it after settings/history are
    resolved.
    """
    confirmed_ids = {event["result_id"] for event in history if event["kind"] == "final"}
    for event in history:
        if event["kind"] == "request":
            with st.chat_message("user"):
                st.markdown(event["user_request"])
                st.caption(event["created_at"])
        elif event["kind"] == "result":
            with st.chat_message("assistant"):
                st.markdown(event["answer"])
                if event.get("sql"):
                    st.code(event["sql"], language="sql")
                if event.get("rows"):
                    st.dataframe(pd.DataFrame(**event["rows"]))
                    if event.get("csv"):
                        st.download_button("Download result", event["csv"], f"sql_result_{event['id']}.csv",
                                           key=f"sql_download_{event['id']}")
                if event.get("execution_error"):
                    st.error(event["execution_error"])
                if event.get("truncated"):
                    st.caption("Showing and saving the first 500 rows.")
                confirmed = event["id"] in confirmed_ids
                accepted = st.checkbox(
                    "I received the correct answer",
                    value=confirmed,
                    disabled=confirmed,
                    key=f"sql_confirm_{event['id']}",
                )
                if accepted and not confirmed:
                    store.log_event(session_id, "final", {
                        "result_id": event["id"], "request_id": event["request_id"],
                        "user_request": event["user_request"], "sql": event.get("sql"),
                        "answer": event["answer"], "rows": event.get("rows"),
                        "source": event.get("source"), "confirmed_by_user": True,
                    })
                    st.rerun()
                st.caption(f"{event.get('model', 'Model')} | {event['created_at']}")
        elif event["kind"] == "guidance_result":
            with st.chat_message("assistant"):
                st.error("A valid SQL query could not be produced for this request.")
                st.markdown(event["explanation"])
                st.markdown("**How to improve the request**")
                st.markdown(event["request_advice"])
                st.caption(f"{event.get('model', 'Model')} | {event['created_at']}")
        elif event["kind"] in ("failure", "pipeline_failure"):
            st.error(f"The SQL pipeline failed: {event['error']}")


def draw_investigation_log(session_id, history):
    """Optionally render syntax attempts and the one semantic review.

    Parameters
    ----------
    session_id : str
        Used to create a checkbox key unique to the active experiment.
    history : list[dict]
        Complete event log containing ``syntax_attempt``, ``semantic_review``,
        and possible ``pipeline_failure`` events.

    Returns
    -------
    None
        Renders nothing beyond the checkbox unless it is selected.

    Operation and calls
    -------------------
    Filters investigation events and displays phase, attempt number, SQL,
    SQLite error or success, model explanation, semantic issues, and failures.
    All rendering uses Streamlit APIs. :func:`render` calls this immediately
    after :func:`draw_conversation`.
    """
    if not st.checkbox("Show SQL investigation log", key=f"sql_show_investigation_{session_id}"):
        return

    events = [event for event in history if event["kind"] in {
        "syntax_attempt", "semantic_review", "pipeline_failure",
    }]
    if not events:
        st.caption("No investigation attempts have been logged.")
        return

    with st.expander("SQL investigation details", expanded=True):
        for event in events:
            if event["kind"] == "syntax_attempt":
                phase = event["phase"].replace("_", " ").title()
                st.markdown(f"**{phase} — attempt {event['attempt']}**")
                st.code(event["sql"], language="sql")
                if event.get("execution_error"):
                    st.error(event["execution_error"])
                else:
                    st.success("SQL executed successfully.")
                if event.get("model_explanation"):
                    st.caption(event["model_explanation"])
            elif event["kind"] == "semantic_review":
                st.markdown("**Semantic investigation**")
                if event.get("user_comment"):
                    st.markdown(f"User clarification: {event['user_comment']}")
                for issue in event.get("semantic_issues", []):
                    st.markdown(f"- {issue}")
                st.code(event["output_sql"], language="sql")
            else:
                st.error(event["error"])
def request_composer(session_id, answers):
    """Render the request form and return its cleaned values.

    Parameters
    ----------
    session_id : str
        Included in the form key so experiments keep independent widget state.
    answers : dict[int, dict]
        Existing answers. Their count changes the form key after a successful
        result. Previous answers are not treated as feedback.

    Returns
    -------
    tuple[bool, str, str]
        ``sent`` indicates form submission, ``message`` is the stripped request,
        and ``semantic_comment`` is the stripped optional clarification.

    Operation and calls
    -------------------
    Uses Streamlit ``form``, ``text_area``, and ``form_submit_button``. It makes
    no model or database calls. :func:`render` calls it once per page run and
    uses its return values to decide whether to start ``submit_request``.
    """
    with st.form(f"sql_message_form_{session_id}_{len(answers)}", clear_on_submit=True):
        message = st.text_area("Request",
                               max_chars=st.session_state.get("input_limit", 0) or None,
                               help="Describe the data question the SQL pipeline should answer.")
        semantic_comment = st.text_area(
            "Semantic clarification (optional)",
            help=("Explain business rules, expected calculations, joins, filters, "
                  "grouping, or expected output."),
            placeholder=("Example: Revenue means quantity multiplied by unit price. "
                         "Cancelled transactions must be excluded."),
        )
        sent = st.form_submit_button("Send")
    return sent, message.strip(), semantic_comment.strip()

def final_and_logs(session_id, history, answers):
    """Show user-confirmed result status and provide the complete audit log.

    Parameters
    ----------
    session_id : str
        Included in the export payload and download filename.
    history : list[dict]
        Complete event stream, including every repair attempt and final event.
    answers : dict[int, dict]
        Successful result events used for result counts.

    Returns
    -------
    None
        Writes status/counts/download controls to Streamlit.

    Operation and calls
    -------------------
    Finds the latest automatically created ``final`` event, counts requests and
    semantic reviews, serializes all state with ``json.dumps``, and renders a
    JSON download. It does not select or modify the final answer. :func:`render`
    calls it at the bottom of every page run.
    """
    finals = [event for event in history if event["kind"] == "final"]
    if finals:
        st.success(f"{len(finals)} result(s) confirmed correct by the user.")
    with st.expander("Final answers and investigation log"):
        request_count = sum(event["kind"] == "request" for event in history)
        semantic_count = sum(
            event["kind"] == "semantic_review"
            for event in history
        )
        st.caption(
            f"{request_count} syntactic requests | "
            f"{len(answers)} syntactic answers | "
            f"{semantic_count} semantic reviews. "
            "All iterations are logged automatically."
        )
        export = {
            "session_id": session_id,
            "request_count": request_count,
            "result_count": len(answers),
            "semantic_result_count": semantic_count,
            "final_results": finals,
            "events": history,
        }
        st.download_button("Download review log (JSON)", json.dumps(export, ensure_ascii=False, indent=2),
                           f"sql_experiment_{session_id}.json", mime="application/json")

def render():
    """Render and coordinate the complete SQL-reflection screen.

    Parameters
    ----------
    None
        Inputs come from Streamlit widgets and persisted experiment state.

    Returns
    -------
    None
        Produces the page and may request a Streamlit rerun.

    Operation and calls
    -------------------
    This is the module's public entry point. ``sections.chatbot_section`` calls
    it when the user selects SQL reflection; the Streamlit test also calls it
    directly. It calls :func:`experiment_picker`, ``store.events`` (defined in
    ``utils/sql_reflection_store.py``), ``resolved_answers`` (defined in
    ``utils/sql_reflection_workflow.py``), :func:`data_and_model_settings`,
    :func:`draw_conversation`, :func:`draw_investigation_log`,
    :func:`request_composer`, and :func:`final_and_logs`.

    On a nonempty submission it calls ``submit_request`` inside a spinner. It
    passes ``build_chat_model`` from ``models.py``, :func:`invoke_sql_response`,
    and both response schemas as injected dependencies. After the workflow has
    persisted its result, ``st.rerun`` reloads history so the result appears.
    """
    # Resolve navigation and persisted state at the beginning of every rerun.
    session_id = experiment_picker()
    notice = st.session_state.pop("sql_data_notice", None)
    if notice:
        st.warning(notice)
    history = store.events(session_id)
    answers = resolved_answers(history)
    settings, data, rules = data_and_model_settings(session_id, history)
    draw_conversation(session_id, history, answers)
    draw_investigation_log(session_id, history)
    sent, message, semantic_comment = request_composer(session_id, answers)
    if sent:
        if not message:
            st.error("Write a request before sending.")
        else:
            # The status label is updated from inside the workflow so the user
            # can see both the current phase and syntax attempt number.
            with st.status("Starting SQL investigation...", expanded=True) as pipeline_status:
                def show_phase(label):
                    """Display the current pipeline phase while the request runs."""
                    pipeline_status.update(label=label, state="running")

                submit_request(session_id, history, message, semantic_comment, settings, data,
                               rules, build_chat_model, invoke_sql_response,
                               QueryReview, SemanticReview, QueryFailureGuidance,
                               show_phase)
                pipeline_status.update(label="SQL investigation finished", state="complete")
            st.rerun()
    final_and_logs(session_id, history, answers)
