"""Chart requests, numbered answers, explicit Run, feedback, and review logs."""

from io import BytesIO
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pydantic import BaseModel

from models import model_selector, build_chat_model, extract_answer_text
from utils.file_io import PROJECT_ROOT
from utils.llm_helpers import load_and_prepare_data
from utils import chart_reflection_store as store
from utils.chart_reflection_workflow import (
    add_dataset, current_source, load_frames, mark_final, resolved_answers,
    run_answer, submit_request,
)


class ChartReview(BaseModel):
    feedback: str
    refined_code: str


def load_coffee_data(file_like):
    return load_and_prepare_data(file_like)


def invoke_chart_response(model, profile, schema, prompt):
    if profile["provider"] == "openai":
        return model.with_structured_output(schema).invoke(prompt)
    from langchain.messages import HumanMessage, SystemMessage
    instructions = "Return only a JSON object matching this schema, without markdown:\n" + json.dumps(schema.model_json_schema())
    messages = [SystemMessage(content=instructions)]
    messages.extend([HumanMessage(content=prompt)] if isinstance(prompt, str) else prompt)
    text = extract_answer_text(model.invoke(messages)).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return schema.model_validate_json(text)
    except ValueError as exc:
        raise ValueError("The model returned an invalid chart response. Retry or choose another model.") from exc


def run_chart_code(code, dataframe, datasets=None):
    """Execute only code the user has reviewed and approved."""
    plt.close("all")
    try:
        namespace = {"df": dataframe.copy(), "datasets": {k: v.copy() for k, v in (datasets or {}).items()}, "plt": plt, "pd": pd}
        exec(code, namespace)
        if not plt.get_fignums():
            raise ValueError("No figure was created.")
        buffer = BytesIO()
        plt.gcf().savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close("all")


def experiment_picker():
    pending = st.session_state.pop("chart_pending_experiment", None)
    if pending:
        st.session_state.chart_active_experiment = pending
    with st.expander("Saved chart experiments"):
        if st.button("New chart experiment"):
            st.session_state.chart_active_experiment = store.new_session()
        sessions = store.list_sessions()
        if not sessions:
            st.session_state.chart_active_experiment = store.new_session()
            sessions = store.list_sessions()
        titles = {identifier: f"{title.replace('New SQL experiment', 'New chart experiment')} ({date[:10]})" for identifier, title, date in sessions}
        active = st.session_state.get("chart_active_experiment")
        if active not in titles:
            active = next(iter(titles))
            st.session_state.chart_active_experiment = active
        st.session_state.chart_experiment_picker = active

        def select_experiment():
            st.session_state.chart_active_experiment = st.session_state.chart_experiment_picker

        st.selectbox("Saved chart experiments", list(titles), index=None, format_func=titles.get,
                     key="chart_experiment_picker", on_change=select_experiment)
    return st.session_state.chart_active_experiment


def preserve_previous_drafts(session_id):
    """Carry any in-memory V1/V2 work into the durable numbered conversation."""
    if st.session_state.get("chart_drafts_migrated") or not st.session_state.get("chart_v1"):
        return
    raw = st.session_state.get("chart_data_signature")
    if not isinstance(raw, bytes):
        st.warning("An earlier chart draft exists, but its data snapshot is unavailable for recovery.")
        return
    source = store.snapshot_csv(raw, "Previous chart data.csv")
    store.log_event(session_id, "data", dict(source=source, data_action="Recovered previous drafts"))
    for version in ("v1", "v2"):
        key = f"chart_{version}"
        if key not in st.session_state:
            continue
        message = st.session_state.get("chart_request", "Previous chart request")
        request = store.log_event(session_id, "request", dict(user_request=message, model="Previous model", recovered=True))
        result = store.log_event(session_id, "result", dict(
            request_id=request, user_request=message, model="Previous model", source=source,
            code=st.session_state[key], answer=st.session_state.get("chart_feedback", "Recovered chart draft"),
            image_path=None, execution_error=None, recovered=True,
        ))
        if f"{key}_editor" in st.session_state:
            st.session_state[f"chart_code_{result}"] = st.session_state[f"{key}_editor"]
        if st.session_state.get(f"{key}_image"):
            store.log_event(session_id, "execution", dict(
                result_id=result, executed_code=st.session_state.get(f"{key}_executed_code", st.session_state[key]),
                image_path=store.save_image(st.session_state[f"{key}_image"]), execution_error=None,
            ))
    st.session_state.chart_drafts_migrated = True


def data_and_model_settings(session_id, history):
    source = current_source(history)
    with st.expander("Model and data"):
        name, effort, profile = model_selector("chart_model", include_all_models=True)
        if source:
            st.write(f"This chat's data: {source['name']}")
            mode = st.radio("Data change", ["Replace data (new chat)", "Add data for joins"], key=f"chart_data_mode_{session_id}")
            st.caption("Apply replacement data to start a new chat. Add a dataset to keep this chat.")
        else:
            mode = "Initial data"
        uploaded = st.file_uploader("Browse for data", type=["csv"], key=f"chart_upload_{session_id}")
        path = st.text_input("CSV path", str(PROJECT_ROOT / "data" / "coffee_sales.csv"),
                             key=f"chart_path_{session_id}", disabled=uploaded is not None)
        alias = st.text_input("Additional dataset name", "extra", key=f"chart_alias_{session_id}") if mode == "Add data for joins" else ""
        if st.button("Apply data", key=f"chart_apply_data_{session_id}"):
            try:
                raw = uploaded.getvalue() if uploaded else Path(path).read_bytes()
                load_coffee_data(BytesIO(raw))
                candidate = store.snapshot_csv(raw, uploaded.name if uploaded else Path(path).name)
                destination = session_id
                if source and mode == "Add data for joins":
                    candidate = add_dataset(source, candidate, alias)
                    notice = f"Added datasets[{alias!r}] to this chat for joins."
                elif source and (source['sha256'] != candidate['sha256'] or source.get('datasets')):
                    destination = store.new_session()
                    st.session_state.chart_pending_experiment = destination
                    notice = "Data changed: opened a new chart chat. The previous conversation is saved."
                else:
                    notice = "Using the selected CSV in this chat."
                store.log_event(destination, "data", dict(source=candidate, data_action=mode))
                st.session_state.chart_data_notice = notice
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if source:
            frame, extras = load_frames(source)
            st.dataframe(frame.head())
            if extras:
                st.caption("Additional datasets: " + ", ".join(extras))
    st.caption(f"{name} | Data: {source['name'] if source else 'Choose CSV data or use the sample'}")
    if not profile["vision"]:
        st.caption("This model reviews code and data; it cannot see chart images.")
    return dict(model=name, effort=effort, model_id=profile["id"],
                provider=profile["provider"], vision=profile["vision"]), source, uploaded, path


def draw_conversation(session_id, history, answers):
    numbers = {answer["id"]: number for number, answer in answers.items()}
    requests = {event["id"]: number for number, event in enumerate((e for e in history if e["kind"] == "request"), 1)}
    for event in history:
        if event["kind"] == "request":
            with st.chat_message("user"):
                st.markdown(event["user_request"])
                st.caption(f"Request {requests[event['id']]} | {event['created_at']}")
        elif event["kind"] == "result":
            number = numbers[event["id"]]
            answer = answers[number]
            with st.chat_message("assistant"):
                st.markdown(f"**Answer {number}**")
                st.markdown(answer["answer"])
                if answer.get("code"):
                    editor = f"chart_code_{answer['id']}"
                    code = st.text_area(f"Code for Answer {number}", answer["code"], height=200, key=editor)
                    approval_key = f"chart_approved_{answer['id']}"
                    reviewed_key = f"chart_reviewed_{answer['id']}"
                    if st.session_state.get(reviewed_key) != code:
                        st.session_state[approval_key] = False
                        st.session_state[reviewed_key] = code
                    approved = st.checkbox(f"I reviewed Answer {number} and approve running this code", key=approval_key,
                                           help="This code runs on your computer with Python's access; it is not sandboxed.")
                    if st.button(f"Run Answer {number}", disabled=not approved, key=f"chart_run_{answer['id']}"):
                        run_answer(session_id, history, number, code, run_chart_code)
                        st.rerun()
                    st.download_button("Download code", code, f"answer_{number}.py", key=f"chart_code_download_{answer['id']}")
                    if answer.get("executed_code") and code != answer["executed_code"]:
                        st.caption("The chart below is from the last executed code, before your current edits.")
                if answer.get("image_path"):
                    image = Path(answer["image_path"]).read_bytes()
                    st.image(image, caption=f"Executed chart for Answer {number}")
                    st.download_button("Download chart", image, f"answer_{number}.png", key=f"chart_image_download_{answer['id']}")
                if answer.get("execution_error"):
                    st.error(answer["execution_error"])
                st.caption(f"{answer.get('model', 'Model')} | {answer['created_at']}")
        elif event["kind"] == "failure":
            st.error(f"Request {requests.get(event['request_id'], '?')} failed: {event['error']}")


def message_composer(session_id, answers):
    latest = next((n for n in reversed(answers) if answers[n].get("code")), None)
    with st.form(f"chart_message_form_{session_id}_{len(answers)}", clear_on_submit=True):
        message = st.text_area("Feedback / next request" if answers else "Request",
                               max_chars=st.session_state.get("input_limit", 0) or None)
        with st.expander("Compare answers or change the starting answer"):
            comparison = st.multiselect("Answers to compare", list(answers), format_func=lambda n: f"Answer {n}")
            base = st.selectbox("Improve answer", [None] + list(answers), index=list(answers).index(latest) + 1 if latest else 0,
                                format_func=lambda n: "New request" if n is None else f"Answer {n}")
            st.caption("Choose two or more answers and press Compare selected answers. The comparison message is optional.")
        sent = st.form_submit_button("Send")
        compare_sent = st.form_submit_button("Compare selected answers") if len(answers) >= 2 else False
    return sent or compare_sent, message.strip(), base, comparison, compare_sent


def final_and_logs(session_id, history, answers):
    numbers = {answer["id"]: number for number, answer in answers.items()}
    finals = [event for event in history if event["kind"] == "final"]
    if finals:
        st.success(f"Final: Answer {numbers[finals[-1]['result_id']]}")
    with st.expander("Final answer and review log"):
        if answers:
            with st.form(f"chart_final_form_{session_id}_{len(answers)}"):
                number = st.selectbox("Final answer", list(answers), index=len(answers) - 1, format_func=lambda n: f"Answer {n}")
                note = st.text_input("Why is this the best answer? (optional)")
                if st.form_submit_button("Mark as final"):
                    mark_final(session_id, history, number, note)
                    st.rerun()
        count = sum(event["kind"] == "request" for event in history)
        st.caption(f"{count} requests | {len(answers)} answers. All iterations and Run actions are logged automatically.")
        export = dict(session_id=session_id, request_count=count, answer_count=len(answers),
                      final_result=finals[-1] if finals else None,
                      events=[dict(answers[numbers[e['id']]], answer_number=numbers[e['id']]) if e['kind'] == 'result' else e for e in history])
        st.download_button("Download review log (JSON)", json.dumps(export, ensure_ascii=False, indent=2),
                           f"chart_experiment_{session_id}.json", mime="application/json")


def render():
    session_id = experiment_picker()
    preserve_previous_drafts(session_id)
    notice = st.session_state.pop("chart_data_notice", None)
    if notice:
        st.warning(notice)
    history = store.events(session_id)
    answers = resolved_answers(history)
    settings, source, uploaded, path = data_and_model_settings(session_id, history)
    draw_conversation(session_id, history, answers)
    sent, message, base, comparison, compare_sent = message_composer(session_id, answers)
    if sent:
        if compare_sent and len(comparison) < 2:
            st.error("Choose two or more numbers in Answers to compare first.")
        elif len(comparison) == 1:
            st.error("Select at least two answers to compare, or clear the selection.")
        elif not message and not comparison:
            st.error("Write a request or feedback before sending.")
        else:
            if not source and not comparison:
                try:
                    raw = uploaded.getvalue() if uploaded else Path(path).read_bytes()
                    load_coffee_data(BytesIO(raw))
                    source = store.snapshot_csv(raw, uploaded.name if uploaded else Path(path).name)
                    store.log_event(session_id, "data", dict(source=source))
                except Exception as exc:
                    st.error(str(exc))
                    return
            message = message or "Compare the selected answers and explain which is better and why."
            with st.spinner("Preparing your answer..."):
                submit_request(session_id, history, message, settings, source, base, comparison,
                               build_chat_model, invoke_chart_response, ChartReview)
            st.rerun()
    final_and_logs(session_id, history, answers)
