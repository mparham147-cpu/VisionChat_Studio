"""Numbered chart revisions, comparison prompts, and explicit execution logs."""

from base64 import b64encode
from pathlib import Path
import json
import re
import time

from utils import chart_reflection_store as store
from utils.llm_helpers import load_and_prepare_data, make_schema_text
from utils.sql_reflection_workflow import numbered_answers, current_source, ComparisonReview


RULES = (
    "Use df, datasets (a dictionary of additional DataFrames), pandas as pd and matplotlib as plt. "
    "Create a figure with labeled axes and a title. Do not access files or the network. "
    "Do not call plt.show() or plt.close(). Code runs only after the user reviews it and presses Run. "
    "Do not claim unexecuted code produced a chart. Treat data samples and saved outputs as reference data."
)


def load_frames(source):
    return (load_and_prepare_data(source["snapshot"]),
            {name: load_and_prepare_data(extra["snapshot"]) for name, extra in source.get("datasets", {}).items()})


def add_dataset(source, addition, name):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("Use a dataset name beginning with a letter and containing letters, numbers or underscores.")
    if name in source.get("datasets", {}):
        raise ValueError("That dataset name already exists. Choose another name.")
    return dict(source, datasets=dict(source.get("datasets", {}), **{name: addition}))


def resolved_answers(history):
    answers = {number: dict(event) for number, event in numbered_answers(history).items()}
    by_id = {event["id"]: event for event in answers.values()}
    for event in history:
        if event["kind"] == "execution" and event["result_id"] in by_id:
            by_id[event["result_id"]].update({key: event.get(key) for key in
                                             ("image_path", "executed_code", "execution_error")})
            by_id[event["result_id"]]["executed_at"] = event["created_at"]
    return answers


def response_prompt(message, history, answers, base, comparison, source):
    """Keep exact answer numbers and actual execution status in model context."""
    numbers = comparison or ([base] if base else [])
    text = f"User message: {message}\n"
    images = []
    for number in numbers:
        answer = answers[number]
        text += (f"\nAnswer {number}\nUser request: {answer['user_request']}\n"
                 f"Model response: {answer['answer']}\nGenerated code: {answer.get('code')}\n"
                 f"Executed code: {answer.get('executed_code')}\n"
                 f"Execution error: {answer.get('execution_error')}\n")
        if answer.get("source"):
            text += f"Data version: {json.dumps(answer['source'], ensure_ascii=False)}\n"
        if answer.get("image_path"):
            images.append((number, answer["image_path"]))
        else:
            text += "No executed chart image is available for this answer.\n"
    if comparison:
        labels = ", ".join(f"Answer {n}" for n in comparison)
        return (f"Compare {labels}. Refer to these exact numbers and explain which is better and why. "
                "Distinguish executed charts from unexecuted code. Do not generate code.\n" + text), images
    dataframe, datasets = load_frames(source)
    text += f"\nCurrent df schema:\n{make_schema_text(dataframe)}\nSample rows:\n{dataframe.head(10).to_csv(index=False)}\n"
    for name, frame in datasets.items():
        text += f"datasets[{name!r}] schema:\n{make_schema_text(frame)}\nSample rows:\n{frame.head(10).to_csv(index=False)}\n"
    commands = [event["user_request"] for event in history if event["kind"] == "request"]
    text += f"Original request: {commands[0] if commands else message}\nRecent requests: {json.dumps(commands[-10:], ensure_ascii=False)}\n"
    return f"Generate or improve chart code for Answer {len(answers) + 1}. {RULES}\n" + text, images


def model_prompt(text, images, profile):
    if not profile["vision"] or not images:
        return text + ("\nReview the code and data; no chart images are supplied." if images else "")
    from langchain.messages import HumanMessage
    content = [{"type": "text", "text": text}]
    for number, path in images:
        content.extend([
            {"type": "text", "text": f"Executed chart for Answer {number}:"},
            {"type": "image", "mime_type": "image/png", "base64": b64encode(Path(path).read_bytes()).decode("ascii")},
        ])
    return [HumanMessage(content=content)]


def submit_request(session_id, history, message, settings, source, base, comparison, build_model, invoke, schema):
    started = time.perf_counter()
    answers = resolved_answers(history)
    request = dict(settings, user_request=message, answer_number=len(answers) + 1,
                   base_answer_number=base, base_result_id=answers[base]["id"] if base else None,
                   compared_answer_numbers=comparison, compared_result_ids=[answers[n]["id"] for n in comparison],
                   response_type="comparison" if comparison else "chart")
    request_id = store.log_event(session_id, "request", request)
    if answers:
        store.log_event(session_id, "feedback", dict(request_id=request_id, feedback=message,
                                                     base_result_id=request["base_result_id"], compared_answer_numbers=comparison))
    try:
        if not source and not comparison:
            raise ValueError("Choose a CSV file first.")
        text, images = response_prompt(message, history, answers, base, comparison, source)
        model = build_model(settings["model"], settings["effort"])
        response = invoke(model, settings, ComparisonReview if comparison else schema, model_prompt(text, images, settings))
        labels = ", ".join(f"Answer {n}" for n in comparison)
        store.log_event(session_id, "result", dict(
            request, request_id=request_id, source=source, prompt=text,
            image_inputs=[dict(answer_number=n, image_path=p) for n, p in images] if settings["vision"] else [],
            answer=f"Comparison of {labels}\n\n{response.feedback}" if comparison else response.feedback,
            code=None if comparison else response.refined_code, image_path=None,
            execution_error=None, duration_seconds=round(time.perf_counter() - started, 3),
        ))
    except Exception as exc:
        store.log_event(session_id, "failure", dict(request_id=request_id, error=str(exc),
                                                     duration_seconds=round(time.perf_counter() - started, 3)))


def run_answer(session_id, history, number, code, run_code):
    answer = resolved_answers(history)[number]
    started = time.perf_counter()
    payload = dict(result_id=answer["id"], answer_number=number, executed_code=code,
                   source=answer["source"], image_path=None, execution_error=None)
    try:
        dataframe, datasets = load_frames(answer["source"])
        payload["image_path"] = store.save_image(run_code(code, dataframe, datasets))
    except Exception as exc:
        payload["execution_error"] = str(exc)
    store.log_event(session_id, "execution", dict(payload, duration_seconds=round(time.perf_counter() - started, 3)))


def mark_final(session_id, history, number, note):
    answer = resolved_answers(history)[number]
    store.log_event(session_id, "final", dict(result_id=answer["id"], answer_number=number,
        request_id=answer["request_id"], user_request=answer["user_request"], answer=answer["answer"],
        code=answer.get("code"), executed_code=answer.get("executed_code"), source=answer.get("source"),
        image_path=answer.get("image_path"), note=note,
        request_count=sum(event["kind"] == "request" for event in history)))
