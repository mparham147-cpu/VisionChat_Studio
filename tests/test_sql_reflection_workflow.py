"""SQL experiment persistence and user-driven iteration checks."""

import sqlite3
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

from utils import sql_reflection_store as store
from utils.database import run_readonly_query


class SQLExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_changed_data_keeps_old_results_reproducible(self):
        first = store.snapshot_data(b"color,units\nred,3\nblue,4\n", "sales.csv", self.root)
        second = store.snapshot_data(b"color,units\nred,8\nblue,4\n", "sales.csv", self.root)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(run_readonly_query("SELECT SUM(units) AS total FROM uploaded_data", first["snapshot"]).iloc[0]["total"], 7)
        self.assertEqual(run_readonly_query("SELECT SUM(units) AS total FROM uploaded_data", second["snapshot"]).iloc[0]["total"], 12)
        self.assertTrue(Path(first["original"]).exists())
        denied = run_readonly_query("DELETE FROM uploaded_data", first["snapshot"])
        self.assertIn("error", denied.columns)

    def test_requests_final_and_failures_survive_reopening(self):
        session = store.new_session(self.root)
        request = store.log_event(session, "request", {"user_request": "Total sales"}, self.root)
        result = store.log_event(session, "result", {"request_id": request, "sql": "SELECT 1"}, self.root)
        next_request = store.log_event(session, "request", {"user_request": "Calculate revenue"}, self.root)
        store.log_event(session, "failure", {"request_id": next_request, "error": "Provider unavailable"}, self.root)
        store.log_event(session, "final", {"result_id": result, "request_count": 2}, self.root)
        history = store.events(session, self.root)
        self.assertEqual(sum(event["kind"] == "request" for event in history), 2)
        self.assertEqual(history[-1]["result_id"], result)
        self.assertEqual(store.list_sessions(self.root)[0][1], "Total sales")
        other = store.new_session(self.root)
        self.assertEqual(store.events(other, self.root), [])

    def test_sqlite_upload_and_duplicate_csv_snapshot(self):
        path = self.root / "input.sqlite"
        connection = sqlite3.connect(path)
        try:
            with connection:
                connection.execute("CREATE TABLE values_table (value INTEGER)")
                connection.execute("INSERT INTO values_table VALUES (9)")
        finally:
            connection.close()
        source = store.snapshot_data(path.read_bytes(), "input.sqlite", self.root)
        self.assertEqual(run_readonly_query("SELECT value FROM values_table", source["snapshot"]).iloc[0]["value"], 9)
        first = store.snapshot_data(b"x\n1\n", "file.csv", self.root)
        second = store.snapshot_data(b"x\n1\n", "file.csv", self.root)
        self.assertEqual(first, second)

    def test_multiple_uploads_are_combined_for_join_queries(self):
        from utils.sql_reflection_workflow import combine_uploaded_sources

        class Upload:
            def __init__(self, name, raw):
                self.name = name
                self.raw = raw

            def getvalue(self):
                return self.raw

        with patch.object(store, "snapshot_data", partial(store.snapshot_data, root=self.root)):
            source = combine_uploaded_sources(None, [
                Upload("orders.csv", b"customer_id,total\n1,10\n2,20\n"),
                Upload("customers.csv", b"customer_id,name\n1,Ada\n2,Lin\n"),
            ])
        joined = run_readonly_query(
            "SELECT o.total, c.name FROM uploaded_data o "
            "JOIN customers_2_uploaded_data c USING (customer_id) ORDER BY o.total",
            source["snapshot"],
        )
        self.assertEqual(joined.values.tolist(), [[10, "Ada"], [20, "Lin"]])

    def test_unconvertible_request_logs_helpful_guidance(self):
        from contextlib import ExitStack
        from sections import sql_reflection_section as section
        from utils.sql_reflection_workflow import submit_request

        source = store.snapshot_data(b"units\n3\n", "input.csv", self.root)
        session = store.new_session(self.root)

        def invoke(_model, _settings, schema, _prompt):
            if schema.__name__ == "QueryFailureGuidance":
                return section.QueryFailureGuidance(
                    explanation="The requested sales table does not exist.",
                    request_advice="Ask for total units from uploaded_data.",
                )
            return section.QueryReview(
                explanation="Trying the requested table.",
                refined_sql="SELECT * FROM sales",
            )

        with ExitStack() as stack:
            for name in ("log_event", "events"):
                stack.enter_context(patch.object(store, name, partial(getattr(store, name), root=self.root)))
            submit_request(
                session, [], "Show sales", "", {
                    "model": "Test model", "effort": None,
                    "provider": "ollama", "model_id": "test",
                }, {
                    "source": source, "uploaded": None, "reuse": True,
                    "previous_source": source, "path": source["snapshot"],
                }, "", lambda *_args: object(), invoke,
                section.QueryReview, section.SemanticReview,
                section.QueryFailureGuidance,
            )

        history = store.events(session, self.root)
        self.assertEqual(sum(event["kind"] == "syntax_attempt" for event in history), 5)
        self.assertEqual(history[-1]["kind"], "guidance_result")
        self.assertIn("does not exist", history[-1]["explanation"])
        self.assertFalse(any(event["kind"] in ("result", "final") for event in history))

    def test_ui_requests_and_user_confirmed_final_result(self):
        from contextlib import ExitStack
        from streamlit.testing.v1 import AppTest
        from sections import sql_reflection_section as section

        source = store.snapshot_data(b"units\n3\n4\n", "input.csv", self.root)
        prompts = []
        syntax_calls = 0

        def respond(model, profile, schema, prompt):
            nonlocal syntax_calls
            prompts.append(prompt)
            if schema.__name__ == "QueryFailureGuidance":
                return schema(explanation="The requested field is unavailable.",
                              request_advice="Ask using columns present in the schema.")
            if schema.__name__ == "SemanticReview":
                return schema(semantic_issues=[], explanation="The query matches the request.",
                              corrected_sql="SELECT SUM(units) AS total FROM uploaded_data")
            syntax_calls += 1
            if syntax_calls == 1:
                return section.QueryReview(explanation="Initial draft", refined_sql="SELECT * FROM missing_table")
            return section.QueryReview(explanation="Computed total", refined_sql="SELECT SUM(units) AS total FROM uploaded_data")

        def select_model(*args, **kwargs):
            name = section.st.selectbox("Model", ["Test model", "Other model"], key="sql_model")
            return name, None, {"id": name, "provider": "ollama"}

        with ExitStack() as stack:
            for name in ("new_session", "list_sessions", "log_event", "events", "snapshot_data"):
                stack.enter_context(patch.object(store, name, partial(getattr(store, name), root=self.root)))
            stack.enter_context(patch.object(section, "model_selector", side_effect=select_model))
            stack.enter_context(patch.object(section, "build_chat_model", return_value=object()))
            stack.enter_context(patch.object(section, "invoke_sql_response", side_effect=respond))
            app = AppTest.from_string("from sections.sql_reflection_section import render\nrender()", default_timeout=15).run()
            self.assertFalse(app.exception)

            def labeled(elements, label):
                return next(element for element in elements if element.label == label)

            labeled(app.text_input, "SQLite database path").set_value(source["snapshot"])
            labeled(app.text_area, "Request").set_value("Total units")
            labeled(app.button, "Send").click().run()
            self.assertFalse(app.exception)
            session = store.list_sessions()[0][0]
            before = store.events(session)
            self.assertEqual(section.resolved_answers(before)[1]["rows"]["data"], [[7]])
            self.assertEqual(sum(event["kind"] == "semantic_review" for event in before), 1)
            first_attempts = [event for event in before if event["kind"] == "syntax_attempt"
                              and event["phase"] == "initial_syntax"]
            self.assertEqual(len(first_attempts), 2)
            self.assertIn("no such table", first_attempts[0]["execution_error"])
            self.assertIsNone(first_attempts[1]["execution_error"])
            self.assertEqual(sum(event["kind"] == "final" for event in before), 0)
            labeled(app.checkbox, "I received the correct answer").check().run()
            self.assertFalse(app.exception)
            before = store.events(session)
            self.assertEqual(sum(event["kind"] == "final" for event in before), 1)
            labeled(app.selectbox, "Model").select("Other model").run()
            self.assertFalse(app.exception)
            self.assertEqual(store.events(session), before)
            self.assertEqual(app.session_state["sql_active_experiment"], session)
            labeled(app.text_area, "Request").set_value("Calculate total units with a clear label")
            labeled(app.button, "Send").click().run()
            self.assertFalse(app.exception)
            self.assertIn("Calculate total units", prompts[-1])
            self.assertFalse(any("Previous SQL" in prompt for prompt in prompts[-2:]))
            self.assertIn("7", prompts[-1])
            session = store.list_sessions()[0][0]
            history = store.events(session)
            answers = section.resolved_answers(history)
            self.assertEqual(list(answers), [1, 2])
            self.assertNotEqual(answers[2]["id"], 2)
            self.assertEqual(sum(event["kind"] == "request" for event in history), 2)
            self.assertEqual(sum(event["kind"] == "final" for event in history), 1)
            [box for box in app.checkbox if box.label == "I received the correct answer"][-1].check().run()
            self.assertFalse(app.exception)
            history = store.events(session)
            self.assertEqual(sum(event["kind"] == "final" for event in history), 2)
            self.assertEqual(history[-1]["kind"], "final")
            self.assertEqual(history[-1]["result_id"], answers[2]["id"])
            self.assertTrue(history[-1]["confirmed_by_user"])
            self.assertEqual(answers[2]["model"], "Other model")

            replacement = store.snapshot_data(b"units\n20\n", "replacement.csv")
            labeled(app.text_input, "SQLite database path").set_value(replacement["snapshot"])
            labeled(app.button, "Apply data").click().run()
            self.assertFalse(app.exception)
            replacement_session = app.session_state["sql_active_experiment"]
            self.assertNotEqual(replacement_session, session)
            self.assertEqual(section.resolved_answers(store.events(replacement_session)), {})
            self.assertEqual(len(section.resolved_answers(store.events(session))), 2)
            self.assertTrue(any("opened a new chat" in warning.value for warning in app.warning))

            labeled(app.selectbox, "Saved SQL experiments").select(session).run()
            self.assertFalse(app.exception)
            labeled(app.selectbox, "Model").select("Test model").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["sql_active_experiment"], session)
            self.assertEqual(labeled(app.selectbox, "Saved SQL experiments").value, session)
            self.assertEqual(len(section.resolved_answers(store.events(session))), 2)
            labeled(app.radio, "Data change").set_value("Add data for joins").run()
            labeled(app.text_input, "SQLite database path").set_value(replacement["snapshot"])
            labeled(app.button, "Apply data").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["sql_active_experiment"], session)
            joined_source = section.current_source(store.events(session))
            joined = run_readonly_query("SELECT a.units, b.units FROM uploaded_data a CROSS JOIN extra_uploaded_data b", joined_source["snapshot"])
            self.assertEqual(len(joined), 2)
            self.assertNotIn("error", joined.columns)
            self.assertEqual(section.current_source(store.events(session))["sha256"], joined_source["sha256"])
            self.assertEqual(section.resolved_answers(store.events(session))[1]["rows"]["data"], [[7]])

    def test_legacy_answers_get_consecutive_numbers_despite_log_gaps(self):
        from utils.sql_reflection_workflow import numbered_answers
        history = [dict(id=16, kind="result"), dict(id=17, kind="data"),
                   dict(id=18, kind="failure"), dict(id=20, kind="result")]
        answers = numbered_answers(history)
        self.assertEqual(list(answers), [1, 2])
        self.assertEqual(answers[1]["id"], 16)
        self.assertEqual(answers[2]["id"], 20)


if __name__ == "__main__":
    unittest.main()
