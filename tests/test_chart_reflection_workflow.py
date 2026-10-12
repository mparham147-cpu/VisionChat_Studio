"""Chart chat continuity, explicit execution, comparison, and data changes."""

from contextlib import ExitStack
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from utils import chart_reflection_store as store
from utils import chart_reflection_workflow as workflow


class ChartReflectionTests(unittest.TestCase):
    def test_ui_model_switch_run_compare_final_and_data_changes(self):
        from streamlit.testing.v1 import AppTest
        from sections import chart_reflection_section as section

        code = "plt.figure()\nplt.bar(df['label'], df['units'])"
        prompts = []

        def select_model(*args, **kwargs):
            name = section.st.selectbox("Model", ["Vision model", "Text model"], key="chart_model")
            provider = "openai" if name == "Vision model" else "ollama"
            return name, None, dict(id=name, provider=provider, vision=name == "Vision model")

        def respond(model, profile, schema, prompt):
            prompts.append(prompt)
            if schema.__name__ == "ComparisonReview":
                return schema(feedback="Answer 2 has a clearer title than Answer 1.")
            return schema(feedback="Suggested chart", refined_code=code)

        def labeled(elements, label):
            return next(element for element in elements if element.label == label)

        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            csv = root / "input.csv"
            csv.write_text("label,units\nA,3\nB,4\n", encoding="utf-8")
            for name in ("new_session", "list_sessions", "log_event", "events", "snapshot_csv", "save_image"):
                stack.enter_context(patch.object(store, name, partial(getattr(store, name), root=root)))
            stack.enter_context(patch.object(section, "model_selector", side_effect=select_model))
            stack.enter_context(patch.object(section, "build_chat_model", return_value=object()))
            stack.enter_context(patch.object(section, "invoke_chart_response", side_effect=respond))
            app = AppTest.from_string("from sections.chart_reflection_section import render\nrender()", default_timeout=20).run()
            self.assertFalse(app.exception)
            labeled(app.text_input, "CSV path").set_value(str(csv))
            labeled(app.text_area, "Request").set_value("Plot units")
            labeled(app.button, "Send").click().run()
            self.assertFalse(app.exception)
            session = store.list_sessions()[0][0]
            history = store.events(session)
            self.assertIsNone(workflow.resolved_answers(history)[1]["image_path"])
            self.assertTrue(labeled(app.button, "Run Answer 1").disabled)
            labeled(app.checkbox, "I reviewed Answer 1 and approve running this code").check().run()
            labeled(app.button, "Run Answer 1").click().run()
            self.assertFalse(app.exception)
            answer = workflow.resolved_answers(store.events(session))[1]
            self.assertTrue(Path(answer["image_path"]).exists())
            self.assertEqual(answer["executed_code"], code)

            labeled(app.text_area, "Code for Answer 1").set_value(code + "\nplt.title('Units')").run()
            self.assertTrue(labeled(app.button, "Run Answer 1").disabled)
            labeled(app.checkbox, "I reviewed Answer 1 and approve running this code").check().run()
            labeled(app.button, "Run Answer 1").click().run()
            self.assertFalse(app.exception)

            history = store.events(session)
            labeled(app.selectbox, "Model").select("Text model").run()
            self.assertFalse(app.exception)
            self.assertEqual(store.events(session), history)
            self.assertEqual(app.session_state["chart_active_experiment"], session)
            labeled(app.text_area, "Feedback / next request").set_value("Improve the title")
            labeled(app.button, "Send").click().run()
            self.assertFalse(app.exception)
            self.assertIsInstance(prompts[-1], str)
            self.assertIn("plt.title('Units')", prompts[-1])

            labeled(app.multiselect, "Answers to compare").set_value([1, 2])
            labeled(app.button, "Compare selected answers").click().run()
            self.assertFalse(app.exception)
            self.assertIn("Compare Answer 1, Answer 2", prompts[-1])
            answers = workflow.resolved_answers(store.events(session))
            self.assertEqual(list(answers), [1, 2, 3])
            self.assertIn("Comparison of Answer 1, Answer 2", answers[3]["answer"])
            labeled(app.selectbox, "Final answer").select(1)
            labeled(app.button, "Mark as final").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(store.events(session)[-1]["answer_number"], 1)

            replacement = root / "replacement.csv"
            replacement.write_text("label,units\nC,20\n", encoding="utf-8")
            labeled(app.text_input, "CSV path").set_value(str(replacement))
            labeled(app.button, "Apply data").click().run()
            self.assertFalse(app.exception)
            self.assertNotEqual(app.session_state["chart_active_experiment"], session)
            self.assertTrue(any("opened a new chart chat" in warning.value for warning in app.warning))
            labeled(app.selectbox, "Saved chart experiments").select(session).run()
            labeled(app.selectbox, "Model").select("Vision model").run()
            self.assertEqual(app.session_state["chart_active_experiment"], session)
            labeled(app.radio, "Data change").set_value("Add data for joins").run()
            labeled(app.text_input, "CSV path").set_value(str(replacement))
            labeled(app.button, "Apply data").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["chart_active_experiment"], session)
            source = workflow.current_source(store.events(session))
            self.assertIn("extra", source["datasets"])
            original = workflow.resolved_answers(store.events(session))[1]["source"]
            self.assertNotIn("datasets", original)
            frame, extras = workflow.load_frames(source)
            self.assertEqual(extras["extra"]["units"].sum(), 20)
            labeled(app.selectbox, "Improve answer").select(1)
            labeled(app.text_area, "Feedback / next request").set_value("Use the additional dataset")
            labeled(app.button, "Send").click().run()
            self.assertFalse(app.exception)
            self.assertIsInstance(prompts[-1], list)
            blocks = prompts[-1][0].content
            self.assertTrue(any(block["type"] == "image" for block in blocks))
            self.assertIn("datasets['extra']", blocks[0]["text"])

    def test_invalid_code_is_logged_without_erasing_generated_answer(self):
        from sections.chart_reflection_section import run_chart_code
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            for name in ("new_session", "events", "log_event", "snapshot_csv", "save_image"):
                stack.enter_context(patch.object(store, name, partial(getattr(store, name), root=root)))
            source = store.snapshot_csv(b"x\n1\n", "input.csv")
            session = store.new_session()
            request = store.log_event(session, "request", dict(user_request="Plot x"))
            store.log_event(session, "result", dict(request_id=request, user_request="Plot x", answer="Code", source=source, code="raise ValueError('bad code')"))
            workflow.run_answer(session, store.events(session), 1, "raise ValueError('bad code')", run_chart_code)
            answers = workflow.resolved_answers(store.events(session))
            self.assertEqual(len(answers), 1)
            self.assertEqual(answers[1]["execution_error"], "bad code")
            self.assertIsNone(answers[1]["image_path"])


if __name__ == "__main__":
    unittest.main()
