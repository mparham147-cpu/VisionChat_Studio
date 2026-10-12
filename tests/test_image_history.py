"""Explicit menu selection and exact image Undo/Redo regressions."""
import unittest
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest
from utils.image_history import initialize_history, record_edit, undo_image, redo_image, reset_image
ROOT=Path(__file__).resolve().parents[1]

class ImageHistoryTests(unittest.TestCase):
    def setUp(self):
        self.original=np.random.RandomState(1).randint(0,256,(20,30,3),dtype=np.uint8)
        self.state={'source_image':self.original.copy(),'result_image':self.original.copy(),'result_label':'Original'}
        initialize_history(self.state)

    def test_previews_record_once_undo_redo_exact(self):
        self.state['edit_pending']=True
        record_edit(self.state)
        self.state['result_image']=np.full_like(self.original,30)
        record_edit(self.state)
        self.state['result_image']=np.full_like(self.original,60)
        self.assertEqual(len(self.state['image_undo']),1)
        self.assertTrue(undo_image(self.state))
        np.testing.assert_array_equal(self.state['result_image'],self.original)
        self.assertIsNone(self.state['effect_name'])
        self.assertTrue(redo_image(self.state))
        self.assertTrue((self.state['result_image']==60).all())

    def test_reset_undo_and_branch(self):
        self.state['result_image']=np.full_like(self.original,60)
        reset_image(self.state)
        np.testing.assert_array_equal(self.state['result_image'],self.original)
        undo_image(self.state)
        self.assertTrue((self.state['result_image']==60).all())
        self.state['edit_pending']=True
        record_edit(self.state)
        self.assertEqual(self.state['image_redo'],[])
        initialize_history(self.state)
        self.assertEqual(self.state['image_undo'],[])

    def test_initial_menus_never_change_color_and_toolbar_every_task(self):
        app=AppTest.from_file(str(ROOT/'app.py')).run()
        self.assertIsNone(app.session_state['section'])
        def click(label):
            next(b for b in app.button if b.label==label).click().run()
            self.assertFalse(app.exception);self.assertFalse(app.error)
        click('Image')
        self.assertIsNone(app.session_state['image_task'])
        app.session_state['source_image']=self.original.copy()
        app.session_state['result_image']=self.original.copy()
        click('Effects')
        self.assertIsNone(app.session_state['effect_group'])
        np.testing.assert_array_equal(app.session_state['result_image'],self.original)
        click('Color and values')
        self.assertIsNone(app.session_state['effect_name'])
        np.testing.assert_array_equal(app.session_state['result_image'],self.original)
        click('Grayscale')
        gray=app.session_state['result_image'].copy()
        self.assertEqual(gray.ndim,2)
        click('Undo')
        np.testing.assert_array_equal(app.session_state['result_image'],self.original)
        self.assertIsNone(app.session_state['effect_name'])
        click('Redo')
        np.testing.assert_array_equal(app.session_state['result_image'],gray)
        click('Reset to original')
        np.testing.assert_array_equal(app.session_state['result_image'],self.original)
        for task in ['Inspect','Save','Upload','Blank canvas']:
            click(task)
            self.assertTrue({'Reset to original','Undo','Redo'}.issubset({b.label for b in app.button}))
