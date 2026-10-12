"""Registry, read-only SQL and sidebar regressions without live services."""
import tempfile, sqlite3, unittest
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest
from image_tools.effects_menu import EFFECTS
from sections.image_section import convert_parameters_for_function
from utils.database import run_readonly_query
from image_tools.drawing import draw_filled_highlight
ROOT=Path(__file__).resolve().parents[1]
class IntegrationTests(unittest.TestCase):
 def test_all_registered_effects(self):
  image=np.random.RandomState(0).randint(0,256,(600,800,3),dtype=np.uint8)
  original=image.copy()
  for name,definition in EFFECTS.items():
   with self.subTest(effect=name):
    values={}
    for p in definition['parameters']:
     if p['type']=='select': value=p['options'][p['default']]
     elif p['type']=='file': value=image.copy()
     elif p['type']=='color': value=(0,0,255)
     else: value=p['default']
     values[p['name']]=value
    output=definition['function'](image,**convert_parameters_for_function(name,values))
    if isinstance(output,tuple): output=output[0]
    self.assertTrue(output.size and np.isfinite(output).all())
    np.testing.assert_array_equal(image,original)
 def test_highlight(self):
  image=np.full((10,10,3),100,np.uint8)
  np.testing.assert_array_equal(draw_filled_highlight(image,(2,2),(5,5),(0,0,255),0),image)
  result=draw_filled_highlight(image,(2,2),(5,5),(0,0,255),1)
  np.testing.assert_array_equal(result[3,3],[0,0,255])
  np.testing.assert_array_equal(result[0,0],image[0,0])
 def test_readonly_sql(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'test.db'
   with sqlite3.connect(path) as c:
    c.execute('CREATE TABLE items (amount INTEGER)');c.execute('INSERT INTO items VALUES (3)')
   for sql in ['DELETE FROM items','DROP TABLE items',"ATTACH DATABASE ':memory:' AS extra",'PRAGMA user_version=1']:
    self.assertIn('error',run_readonly_query(sql,path).columns)
   self.assertEqual(run_readonly_query('SELECT amount FROM items',path).iloc[0,0],3)
 def test_navigation_and_chained_edits(self):
  app=AppTest.from_file(str(ROOT/'app.py')).run()
  def click(label):
   next(b for b in app.button if b.label==label).click().run()
   self.assertFalse(app.exception);self.assertFalse(app.error)
  for name in ['Video','Conversion','ChatBot','File','Help','Settings']: click(name)
  click('Models');app.multiselect[0].set_value(['GPT-4o mini']).run();click('ChatBot');click('Conversation')
  self.assertEqual(app.selectbox[0].options,['GPT-4o mini'])
  self.assertTrue(any(x.label=='Your message' for x in app.text_area))
  click('Image')
  image=np.random.RandomState(1).randint(0,256,(20,30,3),dtype=np.uint8)
  app.session_state['source_image']=image.copy();app.session_state['result_image']=image.copy()
  click('Effects');click('Geometry');click('Flip')
  flipped=app.session_state['result_image'].copy();app.run()
  np.testing.assert_array_equal(flipped,app.session_state['result_image'])
  app.toggle[0].set_value(False).run()
  for _ in range(2):
   next(b for b in app.button if b.label=='Apply').click().run()
   np.testing.assert_array_equal(flipped,app.session_state['result_image'])
  click('Blur');click('Mean blur');np.testing.assert_array_equal(flipped,app.session_state['working_base'])
  next(b for b in app.button if b.label=='Reset to original').click().run()
  np.testing.assert_array_equal(image,app.session_state['result_image'])
  self.assertIsNone(app.session_state['effect_name'])
  click('ChatBot');click('SQL reflection');click('Chart reflection')
  self.assertEqual([b.label for b in app.sidebar.button], ['Image','Video','Conversion','ChatBot','File','Help','Settings'])
  self.assertFalse(any(x.label=='Your message' for x in app.text_area))
