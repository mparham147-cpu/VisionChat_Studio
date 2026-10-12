"""Noise boundary, kernel parsing and filter UI regressions; no external calls."""
import unittest
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest
from image_tools import noise, operations
from sections.image_section import parse_kernel
ROOT = Path(__file__).resolve().parents[1]

class NoiseFilterTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((100, 100, 3), 128, np.uint8)

    def test_noise_repeatability_and_source_preservation(self):
        generators = [
            (noise.apply_salt_noise, (.2, 42)),
            (noise.apply_pepper_noise, (.2, 42)),
            (noise.apply_salt_pepper_noise, (.2, .5, 42)),
            (noise.apply_uniform_noise, (-20, 20, 42)),
            (noise.apply_speckle_noise, (.15, 42)),
            (noise.apply_poisson_noise, (100, 42)),
            (operations.apply_gaussian_noise, (20, 42, 0)),
        ]
        before = self.image.copy()
        for function, arguments in generators:
            with self.subTest(function=function.__name__):
                first = function(self.image, *arguments)
                np.testing.assert_array_equal(first, function(self.image, *arguments))
                self.assertEqual(first.shape, self.image.shape)
                self.assertEqual(first.dtype, np.uint8)
                np.testing.assert_array_equal(self.image, before)

    def test_impulse_boundaries_and_whole_pixels(self):
        np.testing.assert_array_equal(noise.apply_salt_noise(self.image, 0, 0), self.image)
        self.assertTrue((noise.apply_salt_noise(self.image, 1, 0) == 255).all())
        self.assertTrue((noise.apply_pepper_noise(self.image, 1, 0) == 0).all())
        mixed = noise.apply_salt_pepper_noise(self.image, 1, .5, 0)
        self.assertEqual(set(np.unique(mixed)), {0, 255})
        np.testing.assert_array_equal(mixed[:,:,0], mixed[:,:,1])
        self.assertLess(abs(float((mixed[:,:,0]==255).mean())-.5), .03)

    def test_zero_strength_and_black_signal(self):
        np.testing.assert_array_equal(operations.apply_gaussian_noise(self.image, 0, 42), self.image)
        np.testing.assert_array_equal(noise.apply_uniform_noise(self.image, 0, 0, 42), self.image)
        np.testing.assert_array_equal(noise.apply_speckle_noise(self.image, 0, 42), self.image)
        black = np.zeros_like(self.image)
        np.testing.assert_array_equal(noise.apply_poisson_noise(black, 100, 42), black)
        clipped = operations.apply_gaussian_noise(self.image, 0, 42, 255)
        self.assertTrue((clipped==255).all())

    def test_invalid_values(self):
        for function,args in [(noise.apply_salt_noise,(1.1,42)),
                              (noise.apply_salt_pepper_noise,(.1,-.1,42)),
                              (noise.apply_uniform_noise,(10,-10,42)),
                              (noise.apply_speckle_noise,(-1,42)),
                              (noise.apply_poisson_noise,(0,42)),
                              (operations.apply_gaussian_noise,(-1,42))]:
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function(self.image,*args)

    def test_grayscale_noise(self):
        gray = self.image[:,:,0]
        for function,args in [(noise.apply_salt_noise,(.1,42)),
                              (noise.apply_speckle_noise,(.1,42)),
                              (noise.apply_poisson_noise,(100,42))]:
            self.assertEqual(function(gray,*args).shape, gray.shape)

    def test_kernel_identity_and_errors(self):
        kernel = parse_kernel('0,0,0; 0,1,0; 0,0,0')
        np.testing.assert_array_equal(operations.apply_custom_convolution(self.image,kernel,False,0),self.image)
        for bad in ['a,b', '1,2; 3,4', '1,2,3; 4', 'nan', 'inf']:
            with self.subTest(kernel=bad), self.assertRaises(ValueError): parse_kernel(bad)
        with self.assertRaises(ValueError):
            operations.apply_custom_convolution(self.image,np.zeros((3,3)),True,0)

    def test_pipeline_chances_and_threshold(self):
        kwargs = dict(mean_chance=0, gaussian_chance=0, sharpen_chance=0,
                      median_chance=0, flip_chance=0, rotate_chance=0, noise_chance=0)
        result, steps = operations.apply_random_pipeline(self.image,42,**kwargs)
        np.testing.assert_array_equal(result,self.image); self.assertEqual(steps,[])
        kwargs['noise_chance']=1
        result, steps=operations.apply_random_pipeline(self.image,42,noise_sigma=0,noise_mean=10,**kwargs)
        self.assertEqual(steps,['Gaussian noise']); self.assertTrue((result==138).all())
        gray=np.array([[0,128,255]],np.uint8)
        np.testing.assert_array_equal(operations.apply_binary_threshold(gray,128),[[0,0,255]])

    def test_ui_hover_only_and_custom_kernel_errors(self):
        app=AppTest.from_file(str(ROOT/'app.py')).run()
        app.session_state['source_image']=self.image.copy()
        app.session_state['result_image']=self.image.copy()
        def click(label):
            next(b for b in app.button if b.label==label).click().run()
            self.assertFalse(app.exception)
        click('Image'); click('Effects'); click('Noise'); click('Salt and pepper noise')
        self.assertFalse(app.error)
        self.assertFalse(any(' — ' in c.value for c in app.caption))
        first=app.session_state['result_image'].copy()
        app.run(); np.testing.assert_array_equal(first,app.session_state['result_image'])
        click('Detail and noise');click('Custom convolution')
        next(t for t in app.text_area if t.label=='Kernel Text').set_value('invalid').run()
        self.assertTrue(app.error); self.assertFalse(app.exception)
