import unittest
import numpy as np
from prototype_phd.data_utils import generate_qmc_samples

class TestGenerateQMCSamples(unittest.TestCase):
    def test_sample_ranges_and_shape(self):
        param_limits = {'a': (0, 10), 'b': (5, 15)}
        n_samples = 1000
        samples = generate_qmc_samples(param_limits, n_samples)
        
        # Check that the output dictionary has the correct keys.
        self.assertIn('a', samples)
        self.assertIn('b', samples)
        
        # Check that each sample array has the correct shape.
        self.assertEqual(samples['a'].shape, (n_samples,))
        self.assertEqual(samples['b'].shape, (n_samples,))
        
        # Check that sample values are within the defined ranges.
        self.assertTrue(np.all(samples['a'] >= 0))
        self.assertTrue(np.all(samples['a'] <= 10))
        self.assertTrue(np.all(samples['b'] >= 5))
        self.assertTrue(np.all(samples['b'] <= 15))
        
    def test_quasi_randomness(self):
        # Two separate calls should produce different sequences since scrambling is enabled.
        param_limits = {'x': (0, 1)}
        n_samples = 100
        samples1 = generate_qmc_samples(param_limits, n_samples)['x']
        samples2 = generate_qmc_samples(param_limits, n_samples)['x']
        # Check that at least one element is different.
        self.assertFalse(np.allclose(samples1, samples2))

if __name__ == '__main__':
    unittest.main()