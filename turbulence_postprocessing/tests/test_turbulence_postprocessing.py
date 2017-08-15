# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure

import unittest
import numpy as np
import numpy.testing as npt
import turbulence_postprocessing as tp


class TurbulencePostProcessingTestCase(unittest.TestCase):
    """Tests for `turbulence_postprocessing.py`."""

    def setUp(self):

        self.N = np.array([32, 32, 32], dtype=np.int64)
        self.L = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi])
        x = np.linspace(0, self.L[0], self.N[0])
        y = np.linspace(0, self.L[1], self.N[1])
        z = np.linspace(0, self.L[2], self.N[2])

        X, Y, Z = np.meshgrid(x, y, z)
        self.U = np.cos(2 * X) + np.cos(4 * Y) + np.cos(8 * Z)
        self.V = np.cos(4 * X) + np.cos(8 * Y) + np.cos(2 * Z)
        self.W = np.cos(8 * X) + np.cos(2 * Y) + np.cos(4 * Z)

    def test_1Denergy_spectra(self):
        """Is the 1D energy spectra calculation correct?"""
        spectra = tp.energy_spectra([self.U, self.V, self.W], self.N, self.L)

        npt.assert_array_almost_equal(spectra[spectra['name'] == 'E00(k0)'].E,
                                      np.array([4.5619010925292969e-01,
                                                1.4605263152219195e-03,
                                                2.2386120212584928e-01,
                                                6.7442813799552270e-04,
                                                1.0720979017672000e-04,
                                                3.3203539948815836e-05,
                                                1.3551381799540367e-05,
                                                6.4286163047664882e-06,
                                                3.3376728983783624e-06,
                                                1.8279914430695195e-06,
                                                1.0269758955349610e-06,
                                                5.7614524958903159e-07,
                                                3.1222570170418413e-07,
                                                1.5503327531902072e-07,
                                                6.3177488588052565e-08,
                                                1.5006493455757903e-08,
                                                0.0000000000000000e+00]),
                                      decimal=10)

    def test_3Denergy_spectra(self):
        """Is the 3D energy spectra calculation correct?"""
        spectra = tp.energy_spectra([self.U, self.V, self.W], self.N, self.L)
        npt.assert_array_almost_equal(spectra[spectra['name'] == 'E3D'].E,
                                      np.array([1.7063476476572133e-02,
                                                1.4615375922598896e-02,
                                                9.9962258903049606e-01,
                                                2.4447579277973940e-02,
                                                9.5824424787481777e-01,
                                                2.5039243549310627e-02,
                                                1.8173922860806487e-02,
                                                4.2345279320411822e-02,
                                                8.1622388707478410e-01,
                                                8.0940125943099031e-02,
                                                1.1821913455429101e-02,
                                                3.7035253765588423e-03,
                                                1.4513906165973757e-03,
                                                5.9257366520192262e-04,
                                                2.1456841319592457e-04,
                                                4.7812003225643672e-05,
                                                0.0000000000000000e+00]),
                                      decimal=10)

    def test_integral_length_scale_tensor(self):
        """Is the integral length scale tensor calculation correct?"""
        lengthscales = tp.integral_length_scale_tensor(
            [self.U, self.V, self.W], self.N, self.L)

        npt.assert_array_almost_equal(lengthscales,
                                      np.array([[2.1003226357018558,
                                                 2.100322635701855,
                                                 2.1003226357018563],
                                                [2.1003226357018563,
                                                 2.1003226357018558,
                                                 2.1003226357018554],
                                                [2.100322635701855,
                                                 2.1003226357018563,
                                                 2.1003226357018558]]),
                                      decimal=10)


if __name__ == '__main__':
    unittest.main()
