# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure

import unittest
import numpy as np
import numpy.testing as npt
import turbulence_postprocessing.turbulence_postprocessing as tp


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

    def test_energy_spectra(self):
        """Is the energy spectra calculation correct?"""
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

    def test_structure_functions(self):
        """Is the structure function calculation correct?"""
        structure = tp.structure_functions(
            [self.U, self.V, self.W], self.N, self.L)
        npt.assert_array_almost_equal(structure.SL,
                                      np.array([0.,
                                                0.3013132971454286,
                                                1.0238627309018322,
                                                1.7385989137636211,
                                                2.0290518004781566,
                                                1.7328957100929185,
                                                1.0315084708761031,
                                                0.3389158307627682,
                                                0.0581018731482762,
                                                0.3476535108824844,
                                                1.0331229227730068,
                                                1.7120169980026754,
                                                1.9892984566194272,
                                                1.70615536360969,
                                                1.030285607843781,
                                                0.3568253757404142,
                                                0.0782862747188336]),
                                      decimal=10)

        npt.assert_array_almost_equal(structure.ST1,
                                      np.array([0.,
                                                1.0178163823125026,
                                                2.0015211367847403,
                                                1.0150513353640416,
                                                0.1180048334623851,
                                                1.0656938951979218,
                                                1.893143904464391,
                                                0.9886110282366569,
                                                0.2135234550866583,
                                                1.0735734216745583,
                                                1.8132640956728139,
                                                0.9960430563200141,
                                                0.275482055930023,
                                                1.0543843535145818,
                                                1.7709728609211628,
                                                1.0232015273797213,
                                                0.2969253153556487]),
                                      decimal=10)

        npt.assert_array_almost_equal(structure.ST2,
                                      np.array([0.,
                                                0.0785096199929016,
                                                0.301723786914015,
                                                0.6350383967300699,
                                                1.0273238570414358,
                                                1.4187430801470575,
                                                1.7498429304380465,
                                                1.9705439637367888,
                                                2.0476761798337342,
                                                1.9699335870800814,
                                                1.7495128845335444,
                                                1.4202020757179299,
                                                1.0322182020004451,
                                                0.6445799848193108,
                                                0.3161686585952469,
                                                0.0968242917758608,
                                                0.0198170012870652]),
                                      decimal=10)


if __name__ == '__main__':
    unittest.main()
