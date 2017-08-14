import unittest
import numpy as np
import numpy.testing as npt
import turbulence_postprocessing as tp


class TurbulencePostProcessingTestCase(unittest.TestCase):
    """Tests for `turbulence_postprocessing.py`."""

    def setUp(self):

        self.N = [32, 32, 32]
        self.L = [2 * np.pi, 2 * np.pi, 2 * np.pi]
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
        print(spectra)
        npt.assert_array_almost_equal(0, 1)


if __name__ == '__main__':
    unittest.main()
