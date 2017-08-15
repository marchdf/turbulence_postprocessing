# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure

# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
import pandas as pd
import scipy.integrate as spi


# ========================================================================
#
# Function definitions
#
# ========================================================================
def energy_spectra(U, N, L):
    """
    Get the 1D and 3D energy spectra from 3D data.

    :param U: momentum, [:math:`u`, :math:`v`, :math:`w`]
    :type U: list.
    :param N: number of points, [:math:`n_x`, :math:`n_y`, :math:`n_z`]
    :type N: list.
    :param L: domain lengths, [:math:`L_x`, :math:`L_y`, :math:`L_z`]
    :type L: list.
    :return: Dataframe of 1D and 3D energy spectra
    :rtype: dataframe.
    """

    # FFT of fields
    Uf = [np.fft.fftn(U[0]),
          np.fft.fftn(U[1]),
          np.fft.fftn(U[2])]

    # Wavenumbers
    k0 = np.fft.fftfreq(Uf[0].shape[0]) * N[0]
    k1 = np.fft.fftfreq(Uf[0].shape[1]) * N[1]
    k2 = np.fft.fftfreq(Uf[0].shape[2]) * N[2]
    kmax = [k0.max(), k1.max(), k2.max()]
    K = np.meshgrid(k0, k1, k2)
    kmag = np.sqrt(K[0]**2 + K[1]**2 + K[2]**2)
    halfN = np.array([int(n / 2) for n in N], dtype=np.int64)

    # Calculate the 1D energy spectra Eii(kj)
    df = pd.DataFrame(columns=['name', 'k', 'E'])
    for i in range(3):

        # Energy
        Eiif = np.real(Uf[i] * np.conj(Uf[i])) / (np.prod(N)**2)

        # Wavenumber spacing (shell thickness)
        dkdk = float(kmax[(i + 1) % 3]) / N[(i + 1) % 3] \
            * float(kmax[(i + 2) % 3]) / N[(i + 2) % 3]

        for j in range(3):

            # Binning
            kbins = np.arange(0, halfN[j] + 1)
            whichbin = np.digitize(np.abs(K[j]).flat, kbins)
            ncount = np.bincount(whichbin)

            # Summation in each wavenumber bin
            E = np.zeros(len(kbins))
            for n in range(1, len(ncount)):
                E[n - 1] = 2 * np.sum(Eiif.flat[whichbin == n]) * dkdk
            E[E < 1e-13] = 0.0

            # Store the data
            subdf = pd.DataFrame(columns=['name', 'k', 'E'])
            subdf['k'] = kbins
            subdf['E'] = E
            subdf['name'] = 'E{0:d}{0:d}(k{1:d})'.format(i, j)
            df = pd.concat([df, subdf], ignore_index=True)

    # Calculate the 3D energy spectra
    E3D = np.real(Uf[0] * np.conj(Uf[0])
                  + Uf[1] * np.conj(Uf[1])
                  + Uf[2] * np.conj(Uf[2])) / (np.prod(N)**2)

    # Wavenumber spacing (shell thickness)
    dkdkdk = np.prod(kmax) / np.prod(N)

    # Binning
    kbins = np.arange(0, halfN[0] + 1)
    whichbin = np.digitize(kmag.flat, kbins)
    ncount = np.bincount(whichbin)

    # Summation in each wavenumber bin
    E = np.zeros(len(kbins))
    for n in range(1, len(ncount)):
        E[n - 1] = 2 * np.pi * np.sum(E3D.flat[whichbin == n]) * dkdkdk
    E[E < 1e-13] = 0.0

    # Store the data
    subdf = pd.DataFrame(columns=['name', 'k', 'E'])
    subdf['k'] = kbins
    subdf['E'] = E
    subdf['name'] = 'E3D'
    df = pd.concat([df, subdf], ignore_index=True)

    return df


# ========================================================================
def integral_length_scale_tensor(U, N, L):
    """
    Calculate the integral lengthscale tensor.

    :math:`L_{ij} = \\frac{1}{R_{ii}(0)} \\int_0^\\infty R_{ii}(e_j r) \\mathrm{d} r`
    where :math:`R_{ij}(r) = \\langle u_i(x) u_j(x+r) \\rangle`

    :param U: momentum, [:math:`u`, :math:`v`, :math:`w`]
    :type U: list.
    :param N: number of points, [:math:`n_x`, :math:`n_y`, :math:`n_z`]
    :type N: list.
    :param L: domain lengths, [:math:`L_x`, :math:`L_y`, :math:`L_z`]
    :type L: list.
    :return: Array of the lengthscale tensor, :math:`Lij`
    :rtype: array.
    """

    dr = L / N
    Lij = np.zeros((3, 3))
    halfN = np.array([int(n / 2) for n in N], dtype=np.int64)

    for i in range(3):
        for j in range(3):

            Rij = np.zeros(halfN[j] + 1)

            for m in range(N[j]):
                for r in range(halfN[j] + 1):
                    if j == 0:
                        Rij[r] += np.sum((U[i][m, :, :]
                                          * U[i][(m + r) % N[j], :, :]))
                    elif j == 1:
                        Rij[r] += np.sum((U[i][:, m, :]
                                          * U[i][:, (m + r) % N[j], :]))
                    elif j == 2:
                        Rij[r] += np.sum((U[i][:, :, m]
                                          * U[i][:, :, (m + r) % N[j]]))
            Rij = Rij / np.prod(N)
            Lij[i, j] = spi.simps(Rij, dx=dr[j]) / Rij[0]

    return Lij


# ========================================================================
def structure_functions(U, N, L):
    """
    Calculate the longitudinal and transverse structure functions.

    :math:`D_{ij} = \\int_V (u_i(x+r)-u_i(x)) (u_j(x+r)-u_j(x)) \\mathrm{d} V`
    and :math:`S_{L} = D_{00}`, :math:`S_{T1} = D_{11}`, :math:`S_{T2} = D_{22}`.

    :param U: momentum, [:math:`u`, :math:`v`, :math:`w`]
    :type U: list.
    :param N: number of points, [:math:`n_x`, :math:`n_y`, :math:`n_z`]
    :type N: list.
    :param L: domain lengths, [:math:`L_x`, :math:`L_y`, :math:`L_z`]
    :type L: list.
    :return: Dataframe of structure functions
    :rtype: dataframe.
    """

    # Get the structure functions
    SL = np.zeros(N[0])
    ST1 = np.zeros(N[0])
    ST2 = np.zeros(N[0])
    for i in range(N[0]):
        for r in range(N[0]):
            SL[r] += np.sum((U[0][(i + r) % N[0], :, :] - U[0][i, :, :])**2)
            ST1[r] += np.sum((U[1][(i + r) % N[0], :, :] - U[1][i, :, :])**2)
            ST2[r] += np.sum((U[2][(i + r) % N[0], :, :] - U[2][i, :, :])**2)

    # Store the data
    df = pd.DataFrame(columns=['r', 'SL', 'ST1', 'ST2'])
    df['r'] = np.linspace(0, L[0], N[0])
    df['SL'] = SL / np.prod(N)
    df['ST1'] = ST1 / np.prod(N)
    df['ST2'] = ST2 / np.prod(N)

    return df
