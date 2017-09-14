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

    The 3D energy spectrum is defined as (see Eq. 6.188 in Pope):

    - :math:`E_{3D}(k) = \\frac{1}{2} \\int_S \\phi_{ii}(k_0,k_1,k_2) \\mathrm{d}S(k)`

    where :math:`k=\\sqrt{k_0^2 + k_1^2 + k_2^2}`.

    .. note::

       For the 3D spectrum, the integral is approximated by averaging
       :math:`\\phi_{ii}(k_0,k_1,k_2)` over a binned :math:`k` and
       multiplying by the surface area of the sphere at :math:`k`. The
       bins are defined by rounding the wavenumber :math:`k` to the
       closest integer. An average of every :math:`k` in each bin is
       then used as the return value for that bin.

    :param U: momentum, [:math:`u`, :math:`v`, :math:`w`]
    :type U: list
    :param N: number of points, [:math:`n_x`, :math:`n_y`, :math:`n_z`]
    :type N: list
    :param L: domain lengths, [:math:`L_x`, :math:`L_y`, :math:`L_z`]
    :type L: list
    :return: Dataframe of 1D and 3D energy spectra
    :rtype: dataframe
    """

    # =========================================================================
    # Setup

    # Initialize empty dataframe
    df = pd.DataFrame(columns=['name', 'k', 'E'])

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
    kbins = np.hstack((-1e-16,
                       np.arange(0.5, halfN[0] - 1),
                       halfN[0] - 1))

    # Energy in Fourier space
    Ef = 0.5 / (np.prod(N)**2) * (np.absolute(Uf[0])**2
                                  + np.absolute(Uf[1])**2
                                  + np.absolute(Uf[2])**2)

    # Filter the data with ellipsoid filter
    ellipse = (K[0] / kmax[0])**2 \
        + (K[1] / kmax[1])**2 \
        + (K[2] / kmax[2])**2 <= 1.0
    Ef = np.where(ellipse, Ef, np.nan)
    K = np.where(ellipse, K[0], np.nan),\
        np.where(ellipse, K[1], np.nan),\
        np.where(ellipse, K[2], np.nan)
    kmag = np.where(ellipse, kmag, np.nan)

    # =========================================================================
    # 1D spectra Eii(kj)

    for j in range(3):
        jm = (j + 1) % 3
        jn = (j + 2) % 3

        # Binning
        whichbin = np.digitize(np.abs(K[j]).flat, kbins, right=True)
        ncount = np.bincount(whichbin)

        # Average in each wavenumber bin
        E = np.zeros(len(kbins) - 2)
        for k, n in enumerate(range(1, len(kbins) - 1)):
            area = np.pi * kmax[jm] * kmax[jn] * np.sqrt(1 - (k / kmax[j])**2)
            E[k] = np.mean(Ef.flat[whichbin == n]) * area
        E[E < 1e-13] = 0.0

        # Store the data
        subdf = pd.DataFrame(columns=['name', 'k', 'E'])
        subdf['k'] = np.arange(0, kmax[j])
        subdf['E'] = E
        subdf['name'] = 'E{0:d}{0:d}(k{1:d})'.format(j, j)
        df = pd.concat([df, subdf], ignore_index=True)

    # =========================================================================
    # 3D spectrum

    # Multiply spectra by the surface area of the sphere at kmag.
    E3D = 4.0 * np.pi * kmag**2 * Ef

    # Binning
    whichbin = np.digitize(kmag.flat, kbins, right=True)
    ncount = np.bincount(whichbin)

    # Average in each wavenumber bin
    E = np.zeros(len(kbins) - 1)
    kavg = np.zeros(len(kbins) - 1)
    for k, n in enumerate(range(1, len(kbins))):
        E[k] = np.mean(E3D.flat[whichbin == n])
        kavg[k] = np.mean(kmag.flat[whichbin == n])
    E[E < 1e-13] = 0.0

    # Store the data
    subdf = pd.DataFrame(columns=['name', 'k', 'E'])
    subdf['k'] = kavg
    subdf['E'] = E
    subdf['name'] = 'E3D'
    df = pd.concat([df, subdf], ignore_index=True)

    return df


# ========================================================================
def dissipation(U, N, L, viscosity):
    """
    Get the dissipation.

    The dissipation is defined as (see Eq. 6.160 in Pope):

    - :math:`\\epsilon = 2 \\nu \\sum_k k^2 E({\\mathbf{k}})`

    where :math:`k=\\sqrt{k_0^2 + k_1^2 + k_2^2}`.

    :param U: momentum, [:math:`u`, :math:`v`, :math:`w`]
    :type U: list
    :param N: number of points, [:math:`n_x`, :math:`n_y`, :math:`n_z`]
    :type N: list
    :param L: domain lengths, [:math:`L_x`, :math:`L_y`, :math:`L_z`]
    :type L: list
    :param viscosity: kinematic viscosity
    :type viscosity: double
    :return: dissipation
    :rtype: double
    """

    # FFT of fields
    Uf = [np.fft.fftn(U[0]),
          np.fft.fftn(U[1]),
          np.fft.fftn(U[2])]

    # Wavenumbers
    k0 = np.fft.fftfreq(Uf[0].shape[0]) * N[0]
    k1 = np.fft.fftfreq(Uf[0].shape[1]) * N[1]
    k2 = np.fft.fftfreq(Uf[0].shape[2]) * N[2]
    K = np.meshgrid(k0, k1, k2)
    kmag2 = K[0]**2 + K[1]**2 + K[2]**2

    # Energy in Fourier space
    Ef = 0.5 / (np.prod(N)**2) * (np.absolute(Uf[0])**2
                                  + np.absolute(Uf[1])**2
                                  + np.absolute(Uf[2])**2)

    return 2.0 * viscosity * np.sum(kmag2 * Ef)


# ========================================================================
def integral_length_scale_tensor(U, N, L):
    """
    Calculate the integral lengthscale tensor.

    :math:`L_{ij} = \\frac{1}{R_{ii}(0)} \\int_0^\\infty R_{ii}(e_j r) \\mathrm{d} r`
    where :math:`R_{ij}(\\mathbf{r}) = \\langle u_i(\\mathbf{x}) u_j(\\mathbf{x}+\\mathbf{r}) \\rangle`

    :param U: momentum, [:math:`u`, :math:`v`, :math:`w`]
    :type U: list
    :param N: number of points, [:math:`n_x`, :math:`n_y`, :math:`n_z`]
    :type N: list
    :param L: domain lengths, [:math:`L_x`, :math:`L_y`, :math:`L_z`]
    :type L: list
    :return: Array of the lengthscale tensor, :math:`L_{ij}`
    :rtype: array
    """

    dr = L / N
    Lij = np.zeros((3, 3))
    halfN = np.array([int(n / 2) for n in N], dtype=np.int64)

    for i in range(3):
        for j in range(3):

            idxm = (j + 1) % 3
            idxn = (j + 2) % 3

            Uf = np.fft.rfft(U[i], axis=j)

            if j == 0:
                Rii = np.sum(np.fft.irfft(Uf * np.conj(Uf), axis=j)[:halfN[i] + 1, :, :],
                             axis=(idxm, idxn))
            elif j == 1:
                Rii = np.sum(np.fft.irfft(Uf * np.conj(Uf), axis=j)[:, :halfN[i] + 1, :],
                             axis=(idxm, idxn))
            elif j == 2:
                Rii = np.sum(np.fft.irfft(Uf * np.conj(Uf), axis=j)[:, :, :halfN[i] + 1],
                             axis=(idxm, idxn))
            Rii = Rii / np.prod(N)
            Lij[i, j] = spi.simps(Rii, dx=dr[j]) / Rii[0]

    return Lij


# ========================================================================
def structure_functions(U, N, L):
    """
    Calculate the longitudinal and transverse structure functions.

    :math:`D_{ij}(r) = \\int_V (u_i(x+r,y,z)-u_i(x,y,z)) (u_j(x+r,y,z)-u_j(x,y,z)) \\mathrm{d} V`

    and :math:`S_{L} = D_{00}`, :math:`S_{T1} = D_{11}`, :math:`S_{T2} = D_{22}`.

    :param U: momentum, [:math:`u`, :math:`v`, :math:`w`]
    :type U: list
    :param N: number of points, [:math:`n_x`, :math:`n_y`, :math:`n_z`]
    :type N: list
    :param L: domain lengths, [:math:`L_x`, :math:`L_y`, :math:`L_z`]
    :type L: list
    :return: Dataframe of structure functions (:math:`S_{L}`, :math:`S_{T1}`, and :math:`S_{T2}`)
    :rtype: dataframe
    """

    # Get the structure functions
    halfN = np.array([int(n / 2) for n in N], dtype=np.int64)
    SL = np.zeros(halfN[0] + 1)
    ST1 = np.zeros(halfN[0] + 1)
    ST2 = np.zeros(halfN[0] + 1)
    for i in range(N[0]):
        for r in range(halfN[0] + 1):
            SL[r] += np.sum((U[0][(i + r) % N[0], :, :] - U[0][i, :, :])**2)
            ST1[r] += np.sum((U[1][(i + r) % N[0], :, :] - U[1][i, :, :])**2)
            ST2[r] += np.sum((U[2][(i + r) % N[0], :, :] - U[2][i, :, :])**2)

    # Store the data
    df = pd.DataFrame(columns=['r', 'SL', 'ST1', 'ST2'])
    df['r'] = L[0] / N[0] * np.arange(halfN[0] + 1)
    df['SL'] = SL / np.prod(N)
    df['ST1'] = ST1 / np.prod(N)
    df['ST2'] = ST2 / np.prod(N)

    return df
