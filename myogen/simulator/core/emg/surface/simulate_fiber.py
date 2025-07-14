#######################################################################################################
##################################### Initial Explanations ############################################
#######################################################################################################

# This script simulates a single fiber: it must be runned in the script simulate_muscle_XXX.py, which simulates
# this script several times and generates the MUAPs for each MU dectected at each electrode channel.
# Length parameters are in mm, frequencies in kHz, time in ms, conductivities in S/m


#######################################################################################################
##################################### Input Parameters ################################################
#######################################################################################################

# Fs             -> Sampling frequency
# mean_conduction_velocity__mm_s              -> Conduction velocity
# N              -> Number of points in t and z domains
# M              -> Number of points in theta domain
# r              -> Model total radius
# r_bone         -> Bone radius
# th_fat         -> Fat thickness
# th_skin        -> Skin thickness
# R              -> Source position in rho coordinate
# L1             -> Semifiber length (z > 0)
# L2             -> Semifiber length (z < 0)
# zi             -> Innervation zone (z = 0 for the mean innervation zone of the M.U)
# alpha          -> Inclination angle (in degrees) between the eletrode matrix and the muscle fibers
# channels       -> Matrix of electrodes (tuple). The columns are aligned with the muscle fibers if alpha = 0
# electrode_grid_center         -> Matrix of electrodes electrode_grid_center (tuple -> (z electrode_grid_center in mm, theta electrode_grid_center in degrees))
# d_ele          -> Distance between neighboring electrodes
# rele           -> Electrode radius (circular electrodes)
# sig_bone       -> Bone conductivity
# sig_muscle_rho -> Muscle conductivity in rho direction
# sig_muscle_z   -> Muscle conductivity in z direction
# sig_fat        -> Fat conductivity
# sig_skin       -> Skin conductivity


#######################################################################################################
##################################### Model ###########################################################
#######################################################################################################

import math

import numpy as np
from scipy.special import iv as In
from scipy.special import jv as Jn
from scipy.special import kv as Kn

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def In_tilde(K_THETA, x) -> np.ndarray:
    return (In(K_THETA + 1, x) + In(K_THETA - 1, x)) / 2


def Kn_tilde(K_THETA, x) -> np.ndarray:
    return (Kn(K_THETA + 1, x) + Kn(K_THETA - 1, x)) / 2


def simulate_fiber(
    Fs,
    v,
    N,
    M,
    r,
    r_bone,
    th_fat,
    th_skin,
    R,
    L1,
    L2,
    zi,
    alpha,
    channels,
    center,
    d_ele,
    rele,
    sig_muscle_rho,
    sig_muscle_z,
    sig_fat,
    sig_skin,
    sig_bone=0,
    differential=False,
    A_matrix=None,
    B_incomplete=None,
):
    ###################################################################################################
    ## 1. Constants

    ## Model angular frequencies
    k_theta = np.linspace(-(M - 1) / 2, (M - 1) / 2, M)
    k_t = 2 * math.pi * np.linspace(-Fs / 2, Fs / 2, N)
    k_z = (
        k_t / v
    )  ## mean_conduction_velocity__mm_s = z/t => mean_conduction_velocity__mm_s = (1/k_z)/(1/k_t) => mean_conduction_velocity__mm_s = k_t/k_z
    (kt_mesh_kzkt, kz_mesh_kzkt) = np.meshgrid(k_t, k_z)

    ## Model radii -> (Farina, 2004), Figure 1 - b)
    th_muscle = r - th_fat - th_skin - r_bone
    a = r_bone
    b = r_bone + th_muscle
    c = r_bone + th_muscle + th_fat
    d = r_bone + th_muscle + th_fat + th_skin

    ###################################################################################################
    ## 2. I(k_t, k_z
    A = 96  ## mV/mm^3 -> (Farina, 2001), eq (16)
    z = np.linspace(
        -(N - 1) * (v / Fs) / 2, (N - 1) * (v / Fs) / 2, N, dtype=np.longdouble
    )  ## Ts(z) = Ts(t)*mean_conduction_velocity__mm_s = mean_conduction_velocity__mm_s/Fs(t)

    aux = np.zeros_like(z)
    positive_mask = z >= 0
    aux[positive_mask] = (
        A
        * np.exp(-z[positive_mask])
        * (3 * z[positive_mask] ** 2 - z[positive_mask] ** 3)
    )

    psi = -f_minus_t(aux)  ## derivative of Vm(-z) with respect to z
    PSI = np.fft.fftshift(np.fft.fft(psi)) / len(psi)
    PSI_conj = np.conj(PSI)
    PSI_conj = PSI_conj.reshape(-1, len(PSI_conj))
    ones = np.ones((len(PSI_conj[0, :]), 1))
    PSI_mesh_conj = np.dot(ones, PSI_conj)
    I_kzkt = 1j * np.multiply(
        kz_mesh_kzkt / v, np.multiply(PSI_mesh_conj, np.exp(-1j * kz_mesh_kzkt * zi))
    )
    k_eps = kz_mesh_kzkt + kt_mesh_kzkt / v
    k_beta = kz_mesh_kzkt - kt_mesh_kzkt / v
    aux1 = np.multiply(
        np.exp(-1j * k_eps * L1 / 2), np.sinc(k_eps * L1 / 2 / math.pi) * L1
    )
    aux2 = np.multiply(
        np.exp(1j * k_beta * L2 / 2), np.sinc(k_beta * L2 / 2 / math.pi) * L2
    )
    I_kzkt = np.multiply(I_kzkt, (aux1 - aux2))  ## (Farina, 2001), eq (22)
    t = np.linspace(0, (N - 1) / Fs, N)

    ###################################################################################################
    ## 2. H_vc(k_z, k_theta)
    am = a * math.sqrt(sig_muscle_z / sig_muscle_rho)  ## (Farina, 2004), eq (19)
    bm = b * math.sqrt(sig_muscle_z / sig_muscle_rho)
    Rm = R * math.sqrt(sig_muscle_z / sig_muscle_rho)

    i_start, i_end = int(len(k_z) / 2), len(k_z)
    j_start, j_end = int(len(k_theta) / 2), len(k_theta)

    # Create sub-arrays for positive frequencies
    k_z_pos = k_z[i_start:i_end]
    k_theta_pos = k_theta[j_start:j_end]

    K_THETA, K_Z = np.meshgrid(k_theta_pos, k_z_pos, indexing="ij")

    n_theta, n_z = K_THETA.shape

    A = np.zeros((n_theta, n_z, 7, 7))
    B = np.zeros((n_theta, n_z, 7, 1))

    Kn_Rm = Kn(K_THETA, Rm * K_Z)
    In_Rm = In(K_THETA, Rm * K_Z)

    if A_matrix is not None and B_incomplete is not None:
        A = A_matrix
        B = B_incomplete.copy()

    else:
        In_a = In(K_THETA, a * K_Z)
        In_bm = In(K_THETA, bm * K_Z)
        In_c = In(K_THETA, c * K_Z)
        In_d = In(K_THETA, d * K_Z)
        In_b = In(K_THETA, b * K_Z)
        In_am = In(K_THETA, am * K_Z)
        Kn_am = Kn(K_THETA, am * K_Z)
        In_tilde_am = In_tilde(K_THETA, am * K_Z)
        Kn_c = Kn(K_THETA, c * K_Z)
        Kn_d = Kn(K_THETA, d * K_Z)
        Kn_b = Kn(K_THETA, b * K_Z)
        In_tilde_a = In_tilde(K_THETA, a * K_Z)
        Kn_bm = Kn(K_THETA, bm * K_Z)
        In_tilde_bm = In_tilde(K_THETA, bm * K_Z)
        In_tilde_c = In_tilde(K_THETA, c * K_Z)
        In_tilde_d = In_tilde(K_THETA, d * K_Z)
        In_tilde_b = In_tilde(K_THETA, b * K_Z)
        Kn_tilde_am = Kn_tilde(K_THETA, am * K_Z)
        Kn_tilde_bm = Kn_tilde(K_THETA, bm * K_Z)
        Kn_tilde_c = Kn_tilde(K_THETA, c * K_Z)
        Kn_tilde_d = Kn_tilde(K_THETA, d * K_Z)
        Kn_tilde_b = Kn_tilde(K_THETA, b * K_Z)

        A[..., 0, 0] = 1
        A[..., 0, 1] = -In_am / In_bm
        A[..., 0, 2] = -Kn_am / Kn_bm
        # Second Row
        A[..., 1, 0] = sig_bone / In_a * In_tilde_a
        A[..., 1, 1] = -math.sqrt(sig_muscle_rho * sig_muscle_z) / In_bm * In_tilde_am
        A[..., 1, 2] = (
            -math.sqrt(sig_muscle_rho * sig_muscle_z) / Kn_bm * (-1) * Kn_tilde_am
        )
        # Third Row
        A[..., 2, 1] = 1
        A[..., 2, 2] = 1
        A[..., 2, 3] = -In_b / In_c
        A[..., 2, 4] = -Kn_b / Kn_c
        # Fourth Row
        A[..., 3, 1] = math.sqrt(sig_muscle_rho * sig_muscle_z) / In_bm * In_tilde_bm
        A[..., 3, 2] = (
            math.sqrt(sig_muscle_rho * sig_muscle_z) / Kn_bm * (-1) * Kn_tilde_bm
        )
        A[..., 3, 3] = -sig_fat / In_c * In_tilde_b
        A[..., 3, 4] = -sig_fat / Kn_c * (-1) * Kn_tilde_b
        # Fifth Row
        A[..., 4, 3] = 1
        A[..., 4, 4] = 1
        A[..., 4, 5] = -In_c / In_d
        A[..., 4, 6] = -Kn_c / Kn_d
        # Sixth Row
        A[..., 5, 3] = sig_fat / In_c * In_tilde_c
        A[..., 5, 4] = sig_fat / Kn_c * (-1) * Kn_tilde_c
        A[..., 5, 5] = -sig_skin / In_d * In_tilde_c
        A[..., 5, 6] = -sig_skin / Kn_d * (-1) * Kn_tilde_c
        # Seventh Row
        A[..., 6, 5] = sig_skin / In_d * In_tilde_d
        A[..., 6, 6] = sig_skin / Kn_d * (-1) * Kn_tilde_d

        A_matrix = A.copy()

        # Vector b
        B[..., 0, 0] = In_am / sig_muscle_rho
        B[..., 1, 0] = math.sqrt(sig_muscle_z / sig_muscle_rho) * In_tilde_am
        B[..., 2, 0] = Kn_bm / sig_muscle_rho
        B[..., 3, 0] = -math.sqrt(sig_muscle_z / sig_muscle_rho) * (-1) * Kn_tilde_bm

        B_incomplete = B.copy()

    B[..., 0, 0] *= Kn_Rm
    B[..., 1, 0] *= Kn_Rm
    B[..., 2, 0] *= -In_Rm
    B[..., 3, 0] *= In_Rm

    A = A.reshape(-1, 7, 7)
    B = B.reshape(-1, 7, 1)

    # Vector X
    if r_bone == 0:
        A = A[..., 2:, 2:]
        B = B[..., 2:, :]

        if HAS_CUPY:
            # Use GPU for matrix operations if available
            A_gpu = cp.asarray(A)
            B_gpu = cp.asarray(B)

            X = cp.linalg.solve(A_gpu, B_gpu)
            X = cp.asnumpy(X)

            del A_gpu, B_gpu  # Free GPU memory

        else:
            X = np.linalg.solve(A, B)

        X = X.reshape(n_theta, n_z, 5, 1)
        H_vc = X[..., 3, 0] + X[..., 4, 0]
    else:
        if HAS_CUPY:
            # Use GPU for matrix operations if available
            A_gpu = cp.asarray(A)
            B_gpu = cp.asarray(B)

            X = cp.linalg.solve(A_gpu, B_gpu)
            X = cp.asnumpy(X)

            del A_gpu, B_gpu
        else:
            X = np.linalg.solve(A, B)

        X = X.reshape(n_theta, n_z, 7, 1)
        H_vc = X[5, 0] + X[6, 0]

    temp = np.zeros((len(k_z), len(k_theta)))
    temp[i_start:i_end, j_start:j_end] = (
        H_vc.T
    )  # Transpose to match (k_z, k_theta) indexing
    H_vc = temp

    # OPTIMIZATION: More efficient symmetry exploitation
    H_vc_pos_section = H_vc[i_start:i_end, j_start:j_end]
    H_vc[i_start:i_end, :j_start] = np.fliplr(H_vc_pos_section)
    H_vc[:i_start, j_start:j_end] = np.flipud(H_vc_pos_section)
    H_vc[:i_start, :j_start] = np.flipud(np.fliplr(H_vc_pos_section))

    ###################################################################################################
    # 3. H_ele(k_z, k_theta)

    ## H_sf
    (ktheta_mesh_kzktheta, kz_mesh_kzktheta) = np.meshgrid(k_theta, k_z)
    # Rotation Angle -> (Farina, 2004), eq 36
    alpha = alpha * math.pi / 180
    kz_mesh_kzktheta_new = ktheta_mesh_kzktheta / r * math.sin(
        alpha
    ) + kz_mesh_kzktheta * math.cos(alpha)
    ktheta_mesh_kzktheta_new = ktheta_mesh_kzktheta * math.cos(
        alpha
    ) - kz_mesh_kzktheta * r * math.sin(alpha)
    # Spatial Filter - single differential
    # H_sf = np.exp(1j*kz_mesh_kzktheta_new)-np.exp(-1j*kz_mesh_kzktheta_new)
    # Spatial Filter - monopolar
    H_sf = 1

    ## H_size -> (Farina, 2004), eq (33)
    arg = np.sqrt(
        (rele * ktheta_mesh_kzktheta / r) ** 2 + (rele * kz_mesh_kzktheta) ** 2
    )
    H_size = 2 * np.divide(Jn(1, arg), arg)
    auxxx = np.ones(H_size.shape)
    H_size[np.isnan(H_size)] = auxxx[np.isnan(H_size)]

    ## H_ele
    H_ele = np.multiply(H_sf, H_size)

    ###################################################################################################
    ## 4. H_glo(k_z, k_theta)

    ## Detection system
    # pos_z
    pos_z = np.zeros((channels[0], channels[1]))
    if channels[0] % 2 == 1:
        index_center = int((channels[0] - 1) / 2)
        pos_z[index_center, :] = center[0]
        for i in range(1, index_center + 1):
            pos_z[index_center + i, :] = pos_z[index_center + i - 1, :] + d_ele
            pos_z[index_center - i, :] = pos_z[index_center - i + 1, :] - d_ele
    else:
        index_center1 = int(channels[0] / 2)
        index_center2 = index_center1 - 1
        pos_z[index_center1, :] = center[0] + d_ele / 2
        pos_z[index_center2, :] = center[0] - d_ele / 2
        for i in range(1, index_center2 + 1):
            pos_z[index_center1 + i, :] = pos_z[index_center1 + i - 1, :] + d_ele
            pos_z[index_center2 - i, :] = pos_z[index_center2 - i + 1, :] - d_ele
    # pos_theta
    pos_theta = np.zeros((channels[0], channels[1]))
    if channels[1] % 2 == 1:
        index_center = int((channels[1] - 1) / 2)
        pos_theta[:, index_center] = center[1] * math.pi / 180
        for i in range(1, index_center + 1):
            pos_theta[:, index_center + i] = (
                pos_theta[:, index_center + i - 1] + d_ele / r
            )
            pos_theta[:, index_center - i] = (
                pos_theta[:, index_center - i + 1] - d_ele / r
            )
    else:
        index_center1 = int(channels[1] / 2)
        index_center2 = index_center1 - 1
        pos_theta[:, index_center1] = center[1] * math.pi / 180 + d_ele / 2 / r
        pos_theta[:, index_center2] = center[1] * math.pi / 180 - d_ele / 2 / r
        for i in range(1, index_center2 + 1):
            pos_theta[:, index_center1 + i] = (
                pos_theta[:, index_center1 + i - 1] + d_ele / r
            )
            pos_theta[:, index_center2 - i] = (
                pos_theta[:, index_center2 - i + 1] - d_ele / r
            )

    ## Rotated detection system (Farina, 2004), eq (36)
    displacement = center[0] * np.ones(pos_z.shape)
    pos_z_new = (
        -r * math.sin(alpha) * pos_theta
        + math.cos(alpha) * (pos_z - displacement)
        + displacement
    )
    pos_theta_new = (
        math.cos(alpha) * pos_theta + math.sin(alpha) * (pos_z - displacement) / r
    )
    pos_z = pos_z_new
    pos_theta = pos_theta_new
    H_glo = np.multiply(H_vc, H_ele)
    B_kz = np.zeros((channels[0], channels[1], len(k_z)))
    for channel_z in range(0, channels[0]):
        for channel_theta in range(0, channels[1]):
            ## B_kz -> (Farina, 2004), eq (5)
            arg = np.multiply(
                H_glo,
                np.exp(1j * pos_theta[channel_z, channel_theta] * ktheta_mesh_kzktheta)
                * (k_theta[1] - k_theta[0]),
            )
            B_kz[channel_z, channel_theta, :] = sum(np.transpose(arg)) / 2 / math.pi

    ###################################################################################################
    ## 5. phi(t) for each channel

    ## (Farina, 2004), eq (2)
    phi = np.zeros((channels[0], channels[1], len(t)))
    for channel_z in range(0, channels[0]):
        for channel_theta in range(0, channels[1]):
            auxiliar = np.dot(
                np.ones((len(I_kzkt[1, :]), 1)),
                B_kz[channel_z, channel_theta, :].reshape(1, -1),
            )
            auxiliar = np.transpose(auxiliar)
            arg = np.multiply(I_kzkt, auxiliar)
            arg2 = np.multiply(
                arg,
                np.exp(1j * pos_z[channel_z, channel_theta] * kz_mesh_kzkt)
                * (k_z[1] - k_z[0]),
            )
            PHI = sum(arg2)
            phi[channel_z, channel_theta, :] = np.real(
                (np.fft.ifft(np.fft.fftshift(PHI / 2 / math.pi * len(psi))))
            )
    return phi, A_matrix, B_incomplete


def f_minus_t(y):
    y_new = np.zeros(len(y))
    for i in range(0, len(y)):
        y_new[i] = y[-i]
    return y_new


#######################################################################################################
###################################### References #####################################################
#######################################################################################################

# FARINA, D.; MERLETTI, R. A novel approach for precise simulation of the EMG signal detected by surface electrodes.
# IEEE Transactions on Biomedical Engineering, mean_conduction_velocity__mm_s. 48, n. 6, p. 637–646, 2001. DOI: 10.1109/10.923782.

# FARINA, D.; MESIN, L.; MARTINA, S.; MERLETTI, R. A Surface EMG Generation Model With Multilayer Cylindrical
# Description of the Volume Conductor. IEEE Transactions on Biomedical Engineering, mean_conduction_velocity__mm_s. 51, n. 3, p. 415–426,
# 2004. DOI: 10.1109/TBME.2003.820998.
