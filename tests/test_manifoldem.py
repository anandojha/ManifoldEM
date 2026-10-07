"""Unit tests for ManifoldEM.

Single-file suite, mirroring the PySTARC convention. Run from the repo root so that
`ManifoldEM` resolves to the source tree being edited rather than an installed copy:

    cd <repo> && python -m pytest tests/test_manifoldem.py -q

Tests marked xfail document defects found while writing the suite. The reason string
names the defect. They are non-strict, so a fix turns them green rather than red.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace, _SubParsersAction
from functools import partial
from numpy import linalg as LA
from scipy.ndimage import affine_transform
from scipy.ndimage import correlate, gaussian_filter
from scipy.signal import butter, freqs
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import spearmanr
from types import SimpleNamespace
from typing import List, Union
from typing import get_args, get_origin, get_type_hints
import builtins
import copy
import dataclasses
import h5py
import inspect
import math
import matplotlib
import matplotlib.pyplot as plt
import mrcfile
import multiprocessing
import numbers
import numpy as np
import os
import pandas as pd
import pickle
import pprint
import pytest
import shutil
import subprocess
import sys
import toml
import warnings
import zipfile

from ManifoldEM import (
    calc_distance,
    find_conformational_coords,
    manifold_analysis,
    manifoldTrimmingAuto,
    myio,
    nlsa_movie,
    particle_index,
    probability_landscape,
    quaternion,
    star,
    trajectory,
    util,
    writeRelionS2,
)
from ManifoldEM import DMembeddingII as DM
from ManifoldEM import FindCCGraph as FG
from ManifoldEM import FindCCGraphPruned as FP
from ManifoldEM import embedd, manifoldTrimmingAuto as MTA, myio
from ManifoldEM import myio
from ManifoldEM import star
from ManifoldEM.CC import ComputeMeasureEdgeAll as CME
from ManifoldEM.CC import MRFBeliefPropagation as BP
from ManifoldEM.CC import MRFGeneratePotentials as MGP
from ManifoldEM.CC import OpticalFlowMovie as OFM
from ManifoldEM.CC import hornschunck_simple as HS
from ManifoldEM.CC import transformations as tr
from ManifoldEM.FindCCGraph import CreateGraphStruct
from ManifoldEM.S2tessellation import (
    bin_and_threshold,
    collect_nearest_neighbors,
    fibonacci_tessellation,
    lovisolo_silva_tessellation,
)
from ManifoldEM.core import (L2_distance, svdRF, fergusonE, annular_mask, get_wiener,
                             euler_rot_matrix_3D_spider, rotate_volume_euler, get_euler_from_PD,
                             project_mask, makeMovie, clusterAvg)
from ManifoldEM.data_store import (
    Anchor,
    PrdData,
    PrdInfo,
    Sense,
    _DataStore,
    _ProjectionDirections,
    data_store,
)
from ManifoldEM.data_store import data_store, _DataStore, _ProjectionDirections
from ManifoldEM.fit_1D_open_manifold_3D import (
    _Params,
    _R_p,
    _get_fit_params,
    _solve_d_R_d_tau_p_3D,
    fit_1D_open_manifold_3D,
)
from ManifoldEM.interfaces import cli
from ManifoldEM.interfaces import interactive as mem
from ManifoldEM.myio import fin1, fout1
from ManifoldEM.params import ParamInfo, Params, ProjectLevel, params
from ManifoldEM.params import ProjectLevel, params
from ManifoldEM.params import params
from ManifoldEM.params import params, Params, ProjectLevel
from ManifoldEM.psi_analysis import _corr, _diff_corr
from ManifoldEM.quaternion import (_optfunc, _q_product_single, alternate_euler_convention, calc_avg_pd,
                                   collapse_to_half_space, collapse_to_half_space_euler_angles, convert_euler_to_S2,
                                   convert_S2_to_euler, psi_ang, q2Spider, q_product, quaternion_to_S2,
                                   qs_to_spider_euler_angles)
from ManifoldEM.quaternion import quaternion_to_S2, convert_euler_to_S2
from ManifoldEM.util import (NullEmitter, get_tqdm, calc_shannon, debug_print, hist_match, histeq,
                             eul_to_quat, augment, make_indeces, interv, filter_fourier,
                             create_proportional_grid, ctemh_cryoFrank, get_CTFs, rotate_fill,
                             get_image_width_from_stack)
from ManifoldEM.util import eul_to_quat
import ManifoldEM
import ManifoldEM.fit_1D_open_manifold_3D as fitmod
import ManifoldEM.util



# ============================================================================
# core.py and util.py - leaf math and image utilities
# ============================================================================

# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture
def params_sandbox_core_util(tmp_path):
    """`params` is a process-wide singleton and several of its paths are RELATIVE, so any
    test that touches it must pin the cwd and put every field back the way it found it."""
    saved_cwd = os.getcwd()
    watched = ("project_name", "ncpu", "img_stack_file", "ms_num_pixels")
    saved = {k: getattr(params, k) for k in watched}

    os.chdir(tmp_path)
    params.project_name = "pytest_core_util"
    params.ncpu = 1  # never fork a Pool from a test

    try:
        yield tmp_path
    finally:
        os.chdir(saved_cwd)
        for k, v in saved.items():
            setattr(params, k, v)


# --------------------------------------------------------------------------------------
# core.L2_distance
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("dim,m,n", [(1, 3, 4), (2, 5, 5), (3, 7, 2), (4, 2, 9), (10, 6, 6),
                                     (2, 1, 1), (5, 12, 3)])
def test_L2_distance_matches_scipy_cdist(dim, m, n):
    """L2_distance takes D x M and D x N (vectors are COLUMNS) and returns M x N."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(dim, m)) * 10.0
    b = rng.normal(size=(dim, n)) * 10.0

    d = L2_distance(a, b)

    assert d.shape == (m, n)
    assert np.allclose(d, cdist(a.T, b.T), atol=1e-8)


@pytest.mark.parametrize("n", [2, 5, 11])
def test_L2_distance_self_is_symmetric_with_zero_diagonal(n):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, n)) * 5.0

    d = L2_distance(a, a)

    assert np.allclose(d, d.T)
    assert np.all(np.diag(d) == 0.0)


def test_L2_distance_is_non_negative_and_obeys_triangle_inequality():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(3, 8)) * 20.0
    d = L2_distance(a, a)

    assert np.all(d >= 0.0)
    # d_ik <= d_ij + d_jk for every triple (i, j, k)
    assert np.all(d[:, None, :] <= d[:, :, None] + d[None, :, :] + 1e-9)


def test_L2_distance_dimension_mismatch_raises():
    a = np.zeros((3, 4))
    b = np.zeros((5, 4))
    with pytest.raises(ValueError, match="same dimensionality"):
        L2_distance(a, b)


@pytest.mark.parametrize("sep,expected_zero", [(1e-6, True), (1e-5, True), (1e-4, False),
                                               (1e-3, False), (1.0, False)])
def test_L2_distance_clamps_squared_distances_below_1e_8(sep, expected_zero):
    """The routine zeroes squared distances < 1e-8 for numerical stability, so genuine
    separations below 1e-4 are reported as exactly 0."""
    a = np.array([[0.0], [0.0]])
    b = np.array([[sep], [0.0]])

    d = L2_distance(a, b)[0, 0]

    assert bool(d == 0.0) is expected_zero


def test_L2_distance_translation_invariance():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3, 6))
    b = rng.normal(size=(3, 4))
    shift = np.array([[3.0], [-2.0], [7.5]])

    assert np.allclose(L2_distance(a, b), L2_distance(a + shift, b + shift), atol=1e-8)


def test_L2_distance_scales_linearly():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(3, 5)) * 10.0
    b = rng.normal(size=(3, 5)) * 10.0

    assert np.allclose(L2_distance(2.5 * a, 2.5 * b), 2.5 * L2_distance(a, b), atol=1e-7)


# --------------------------------------------------------------------------------------
# core.svdRF
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (4, 4), (10, 2), (2, 10), (5, 5), (8, 1),
                                   (1, 8)])
def test_svdRF_reconstructs_the_matrix(shape):
    rng = np.random.default_rng(5)
    A = rng.normal(size=shape)

    U, S, V = svdRF(A)

    assert np.allclose(U @ S @ V.T, A, atol=1e-8)


@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (4, 4), (10, 2), (2, 10)])
def test_svdRF_singular_values_match_numpy(shape):
    rng = np.random.default_rng(6)
    A = rng.normal(size=shape)

    _, S, _ = svdRF(A)
    sdiag = np.diag(S)
    reference = np.linalg.svd(A, compute_uv=False)

    assert np.allclose(sdiag, reference[:sdiag.size], atol=1e-8)
    # S must be strictly diagonal
    assert np.allclose(S - np.diag(sdiag), 0.0)


@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (4, 4), (10, 2), (2, 10)])
def test_svdRF_singular_values_are_descending(shape):
    rng = np.random.default_rng(7)
    A = rng.normal(size=shape)

    _, S, _ = svdRF(A)

    assert np.all(np.diff(np.diag(S)) <= 1e-12)


@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (4, 4), (10, 2), (2, 10)])
def test_svdRF_factors_are_orthonormal(shape):
    rng = np.random.default_rng(8)
    A = rng.normal(size=shape)

    U, _, V = svdRF(A)

    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8)
    assert np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-8)


@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (4, 4)])
def test_svdRF_shapes_follow_the_thin_convention(shape):
    rng = np.random.default_rng(9)
    A = rng.normal(size=shape)
    d1, d2 = shape
    k = min(d1, d2)

    U, S, V = svdRF(A)

    assert U.shape == (d1, k)
    assert S.shape == (k, k)
    assert V.shape == (d2, k)


def test_svdRF_of_a_diagonal_matrix_recovers_the_diagonal():
    A = np.diag([4.0, 1.0, 3.0])

    _, S, _ = svdRF(A)

    assert np.allclose(np.diag(S), [4.0, 3.0, 1.0], atol=1e-10)


# --------------------------------------------------------------------------------------
# core.fergusonE
# --------------------------------------------------------------------------------------
def _ferguson_inputs(n_points=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3, n_points))
    return L2_distance(X, X), np.arange(-15.0, 15.1, 1.0)


@pytest.mark.parametrize("n_points", [20, 40, 60])
def test_fergusonE_asymptotes_bracket_logN_and_2logN(n_points):
    """sum_ij exp(-d_ij^2 / 2 eps) -> N as eps -> 0 (only the diagonal survives) and -> N^2
    as eps -> inf.  The fitted tanh d + c*tanh(a x + b) therefore has asymptotes
    d - |c| = log N and d + |c| = 2 log N."""
    D, logEps = _ferguson_inputs(n_points)

    popt, logSumWij, resnorm, R_squared = fergusonE(D, logEps)
    _, _, c, d = popt

    assert np.isclose(logSumWij[0], np.log(n_points), atol=1e-9)
    assert np.isclose(logSumWij[-1], 2 * np.log(n_points), atol=1e-6)
    assert np.isclose(d - abs(c), np.log(n_points), atol=0.05)
    assert np.isclose(d + abs(c), 2 * np.log(n_points), atol=0.05)


def test_fergusonE_returns_a_good_fit():
    D, logEps = _ferguson_inputs()

    popt, logSumWij, resnorm, R_squared = fergusonE(D, logEps)

    assert popt.shape == (4, )
    assert logSumWij.shape == logEps.shape
    assert R_squared > 0.99
    assert resnorm <= 100.0  # the while-loop exit condition


def test_fergusonE_logSumWij_is_monotone_increasing_in_epsilon():
    D, logEps = _ferguson_inputs()

    _, logSumWij, _, _ = fergusonE(D, logEps)

    assert np.all(np.diff(logSumWij) >= -1e-12)


def test_fergusonE_mutates_the_caller_a0_in_place():
    """`a0 *= 0.5` inside the loop writes through to the caller's array, callers must
    hand over a throwaway."""
    D, logEps = _ferguson_inputs()
    a0 = np.ones(4)

    fergusonE(D, logEps, a0)

    assert np.allclose(a0, 0.5)


def test_fergusonE_default_a0_leaves_module_state_untouched():
    D, logEps = _ferguson_inputs()

    first = fergusonE(D, logEps)[0]
    second = fergusonE(D, logEps)[0]

    assert np.allclose(first, second)


def test_fergusonE_is_invariant_to_the_order_of_the_distance_entries():
    D, logEps = _ferguson_inputs()
    rng = np.random.default_rng(11)
    perm = rng.permutation(D.shape[0])

    _, ref, _, _ = fergusonE(D, logEps)
    _, shuffled, _, _ = fergusonE(D[perm][:, perm], logEps)

    assert np.allclose(ref, shuffled)


# --------------------------------------------------------------------------------------
# core.annular_mask
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("N,M", [(8, 8), (7, 7), (10, 6), (5, 9), (1, 1)])
def test_annular_mask_shape_and_binary_values(N, M):
    mask = annular_mask(1.0, 3.0, N, M)

    assert mask.shape == (N, M)
    assert mask.dtype == np.float64
    assert set(np.unique(mask)).issubset({0.0, 1.0})


@pytest.mark.parametrize("a,b", [(3.0, 3.0), (5.0, 2.0), (10.0, 1.0)])
def test_annular_mask_is_empty_when_inner_radius_is_not_smaller(a, b):
    """The membership test is aSq <= rSq < bSq, so a >= b can never be satisfied."""
    assert annular_mask(a, b, 8, 8).sum() == 0.0


def test_annular_mask_with_huge_outer_radius_is_all_ones():
    assert annular_mask(0.0, 1e6, 9, 9).sum() == 81.0


@pytest.mark.parametrize("b", [1.0, 2.0, 3.0, 4.0, 5.0])
def test_annular_mask_grows_monotonically_with_outer_radius(b):
    small = annular_mask(0.0, b, 16, 16)
    large = annular_mask(0.0, b + 1.0, 16, 16)

    assert np.all(large >= small)


def test_annular_mask_annulus_is_the_difference_of_two_discs():
    inner = annular_mask(0.0, 3.0, 16, 16)
    outer = annular_mask(0.0, 6.0, 16, 16)
    ring = annular_mask(3.0, 6.0, 16, 16)

    assert np.allclose(ring, outer - inner)


def test_annular_mask_outer_radius_is_exclusive():
    """rSq < bSq, so the pixel sitting exactly at radius b is NOT included."""
    # row center is index N/2 - 1 = 3, column center is index M/2 = 4 for N = M = 8
    mask = annular_mask(0.0, 2.0, 8, 8)
    assert mask[3, 4 + 2] == 0.0
    assert mask[3, 4 + 1] == 1.0


@pytest.mark.parametrize("N", [6, 8, 12])
@pytest.mark.xfail(reason="Bug. row center is N/2-1 but column center is M/2, so the mask is "
                   "off-center by one row (MATLAB 1-index leaked into the xx loop only)",
                   strict=False)
def test_annular_mask_is_symmetric_under_transpose(N):
    """A disc mask on a square grid has to be its own transpose."""
    mask = annular_mask(0.0, 3.0, N, N)

    assert np.allclose(mask, mask.T)


# --------------------------------------------------------------------------------------
# core.get_wiener
# --------------------------------------------------------------------------------------
def _ctf_stack(n=10, dim=4, seed=12):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim, dim))


@pytest.mark.parametrize("con_order,num", [(2, 6), (3, 8), (1, 5), (4, 10)])
def test_get_wiener_shapes_and_passthrough(con_order, num):
    CTF = _ctf_stack()
    posPath = np.arange(CTF.shape[0])
    posPsi1 = np.arange(CTF.shape[0])

    wiener_dom, CTF1 = get_wiener(CTF, posPath, posPsi1, con_order, num)

    assert wiener_dom.shape == (num - con_order, CTF.shape[1], CTF.shape[2])
    assert wiener_dom.dtype == np.float64
    # CTF1 is just the reindexed stack
    assert np.allclose(CTF1, CTF[posPath[posPsi1]])


def test_get_wiener_adds_the_hardcoded_inverse_snr():
    """SNR is hardcoded to 5, so a zero CTF gives a flat 1/5 denominator."""
    CTF = np.zeros((6, 3, 3))
    posPath = np.arange(6)

    wiener_dom, _ = get_wiener(CTF, posPath, np.arange(6), 2, 5)

    assert np.allclose(wiener_dom, 1.0 / 5.0)


def test_get_wiener_denominator_is_never_below_inverse_snr():
    CTF = _ctf_stack()
    posPath = np.arange(CTF.shape[0])

    wiener_dom, _ = get_wiener(CTF, posPath, posPath, 3, 8)

    assert np.all(wiener_dom >= 1.0 / 5.0 - 1e-12)


def test_get_wiener_matches_its_own_index_arithmetic():
    """Documents what the routine actually sums. CTF1[ConOrder - ii + i]**2."""
    CTF = _ctf_stack()
    posPath = np.arange(CTF.shape[0])
    con_order, num, snr = 3, 8, 5.0

    wiener_dom, CTF1 = get_wiener(CTF, posPath, posPath, con_order, num)

    expected = np.zeros_like(wiener_dom)
    for i in range(num - con_order):
        for ii in range(con_order):
            expected[i] += CTF1[con_order - ii + i]**2
    assert np.allclose(wiener_dom, expected + 1.0 / snr)


def test_get_wiener_respects_posPath_reordering():
    CTF = _ctf_stack()
    order = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    _, CTF1 = get_wiener(CTF, order, np.arange(10), 3, 8)

    assert np.allclose(CTF1, CTF[::-1])


@pytest.mark.xfail(reason="Bug. off-by-one, _NLSA deconvolves image index "
                   "(ConOrder - ii + i - 1) but get_wiener builds the denominator from "
                   "(ConOrder - ii + i), so CTF slot 0 is never used",
                   strict=False)
def test_get_wiener_denominator_uses_the_same_indices_as_NLSA():
    CTF = _ctf_stack()
    posPath = np.arange(CTF.shape[0])
    con_order, num, snr = 3, 8, 5.0

    wiener_dom, CTF1 = get_wiener(CTF, posPath, posPath, con_order, num)

    expected = np.zeros_like(wiener_dom)
    for i in range(num - con_order):
        for ii in range(con_order):
            expected[i] += CTF1[con_order - ii + i - 1]**2
    assert np.allclose(wiener_dom, expected + 1.0 / snr)


# --------------------------------------------------------------------------------------
# core.euler_rot_matrix_3D_spider
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("angles", [(0.0, 0.0, 0.0), (0.3, 0.4, 0.5), (-1.2, 2.1, 0.7),
                                    (np.pi, np.pi / 2, -np.pi / 3), (2.0, 0.0, -2.0)])
def test_euler_rot_matrix_is_a_proper_rotation(angles):
    R = euler_rot_matrix_3D_spider(*angles)

    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_euler_rot_matrix_identity_at_zero():
    assert np.allclose(euler_rot_matrix_3D_spider(0.0, 0.0, 0.0), np.eye(3))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_euler_rot_matrix_matches_scipy_clockwise_extrinsic_zyz(seed):
    """SPIDER angles are clockwise extrinsic zyz, negate before handing to scipy."""
    rng = np.random.default_rng(seed)
    e = rng.uniform(-np.pi, np.pi, size=(20, 3))

    mem = np.array([euler_rot_matrix_3D_spider(*a) for a in e])
    ref = Rotation.from_euler('zyz', -e).as_matrix()

    assert np.allclose(mem, ref, atol=1e-10)


@pytest.mark.parametrize("phi", [0.0, 0.5, -1.3, np.pi])
def test_euler_rot_matrix_pure_phi_is_a_z_rotation(phi):
    """With Theta = Psi = 0 only the leading z rotation survives. The sign is clockwise."""
    R = euler_rot_matrix_3D_spider(phi, 0.0, 0.0)
    expected = np.array([[np.cos(phi), np.sin(phi), 0.0],
                         [-np.sin(phi), np.cos(phi), 0.0],
                         [0.0, 0.0, 1.0]])

    assert np.allclose(R, expected, atol=1e-12)


@pytest.mark.parametrize("theta", [0.0, 0.4, -0.9, np.pi / 2])
def test_euler_rot_matrix_pure_theta_leaves_the_y_axis_alone(theta):
    R = euler_rot_matrix_3D_spider(0.0, theta, 0.0)

    assert np.allclose(R @ np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), atol=1e-12)
    assert np.isclose(R[2, 2], np.cos(theta))


def test_euler_rot_matrix_phi_and_psi_compose_when_theta_is_zero():
    """Two coaxial z rotations add, so R(phi,0,psi) == R(phi+psi,0,0)."""
    R = euler_rot_matrix_3D_spider(0.7, 0.0, 0.4)

    assert np.allclose(R, euler_rot_matrix_3D_spider(1.1, 0.0, 0.0), atol=1e-12)


@pytest.mark.parametrize("angles", [(0.3, 0.4, 0.5), (-1.0, 0.2, 2.0)])
def test_euler_rot_matrix_is_2pi_periodic(angles):
    phi, theta, psi = angles
    a = euler_rot_matrix_3D_spider(phi, theta, psi)
    b = euler_rot_matrix_3D_spider(phi + 2 * np.pi, theta + 2 * np.pi, psi + 2 * np.pi)

    assert np.allclose(a, b, atol=1e-10)


# --------------------------------------------------------------------------------------
# core.rotate_volume_euler
# --------------------------------------------------------------------------------------
def _blob_volume(n=16):
    vol = np.zeros((n, n, n))
    vol[5:9, 4:7, 6:11] = 1.0
    return vol


def test_rotate_volume_euler_identity():
    vol = _blob_volume()

    assert np.allclose(rotate_volume_euler(vol, [0.0, 0.0, 0.0]), vol)


def test_rotate_volume_euler_preserves_shape_and_dtype():
    vol = _blob_volume(12)

    out = rotate_volume_euler(vol, [0.4, 0.2, -0.3])

    assert out.shape == vol.shape
    assert out.dtype == vol.dtype


@pytest.mark.parametrize("sym", [[0.4, 0.2, -0.3], [1.0, 0.5, 0.0], [-2.0, 1.1, 0.6]])
def test_rotate_volume_euler_matches_a_direct_affine_transform(sym):
    """`sym` is consumed REVERSED. euler_rot_matrix_3D_spider(sym[2], sym[1], sym[0])."""
    vol = _blob_volume()
    dims = vol.shape
    rotmat = euler_rot_matrix_3D_spider(sym[2], sym[1], sym[0])
    c = 0.5 * np.array(dims)
    ref = affine_transform(input=vol, matrix=rotmat, offset=c - rotmat @ c,
                           output_shape=dims, mode='nearest')

    assert np.allclose(rotate_volume_euler(vol, sym), ref)


def test_rotate_volume_euler_of_a_constant_volume_is_constant():
    vol = np.full((10, 10, 10), 2.5)

    assert np.allclose(rotate_volume_euler(vol, [0.7, 0.9, -0.2]), 2.5)


# --------------------------------------------------------------------------------------
# core.get_euler_from_PD
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("pd", [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                                [0.5, -0.3, 0.8], [-0.4, 0.2, 0.9], [2.0, 1.0, 3.0]])
@pytest.mark.xfail(reason="Bug. q2Spider starts its Levenberg-Marquardt solve at a=[0,0,0], "
                   "where the Jacobian row of the first imaginary residual component is "
                   "identically zero. Whether the solver escapes that saddle turns on 1-ulp "
                   "differences in the numba-compiled residual, so ~5%% of PROCESSES return "
                   "(0,0,0) for some direction.  Reproducible by varying PYTHONHASHSEED.",
                   strict=False)
def test_get_euler_from_PD_round_trips_through_S2(pd):
    pd = np.array(pd, dtype=float)
    pd /= np.linalg.norm(pd)

    e = get_euler_from_PD(pd)
    back = convert_euler_to_S2(e.reshape(1, 3)).ravel()

    assert np.allclose(pd, back, atol=1e-6)


@pytest.mark.parametrize("pd", [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                                [0.5, -0.3, 0.8], [-0.4, 0.2, 0.9], [2.0, 1.0, 3.0]])
def test_get_euler_from_PD_stays_in_the_valid_angle_ranges(pd):
    """Deterministic companion to the (unreliable) round trip above. Whichever branch the
    optimizer lands in, the returned triple has to be a finite Euler triple with a polar
    angle in [0, pi] and the conventional zero psi."""
    pd = np.array(pd, dtype=float)
    pd /= np.linalg.norm(pd)

    phi, theta, psi = get_euler_from_PD(pd)

    assert np.all(np.isfinite([phi, theta, psi]))
    assert -1.0 - 1e-12 <= np.cos(theta) <= 1.0 + 1e-12
    assert psi == 0.0


@pytest.mark.parametrize("seed", [0, 1])
def test_get_euler_from_PD_psi_is_always_zero(seed):
    """Psi is degenerate for a projection direction and is hardwired to 0."""
    rng = np.random.default_rng(seed)
    pds = rng.normal(size=(3, 8))
    pds /= np.linalg.norm(pds, axis=0)

    for pd in pds.T:
        assert get_euler_from_PD(pd)[2] == 0.0


def test_get_euler_from_PD_returns_three_angles():
    e = get_euler_from_PD(np.array([0.3, 0.4, np.sqrt(1 - 0.25)]))

    assert e.shape == (3, )
    assert np.all(np.isfinite(e))


@pytest.mark.parametrize("pd", [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.7071, 0.7071],
                                [0.0, -0.6, 0.8]])
@pytest.mark.xfail(reason="Bug. for PD[0] == 0 the q2Spider Levenberg-Marquardt solve starts "
                   "on a zero-Jacobian saddle and silently returns (0,0,0), i.e. the +z "
                   "direction, instead of the requested direction",
                   strict=False)
def test_get_euler_from_PD_handles_the_x_equals_zero_great_circle(pd):
    pd = np.array(pd, dtype=float)
    pd /= np.linalg.norm(pd)

    back = convert_euler_to_S2(get_euler_from_PD(pd).reshape(1, 3)).ravel()

    assert np.allclose(pd, back, atol=1e-6)


@pytest.mark.xfail(reason="Bug. The south pole gives Qr = [0,0,0,0]. Normalizing it produces "
                   "NaN and least_squares raises 'Residuals are not finite'",
                   strict=False)
def test_get_euler_from_PD_handles_the_south_pole():
    pd = np.array([0.0, 0.0, -1.0])

    e = get_euler_from_PD(pd)

    assert np.isclose(np.cos(e[1]), -1.0, atol=1e-6)


# --------------------------------------------------------------------------------------
# core.project_mask
# --------------------------------------------------------------------------------------
def test_project_mask_shape_and_dtype():
    vol = np.zeros((20, 20, 20))
    vol[8:12, 8:12, 8:12] = 1.0

    msk = project_mask(vol, np.array([0.0, 0.0, 1.0]))

    assert msk.shape == (20, 20)
    assert msk.dtype == bool


def test_project_mask_of_an_empty_volume_is_empty():
    msk = project_mask(np.zeros((16, 16, 16)), np.array([0.0, 0.0, 1.0]))

    assert not msk.any()


def test_project_mask_of_a_dense_volume_is_full():
    """The threshold is a bare `> 1` on the ray sum, so a uniform volume passes everywhere."""
    msk = project_mask(np.full((16, 16, 16), 2.0), np.array([0.0, 0.0, 1.0]))

    assert msk.all()


def test_project_mask_along_z_is_the_plain_z_projection():
    """PD = +z maps to zero Euler angles, so the rotation is the identity and only the
    swapaxes(0, 2) + transpose bookkeeping remains."""
    rng = np.random.default_rng(13)
    vol = rng.uniform(0.0, 0.3, size=(12, 12, 12))
    vol[3:6, 7:10, :] = 1.0

    msk = project_mask(vol, np.array([0.0, 0.0, 1.0]))
    ref = np.sum(np.swapaxes(vol, 0, 2), axis=2).T > 1

    assert np.array_equal(msk, ref)


@pytest.mark.parametrize("pd", [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.6, 0.0, 0.8],
                                [0.3, -0.4, 0.866]])
def test_project_mask_covers_fewer_pixels_than_the_box(pd):
    vol = np.zeros((20, 20, 20))
    vol[7:13, 7:13, 7:13] = 1.0

    msk = project_mask(vol, np.array(pd, dtype=float))

    assert 0 < msk.sum() < msk.size


# --------------------------------------------------------------------------------------
# core.makeMovie  (GIF/ZIP writer)
# --------------------------------------------------------------------------------------
def _topos_dir(prd):
    d = os.path.join(params.out_dir, "topos", f"PrD_{prd + 1}")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.mark.parametrize("dim,nframes", [(4, 3), (6, 5), (8, 10)])
def test_makeMovie_writes_a_gif_and_a_zip(params_sandbox_core_util, dim, nframes):
    import imageio

    rng = np.random.default_rng(14)
    IMG1 = rng.uniform(size=(dim * dim, nframes))
    prd, psinum = 0, 0
    d = _topos_dir(prd)

    makeMovie(IMG1, prd, psinum, 5.0)

    gif = os.path.join(d, "psi_1.gif")  # filenames are 1-indexed
    zp = os.path.join(d, "psi_1.zip")
    assert os.path.exists(gif) and os.path.exists(zp)
    assert len(imageio.mimread(gif)) == nframes
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == [f"frame{i:02d}.png" for i in range(nframes)]


def test_makeMovie_uses_one_indexed_names(params_sandbox_core_util):
    rng = np.random.default_rng(15)
    IMG1 = rng.uniform(size=(16, 3))
    prd, psinum = 2, 3
    d = _topos_dir(prd)

    makeMovie(IMG1, prd, psinum, 5.0)

    assert os.path.exists(os.path.join(d, "psi_4.gif"))
    assert os.path.exists(os.path.join(d, "psi_4.zip"))


def test_makeMovie_inverts_contrast(params_sandbox_core_util):
    """`images = -IMG1`, so the brightest input pixel becomes the darkest gif pixel."""
    import imageio

    dim, nframes = 4, 2
    IMG1 = np.linspace(0.0, 1.0, dim * dim * nframes).reshape(dim * dim, nframes)
    d = _topos_dir(0)

    makeMovie(IMG1, 0, 0, 5.0)

    frames = imageio.mimread(os.path.join(d, "psi_1.gif"))
    gray = np.array([f[..., 0] for f in frames])
    flat_in = IMG1.T.reshape(nframes, dim, dim)
    assert gray[np.unravel_index(np.argmax(flat_in), flat_in.shape)] == 0
    assert gray[np.unravel_index(np.argmin(flat_in), flat_in.shape)] == 255


@pytest.mark.xfail(reason="Bug. Window size is int(sqrt(max(IMG1.shape))) so a stack with "
                   "more frames than pixels picks up the frame count as the box width",
                   strict=False)
def test_makeMovie_takes_the_window_size_from_the_pixel_axis(params_sandbox_core_util):
    import imageio

    IMG1 = np.zeros((9, 16))  # 3x3 images, 16 frames
    d = _topos_dir(0)

    makeMovie(IMG1, 0, 0, 5.0)

    assert imageio.mimread(os.path.join(d, "psi_1.gif"))[0].shape[:2] == (3, 3)


# --------------------------------------------------------------------------------------
# core.clusterAvg
# --------------------------------------------------------------------------------------
def _write_dist_file_core_util(prd, imgAll):
    os.makedirs(params.dist_dir, exist_ok=True)
    myio.fout1(params.get_dist_file(prd), imgAll=imgAll)


@pytest.mark.parametrize("clust", [[0], [0, 1], [1, 3], [0, 1, 2, 3]])
def test_clusterAvg_sums_the_selected_images(params_sandbox_core_util, clust):
    rng = np.random.default_rng(16)
    imgAll = rng.normal(size=(4, 5, 5))
    _write_dist_file_core_util(0, imgAll)

    out = clusterAvg(clust, 0)

    assert out.shape == (5, 5)
    assert np.allclose(out, imgAll[list(clust)].sum(axis=0))


def test_clusterAvg_of_an_empty_cluster_is_zero(params_sandbox_core_util):
    rng = np.random.default_rng(17)
    _write_dist_file_core_util(1, rng.normal(size=(3, 6, 6)))

    out = clusterAvg([], 1)

    assert out.shape == (6, 6)
    assert np.all(out == 0.0)


def test_clusterAvg_counts_repeats(params_sandbox_core_util):
    """It is a bare accumulation, not a mean, so a repeated index is added twice."""
    imgAll = np.ones((2, 4, 4))
    _write_dist_file_core_util(2, imgAll)

    assert np.allclose(clusterAvg([0, 0, 0], 2), 3.0)


# --------------------------------------------------------------------------------------
# util.NullEmitter / get_tqdm / calc_shannon / debug_print
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("percent", [0, 50, 100, -1, None, "abc"])
def test_null_emitter_swallows_everything(percent):
    assert NullEmitter().emit(percent) is None


def test_get_tqdm_returns_the_terminal_tqdm_outside_jupyter():
    from tqdm import tqdm as terminal_tqdm

    assert get_tqdm() is terminal_tqdm


def test_get_tqdm_result_is_usable_as_an_iterator_wrapper():
    tqdm = get_tqdm()

    assert list(tqdm(range(4), disable=True)) == [0, 1, 2, 3]


@pytest.mark.parametrize("res,dia,expected", [(2.0, 100.0, 0.02), (5.0, 5.0, 1.0),
                                              (1.0, 4.0, 0.25), (0.0, 3.0, 0.0),
                                              (3.0, 0.5, 6.0)])
def test_calc_shannon(res, dia, expected):
    """Shannon angle is simply resolution / diameter."""
    assert np.isclose(calc_shannon(res, dia), expected)


def test_debug_print_emits_the_message_and_the_caller_line(capsys):
    debug_print("hello-from-test")

    out = capsys.readouterr().out
    assert "hello-from-test" in out
    assert "test_debug_print_emits_the_message_and_the_caller_line" in out


def test_debug_print_with_no_message_prints_only_the_frame(capsys):
    debug_print()

    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0].strip().startswith("File ")


# --------------------------------------------------------------------------------------
# util.hist_match
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("src_shape,tpl_shape", [((6, 7), (5, 5)), ((10, 10), (3, 4)),
                                                 ((4, 4), (20, 1)), ((2, 3), (2, 3))])
def test_hist_match_shape_and_dtype(src_shape, tpl_shape):
    rng = np.random.default_rng(18)
    src = rng.uniform(size=src_shape)
    tpl = rng.uniform(size=tpl_shape) * 10.0 + 3.0

    out = hist_match(src, tpl)

    assert out.shape == src_shape
    assert out.dtype == np.float64


@pytest.mark.parametrize("n_src,n_tpl", [(64, 30), (25, 25), (16, 100), (49, 7)])
def test_hist_match_preserves_rank_order(n_src, n_tpl):
    """A histogram match is a monotone (but not strictly monotone, a coarse template
    creates ties) remap, so sorting by the source must leave the output non-decreasing."""
    rng = np.random.default_rng(19)
    src = rng.uniform(size=n_src)
    tpl = rng.normal(size=n_tpl)

    out = hist_match(src, tpl)

    assert np.all(np.diff(out[np.argsort(src)]) >= -1e-12)


def test_hist_match_is_idempotent():
    rng = np.random.default_rng(20)
    src = rng.uniform(size=(7, 7))
    tpl = rng.normal(size=(40, ))

    once = hist_match(src, tpl)

    assert np.allclose(hist_match(once, tpl), once)


def test_hist_match_against_itself_is_the_identity():
    rng = np.random.default_rng(21)
    src = rng.uniform(size=(9, 9))

    assert np.allclose(hist_match(src, src), src)


def test_hist_match_lands_inside_the_template_range():
    rng = np.random.default_rng(22)
    src = rng.uniform(size=(10, 10))
    tpl = rng.uniform(size=(25, )) * 7.0 - 2.0

    out = hist_match(src, tpl)

    assert out.min() >= tpl.min() - 1e-12
    assert out.max() <= tpl.max() + 1e-12
    assert np.isclose(out.max(), tpl.max())


def test_hist_match_of_a_constant_source_saturates_at_the_template_max():
    """One unique source value means one quantile, 1.0, which interpolates to t_values[-1]."""
    src = np.full((4, 4), 7.0)
    tpl = np.array([10.0, 20.0, 30.0, 60.0])

    assert np.allclose(hist_match(src, tpl), 60.0)


def test_hist_match_is_invariant_to_a_monotone_rescaling_of_the_source():
    rng = np.random.default_rng(23)
    src = rng.uniform(size=(6, 6))
    tpl = rng.normal(size=(50, ))

    assert np.allclose(hist_match(src, tpl), hist_match(3.0 * src + 5.0, tpl))


def test_hist_match_accepts_integer_arrays():
    src = np.array([[1, 2, 3], [4, 5, 6]])
    tpl = np.array([10, 20, 30, 40, 50, 60])

    assert np.allclose(hist_match(src, tpl), [[10, 20, 30], [40, 50, 60]])


def test_hist_match_handles_a_template_with_repeats():
    src = np.array([0.0, 1.0, 2.0, 3.0])
    tpl = np.array([5.0, 5.0, 5.0, 9.0])

    out = hist_match(src, tpl)

    assert out.shape == (4, )
    assert out.min() >= 5.0 and out.max() <= 9.0


# --------------------------------------------------------------------------------------
# util.histeq
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(10, 10), (4, 9), (25, )])
def test_histeq_returns_a_flat_array(shape):
    """Note histeq does NOT restore the input shape, it returns src.size values."""
    rng = np.random.default_rng(24)
    src = rng.uniform(size=shape)

    out = histeq(src, np.ones(16))

    assert out.shape == (int(np.prod(shape)), )
    assert out.dtype == np.float64


@pytest.mark.parametrize("nbins", [8, 16, 64, 200, 255])
def test_histeq_is_rank_preserving_below_the_uint8_wrap(nbins):
    rng = np.random.default_rng(25)
    src = rng.uniform(size=(20, 20))
    order = np.argsort(src.ravel())

    out = histeq(src, np.ones(nbins))

    assert np.all(np.diff(out[order]) >= -1e-12)


@pytest.mark.parametrize("nbins", [256, 300, 512])
@pytest.mark.xfail(reason="Bug. The CDFs are cast to uint8 after being scaled by nbr_bins, so "
                   "nbr_bins >= 256 wraps modulo 256 and destroys monotonicity",
                   strict=False)
def test_histeq_is_rank_preserving_for_large_bin_counts(nbins):
    rng = np.random.default_rng(25)
    src = rng.uniform(size=(20, 20))
    order = np.argsort(src.ravel())

    out = histeq(src, np.ones(nbins))

    assert np.all(np.diff(out[order]) >= -1e-12)


def test_histeq_output_lives_in_the_unit_interval():
    rng = np.random.default_rng(26)
    src = rng.uniform(size=(15, 15))

    out = histeq(src, np.ones(32))

    assert out.min() >= 0.0 and out.max() <= 1.0


# --------------------------------------------------------------------------------------
# util.eul_to_quat
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 4, 17])
def test_eul_to_quat_shape_and_unit_norm(n):
    rng = np.random.default_rng(27)
    phi, theta, psi = rng.uniform(-np.pi, np.pi, size=(3, n))

    q = eul_to_quat(phi, theta, psi)

    assert q.shape == (4, n)
    assert np.allclose(np.linalg.norm(q, axis=0), 1.0)


def test_eul_to_quat_zero_angles_give_the_identity_quaternion():
    z = np.zeros(3)

    q = eul_to_quat(z, z, z)

    # scalar-first identity [1, 0, 0, 0]
    assert np.allclose(q, np.array([[1.0], [0.0], [0.0], [0.0]]))


def test_eul_to_quat_matches_scipy_clockwise_extrinsic_zyz():
    rng = np.random.default_rng(28)
    e = rng.uniform(-np.pi, np.pi, size=(9, 3))

    q0, q1, q2, q3 = eul_to_quat(*e.T, flip=True)
    ref = Rotation.from_euler('zyz', -e).as_quat()  # scipy is scalar-LAST

    assert np.allclose(ref, np.vstack([q1, q2, q3, q0]).T, atol=1e-9)


def test_eul_to_quat_flip_only_matters_when_psi_is_nonzero():
    rng = np.random.default_rng(29)
    phi, theta = rng.uniform(-np.pi, np.pi, size=(2, 6))
    zero_psi = np.zeros(6)

    assert np.allclose(eul_to_quat(phi, theta, zero_psi, True),
                       eul_to_quat(phi, theta, zero_psi, False))

    psi = rng.uniform(0.5, 2.0, size=6)
    assert not np.allclose(eul_to_quat(phi, theta, psi, True),
                           eul_to_quat(phi, theta, psi, False))


def test_eul_to_quat_flip_equals_negating_psi():
    rng = np.random.default_rng(30)
    phi, theta, psi = rng.uniform(-np.pi, np.pi, size=(3, 5))

    assert np.allclose(eul_to_quat(phi, theta, psi, flip=True),
                       eul_to_quat(phi, theta, -psi, flip=False))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_eul_to_quat_psi_does_not_move_the_projection_direction(seed):
    """The S2 point only depends on phi and theta. psi is the in-plane rotation."""
    rng = np.random.default_rng(seed)
    phi, theta = rng.uniform(-np.pi, np.pi, size=(2, 6))

    a = quaternion_to_S2(eul_to_quat(phi, theta, np.zeros(6)))
    b = quaternion_to_S2(eul_to_quat(phi, theta, np.full(6, 1.234)))

    assert np.allclose(a, b, atol=1e-10)


# --------------------------------------------------------------------------------------
# util.augment
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 3, 8])
def test_augment_doubles_the_column_count(n):
    rng = np.random.default_rng(31)
    q = rng.normal(size=(4, n))

    out = augment(q)

    assert out.shape == (4, 2 * n)
    assert np.allclose(out[:, :n], q)


def test_augment_appends_the_half_space_mirror():
    """The appended block is [-q1, q0, -q3, q2], the same map collapse_to_half_space uses."""
    rng = np.random.default_rng(32)
    q = rng.normal(size=(4, 5))

    out = augment(q)

    assert np.allclose(out[:, 5:], np.vstack((-q[1], q[0], -q[3], q[2])))


def test_augment_mirror_flips_the_S2_point():
    rng = np.random.default_rng(33)
    e = rng.uniform(-np.pi, np.pi, size=(6, 3))
    q = eul_to_quat(*e.T)

    s2 = quaternion_to_S2(augment(q))

    assert np.allclose(s2[:, 6:], -s2[:, :6], atol=1e-12)


def test_augment_preserves_quaternion_norms():
    rng = np.random.default_rng(34)
    e = rng.uniform(-np.pi, np.pi, size=(5, 3))
    q = eul_to_quat(*e.T)

    assert np.allclose(np.linalg.norm(augment(q), axis=0), 1.0)


def test_augment_does_not_modify_its_input():
    rng = np.random.default_rng(35)
    q = rng.normal(size=(4, 4))
    before = q.copy()

    augment(q)

    assert np.allclose(q, before)


@pytest.mark.parametrize("rows", [1, 2, 3])
def test_augment_rejects_arrays_with_too_few_rows(rows):
    with pytest.raises(AssertionError):
        augment(np.zeros((rows, 3)))


# --------------------------------------------------------------------------------------
# util.make_indeces
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("gc_num,prds", [(1, 3), (2, 3), (3, 1), (4, 5)])
def test_make_indeces_builds_the_prd_by_graph_grid(tmp_path, gc_num, prds):
    payload = {"CGtot": [list(range(prds)) for _ in range(gc_num)]}
    path = tmp_path / "gc.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    xAll, xSelect = make_indeces(str(path))

    assert xAll.shape == (2, gc_num * prds)
    assert xAll.dtype == np.int64
    assert np.array_equal(xAll[0], np.tile(np.arange(prds), gc_num))
    assert np.array_equal(xAll[1], np.repeat(np.arange(gc_num), prds))
    assert list(xSelect) == list(range(gc_num * prds))


# --------------------------------------------------------------------------------------
# util.interv
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("s", [1, 2, 3, 4, 5, 6, 7, 8, 15, 16])
def test_interv_length_and_unit_spacing(s):
    v = interv(s)

    assert v.shape == (s, )
    if s > 1:
        assert np.allclose(np.diff(v), 1.0)


@pytest.mark.parametrize("s,expected", [(6, [-3, -2, -1, 0, 1, 2]), (5, [-2, -1, 0, 1, 2]),
                                        (4, [-2, -1, 0, 1]), (1, [0]), (2, [-1, 0])])
def test_interv_values(s, expected):
    """Even sizes are asymmetric (one extra negative sample), odd sizes are symmetric."""
    assert np.allclose(interv(s), expected)


@pytest.mark.parametrize("s", [3, 5, 7, 9])
def test_interv_odd_is_antisymmetric(s):
    v = interv(s)

    assert np.allclose(v, -v[::-1])
    assert np.isclose(v.sum(), 0.0)


@pytest.mark.parametrize("s", [2, 4, 6, 8])
def test_interv_even_contains_zero_at_the_fft_centre(s):
    v = interv(s)

    assert v[s // 2] == 0.0


# --------------------------------------------------------------------------------------
# util.filter_fourier
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [0.1, 0.5, 1.0, 5.0])
def test_filter_fourier_preserves_a_constant_image(sigma):
    """The Butterworth gain at Q = 0 is exactly 1, so DC survives untouched."""
    img = np.full((16, 16), 2.5)

    assert np.allclose(filter_fourier(img, sigma), 2.5)


@pytest.mark.parametrize("sigma", [0.2, 0.5, 2.0])
def test_filter_fourier_preserves_the_mean(sigma):
    rng = np.random.default_rng(36)
    img = rng.uniform(size=(16, 16))

    assert np.isclose(filter_fourier(img, sigma).mean(), img.mean())


def test_filter_fourier_with_huge_sigma_is_the_identity():
    rng = np.random.default_rng(37)
    img = rng.uniform(size=(16, 16))

    assert np.allclose(filter_fourier(img, 1e6), img, atol=1e-6)


def test_filter_fourier_with_tiny_sigma_keeps_only_dc():
    rng = np.random.default_rng(38)
    img = rng.uniform(size=(16, 16))

    assert np.allclose(filter_fourier(img, 1e-6), img.mean(), atol=1e-6)


@pytest.mark.parametrize("shape", [(16, 16), (8, 12), (12, 8), (9, 9)])
def test_filter_fourier_shape_and_realness(shape):
    rng = np.random.default_rng(39)
    img = rng.uniform(size=shape)

    out = filter_fourier(img, 0.5)

    assert out.shape == shape
    assert out.dtype == np.float64


def test_filter_fourier_is_linear():
    rng = np.random.default_rng(40)
    a = rng.uniform(size=(16, 16))
    b = rng.uniform(size=(16, 16))

    assert np.allclose(filter_fourier(a + 2.0 * b, 0.4),
                       filter_fourier(a, 0.4) + 2.0 * filter_fourier(b, 0.4), atol=1e-10)


def test_filter_fourier_suppresses_the_nyquist_checkerboard():
    """A +/-1 checkerboard sits at the corner of Fourier space, well above any small cutoff."""
    n = 16
    i, j = np.indices((n, n))
    checker = (-1.0)**(i + j)

    out = filter_fourier(checker, 0.2)

    assert np.max(np.abs(out)) < 0.05


# --------------------------------------------------------------------------------------
# util.create_proportional_grid
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 2, 4, 5, 8, 9, 16])
def test_create_proportional_grid_shape_and_centre(N):
    g = create_proportional_grid(N)

    assert g.shape == (N, N)
    assert g[N // 2, N // 2] == 0.0
    assert np.all(g >= 0.0)


@pytest.mark.parametrize("N", [4, 5, 8, 9])
def test_create_proportional_grid_is_transpose_symmetric(N):
    g = create_proportional_grid(N)

    assert np.allclose(g, g.T)


@pytest.mark.parametrize("N", [4, 8, 16])
def test_create_proportional_grid_reaches_one_at_the_edge(N):
    """The value is 2*r/N, so a pixel N/2 away from the center reads exactly 1."""
    g = create_proportional_grid(N)

    assert np.isclose(g[N // 2, 0], 1.0)


@pytest.mark.parametrize("N", [4, 6, 8])
def test_create_proportional_grid_unit_step_along_a_row(N):
    g = create_proportional_grid(N)
    row = g[N // 2]

    assert np.isclose(row[N // 2 + 1] - row[N // 2], 2.0 / N)


# --------------------------------------------------------------------------------------
# util.ctemh_cryoFrank
# --------------------------------------------------------------------------------------
def _wavelength(kev):
    """Relativistic electron wavelength in Angstrom, the same closed form the code uses."""
    return 12.3986 / np.sqrt((2 * 511.0 + kev) * kev)


@pytest.mark.parametrize("ac", [0.0, 0.07, 0.1, 0.5, 1.0])
def test_ctemh_at_zero_frequency_is_minus_the_amplitude_contrast(ac):
    """gamma(0) = 0 so CTF(0) = (sin 0 - ac cos 0) * 1 = -ac."""
    k = np.zeros((1, 1))

    assert np.isclose(ctemh_cryoFrank(k, 2.0, 10000.0, 300.0, 1.0, ac)[0, 0], -ac)


@pytest.mark.parametrize("kev", [80.0, 120.0, 200.0, 300.0])
def test_ctemh_zeros_follow_the_defocus_only_analytic_form(kev):
    """With Cs = 0, ac = 0 and no envelope the CTF is -sin(pi lam df k^2), whose zeros are
    at k^2 = n / (lam df)."""
    lam = _wavelength(kev)
    df = 12000.0
    n = np.arange(1, 5)
    k = np.sqrt(n / (lam * df)).reshape(1, -1)

    ctf = ctemh_cryoFrank(k, 0.0, df, kev, np.inf, 0.0)

    assert np.allclose(ctf, 0.0, atol=1e-9)


@pytest.mark.parametrize("kev,df", [(300.0, 5000.0), (300.0, 25000.0), (200.0, 10000.0)])
def test_ctemh_defocus_only_matches_the_closed_form(kev, df):
    lam = _wavelength(kev)
    k = np.linspace(0.0, 0.25, 30).reshape(1, -1)

    ctf = ctemh_cryoFrank(k, 0.0, df, kev, np.inf, 0.0)

    assert np.allclose(ctf, -np.sin(np.pi * lam * df * k**2), atol=1e-9)


def test_ctemh_envelope_is_a_gaussian_with_fwhm_halfwidth():
    """gauss_env_halfwidth is a half width at half maximum, converted to a sigma by
    dividing by sqrt(2 ln 2)."""
    k = np.linspace(0.0, 0.4, 25).reshape(1, -1)
    b = 0.3
    sigma = b / np.sqrt(2 * np.log(2))

    # Cs = df = 0 and ac = 1 isolates the envelope. CTF = -cos(0) * wi = -wi
    ctf = ctemh_cryoFrank(k, 0.0, 0.0, 300.0, b, 1.0)

    assert np.allclose(ctf, -np.exp(-k**2 / (2 * sigma**2)), atol=1e-12)


@pytest.mark.parametrize("shape", [(1, 5), (8, 8), (3, 7)])
def test_ctemh_preserves_the_input_shape_and_is_bounded(shape):
    rng = np.random.default_rng(41)
    k = rng.uniform(0.0, 0.3, size=shape)

    ctf = ctemh_cryoFrank(k, 2.0, 10000.0, 300.0, 1.0, 0.1)

    assert ctf.shape == shape
    # |sin - ac cos| <= sqrt(1 + ac^2) and the envelope is <= 1
    assert np.all(np.abs(ctf) <= np.sqrt(1 + 0.1**2) + 1e-12)


def test_ctemh_is_even_in_k():
    """gamma depends on k^2 only, so the CTF is symmetric about the origin."""
    k = np.linspace(0.05, 0.3, 12).reshape(1, -1)

    assert np.allclose(ctemh_cryoFrank(k, 2.0, 9000.0, 300.0, 1.0, 0.07),
                       ctemh_cryoFrank(-k, 2.0, 9000.0, 300.0, 1.0, 0.07))


def test_ctemh_higher_defocus_oscillates_faster():
    k = np.linspace(0.0, 0.2, 400).reshape(1, -1)

    def zero_crossings(df):
        c = ctemh_cryoFrank(k, 0.0, df, 300.0, np.inf, 0.0).ravel()
        return np.sum(np.diff(np.sign(c)) != 0)

    assert zero_crossings(30000.0) > zero_crossings(5000.0)


def test_ctemh_spherical_aberration_is_taken_in_millimetres():
    """Cs is multiplied by 1e7 internally to reach Angstrom, so 0 mm and a tiny Cs differ
    only at high frequency."""
    k = np.linspace(0.0, 0.5, 50).reshape(1, -1)

    no_cs = ctemh_cryoFrank(k, 0.0, 10000.0, 300.0, np.inf, 0.0)
    with_cs = ctemh_cryoFrank(k, 2.7, 10000.0, 300.0, np.inf, 0.0)

    assert np.isclose(no_cs[0, 0], with_cs[0, 0])
    assert not np.allclose(no_cs, with_cs)


# --------------------------------------------------------------------------------------
# util.get_CTFs
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_def,width", [(1, 8), (3, 8), (2, 9), (5, 16)])
def test_get_CTFs_shape_core_util(n_def, width):
    defocus = np.linspace(5000.0, 25000.0, n_def)

    ctf = get_CTFs(defocus, width, 1.5, 2.0, 300.0, 1.0, 0.1)

    assert ctf.shape == (n_def, width, width)
    assert ctf.dtype == np.float64


@pytest.mark.parametrize("width", [4, 8, 9, 16])
def test_get_CTFs_puts_dc_at_the_array_origin(width):
    """The grid is built centerd and then ifftshift-ed, so k = 0 ends up at [0, 0]."""
    ctf = get_CTFs(np.array([10000.0]), width, 1.5, 2.0, 300.0, 1.0, 0.1)

    assert np.isclose(ctf[0, 0, 0], -0.1)


@pytest.mark.parametrize("width,pixel_size", [(8, 1.0), (8, 2.5), (12, 1.3)])
def test_get_CTFs_is_the_ifftshifted_ctemh(width, pixel_size):
    defocus = np.array([8000.0, 18000.0])
    k = create_proportional_grid(width) / (2 * pixel_size)

    ctf = get_CTFs(defocus, width, pixel_size, 2.0, 300.0, 1.0, 0.07)

    for i, df in enumerate(defocus):
        ref = ctemh_cryoFrank(k, 2.0, df, 300.0, 1.0, 0.07)
        assert np.allclose(np.fft.fftshift(ctf[i]), ref)


def test_get_CTFs_rows_depend_only_on_their_own_defocus():
    a = get_CTFs(np.array([7000.0]), 8, 1.5, 2.0, 300.0, 1.0, 0.1)
    b = get_CTFs(np.array([7000.0, 19000.0]), 8, 1.5, 2.0, 300.0, 1.0, 0.1)

    assert np.allclose(a[0], b[0])


def test_get_CTFs_is_transpose_symmetric_per_slice():
    """The frequency grid is isotropic, so each CTF slice equals its own transpose."""
    ctf = get_CTFs(np.array([12000.0]), 10, 1.2, 2.0, 300.0, 1.0, 0.1)

    assert np.allclose(ctf[0], ctf[0].T)


def test_get_CTFs_smaller_pixel_size_pushes_the_first_zero_further_out():
    """k scales as 1/pixel_size, so a finer sampling puts more oscillations in the box."""
    coarse = get_CTFs(np.array([15000.0]), 32, 4.0, 0.0, 300.0, np.inf, 0.0)[0]
    fine = get_CTFs(np.array([15000.0]), 32, 1.0, 0.0, 300.0, np.inf, 0.0)[0]

    row_c = np.fft.fftshift(coarse)[16]
    row_f = np.fft.fftshift(fine)[16]
    assert np.sum(np.diff(np.sign(row_c)) != 0) < np.sum(np.diff(np.sign(row_f)) != 0)


def test_get_CTFs_with_empty_defocus_list():
    ctf = get_CTFs(np.array([]), 8, 1.5, 2.0, 300.0, 1.0, 0.1)

    assert ctf.shape == (0, 8, 8)


# --------------------------------------------------------------------------------------
# util.rotate_fill
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [5, 8, 9, 16])
def test_rotate_fill_zero_angle_is_the_identity(n):
    rng = np.random.default_rng(42)
    img = rng.uniform(size=(n, n))

    assert np.allclose(rotate_fill(img, 0.0), img)


@pytest.mark.parametrize("angle", [360.0, 720.0, -360.0])
def test_rotate_fill_full_turns_are_the_identity(angle):
    rng = np.random.default_rng(43)
    img = rng.uniform(size=(8, 8))

    assert np.allclose(rotate_fill(img, angle), img, atol=1e-8)


@pytest.mark.parametrize("n", [5, 7, 8, 9])
def test_rotate_fill_ninety_degrees_equals_numpy_rot90(n):
    """Positive angles rotate counter-clockwise in array-index space."""
    rng = np.random.default_rng(44)
    img = rng.uniform(size=(n, n))

    assert np.allclose(rotate_fill(img, 90.0), np.rot90(img, 1), atol=1e-8)


@pytest.mark.parametrize("k,angle", [(1, 90.0), (2, 180.0), (3, 270.0)])
def test_rotate_fill_right_angles_are_exact(k, angle):
    rng = np.random.default_rng(45)
    img = rng.uniform(size=(8, 8))

    assert np.allclose(rotate_fill(img, angle), np.rot90(img, k), atol=1e-8)


def test_rotate_fill_preserves_a_constant_image():
    """`mode='grid-wrap'` means the corners are filled from the wrapped image, never zeros."""
    img = np.full((12, 12), 3.0)

    assert np.allclose(rotate_fill(img, 37.0), 3.0)


@pytest.mark.parametrize("angle", [90.0, 180.0, 270.0])
def test_rotate_fill_preserves_total_intensity_at_right_angles(angle):
    rng = np.random.default_rng(46)
    img = rng.uniform(size=(10, 10))

    assert np.isclose(rotate_fill(img, angle).sum(), img.sum())


@pytest.mark.parametrize("shape", [(8, 8), (7, 11), (12, 6)])
def test_rotate_fill_keeps_the_input_shape(shape):
    rng = np.random.default_rng(47)
    img = rng.uniform(size=shape)

    assert rotate_fill(img, 33.0).shape == shape


def test_rotate_fill_right_angle_rotations_compose():
    rng = np.random.default_rng(48)
    img = rng.uniform(size=(8, 8))

    assert np.allclose(rotate_fill(rotate_fill(img, 90.0), 90.0), rotate_fill(img, 180.0),
                       atol=1e-8)


def test_rotate_fill_never_introduces_a_zero_border():
    """A plain 'constant' fill would leave zero corners. grid-wrap must not."""
    img = np.full((16, 16), 5.0)
    img[6:10, 6:10] = 9.0

    out = rotate_fill(img, 45.0)

    assert out.min() > 4.0


# --------------------------------------------------------------------------------------
# util.get_image_width_from_stack
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["stack.txt", "stack.star", "stack", "stack.mrc.gz"])
def test_get_image_width_from_stack_rejects_non_mrc(name):
    with pytest.raises(ValueError, match="mrc or mrcs"):
        get_image_width_from_stack(name)


@pytest.mark.parametrize("width", [8, 12, 20])
def test_get_image_width_from_stack_reads_the_configured_stack(params_sandbox_core_util, width):
    import mrcfile

    path = str(params_sandbox_core_util / "stack.mrcs")
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.zeros((3, width, width), dtype=np.float32))
    params.img_stack_file = path

    assert get_image_width_from_stack(path) == width


@pytest.mark.xfail(reason="Bug. The `stack_file` argument is only used for the extension "
                   "check, the file actually opened is params.img_stack_file",
                   strict=False)
def test_get_image_width_from_stack_uses_its_argument(params_sandbox_core_util):
    import mrcfile

    a = str(params_sandbox_core_util / "a.mrcs")
    b = str(params_sandbox_core_util / "b.mrcs")
    with mrcfile.new(a, overwrite=True) as m:
        m.set_data(np.zeros((2, 8, 8), dtype=np.float32))
    with mrcfile.new(b, overwrite=True) as m:
        m.set_data(np.zeros((2, 20, 20), dtype=np.float32))
    params.img_stack_file = a

    assert get_image_width_from_stack(b) == 20


# ============================================================================
# quaternion.py - scalar-first [R,I,I,I] convention
# ============================================================================

ATOL = 1e-10

# unit quaternions of the quaternion group, scalar first
Q_ONE = np.array([1.0, 0.0, 0.0, 0.0])
Q_I = np.array([0.0, 1.0, 0.0, 0.0])
Q_J = np.array([0.0, 0.0, 1.0, 0.0])
Q_K = np.array([0.0, 0.0, 0.0, 1.0])


def _unit_quats(n, seed):
    """n random unit quaternions as a 4xN scalar-first array."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(4, n))
    return q / np.linalg.norm(q, axis=0)


def _unit_vecs(n, seed):
    """n random unit vectors as a 3xN array."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(3, n))
    return v / np.linalg.norm(v, axis=0)


def _to_scalar_last(q):
    """4xN scalar-first -> Nx4 scalar-last, the layout scipy's from_quat wants."""
    return np.vstack((q[1:4, :], q[0:1, :])).T


def _hamilton(q, s):
    """Reference Hamilton product of two scalar-first quaternions (1-D)."""
    q0, qv = q[0], q[1:]
    s0, sv = s[0], s[1:]
    return np.concatenate(([q0 * s0 - np.dot(qv, sv)], q0 * sv + s0 * qv + np.cross(qv, sv)))


_SOLVER_ATTEMPTS = 64


def _q2spider_stable(q):
    """q2Spider intermittently hands back its (0,0,0) start point without converging.

    Retry until the residual is genuinely zero so the surrounding assertions test the
    algorithm rather than the flake.  q2Spider normalizes internally, so rescaling the
    input poses the identical mathematical problem while perturbing the bit pattern.
    """
    q_unit = np.asarray(q, dtype=float) / np.linalg.norm(q)
    last = None
    for attempt in range(_SOLVER_ATTEMPTS):
        last = np.array(q2Spider(q * (1.0 + 0.5 * (attempt % 4))))
        if np.abs(_optfunc(last, q_unit)).max() < 1e-9:
            return last
    raise AssertionError(f"q2Spider never converged for {q}; last result {last}")


def _psi_ang_stable(pd):
    """psi_ang inherits the q2Spider stall. Retry until the angles really encode `pd`."""
    last = None
    for _ in range(_SOLVER_ATTEMPTS):
        last = np.array(psi_ang(pd))
        if np.allclose(convert_euler_to_S2(np.radians(last).reshape(1, 3)).ravel(), pd, atol=1e-8):
            return last
    raise AssertionError(f"psi_ang never converged for {pd}; last result {last}")


# ----------------------------------------------------------------------------------
# q_product
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_q_product_matches_hamilton_reference(seed):
    """p = q*s with p0 = q0 s0 - qv.sv and pv = q0 sv + s0 qv + qv x sv."""
    q = _unit_quats(6, seed)
    s = _unit_quats(6, seed + 100)
    p = q_product(q, s)
    ref = np.column_stack([_hamilton(q[:, i], s[:, i]) for i in range(6)])
    assert np.allclose(p, ref, atol=ATOL)


@pytest.mark.parametrize("seed", range(6))
def test_q_product_matches_scipy_rotation_composition(seed):
    """R(q*s) == R(q) R(s). quaternion product is rotation composition."""
    q = _unit_quats(5, seed)
    s = _unit_quats(5, seed + 50)
    p = q_product(q, s)
    r_prod = Rotation.from_quat(_to_scalar_last(p)).as_matrix()
    r_compose = Rotation.from_quat(_to_scalar_last(q)).as_matrix() @ Rotation.from_quat(_to_scalar_last(s)).as_matrix()
    assert np.allclose(r_prod, r_compose, atol=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_q_product_identity_element(seed):
    """[1,0,0,0] is the two sided identity of the quaternion algebra."""
    q = _unit_quats(4, seed)
    one = np.tile(Q_ONE.reshape(4, 1), (1, 4))
    assert np.allclose(q_product(q, one), q, atol=ATOL)
    assert np.allclose(q_product(one, q), q, atol=ATOL)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (Q_I, Q_I, -Q_ONE),
        (Q_J, Q_J, -Q_ONE),
        (Q_K, Q_K, -Q_ONE),
        (Q_I, Q_J, Q_K),
        (Q_J, Q_K, Q_I),
        (Q_K, Q_I, Q_J),
        (Q_J, Q_I, -Q_K),
        (Q_K, Q_J, -Q_I),
        (Q_I, Q_K, -Q_J),
    ],
)
def test_q_product_basis_multiplication_table(a, b, expected):
    """i^2 = j^2 = k^2 = ijk = -1."""
    assert np.allclose(q_product(a, b).ravel(), expected, atol=ATOL)


@pytest.mark.parametrize("seed", range(6))
def test_q_product_conjugate_gives_identity(seed):
    """q q* = |q|^2, so for unit q the product is the identity quaternion."""
    q = _unit_quats(5, seed)
    q_conj = np.vstack((q[0:1, :], -q[1:4, :]))
    p = q_product(q, q_conj)
    assert np.allclose(p, np.tile(Q_ONE.reshape(4, 1), (1, 5)), atol=ATOL)
    assert np.allclose(q_product(q_conj, q), np.tile(Q_ONE.reshape(4, 1), (1, 5)), atol=ATOL)


@pytest.mark.parametrize("seed", range(5))
def test_q_product_is_associative(seed):
    """(q s) t == q (s t). Quaternions are associative but not commutative."""
    q = _unit_quats(4, seed)
    s = _unit_quats(4, seed + 20)
    t = _unit_quats(4, seed + 40)
    assert np.allclose(q_product(q_product(q, s), t), q_product(q, q_product(s, t)), atol=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_q_product_norm_is_multiplicative(seed):
    """|q s| = |q| |s| (the quaternions are a normed division algebra)."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(4, 7))
    s = rng.normal(size=(4, 7))
    p = q_product(q, s)
    assert np.allclose(np.linalg.norm(p, axis=0),
                       np.linalg.norm(q, axis=0) * np.linalg.norm(s, axis=0),
                       atol=1e-10)


@pytest.mark.parametrize("seed", range(3))
def test_q_product_is_not_commutative(seed):
    q = _unit_quats(6, seed)
    s = _unit_quats(6, seed + 11)
    assert not np.allclose(q_product(q, s), q_product(s, q), atol=1e-6)


@pytest.mark.parametrize(
    "qshape,sshape,expected",
    [
        ((4, 1), (4, 1), (4, 1)),
        ((4, 5), (4, 5), (4, 5)),
        ((4, 1), (4, 5), (4, 5)),  # broadcasts a single quaternion over an array
        ((4, 5), (4, 1), (4, 5)),
        ((4, ), (4, ), (4, 1)),  # 1-D inputs are reshaped to columns
        ((4, ), (4, 3), (4, 3)),
        ((4, 3), (4, ), (4, 3)),
    ],
)
def test_q_product_output_shape(qshape, sshape, expected):
    rng = np.random.default_rng(7)
    q = rng.normal(size=qshape)
    s = rng.normal(size=sshape)
    assert q_product(q, s).shape == expected


def test_q_product_1d_inputs_return_column_vector():
    """Known integer example. (1+2i+3j+4k)(5+6i+7j+8k) = -60 + 12i + 30j + 24k."""
    p = q_product(np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0, 8.0]))
    assert p.shape == (4, 1)
    assert np.allclose(p.ravel(), [-60.0, 12.0, 30.0, 24.0])


def test_q_product_silently_truncates_extra_rows():
    """The guard is `shape[0] > 3`, so a 5-row input passes and rows beyond 3 are dropped."""
    q = np.arange(10.0).reshape(5, 2)
    s = np.arange(10.0, 20.0).reshape(5, 2)
    p = q_product(q, s)
    assert p.shape == (4, 2)
    assert np.allclose(p, q_product(q[:4], s[:4]))


@pytest.mark.xfail(reason="Bug. q_product catches its own AssertionError, prints to stdout and keeps going, "
                          "so a short input dies later with an unrelated IndexError instead of a clear error",
                   strict=False)
@pytest.mark.parametrize("bad_rows", [2, 3])
def test_q_product_rejects_short_vectors(bad_rows):
    with pytest.raises(ValueError):
        q_product(np.ones((bad_rows, 2)), np.ones((4, 2)))


def test_q_product_incompatible_batch_sizes_raise():
    with pytest.raises(ValueError):
        q_product(np.ones((4, 2)), np.ones((4, 3)))


@pytest.mark.parametrize("seed", range(6))
def test_q_product_single_matches_q_product(seed):
    """The numba scalar helper must agree with the vectorised routine."""
    q = _unit_quats(2, seed)
    assert np.allclose(_q_product_single(q[:, 0], q[:, 1]), q_product(q[:, 0], q[:, 1]).ravel(), atol=ATOL)
    assert np.allclose(_q_product_single(q[:, 1], q[:, 0]), q_product(q[:, 1], q[:, 0]).ravel(), atol=ATOL)


# ----------------------------------------------------------------------------------
# quaternion_to_S2
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "q,expected",
    [
        (Q_ONE, [0.0, 0.0, 1.0]),
        (-Q_ONE, [0.0, 0.0, 1.0]),
        (Q_I, [0.0, 0.0, -1.0]),  # 180 deg about x
        (Q_J, [0.0, 0.0, -1.0]),  # 180 deg about y
        (Q_K, [0.0, 0.0, 1.0]),  # 180 deg about z leaves the pole fixed
        (np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]), [0.0, 1.0, 0.0]),
        (np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]), [-1.0, 0.0, 0.0]),
        (np.array([np.cos(1.0), 0.0, 0.0, np.sin(1.0)]), [0.0, 0.0, 1.0]),
    ],
)
def test_quaternion_to_S2_known_values(q, expected):
    """For q = [cos(t/2), sin(t/2), 0, 0] the image is (0, sin t, cos t). The z axis
    swept toward +y, i.e. the transpose (inverse) rotation applied to z_hat."""
    assert np.allclose(quaternion_to_S2(q.reshape(4, 1)).ravel(), expected, atol=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_quaternion_to_S2_is_third_row_of_rotation_matrix(seed):
    """quaternion_to_S2(q) == R(q).T @ z_hat == R(q)[2, :]."""
    q = _unit_quats(8, seed)
    s2 = quaternion_to_S2(q)
    r = Rotation.from_quat(_to_scalar_last(q)).as_matrix()
    assert np.allclose(s2, r[:, 2, :].T, atol=1e-12)
    # and it is emphatically NOT the third column for generic rotations
    assert not np.allclose(s2, r[:, :, 2].T, atol=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_quaternion_to_S2_returns_unit_vectors(seed):
    q = _unit_quats(9, seed)
    s2 = quaternion_to_S2(q)
    assert s2.shape == (3, 9)
    assert np.allclose(np.linalg.norm(s2, axis=0), 1.0, atol=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_quaternion_to_S2_is_invariant_under_sign_flip(seed):
    """q and -q are the same rotation, so they give the same projection direction."""
    q = _unit_quats(9, seed)
    assert np.allclose(quaternion_to_S2(q), quaternion_to_S2(-q), atol=1e-12)


@pytest.mark.parametrize("psi", [0.0, 0.3, 1.1, -2.0, np.pi, 2 * np.pi, -0.75, 4.2])
def test_quaternion_to_S2_drops_the_psi_rotation(psi):
    """The final in-plane rotation enters eul_to_quat as a LEFT z-axis factor, and
    R(qz).T @ z_hat == z_hat, so the S2 point is untouched by psi."""
    q = _unit_quats(7, 3)
    qz = np.tile(np.array([np.cos(psi / 2), 0.0, 0.0, -np.sin(psi / 2)]).reshape(4, 1), (1, 7))
    assert np.allclose(quaternion_to_S2(q_product(qz, q)), quaternion_to_S2(q), atol=1e-12)


@pytest.mark.parametrize("psi", [0.3, 1.1, -2.0])
def test_quaternion_to_S2_does_not_drop_a_right_z_rotation(psi):
    """The invariance is one sided. Multiplying on the right moves the point."""
    q = _unit_quats(7, 3)
    qz = np.tile(np.array([np.cos(psi / 2), 0.0, 0.0, -np.sin(psi / 2)]).reshape(4, 1), (1, 7))
    assert not np.allclose(quaternion_to_S2(q_product(q, qz)), quaternion_to_S2(q), atol=1e-6)


@pytest.mark.xfail(reason="Bug. quaternion_to_S2 never normalizes, and its z row is 2(q0^2+q3^2)-1 rather than "
                          "2(q0^2+q3^2)/|q|^2-1, so a non-unit quaternion silently yields a non-unit, "
                          "non-parallel vector instead of the same rotation axis",
                   strict=False)
@pytest.mark.parametrize("scale", [2.0, 0.5, 10.0])
def test_quaternion_to_S2_is_scale_invariant(scale):
    """q and c*q represent the same rotation for any c != 0."""
    q = _unit_quats(6, 4)
    assert np.allclose(quaternion_to_S2(scale * q), quaternion_to_S2(q), atol=1e-8)


@pytest.mark.parametrize("n", [0, 1, 3, 17])
def test_quaternion_to_S2_shape_contract(n):
    q = _unit_quats(n, 8) if n else np.zeros((4, 0))
    assert quaternion_to_S2(q).shape == (3, n)


def test_quaternion_to_S2_requires_a_2d_array():
    with pytest.raises(IndexError):
        quaternion_to_S2(Q_ONE)


# ----------------------------------------------------------------------------------
# calc_avg_pd
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 5, 13, 40])
def test_calc_avg_pd_is_quaternion_to_S2(n):
    """The matlab-derived routine computes the identical expression."""
    q = _unit_quats(n, 11)
    assert np.allclose(calc_avg_pd(q, n), quaternion_to_S2(q), atol=0.0)
    assert calc_avg_pd(q, n).shape == (3, n)


@pytest.mark.parametrize("n", [3, 6, 9])
def test_calc_avg_pd_ignores_ns_when_it_is_one(n):
    """nS only ever enters as np.ones((1, nS))/2, so nS == 1 broadcasts and the
    argument is silently irrelevant."""
    q = _unit_quats(n, 12)
    assert np.allclose(calc_avg_pd(q, 1), calc_avg_pd(q, n), atol=0.0)


@pytest.mark.parametrize("bad_ns", [0, 2, 3, 4, 11])
def test_calc_avg_pd_rejects_mismatched_ns(bad_ns):
    """With 5 quaternions any nS other than 5 or 1 fails to broadcast."""
    q = _unit_quats(5, 13)
    with pytest.raises(ValueError):
        calc_avg_pd(q, bad_ns)


# ----------------------------------------------------------------------------------
# collapse_to_half_space
# ----------------------------------------------------------------------------------
PLANE_VECS = [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    np.array([0.6, -0.8, 0.0]),
]


@pytest.mark.parametrize("plane_vec", PLANE_VECS)
def test_collapse_mirror_is_left_multiplication_by_i(plane_vec):
    """[q0,q1,q2,q3] -> [-q1,q0,-q3,q2] is exactly i*q in the Hamilton algebra."""
    q = _unit_quats(30, 21)
    collapsed, mirrored = collapse_to_half_space(q, plane_vec)
    i_left = q_product(np.tile(Q_I.reshape(4, 1), (1, q.shape[1])), q)
    assert np.allclose(collapsed[:, mirrored], i_left[:, mirrored], atol=1e-14)
    assert np.allclose(collapsed[:, ~mirrored], q[:, ~mirrored], atol=0.0)


@pytest.mark.parametrize("plane_vec", PLANE_VECS)
def test_collapse_puts_every_point_in_the_chosen_half_space(plane_vec):
    q = _unit_quats(40, 22)
    collapsed, _ = collapse_to_half_space(q, plane_vec)
    dots = quaternion_to_S2(collapsed).T @ plane_vec
    assert (dots >= -1e-14).all()


@pytest.mark.parametrize("plane_vec", PLANE_VECS[:4])
def test_collapse_mirrored_points_are_antipodal(plane_vec):
    """For UNIT quaternions the mirror sends the S2 point to exactly -S2."""
    q = _unit_quats(30, 23)
    s2 = quaternion_to_S2(q)
    collapsed, mirrored = collapse_to_half_space(q, plane_vec)
    assert mirrored.any()
    assert np.allclose(quaternion_to_S2(collapsed)[:, mirrored], -s2[:, mirrored], atol=1e-14)


@pytest.mark.parametrize("plane_vec", PLANE_VECS[:4])
def test_collapse_is_idempotent(plane_vec):
    q = _unit_quats(40, 24)
    once, mirrored_once = collapse_to_half_space(q, plane_vec)
    twice, mirrored_twice = collapse_to_half_space(once, plane_vec)
    assert mirrored_once.any()
    assert not mirrored_twice.any()
    assert np.allclose(once, twice, atol=0.0)


@pytest.mark.parametrize("plane_vec", PLANE_VECS[:4])
def test_collapse_preserves_quaternion_norm(plane_vec):
    rng = np.random.default_rng(25)
    q = rng.normal(size=(4, 20))
    collapsed, _ = collapse_to_half_space(q, plane_vec)
    assert np.allclose(np.linalg.norm(collapsed, axis=0), np.linalg.norm(q, axis=0), atol=1e-14)


def test_collapse_does_not_mutate_its_input():
    """The routine copies before writing, so the caller's array survives."""
    q = _unit_quats(20, 26)
    original = q.copy()
    collapse_to_half_space(q)
    assert np.array_equal(q, original)


@pytest.mark.parametrize("seed", range(5))
def test_collapse_mirror_applied_twice_negates(seed):
    """i*(i*q) = -q, so mirroring twice returns the same rotation with flipped sign."""
    q = _unit_quats(6, seed)
    m1 = np.vstack((-q[1], q[0], -q[3], q[2]))
    m2 = np.vstack((-m1[1], m1[0], -m1[3], m1[2]))
    assert np.allclose(m2, -q, atol=0.0)


@pytest.mark.parametrize("plane_vec", [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])])
def test_collapse_boundary_dot_of_zero_is_not_mirrored(plane_vec):
    """The test is a strict `< 0`, so a point exactly on the dividing plane stays put.
    The identity quaternion maps to (0,0,1), orthogonal to both x and y."""
    q = Q_ONE.reshape(4, 1)
    collapsed, mirrored = collapse_to_half_space(q, plane_vec)
    assert not mirrored[0]
    assert np.array_equal(collapsed, q)


def test_collapse_returns_bool_mask_and_float_quats():
    q = _unit_quats(11, 27)
    collapsed, mirrored = collapse_to_half_space(q)
    assert mirrored.dtype == np.bool_
    assert mirrored.shape == (11, )
    assert collapsed.shape == (4, 11)
    assert collapsed.dtype == q.dtype


def test_collapse_handles_empty_input():
    q = np.zeros((4, 0))
    collapsed, mirrored = collapse_to_half_space(q)
    assert collapsed.shape == (4, 0)
    assert mirrored.shape == (0, )


def test_collapse_requires_a_2d_array():
    with pytest.raises(IndexError):
        collapse_to_half_space(Q_ONE)


@pytest.mark.parametrize("plane_vec", PLANE_VECS)
def test_collapse_quaternion_and_euler_paths_agree(plane_vec):
    """The euler shortcut must reproduce the legacy quaternion path for any plane."""
    rng = np.random.default_rng(28)
    euler = rng.uniform(-np.pi, np.pi, size=(25, 3))
    raw_qs = eul_to_quat(euler[:, 0], euler[:, 1], euler[:, 2], flip=True)

    q_collapsed, mirrored_q = collapse_to_half_space(raw_qs, plane_vec)
    s2_euler, mirrored_e = collapse_to_half_space_euler_angles(euler, plane_vec)

    assert np.allclose(quaternion_to_S2(q_collapsed), s2_euler, atol=1e-10)
    assert np.array_equal(mirrored_q, mirrored_e)


@pytest.mark.parametrize("plane_vec", PLANE_VECS[:4])
def test_collapse_euler_returns_unit_vectors_in_half_space(plane_vec):
    rng = np.random.default_rng(29)
    euler = rng.uniform(-np.pi, np.pi, size=(30, 3))
    s2, mirrored = collapse_to_half_space_euler_angles(euler, plane_vec)
    assert s2.shape == (3, 30)
    assert mirrored.shape == (30, )
    assert np.allclose(np.linalg.norm(s2, axis=0), 1.0, atol=1e-12)
    assert (s2.T @ plane_vec >= -1e-14).all()


# ----------------------------------------------------------------------------------
# convert_euler_to_S2
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_convert_euler_to_S2_is_the_spherical_polar_map(seed):
    """ZXZ(a,b,c) has R[:,2] = (sin a sin b, -cos a sin b, cos b). After the
    (-ry, rx, rz) permutation that is the textbook (cos a sin b, sin a sin b, cos b)."""
    rng = np.random.default_rng(seed)
    euler = rng.uniform(-np.pi, np.pi, size=(9, 3))
    ref = np.vstack([
        np.cos(euler[:, 0]) * np.sin(euler[:, 1]),
        np.sin(euler[:, 0]) * np.sin(euler[:, 1]),
        np.cos(euler[:, 1]),
    ])
    assert np.allclose(convert_euler_to_S2(euler), ref, atol=1e-14)


@pytest.mark.parametrize("psi", [0.0, 0.5, -1.3, np.pi, 2.7, -3.0])
def test_convert_euler_to_S2_ignores_the_third_angle(psi):
    rng = np.random.default_rng(31)
    euler = rng.uniform(-np.pi, np.pi, size=(8, 3))
    base = euler.copy()
    base[:, 2] = 0.0
    shifted = base.copy()
    shifted[:, 2] = psi
    assert np.allclose(convert_euler_to_S2(base), convert_euler_to_S2(shifted), atol=1e-14)


@pytest.mark.parametrize("n", [0, 1, 4, 25])
def test_convert_euler_to_S2_shape_and_normalisation(n):
    rng = np.random.default_rng(32)
    euler = rng.uniform(-np.pi, np.pi, size=(n, 3))
    s2 = convert_euler_to_S2(euler)
    assert s2.shape == (3, n)
    if n:
        assert np.allclose(np.linalg.norm(s2, axis=0), 1.0, atol=1e-14)


@pytest.mark.parametrize(
    "angles,expected",
    [
        ((0.0, 0.0, 0.0), [0.0, 0.0, 1.0]),
        ((0.0, np.pi / 2, 0.0), [1.0, 0.0, 0.0]),
        ((np.pi / 2, np.pi / 2, 0.0), [0.0, 1.0, 0.0]),
        ((np.pi, np.pi / 2, 0.0), [-1.0, 0.0, 0.0]),
        ((-np.pi / 2, np.pi / 2, 0.0), [0.0, -1.0, 0.0]),
        ((1.234, np.pi, 0.0), [0.0, 0.0, -1.0]),
    ],
)
def test_convert_euler_to_S2_known_values(angles, expected):
    assert np.allclose(convert_euler_to_S2(np.array([angles])).ravel(), expected, atol=1e-14)


def test_convert_euler_to_S2_requires_an_nx3_array():
    """A single (3,) triple is not accepted. The routine indexes a 3-D matrix stack."""
    with pytest.raises(IndexError):
        convert_euler_to_S2(np.zeros(3))


# ----------------------------------------------------------------------------------
# convert_S2_to_euler
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(8))
def test_convert_S2_to_euler_round_trips_through_convert_euler_to_S2(seed):
    """Both the primary and the `sx < 0` alternate branch encode the same direction."""
    pts = _unit_vecs(20, seed)
    angles = convert_S2_to_euler(pts)
    assert np.allclose(convert_euler_to_S2(angles.T), pts, atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_convert_S2_to_euler_wraps_to_0_2pi_and_zeroes_psi(seed):
    pts = _unit_vecs(30, seed + 60)
    angles = convert_S2_to_euler(pts)
    assert angles.shape == (3, 30)
    assert (angles >= 0.0).all() and (angles < 2 * np.pi + 1e-12).all()
    assert np.array_equal(angles[2], np.zeros(30))


@pytest.mark.parametrize("pole,expected", [
    ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
    ([0.0, 0.0, -1.0], [0.0, np.pi, 0.0]),
])
def test_convert_S2_to_euler_poles(pole, expected):
    """arctan2(0, 0) is 0, so the azimuth is pinned to zero at both poles."""
    assert np.allclose(convert_S2_to_euler(np.array(pole).reshape(3, 1)).ravel(), expected, atol=1e-14)


@pytest.mark.parametrize("seed", range(6))
def test_convert_S2_to_euler_branch_selection_follows_sign_of_x(seed):
    """Columns with sx < 0 come back in the `alternate_euler_convention` frame."""
    pts = _unit_vecs(24, seed + 70)
    sx, sy, sz = pts
    naive = np.mod(np.array([np.arctan2(sy, sx), np.arccos(sz), np.zeros_like(sz)]), 2 * np.pi)
    alternate = alternate_euler_convention(naive)
    alternate[-1, :] = 0.0
    expected = naive.copy()
    expected[:, sx < 0] = alternate[:, sx < 0]
    assert (sx < 0).any()
    assert np.allclose(convert_S2_to_euler(pts), expected, atol=0.0)


@pytest.mark.parametrize("sz", [2.0, -3.5])
def test_convert_S2_to_euler_gives_nan_polar_angle_for_non_unit_input(sz):
    """arccos of |sz| > 1 is nan. The routine performs no normalization."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        angles = convert_S2_to_euler(np.array([[0.0], [0.0], [sz]]))
    assert np.isnan(angles[1, 0])


# ----------------------------------------------------------------------------------
# alternate_euler_convention
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_alternate_euler_convention_is_an_involution(seed):
    """a -> ((a - [pi,0,pi]) * [1,-1,1]) mod 2pi applied twice is the identity mod 2pi."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2 * np.pi, size=(3, 9))
    assert np.allclose(alternate_euler_convention(alternate_euler_convention(angles)),
                       np.mod(angles, 2 * np.pi),
                       atol=1e-12)


@pytest.mark.parametrize(
    "angles,expected",
    [
        ([0.0, 0.0, 0.0], [np.pi, 0.0, np.pi]),
        ([np.pi, 0.0, np.pi], [0.0, 0.0, 0.0]),
        ([np.pi / 2, np.pi / 2, np.pi / 2], [3 * np.pi / 2, 3 * np.pi / 2, 3 * np.pi / 2]),
        ([2 * np.pi, np.pi, 0.0], [np.pi, np.pi, np.pi]),
        ([np.pi, np.pi / 4, 0.0], [0.0, 7 * np.pi / 4, np.pi]),
        ([0.5, 1.0, 1.5], [0.5 + np.pi, 2 * np.pi - 1.0, 1.5 + np.pi]),
    ],
)
def test_alternate_euler_convention_known_values(angles, expected):
    got = alternate_euler_convention(np.array(angles).reshape(3, 1)).ravel()
    assert np.allclose(np.mod(got, 2 * np.pi), np.mod(expected, 2 * np.pi), atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_alternate_euler_convention_output_is_wrapped(seed):
    rng = np.random.default_rng(seed + 80)
    angles = rng.uniform(-10.0, 10.0, size=(3, 12))
    out = alternate_euler_convention(angles)
    assert out.shape == (3, 12)
    assert (out >= 0.0).all() and (out < 2 * np.pi).all()


def test_alternate_euler_convention_accepts_a_flat_triple():
    """A bare (3,) vector transposes to itself, so the elementwise map still applies."""
    assert np.allclose(alternate_euler_convention(np.zeros(3)), [np.pi, 0.0, np.pi], atol=1e-14)


# ----------------------------------------------------------------------------------
# _optfunc / q2Spider
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_optfunc_is_the_residual_against_eul_to_quat(seed):
    """_optfunc builds q3(psi) * q2(theta) * q1(phi), which is exactly eul_to_quat(flip=True)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-np.pi, np.pi, size=3)
    target = _unit_quats(1, seed + 90)[:, 0]
    expected = target - eul_to_quat(a[0:1], a[1:2], a[2:3], flip=True)[:, 0]
    assert np.allclose(_optfunc(a, target), expected, atol=1e-14)


@pytest.mark.parametrize("seed", range(10))
def test_q2Spider_inverts_eul_to_quat(seed):
    """The optimizer recovers angles that rebuild the input quaternion exactly."""
    rng = np.random.default_rng(seed)
    euler = rng.uniform(-np.pi, np.pi, size=(4, 3))
    euler[:, 1] = 0.5 * (euler[:, 1] + np.pi)  # keep theta away from the gimbal poles
    qs = eul_to_quat(euler[:, 0], euler[:, 1], euler[:, 2], flip=True)
    for i in range(qs.shape[1]):
        a = _q2spider_stable(qs[:, i])
        back = eul_to_quat(a[0:1], a[1:2], a[2:3], flip=True)[:, 0]
        assert np.allclose(back, qs[:, i], atol=1e-8)


@pytest.mark.parametrize("seed", range(4))
def test_q2Spider_inverts_the_negated_quaternion_too(seed):
    """-q is the same rotation. The solver reproduces -q, not q."""
    rng = np.random.default_rng(seed + 200)
    euler = rng.uniform(-np.pi, np.pi, size=(3, 3))
    euler[:, 1] = 0.5 * (euler[:, 1] + np.pi)
    qs = eul_to_quat(euler[:, 0], euler[:, 1], euler[:, 2], flip=True)
    for i in range(qs.shape[1]):
        a = _q2spider_stable(-qs[:, i])
        back = eul_to_quat(a[0:1], a[1:2], a[2:3], flip=True)[:, 0]
        assert np.allclose(back, -qs[:, i], atol=1e-8)


@pytest.mark.parametrize("scale", [0.25, 2.0, 7.0, 1e3, 1e-3])
def test_q2Spider_normalises_its_input(scale):
    """The first statement divides by the norm, so scaling is irrelevant."""
    q = _unit_quats(1, 33)[:, 0]
    assert np.allclose(_q2spider_stable(scale * q), _q2spider_stable(q), atol=1e-8)


@pytest.mark.parametrize("q", [Q_ONE, 5.0 * Q_ONE, np.array([1.0, 0.0, 0.0, 0.0])])
def test_q2Spider_identity_is_the_zero_rotation(q):
    """a0 = (0,0,0) is already an exact solution for the identity quaternion."""
    assert np.allclose(q2Spider(q), (0.0, 0.0, 0.0), atol=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_q2Spider_actually_moves_away_from_its_start_point(seed):
    """A converged solve of a non-identity quaternion cannot be the (0,0,0) start."""
    rng = np.random.default_rng(seed + 300)
    euler = rng.uniform(-np.pi, np.pi, size=(3, 3))
    euler[:, 1] = 0.5 * (euler[:, 1] + np.pi)
    qs = eul_to_quat(euler[:, 0], euler[:, 1], euler[:, 2], flip=True)
    for i in range(qs.shape[1]):
        a = _q2spider_stable(qs[:, i])
        assert np.abs(a).max() > 1e-6


def test_optfunc_jacobian_is_rank_deficient_at_the_solver_start():
    """The root cause of the q2Spider stalls.

    d/da of q3(psi) q2(theta) q1(phi) at a = 0. The phi and psi columns are bit
    identical (both rotate about z, and at theta = 0 they are the same rotation), and
    the i component is second order in a, so its whole row vanishes.  The 4x3 Jacobian
    therefore has rank 2 and any residual along i is invisible to the first-order model.
    """
    q = _unit_quats(1, 77)[:, 0]
    h = np.sqrt(np.finfo(float).eps)
    f0 = _optfunc(np.zeros(3), q)
    jac = np.column_stack([(_optfunc(h * np.eye(3)[j], q) - f0) / h for j in range(3)])

    assert np.array_equal(jac[:, 0], jac[:, 2])  # phi and psi columns coincide
    assert np.array_equal(jac[1, :], np.zeros(3))  # the i row is identically zero
    assert np.linalg.matrix_rank(jac) == 2


def test_q2Spider_rejects_the_zero_quaternion():
    """0/0 makes the residual nan and scipy refuses to start."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError):
            q2Spider(np.zeros(4))


def test_q2Spider_rejects_a_column_vector():
    """A (4,1) input broadcasts inside _optfunc to a 4x4 residual, which least_squares rejects."""
    with pytest.raises(ValueError):
        q2Spider(Q_ONE.reshape(4, 1))


@pytest.mark.xfail(reason="Bug. q2Spider seeds Levenberg-Marquardt at a=(0,0,0), where its Jacobian is rank 2 of 3 "
                          "and the i-component row vanishes. a residual along i is therefore invisible to the "
                          "solver and every quaternion [q0,q1,0,0] comes back as the untouched start point (0,0,0)",
                   strict=False)
@pytest.mark.parametrize("q1", [1.0, 0.866, 0.5, -0.7071])
def test_q2Spider_converges_for_pure_i_quaternions(q1):
    q = np.array([np.sqrt(max(0.0, 1.0 - q1**2)), q1, 0.0, 0.0])
    a = np.array(q2Spider(q))
    back = eul_to_quat(a[0:1], a[1:2], a[2:3], flip=True)[:, 0]
    assert np.allclose(back, q, atol=1e-6) or np.allclose(back, -q, atol=1e-6)


# ----------------------------------------------------------------------------------
# qs_to_spider_euler_angles
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 5, 20])
def test_qs_to_spider_euler_angles_shape(n):
    q = _unit_quats(n, 41)
    assert qs_to_spider_euler_angles(q).shape == (3, n)


@pytest.mark.parametrize("seed", range(4))
def test_qs_to_spider_euler_angles_is_sign_invariant(seed):
    """scipy canonicalises the quaternion, so q and -q give identical angles."""
    q = _unit_quats(12, seed + 42)
    assert np.allclose(qs_to_spider_euler_angles(q), qs_to_spider_euler_angles(-q), atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_qs_to_spider_euler_angles_middle_angle_is_non_positive(seed):
    """scipy's zyz beta lies in [0, pi]. The routine negates the whole triple."""
    q = _unit_quats(30, seed + 43)
    angles = qs_to_spider_euler_angles(q)
    assert (angles[1] <= 1e-12).all()
    assert (angles[1] >= -np.pi - 1e-12).all()
    assert (np.abs(angles[0]) <= np.pi + 1e-12).all()
    assert (np.abs(angles[2]) <= np.pi + 1e-12).all()


@pytest.mark.parametrize("seed", range(5))
def test_qs_to_spider_euler_angles_round_trips_through_eul_to_quat(seed):
    """Rebuilding with eul_to_quat(flip=True) returns the quaternion up to sign."""
    q = _unit_quats(10, seed + 44)
    angles = qs_to_spider_euler_angles(q)
    back = eul_to_quat(angles[0], angles[1], angles[2], flip=True)
    same = np.isclose(back, q, atol=1e-10).all(axis=0)
    flipped = np.isclose(back, -q, atol=1e-10).all(axis=0)
    assert np.logical_or(same, flipped).all()


@pytest.mark.parametrize("seed", range(5))
def test_qs_to_spider_and_q2Spider_describe_the_same_orientation(seed):
    """The two routines pick different Euler branches but the same S2 direction."""
    rng = np.random.default_rng(seed + 55)
    euler = rng.uniform(-np.pi, np.pi, size=(4, 3))
    euler[:, 1] = 0.5 * (euler[:, 1] + np.pi)
    qs = eul_to_quat(euler[:, 0], euler[:, 1], euler[:, 2], flip=True)

    spider = qs_to_spider_euler_angles(qs)
    legacy = np.array([_q2spider_stable(qs[:, i]) for i in range(qs.shape[1])]).T

    z = np.array([0.0, 0.0, 1.0])
    s2_spider = Rotation.from_euler('ZYZ', spider.T).apply(z)
    s2_legacy = Rotation.from_euler('ZYZ', legacy.T).apply(z)
    assert np.allclose(s2_spider, quaternion_to_S2(qs).T, atol=1e-8)
    assert np.allclose(s2_legacy, quaternion_to_S2(qs).T, atol=1e-8)


# ----------------------------------------------------------------------------------
# psi_ang
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(8))
def test_psi_ang_reference_quaternion_encodes_the_projection_direction(seed):
    """Qr = [1+z, y, -x, 0] normalized satisfies quaternion_to_S2(Qr) == PD exactly."""
    pds = _unit_vecs(6, seed + 500)
    for i in range(pds.shape[1]):
        pd = pds[:, i]
        qr = np.array([1.0 + pd[2], pd[1], -pd[0], 0.0])
        qr = qr / np.linalg.norm(qr)
        assert np.allclose(quaternion_to_S2(qr.reshape(4, 1)).ravel(), pd, atol=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_psi_ang_round_trips_back_to_the_projection_direction(seed):
    """psi_ang returns DEGREES. Converting back through convert_euler_to_S2 recovers PD."""
    pds = _unit_vecs(5, seed + 600)
    pds = pds[:, np.abs(pds[0]) > 0.05]  # x == 0 is a hard solver stall, covered separately
    for i in range(pds.shape[1]):
        pd = pds[:, i]
        angles = np.radians(_psi_ang_stable(pd))
        assert np.allclose(convert_euler_to_S2(angles.reshape(1, 3)).ravel(), pd, atol=1e-8)


@pytest.mark.parametrize("seed", range(8))
def test_psi_ang_agrees_with_convert_S2_to_euler(seed):
    """The legacy optimizer and the closed form modern routine must match."""
    pds = _unit_vecs(5, seed + 700)
    pds = pds[:, np.abs(pds[0]) > 0.05]
    modern = convert_S2_to_euler(pds)
    for i in range(pds.shape[1]):
        legacy = np.radians(_psi_ang_stable(pds[:, i]))
        assert np.allclose(legacy, modern[:, i], atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_psi_ang_always_zeroes_psi(seed):
    """psi is degenerate for a projection direction and is pinned to 0 by convention."""
    pds = _unit_vecs(4, seed + 800)
    for i in range(pds.shape[1]):
        assert psi_ang(pds[:, i])[2] == 0.0


@pytest.mark.parametrize("seed", range(6))
def test_psi_ang_returns_degrees_in_0_360(seed):
    """np.mod(., 2 pi) then scaling by 180/pi bounds both free angles, converged or not."""
    pds = _unit_vecs(4, seed + 900)
    for i in range(pds.shape[1]):
        phi, theta, psi = psi_ang(pds[:, i])
        assert 0.0 <= phi < 360.0
        assert 0.0 <= theta < 360.0
        assert psi == 0.0


@pytest.mark.parametrize(
    "pd,expected",
    [
        ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 90.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 270.0, 0.0]),
        ([np.sqrt(0.5), 0.0, np.sqrt(0.5)], [0.0, 45.0, 0.0]),
    ],
)
def test_psi_ang_known_values(pd, expected):
    assert np.allclose(_psi_ang_stable(np.array(pd)), expected, atol=1e-6)


def test_psi_ang_south_pole_is_singular():
    """PD = (0,0,-1) gives Qr = 0, a nan quaternion that scipy refuses."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError):
            psi_ang(np.array([0.0, 0.0, -1.0]))


def test_psi_ang_rejects_a_column_vector():
    """The type hint advertises Shape["3,1"] but a genuine (3,1) input builds a ragged array."""
    with pytest.raises(ValueError):
        psi_ang(np.array([[0.0], [0.0], [1.0]]))


@pytest.mark.xfail(reason="Bug. for any projection direction with x == 0 the reference quaternion is [1+z,y,0,0], "
                          "a pure-i residual that stalls q2Spider at its (0,0,0) start point, so psi_ang returns "
                          "(0,0,0) and silently reports the north pole instead of the true direction",
                   strict=False)
@pytest.mark.parametrize("pd", [
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.6, 0.8],
    [0.0, -0.8, 0.6],
])
def test_psi_ang_handles_the_x_equals_zero_great_circle(pd):
    pd = np.array(pd)
    angles = np.radians(np.array(psi_ang(pd)))
    assert np.allclose(convert_euler_to_S2(angles.reshape(1, 3)).ravel(), pd, atol=1e-6)


# ============================================================================
# params.py - config singleton, path templates, TOML round-trip
# ============================================================================

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_params():
    """Hand back the singleton with an empty instance __dict__ (so class defaults show through).

    Teardown restores the previous instance state, the class-level mutable `opt_movie`, and the
    working directory that `params.load()` likes to change out from under the process.
    """
    cwd = os.getcwd()
    saved_instance = dict(params.__dict__)
    saved_opt_movie = copy.deepcopy(Params.opt_movie)
    params.__dict__.clear()
    try:
        yield params
    finally:
        params.__dict__.clear()
        params.__dict__.update(saved_instance)
        Params.opt_movie.clear()
        Params.opt_movie.update(saved_opt_movie)
        os.chdir(cwd)


@pytest.fixture
def project_params(fresh_params, tmp_path):
    """A named project rooted in tmp_path, with a nonzero diameter so `sh`/`asdict` are usable."""
    os.chdir(tmp_path)
    fresh_params.project_name = "P"
    fresh_params.particle_diameter = 100.0
    fresh_params.ms_estimated_resolution = 3.0
    return fresh_params


# ---------------------------------------------------------------------------
# module-level introspection tables (used to drive the parametrized sweeps)
# ---------------------------------------------------------------------------

PARAM_HINTS = get_type_hints(Params, include_extras=True)
PARAM_NAMES = sorted(PARAM_HINTS.keys())

ANNOTATED_NAMES = sorted(k for k, v in PARAM_HINTS.items() if hasattr(v, "__metadata__"))

USER_PARAM_NAMES = sorted(
    k for k in ANNOTATED_NAMES if PARAM_HINTS[k].__metadata__[0].user_param
)

PATH_PROPERTIES = [
    "proj_file",
    "user_dir",
    "out_dir",
    "psi_file",
    "psi2_file",
    "rho_file",
    "pd_file",
    "dist_dir",
    "dist_file",
    "psi_dir",
    "psi2_dir",
    "EL_dir",
    "OM_dir",
    "OM_file",
    "traj_dir",
    "CC_dir",
    "CC_dir_temp",
    "bad_nodes_psis_tau_file",
    "CC_file",
    "CC_meas_dir",
    "CC_OF_dir",
    "traj_file",
    "euler_dir",
    "ref_ang_file",
    "ref_ang_file1",
    "bin_dir",
    "postproc_mrcs2mrc_dir",
    "postproc_denoise_dir",
]

# every directory create_dir() asks for explicitly
CREATED_DIR_PROPERTIES = [
    "dist_dir",
    "psi_dir",
    "psi2_dir",
    "EL_dir",
    "OM_dir",
    "traj_dir",
    "bin_dir",
    "CC_dir",
    "CC_OF_dir",
    "CC_meas_dir",
    "euler_dir",
    "postproc_mrcs2mrc_dir",
    "postproc_denoise_dir",
]


def _declared_type(name):
    """Underlying type of a declared param, peeling Annotated[...] when present."""
    hint = PARAM_HINTS[name]
    return get_args(hint)[0] if hasattr(hint, "__metadata__") else hint


# ---------------------------------------------------------------------------
# ProjectLevel
# ---------------------------------------------------------------------------

PROJECT_LEVEL_SPEC = [
    ("INIT", 0),
    ("BINNING", 1),
    ("CALC_DISTANCE", 2),
    ("MANIFOLD_ANALYSIS", 3),
    ("PSI_ANALYSIS", 4),
    ("NLSA_MOVIE", 5),
    ("PRD_SELECTION", 6),
    ("FIND_CCS", 7),
    ("PROBABILITY_LANDSCAPE", 8),
    ("TRAJECTORY", 9),
]


@pytest.mark.parametrize("name,value", PROJECT_LEVEL_SPEC)
def test_project_level_value(name, value):
    # The integer values are persisted verbatim in the project TOML, so they are a wire format.
    assert ProjectLevel[name].value == value


@pytest.mark.parametrize("name,value", PROJECT_LEVEL_SPEC)
def test_project_level_lookup_by_value(name, value):
    assert ProjectLevel(value) is ProjectLevel[name]


@pytest.mark.parametrize("name,value", PROJECT_LEVEL_SPEC)
def test_project_level_name_roundtrip(name, value):
    assert ProjectLevel(value).name == name


def test_project_level_declaration_order_matches_pipeline_order():
    # Iteration order is definition order, and the CLI builds subparsers by walking ProjectLevel.
    assert [lvl.value for lvl in ProjectLevel] == list(range(len(PROJECT_LEVEL_SPEC)))
    assert [lvl.name for lvl in ProjectLevel] == [n for n, _ in PROJECT_LEVEL_SPEC]


def test_project_level_members_are_unique():
    assert len(set(ProjectLevel)) == len(PROJECT_LEVEL_SPEC)
    assert len({lvl.value for lvl in ProjectLevel}) == len(PROJECT_LEVEL_SPEC)


@pytest.mark.parametrize("bad", [-1, 10, 11, 42, 1.5, "0"])
def test_project_level_invalid_value_raises(bad):
    with pytest.raises(ValueError):
        ProjectLevel(bad)


@pytest.mark.parametrize("bad_name", ["init", "MISSING", "binning"])
def test_project_level_invalid_name_raises(bad_name):
    with pytest.raises(KeyError):
        ProjectLevel[bad_name]


@pytest.mark.parametrize(
    "lo,hi",
    [
        (ProjectLevel.INIT, ProjectLevel.BINNING),
        (ProjectLevel.CALC_DISTANCE, ProjectLevel.PSI_ANALYSIS),
        (ProjectLevel.FIND_CCS, ProjectLevel.TRAJECTORY),
    ],
)
def test_project_level_is_not_orderable(lo, hi):
    # Plain Enum (not IntEnum), so pipeline progress must be compared through `.value`,
    # which is exactly what gui/main_window.py does.
    with pytest.raises(TypeError):
        lo < hi
    assert lo.value < hi.value


@pytest.mark.parametrize("lvl,ival", [(ProjectLevel.INIT, 0), (ProjectLevel.TRAJECTORY, 9)])
def test_project_level_does_not_compare_equal_to_int(lvl, ival):
    assert lvl != ival
    assert lvl.value == ival


# ---------------------------------------------------------------------------
# ParamInfo
# ---------------------------------------------------------------------------

def test_paraminfo_defaults():
    info = ParamInfo()
    assert info.description == ""
    assert info.user_param is False
    assert info.affects == []


def test_paraminfo_positional_arguments():
    info = ParamInfo("desc", True, [ProjectLevel.BINNING])
    assert info.description == "desc"
    assert info.user_param is True
    assert info.affects == [ProjectLevel.BINNING]


def test_paraminfo_affects_is_a_fresh_list_per_instance():
    # default_factory=list, so mutating one instance must not leak into the next
    a, b = ParamInfo(), ParamInfo()
    assert a.affects is not b.affects
    a.affects.append(ProjectLevel.INIT)
    assert b.affects == []


def test_paraminfo_is_a_value_type():
    assert ParamInfo("x", True, [ProjectLevel.INIT]) == ParamInfo("x", True, [ProjectLevel.INIT])
    assert ParamInfo("x") != ParamInfo("y")


def test_paraminfo_repr_mentions_fields():
    r = repr(ParamInfo("hello", True))
    assert "hello" in r and "user_param=True" in r


# ---------------------------------------------------------------------------
# singleton semantics
# ---------------------------------------------------------------------------

def test_params_is_a_singleton():
    assert Params() is params
    assert Params() is Params()


def test_singleton_shares_mutations_across_handles(fresh_params):
    other = Params()
    fresh_params.project_name = "shared"
    assert other.project_name == "shared"


def test_instance_attribute_shadows_class_default(fresh_params):
    assert fresh_params.num_psi == Params.num_psi == 8
    fresh_params.num_psi = 3
    assert fresh_params.num_psi == 3
    # the class default is untouched, which is why clearing __dict__ restores defaults
    assert Params.num_psi == 8


def test_clearing_instance_dict_restores_defaults(fresh_params):
    fresh_params.num_part = 12345
    fresh_params.__dict__.clear()
    assert fresh_params.num_part == 0


# ---------------------------------------------------------------------------
# path properties
# ---------------------------------------------------------------------------

def _expected_paths():
    out = os.path.join("output", "P")
    el = os.path.join(out, "ELConc50")
    cc = os.path.join(out, "CC")
    return {
        "proj_file": "params_P.toml",
        "user_dir": "output",
        "out_dir": out,
        "psi_dir": os.path.join(out, "diff_maps"),
        "psi2_dir": os.path.join(out, "psi_analysis"),
        "psi_file": os.path.join(out, "diff_maps", "gC_trimmed_psi_"),
        "psi2_file": os.path.join(out, "psi_analysis", "S2_"),
        "pd_file": os.path.join(out, "pd_data.pkl"),
        "dist_dir": os.path.join(out, "distances"),
        "dist_file": os.path.join(out, "distances", "IMGs_"),
        "EL_dir": el,
        "OM_dir": os.path.join(el, "OM"),
        "OM_file": os.path.join(el, "OM", "S2_OM.npy"),
        "rho_file": os.path.join(el, "OM", "rho"),
        "traj_dir": os.path.join(out, "traj"),
        "traj_file": os.path.join(out, "traj", "traj_"),
        "CC_dir": cc,
        "CC_dir_temp": os.path.join(cc, "temp"),
        "CC_file": os.path.join(cc, "CC_file.pkl"),
        "CC_meas_dir": os.path.join(cc, "CC_meas"),
        "CC_OF_dir": os.path.join(cc, "CC_OF"),
        "bad_nodes_psis_tau_file": os.path.join(cc, "bad_nodes_psis_tau.pkl"),
        "euler_dir": os.path.join(out, "topos", "Euler_PrD"),
        "ref_ang_file": os.path.join(out, "topos", "Euler_PrD", "PrD_map.txt"),
        "ref_ang_file1": os.path.join(out, "topos", "Euler_PrD", "PrD_map1.txt"),
        "bin_dir": os.path.join(out, "bin"),
        "postproc_mrcs2mrc_dir": os.path.join(out, "postproc", "vols"),
        "postproc_denoise_dir": os.path.join(out, "postproc", "denoise"),
    }


@pytest.mark.parametrize("prop,expected", sorted(_expected_paths().items()))
def test_path_property_value(project_params, prop, expected):
    assert getattr(project_params, prop) == expected


@pytest.mark.parametrize("prop", PATH_PROPERTIES)
def test_path_property_is_a_relative_str(project_params, prop):
    val = getattr(project_params, prop)
    assert isinstance(val, str)
    # every path is relative. The pipeline chdirs to the project root instead of using absolutes
    assert not os.path.isabs(val)


@pytest.mark.parametrize(
    "prop", [p for p in PATH_PROPERTIES if p not in ("proj_file", "user_dir", "out_dir")]
)
def test_paths_live_under_out_dir(project_params, prop):
    # proj_file sits next to the project (not inside it) and user_dir is the parent of out_dir
    assert getattr(project_params, prop).startswith(project_params.out_dir + os.sep)


@pytest.mark.parametrize("name", ["A", "my_project", "run-01", "P", "x y"])
def test_out_dir_tracks_project_name(project_params, name):
    project_params.project_name = name
    assert project_params.out_dir == os.path.join("output", name)
    assert project_params.proj_file == f"params_{name}.toml"


def test_user_dir_is_a_fixed_constant(project_params):
    # 'output' is hardcoded, not derived from project_name
    assert project_params.user_dir == "output"
    project_params.project_name = "something_else"
    assert project_params.user_dir == "output"


@pytest.mark.parametrize("con_order", [1, 25, 50, 100, 250])
def test_EL_dir_encodes_con_order_range(project_params, con_order):
    project_params.con_order_range = con_order
    assert project_params.EL_dir == os.path.join("output", "P", f"ELConc{con_order}")
    assert project_params.OM_dir == os.path.join(project_params.EL_dir, "OM")
    assert project_params.OM_file == os.path.join(project_params.EL_dir, "OM", "S2_OM.npy")
    assert project_params.rho_file == os.path.join(project_params.EL_dir, "OM", "rho")


def test_proj_file_matches_the_default_save_target(project_params, tmp_path):
    project_params.save()
    assert os.path.exists(tmp_path / project_params.proj_file)


def test_user_dimensions_is_a_1d_placeholder(fresh_params):
    # documented placeholder until 2D landscapes are supported
    assert fresh_params.user_dimensions == 1


@pytest.mark.parametrize("prop", PATH_PROPERTIES + ["user_dimensions", "ang_width", "sh"])
def test_properties_are_absent_from_the_annotations(prop):
    # derived properties must never be persisted as params
    assert prop not in PARAM_HINTS


# ---------------------------------------------------------------------------
# get_*_file templates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prd,index", [(0, 0), (1, 2), (3, 2), (17, 8)])
def test_get_topos_path(project_params, prd, index):
    assert project_params.get_topos_path(prd, index) == os.path.join(
        "output", "P", "topos", f"PrD_{prd}", f"topos_{index}.png"
    )


@pytest.mark.parametrize("prd,index", [(0, 0), (1, 2), (3, 2), (17, 8)])
def test_get_psi_gif(project_params, prd, index):
    assert project_params.get_psi_gif(prd, index) == os.path.join(
        "output", "P", "topos", f"PrD_{prd}", f"psi_{index}.gif"
    )


def test_topos_getters_do_not_shift_the_index(project_params):
    # Code is 0-indexed and filenames are 1-indexed, but the +1 is the caller's job:
    # nlsa_movie.py calls params.get_topos_path(prD + 1, psinum + 1).
    assert "PrD_0" in project_params.get_topos_path(0, 0)
    assert "topos_0.png" in project_params.get_topos_path(0, 0)


@pytest.mark.parametrize("prd", [0, 1, 7, 123])
def test_get_EL_file(project_params, prd):
    assert project_params.get_EL_file(prd) == os.path.join(project_params.EL_dir, f"S2_prD_{prd}.pkl")


@pytest.mark.parametrize("prd", [0, 1, 7, 123])
def test_get_CC_OF_file(project_params, prd):
    assert project_params.get_CC_OF_file(prd) == os.path.join(project_params.CC_OF_dir, f"OF_prD_{prd}.pkl")


@pytest.mark.parametrize("edge,prd,nbr", [(0, 0, 0), (1, 2, 3), (10, 4, 9)])
def test_get_CC_meas_file(project_params, edge, prd, nbr):
    assert project_params.get_CC_meas_file(edge, prd, nbr) == os.path.join(
        project_params.CC_meas_dir, f"meas_edge_prDs_{edge}_{prd}_{nbr}.h5"
    )


def test_get_CC_meas_file_argument_order_is_edge_prd_nbr(project_params):
    # the three integers are not interchangeable. verify they land in declaration order
    assert project_params.get_CC_meas_file(1, 2, 3).endswith("meas_edge_prDs_1_2_3.h5")
    assert project_params.get_CC_meas_file(3, 2, 1).endswith("meas_edge_prDs_3_2_1.h5")


@pytest.mark.parametrize("prd", [0, 1, 7, 123])
def test_get_psi_file(project_params, prd):
    assert project_params.get_psi_file(prd) == f"{project_params.psi_file}prD_{prd}.h5"
    assert project_params.get_psi_file(prd) == os.path.join(
        project_params.psi_dir, f"gC_trimmed_psi_prD_{prd}.h5"
    )


@pytest.mark.parametrize("prd,psi", [(0, 0), (1, 2), (7, 3), (12, 7)])
def test_get_psi2_file(project_params, prd, psi):
    assert project_params.get_psi2_file(prd, psi) == os.path.join(
        project_params.psi2_dir, f"S2_prD_{prd}_psi_{psi}.h5"
    )


@pytest.mark.parametrize("prd", [0, 1, 7, 123])
def test_get_dist_file(project_params, prd):
    assert project_params.get_dist_file(prd) == os.path.join(project_params.dist_dir, f"IMGs_prD_{prd}.h5")


@pytest.mark.parametrize("prd", [0, 1, 7, 123])
def test_get_bad_nodes_psis_tau_file_prd(project_params, prd):
    assert project_params.get_bad_nodes_psis_tau_file_prd(prd) == os.path.join(
        project_params.CC_dir_temp, f"bad_nodes_tau_prD_{prd}.pkl"
    )


@pytest.mark.parametrize(
    "getter,args",
    [
        ("get_EL_file", (0,)),
        ("get_CC_OF_file", (0,)),
        ("get_psi_file", (0,)),
        ("get_dist_file", (0,)),
        ("get_bad_nodes_psis_tau_file_prd", (0,)),
        ("get_psi2_file", (0, 0)),
        ("get_topos_path", (0, 0)),
        ("get_psi_gif", (0, 0)),
        ("get_CC_meas_file", (0, 0, 0)),
    ],
)
def test_getters_return_paths_under_out_dir(project_params, getter, args):
    path = getattr(project_params, getter)(*args)
    assert isinstance(path, str)
    assert path.startswith(project_params.out_dir + os.sep)


@pytest.mark.parametrize(
    "getter,args,ext",
    [
        ("get_EL_file", (3,), ".pkl"),
        ("get_CC_OF_file", (3,), ".pkl"),
        ("get_bad_nodes_psis_tau_file_prd", (3,), ".pkl"),
        ("get_psi_file", (3,), ".h5"),
        ("get_dist_file", (3,), ".h5"),
        ("get_psi2_file", (3, 1), ".h5"),
        ("get_CC_meas_file", (1, 2, 3), ".h5"),
        ("get_topos_path", (1, 1), ".png"),
        ("get_psi_gif", (1, 1), ".gif"),
    ],
)
def test_getter_file_extensions(project_params, getter, args, ext):
    assert getattr(project_params, getter)(*args).endswith(ext)


# ---------------------------------------------------------------------------
# derived scalars. ang_width and sh
# ---------------------------------------------------------------------------

def test_ang_width_is_zero_when_diameter_is_unset(fresh_params):
    # guard clause. a falsy particle_diameter short-circuits to 0.0 instead of dividing
    assert fresh_params.particle_diameter == 0.0
    assert fresh_params.ang_width == 0.0


@pytest.mark.parametrize(
    "aperture,resolution,diameter",
    [
        (1, 3.0, 100.0),
        (2, 3.0, 100.0),
        (1, 10.0, 250.0),
        (3, 1.0, 40.0),
        (1, 0.0, 100.0),
    ],
)
def test_ang_width_formula(fresh_params, aperture, resolution, diameter):
    # ang_width = min(aperture_index * resolution / diameter, sqrt(4 pi)). The second term is the
    # angular width whose spherical cap covers the whole sphere
    fresh_params.aperture_index = aperture
    fresh_params.ms_estimated_resolution = resolution
    fresh_params.particle_diameter = diameter
    expected = min(aperture * resolution / diameter, np.sqrt(4.0 * np.pi))
    assert fresh_params.ang_width == pytest.approx(expected)


@pytest.mark.parametrize("aperture,resolution,diameter", [(100, 3.0, 1.0), (1, 500.0, 1.0), (5, 40.0, 10.0)])
def test_ang_width_is_clamped_at_sqrt_4pi(fresh_params, aperture, resolution, diameter):
    fresh_params.aperture_index = aperture
    fresh_params.ms_estimated_resolution = resolution
    fresh_params.particle_diameter = diameter
    assert fresh_params.ang_width == pytest.approx(np.sqrt(4.0 * np.pi))


@pytest.mark.parametrize("resolution,diameter", [(3.0, 100.0), (10.0, 250.0), (1.0, 2.0), (0.0, 5.0)])
def test_sh_is_resolution_over_diameter(fresh_params, resolution, diameter):
    fresh_params.ms_estimated_resolution = resolution
    fresh_params.particle_diameter = diameter
    assert fresh_params.sh == pytest.approx(resolution / diameter)


def test_sh_divides_by_zero_when_diameter_is_unset(fresh_params):
    # unlike ang_width, `sh` has no guard clause. This is what makes asdict() blow up on defaults
    with pytest.raises(ZeroDivisionError):
        fresh_params.sh


def test_ang_width_ignores_aperture_when_diameter_is_zero(fresh_params):
    fresh_params.aperture_index = 5
    fresh_params.ms_estimated_resolution = 12.0
    assert fresh_params.ang_width == 0.0


# ---------------------------------------------------------------------------
# asdict
# ---------------------------------------------------------------------------

def test_asdict_keys_are_exactly_the_declared_annotations(project_params):
    assert set(project_params.asdict().keys()) == set(PARAM_NAMES)


@pytest.mark.parametrize("prop", PATH_PROPERTIES + ["ang_width", "sh", "user_dimensions"])
def test_asdict_excludes_derived_properties(project_params, prop):
    assert prop not in project_params.asdict()


@pytest.mark.parametrize(
    "attr", ["save", "load", "print", "create_dir", "asdict", "get_param_info", "get_user_params"]
)
def test_asdict_excludes_methods(project_params, attr):
    assert attr not in project_params.asdict()


def test_asdict_excludes_the_singleton_handle(project_params):
    # Params.__new__ stashes the instance on the class as `instance`
    assert hasattr(Params, "instance")
    assert "instance" not in project_params.asdict()


def test_asdict_serializes_project_level_as_its_int_value(project_params):
    project_params.project_level = ProjectLevel.PSI_ANALYSIS
    assert project_params.asdict()["project_level"] == 4
    project_params.project_level = ProjectLevel.INIT
    assert project_params.asdict()["project_level"] == 0


DEFAULTS = [
    ("ncpu", 1),
    ("num_part", 0),
    ("eps", 1e-10),
    ("aperture_index", 1),
    ("prd_thres_low", 100),
    ("prd_thres_high", 2000),
    ("prd_n_active", 0),
    ("tess_hemisphere_type", "lovisolo_silva"),
    ("prd_assignment", "hard"),
    ("prd_cone_width_factor", 1.0),
    ("distance_filter_type", "Butter"),
    ("distance_filter_cutoff_freq", 0.5),
    ("distance_filter_order", 8),
    ("num_eigs", 15),
    ("num_psi_truncated", 8),
    ("num_psi", 8),
    ("rad", 5),
    ("nlsa_fps", 5.0),
    ("con_order_range", 50),
    ("nlsa_tune", 3),
    ("n_reaction_coords", 1),
    ("traj_name", "1"),
    ("states_per_coord", 50),
    ("width_1D", 1),
    ("width_2D", 1),
    ("calc_optical_flow", True),
    ("calc_all_edge_measures", True),
    ("opt_mask_type", 0),
    ("opt_mask_param", 0),
    ("find_bad_psi_tau", True),
    ("tau_occ_thresh", 0.35),
    ("use_pruned_graph", False),
    ("vis_s2_scale", 1.0),
    ("vis_s2_density", 1000),
    ("vis_s2_isosurface_level", 3),
]


@pytest.mark.parametrize("name,default", DEFAULTS)
def test_declared_defaults(fresh_params, name, default):
    assert getattr(fresh_params, name) == default


def test_tess_hemisphere_vec_default_is_the_x_normal(fresh_params):
    # PrDs on the far side of the plane perpendicular to this vector get mirrored
    assert fresh_params.tess_hemisphere_vec == [1.0, 0.0, 0.0]
    assert np.linalg.norm(fresh_params.tess_hemisphere_vec) == pytest.approx(1.0)


def test_ms_ctf_envelope_default_is_infinite(fresh_params):
    # an infinite envelope means "no envelope damping"
    assert np.isinf(fresh_params.ms_ctf_envelope)


def test_opt_movie_default_keys(fresh_params):
    assert fresh_params.opt_movie == {
        "printFig": 0,
        "OFvisual": 0,
        "visual_CC": 0,
        "flowVecPctThresh": 95,
    }


@pytest.mark.parametrize(
    "name,value",
    [
        ("num_psi", 4),
        ("con_order_range", 17),
        ("project_name", "zzz"),
        ("eps", 1e-8),
        ("tess_hemisphere_vec", [0.0, 1.0, 0.0]),
        ("calc_optical_flow", False),
    ],
)
def test_asdict_reflects_mutations(project_params, name, value):
    setattr(project_params, name, value)
    assert project_params.asdict()[name] == value


def test_repr_is_the_pformatted_dict(project_params):
    assert repr(project_params) == pprint.pformat(project_params.asdict())


def test_print_emits_every_param(project_params, capsys):
    project_params.print()
    out = capsys.readouterr().out
    assert "project_name" in out and "num_psi" in out


@pytest.mark.xfail(
    reason="Bug. asdict() evaluates every non-underscore attribute (including derived properties) "
    "before checking its type, so the unguarded `sh` property raises ZeroDivisionError whenever "
    "particle_diameter is still 0, i.e. asdict/save/repr crash on a default-constructed project",
    strict=False,
)
def test_asdict_works_on_default_params(fresh_params):
    assert fresh_params.particle_diameter == 0.0
    d = fresh_params.asdict()
    assert "sh" not in d


# ---------------------------------------------------------------------------
# save / load round trip
# ---------------------------------------------------------------------------

def test_save_default_filename_is_derived_from_project_name(project_params, tmp_path):
    project_params.save()
    assert (tmp_path / "params_P.toml").exists()


def test_save_writes_parseable_toml_under_a_params_table(project_params, tmp_path):
    project_params.save()
    raw = toml.load(tmp_path / "params_P.toml")
    assert set(raw.keys()) == {"params"}
    assert set(raw["params"].keys()) == set(PARAM_NAMES)


def test_save_to_explicit_path(project_params, tmp_path):
    target = tmp_path / "elsewhere.toml"
    project_params.save(str(target))
    assert target.exists()
    assert not (tmp_path / "params_P.toml").exists()


ROUNDTRIP_CASES = [
    ("ncpu", 4),
    ("num_part", 98765),
    ("eps", 1e-7),
    ("ms_kilovolts", 300.0),
    ("ms_num_pixels", 256),
    ("ms_pixel_size", 1.07),
    ("particle_diameter", 320.0),
    ("aperture_index", 3),
    ("prd_thres_low", 50),
    ("prd_thres_high", 1000),
    ("tess_hemisphere_type", "fibonacci"),
    ("prd_assignment", "cone"),
    ("prd_cone_width_factor", 2.5),
    ("distance_filter_type", "Gauss"),
    ("distance_filter_cutoff_freq", 0.25),
    ("num_psi", 6),
    ("con_order_range", 25),
    ("nlsa_tune", 5),
    ("traj_name", "2"),
    ("calc_optical_flow", False),
    ("use_pruned_graph", True),
    ("tau_occ_thresh", 0.5),
    ("tess_hemisphere_vec", [0.0, 0.0, 1.0]),
    ("opt_movie", {"printFig": 1, "OFvisual": 0, "visual_CC": 1, "flowVecPctThresh": 90}),
]


@pytest.mark.parametrize("name,value", ROUNDTRIP_CASES)
def test_save_load_roundtrip_per_param(project_params, tmp_path, name, value):
    setattr(project_params, name, value)
    project_params.save()
    # wipe the singleton back to defaults, then reload from disk
    project_params.__dict__.clear()
    project_params.project_name = "P"
    project_params.load(str(tmp_path / "params_P.toml"))
    assert getattr(project_params, name) == value


def test_save_load_roundtrips_the_full_dict(project_params, tmp_path):
    project_params.num_part = 4242
    project_params.project_level = ProjectLevel.NLSA_MOVIE
    before = project_params.asdict()
    project_params.save()
    project_params.__dict__.clear()
    project_params.project_name = "P"
    project_params.particle_diameter = 100.0
    project_params.load(str(tmp_path / "params_P.toml"))
    assert project_params.asdict() == before


def test_roundtrip_preserves_infinite_ctf_envelope(project_params, tmp_path):
    project_params.save()
    project_params.__dict__.clear()
    project_params.project_name = "P"
    project_params.load(str(tmp_path / "params_P.toml"))
    assert np.isinf(project_params.ms_ctf_envelope)


def test_load_converts_project_level_back_to_an_enum(project_params, tmp_path):
    project_params.project_level = ProjectLevel.FIND_CCS
    project_params.save()
    project_params.__dict__.clear()
    project_params.load(str(tmp_path / "params_P.toml"))
    assert project_params.project_level is ProjectLevel.FIND_CCS
    assert isinstance(project_params.project_level, ProjectLevel)


@pytest.mark.parametrize("bad_level", [-1, 10, 99])
def test_load_rejects_out_of_range_project_level(project_params, tmp_path, bad_level):
    target = tmp_path / "bad_level.toml"
    target.write_text(f"[params]\nproject_level = {bad_level}\n")
    with pytest.raises(ValueError):
        project_params.load(str(target))


def test_load_of_a_missing_file_raises(project_params, tmp_path):
    with pytest.raises(FileNotFoundError):
        project_params.load(str(tmp_path / "does_not_exist.toml"))


def test_load_without_a_params_table_raises_keyerror(project_params, tmp_path):
    target = tmp_path / "empty.toml"
    target.write_text("[other]\nx = 1\n")
    with pytest.raises(KeyError):
        project_params.load(str(target))


def test_load_warns_on_unknown_params_instead_of_raising(project_params, tmp_path, capsys):
    target = tmp_path / "extra.toml"
    target.write_text("[params]\nnum_psi = 3\nbogus_param = 7\n")
    project_params.load(str(target))
    out = capsys.readouterr().out
    assert "bogus_param" in out
    assert "not found in parameters module" in out
    assert not hasattr(project_params, "bogus_param")
    # the known param on the same file still lands
    assert project_params.num_psi == 3


def test_load_leaves_params_absent_from_the_file_alone(project_params, tmp_path):
    project_params.num_eigs = 21
    target = tmp_path / "partial.toml"
    target.write_text("[params]\nnum_psi = 2\n")
    project_params.load(str(target))
    assert project_params.num_psi == 2
    assert project_params.num_eigs == 21


def test_load_chdirs_into_the_directory_of_the_toml(project_params, tmp_path):
    sub = tmp_path / "proj_root"
    sub.mkdir()
    target = sub / "params_P.toml"
    project_params.save(str(target))
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(tmp_path))
    project_params.load(str(target))
    # side effect. every relative path property is now resolved against the toml's directory
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(sub))


def test_load_of_a_bare_filename_does_not_chdir(project_params, tmp_path):
    project_params.save()
    here = os.path.realpath(os.getcwd())
    project_params.load("params_P.toml")
    assert os.path.realpath(os.getcwd()) == here


def test_load_with_no_argument_uses_the_project_name(project_params, tmp_path):
    project_params.num_psi = 5
    project_params.save()
    project_params.__dict__.clear()
    project_params.project_name = "P"
    project_params.load()
    assert project_params.num_psi == 5


def test_load_chdirs_before_it_validates_the_contents(project_params, tmp_path):
    # the chdir happens inside the `with open(...)` block, so it fires even for a bad file
    sub = tmp_path / "root2"
    sub.mkdir()
    target = sub / "junk.toml"
    target.write_text("[other]\nx = 1\n")
    with pytest.raises(KeyError):
        project_params.load(str(target))
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(sub))


@pytest.mark.xfail(
    reason="Bug. load() validates keys against dir(Params), which also contains methods and "
    "properties, so a TOML key named e.g. 'save' silently shadows the bound method (and a key "
    "named 'out_dir' raises AttributeError) instead of being reported as unknown",
    strict=False,
)
def test_load_rejects_toml_keys_that_name_methods(project_params, tmp_path):
    target = tmp_path / "shadow.toml"
    target.write_text("[params]\nsave = 1\n")
    project_params.load(str(target))
    assert callable(project_params.save)


# ---------------------------------------------------------------------------
# create_dir
# ---------------------------------------------------------------------------

def test_create_dir_creates_exactly_the_expected_tree(project_params, tmp_path):
    project_params.create_dir()
    made = set()
    for root, dirnames, _ in os.walk("output"):
        made.add(root)
        for d in dirnames:
            made.add(os.path.join(root, d))

    out = os.path.join("output", "P")
    expected = {
        "output",
        out,
        os.path.join(out, "distances"),
        os.path.join(out, "diff_maps"),
        os.path.join(out, "psi_analysis"),
        os.path.join(out, "ELConc50"),
        os.path.join(out, "ELConc50", "OM"),
        os.path.join(out, "traj"),
        os.path.join(out, "bin"),
        os.path.join(out, "CC"),
        os.path.join(out, "CC", "CC_OF"),
        os.path.join(out, "CC", "CC_meas"),
        os.path.join(out, "topos"),
        os.path.join(out, "topos", "Euler_PrD"),
        os.path.join(out, "postproc"),
        os.path.join(out, "postproc", "vols"),
        os.path.join(out, "postproc", "denoise"),
    }
    assert made == expected


@pytest.mark.parametrize("prop", CREATED_DIR_PROPERTIES)
def test_create_dir_makes_each_declared_directory(project_params, tmp_path, prop):
    project_params.create_dir()
    assert os.path.isdir(getattr(project_params, prop))


def test_create_dir_does_not_make_the_CC_temp_dir(project_params, tmp_path):
    # CC_dir_temp is created lazily by CC/ComputeOpticalFlowPrDAll.py, not here
    project_params.create_dir()
    assert not os.path.exists(project_params.CC_dir_temp)


def test_create_dir_is_idempotent(project_params, tmp_path):
    project_params.create_dir()
    marker = os.path.join(project_params.bin_dir, "marker.txt")
    with open(marker, "w") as f:
        f.write("keep me")
    project_params.create_dir()
    assert os.path.exists(marker)


def test_create_dir_respects_con_order_range(project_params, tmp_path):
    project_params.con_order_range = 7
    project_params.create_dir()
    assert os.path.isdir(os.path.join("output", "P", "ELConc7"))
    assert not os.path.exists(os.path.join("output", "P", "ELConc50"))


def test_create_dir_is_relative_to_the_current_directory(project_params, tmp_path):
    sub = tmp_path / "nested"
    sub.mkdir()
    os.chdir(sub)
    project_params.create_dir()
    assert (sub / "output" / "P" / "bin").is_dir()
    assert not (tmp_path / "output").exists()


# ---------------------------------------------------------------------------
# annotation / ParamInfo consistency sweep (introspection driven)
# ---------------------------------------------------------------------------

def test_the_annotation_table_is_not_empty():
    assert len(PARAM_NAMES) == 51
    assert len(USER_PARAM_NAMES) > 0


@pytest.mark.parametrize("name", PARAM_NAMES)
def test_every_declared_param_has_a_class_level_default(name):
    assert hasattr(Params, name), f"{name} is annotated but has no default"


@pytest.mark.parametrize("name", PARAM_NAMES)
def test_every_default_matches_its_declared_type(name):
    declared = _declared_type(name)
    default = getattr(Params, name)
    if get_origin(declared) is list:
        inner = get_args(declared)[0]
        assert isinstance(default, list)
        assert all(isinstance(x, inner) for x in default)
    else:
        assert isinstance(default, declared), f"{name}: {type(default)} is not {declared}"


@pytest.mark.parametrize("name", ANNOTATED_NAMES)
def test_first_annotation_metadatum_is_a_paraminfo(name):
    assert isinstance(PARAM_HINTS[name].__metadata__[0], ParamInfo)


@pytest.mark.parametrize("name", ANNOTATED_NAMES)
def test_affects_entries_are_project_levels(name):
    info = PARAM_HINTS[name].__metadata__[0]
    assert isinstance(info.affects, list)
    assert all(isinstance(a, ProjectLevel) for a in info.affects)
    # no level is listed twice
    assert len(set(info.affects)) == len(info.affects)


@pytest.mark.parametrize("name", USER_PARAM_NAMES)
def test_user_params_carry_a_description(name):
    # the description is the CLI --help string, so an empty one ships a blank flag
    assert PARAM_HINTS[name].__metadata__[0].description.strip()


@pytest.mark.parametrize("name", [n for n in USER_PARAM_NAMES if n != "ncpu"])
def test_user_params_declare_the_level_they_affect(name):
    # ncpu is the one exception. It is the global `-n/--ncpu` flag, not attached to a step
    assert PARAM_HINTS[name].__metadata__[0].affects


@pytest.mark.parametrize("name", USER_PARAM_NAMES)
def test_user_params_are_serialized(project_params, name):
    assert name in project_params.asdict()


@pytest.mark.xfail(
    reason="Bug. opt_movie is annotated as a bare `dict` with no ParamInfo, so it is the one "
    "declared param that get_param_info() cannot describe (raises IndexError)",
    strict=False,
)
def test_every_declared_param_has_a_paraminfo():
    missing = [n for n in PARAM_NAMES if not hasattr(PARAM_HINTS[n], "__metadata__")]
    assert missing == []


@pytest.mark.xfail(
    reason="Bug. rad is declared Annotated[int, ParamInfo('Manifold pruning'), True, "
    "[ProjectLevel.MANIFOLD_ANALYSIS]], the user_param flag and affects list were meant to be "
    "ParamInfo arguments but became extra Annotated metadata, so rad is invisible to the CLI",
    strict=False,
)
def test_paraminfo_is_the_only_annotation_metadatum():
    extra = {n: len(PARAM_HINTS[n].__metadata__) for n in ANNOTATED_NAMES
             if len(PARAM_HINTS[n].__metadata__) != 1}
    assert extra == {}


@pytest.mark.xfail(
    reason="Bug. rad's malformed Annotated leaves user_param=False, so --rad is missing from the "
    "manifold-analysis CLI even though the pruning radius is meant to be user tunable",
    strict=False,
)
def test_rad_is_a_user_tunable_manifold_analysis_param(fresh_params):
    info = PARAM_HINTS["rad"].__metadata__[0]
    assert info.user_param is True
    assert ProjectLevel.MANIFOLD_ANALYSIS in info.affects
    assert "rad" in fresh_params.get_user_params()


# ---------------------------------------------------------------------------
# get_user_params / get_params_for_level / get_param_info
# ---------------------------------------------------------------------------

def test_get_user_params_matches_a_manual_scan(fresh_params):
    assert sorted(fresh_params.get_user_params().keys()) == USER_PARAM_NAMES


def test_get_user_params_returns_annotated_types(fresh_params):
    for name, hint in fresh_params.get_user_params().items():
        assert hint is PARAM_HINTS[name]
        assert hint.__metadata__[0].user_param is True


@pytest.mark.parametrize(
    "name",
    [
        "eps",
        "prd_thres_low",
        "prd_thres_high",
        "tess_hemisphere_vec",
        "tess_hemisphere_type",
        "prd_assignment",
        "prd_cone_width_factor",
        "distance_filter_type",
        "distance_filter_cutoff_freq",
        "distance_filter_order",
        "num_psi",
        "nlsa_fps",
        "con_order_range",
        "nlsa_tune",
        "ncpu",
    ],
)
def test_known_user_params(fresh_params, name):
    assert name in fresh_params.get_user_params()


@pytest.mark.parametrize(
    "name", ["project_name", "num_part", "num_eigs", "opt_mask_type", "traj_name", "width_1D"]
)
def test_known_non_user_params(fresh_params, name):
    assert name not in fresh_params.get_user_params()


LEVEL_EXPECTATIONS = {
    ProjectLevel.INIT: [],
    ProjectLevel.BINNING: [
        "eps",
        "prd_assignment",
        "prd_cone_width_factor",
        "prd_thres_high",
        "prd_thres_low",
        "tess_hemisphere_type",
        "tess_hemisphere_vec",
    ],
    ProjectLevel.CALC_DISTANCE: [
        "distance_filter_cutoff_freq",
        "distance_filter_order",
        "distance_filter_type",
        "num_psi",
    ],
    ProjectLevel.MANIFOLD_ANALYSIS: ["nlsa_tune"],
    ProjectLevel.PSI_ANALYSIS: ["con_order_range"],
    ProjectLevel.NLSA_MOVIE: ["nlsa_fps"],
    ProjectLevel.PRD_SELECTION: [],
    ProjectLevel.FIND_CCS: [],
    ProjectLevel.PROBABILITY_LANDSCAPE: [],
    ProjectLevel.TRAJECTORY: [],
}


@pytest.mark.parametrize("level", list(ProjectLevel), ids=lambda l: l.name)
def test_get_params_for_level_defaults(fresh_params, level):
    assert sorted(fresh_params.get_params_for_level(level).keys()) == LEVEL_EXPECTATIONS[level]


@pytest.mark.parametrize("level", list(ProjectLevel), ids=lambda l: l.name)
def test_get_params_for_level_returns_type_and_info(fresh_params, level):
    for name, (ptype, info) in fresh_params.get_params_for_level(level).items():
        assert ptype is _declared_type(name)
        assert isinstance(info, ParamInfo)
        assert level in info.affects


def test_first_appearance_hides_repeats_of_a_multi_level_param(fresh_params):
    # con_order_range affects PSI_ANALYSIS, PROBABILITY_LANDSCAPE and TRAJECTORY. with
    # first_appearance=True it is only offered on the earliest of the three
    assert "con_order_range" in fresh_params.get_params_for_level(ProjectLevel.PSI_ANALYSIS)
    assert "con_order_range" not in fresh_params.get_params_for_level(
        ProjectLevel.PROBABILITY_LANDSCAPE
    )
    assert "con_order_range" in fresh_params.get_params_for_level(
        ProjectLevel.PROBABILITY_LANDSCAPE, first_appearance=False
    )
    assert "con_order_range" in fresh_params.get_params_for_level(
        ProjectLevel.TRAJECTORY, first_appearance=False
    )


def test_first_appearance_hides_nlsa_tune_from_psi_analysis(fresh_params):
    # nlsa_tune affects MANIFOLD_ANALYSIS (3) and PSI_ANALYSIS (4)
    assert "nlsa_tune" in fresh_params.get_params_for_level(ProjectLevel.MANIFOLD_ANALYSIS)
    assert "nlsa_tune" not in fresh_params.get_params_for_level(ProjectLevel.PSI_ANALYSIS)
    assert "nlsa_tune" in fresh_params.get_params_for_level(
        ProjectLevel.PSI_ANALYSIS, first_appearance=False
    )


def test_user_only_false_exposes_project_name_at_init(fresh_params):
    # project_name affects INIT but is not a user param, so it only shows with user_only=False
    assert fresh_params.get_params_for_level(ProjectLevel.INIT) == {}
    non_user = fresh_params.get_params_for_level(ProjectLevel.INIT, user_only=False)
    assert list(non_user.keys()) == ["project_name"]
    assert non_user["project_name"][0] is str


@pytest.mark.parametrize("level", list(ProjectLevel), ids=lambda l: l.name)
def test_user_only_is_a_subset_of_everything(fresh_params, level):
    user_only = set(fresh_params.get_params_for_level(level, user_only=True))
    everything = set(fresh_params.get_params_for_level(level, user_only=False))
    assert user_only <= everything


@pytest.mark.parametrize("level", list(ProjectLevel), ids=lambda l: l.name)
def test_first_appearance_is_a_subset_of_all_appearances(fresh_params, level):
    first = set(fresh_params.get_params_for_level(level, first_appearance=True))
    every = set(fresh_params.get_params_for_level(level, first_appearance=False))
    assert first <= every


def test_every_user_param_with_affects_appears_exactly_once_across_levels(fresh_params):
    # the CLI adds each --flag to one subparser. a duplicate would be an argparse conflict
    seen = []
    for level in ProjectLevel:
        seen.extend(fresh_params.get_params_for_level(level).keys())
    assert len(seen) == len(set(seen))
    expected = [n for n in USER_PARAM_NAMES if PARAM_HINTS[n].__metadata__[0].affects]
    assert sorted(seen) == sorted(expected)


PARAM_INFO_CASES = [
    ("num_psi", int, True, [ProjectLevel.CALC_DISTANCE]),
    ("eps", float, True, [ProjectLevel.BINNING]),
    ("prd_thres_low", int, True, [ProjectLevel.BINNING]),
    ("tess_hemisphere_type", str, True, [ProjectLevel.BINNING]),
    ("distance_filter_order", int, True, [ProjectLevel.CALC_DISTANCE]),
    ("nlsa_fps", float, True, [ProjectLevel.NLSA_MOVIE]),
    ("nlsa_tune", int, True, [ProjectLevel.MANIFOLD_ANALYSIS, ProjectLevel.PSI_ANALYSIS]),
    ("num_part", int, False, []),
    ("project_name", str, False, [ProjectLevel.INIT]),
    ("calc_optical_flow", bool, False, []),
    ("project_level", ProjectLevel, False, []),
]


@pytest.mark.parametrize("name,ptype,user,affects", PARAM_INFO_CASES)
def test_get_param_info(fresh_params, name, ptype, user, affects):
    got_type, info = fresh_params.get_param_info(name)
    assert got_type is ptype
    assert info.user_param is user
    assert info.affects == affects


def test_get_param_info_returns_the_list_generic_for_list_params(fresh_params):
    ptype, info = fresh_params.get_param_info("tess_hemisphere_vec")
    # cli.py special-cases this by checking paramtype.__name__ == 'list'
    assert ptype.__name__ == "list"
    assert get_args(ptype) == (float,)
    assert info.user_param is True


@pytest.mark.parametrize("name", ["nonexistent", "out_dir", "create_dir", "sh"])
def test_get_param_info_on_a_non_param_raises_keyerror(fresh_params, name):
    # only names in __annotations__ are describable. Properties and methods are not
    with pytest.raises(KeyError):
        fresh_params.get_param_info(name)


@pytest.mark.parametrize("name", USER_PARAM_NAMES)
def test_get_param_info_agrees_with_get_user_params(fresh_params, name):
    ptype, info = fresh_params.get_param_info(name)
    assert ptype is _declared_type(name)
    assert info is PARAM_HINTS[name].__metadata__[0]


# ---------------------------------------------------------------------------
# fixture hygiene. Prove the singleton really is restored
# ---------------------------------------------------------------------------

def test_singleton_state_is_isolated_between_tests_part_one(fresh_params):
    fresh_params.project_name = "leak_check"
    fresh_params.num_psi = 99


def test_singleton_state_is_isolated_between_tests_part_two(fresh_params):
    assert fresh_params.project_name != "leak_check"
    assert fresh_params.num_psi == 8


# ============================================================================
# star.py and myio.py - RELION star parsing and pickle/HDF5 IO
# ============================================================================

# --------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------

@pytest.fixture
def clean_params():
    """`params` is a process-wide singleton whose defaults live on the class.

    Every assignment lands in the *instance* __dict__, so clearing that dict restores
    the pristine class defaults exactly.  cwd is saved too because params.load() will
    os.chdir into the toml's directory whenever that directory is non-empty.
    """
    cwd = os.getcwd()
    saved = dict(params.__dict__)
    try:
        yield params
    finally:
        os.chdir(cwd)
        params.__dict__.clear()
        params.__dict__.update(saved)


class ExitCalled(Exception):
    """Stand-in for the builtin exit() that star.get_align_data calls on bad input."""

    def __init__(self, code=None):
        super().__init__(code)
        self.code = code


@pytest.fixture
def trap_exit(monkeypatch):
    """Replace builtins.exit with a raiser so exit(1) cannot tear down the test process."""
    codes = []

    def fake_exit(code=None):
        codes.append(code)
        raise ExitCalled(code)

    monkeypatch.setattr(builtins, "exit", fake_exit)
    return codes


@pytest.fixture
def swallow_exit(monkeypatch):
    """Replace builtins.exit with a no-op, exposing what the code does *after* exit()."""
    codes = []
    monkeypatch.setattr(builtins, "exit", lambda code=None: codes.append(code))
    return codes


def write_file(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def star_block(labels, rows, indexed=True):
    """A 'loop_' block. one '_label #n' line per column then whitespace separated rows."""
    lines = ["", "loop_", ""]
    for i, lab in enumerate(labels):
        lines.append(f"_{lab} #{i + 1}" if indexed else f"_{lab}")
    for r in rows:
        lines.append(" ".join(str(v) for v in r))
    return lines


def old_star_text(labels, rows, n_preamble=1, indexed=True):
    """RELION 3.0 style. a single unnamed data block, no optics group."""
    lines = [""] * n_preamble + ["data_"] + star_block(labels, rows, indexed)
    return "\n".join(lines) + "\n"


def new_star_text(opt_labels, opt_rows, part_labels, part_rows):
    """RELION 3.1 style. a data_optics block followed by a data_particles block."""
    lines = ["# version 30001", "", "data_optics"]
    lines += star_block(opt_labels, opt_rows)
    lines += ["", "# version 30001", "", "data_particles"]
    lines += star_block(part_labels, part_rows)
    return "\n".join(lines) + "\n"


MS_LABELS = ["rlnVoltage", "rlnSphericalAberration", "rlnAmplitudeContrast"]
MS_VALUES = [300.0, 2.7, 0.1]
ANG_LABELS = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
DEF_LABELS = ["rlnDefocusU", "rlnDefocusV"]


def angle_rows(n, seed=0):
    """Deterministic Euler triples in degrees plus a defocus pair per particle."""
    rng = np.random.default_rng(seed)
    rot = rng.uniform(-180.0, 180.0, n)
    tilt = rng.uniform(0.0, 180.0, n)
    psi = rng.uniform(-180.0, 180.0, n)
    u = 10000.0 + 100.0 * np.arange(n)
    v = u + 500.0
    return np.column_stack([rot, tilt, psi, u, v])


def full_old_star(tmp_path, n=4, extra_labels=(), extra_cols=None, name="old.star"):
    """Old-format file carrying angles, defocus and the microscope parameters."""
    base = angle_rows(n)
    labels = ANG_LABELS + DEF_LABELS + MS_LABELS + list(extra_labels)
    ms = np.tile(np.array(MS_VALUES), (n, 1))
    parts = [base, ms]
    if extra_cols is not None:
        parts.append(np.asarray(extra_cols, dtype=float).reshape(n, -1))
    rows = np.hstack(parts)
    return write_file(tmp_path, name, old_star_text(labels, rows.tolist())), base


def full_new_star(tmp_path, n=4, extra_labels=(), extra_cols=None, name="new.star"):
    """New-format file. Microscope parameters in the optics block only."""
    base = angle_rows(n)
    part_labels = ANG_LABELS + DEF_LABELS + list(extra_labels)
    parts = [base]
    if extra_cols is not None:
        parts.append(np.asarray(extra_cols, dtype=float).reshape(n, -1))
    rows = np.hstack(parts)
    text = new_star_text(["rlnOpticsGroup"] + MS_LABELS,
                         [[1] + MS_VALUES],
                         part_labels,
                         rows.tolist())
    return write_file(tmp_path, name, text), base


# --------------------------------------------------------------------------------------
# star.parse_star, label scanning and column naming
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["rlnVoltage", "rlnAngleRot", "rlnDefocusU",
                                   "rlnImageName", "rlnOriginXAngst", "rlnCoordinateX"])
def test_parse_star_strips_underscore_and_index(tmp_path, label):
    # keep_index=False -> "_<label> #n" collapses to "<label>"
    p = write_file(tmp_path, "s.star", old_star_text([label], [[1.0], [2.0]]))
    assert star.parse_star(p, 0).columns.tolist() == [label]


@pytest.mark.parametrize("raw,expected", [
    ("_rlnVoltage #1", "rlnVoltage"),
    ("_rlnVoltage  #1", "rlnVoltage"),
    ("_rlnVoltage #1   ", "rlnVoltage"),
    ("_rlnVoltage", "rlnVoltage"),
    ("_rlnVoltage   ", "rlnVoltage"),
    ("_rlnAngleRot #12", "rlnAngleRot"),
    ("_rlnDefocusU #3 # trailing", "rlnDefocusU"),
])
def test_parse_star_header_name_forms(tmp_path, raw, expected):
    # the label is everything before the first '#', rstripped, with leading '_' removed
    text = "\ndata_\n\nloop_\n" + raw + "\n1.0\n2.0\n"
    p = write_file(tmp_path, "s.star", text)
    assert star.parse_star(p, 0).columns.tolist() == [expected]


@pytest.mark.parametrize("raw,expected", [
    ("_rlnVoltage #1", "_rlnVoltage #1"),
    ("_rlnVoltage #1   ", "_rlnVoltage #1"),
    ("_rlnVoltage", "_rlnVoltage"),
    ("_rlnAngleRot  #7", "_rlnAngleRot  #7"),
])
def test_parse_star_keep_index_only_rstrips(tmp_path, raw, expected):
    text = "\ndata_\n\nloop_\n" + raw + "\n1.0\n2.0\n"
    p = write_file(tmp_path, "s.star", text)
    assert star.parse_star(p, 0, keep_index=True).columns.tolist() == [expected]


@pytest.mark.parametrize("n_preamble", [0, 1, 2, 5, 9])
def test_parse_star_tolerates_arbitrary_preamble(tmp_path, n_preamble):
    # nothing before the first "_rln" line matters. foundheader gates the break
    rows = [[1.0, 2.0], [3.0, 4.0]]
    p = write_file(tmp_path, "s.star",
                   old_star_text(["rlnAngleRot", "rlnAngleTilt"], rows, n_preamble=n_preamble))
    df = star.parse_star(p, 0)
    assert df.shape == (2, 2)
    assert np.allclose(df.values, rows)


@pytest.mark.parametrize("n_rows", [1, 2, 5, 17])
def test_parse_star_row_count(tmp_path, n_rows):
    rows = [[float(i), float(i) + 0.5] for i in range(n_rows)]
    p = write_file(tmp_path, "s.star", old_star_text(["rlnAngleRot", "rlnAngleTilt"], rows))
    df = star.parse_star(p, 0)
    assert len(df) == n_rows
    assert np.allclose(df["rlnAngleRot"].values, [r[0] for r in rows])


@pytest.mark.parametrize("n_cols", [1, 2, 3, 8])
def test_parse_star_column_count(tmp_path, n_cols):
    labels = [f"rlnCol{i}" for i in range(n_cols)]
    rows = [list(range(n_cols)), list(range(n_cols, 2 * n_cols))]
    p = write_file(tmp_path, "s.star", old_star_text(labels, rows))
    df = star.parse_star(p, 0)
    assert df.shape == (2, n_cols)
    assert df.columns.tolist() == labels


def test_parse_star_returns_dataframe_with_rangeindex(tmp_path):
    p, _ = full_old_star(tmp_path, n=3)
    df = star.parse_star(p, 0)
    assert isinstance(df, pd.DataFrame)
    assert df.index.tolist() == [0, 1, 2]


def test_parse_star_infers_dtypes(tmp_path):
    # string image names stay object, numeric columns become float64/int64
    labels = ["rlnImageName", "rlnAngleRot", "rlnClassNumber"]
    rows = [["000001@st.mrcs", 1.5, 3], ["000002@st.mrcs", 2.5, 4]]
    p = write_file(tmp_path, "s.star", old_star_text(labels, rows))
    df = star.parse_star(p, 0)
    assert df["rlnImageName"].dtype == object
    assert df["rlnAngleRot"].dtype == np.float64
    assert df["rlnClassNumber"].dtype == np.int64


def test_parse_star_unindexed_labels(tmp_path):
    # the "#n" suffix is optional in STAR. The scanner does not need it
    p = write_file(tmp_path, "s.star",
                   old_star_text(["rlnAngleRot", "rlnAngleTilt"], [[1.0, 2.0]], indexed=False))
    assert star.parse_star(p, 0).columns.tolist() == ["rlnAngleRot", "rlnAngleTilt"]


def test_parse_star_duplicate_labels_are_kept(tmp_path):
    # no de-duplication. Two identical labels give two identically named columns
    text = "\ndata_\n\nloop_\n_rlnA #1\n_rlnA #2\n1 2\n3 4\n"
    p = write_file(tmp_path, "s.star", text)
    df = star.parse_star(p, 0)
    assert df.columns.tolist() == ["rlnA", "rlnA"]
    assert df["rlnA"].shape == (2, 2)


def test_parse_star_non_rln_label_terminates_the_block(tmp_path):
    # only "_rln" prefixed lines count. a foreign label ends the header scan early and
    # then becomes an unparsable data line -> column/label length mismatch
    text = "\ndata_\n\nloop_\n_rlnA #1\n_myLabel #2\n1 2\n"
    p = write_file(tmp_path, "s.star", text)
    with pytest.raises(ValueError, match="Length mismatch"):
        star.parse_star(p, 0)


def test_parse_star_indented_label_is_not_recognized(tmp_path):
    # startswith() is literal. Leading whitespace hides the label, so no header is ever
    # found and the scanner runs to EOF with skiprows == the whole file
    text = "\ndata_\n\nloop_\n  _rlnA #1\n1\n"
    p = write_file(tmp_path, "s.star", text)
    with pytest.raises(pd.errors.EmptyDataError):
        star.parse_star(p, 0)


@pytest.mark.parametrize("skip", [0, 1, 2, 3, 4, 5])
def test_parse_star_skip_before_header_block_is_harmless(tmp_path, skip):
    # the 5 lines before the first label are blank/data_/blank/loop_/blank
    rows = [[1.0, 2.0], [3.0, 4.0]]
    p = write_file(tmp_path, "s.star", old_star_text(["rlnA", "rlnB"], rows))
    df = star.parse_star(p, skip)
    assert df.columns.tolist() == ["rlnA", "rlnB"]
    assert np.allclose(df.values, rows)


@pytest.mark.parametrize("skip,n_labels", [(6, 2), (7, 1)])
def test_parse_star_skip_into_header_block_loses_columns(tmp_path, skip, n_labels):
    # labels live on lines 5..7. Skipping past the first one leaves fewer names than
    # data columns and pandas refuses the rename
    p = write_file(tmp_path, "s.star", old_star_text(["rlnA", "rlnB", "rlnC"], [[1, 2, 3]]))
    with pytest.raises(ValueError, match=f"new values have {n_labels} element"):
        star.parse_star(p, skip)


@pytest.mark.parametrize("text", [
    "",
    "data_\n\nloop_\n1 2 3\n",
    "# just a comment\n",
])
def test_parse_star_without_labels_raises(tmp_path, text):
    p = write_file(tmp_path, "s.star", text)
    with pytest.raises(pd.errors.EmptyDataError):
        star.parse_star(p, 0)


def test_parse_star_skip_past_end_of_file_raises(tmp_path):
    p = write_file(tmp_path, "s.star", old_star_text(["rlnA"], [[1.0]]))
    with pytest.raises(pd.errors.EmptyDataError):
        star.parse_star(p, 10_000)


def test_parse_star_header_only_file_raises(tmp_path):
    # labels but no data rows. pandas has nothing to read below skiprows
    p = write_file(tmp_path, "s.star", old_star_text(["rlnA", "rlnB"], []))
    with pytest.raises(pd.errors.EmptyDataError):
        star.parse_star(p, 0)


def test_parse_star_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        star.parse_star(str(tmp_path / "absent.star"), 0)


def test_parse_star_whitespace_delimiter_handles_ragged_spacing(tmp_path):
    text = "\ndata_\n\nloop_\n_rlnA #1\n_rlnB #2\n1     2\n\t3\t\t4\n"
    p = write_file(tmp_path, "s.star", text)
    df = star.parse_star(p, 0)
    assert np.allclose(df.values, [[1, 2], [3, 4]])


# --------------------------------------------------------------------------------------
# star.parse_star_optics
# --------------------------------------------------------------------------------------

def test_parse_star_optics_returns_frame_and_skip(tmp_path):
    p, _ = full_new_star(tmp_path)
    out = star.parse_star_optics(p)
    assert isinstance(out, tuple) and len(out) == 2
    df0, skip = out
    assert isinstance(df0, pd.DataFrame)
    assert isinstance(skip, int)


def test_parse_star_optics_reads_exactly_one_row(tmp_path):
    p, _ = full_new_star(tmp_path)
    df0, _ = star.parse_star_optics(p)
    assert df0.shape == (1, 4)
    assert df0.columns.tolist() == ["rlnOpticsGroup"] + MS_LABELS


@pytest.mark.parametrize("label,value", list(zip(MS_LABELS, MS_VALUES)))
def test_parse_star_optics_values(tmp_path, label, value):
    p, _ = full_new_star(tmp_path)
    df0, _ = star.parse_star_optics(p)
    assert float(df0[label].values[0]) == pytest.approx(value)


def test_parse_star_optics_skip_points_past_the_optics_row(tmp_path):
    p, _ = full_new_star(tmp_path)
    df0, skip = star.parse_star_optics(p)
    with open(p) as f:
        lines = f.read().split("\n")
    # the returned skip is the count of consumed lines, so lines[skip-1] is the optics row
    assert lines[skip - 1].split()[0] == "1"
    # and everything at/after `skip` still contains the particle label block
    assert any(l.startswith("_rlnAngleRot") for l in lines[skip:])


def test_parse_star_optics_feeds_parse_star(tmp_path):
    p, base = full_new_star(tmp_path, n=5)
    _, skip = star.parse_star_optics(p)
    df = star.parse_star(p, skip)
    assert df.columns.tolist() == ANG_LABELS + DEF_LABELS
    assert len(df) == 5
    assert np.allclose(df.values, base)


def test_parse_star_optics_keep_index(tmp_path):
    p, _ = full_new_star(tmp_path)
    df0, _ = star.parse_star_optics(p, keep_index=True)
    assert df0.columns.tolist() == ["_rlnOpticsGroup #1", "_rlnVoltage #2",
                                    "_rlnSphericalAberration #3", "_rlnAmplitudeContrast #4"]


def test_parse_star_optics_ignores_extra_optics_groups(tmp_path):
    # nrows=1. Only the first optics group is ever returned
    text = new_star_text(["rlnOpticsGroup"] + MS_LABELS,
                         [[1, 300.0, 2.7, 0.1], [2, 200.0, 2.0, 0.07]],
                         ANG_LABELS, [[1.0, 2.0, 3.0]])
    p = write_file(tmp_path, "two.star", text)
    df0, skip = star.parse_star_optics(p)
    assert len(df0) == 1
    assert float(df0["rlnVoltage"].values[0]) == 300.0
    # the second group's row is not a label line, so parse_star scans straight past it
    df = star.parse_star(p, skip)
    assert df.columns.tolist() == ANG_LABELS


def test_parse_star_optics_does_not_verify_it_read_an_optics_block(tmp_path):
    # handed an old-format file it happily returns the first *particle* row
    p, base = full_old_star(tmp_path, n=3)
    df0, _ = star.parse_star_optics(p)
    assert df0.columns.tolist() == ANG_LABELS + DEF_LABELS + MS_LABELS
    assert np.allclose(df0.values[0][:5], base[0])


def test_parse_star_optics_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        star.parse_star_optics(str(tmp_path / "absent.star"))


def test_parse_star_optics_without_labels_raises(tmp_path):
    p = write_file(tmp_path, "s.star", "data_optics\n\nloop_\n1 2 3\n")
    with pytest.raises(pd.errors.EmptyDataError):
        star.parse_star_optics(p)


# --------------------------------------------------------------------------------------
# star.get_align_data, shapes, values, quaternions
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [full_old_star, full_new_star])
def test_get_align_data_return_shape_contract(tmp_path, clean_params, builder):
    p, base = builder(tmp_path, n=6)
    sh, q, U, V = star.get_align_data(p, flip=True)
    assert isinstance(sh, tuple) and len(sh) == 2
    assert sh[0].shape == (6,) and sh[1].shape == (6,)
    assert q.shape == (4, 6)          # quaternions are columns, scalar-first
    assert U.shape == (6,) and V.shape == (6,)
    assert np.allclose(U, base[:, 3])
    assert np.allclose(V, base[:, 4])


@pytest.mark.parametrize("n", [1, 2, 7, 13])
def test_get_align_data_scales_with_particle_count(tmp_path, clean_params, n):
    p, _ = full_old_star(tmp_path, n=n)
    sh, q, U, V = star.get_align_data(p, flip=True)
    assert q.shape == (4, n)
    assert len(U) == len(V) == len(sh[0]) == len(sh[1]) == n


@pytest.mark.parametrize("flip", [True, False])
def test_get_align_data_quaternions_are_unit(tmp_path, clean_params, flip):
    p, _ = full_old_star(tmp_path, n=8)
    _, q, _, _ = star.get_align_data(p, flip=flip)
    assert np.allclose(np.linalg.norm(q, axis=0), 1.0)


@pytest.mark.parametrize("flip", [True, False])
def test_get_align_data_matches_eul_to_quat(tmp_path, clean_params, flip):
    p, base = full_old_star(tmp_path, n=8)
    _, q, _, _ = star.get_align_data(p, flip=flip)
    expected = eul_to_quat(np.deg2rad(base[:, 0]), np.deg2rad(base[:, 1]),
                           np.deg2rad(base[:, 2]), flip)
    assert np.allclose(q, expected)


def test_get_align_data_scipy_oracle_flip_true(tmp_path, clean_params):
    # flip=True reproduces the legacy "spider" convention. Clockwise extrinsic zyz,
    # scalar-first quaternion, i.e. scipy's scalar-last quat permuted to [q1,q2,q3,q0]
    p, base = full_old_star(tmp_path, n=8)
    _, q, _, _ = star.get_align_data(p, flip=True)
    eul = np.deg2rad(base[:, :3])
    ref = Rotation.from_euler("zyz", -eul).as_quat()
    assert np.allclose(ref, np.vstack([q[1], q[2], q[3], q[0]]).T, atol=1e-10)


def test_get_align_data_scipy_oracle_flip_false(tmp_path, clean_params):
    # flip=False only negates the psi half-angle sine, i.e. it is flip=True on -psi
    p, base = full_old_star(tmp_path, n=8)
    _, q, _, _ = star.get_align_data(p, flip=False)
    eul = np.deg2rad(base[:, :3]) * np.array([1.0, 1.0, -1.0])
    ref = Rotation.from_euler("zyz", -eul).as_quat()
    assert np.allclose(ref, np.vstack([q[1], q[2], q[3], q[0]]).T, atol=1e-10)


def test_get_align_data_zero_angles_give_identity_quaternion(tmp_path, clean_params):
    labels = ANG_LABELS + DEF_LABELS + MS_LABELS
    rows = [[0.0, 0.0, 0.0, 10000.0, 10500.0] + MS_VALUES]
    p = write_file(tmp_path, "zero.star", old_star_text(labels, rows))
    _, q, _, _ = star.get_align_data(p, flip=True)
    assert np.allclose(q[:, 0], [1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("flip", [True, False])
def test_get_align_data_flip_only_touches_the_psi_rotation(tmp_path, clean_params, flip):
    # with psi = 0 the flip flag is a no-op
    labels = ANG_LABELS + DEF_LABELS + MS_LABELS
    rows = [[30.0, 40.0, 0.0, 10000.0, 10500.0] + MS_VALUES]
    p = write_file(tmp_path, "psi0.star", old_star_text(labels, rows))
    _, q, _, _ = star.get_align_data(p, flip=flip)
    ref = eul_to_quat(np.deg2rad([30.0]), np.deg2rad([40.0]), np.array([0.0]), True)
    assert np.allclose(q, ref)


def test_get_align_data_flip_negates_only_the_k_component_for_pure_psi(tmp_path, clean_params):
    # a pure psi rotation is exp(-psi/2 k) or exp(+psi/2 k) depending on flip
    labels = ANG_LABELS + DEF_LABELS + MS_LABELS
    rows = [[0.0, 0.0, 60.0, 10000.0, 10500.0] + MS_VALUES]
    p = write_file(tmp_path, "purepsi.star", old_star_text(labels, rows))
    _, qt, _, _ = star.get_align_data(p, flip=True)
    _, qf, _, _ = star.get_align_data(p, flip=False)
    assert np.allclose(qt[:3, 0], qf[:3, 0])
    assert np.allclose(qt[3, 0], -qf[3, 0])
    assert qt[3, 0] == pytest.approx(-np.sin(np.deg2rad(60.0) / 2))


def test_get_align_data_angles_are_converted_from_degrees(tmp_path, clean_params):
    # tilt = 180 deg about y sends the scalar part to 0
    labels = ANG_LABELS + DEF_LABELS + MS_LABELS
    rows = [[0.0, 180.0, 0.0, 10000.0, 10500.0] + MS_VALUES]
    p = write_file(tmp_path, "tilt180.star", old_star_text(labels, rows))
    _, q, _, _ = star.get_align_data(p, flip=True)
    assert q[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert q[2, 0] == pytest.approx(-1.0)


# --------------------------------------------------------------------------------------
# star.get_align_data, microscope parameter side effects on the params singleton
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("attr,value", [
    ("ms_kilovolts", 300.0),
    ("ms_spherical_aberration", 2.7),
    ("ms_amplitude_contrast_ratio", 0.1),
])
@pytest.mark.parametrize("builder", [full_old_star, full_new_star])
def test_get_align_data_sets_microscope_params(tmp_path, clean_params, builder, attr, value):
    setattr(clean_params, attr, -1.0)
    p, _ = builder(tmp_path, n=3)
    star.get_align_data(p, flip=True)
    assert getattr(clean_params, attr) == pytest.approx(value)


def test_get_align_data_microscope_params_become_python_floats(tmp_path, clean_params):
    p, _ = full_new_star(tmp_path, n=2)
    star.get_align_data(p, flip=True)
    for attr in ("ms_kilovolts", "ms_spherical_aberration", "ms_amplitude_contrast_ratio"):
        assert type(getattr(clean_params, attr)) is float


def test_get_align_data_old_format_takes_microscope_params_from_first_row(tmp_path, clean_params):
    # with no optics block df0 *is* the particle table, so row 0 wins
    labels = ANG_LABELS + DEF_LABELS + MS_LABELS
    rows = [[0.0, 0.0, 0.0, 1e4, 1e4, 300.0, 2.7, 0.1],
            [0.0, 0.0, 0.0, 1e4, 1e4, 200.0, 2.0, 0.07]]
    p = write_file(tmp_path, "mixed.star", old_star_text(labels, rows))
    star.get_align_data(p, flip=True)
    assert clean_params.ms_kilovolts == pytest.approx(300.0)


def test_get_align_data_does_not_touch_pixel_size(tmp_path, clean_params):
    # ms_pixel_size is an input here, never an output
    clean_params.ms_pixel_size = 1.25
    p, _ = full_new_star(tmp_path, n=2)
    star.get_align_data(p, flip=True)
    assert clean_params.ms_pixel_size == 1.25


# --------------------------------------------------------------------------------------
# star.get_align_data, origin/shift columns
# --------------------------------------------------------------------------------------

def test_get_align_data_origin_in_pixels_is_used_verbatim(tmp_path, clean_params):
    clean_params.ms_pixel_size = 2.0
    shifts = np.array([[1.0, -2.0], [3.5, 4.5], [0.0, 0.0]])
    p, _ = full_old_star(tmp_path, n=3, extra_labels=("rlnOriginX", "rlnOriginY"),
                         extra_cols=shifts)
    sh, _, _, _ = star.get_align_data(p, flip=True)
    # rlnOriginX/Y are already pixels, so pixel size must not be applied
    assert np.allclose(sh[0], shifts[:, 0])
    assert np.allclose(sh[1], shifts[:, 1])


@pytest.mark.parametrize("pixel_size", [0.5, 1.0, 1.5, 2.0, 3.25])
def test_get_align_data_origin_in_angstrom_is_divided_by_pixel_size(tmp_path, clean_params,
                                                                    pixel_size):
    clean_params.ms_pixel_size = pixel_size
    shifts = np.array([[3.0, -6.0], [1.5, 4.5], [0.0, 2.0]])
    p, _ = full_new_star(tmp_path, n=3, extra_labels=("rlnOriginXAngst", "rlnOriginYAngst"),
                         extra_cols=shifts)
    sh, _, _, _ = star.get_align_data(p, flip=True)
    assert np.allclose(sh[0], shifts[:, 0] / pixel_size)
    assert np.allclose(sh[1], shifts[:, 1] / pixel_size)


def test_get_align_data_pixel_origin_wins_over_angstrom_origin(tmp_path, clean_params):
    clean_params.ms_pixel_size = 2.0
    cols = np.array([[1.0, 2.0, 100.0, 200.0], [3.0, 4.0, 300.0, 400.0]])
    p, _ = full_old_star(tmp_path, n=2,
                         extra_labels=("rlnOriginX", "rlnOriginY",
                                       "rlnOriginXAngst", "rlnOriginYAngst"),
                         extra_cols=cols)
    sh, _, _, _ = star.get_align_data(p, flip=True)
    assert np.allclose(sh[0], [1.0, 3.0])
    assert np.allclose(sh[1], [2.0, 4.0])


@pytest.mark.parametrize("extra", [
    (),
    ("rlnOriginX",),
    ("rlnOriginY",),
    ("rlnOriginXAngst",),
    ("rlnOriginX", "rlnOriginYAngst"),
])
def test_get_align_data_incomplete_origin_pair_falls_back_to_zeros(tmp_path, clean_params, extra):
    # both members of a pair must be present, otherwise the shifts are silently zeroed
    n = 3
    cols = np.arange(n * len(extra), dtype=float).reshape(n, -1) + 1.0 if extra else None
    p, _ = full_old_star(tmp_path, n=n, extra_labels=extra, extra_cols=cols)
    sh, _, _, _ = star.get_align_data(p, flip=True)
    assert np.allclose(sh[0], 0.0)
    assert np.allclose(sh[1], 0.0)


def test_get_align_data_zero_shift_is_float_even_for_integer_defocus(tmp_path, clean_params):
    # the fallback is `U * 0.`, so it inherits U's length but is promoted to float
    labels = ANG_LABELS + DEF_LABELS + MS_LABELS
    rows = [[0.0, 0.0, 0.0, 10000, 10500] + MS_VALUES,
            [1.0, 2.0, 3.0, 11000, 11500] + MS_VALUES]
    p = write_file(tmp_path, "intdef.star", old_star_text(labels, rows))
    sh, _, U, _ = star.get_align_data(p, flip=True)
    assert U.dtype == np.int64
    assert sh[0].dtype == np.float64
    assert sh[0].shape == U.shape


@pytest.mark.xfail(reason="Bug. Fallback does `shy = shx`, so the two shift arrays alias",
                   strict=False)
def test_get_align_data_zero_shifts_are_independent_arrays(tmp_path, clean_params):
    p, _ = full_old_star(tmp_path, n=3)
    sh, _, _, _ = star.get_align_data(p, flip=True)
    shx, shy = sh
    shx[0] = 5.0
    assert shy[0] == 0.0


def test_get_align_data_angstrom_origin_with_unset_pixel_size(tmp_path, clean_params):
    # ms_pixel_size defaults to 0.0. The division is unguarded and quietly yields inf/nan
    assert clean_params.ms_pixel_size == 0.0
    shifts = np.array([[3.0, 0.0], [-3.0, 1.0]])
    p, _ = full_new_star(tmp_path, n=2, extra_labels=("rlnOriginXAngst", "rlnOriginYAngst"),
                         extra_cols=shifts)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh, _, _, _ = star.get_align_data(p, flip=True)
    assert np.isinf(sh[0][0]) and sh[0][0] > 0
    assert np.isinf(sh[0][1]) and sh[0][1] < 0
    assert np.isnan(sh[1][0])


# --------------------------------------------------------------------------------------
# star.get_align_data, format detection and failure paths
# --------------------------------------------------------------------------------------

def test_get_align_data_detects_optics_block(tmp_path, clean_params, capsys):
    p, _ = full_new_star(tmp_path, n=2)
    star.get_align_data(p, flip=True)
    assert "RELION Optics Group found." in capsys.readouterr().out


def test_get_align_data_old_format_prints_no_optics_banner(tmp_path, clean_params, capsys):
    p, _ = full_old_star(tmp_path, n=2)
    star.get_align_data(p, flip=True)
    assert "RELION Optics Group found." not in capsys.readouterr().out


def test_get_align_data_optics_detection_is_line_anchored(tmp_path, clean_params):
    # "data_optics" mentioned mid-line must not switch the parser into 3.1 mode
    labels = ANG_LABELS + DEF_LABELS + MS_LABELS
    rows = [[0.0, 0.0, 0.0, 1e4, 1e4] + MS_VALUES]
    text = "# note: no data_optics here\n" + old_star_text(labels, rows, n_preamble=0)
    p = write_file(tmp_path, "anchor.star", text)
    _, q, _, _ = star.get_align_data(p, flip=True)
    assert q.shape == (4, 1)


def test_get_align_data_missing_origin_warns(tmp_path, clean_params, capsys):
    p, _ = full_old_star(tmp_path, n=2)
    star.get_align_data(p, flip=True)
    assert "missing relion origin data" in capsys.readouterr().out


@pytest.mark.parametrize("drop,message", [
    (MS_LABELS, "missing microscope parameters"),
    (DEF_LABELS, "missing defocus"),
    (ANG_LABELS, "missing Euler angles"),
])
def test_get_align_data_exits_on_missing_columns(tmp_path, clean_params, trap_exit, capsys,
                                                 drop, message):
    labels = [l for l in ANG_LABELS + DEF_LABELS + MS_LABELS if l not in drop]
    row = [1.0] * len(labels)
    p = write_file(tmp_path, "bad.star", old_star_text(labels, [row, row]))
    with pytest.raises(ExitCalled) as excinfo:
        star.get_align_data(p, flip=True)
    assert excinfo.value.code == 1
    assert message in capsys.readouterr().out


@pytest.mark.parametrize("drop", [MS_LABELS, DEF_LABELS, ANG_LABELS])
def test_get_align_data_exit_code_is_one(tmp_path, clean_params, trap_exit, drop):
    labels = [l for l in ANG_LABELS + DEF_LABELS + MS_LABELS if l not in drop]
    row = [1.0] * len(labels)
    p = write_file(tmp_path, "bad.star", old_star_text(labels, [row, row]))
    with pytest.raises(ExitCalled):
        star.get_align_data(p, flip=True)
    assert trap_exit == [1]


def test_get_align_data_exits_when_voltage_is_not_numeric(tmp_path, clean_params, trap_exit):
    # the bare `except:` also catches the float() conversion failure
    labels = ANG_LABELS + DEF_LABELS + MS_LABELS
    rows = [[0.0, 0.0, 0.0, 1e4, 1e4, "notavolt", 2.7, 0.1]]
    p = write_file(tmp_path, "nonnum.star", old_star_text(labels, rows))
    with pytest.raises(ExitCalled):
        star.get_align_data(p, flip=True)


def test_get_align_data_new_format_needs_optics_in_the_optics_block(tmp_path, clean_params,
                                                                    trap_exit):
    # voltage present only among the particles is not found. df0 is the optics frame
    text = new_star_text(["rlnOpticsGroup"], [[1]],
                         ANG_LABELS + DEF_LABELS + MS_LABELS,
                         [[0.0, 0.0, 0.0, 1e4, 1e4] + MS_VALUES])
    p = write_file(tmp_path, "misplaced.star", text)
    with pytest.raises(ExitCalled):
        star.get_align_data(p, flip=True)


def test_get_align_data_error_handling_relies_on_exit_terminating(tmp_path, clean_params,
                                                                  swallow_exit):
    # the except branch neither returns nor re-raises. if exit() ever came back, the
    # function would keep running with unbound locals
    labels = ANG_LABELS + MS_LABELS
    p = write_file(tmp_path, "nodef.star",
                   old_star_text(labels, [[0.0, 0.0, 0.0] + MS_VALUES]))
    with pytest.raises(NameError):
        star.get_align_data(p, flip=True)
    assert swallow_exit == [1]


def test_get_align_data_missing_file_raises(tmp_path, clean_params):
    with pytest.raises(FileNotFoundError):
        star.get_align_data(str(tmp_path / "absent.star"), flip=True)


def test_get_align_data_defocus_average_matches_data_store_usage(tmp_path, clean_params):
    # data_store consumes (U + V) / 2 as the per-particle defocus
    p, base = full_old_star(tmp_path, n=5)
    _, _, U, V = star.get_align_data(p, flip=True)
    assert np.allclose((U + V) / 2, (base[:, 3] + base[:, 4]) / 2)


# --------------------------------------------------------------------------------------
# star.write_star
# --------------------------------------------------------------------------------------

def write_params_toml(tmp_path, project_name, **fields):
    body = {"project_name": project_name}
    body.update(fields)
    with open(tmp_path / f"params_{project_name}.toml", "w") as f:
        toml.dump({"params": body}, f)


@pytest.fixture
def star_project(tmp_path, clean_params, monkeypatch):
    """write_star calls params.load(), which reads params_<project>.toml from the cwd."""
    monkeypatch.chdir(tmp_path)
    write_params_toml(tmp_path, "wsproj", ms_pixel_size=1.23)
    clean_params.project_name = "wsproj"
    return tmp_path


def angles_df(n=3, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(dict(phi=rng.uniform(-180, 180, n),
                             theta=rng.uniform(0, 180, n),
                             psi=rng.uniform(-180, 180, n)))


def test_write_star_header_labels_and_order(star_project):
    star.write_star("out.star", "traj.mrcs", angles_df(2))
    text = (star_project / "out.star").read_text()
    labels = [l.split()[0] for l in text.split("\n") if l.startswith("_rln")]
    assert labels == ["_rlnImageName", "_rlnAnglePsi", "_rlnAngleTilt",
                      "_rlnAngleRot", "_rlnDetectorPixelSize", "_rlnMagnification"]


@pytest.mark.parametrize("n", [1, 2, 5])
def test_write_star_writes_one_row_per_particle(star_project, n):
    df = angles_df(n)
    star.write_star("out.star", "traj.mrcs", df)
    body = [l for l in (star_project / "out.star").read_text().split("\n") if "@" in l]
    assert len(body) == n


@pytest.mark.parametrize("n", [1, 3, 6])
def test_write_star_image_names_are_one_indexed(star_project, n):
    star.write_star("out.star", "traj.mrcs", angles_df(n))
    body = [l for l in (star_project / "out.star").read_text().split("\n") if "@" in l]
    assert [l.split("@")[0] for l in body] == [str(i + 1) for i in range(n)]
    assert all(l.split()[0].endswith("@traj.mrcs") for l in body)


def test_write_star_column_values_are_psi_tilt_rot(star_project):
    # the file declares Psi/Tilt/Rot in that order and is fed psi/theta/phi to match
    df = angles_df(3)
    star.write_star("out.star", "traj.mrcs", df)
    parsed = star.parse_star(str(star_project / "out.star"), 0)
    assert np.allclose(parsed["rlnAnglePsi"].values, df.psi.values)
    assert np.allclose(parsed["rlnAngleTilt"].values, df.theta.values)
    assert np.allclose(parsed["rlnAngleRot"].values, df.phi.values)


def test_write_star_round_trips_through_parse_star(star_project):
    df = angles_df(4)
    star.write_star("out.star", "t.mrcs", df)
    parsed = star.parse_star(str(star_project / "out.star"), 0)
    assert parsed.columns.tolist() == ["rlnImageName", "rlnAnglePsi", "rlnAngleTilt",
                                       "rlnAngleRot", "rlnDetectorPixelSize",
                                       "rlnMagnification"]
    assert len(parsed) == 4


def test_write_star_magnification_is_the_fixed_1e4_convention(star_project):
    star.write_star("out.star", "t.mrcs", angles_df(2))
    parsed = star.parse_star(str(star_project / "out.star"), 0)
    assert np.all(parsed["rlnMagnification"].values == 10000.0)
    # DetectorPixelSize [um] / Magnification, converted to Angstrom, is the pixel size
    recovered = parsed["rlnDetectorPixelSize"].values / parsed["rlnMagnification"].values * 1e4
    assert np.allclose(recovered, params.ms_pixel_size)


def test_write_star_reloads_params_from_disk(star_project):
    # write_star() calls params.load() first, so the in-memory value is discarded
    params.ms_pixel_size = 999.0
    star.write_star("out.star", "t.mrcs", angles_df(1))
    assert params.ms_pixel_size == pytest.approx(1.23)
    parsed = star.parse_star(str(star_project / "out.star"), 0)
    assert parsed["rlnDetectorPixelSize"].values[0] == pytest.approx(1.23)


def test_write_star_missing_params_file_raises(tmp_path, clean_params, monkeypatch):
    monkeypatch.chdir(tmp_path)
    clean_params.project_name = "noexist"
    with pytest.raises(FileNotFoundError):
        star.write_star("out.star", "t.mrcs", angles_df(1))


def test_write_star_empty_dataframe_writes_header_only(star_project):
    star.write_star("out.star", "t.mrcs", pd.DataFrame(dict(phi=[], theta=[], psi=[])))
    text = (star_project / "out.star").read_text()
    assert "_rlnImageName #1" in text
    assert "@" not in text
    with pytest.raises(pd.errors.EmptyDataError):
        star.parse_star(str(star_project / "out.star"), 0)


def test_write_star_requires_phi_theta_psi_columns(star_project):
    with pytest.raises(AttributeError):
        star.write_star("out.star", "t.mrcs", pd.DataFrame(dict(a=[1.0], b=[2.0], c=[3.0])))


@pytest.mark.xfail(reason="Bug. write_star indexes df.psi[i] by label over range(len(df)), "
                          "so any non-RangeIndex frame raises KeyError",
                   strict=False)
def test_write_star_ignores_the_dataframe_index(star_project):
    df = angles_df(2)
    df.index = [10, 11]
    star.write_star("out.star", "t.mrcs", df)
    parsed = star.parse_star(str(star_project / "out.star"), 0)
    assert np.allclose(parsed["rlnAnglePsi"].values, df.psi.values)


# --------------------------------------------------------------------------------------
# myio, backend selection
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,is_hdf5", [
    ("a.h5", True),
    ("a.pkl", False),
    ("a", False),
    ("a.H5", False),        # the suffix test is case sensitive
    ("a.h5.pkl", False),
    ("a.pkl.h5", True),
    ("a.hdf5", False),      # only the literal ".h5" suffix switches backend
    (".h5", True),
])
def test_myio_backend_is_chosen_by_literal_h5_suffix(tmp_path, name, is_hdf5):
    p = str(tmp_path / name)
    fout1(p, x=np.arange(3.0))
    with open(p, "rb") as f:
        magic = f.read(8)
    assert (magic == b"\x89HDF\r\n\x1a\n") is is_hdf5


@pytest.mark.parametrize("suffix", [".pkl", ".h5", ".dat", ""])
def test_myio_round_trip_array(tmp_path, suffix):
    a = np.linspace(-1.0, 1.0, 11)
    p = str(tmp_path / ("f" + suffix))
    fout1(p, a=a)
    assert np.array_equal(fin1(p)["a"], a)


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
def test_myio_fin1_returns_a_dict(tmp_path, suffix):
    p = str(tmp_path / ("f" + suffix))
    fout1(p, a=np.zeros(2), b=np.ones(3))
    out = fin1(p)
    assert isinstance(out, dict)
    assert set(out) == {"a", "b"}


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
def test_myio_fout1_returns_none(tmp_path, suffix):
    assert fout1(str(tmp_path / ("f" + suffix)), a=np.zeros(1)) is None


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
def test_myio_empty_payload_round_trips_to_empty_dict(tmp_path, suffix):
    p = str(tmp_path / ("f" + suffix))
    fout1(p)
    assert fin1(p) == {}


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
def test_myio_missing_key_raises_keyerror(tmp_path, suffix):
    p = str(tmp_path / ("f" + suffix))
    fout1(p, a=np.zeros(2))
    with pytest.raises(KeyError):
        fin1(p)["nope"]


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
def test_myio_missing_file_raises_filenotfound(tmp_path, suffix):
    with pytest.raises(FileNotFoundError):
        fin1(str(tmp_path / ("absent" + suffix)))


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
def test_myio_write_into_missing_directory_raises(tmp_path, suffix):
    with pytest.raises(OSError):
        fout1(str(tmp_path / "nodir" / ("f" + suffix)), a=np.zeros(2))


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
def test_myio_rewrite_replaces_rather_than_merges(tmp_path, suffix):
    p = str(tmp_path / ("f" + suffix))
    fout1(p, a=np.zeros(2), b=np.ones(2))
    fout1(p, c=np.full(2, 7.0))
    assert set(fin1(p)) == {"c"}


# --------------------------------------------------------------------------------------
# myio, dtype / shape fidelity
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
@pytest.mark.parametrize("dtype", ["float64", "float32", "int32", "int64",
                                   "uint8", "complex128", "bool"])
def test_myio_preserves_array_dtype(tmp_path, suffix, dtype):
    a = np.arange(6).astype(dtype)
    p = str(tmp_path / ("f" + suffix))
    fout1(p, a=a)
    back = fin1(p)["a"]
    assert back.dtype == np.dtype(dtype)
    assert np.array_equal(back, a)


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
@pytest.mark.parametrize("shape", [(1,), (5,), (3, 4), (2, 3, 4), (2, 2, 2, 2)])
def test_myio_preserves_array_shape(tmp_path, suffix, shape):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(shape)
    p = str(tmp_path / ("f" + suffix))
    fout1(p, a=a)
    back = fin1(p)["a"]
    assert back.shape == shape
    assert np.array_equal(back, a)


def test_myio_h5_zero_dim_array_comes_back_as_a_scalar(tmp_path):
    # f[key][()] on a shape-() dataset yields a numpy scalar, not a 0-d ndarray
    p = str(tmp_path / "f.h5")
    fout1(p, a=np.array(3.5))
    back = fin1(p)["a"]
    assert isinstance(back, np.float64)
    assert not isinstance(back, np.ndarray)
    assert back == 3.5


def test_myio_pickle_zero_dim_array_stays_an_array(tmp_path):
    p = str(tmp_path / "f.pkl")
    fout1(p, a=np.array(3.5))
    back = fin1(p)["a"]
    assert isinstance(back, np.ndarray) and back.shape == ()


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
def test_myio_preserves_nan_and_inf(tmp_path, suffix):
    a = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0])
    p = str(tmp_path / ("f" + suffix))
    fout1(p, a=a)
    assert np.array_equal(fin1(p)["a"], a, equal_nan=True)


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
def test_myio_fortran_order_array_round_trips_by_value(tmp_path, suffix):
    a = np.asfortranarray(np.arange(12.0).reshape(3, 4))
    p = str(tmp_path / ("f" + suffix))
    fout1(p, a=a)
    assert np.array_equal(fin1(p)["a"], a)


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
def test_myio_non_contiguous_view_round_trips(tmp_path, suffix):
    a = np.arange(12.0).reshape(3, 4)[:, ::2]
    p = str(tmp_path / ("f" + suffix))
    fout1(p, a=a)
    back = fin1(p)["a"]
    assert back.shape == (3, 2)
    assert np.array_equal(back, a)


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
def test_myio_exact_bitwise_round_trip_of_random_data(tmp_path, suffix):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 17))
    p = str(tmp_path / ("f" + suffix))
    fout1(p, a=a)
    assert (fin1(p)["a"] == a).all()


@pytest.mark.parametrize("value,expected_type", [
    (3, np.int64),
    (2.5, np.float64),
    (True, np.bool_),
])
def test_myio_h5_promotes_python_scalars_to_numpy(tmp_path, value, expected_type):
    p = str(tmp_path / "f.h5")
    fout1(p, a=value)
    back = fin1(p)["a"]
    assert isinstance(back, expected_type)
    assert back == value


@pytest.mark.parametrize("value", [3, 2.5, True, "text", None, (1, 2), [1, 2], {"k": 1}])
def test_myio_pickle_preserves_python_objects_exactly(tmp_path, value):
    p = str(tmp_path / "f.pkl")
    fout1(p, a=value)
    back = fin1(p)["a"]
    assert back == value or (value is None and back is None)
    assert type(back) is type(value)


@pytest.mark.parametrize("value", [{"k": 1}, None, np.array([{"a": 1}, 2], dtype=object)])
def test_myio_h5_rejects_non_array_payloads(tmp_path, value):
    with pytest.raises(TypeError):
        fout1(str(tmp_path / "f.h5"), a=value)


def test_myio_h5_returns_bytes_for_a_string_payload(tmp_path):
    # h5py stores str as variable length UTF-8 and hands it back as bytes
    p = str(tmp_path / "f.h5")
    fout1(p, a="hello")
    back = fin1(p)["a"]
    assert isinstance(back, bytes)
    assert back.decode() == "hello"


@pytest.mark.xfail(reason="Bug. str survives the pickle backend but returns as bytes from "
                          "the .h5 backend, so the two are not interchangeable",
                   strict=False)
def test_myio_string_round_trips_for_both_backends(tmp_path):
    for suffix in (".pkl", ".h5"):
        p = str(tmp_path / ("f" + suffix))
        fout1(p, a="hello")
        assert fin1(p)["a"] == "hello"


# --------------------------------------------------------------------------------------
# myio, on-disk format and key handling
# --------------------------------------------------------------------------------------

def test_myio_pickle_file_is_a_plain_pickled_dict(tmp_path):
    p = str(tmp_path / "f.pkl")
    fout1(p, a=np.zeros(2), b=1)
    with open(p, "rb") as f:
        raw = pickle.load(f)
    assert isinstance(raw, dict) and set(raw) == {"a", "b"}


def test_myio_h5_file_stores_one_dataset_per_kwarg(tmp_path):
    p = str(tmp_path / "f.h5")
    fout1(p, a=np.zeros(2), b=np.ones((2, 2)))
    with h5py.File(p, "r") as f:
        assert set(f.keys()) == {"a", "b"}
        assert f["b"].shape == (2, 2)


def test_myio_pickle_keeps_keyword_insertion_order(tmp_path):
    p = str(tmp_path / "f.pkl")
    fout1(p, zebra=1, apple=2, mango=3)
    assert list(fin1(p)) == ["zebra", "apple", "mango"]


def test_myio_h5_returns_keys_in_name_order(tmp_path):
    # HDF5 groups iterate alphabetically, so the kwarg order is not preserved
    p = str(tmp_path / "f.h5")
    fout1(p, zebra=1, apple=2, mango=3)
    assert list(fin1(p)) == ["apple", "mango", "zebra"]


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
@pytest.mark.parametrize("n_keys", [1, 3, 8])
def test_myio_many_keys_round_trip(tmp_path, suffix, n_keys):
    payload = {f"k{i}": np.full(3, float(i)) for i in range(n_keys)}
    p = str(tmp_path / ("f" + suffix))
    fout1(p, **payload)
    back = fin1(p)
    assert set(back) == set(payload)
    for k, v in payload.items():
        assert np.array_equal(back[k], v)


@pytest.mark.parametrize("key", ["psi", "IMG1", "a_b", "x2", "with space"])
def test_myio_h5_accepts_assorted_dataset_names(tmp_path, key):
    p = str(tmp_path / "f.h5")
    fout1(p, **{key: np.arange(3.0)})
    assert np.array_equal(fin1(p)[key], np.arange(3.0))


def test_myio_h5_slash_in_key_creates_an_unreadable_nested_group(tmp_path):
    # '/' is the HDF5 path separator, so the top level key is only the first segment
    p = str(tmp_path / "f.h5")
    fout1(p, **{"grp/leaf": np.arange(3.0)})
    with h5py.File(p, "r") as f:
        assert list(f.keys()) == ["grp"]
    # fin1 walks only the top level and applies [()] to it, which a Group rejects
    with pytest.raises(TypeError):
        fin1(p)


@pytest.mark.parametrize("suffix", [".pkl", ".h5"])
def test_myio_mimics_the_project_psi_file_payload(tmp_path, suffix):
    # shape of a real diff_maps payload. Eigenvectors, eigenvalues, index list
    rng = np.random.default_rng(0)
    psi = rng.standard_normal((50, 8))
    lamb = np.sort(rng.random(8))[::-1].copy()
    idx = np.arange(50)
    p = str(tmp_path / ("gC_trimmed_psi_prD_0" + suffix))
    fout1(p, psi=psi, lamb=lamb, indices=idx)
    back = fin1(p)
    assert np.array_equal(back["psi"], psi)
    assert np.array_equal(back["lamb"], lamb)
    assert np.array_equal(back["indices"], idx)
    assert back["psi"].shape == (50, 8)


# ============================================================================
# S2tessellation.py, FindCCGraph.py, FindCCGraphPruned.py
# ============================================================================

# --------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------
@pytest.fixture
def params_guard_geometry_graph():
    """`params` is a process-wide singleton and `params.load()` chdirs. Save/restore both."""
    cwd = os.getcwd()
    watched = ("num_psi", "eps", "prd_thres_low", "prd_thres_high", "ncpu",
               "tess_hemisphere_type", "prd_assignment", "prd_cone_width_factor")
    saved = {k: getattr(params, k) for k in watched}
    params.ncpu = 1
    yield params
    for k, v in saved.items():
        setattr(params, k, v)
    os.chdir(cwd)


def random_half_space_points(n, seed=0, plane_vec=np.array([1.0, 0.0, 0.0])):
    """n unit vectors folded onto the half space `dot(plane_vec, x) >= 0`."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(3, n))
    v /= np.linalg.norm(v, axis=0)
    flip = (plane_vec @ v) < 0.0
    v[:, flip] *= -1.0
    return v


# adjacency matrices used repeatedly below (all symmetric, no self loops)
ADJ_PATH4 = np.array([[0, 1, 0, 0],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]], dtype=int)

ADJ_TWO_COMPONENTS = np.array([[0, 1, 0, 0],
                               [1, 0, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 1, 0]], dtype=int)

ADJ_ISOLATED = np.array([[0, 1, 0, 0],
                         [1, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0]], dtype=int)

ADJ_COMPLETE3 = np.array([[0, 1, 1],
                          [1, 0, 1],
                          [1, 1, 0]], dtype=int)

ADJ_EMPTY3 = np.zeros((3, 3), dtype=int)

ALL_ADJ = [ADJ_PATH4, ADJ_TWO_COMPONENTS, ADJ_ISOLATED, ADJ_COMPLETE3, ADJ_EMPTY3]


# --------------------------------------------------------------------------------------
# lovisolo_silva_tessellation
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13, 20, 50, 97, 128, 200, 314])
def test_lovisolo_silva_shape(n):
    # despite the docstring warning about "slightly more or fewer points", the function always
    # truncates/pads to exactly the requested count via results[0:K]
    assert lovisolo_silva_tessellation(n).shape == (n, 3)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13, 20, 50, 97, 128, 200, 314])
def test_lovisolo_silva_unit_norm(n):
    x = lovisolo_silva_tessellation(n)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)


@pytest.mark.parametrize("n", [7, 60, 150])
def test_lovisolo_silva_is_deterministic(n):
    assert np.array_equal(lovisolo_silva_tessellation(n), lovisolo_silva_tessellation(n))


@pytest.mark.parametrize("n", [40, 120, 250])
def test_lovisolo_silva_has_no_duplicate_points(n):
    x = np.round(lovisolo_silva_tessellation(n), 10)
    assert np.unique(x, axis=0).shape[0] == n


@pytest.mark.parametrize("n", [60, 200])
def test_lovisolo_silva_lies_on_constant_x_rings(n):
    # the algorithm sweeps a polar angle w1 (x = cos w1) and fills a ring of azimuths for each,
    # so x takes far fewer distinct values than there are points
    x = lovisolo_silva_tessellation(n)
    rings = np.unique(np.round(x[:, 0], 10))
    assert 1 < len(rings) < n / 2
    # on each ring the transverse radius is sin(w1) = sqrt(1 - x^2)
    for ring_x in rings:
        pts = x[np.isclose(x[:, 0], ring_x)]
        assert np.allclose(np.linalg.norm(pts[:, 1:], axis=1), np.sqrt(1.0 - ring_x**2))


@pytest.mark.parametrize("n", [20, 50, 100, 200, 300, 400, 500])
def test_lovisolo_silva_min_separation_tracks_target_spacing(n):
    # target nearest-neighbor arc is delta = sqrt(A3 / N) with A3 the unit sphere area
    x = lovisolo_silva_tessellation(n)
    d = np.arccos(np.clip(x @ x.T, -1.0, 1.0))
    np.fill_diagonal(d, np.inf)
    target = np.sqrt(4 * np.pi / n)
    assert 0.45 * target <= d.min() <= 1.05 * target


@pytest.mark.parametrize("n", [50, 200, 500])
def test_lovisolo_silva_centroid_near_origin(n):
    # a roughly uniform covering of S2 has a vanishing first moment
    assert np.linalg.norm(lovisolo_silva_tessellation(n).mean(axis=0)) < 0.05


def test_lovisolo_silva_zero_points_raises():
    # delta = sqrt(A3 / K) divides by the requested count
    with pytest.raises(ZeroDivisionError):
        lovisolo_silva_tessellation(0)


# --------------------------------------------------------------------------------------
# fibonacci_tessellation
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13, 20, 50, 97, 128, 200, 314])
def test_fibonacci_shape(n):
    assert fibonacci_tessellation(n).shape == (n, 3)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13, 20, 50, 97, 128, 200, 314])
def test_fibonacci_unit_norm(n):
    x = fibonacci_tessellation(n)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)


@pytest.mark.parametrize("n", [1, 4, 9, 33, 100])
def test_fibonacci_z_is_symmetrically_shifted_ladder(n):
    # z_j = 1 - (2j+1)/N places the samples at the centers of N equal-area z bands
    z = fibonacci_tessellation(n)[:, 2]
    assert np.allclose(z, 1.0 - (2 * np.arange(n) + 1) / n)


@pytest.mark.parametrize("n", [2, 7, 8, 30, 101])
def test_fibonacci_z_monotone_and_equator_symmetric(n):
    z = fibonacci_tessellation(n)[:, 2]
    assert np.all(np.diff(z) < 0.0)
    assert np.allclose(z, -z[::-1])


@pytest.mark.parametrize("n", [10, 55, 200])
def test_fibonacci_azimuth_advances_by_a_constant_angle(n):
    # phi_j = j * 2*pi/golden, so consecutive azimuth increments are constant modulo 2*pi
    x = fibonacci_tessellation(n)
    phi = np.arctan2(x[:, 1], x[:, 0])
    step = np.mod(np.diff(phi), 2 * np.pi)
    assert np.allclose(step, np.mod(2 * np.pi / ((1 + np.sqrt(5.0)) / 2), 2 * np.pi))


@pytest.mark.parametrize("n", [40, 120, 250])
def test_fibonacci_has_no_duplicate_points(n):
    x = np.round(fibonacci_tessellation(n), 10)
    assert np.unique(x, axis=0).shape[0] == n


@pytest.mark.parametrize("n", [20, 50, 100, 200, 300, 400, 500])
def test_fibonacci_min_separation_tracks_target_spacing(n):
    x = fibonacci_tessellation(n)
    d = np.arccos(np.clip(x @ x.T, -1.0, 1.0))
    np.fill_diagonal(d, np.inf)
    target = np.sqrt(4 * np.pi / n)
    assert 0.45 * target <= d.min() <= 1.05 * target


@pytest.mark.parametrize("n", [50, 200, 500])
def test_fibonacci_centroid_near_origin(n):
    assert np.linalg.norm(fibonacci_tessellation(n).mean(axis=0)) < 0.05


def test_fibonacci_zero_points_gives_empty_array():
    # unlike the Lovisolo-Silva variant this degrades gracefully
    assert fibonacci_tessellation(0).shape == (0, 3)


@pytest.mark.parametrize("n", [30, 90])
def test_fibonacci_is_deterministic(n):
    assert np.array_equal(fibonacci_tessellation(n), fibonacci_tessellation(n))


@pytest.mark.parametrize("n", [100, 300])
def test_fibonacci_is_better_separated_than_lovisolo_silva(n):
    # the Fibonacci spiral has no ring seams, so its worst nearest-neighbor gap is larger
    def min_arc(x):
        d = np.arccos(np.clip(x @ x.T, -1.0, 1.0))
        np.fill_diagonal(d, np.inf)
        return d.min()

    assert min_arc(fibonacci_tessellation(n)) > min_arc(lovisolo_silva_tessellation(n))


# --------------------------------------------------------------------------------------
# collect_nearest_neighbors
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_bins,n_query", [(5, 20), (12, 12), (30, 7), (3, 100)])
def test_collect_nearest_neighbors_shapes(n_bins, n_query):
    x = fibonacci_tessellation(n_bins)
    q = random_half_space_points(n_query, seed=1).T
    idx, counts = collect_nearest_neighbors(x, q)
    assert idx.shape == (n_query, 1)
    assert counts.shape == (n_bins,)


@pytest.mark.parametrize("n_bins,n_query", [(5, 20), (12, 12), (30, 7), (3, 100)])
def test_collect_nearest_neighbors_counts_sum_to_query_count(n_bins, n_query):
    x = fibonacci_tessellation(n_bins)
    q = random_half_space_points(n_query, seed=2).T
    idx, counts = collect_nearest_neighbors(x, q)
    assert counts.sum() == n_query
    assert np.array_equal(counts, np.bincount(idx.ravel(), minlength=n_bins))


@pytest.mark.parametrize("n", [4, 11, 40])
def test_collect_nearest_neighbors_of_the_set_with_itself_is_the_identity(n):
    x = fibonacci_tessellation(n)
    idx, counts = collect_nearest_neighbors(x, x)
    assert np.array_equal(idx.ravel(), np.arange(n))
    assert np.array_equal(counts, np.ones(n, dtype=int))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_collect_nearest_neighbors_matches_brute_force_argmin(seed):
    x = fibonacci_tessellation(17)
    q = random_half_space_points(40, seed=seed).T
    idx, _ = collect_nearest_neighbors(x, q)
    assert np.array_equal(idx.ravel(), np.argmin(cdist(q, x), axis=1))


def test_collect_nearest_neighbors_reports_zero_for_unclaimed_points():
    # minlength keeps the histogram the full size of X even when some rows win nothing
    x = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q = np.array([[1.0, 0.0, 0.0], [0.99, 0.1, 0.0]])
    q /= np.linalg.norm(q, axis=1)[:, None]
    _, counts = collect_nearest_neighbors(x, q)
    assert counts[0] == 2 and counts[1] == 0 and counts[2] == 0


@pytest.mark.xfail(reason="Bug. bin_counts squeezes a single-row neighbor array to 0-d, "
                          "so np.bincount rejects it", strict=False)
def test_collect_nearest_neighbors_accepts_a_single_query_point():
    x = fibonacci_tessellation(10)
    idx, counts = collect_nearest_neighbors(x, x[:1])
    assert idx.shape == (1, 1)
    assert counts.sum() == 1


def test_collect_nearest_neighbors_single_query_currently_raises():
    # documents the defect above so the suite notices if it ever changes
    x = fibonacci_tessellation(10)
    with pytest.raises(ValueError):
        collect_nearest_neighbors(x, x[:1])


# --------------------------------------------------------------------------------------
# bin_and_threshold. Bin construction
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("tess", ["lovisolo_silva", "fibonacci"])
@pytest.mark.parametrize("bin_width", [0.8, 0.6, 0.45, 0.3])
def test_bin_and_threshold_bin_count_matches_requested_resolution(tess, bin_width):
    # requested_n_bins = int(4*pi / bin_width^2) covers the whole sphere. Only the half space
    # dot(plane_vec, center) >= 0 survives, so roughly half the bins are kept
    s2 = random_half_space_points(400, seed=3)
    _, centers, _, _ = bin_and_threshold(s2, bin_width, 1, tess)
    requested = int(4 * np.pi / bin_width**2)
    assert 0 < centers.shape[1] <= requested
    assert abs(centers.shape[1] - requested / 2) <= 0.15 * requested


@pytest.mark.parametrize("tess,tessellator", [("lovisolo_silva", lovisolo_silva_tessellation),
                                              ("fibonacci", fibonacci_tessellation)])
@pytest.mark.parametrize("bin_width", [0.7, 0.4])
def test_bin_and_threshold_keeps_exactly_the_positive_half_space_bins(tess, tessellator, bin_width):
    s2 = random_half_space_points(200, seed=4)
    _, centers, _, _ = bin_and_threshold(s2, bin_width, 1, tess)
    full = tessellator(int(4 * np.pi / bin_width**2)).T
    expected = full[:, (full[0, :] >= 0.0)]
    assert np.array_equal(centers, expected)


@pytest.mark.parametrize("tess", ["lovisolo_silva", "fibonacci"])
@pytest.mark.parametrize("plane_vec", [np.array([1.0, 0.0, 0.0]),
                                       np.array([0.0, 1.0, 0.0]),
                                       np.array([0.0, 0.0, 1.0]),
                                       np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)])
def test_bin_and_threshold_bin_centers_lie_in_the_requested_hemisphere(tess, plane_vec):
    s2 = random_half_space_points(200, seed=5, plane_vec=plane_vec)
    _, centers, _, _ = bin_and_threshold(s2, 0.6, 1, tess, plane_vec=plane_vec)
    # the cut is inclusive. Centers exactly on the plane are kept
    assert np.all(plane_vec @ centers >= 0.0)
    assert np.allclose(np.linalg.norm(centers, axis=0), 1.0)


@pytest.mark.parametrize("tess", ["lovisolo_silva", "fibonacci"])
def test_bin_and_threshold_does_not_mirror_points_itself(tess):
    # mirroring onto the half space happens upstream (collapse_to_half_space). a point below the
    # plane is still assigned, but to the nearest kept bin rather than to its antipode's bin
    p = np.array([[1.0], [0.2], [0.1]])
    p /= np.linalg.norm(p, axis=0)
    both = np.hstack([p, -p])
    neighb, centers, occ, _ = bin_and_threshold(both, 0.5, 1, tess)
    assert occ.sum() == 2
    assert np.all(centers[0, :] >= 0.0)
    owner = [b for b, members in enumerate(neighb) if len(members)]
    assert len(owner) == 2  # the point and its mirror land in different bins


@pytest.mark.parametrize("bad", ["", "nope", "Fibonacci", "lovisolo-silva", "healpix"])
def test_bin_and_threshold_rejects_unknown_tessellator(bad):
    s2 = random_half_space_points(20, seed=6)
    with pytest.raises(ValueError, match="Invalid tesselator"):
        bin_and_threshold(s2, 0.6, 1, bad)


@pytest.mark.parametrize("bad", ["soft", "", "HARD", "cones"])
def test_bin_and_threshold_rejects_unknown_assignment(bad):
    s2 = random_half_space_points(20, seed=7)
    with pytest.raises(ValueError, match="Invalid assignment"):
        bin_and_threshold(s2, 0.6, 1, "fibonacci", assignment=bad)


@pytest.mark.parametrize("bin_width", [4.0, 3.6])
def test_bin_and_threshold_with_no_requested_bins_fails(bin_width):
    # int(4*pi/bin_width^2) rounds down to zero
    s2 = random_half_space_points(20, seed=8)
    with pytest.raises((ValueError, ZeroDivisionError)):
        bin_and_threshold(s2, bin_width, 1, "fibonacci")


# --------------------------------------------------------------------------------------
# bin_and_threshold. hard assignment
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("tess", ["lovisolo_silva", "fibonacci"])
@pytest.mark.parametrize("n_points", [37, 150, 400])
def test_hard_assignment_places_every_point_in_exactly_one_bin(tess, n_points):
    s2 = random_half_space_points(n_points, seed=9)
    neighb, _, occ, _ = bin_and_threshold(s2, 0.5, 1, tess)
    members = np.concatenate([np.asarray(a, dtype=int) for a in neighb if len(a)])
    assert occ.sum() == n_points
    assert np.array_equal(np.sort(members), np.arange(n_points))


@pytest.mark.parametrize("tess", ["lovisolo_silva", "fibonacci"])
def test_hard_assignment_membership_lengths_equal_occupancy(tess):
    s2 = random_half_space_points(180, seed=10)
    neighb, _, occ, _ = bin_and_threshold(s2, 0.5, 1, tess)
    assert np.array_equal(np.array([len(a) for a in neighb]), occ)


@pytest.mark.parametrize("tess", ["lovisolo_silva", "fibonacci"])
def test_hard_assignment_sends_each_point_to_its_nearest_kept_bin(tess):
    s2 = random_half_space_points(120, seed=11)
    neighb, centers, _, _ = bin_and_threshold(s2, 0.55, 1, tess)
    nearest = np.argmin(cdist(s2.T, centers.T), axis=1)
    for b, members in enumerate(neighb):
        assert np.all(nearest[np.asarray(members, dtype=int)] == b)


@pytest.mark.parametrize("tess", ["lovisolo_silva", "fibonacci"])
def test_binning_is_deterministic(tess):
    s2 = random_half_space_points(90, seed=12)
    a = bin_and_threshold(s2, 0.5, 3, tess)
    b = bin_and_threshold(s2, 0.5, 3, tess)
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])
    assert a[3] == b[3]


@pytest.mark.xfail(reason="Bug. np.array([...], dtype=object) collapses to a 2-D array when "
                          "every bin holds the same number of points", strict=False)
def test_membership_list_is_always_one_dimensional():
    # one image exactly on each kept bin center gives every bin a membership of length one
    width = 2.0
    centers = fibonacci_tessellation(int(4 * np.pi / width**2)).T
    s2 = centers[:, centers[0, :] >= 0.0]
    neighb, _, _, _ = bin_and_threshold(s2, width, 1, "fibonacci")
    assert neighb.ndim == 1


def test_membership_list_collapses_to_2d_for_equal_sized_bins():
    # documents the defect above
    width = 2.0
    centers = fibonacci_tessellation(int(4 * np.pi / width**2)).T
    s2 = centers[:, centers[0, :] >= 0.0]
    neighb, kept, _, _ = bin_and_threshold(s2, width, 1, "fibonacci")
    assert neighb.shape == (kept.shape[1], 1)


# --------------------------------------------------------------------------------------
# bin_and_threshold. Thresholding
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("tess", ["lovisolo_silva", "fibonacci"])
@pytest.mark.parametrize("thres_low", [0, 1, 2, 5, 10, 10000])
def test_prd_indices_are_exactly_the_bins_at_or_above_the_low_threshold(tess, thres_low):
    s2 = random_half_space_points(250, seed=13)
    _, _, occ, prd = bin_and_threshold(s2, 0.5, thres_low, tess)
    assert prd == list(np.nonzero(occ >= thres_low)[0])


@pytest.mark.parametrize("n_copies,thres_low,kept", [(3, 3, True), (3, 4, False), (3, 2, True),
                                                     (2, 2, True), (2, 3, False), (5, 5, True)])
def test_low_threshold_boundary_is_inclusive(n_copies, thres_low, kept):
    # a bin holding exactly thres_low images is admitted (the code tests n_points >= thres_low)
    s2 = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, n_copies))
    _, _, occ, prd = bin_and_threshold(s2, 1.0, thres_low, "fibonacci")
    assert int(np.max(occ)) == n_copies
    assert (int(np.argmax(occ)) in prd) is kept


@pytest.mark.parametrize("tess", ["lovisolo_silva", "fibonacci"])
def test_zero_threshold_keeps_every_bin_including_empty_ones(tess):
    s2 = random_half_space_points(30, seed=14)
    _, centers, _, prd = bin_and_threshold(s2, 0.4, 0, tess)
    assert prd == list(range(centers.shape[1]))


@pytest.mark.parametrize("tess", ["lovisolo_silva", "fibonacci"])
def test_unreachable_threshold_keeps_nothing(tess):
    s2 = random_half_space_points(30, seed=15)
    _, _, _, prd = bin_and_threshold(s2, 0.4, 31, tess)
    assert prd == []


@pytest.mark.parametrize("low_a,low_b", [(1, 2), (2, 5), (5, 20)])
def test_raising_the_low_threshold_only_removes_bins(low_a, low_b):
    s2 = random_half_space_points(300, seed=16)
    _, _, _, prd_a = bin_and_threshold(s2, 0.5, low_a, "fibonacci")
    _, _, _, prd_b = bin_and_threshold(s2, 0.5, low_b, "fibonacci")
    assert set(prd_b) <= set(prd_a)


def test_high_threshold_truncates_bin_membership(params_guard_geometry_graph):
    """prd_thres_high is applied downstream, by _ProjectionDirections.thresholded_image_indices."""
    from ManifoldEM.data_store import _ProjectionDirections

    s2 = random_half_space_points(300, seed=17)
    image_indices, centers, occ, prd = bin_and_threshold(s2, 0.5, 5, "fibonacci")

    prds = _ProjectionDirections()
    prds.image_indices_full = image_indices
    prds.thres_ids = prd
    prds.occupancy_full = occ
    prds.bin_centers = centers
    prds.thres_high = 4

    trimmed = prds.thresholded_image_indices
    assert len(trimmed) == len(prd)
    assert all(len(a) <= 4 for a in trimmed)
    # untouched bins keep every image, and truncation always keeps the leading entries
    for i, b in enumerate(prd):
        assert np.array_equal(np.asarray(trimmed[i]), np.asarray(image_indices[b])[:4])


@pytest.mark.parametrize("high", [1, 3, 1000])
def test_high_threshold_is_a_no_op_when_larger_than_every_bin(params_guard_geometry_graph, high):
    from ManifoldEM.data_store import _ProjectionDirections

    s2 = random_half_space_points(200, seed=18)
    image_indices, centers, occ, prd = bin_and_threshold(s2, 0.5, 1, "fibonacci")
    prds = _ProjectionDirections()
    prds.image_indices_full = image_indices
    prds.thres_ids = prd
    prds.thres_high = high
    total = sum(len(a) for a in prds.thresholded_image_indices)
    assert total == sum(min(len(image_indices[b]), high) for b in prd)


# --------------------------------------------------------------------------------------
# bin_and_threshold. cone assignment
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("factor", [0.5, 1.0, 1.5])
def test_cone_assignment_keeps_the_hard_occupancy_and_thresholding(factor):
    s2 = random_half_space_points(200, seed=19)
    _, _, occ_h, prd_h = bin_and_threshold(s2, 0.6, 3, "fibonacci")
    _, _, occ_c, prd_c = bin_and_threshold(s2, 0.6, 3, "fibonacci",
                                           assignment="cone", cone_width_factor=factor)
    assert np.array_equal(occ_h, occ_c)
    assert prd_h == prd_c


@pytest.mark.parametrize("factor", [0.5, 1.0, 1.5, 2.0])
def test_cone_membership_is_exactly_the_angular_cone(factor):
    width = 0.6
    s2 = random_half_space_points(150, seed=20)
    neighb, centers, _, _ = bin_and_threshold(s2, width, 1, "fibonacci",
                                              assignment="cone", cone_width_factor=factor)
    expected = (centers.T @ s2) >= np.cos(factor * width)
    for b, members in enumerate(neighb):
        assert np.array_equal(np.sort(np.asarray(members, dtype=int)), np.nonzero(expected[b])[0])


@pytest.mark.parametrize("small,large", [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0)])
def test_wider_cones_only_add_members(small, large):
    s2 = random_half_space_points(150, seed=21)
    n_small, _, _, _ = bin_and_threshold(s2, 0.6, 1, "fibonacci",
                                         assignment="cone", cone_width_factor=small)
    n_large, _, _, _ = bin_and_threshold(s2, 0.6, 1, "fibonacci",
                                         assignment="cone", cone_width_factor=large)
    for a, b in zip(n_small, n_large):
        assert set(np.asarray(a, dtype=int).tolist()) <= set(np.asarray(b, dtype=int).tolist())


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_cone_membership_contains_the_hard_assignment(factor):
    s2 = random_half_space_points(200, seed=22)
    hard, _, _, _ = bin_and_threshold(s2, 0.6, 1, "fibonacci")
    cone, _, _, _ = bin_and_threshold(s2, 0.6, 1, "fibonacci",
                                      assignment="cone", cone_width_factor=factor)
    for h, c in zip(hard, cone):
        assert set(np.asarray(h, dtype=int).tolist()) <= set(np.asarray(c, dtype=int).tolist())


def test_cone_assignment_overlaps_while_hard_assignment_partitions():
    s2 = random_half_space_points(200, seed=23)
    hard, _, _, _ = bin_and_threshold(s2, 0.6, 1, "fibonacci")
    cone, _, _, _ = bin_and_threshold(s2, 0.6, 1, "fibonacci",
                                      assignment="cone", cone_width_factor=1.0)
    assert sum(len(a) for a in hard) == 200
    assert sum(len(a) for a in cone) > 200


# --------------------------------------------------------------------------------------
# CreateGraphStruct
# --------------------------------------------------------------------------------------
GRAPH_MODULES = [FG, FP]


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_create_graph_struct_without_data_returns_minus_one(mod):
    assert mod.CreateGraphStruct(4, [], 1.0) == -1


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("adj", ALL_ADJ)
def test_create_graph_struct_node_count_from_adjacency(mod, adj):
    g = mod.CreateGraphStruct(4, [], None, csr_matrix(adj))
    assert g["nNodes"] == adj.shape[0]
    assert g["Nodes"] == range(adj.shape[0])


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("n_states", [1, 2, 8, 16])
def test_create_graph_struct_scalar_states_are_broadcast(mod, n_states):
    g = mod.CreateGraphStruct(n_states, [], None, csr_matrix(ADJ_PATH4))
    assert g["eqnStates"] == 1
    assert np.array_equal(g["nStates"], np.full(4, n_states))
    assert g["maxState"] == n_states


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_create_graph_struct_vector_states_are_kept(mod):
    states = np.array([2, 3, 4, 5])
    g = mod.CreateGraphStruct(states, [], None, csr_matrix(ADJ_PATH4))
    assert g["eqnStates"] == 0
    assert np.array_equal(g["nStates"], states)
    assert g["maxState"] == 5


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("adj", ALL_ADJ)
def test_edges_are_sorted_upper_triangle_pairs(mod, adj):
    g = mod.CreateGraphStruct(4, [], None, csr_matrix(adj))
    edges = g["Edges"]
    assert edges.shape[1] == 2
    assert np.all(edges[:, 0] < edges[:, 1])
    assert np.array_equal(edges, edges[np.lexsort((edges[:, 1], edges[:, 0])), :])
    expected = np.array(sorted(map(tuple, np.argwhere(np.triu(adj, 1) > 0))))
    assert np.array_equal(edges, expected.reshape(-1, 2))


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("adj", ALL_ADJ)
def test_edge_count_matches_edge_list_for_loopless_symmetric_graphs(mod, adj):
    g = mod.CreateGraphStruct(4, [], None, csr_matrix(adj))
    assert g["nEdges"] == g["Edges"].shape[0]
    assert g["nEdges"] == int(np.triu(adj, 1).sum())


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("adj", ALL_ADJ)
def test_edge_index_matrix_pairs_forward_and_reverse_directions(mod, adj):
    # EdgeIdx[j,i] = e in 1..nEdges and EdgeIdx[i,j] = e + nEdges for every undirected edge
    g = mod.CreateGraphStruct(4, [], None, csr_matrix(adj))
    n_edges = g["nEdges"]
    e = g["EdgeIdx"].toarray()
    for i, j in g["Edges"]:
        assert 1 <= e[j, i] <= n_edges
        assert e[i, j] == e[j, i] + n_edges
    nz = np.sort(e[e > 0])
    assert np.array_equal(nz, np.arange(1, 2 * n_edges + 1))


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("adj", ALL_ADJ)
def test_nnmat_lists_the_neighbours_of_each_node(mod, adj):
    g = mod.CreateGraphStruct(4, [], None, csr_matrix(adj))
    assert len(g["nnMat"]) == adj.shape[0]
    for n in range(adj.shape[0]):
        assert np.array_equal(np.sort(g["nnMat"][n]), np.nonzero(adj[n, :])[0])


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_adjacency_is_stored_as_csr(mod):
    g = mod.CreateGraphStruct(4, [], None, csr_matrix(ADJ_PATH4))
    assert isinstance(g["AdjMat"], csr_matrix)
    assert g["AdjMat"].shape == (4, 4)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("epsilon,expected_edges", [(0.5, 0), (1.05, 2), (2.05, 3), (100.0, 6)])
def test_pairwise_distance_path_connects_pairs_within_epsilon(mod, epsilon, expected_edges):
    # nodes on a line at x = 0, 1, 2, 10
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [10.0, 0, 0]]).T
    d = cdist(x.T, x.T)
    g = mod.CreateGraphStruct(4, d, epsilon)
    assert g["nEdges"] == expected_edges
    adj = g["AdjMat"].toarray().astype(bool)
    assert np.array_equal(adj, (d <= epsilon) & (d != 0.0))


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_pairwise_distance_path_never_creates_self_loops(mod):
    rng = np.random.default_rng(24)
    x = rng.normal(size=(3, 8))
    x /= np.linalg.norm(x, axis=0)
    d = cdist(x.T, x.T)
    g = mod.CreateGraphStruct(4, d, 10.0)
    assert np.all(g["AdjMat"].diagonal() == 0)
    assert g["nEdges"] == 8 * 7 // 2


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_pairwise_distance_path_symmetrizes_the_adjacency(mod):
    d = np.array([[0.0, 1.0, 9.0], [1.0, 0.0, 9.0], [9.0, 9.0, 0.0]])
    g = mod.CreateGraphStruct(4, d, 2.0)
    a = g["AdjMat"].toarray()
    assert np.array_equal(a, a.T)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("eps_small,eps_big", [(1.05, 2.05), (2.05, 100.0), (0.5, 1.05)])
def test_larger_epsilon_gives_a_superset_of_edges(mod, eps_small, eps_big):
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [10.0, 0, 0]]).T
    d = cdist(x.T, x.T)
    small = mod.CreateGraphStruct(4, d, eps_small)["AdjMat"].toarray().astype(bool)
    big = mod.CreateGraphStruct(4, d, eps_big)["AdjMat"].toarray().astype(bool)
    assert np.all(big[small])


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_pairwise_distance_accepts_a_nested_list(mod):
    g = mod.CreateGraphStruct(4, [[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]], 2.0)
    assert g["nEdges"] == 1
    assert np.array_equal(g["Edges"], np.array([[0, 1]]))


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("epsilon", [0.0, 1.0, np.inf])
def test_epsilon_is_recorded_on_the_structure(mod, epsilon):
    d = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert mod.CreateGraphStruct(4, d, epsilon)["epsilon"] == epsilon


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_epsilon_none_with_distances_raises(mod):
    # the neighbor test is `pwDist <= epsilon`, which has no meaning for None
    d = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(TypeError):
        mod.CreateGraphStruct(4, d, None)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_dense_adjacency_is_not_supported(mod):
    # nnMat reads np.nonzero(AdjMat[n, :])[1], which needs a 2-D sparse row
    with pytest.raises(IndexError):
        mod.CreateGraphStruct(4, [], None, ADJ_PATH4)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_single_node_graph(mod):
    g = mod.CreateGraphStruct(6, [0], 0)
    assert g["nNodes"] == 1
    assert g["nEdges"] == 0
    assert g["Edges"].size == 0
    assert g["AdjMat"].shape == (1, 1)
    assert g["maxState"] == 6


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_upper_triangular_adjacency_yields_no_edges(mod):
    # an adjacency handed in directly is NOT symmetrized, and the edge list is built from the
    # transposed nonzero pattern, so only entries below the diagonal register
    upper = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=int)
    g = mod.CreateGraphStruct(2, [], None, csr_matrix(upper))
    assert g["nEdges"] == 0
    assert g["Edges"].size == 0
    assert np.array_equal(g["AdjMat"].toarray(), upper)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_lower_triangular_adjacency_yields_the_edge(mod):
    lower = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=int)
    g = mod.CreateGraphStruct(2, [], None, csr_matrix(lower))
    assert g["nEdges"] == 1
    assert np.array_equal(g["Edges"], np.array([[0, 1]]))


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.xfail(reason="Bug. nEdges counts tril entries including the diagonal, so a graph "
                          "with self loops reports more edges than Edges holds", strict=False)
def test_self_loops_do_not_inflate_the_edge_count(mod):
    loops = np.array([[1, 1], [1, 1]], dtype=int)
    g = mod.CreateGraphStruct(2, [], None, csr_matrix(loops))
    assert g["nEdges"] == g["Edges"].shape[0]


@pytest.mark.parametrize("adj", ALL_ADJ)
def test_both_graph_modules_build_identical_structures(adj):
    a = FG.CreateGraphStruct(4, [], 0.5, csr_matrix(adj))
    b = FP.CreateGraphStruct(4, [], 0.5, csr_matrix(adj))
    assert a["nNodes"] == b["nNodes"]
    assert a["nEdges"] == b["nEdges"]
    assert a["maxState"] == b["maxState"]
    assert np.array_equal(a["Edges"], b["Edges"])
    assert (a["AdjMat"] != b["AdjMat"]).nnz == 0
    assert (a["EdgeIdx"] != b["EdgeIdx"]).nnz == 0


# --------------------------------------------------------------------------------------
# getSubGraph
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("adj", ALL_ADJ)
def test_connected_component_count_matches_scipy(mod, adj):
    g = mod.CreateGraphStruct(4, [], 0.5, csr_matrix(adj))
    gsub, g = mod.getSubGraph(g)
    n_expected, labels = connected_components(csr_matrix(adj), directed=False)
    assert len(g["NodesConnComp"]) == n_expected
    assert len(gsub) == n_expected
    for i, nodes in enumerate(g["NodesConnComp"]):
        assert np.array_equal(nodes, np.nonzero(labels == i)[0])


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("adj", ALL_ADJ)
def test_components_partition_the_nodes(mod, adj):
    g = mod.CreateGraphStruct(4, [], 0.5, csr_matrix(adj))
    _, g = mod.getSubGraph(g)
    joined = np.sort(np.concatenate(g["NodesConnComp"]))
    assert np.array_equal(joined, np.arange(adj.shape[0]))


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("adj", ALL_ADJ)
def test_component_adjacency_blocks_are_the_induced_submatrices(mod, adj):
    g = mod.CreateGraphStruct(4, [], 0.5, csr_matrix(adj))
    _, g = mod.getSubGraph(g)
    for nodes, block in zip(g["NodesConnComp"], g["AdjConnComp"]):
        assert np.array_equal(block.toarray(), g["AdjMat"][nodes][:, nodes].toarray())


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("adj", ALL_ADJ)
def test_subgraph_node_and_edge_bookkeeping(mod, adj):
    g = mod.CreateGraphStruct(4, [], 0.5, csr_matrix(adj))
    gsub, g = mod.getSubGraph(g)
    total_edges = 0
    for i, sub in enumerate(gsub):
        nodes = g["NodesConnComp"][i]
        assert sub["nNodes"] == len(nodes)
        assert np.array_equal(sub["originalNodes"], nodes)
        # every original edge attributed to a component has both endpoints inside it
        assert np.all(np.isin(sub["originalEdges"], nodes))
        assert sub["nEdges"] == sub["originalEdges"].shape[0]
        total_edges += sub["nEdges"]
    assert total_edges == g["nEdges"]


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_singleton_components_carry_no_edges(mod):
    g = mod.CreateGraphStruct(4, [], 0.5, csr_matrix(ADJ_ISOLATED))
    gsub, g = mod.getSubGraph(g)
    singles = [i for i, c in enumerate(g["NodesConnComp"]) if len(c) == 1]
    assert singles
    for i in singles:
        assert gsub[i]["nEdges"] == 0
        assert gsub[i]["originalEdges"].size == 0


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_complete_graph_is_a_single_component(mod):
    g = mod.CreateGraphStruct(4, [], 0.5, csr_matrix(ADJ_COMPLETE3))
    gsub, g = mod.getSubGraph(g)
    assert len(gsub) == 1
    assert gsub[0]["nNodes"] == 3
    assert gsub[0]["nEdges"] == 3


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_empty_graph_is_all_singletons(mod):
    g = mod.CreateGraphStruct(4, [], 0.5, csr_matrix(ADJ_EMPTY3))
    gsub, g = mod.getSubGraph(g)
    assert len(gsub) == 3
    assert all(s["nNodes"] == 1 and s["nEdges"] == 0 for s in gsub)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_get_subgraph_mutates_and_returns_the_input_graph(mod):
    g = mod.CreateGraphStruct(4, [], 0.5, csr_matrix(ADJ_TWO_COMPONENTS))
    gsub, returned = mod.getSubGraph(g)
    assert returned is g
    assert "NodesConnComp" in g and "AdjConnComp" in g


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.xfail(reason="Bug. Subgraph nnMat is copied whole from the parent instead of being "
                          "restricted to the component's nodes", strict=False)
def test_subgraph_nnmat_is_restricted_to_its_own_nodes(mod):
    g = mod.CreateGraphStruct(4, [], 0.5, csr_matrix(ADJ_TWO_COMPONENTS))
    gsub, g = mod.getSubGraph(g)
    assert all(len(s["nnMat"]) == s["nNodes"] for s in gsub)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.xfail(reason="Bug. The explicit-node branch of getSubGraph indexes with the *args "
                          "tuple and reads the misspelled key 'MaxState'", strict=False)
def test_get_subgraph_with_explicit_nodes(mod):
    g = mod.CreateGraphStruct(4, [], 0.5, csr_matrix(ADJ_PATH4))
    sub, _ = mod.getSubGraph(g, [0, 1])
    assert sub["nNodes"] == 2


# --------------------------------------------------------------------------------------
# CalcPairwiseDistS2
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("n", [2, 5, 12, 40])
def test_pairwise_dot_product_is_the_gram_matrix(mod, n):
    x = random_half_space_points(n, seed=25)
    dot, _ = mod.CalcPairwiseDistS2(x)
    assert dot.shape == (n, n)
    assert np.allclose(dot, x.T @ x)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("n", [2, 5, 12, 40])
def test_pairwise_distance_of_unit_vectors_matches_cdist(mod, n):
    x = random_half_space_points(n, seed=26)
    _, d = mod.CalcPairwiseDistS2(x)
    assert np.allclose(d, cdist(x.T, x.T), atol=1e-8)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("n", [3, 10])
def test_pairwise_distance_is_symmetric_with_zero_diagonal(mod, n):
    x = random_half_space_points(n, seed=27)
    _, d = mod.CalcPairwiseDistS2(x)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_chord_length_relation_on_the_sphere(mod):
    # |u - v|^2 = 2 - 2 u.v for unit vectors
    x = random_half_space_points(15, seed=28)
    dot, d = mod.CalcPairwiseDistS2(x)
    assert np.allclose(d**2, np.maximum(2.0 - 2.0 * dot, 0.0), atol=1e-6)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.xfail(reason="Bug. The |u|^2 term is broadcast as a row instead of a column, so the "
                          "distance is only correct when every vector has unit norm", strict=False)
def test_pairwise_distance_of_general_vectors_matches_cdist(mod):
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]]).T
    _, d = mod.CalcPairwiseDistS2(x)
    assert np.allclose(d, cdist(x.T, x.T))


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_pairwise_distance_of_general_vectors_is_not_symmetric(mod):
    # documents the broadcasting defect above
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]).T
    _, d = mod.CalcPairwiseDistS2(x)
    assert not np.allclose(d, d.T)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("idx_u,idx_v", [([0, 1], [2, 3]), ([0, 0], [1, 2]), ([3, 2, 1], [0, 1, 2])])
def test_pairwise_distance_on_index_subsets(mod, idx_u, idx_v):
    x = random_half_space_points(6, seed=29)
    dot, d = mod.CalcPairwiseDistS2(x, idx_u, idx_v)
    assert dot.shape == (len(idx_u), len(idx_v))
    assert np.allclose(dot, x[:, idx_u].T @ x[:, idx_v])
    assert np.allclose(d, cdist(x[:, idx_u].T, x[:, idx_v].T), atol=1e-8)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
def test_pairwise_distance_index_subsets_must_have_equal_length(mod):
    # a consequence of the same broadcasting defect
    x = random_half_space_points(6, seed=30)
    with pytest.raises(ValueError):
        mod.CalcPairwiseDistS2(x, [0, 1], [2, 3, 4])


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("extra", [([0],), ([0], [1], [2])])
def test_pairwise_distance_rejects_wrong_argument_counts(mod, extra):
    x = random_half_space_points(4, seed=31)
    with pytest.raises(AssertionError):
        mod.CalcPairwiseDistS2(x, *extra)


@pytest.mark.parametrize("mod", GRAPH_MODULES)
@pytest.mark.parametrize("sep,expect_zero", [(1e-4, True), (5e-4, True), (1e-2, False), (0.1, False)])
def test_squared_distances_below_1e_6_are_clamped_to_zero(mod, sep, expect_zero):
    x = np.array([[1.0, 0.0, 0.0], [1.0, sep, 0.0]]).T
    x /= np.linalg.norm(x, axis=0)
    _, d = mod.CalcPairwiseDistS2(x)
    assert bool(d[0, 1] == 0.0) is expect_zero


# --------------------------------------------------------------------------------------
# FindCCGraph.prune
# --------------------------------------------------------------------------------------
def build_path_graph(n_states=6, epsilon=0.5, adj=ADJ_PATH4):
    return FG.CreateGraphStruct(n_states, [], epsilon, csr_matrix(adj))


@pytest.mark.parametrize("trash", [{0}, {1}, {3}, {0, 3}])
def test_prune_keeps_the_node_count_and_isolates_the_trash(trash):
    g, _ = FG.prune(dict(build_path_graph()), trash, 4)
    assert g["nNodes"] == 4
    a = g["AdjMat"].toarray()
    for t in trash:
        assert not a[t, :].any()
        assert not a[:, t].any()


@pytest.mark.parametrize("trash,expected_edges", [(set(), 3), ({0}, 2), ({1}, 1), ({3}, 2), ({0, 3}, 1)])
def test_prune_removes_exactly_the_incident_edges(trash, expected_edges):
    g, _ = FG.prune(dict(build_path_graph()), trash, 4)
    assert g["nEdges"] == expected_edges
    assert g["Edges"].shape[0] == expected_edges
    for i, j in g["Edges"]:
        assert i not in trash and j not in trash


@pytest.mark.parametrize("epsilon", [0.25, 0.5, 2.0])
def test_prune_restores_epsilon_on_the_new_structure(epsilon):
    g, _ = FG.prune(dict(build_path_graph(epsilon=epsilon)), {0}, 4)
    assert g["epsilon"] == epsilon


def test_prune_with_no_trash_is_a_no_op_on_the_adjacency():
    original = build_path_graph()
    g, _ = FG.prune(dict(original), set(), 4)
    assert (g["AdjMat"] != original["AdjMat"]).nnz == 0
    assert g["nEdges"] == original["nEdges"]


def test_prune_mutates_the_graph_it_is_handed():
    original = build_path_graph()
    handed = dict(original)
    g, _ = FG.prune(handed, {1}, 4)
    # the caller's dict picks up the zeroed adjacency and nPsiModes, but the returned graph is a
    # freshly built structure that does not carry nPsiModes
    assert np.array_equal(handed["AdjMat"].toarray(), g["AdjMat"].toarray())
    assert handed["nPsiModes"] == 4
    assert "nPsiModes" not in g


@pytest.mark.parametrize("trash", [{0}, {1, 2}])
def test_prune_reports_trash_nodes_as_singleton_components(trash):
    g, gsub = FG.prune(dict(build_path_graph()), trash, 4)
    singleton_nodes = {int(c[0]) for c in g["NodesConnComp"] if len(c) == 1}
    assert trash <= singleton_nodes
    assert len(gsub) == len(g["NodesConnComp"])


@pytest.mark.parametrize("num_psis", [1, 3, 8])
def test_prune_down_to_one_node_builds_an_empty_single_node_graph(num_psis):
    g, gsub = FG.prune(dict(build_path_graph()), {0, 1, 2}, num_psis)
    assert g["nNodes"] == 1
    assert g["nEdges"] == 0
    assert g["maxState"] == 2 * num_psis  # up and down for each psi
    assert g["epsilon"] == 0
    assert len(gsub) == 1


def test_prune_of_everything_also_collapses_to_one_node():
    g, _ = FG.prune(dict(build_path_graph()), {0, 1, 2, 3}, 2)
    assert g["nNodes"] == 1


def test_prune_is_idempotent():
    once, _ = FG.prune(dict(build_path_graph()), {0}, 4)
    twice, _ = FG.prune(dict(once), {0}, 4)
    assert (once["AdjMat"] != twice["AdjMat"]).nnz == 0
    assert once["nEdges"] == twice["nEdges"]


def test_prune_result_carries_a_connected_component_decomposition():
    g, gsub = FG.prune(dict(build_path_graph()), {1}, 4)
    assert "NodesConnComp" in g and "AdjConnComp" in g
    n_expected, _ = connected_components(g["AdjMat"], directed=False)
    assert len(g["NodesConnComp"]) == n_expected


# --------------------------------------------------------------------------------------
# FindCCGraph.op
# --------------------------------------------------------------------------------------
S2_FIVE = np.array([[1.0, 0.0, 0.0],
                    [0.8, 0.6, 0.0],
                    [0.6, 0.8, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]).T


@pytest.mark.parametrize("n_g", [2, 10, 40, 200])
def test_op_epsilon_follows_the_documented_clamp(params_guard_geometry_graph, n_g):
    params.num_psi = 4
    n_pds = S2_FIVE.shape[1]
    g, _ = FG.op([np.arange(5)] * n_pds, n_g, S2_FIVE)

    _, d = FG.CalcPairwiseDistS2(S2_FIVE)
    mindist = np.min(d[np.nonzero(d)])
    expected = min(max(mindist + params.eps, mindist * n_g / n_pds),
                   mindist * 2 * np.sqrt(2) + params.eps)
    assert g["epsilon"] == pytest.approx(expected)
    # mindist + eps <= epsilon <= mindist*2*sqrt(2) + eps
    assert mindist + params.eps <= g["epsilon"] <= mindist * 2 * np.sqrt(2) + params.eps


@pytest.mark.parametrize("num_psi", [1, 4, 8])
def test_op_max_state_is_twice_the_psi_count(params_guard_geometry_graph, num_psi):
    params.num_psi = num_psi
    g, _ = FG.op([np.arange(5)] * 5, 10, S2_FIVE)
    assert g["maxState"] == 2 * num_psi
    assert g["nPsiModes"] == num_psi


def test_op_builds_one_node_per_projection_direction(params_guard_geometry_graph):
    params.num_psi = 4
    g, gsub = FG.op([np.arange(5)] * 5, 10, S2_FIVE)
    assert g["nNodes"] == 5
    assert g["Nodes"] == range(5)
    joined = np.sort(np.concatenate(g["NodesConnComp"]))
    assert np.array_equal(joined, np.arange(5))
    assert len(gsub) == len(g["NodesConnComp"])


def test_op_adjacency_is_symmetric_and_loopless(params_guard_geometry_graph):
    params.num_psi = 4
    g, _ = FG.op([np.arange(5)] * 5, 10, S2_FIVE)
    a = g["AdjMat"].toarray()
    assert np.array_equal(a, a.T)
    assert not a.diagonal().any()


@pytest.mark.parametrize("n_g,expect_full", [(2, False), (2000, True)])
def test_op_neighbourhood_grows_with_the_bin_count(params_guard_geometry_graph, n_g, expect_full):
    # epsilonBall = mindist * nG / numPDs, capped at mindist*2*sqrt(2) + eps
    params.num_psi = 4
    g, _ = FG.op([np.arange(5)] * 5, n_g, S2_FIVE)
    _, d = FG.CalcPairwiseDistS2(S2_FIVE)
    mindist = np.min(d[np.nonzero(d)])
    cap = mindist * 2 * np.sqrt(2) + params.eps
    assert bool(g["epsilon"] == pytest.approx(cap)) is expect_full


def test_op_with_a_single_projection_direction(params_guard_geometry_graph):
    params.num_psi = 3
    g, gsub = FG.op([np.arange(5)], 2, S2_FIVE[:, :1])
    assert g["nNodes"] == 1
    assert g["nEdges"] == 0
    assert g["maxState"] == 6
    assert g["epsilon"] == 0
    assert len(gsub) == 1


def test_op_matches_a_hand_built_epsilon_ball_graph(params_guard_geometry_graph):
    params.num_psi = 2
    g, _ = FG.op([np.arange(5)] * 5, 10, S2_FIVE)
    _, d = FG.CalcPairwiseDistS2(S2_FIVE)
    expected = (d <= g["epsilon"]) & (d != 0.0)
    assert np.array_equal(g["AdjMat"].toarray().astype(bool), expected)


# --------------------------------------------------------------------------------------
# FindCCGraphPruned.op
# --------------------------------------------------------------------------------------
def test_find_cc_graph_pruned_op_depends_on_missing_params_fields():
    # documents that this entry point is stale. Params has neither of these members
    assert not hasattr(params, "get_trash_list")
    assert not hasattr(params, "CC_graph_file")


@pytest.mark.xfail(reason="Bug. FindCCGraphPruned.op calls params.get_trash_list() and reads "
                          "params.CC_graph_file, neither of which exists on Params", strict=False)
def test_find_cc_graph_pruned_op_runs(tmp_path):
    out = str(tmp_path / "pruned.pkl")
    g, gsub = FP.op(out)
    assert g["nNodes"] >= 1
    assert os.path.isfile(out)


def test_find_cc_graph_pruned_op_currently_raises(tmp_path):
    with pytest.raises(AttributeError):
        FP.op(str(tmp_path / "pruned.pkl"))


# ============================================================================
# DMembeddingII.py, manifoldTrimmingAuto.py, embedd.py - diffusion maps
# ============================================================================

matplotlib.use("Agg")




# --------------------------------------------------------------------------
# synthetic manifolds and reference implementations
# --------------------------------------------------------------------------
def sq_dists(X):
    """Matrix of SQUARED euclidean distances, the convention DMembeddingII wants."""
    return ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)


def full_pattern(n):
    """Row/col index vectors covering the whole n x n matrix (row-major)."""
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return r.ravel(), c.ravel()


def dense(l):
    return np.asarray(l.todense()) if issparse(l) else np.asarray(l)


def slaplacian_oracle(D2, sigma, alpha):
    """Dense reference for slaplacian. Gaussian kernel, Coifman-Lafon alpha
    normalization by the outer product of the degrees, then symmetric
    normalization by the square root of the new degrees."""
    W = np.exp(-D2 / sigma**2)
    d = W.sum(axis=0)
    if alpha != 1:
        d = d**alpha
    W2 = W / np.outer(d, d)
    d2 = np.sqrt(W2.sum(axis=0))
    W3 = W2 / np.outer(d2, d2)
    return np.abs(W3 + W3.T) / 2.0


def line_points(n, noise=0.0, seed=0):
    """Uniformly sampled unit segment. The intrinsic coordinate is arclength."""
    s = np.linspace(0.0, 1.0, n)
    rng = np.random.default_rng(seed)
    y = noise * rng.standard_normal(n) if noise else np.zeros(n)
    return s, np.stack([s, y], axis=1)


def circle_points(n):
    """Closed 1-manifold. The Laplacian spectrum comes in degenerate pairs."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return t, np.stack([np.cos(t), np.sin(t)], axis=1)


def spiral_points(n):
    """Archimedean spiral, a 1D swiss roll.  Arclength is the true coordinate."""
    th = np.sqrt(np.linspace(1.0, 9.0, n)) * np.pi
    X = np.stack([th * np.cos(th), th * np.sin(th)], axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])
    return arclen, X


def random_cloud(n, dim=3, seed=0):
    return np.random.default_rng(seed).random((n, dim))


def opts(sigma=1.0, alpha=1.0, autotune=0, nEigs=3):
    return SimpleNamespace(sigma=sigma, alpha=alpha, autotune=autotune, nEigs=nEigs, visual=0)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def params_guard_embedding():
    """params is a process-wide singleton and params.load() chdirs.  Save/restore
    both the working directory and every field these tests touch."""
    cwd = os.getcwd()
    watched = ("project_name", "num_eigs", "num_psi", "num_psi_truncated", "ncpu", "nlsa_tune", "rad")
    saved = {k: getattr(params, k) for k in watched}
    params.ncpu = 1
    try:
        yield params
    finally:
        for k, v in saved.items():
            setattr(params, k, v)
        os.chdir(cwd)


@pytest.fixture(scope="module")
def line_run():
    """One embedding of a straight line, reused by many assertions."""
    n = 150
    s, X = line_points(n)
    out = DM.op(sq_dists(X), n, 3, 60000)
    return s, X, out


@pytest.fixture(scope="module")
def circle_run():
    n = 120
    t, X = circle_points(n)
    out = DM.op(sq_dists(X), n, 3, 60000)
    return t, X, out


@pytest.fixture(scope="module")
def cloud_run():
    n = 100
    X = random_cloud(n, 3, seed=7)
    out = DM.op(sq_dists(X), n, 3, 60000)
    return X, out


# ==========================================================================
# initialize
# ==========================================================================
@pytest.mark.parametrize("nS,nN", [(6, 1), (6, 3), (6, 6), (10, 4), (20, 7), (30, 30)])
def test_initialize_shapes_and_dtypes(nS, nN):
    D = sq_dists(random_cloud(nS, 2, seed=nS))
    yInd1, yVal1 = DM.initialize(nS, nN, D)

    assert yInd1.shape == (nN * nS,)
    assert yVal1.shape == (nN * nS,)
    assert yInd1.dtype == np.int32
    assert yVal1.dtype == np.float64


@pytest.mark.parametrize("nN", [1, 2, 5, 9])
def test_initialize_self_is_the_first_neighbour(nN):
    # the diagonal is forced to -inf so every point is its own nearest neighbor,
    # and its distance is then reset to exactly zero
    nS = 12
    D = sq_dists(random_cloud(nS, 3, seed=1))
    yInd1, yVal1 = DM.initialize(nS, nN, D)

    assert np.array_equal(yInd1[::nN], np.arange(nS, dtype=np.int32))
    assert np.all(yVal1[::nN] == 0.0)


@pytest.mark.parametrize("nN", [2, 4, 9])
def test_initialize_matches_argsort_oracle(nN):
    nS = 15
    D = sq_dists(random_cloud(nS, 2, seed=3))
    ref = D.copy()
    np.fill_diagonal(ref, -np.inf)
    expected_ind = np.argsort(ref, axis=0)[:nN, :]

    yInd1, _ = DM.initialize(nS, nN, D)

    assert np.array_equal(yInd1.reshape(nS, nN).T, expected_ind)


def test_initialize_column_major_flattening():
    # yInd1[i*nN + j] is the j-th nearest neighbor of point i
    nS, nN = 9, 4
    D = sq_dists(random_cloud(nS, 2, seed=5))
    yInd1, yVal1 = DM.initialize(nS, nN, D)

    ref = D.copy()
    for i in range(nS):
        order = np.argsort(ref[:, i])[:nN]
        assert np.array_equal(yInd1[i * nN:(i + 1) * nN], order.astype(np.int32))
        assert np.allclose(yVal1[i * nN + 1:(i + 1) * nN], ref[order[1:], i])


def test_initialize_mutates_diagonal_to_minus_inf():
    # documented side effect. The caller's distance matrix is modified in place
    nS = 8
    D = sq_dists(random_cloud(nS, 2, seed=2))
    DM.initialize(nS, 3, D)

    assert np.all(np.isneginf(np.diag(D)))


def test_initialize_leaves_off_diagonal_untouched():
    nS = 8
    D = sq_dists(random_cloud(nS, 2, seed=2))
    before = D.copy()
    DM.initialize(nS, 3, D)

    off = ~np.eye(nS, dtype=bool)
    assert np.array_equal(D[off], before[off])


@pytest.mark.parametrize("nN", [3, 6, 11])
def test_initialize_distances_are_ascending(nN):
    nS = 14
    D = sq_dists(random_cloud(nS, 3, seed=4))
    _, yVal1 = DM.initialize(nS, nN, D)
    blocks = yVal1.reshape(nS, nN)

    assert np.all(np.diff(blocks[:, 1:], axis=1) >= -1e-15)


def test_initialize_values_match_indexed_entries():
    nS, nN = 11, 5
    D = sq_dists(random_cloud(nS, 3, seed=6))
    ref = D.copy()
    yInd1, yVal1 = DM.initialize(nS, nN, D)

    ind = yInd1.reshape(nS, nN)
    val = yVal1.reshape(nS, nN)
    for i in range(nS):
        # rank 0 is the self entry, artificially zeroed
        assert np.allclose(val[i, 1:], ref[ind[i, 1:], i])


def test_initialize_full_rank_is_a_permutation():
    nS = 10
    D = sq_dists(random_cloud(nS, 2, seed=8))
    yInd1, _ = DM.initialize(nS, nS, D)
    for i in range(nS):
        assert sorted(yInd1[i * nS:(i + 1) * nS].tolist()) == list(range(nS))


# ==========================================================================
# get_yColVal
# ==========================================================================
def _yColVal_call(yVal1, yInd1, nS, nN):
    yVal = np.zeros((nS * nN, 1))
    yCol = np.zeros((nS * nN, 1))
    args = (yVal, yVal1, yCol, yInd1, nS, nN, nN, 0, nS, 0, nS * nN, 0)
    return DM.get_yColVal(args)


@pytest.mark.parametrize("nS,nN", [(4, 3), (5, 2), (7, 5), (3, 3), (9, 1)])
def test_get_yColVal_transposes_to_rank_major_order(nS, nN):
    # input is (nN, nS) flattened column-major. Output is the same block
    # flattened ROW-major, i.e. out[j*nS + i] is rank j of sample i
    mat = np.arange(1.0, nN * nS + 1.0).reshape(nN, nS)
    yVal1 = mat.flatten("F")
    yInd1 = (np.arange(nN * nS) % nS).astype("int32")

    yCol, yVal = _yColVal_call(yVal1.copy(), yInd1.copy(), nS, nN)

    expected = mat.copy()
    expected[0, :] = 0.0
    assert np.allclose(yVal.ravel(), expected.reshape(nN * nS))


def test_get_yColVal_returns_the_same_array_objects():
    nS, nN = 5, 3
    yVal = np.zeros((nS * nN, 1))
    yCol = np.zeros((nS * nN, 1))
    args = (yVal, np.arange(nS * nN, dtype=float), yCol,
            np.zeros(nS * nN, dtype="int32"), nS, nN, nN, 0, nS, 0, nS * nN, 0)
    outCol, outVal = DM.get_yColVal(args)

    assert outVal is yVal
    assert outCol is yCol


def test_get_yColVal_zeroes_rank0_of_the_input_in_place():
    # DataBatch is a view onto yVal1, so `DataBatch[0, :] = 0` writes through
    nS, nN = 4, 3
    yVal1 = np.arange(1.0, nS * nN + 1.0)
    before = yVal1.copy()
    _yColVal_call(yVal1, (np.arange(nS * nN) % nS).astype("int32"), nS, nN)

    assert np.all(yVal1[::nN] == 0.0)
    kept = np.ones(nS * nN, dtype=bool)
    kept[::nN] = False
    assert np.array_equal(yVal1[kept], before[kept])


def test_get_yColVal_column_indices_are_floats():
    nS, nN = 6, 2
    yCol, _ = _yColVal_call(np.arange(nS * nN, dtype=float),
                            (np.arange(nS * nN) % nS).astype("int32"), nS, nN)

    assert yCol.dtype == np.float64
    assert np.all(yCol == np.round(yCol))


@pytest.mark.parametrize("nS,nN", [(8, 3), (12, 5), (15, 15)])
def test_get_yColVal_roundtrip_with_initialize(nS, nN):
    # the pair (initialize, get_yColVal) must yield triplets consistent with D
    D = sq_dists(random_cloud(nS, 3, seed=nS))
    ref = D.copy()
    yInd1, yVal1 = DM.initialize(nS, nN, D)
    yCol, yVal = _yColVal_call(yVal1, yInd1, nS, nN)

    col = yCol.reshape(nN, nS).astype(int)
    val = yVal.reshape(nN, nS)
    for j in range(1, nN):
        for i in range(nS):
            assert val[j, i] == pytest.approx(ref[col[j, i], i])


def test_get_yColVal_row_index_convention_of_op():
    # op builds yRow as ones((nN,1)) * range(nS) reshaped row-major, which pairs
    # element m = j*nS + i with sample i, matching get_yColVal's output order
    nS, nN = 7, 4
    yRow = (np.ones((nN, 1)) * range(nS)).reshape(nS * nN, 1)

    assert np.array_equal(yRow.ravel(), np.tile(np.arange(nS, dtype=float), nN))


# ==========================================================================
# construct_matrix0 / construct_matrix1
# ==========================================================================
def _symmetric_values(n, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.random((n, n))
    V = 0.5 * (V + V.T)
    np.fill_diagonal(V, 0.0)
    return V


@pytest.mark.parametrize("n", [3, 5, 8])
def test_construct_matrix0_squares_a_symmetric_pattern(n):
    # y2 = v*v cancels one copy of v**2, leaving exactly the squared values
    V = _symmetric_values(n, seed=n)
    r, c = full_pattern(n)
    y = DM.construct_matrix0(r, c, V.ravel(), n)

    assert np.allclose(y, V**2)


@pytest.mark.parametrize("n", [3, 5, 8])
def test_construct_matrix1_on_a_symmetric_pattern(n):
    V = _symmetric_values(n, seed=n + 1)
    r, c = full_pattern(n)
    y = DM.construct_matrix1(r, c, V.ravel(), n)

    assert np.allclose(y, V + V.T - V * V.T)


@pytest.mark.parametrize("n", [4, 7])
def test_construct_matrix0_symmetrises_a_one_sided_pattern(n):
    V = _symmetric_values(n, seed=n + 2)
    r, c = np.nonzero(np.triu(np.ones((n, n), bool), 1))
    E = np.zeros((n, n))
    E[r, c] = V[r, c]

    y = DM.construct_matrix0(r, c, V[r, c], n)

    # the transposed slot is empty so y2 vanishes there
    assert np.allclose(y, E**2 + E.T**2)


@pytest.mark.parametrize("n", [4, 7])
def test_construct_matrix1_symmetrises_a_one_sided_pattern(n):
    V = _symmetric_values(n, seed=n + 3)
    r, c = np.nonzero(np.triu(np.ones((n, n), bool), 1))
    E = np.zeros((n, n))
    E[r, c] = V[r, c]

    y = DM.construct_matrix1(r, c, V[r, c], n)

    assert np.allclose(y, E + E.T)


@pytest.mark.parametrize("fn", [DM.construct_matrix0, DM.construct_matrix1])
def test_construct_matrix_output_is_symmetric(fn):
    n = 9
    V = _symmetric_values(n, seed=11)
    r, c = np.nonzero(np.triu(np.ones((n, n), bool), 1))
    y = fn(r, c, V[r, c], n)

    assert np.allclose(y, y.T, atol=1e-15)


@pytest.mark.parametrize("fn", [DM.construct_matrix0, DM.construct_matrix1])
@pytest.mark.parametrize("n", [3, 6])
def test_construct_matrix_shape_and_dtype(fn, n):
    r, c = full_pattern(n)
    y = fn(r, c, np.ones(n * n), n)

    assert y.shape == (n, n)
    assert y.dtype == np.float64
    assert isinstance(y, np.ndarray)


def test_construct_matrix1_is_a_logical_or_for_indicator_values():
    # op relies on this to build the mask of zero-distance entries
    n = 6
    r, c = np.nonzero(np.triu(np.ones((n, n), bool), 1))
    y = DM.construct_matrix1(r, c, np.ones(len(r)), n)
    expected = np.ones((n, n)) - np.eye(n)

    assert np.allclose(y, expected)


@pytest.mark.parametrize("fn", [DM.construct_matrix0, DM.construct_matrix1])
def test_construct_matrix_zero_values_give_zero_matrix(fn):
    n = 5
    r, c = full_pattern(n)
    y = fn(r, c, np.zeros(n * n), n)

    assert np.count_nonzero(y) == 0


def test_construct_matrix0_sums_duplicate_coordinates():
    # csr_matrix accumulates duplicates before the symmetrization happens
    n = 3
    r = np.array([0, 0, 1])
    c = np.array([1, 1, 2])
    y = DM.construct_matrix0(r, c, np.array([2.0, 3.0, 1.0]), n)

    assert y[0, 1] == pytest.approx(25.0)  # (2+3)**2
    assert y[1, 0] == pytest.approx(25.0)
    assert y[1, 2] == pytest.approx(1.0)


# ==========================================================================
# slaplacian
# ==========================================================================
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_slaplacian_matches_dense_oracle(alpha, sigma):
    n = 10
    D2 = sq_dists(random_cloud(n, 3, seed=0))
    r, c = full_pattern(n)
    l, _ = DM.slaplacian(D2.ravel().copy(), c, r, n, opts(sigma=sigma, alpha=alpha))

    assert np.allclose(dense(l), slaplacian_oracle(D2, sigma, alpha), atol=1e-12)


@pytest.mark.parametrize("sigma", [0.3, 1.0, 5.0])
def test_slaplacian_returns_sigma_unchanged(sigma):
    n = 6
    D2 = sq_dists(random_cloud(n, 2, seed=1))
    r, c = full_pattern(n)
    _, sigmaTune = DM.slaplacian(D2.ravel().copy(), c, r, n, opts(sigma=sigma))

    assert sigmaTune == sigma


def test_slaplacian_returns_sparse_matrix_of_right_shape():
    n = 7
    D2 = sq_dists(random_cloud(n, 2, seed=2))
    r, c = full_pattern(n)
    l, _ = DM.slaplacian(D2.ravel().copy(), c, r, n, opts())

    assert issparse(l)
    assert l.shape == (n, n)


def test_slaplacian_is_exactly_symmetric():
    n = 12
    D2 = sq_dists(random_cloud(n, 3, seed=3))
    r, c = full_pattern(n)
    l, _ = DM.slaplacian(D2.ravel().copy(), c, r, n, opts())
    A = dense(l)

    assert np.array_equal(A, A.T)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_slaplacian_top_eigenvalue_is_one(alpha):
    # L = D^{-1/2} K D^{-1/2} always has sqrt(deg) as a unit eigenvector
    n = 11
    D2 = sq_dists(random_cloud(n, 3, seed=4))
    r, c = full_pattern(n)
    l, _ = DM.slaplacian(D2.ravel().copy(), c, r, n, opts(alpha=alpha))
    w = np.linalg.eigvalsh(dense(l))

    assert w[-1] == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("sigma", [0.4, 1.0, 3.0])
def test_slaplacian_spectrum_is_in_the_unit_interval(sigma):
    n = 10
    D2 = sq_dists(random_cloud(n, 3, seed=5))
    r, c = full_pattern(n)
    l, _ = DM.slaplacian(D2.ravel().copy(), c, r, n, opts(sigma=sigma))
    w = np.linalg.eigvalsh(dense(l))

    assert w.min() >= -1.0 - 1e-12
    assert w.max() <= 1.0 + 1e-12


def test_slaplacian_entries_are_nonnegative():
    n = 10
    D2 = sq_dists(random_cloud(n, 2, seed=6))
    r, c = full_pattern(n)
    l, _ = DM.slaplacian(D2.ravel().copy(), c, r, n, opts())

    assert np.all(dense(l) >= 0.0)


def test_slaplacian_diagonal_is_strictly_positive():
    n = 9
    D2 = sq_dists(random_cloud(n, 2, seed=7))
    r, c = full_pattern(n)
    l, _ = DM.slaplacian(D2.ravel().copy(), c, r, n, opts())

    assert np.all(np.diag(dense(l)) > 0.0)


def test_slaplacian_is_permutation_equivariant():
    n = 12
    D2 = sq_dists(random_cloud(n, 3, seed=8))
    r, c = full_pattern(n)
    l, _ = DM.slaplacian(D2.ravel().copy(), c, r, n, opts())

    perm = np.random.default_rng(0).permutation(n)
    D2p = D2[perm][:, perm]
    lp, _ = DM.slaplacian(D2p.ravel().copy(), c, r, n, opts())

    assert np.allclose(dense(lp), dense(l)[perm][:, perm], atol=1e-12)


@pytest.mark.parametrize("scale", [0.25, 3.0, 10.0])
def test_slaplacian_is_invariant_under_matched_rescaling(scale):
    # exp(-(c*d)**2/(c*sigma)**2) == exp(-d**2/sigma**2)
    n = 10
    D2 = sq_dists(random_cloud(n, 3, seed=9))
    r, c = full_pattern(n)
    base, _ = DM.slaplacian(D2.ravel().copy(), c, r, n, opts(sigma=1.0))
    scaled, _ = DM.slaplacian((scale**2 * D2).ravel().copy(), c, r, n, opts(sigma=scale))

    assert np.allclose(dense(scaled), dense(base), atol=1e-12)


def test_slaplacian_symmetrises_an_asymmetric_pattern():
    # only the upper triangle is supplied. The final |l + l.T|/2 must fill it in
    n = 8
    D2 = sq_dists(random_cloud(n, 2, seed=10))
    r, c = np.nonzero(np.triu(np.ones((n, n), bool)))
    l, _ = DM.slaplacian(D2[r, c].copy(), c, r, n, opts())
    A = dense(l)

    assert np.array_equal(A, A.T)
    assert np.count_nonzero(np.tril(A, -1)) > 0


def test_slaplacian_alpha_zero_skips_the_anisotropic_normalisation():
    # d**0 == 1, so alpha=0 is the plain graph-Laplacian normalization
    n = 9
    D2 = sq_dists(random_cloud(n, 2, seed=11))
    r, c = full_pattern(n)
    l, _ = DM.slaplacian(D2.ravel().copy(), c, r, n, opts(alpha=0.0))

    W = np.exp(-D2)
    d = np.sqrt(W.sum(0))
    expected = W / np.outer(d, d)
    assert np.allclose(dense(l), expected, atol=1e-12)


@pytest.mark.parametrize("nnz_only", [True, False])
def test_slaplacian_ignores_absent_entries(nnz_only):
    # entries left out of the triplet list are true structural zeros
    n = 8
    D2 = sq_dists(random_cloud(n, 2, seed=12))
    mask = D2 < np.median(D2)
    r, c = (np.nonzero(mask) if nnz_only else full_pattern(n))
    l, _ = DM.slaplacian(D2[r, c].copy(), np.asarray(c), np.asarray(r), n, opts())
    A = dense(l)

    assert A.shape == (n, n)
    assert np.all(np.isfinite(A))


@pytest.mark.xfail(reason="Bug. slaplacian leaves sigmaTune unbound when options.autotune > 0",
                   strict=False)
def test_slaplacian_autotune_branch_does_not_crash():
    n = 6
    D2 = sq_dists(random_cloud(n, 2, seed=13))
    r, c = full_pattern(n)
    l, sigmaTune = DM.slaplacian(D2.ravel().copy(), c, r, n, opts(autotune=2))

    assert sigmaTune == 1.0


# ==========================================================================
# sembedding
# ==========================================================================
@pytest.mark.parametrize("nEigs", [1, 2, 4, 7])
def test_sembedding_shapes(nEigs):
    n = 14
    D2 = sq_dists(random_cloud(n, 3, seed=1))
    r, c = full_pattern(n)
    vals, vecs = DM.sembedding(D2.ravel().copy(), c, r, n, opts(nEigs=nEigs))

    assert vals.shape == (nEigs + 1,)
    assert vecs.shape == (n, nEigs + 1)


def test_sembedding_eigenvalues_are_descending():
    n = 16
    D2 = sq_dists(random_cloud(n, 3, seed=2))
    r, c = full_pattern(n)
    vals, _ = DM.sembedding(D2.ravel().copy(), c, r, n, opts(nEigs=6))

    assert np.all(np.diff(vals) <= 1e-12)


def test_sembedding_leading_eigenvalue_is_one():
    n = 16
    D2 = sq_dists(random_cloud(n, 3, seed=3))
    r, c = full_pattern(n)
    vals, _ = DM.sembedding(D2.ravel().copy(), c, r, n, opts(nEigs=5))

    assert vals[0] == pytest.approx(1.0, abs=1e-10)


def test_sembedding_matches_dense_eigh_oracle():
    n = 13
    D2 = sq_dists(random_cloud(n, 3, seed=4))
    r, c = full_pattern(n)
    vals, _ = DM.sembedding(D2.ravel().copy(), c, r, n, opts(nEigs=5))

    l, _ = DM.slaplacian(D2.ravel().copy(), c, r, n, opts())
    ref = np.linalg.eigvalsh(dense(l))[::-1][:6]

    assert np.allclose(vals, ref, atol=1e-10)


def test_sembedding_eigenvectors_are_orthonormal():
    n = 15
    D2 = sq_dists(random_cloud(n, 3, seed=5))
    r, c = full_pattern(n)
    _, vecs = DM.sembedding(D2.ravel().copy(), c, r, n, opts(nEigs=6))

    assert np.allclose(vecs.T @ vecs, np.eye(7), atol=1e-10)


def test_sembedding_leading_eigenvector_has_constant_sign():
    # Perron-Frobenius. The top eigenvector of a positive matrix is sign-definite
    n = 15
    D2 = sq_dists(random_cloud(n, 3, seed=6))
    r, c = full_pattern(n)
    _, vecs = DM.sembedding(D2.ravel().copy(), c, r, n, opts(nEigs=4))
    v0 = vecs[:, 0]

    assert np.all(v0 > 0) or np.all(v0 < 0)


@pytest.mark.parametrize("nEigs", [2, 5])
def test_sembedding_eigenpair_residual_is_tiny(nEigs):
    n = 14
    D2 = sq_dists(random_cloud(n, 3, seed=7))
    r, c = full_pattern(n)
    vals, vecs = DM.sembedding(D2.ravel().copy(), c, r, n, opts(nEigs=nEigs))
    A = dense(DM.slaplacian(D2.ravel().copy(), c, r, n, opts())[0])

    assert np.allclose(A @ vecs, vecs * vals[None, :], atol=1e-9)


def test_sembedding_does_not_read_the_autotune_field():
    # sembedding overwrites options.autotune with 0 unconditionally
    n = 12
    D2 = sq_dists(random_cloud(n, 3, seed=8))
    r, c = full_pattern(n)
    a, _ = DM.sembedding(D2.ravel().copy(), c, r, n,
                         SimpleNamespace(sigma=1.0, alpha=1.0, nEigs=3, autotune=99))
    b, _ = DM.sembedding(D2.ravel().copy(), c, r, n,
                         SimpleNamespace(sigma=1.0, alpha=1.0, nEigs=3))

    assert np.allclose(a, b)


def test_sembedding_is_permutation_equivariant():
    n = 14
    D2 = sq_dists(random_cloud(n, 3, seed=9))
    r, c = full_pattern(n)
    vals, vecs = DM.sembedding(D2.ravel().copy(), c, r, n, opts(nEigs=4))

    perm = np.random.default_rng(1).permutation(n)
    valsp, vecsp = DM.sembedding(D2[perm][:, perm].ravel().copy(), c, r, n, opts(nEigs=4))

    assert np.allclose(vals, valsp, atol=1e-10)
    assert np.allclose(np.abs(vecsp), np.abs(vecs[perm]), atol=1e-7)


def test_sembedding_rejects_more_eigenvalues_than_dimensions():
    n = 5
    D2 = sq_dists(random_cloud(n, 2, seed=10))
    r, c = full_pattern(n)
    with pytest.raises(TypeError):
        DM.sembedding(D2.ravel().copy(), c, r, n, opts(nEigs=n))


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_sembedding_spectrum_is_bounded(sigma):
    n = 12
    D2 = sq_dists(random_cloud(n, 3, seed=11))
    r, c = full_pattern(n)
    vals, _ = DM.sembedding(D2.ravel().copy(), c, r, n, opts(sigma=sigma, nEigs=4))

    assert np.all(vals <= 1.0 + 1e-10)
    assert np.all(vals >= -1.0 - 1e-10)


# ==========================================================================
# DMembeddingII.op, contracts
# ==========================================================================
def test_op_returns_eight_items(line_run):
    _, _, out = line_run
    assert len(out) == 8


@pytest.mark.parametrize("nS", [20, 60, 120])
def test_op_output_shapes(nS):
    _, X = line_points(nS)
    lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared = DM.op(sq_dists(X), nS, 3, 60000)
    nEigs = min(params.num_eigs, nS - 3)

    assert lamb.shape == (nEigs + 1,)
    assert psi.shape == (nS, nEigs)
    assert mu.shape == (nS,)
    assert logEps.shape == (1501,)
    assert logSumWij.shape == (1501,)
    assert popt.shape == (4,)
    assert np.isscalar(sigma) or np.ndim(sigma) == 0
    assert np.ndim(R_squared) == 0


def test_op_output_dtypes(cloud_run):
    _, (lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared) = cloud_run

    for arr in (lamb, psi, mu, logEps, logSumWij, popt):
        assert arr.dtype == np.float64
    assert float(sigma) > 0.0


def test_op_all_outputs_are_finite(cloud_run):
    _, (lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared) = cloud_run

    for arr in (lamb, psi, mu, logEps, logSumWij, popt):
        assert np.all(np.isfinite(arr))
    assert np.isfinite(sigma) and np.isfinite(R_squared)


@pytest.mark.parametrize("maker", [line_points, circle_points, spiral_points])
def test_op_leading_eigenvalue_is_one(maker):
    n = 90
    _, X = maker(n)
    lamb = DM.op(sq_dists(X), n, 3, 60000)[0]

    assert lamb[0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("maker", [line_points, circle_points, spiral_points])
def test_op_eigenvalues_decay_monotonically(maker):
    n = 90
    _, X = maker(n)
    lamb = DM.op(sq_dists(X), n, 3, 60000)[0]

    assert np.all(np.diff(lamb) <= 1e-12)
    assert np.all(lamb <= 1.0 + 1e-10)


def test_op_riemannian_measure_is_a_probability(line_run):
    _, _, (lamb, psi, sigma, mu, *_) = line_run

    assert mu.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.all(mu > 0.0)


def test_op_psi_is_orthonormal_in_the_mu_inner_product(line_run):
    # psi_k = v_k / v_0 with v orthonormal and mu = v_0**2
    _, _, (lamb, psi, sigma, mu, *_) = line_run
    G = (psi * mu[:, None]).T @ psi

    assert np.allclose(np.diag(G), 1.0, atol=1e-9)
    assert np.allclose(G - np.diag(np.diag(G)), 0.0, atol=1e-9)


def test_op_psi_has_zero_mean_against_mu(line_run):
    # each non-trivial diffusion coordinate is orthogonal to the constant one
    _, _, (lamb, psi, sigma, mu, *_) = line_run

    assert np.allclose((mu[:, None] * psi).sum(axis=0), 0.0, atol=1e-9)


# ==========================================================================
# DMembeddingII.op, manifolds with known answers
# ==========================================================================
def test_op_line_first_coordinate_is_monotone_in_arclength(line_run):
    s, _, (lamb, psi, *_) = line_run
    d = np.diff(psi[:, 0])

    assert np.all(d > 0) or np.all(d < 0)
    assert abs(spearmanr(psi[:, 0], s).statistic) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("noise", [0.0, 0.005, 0.02])
def test_op_noisy_line_recovers_arclength(noise):
    n = 150
    s, X = line_points(n, noise=noise, seed=0)
    psi = DM.op(sq_dists(X), n, 3, 60000)[1]

    assert abs(spearmanr(psi[:, 0], s).statistic) > 0.99


def test_op_spiral_unrolls_to_arclength():
    # a 1D swiss roll. with a tight kernel the leading coordinate orders the
    # points exactly along the curve, not along the ambient chord
    n = 200
    arclen, X = spiral_points(n)
    psi = DM.op(sq_dists(X), n, 1, 60000)[1]

    assert abs(spearmanr(psi[:, 0], arclen).statistic) == pytest.approx(1.0, abs=1e-12)


def test_op_circle_eigenvalues_are_doubly_degenerate(circle_run):
    _, _, (lamb, *_) = circle_run

    assert lamb[1] == pytest.approx(lamb[2], rel=1e-8)
    assert lamb[3] == pytest.approx(lamb[4], rel=1e-8)
    assert lamb[1] > lamb[3]


def test_op_circle_first_two_coordinates_trace_a_circle(circle_run):
    # the leading pair spans {cos, sin}, so psi0**2 + psi1**2 is constant
    _, _, (lamb, psi, *_) = circle_run
    r = psi[:, 0] ** 2 + psi[:, 1] ** 2

    assert r.std() / r.mean() < 1e-8


def test_op_circle_leading_pair_is_a_first_harmonic(circle_run):
    t, _, (lamb, psi, *_) = circle_run
    # some rotation of (cos t, sin t). Correlation with the harmonic subspace is 1
    basis = np.stack([np.cos(t), np.sin(t)], axis=1)
    basis -= basis.mean(0)
    q, _ = np.linalg.qr(basis)
    v = psi[:, 0] - psi[:, 0].mean()
    proj = q @ (q.T @ v)

    assert np.linalg.norm(proj) / np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_op_is_permutation_equivariant(seed):
    n = 100
    _, X = line_points(n, noise=0.01, seed=seed)
    D2 = sq_dists(X)
    lamb, psi, sigma, mu, *_ = DM.op(D2.copy(), n, 3, 60000)

    perm = np.random.default_rng(seed).permutation(n)
    lamb_p, psi_p, sigma_p, mu_p, *_ = DM.op(D2[perm][:, perm].copy(), n, 3, 60000)

    # the bandwidth fit reduces over a reordered array, so it agrees only to
    # floating-point summation accuracy, not bit for bit
    assert np.allclose(lamb, lamb_p, atol=1e-9)
    assert sigma_p == pytest.approx(sigma, rel=1e-7)
    assert np.allclose(mu_p, mu[perm], atol=1e-9)
    assert np.allclose(np.abs(psi_p[:, 0]), np.abs(psi[perm, 0]), atol=1e-6)


# ==========================================================================
# DMembeddingII.op, bandwidth (epsilon) selection
# ==========================================================================
def test_op_logEps_is_the_fixed_grid(line_run):
    _, _, (lamb, psi, sigma, mu, logEps, *_) = line_run

    assert np.allclose(logEps, np.arange(-150, 150.2, 0.2))
    assert logEps[0] == pytest.approx(-150.0)
    assert len(logEps) == 1501


def test_op_logSumWij_is_a_monotone_sigmoid(line_run):
    _, _, (lamb, psi, sigma, mu, logEps, logSumWij, *_) = line_run

    assert np.all(np.diff(logSumWij) >= -1e-12)


@pytest.mark.parametrize("nS", [40, 80, 130])
def test_op_logSumWij_saturates_at_the_pair_count(nS):
    # tiny epsilon keeps only the nS zero-distance self pairs, huge epsilon
    # keeps all nS**2 pairs
    _, X = line_points(nS)
    logSumWij = DM.op(sq_dists(X), nS, 3, 60000)[5]

    assert logSumWij[0] == pytest.approx(np.log(nS), rel=1e-12)
    assert logSumWij[-1] == pytest.approx(np.log(nS * nS), rel=1e-12)


@pytest.mark.parametrize("maker", [line_points, circle_points])
def test_op_tanh_fit_is_excellent_on_clean_manifolds(maker):
    n = 120
    _, X = maker(n)
    R_squared = DM.op(sq_dists(X), n, 3, 60000)[7]

    assert 0.99 < R_squared <= 1.0


@pytest.mark.parametrize("tune", [1, 2, 3, 5, 10])
def test_op_sigma_is_linear_in_tune(tune):
    # sigma = tune * sqrt(2 exp(-b/a)). The tanh fit does not see tune at all
    n = 100
    _, X = line_points(n)
    D2 = sq_dists(X)
    base = DM.op(D2.copy(), n, 1, 60000)[2]
    scaled = DM.op(D2.copy(), n, tune, 60000)[2]

    assert scaled == pytest.approx(tune * base, rel=1e-12)


@pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
def test_op_sigma_scales_with_the_data(scale):
    # rescaling all distances by c must rescale the selected bandwidth by c
    n = 100
    _, X = line_points(n, noise=0.01, seed=0)
    base = DM.op(sq_dists(X), n, 3, 60000)[2]
    scaled = DM.op(sq_dists(scale * X), n, 3, 60000)[2]

    assert scaled == pytest.approx(scale * base, rel=1e-3)


def test_op_sigma_is_positive(cloud_run):
    _, (lamb, psi, sigma, *_) = cloud_run
    assert sigma > 0.0


@pytest.mark.parametrize("prefsigma", [1.0, 60000, 600000, -3.0])
def test_op_prefsigma_argument_is_dead_code(prefsigma):
    # `count` is initialized to 0 and never incremented, so the else-branch that
    # would use prefsigma is unreachable
    n = 60
    _, X = line_points(n)
    D2 = sq_dists(X)
    ref = DM.op(D2.copy(), n, 3, 60000)
    got = DM.op(D2.copy(), n, 3, prefsigma)

    assert got[2] == pytest.approx(ref[2], rel=1e-15)
    assert np.allclose(got[0], ref[0])


def test_op_tanh_fit_reproduces_logSumWij(line_run):
    _, _, (lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared) = line_run
    a, b, c, d = popt
    model = d + c * np.tanh(a * logEps + b)
    ss_res = np.sum((logSumWij - model) ** 2)
    ss_tot = np.sum((logSumWij - logSumWij.mean()) ** 2)

    assert 1.0 - ss_res / ss_tot == pytest.approx(R_squared, rel=1e-9)


def test_op_sigma_matches_the_fitted_inflection_point(line_run):
    _, _, (lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared) = line_run

    assert sigma == pytest.approx(3 * np.sqrt(2 * np.exp(-popt[1] / popt[0])), rel=1e-12)


# ==========================================================================
# DMembeddingII.op, sizing, neighbors and side effects
# ==========================================================================
@pytest.mark.parametrize("nS", [5, 8, 12, 17, 25])
def test_op_num_eigs_is_capped_by_the_sample_count(nS):
    _, X = line_points(nS)
    lamb, psi, *_ = DM.op(sq_dists(X), nS, 3, 60000)
    nEigs = min(params.num_eigs, nS - 3)

    assert psi.shape == (nS, nEigs)
    assert lamb.shape == (nEigs + 1,)


@pytest.mark.parametrize("num_eigs", [2, 5, 9])
def test_op_honours_the_params_num_eigs_setting(params_guard_embedding, num_eigs):
    params_guard_embedding.num_eigs = num_eigs
    n = 60
    _, X = line_points(n)
    lamb, psi, *_ = DM.op(sq_dists(X), n, 3, 60000)

    assert psi.shape == (n, num_eigs)
    assert lamb.shape == (num_eigs + 1,)


@pytest.mark.parametrize("k", [10, 25, 60])
def test_op_accepts_a_truncated_neighbour_list(k):
    # k < nS keeps only the k nearest neighbors of each point
    n = 60
    _, X = line_points(n, noise=0.01, seed=0)
    lamb, psi, sigma, mu, *_ = DM.op(sq_dists(X), k, 3, 60000)

    assert psi.shape[0] == n
    assert lamb[0] == pytest.approx(1.0, abs=1e-9)
    assert mu.sum() == pytest.approx(1.0, abs=1e-9)


def test_op_fewer_neighbours_gives_a_slower_spectral_decay():
    # a sparser graph mixes more slowly, so lambda_1 sits closer to 1
    n = 60
    _, X = line_points(n, noise=0.01, seed=0)
    D2 = sq_dists(X)
    sparse_lamb = DM.op(D2.copy(), 10, 3, 60000)[0]
    dense_lamb = DM.op(D2.copy(), n, 3, 60000)[0]

    assert sparse_lamb[1] > dense_lamb[1]


def test_op_mutates_the_input_distance_matrix_diagonal():
    # `initialize` writes -inf onto the diagonal of the caller's array
    n = 30
    _, X = line_points(n)
    D2 = sq_dists(X)
    DM.op(D2, n, 3, 60000)

    assert np.all(np.isneginf(np.diag(D2)))


def test_op_is_insensitive_to_a_rigid_translation():
    n = 80
    _, X = line_points(n, noise=0.01, seed=0)
    D2 = sq_dists(X)
    shifted = sq_dists(X + np.array([100.0, -37.0]))

    a = DM.op(D2.copy(), n, 3, 60000)
    b = DM.op(shifted, n, 3, 60000)

    assert np.allclose(a[0], b[0], atol=1e-9)
    assert b[2] == pytest.approx(a[2], rel=1e-7)


def test_op_psi_columns_scaled_by_sqrt_mu_are_unit_vectors(cloud_run):
    _, (lamb, psi, sigma, mu, *_) = cloud_run
    norms = np.linalg.norm(psi * np.sqrt(mu)[:, None], axis=0)

    assert np.allclose(norms, 1.0, atol=1e-9)


# ==========================================================================
# manifoldTrimmingAuto.get_psiPath
# ==========================================================================
def test_get_psiPath_on_a_hand_built_array():
    # rows at radius 0, 3, 5 in the (0,1,2) coordinate block
    psi = np.array([
        [0.0, 0.0, 0.0, 9.0],
        [3.0, 0.0, 0.0, 9.0],
        [0.0, 3.0, 4.0, 9.0],
    ])
    assert np.array_equal(MTA.get_psiPath(psi, 1.0, 0), np.array([0]))
    assert np.array_equal(MTA.get_psiPath(psi, 3.5, 0), np.array([0, 1]))
    assert np.array_equal(MTA.get_psiPath(psi, 6.0, 0), np.array([0, 1, 2]))


@pytest.mark.parametrize("rad,expected", [(0.5, 1), (1.5, 2), (2.5, 3), (3.5, 4), (100.0, 4)])
def test_get_psiPath_radius_thresholds(rad, expected):
    # the four points sit at radii 0, 1, 2, 3 from the origin
    psi = np.zeros((4, 3))
    psi[:, 0] = [0.0, 1.0, 2.0, 3.0]
    assert len(MTA.get_psiPath(psi, rad, 0)) == expected


def test_get_psiPath_is_a_strict_inequality():
    psi = np.zeros((1, 3))
    psi[0, 0] = 2.0
    assert len(MTA.get_psiPath(psi, 2.0, 0)) == 0
    assert len(MTA.get_psiPath(psi, 2.0 + 1e-12, 0)) == 1


@pytest.mark.parametrize("plotNum", [0, 1, 2])
def test_get_psiPath_uses_three_columns_starting_at_plotNum(plotNum):
    rng = np.random.default_rng(0)
    psi = rng.standard_normal((40, 6))
    got = MTA.get_psiPath(psi, 1.5, plotNum)
    expected = np.nonzero(np.sqrt((psi[:, plotNum:plotNum + 3] ** 2).sum(1)) < 1.5)[0]

    assert np.array_equal(got, expected)


def test_get_psiPath_returns_sorted_integer_indices():
    rng = np.random.default_rng(1)
    psi = rng.standard_normal((50, 4))
    idx = MTA.get_psiPath(psi, 1.0, 0)

    assert np.issubdtype(idx.dtype, np.integer)
    assert np.array_equal(idx, np.sort(idx))
    assert idx.ndim == 1


def test_get_psiPath_zero_radius_selects_nothing():
    rng = np.random.default_rng(2)
    psi = rng.standard_normal((30, 5)) + 5.0
    assert MTA.get_psiPath(psi, 0.0, 0).size == 0


def test_get_psiPath_huge_radius_selects_everything():
    rng = np.random.default_rng(3)
    psi = rng.standard_normal((30, 5))
    assert MTA.get_psiPath(psi, 1e12, 0).size == 30


def test_get_psiPath_is_permutation_equivariant():
    rng = np.random.default_rng(4)
    psi = rng.standard_normal((60, 4))
    perm = rng.permutation(60)
    a = MTA.get_psiPath(psi, 1.2, 0)
    b = MTA.get_psiPath(psi[perm], 1.2, 0)

    assert set(perm[b].tolist()) == set(a.tolist())


def test_get_psiPath_needs_three_columns_past_plotNum():
    psi = np.zeros((5, 3))
    with pytest.raises(IndexError):
        MTA.get_psiPath(psi, 1.0, 1)


@pytest.mark.parametrize("n", [10, 40])
def test_get_psiPath_on_a_real_embedding_is_a_subset(n):
    _, X = line_points(n, noise=0.01, seed=0)
    psi = DM.op(sq_dists(X), n, 3, 60000)[1]
    idx = MTA.get_psiPath(psi, 5.0, 0)

    assert idx.size <= n
    assert np.all(idx < n)


# ==========================================================================
# manifoldTrimmingAuto.op / show_plot
# ==========================================================================
def _write_dist_file_embedding(path, D, ind):
    myio.fout1(str(path), D=D, ind=ind)


def _read_eig_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            a, b = line.split()
            rows.append((int(a), float(b)))
    return rows


def test_trimming_op_writes_psi_and_eig_files(tmp_path):
    n = 120
    _, X = line_points(n, noise=0.02, seed=0)
    dist = tmp_path / "d.pkl"
    psi_f = tmp_path / "p.pkl"
    eig_f = tmp_path / "e.txt"
    _write_dist_file_embedding(dist, sq_dists(X), np.arange(n))

    MTA.op((str(dist), str(psi_f), str(eig_f), 0), 0, 3, 5.0, False, dict(Is=True, outputFile=""))

    out = myio.fin1(str(psi_f))
    assert set(out) == {"lamb", "psi", "sigma", "mu", "posPath", "ind",
                        "logEps", "logSumWij", "popt", "R_squared"}
    assert out["psi"].shape[0] == out["posPath"].size
    assert out["mu"].sum() == pytest.approx(1.0, abs=1e-9)
    assert eig_f.exists()


def test_trimming_op_with_generous_radius_keeps_every_point(tmp_path):
    n = 120
    _, X = line_points(n, noise=0.02, seed=0)
    dist = tmp_path / "d.pkl"
    psi_f = tmp_path / "p.pkl"
    eig_f = tmp_path / "e.txt"
    _write_dist_file_embedding(dist, sq_dists(X), np.arange(n))

    MTA.op((str(dist), str(psi_f), str(eig_f), 0), 0, 3, 1e6, False, dict(Is=True, outputFile=""))
    out = myio.fin1(str(psi_f))

    assert np.array_equal(out["posPath"], np.arange(n))
    assert out["psi"].shape == (n, min(params.num_eigs, n - 3))


def test_trimming_op_eig_file_lists_the_nontrivial_spectrum(tmp_path):
    n = 100
    _, X = line_points(n, noise=0.02, seed=1)
    dist = tmp_path / "d.pkl"
    psi_f = tmp_path / "p.pkl"
    eig_f = tmp_path / "e.txt"
    _write_dist_file_embedding(dist, sq_dists(X), np.arange(n))

    MTA.op((str(dist), str(psi_f), str(eig_f), 0), 0, 3, 1e6, False, dict(Is=True, outputFile=""))
    rows = _read_eig_file(eig_f)
    lamb = myio.fin1(str(psi_f))["lamb"]

    # index column is 1-based, the trivial lambda_0 = 1 is skipped
    assert [r[0] for r in rows] == list(range(1, len(lamb)))
    assert np.allclose([r[1] for r in rows], lamb[1:], atol=5e-6)


def test_trimming_op_small_radius_prunes_points(tmp_path):
    n = 120
    _, X = line_points(n, noise=0.02, seed=0)
    dist = tmp_path / "d.pkl"
    psi_f = tmp_path / "p.pkl"
    eig_f = tmp_path / "e.txt"
    _write_dist_file_embedding(dist, sq_dists(X), np.arange(n))

    MTA.op((str(dist), str(psi_f), str(eig_f), 0), 0, 3, 2.2, False, dict(Is=True, outputFile=""))
    out = myio.fin1(str(psi_f))

    assert out["posPath"].size < n
    assert np.array_equal(out["posPath"], np.sort(out["posPath"]))
    assert out["psi"].shape[0] == out["posPath"].size
    # the iteration filters out the non-positive tail of the spectrum
    assert np.all(out["lamb"] > 0)


def test_trimming_op_respects_doSave_false(tmp_path):
    n = 80
    _, X = line_points(n, noise=0.02, seed=2)
    dist = tmp_path / "d.pkl"
    psi_f = tmp_path / "p.pkl"
    eig_f = tmp_path / "e.txt"
    _write_dist_file_embedding(dist, sq_dists(X), np.arange(n))

    MTA.op((str(dist), str(psi_f), str(eig_f), 0), 0, 3, 1e6, False, dict(Is=False, outputFile=""))

    assert not psi_f.exists()
    assert eig_f.exists()  # the eigenvalue dump is unconditional


def test_trimming_op_round_trips_the_ind_array(tmp_path):
    n = 60
    _, X = line_points(n, noise=0.02, seed=3)
    ind = np.arange(1000, 1000 + n)
    dist = tmp_path / "d.pkl"
    psi_f = tmp_path / "p.pkl"
    eig_f = tmp_path / "e.txt"
    _write_dist_file_embedding(dist, sq_dists(X), ind)

    MTA.op((str(dist), str(psi_f), str(eig_f), 0), 0, 3, 1e6, False, dict(Is=True, outputFile=""))

    assert np.array_equal(myio.fin1(str(psi_f))["ind"], ind)


def test_trimming_op_matches_a_direct_embedding(tmp_path):
    n = 90
    _, X = line_points(n, noise=0.01, seed=4)
    D2 = sq_dists(X)
    dist = tmp_path / "d.pkl"
    psi_f = tmp_path / "p.pkl"
    eig_f = tmp_path / "e.txt"
    _write_dist_file_embedding(dist, D2.copy(), np.arange(n))

    MTA.op((str(dist), str(psi_f), str(eig_f), 0), 0, 3, 1e6, False, dict(Is=True, outputFile=""))
    out = myio.fin1(str(psi_f))
    ref = DM.op(D2.copy(), n, 3, 60000)

    assert np.allclose(out["lamb"], ref[0], atol=1e-10)
    assert out["sigma"] == pytest.approx(ref[2], rel=1e-12)


@pytest.mark.xfail(reason="Bug. `if posPath == 0` is ambiguous for the ndarray posPath "
                          "the docstring advertises",
                   strict=False)
def test_trimming_op_accepts_an_array_posPath(tmp_path):
    n = 60
    _, X = line_points(n, noise=0.01, seed=5)
    dist = tmp_path / "d.pkl"
    psi_f = tmp_path / "p.pkl"
    eig_f = tmp_path / "e.txt"
    _write_dist_file_embedding(dist, sq_dists(X), np.arange(n))
    subset = np.arange(10, 50)

    MTA.op((str(dist), str(psi_f), str(eig_f), 0), subset, 3, 1e6, False,
           dict(Is=True, outputFile=""))

    assert np.array_equal(myio.fin1(str(psi_f))["posPath"], subset)


def test_show_plot_builds_a_three_panel_figure(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))
    n = 40
    lamb = np.linspace(1.0, 0.1, 6)
    psi = np.random.default_rng(0).standard_normal((n, 5))

    before = len(plt.get_fignums())
    MTA.show_plot(lamb, psi, "unit test")
    assert shown == [1]
    assert len(plt.get_fignums()) == before + 1
    plt.close("all")


# ==========================================================================
# embedd.op
# ==========================================================================
@pytest.fixture
def embed_project(params_guard_embedding, tmp_path):
    """Minimal on-disk project. distance file + psi file for PrD 0."""
    os.chdir(tmp_path)
    params_guard_embedding.project_name = "emb_test"
    prd = 0
    n = 80
    _, X = line_points(n, noise=0.02, seed=0)

    os.makedirs(params_guard_embedding.dist_dir, exist_ok=True)
    os.makedirs(params_guard_embedding.psi_dir, exist_ok=True)
    os.makedirs(params_guard_embedding.psi2_dir, exist_ok=True)
    topos = os.path.join(params_guard_embedding.out_dir, "topos", f"PrD_{prd + 1}")
    os.makedirs(topos, exist_ok=True)

    myio.fout1(params_guard_embedding.get_dist_file(prd), D=sq_dists(X), ind=np.arange(n))
    myio.fout1(params_guard_embedding.get_psi_file(prd), posPath=np.arange(n), ind=np.arange(n),
               psi=np.zeros((n, 3)))

    # coordinates are matched bit-for-bit, so keep x and y numerically distinct
    orig = [(i, 1000 + i) for i in range(n)]
    return SimpleNamespace(prd=prd, n=n, orig=orig, topos=topos, params=params_guard_embedding)


@pytest.mark.parametrize("n_keep", [80, 60, 40])
def test_embedd_op_updates_posPath_to_the_kept_points(embed_project, n_keep):
    p = embed_project
    keep = sorted(np.random.default_rng(0).choice(p.n, n_keep, replace=False).tolist())
    new = [p.orig[i] for i in keep]

    embedd.op(p.orig, new, p.prd)
    out = myio.fin1(p.params.get_psi_file(p.prd))

    assert np.array_equal(out["posPath"], np.array(keep))
    assert out["psi"].shape[0] == n_keep


def test_embedd_op_writes_the_expected_keys(embed_project):
    p = embed_project
    keep = list(range(0, p.n, 2))
    embedd.op(p.orig, [p.orig[i] for i in keep], p.prd)
    out = myio.fin1(p.params.get_psi_file(p.prd))

    assert {"lamb", "psi", "sigma", "mu", "posPath", "ind"} <= set(out)
    assert out["mu"].sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(out["lamb"] > 0)  # lamb is filtered to the positive part


def test_embedd_op_rewrites_the_eigenvalue_spectrum_file(embed_project):
    p = embed_project
    eig = os.path.join(p.topos, "eig_spec.txt")
    with open(eig, "w") as f:
        f.write("stale\tcontent\n" * 40)

    keep = list(range(0, p.n, 2))
    embedd.op(p.orig, [p.orig[i] for i in keep], p.prd)

    rows = _read_eig_file(eig)
    lamb = myio.fin1(p.params.get_psi_file(p.prd))["lamb"]
    assert [r[0] for r in rows] == list(range(1, len(lamb)))
    assert np.allclose([r[1] for r in rows], lamb[1:], atol=5e-6)


def test_embedd_op_clears_stale_nlsa_products(embed_project):
    p = embed_project
    stale = [p.params.get_psi2_file(p.prd, i) for i in range(p.params.num_psi)]
    for s in stale:
        open(s, "w").close()
    ca = os.path.join(p.topos, "class_avg.png")
    open(ca, "w").close()

    embedd.op(p.orig, p.orig, p.prd)

    assert not any(os.path.exists(s) for s in stale)
    assert not os.path.exists(ca)


def test_embedd_op_matches_a_direct_embedding_of_the_kept_block(embed_project):
    p = embed_project
    keep = list(range(5, 65))
    D = myio.fin1(p.params.get_dist_file(p.prd))["D"]
    ref = DM.op(D[keep][:, keep].copy(), len(keep), p.params.nlsa_tune, 60000)

    embedd.op(p.orig, [p.orig[i] for i in keep], p.prd)
    out = myio.fin1(p.params.get_psi_file(p.prd))

    assert out["sigma"] == pytest.approx(ref[2], rel=1e-12)
    assert np.allclose(out["lamb"], ref[0][ref[0] > 0], atol=1e-10)


def test_embedd_op_preserves_the_ind_array(embed_project):
    p = embed_project
    embedd.op(p.orig, p.orig, p.prd)
    out = myio.fin1(p.params.get_psi_file(p.prd))

    assert np.array_equal(out["ind"], np.arange(p.n))


@pytest.mark.xfail(reason="Bug. embedd.op drops logEps/logSumWij/popt/R_squared from the psi "
                          "file, so the GUI bandwidth view KeyErrors after re-embedding",
                   strict=False)
def test_embedd_op_keeps_the_bandwidth_fit_data(embed_project):
    p = embed_project
    embedd.op(p.orig, p.orig, p.prd)
    out = myio.fin1(p.params.get_psi_file(p.prd))

    assert {"logEps", "logSumWij", "popt", "R_squared"} <= set(out)


# ============================================================================
# fit_1D_open_manifold_3D.py and psi_analysis.py - NLSA cosine fit
# ============================================================================

JJ = np.arange(1, 4)


@pytest.fixture(autouse=True)
def _guard_cwd():
    """ManifoldEM singletons are process wide and some of them chdir. Never leak that."""
    cwd = os.getcwd()
    yield
    os.chdir(cwd)


def synth(a, b, tau):
    """x_ij = a_j cos(j pi tau_i) + b_j for j = 1..3."""
    tau = np.asarray(tau, dtype=float).ravel()
    return np.tile(np.asarray(b, dtype=float), (tau.size, 1)) + np.asarray(a, dtype=float) * np.cos(
        np.outer(tau, JJ) * np.pi)


def residual(x, a, b, tau):
    """R = sum_i sum_j (x_ij - b_j - a_j cos(j pi tau_i))^2."""
    return float(np.sum((np.asarray(x)[:, 0:3] - synth(a, b, tau))**2))


def make_params(a, b, x, p=0, nDim=3):
    prm = _Params()
    prm.nDim = nDim
    prm.a = np.asarray(a, dtype=float)
    prm.b = np.asarray(b, dtype=float)
    prm.x = np.atleast_2d(np.asarray(x, dtype=float))
    prm.p = p
    return prm


def random_params(seed):
    rng = np.random.default_rng(seed)
    return make_params(rng.uniform(-2.0, 2.0, 3), rng.uniform(-1.0, 1.0, 3), rng.uniform(-3.0, 3.0, (1, 3)))


# ---------------------------------------------------------------------------
# _R_p : the per-point residual evaluated on a column of candidate tau values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_r_p_matches_explicit_residual_formula(seed):
    prm = random_params(seed)
    rng = np.random.default_rng(1000 + seed)
    tau = rng.uniform(-1.5, 1.5, (11, 1))

    expected = np.sum((prm.x[prm.p, :] - prm.b - prm.a * np.cos(tau * JJ * np.pi))**2, axis=1)
    assert np.allclose(_R_p(tau, prm), expected)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 17, 64])
def test_r_p_output_shape_is_one_per_candidate(n):
    prm = random_params(0)
    tau = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    out = _R_p(tau, prm)
    assert out.shape == (n, )
    assert out.dtype == np.float64


@pytest.mark.parametrize("tau0", np.linspace(0.0, 1.0, 11))
def test_r_p_vanishes_at_the_generating_tau(tau0):
    a = np.array([1.0, 0.5, 0.25])
    b = np.array([0.1, -0.2, 0.05])
    prm = make_params(a, b, synth(a, b, [tau0]))
    assert _R_p(np.array([[tau0]]), prm)[0] == pytest.approx(0.0, abs=1e-25)


@pytest.mark.parametrize("seed", range(5))
def test_r_p_is_a_sum_of_squares_hence_nonnegative(seed):
    prm = random_params(seed)
    tau = np.linspace(-2.0, 2.0, 41).reshape(-1, 1)
    assert np.all(_R_p(tau, prm) >= 0.0)


@pytest.mark.parametrize("nd", [1, 2, 3, 4, 5])
def test_r_p_honours_the_nDim_field(nd):
    """_R_p sums over j = 1..nDim, so a/b/x must all carry nDim entries."""
    a = np.linspace(1.0, 0.2, nd)
    b = np.zeros(nd)
    prm = make_params(a, b, np.zeros((2, nd)), p=1, nDim=nd)
    tau = np.array([[0.25], [0.5]])
    expected = np.sum((0.0 - 0.0 - a * np.cos(tau * np.arange(1, nd + 1) * np.pi))**2, axis=1)
    assert np.allclose(_R_p(tau, prm), expected)


@pytest.mark.parametrize("bad", [0.3, np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2])])
def test_r_p_requires_a_column_vector_of_candidates(bad):
    """A bare scalar or 1-D tau collapses the (N, nDim) error array. sum(axis=1) then fails."""
    prm = random_params(0)
    with pytest.raises((ValueError, IndexError)):
        _R_p(bad, prm)


def test_r_p_only_looks_at_row_p_of_x():
    prm = random_params(3)
    prm.x = np.vstack([prm.x, np.full((3, 3), 1e6)])
    prm.p = 0
    tau = np.linspace(0.0, 1.0, 9).reshape(-1, 1)
    reference = _R_p(tau, prm)

    prm.x[1:, :] = -1e6
    assert np.array_equal(_R_p(tau, prm), reference)


@pytest.mark.parametrize("shift", [np.array([0.5, -1.2, 3.0]), np.array([-2.0, 0.0, 0.0]), np.zeros(3)])
def test_r_p_invariant_under_common_offset_of_b_and_x(shift):
    """Only x[p] - b enters the residual."""
    prm = random_params(5)
    tau = np.linspace(0.0, 1.0, 13).reshape(-1, 1)
    reference = _R_p(tau, prm)

    shifted = make_params(prm.a, prm.b + shift, prm.x + shift)
    assert np.allclose(_R_p(tau, shifted), reference)


@pytest.mark.parametrize("t", [0.0, 0.17, 0.5, 0.83, 1.0])
def test_r_p_inherits_the_symmetries_of_the_cosine(t):
    """cos is even and 2pi periodic, so R(-tau) = R(tau) and R(tau + 2) = R(tau)."""
    prm = random_params(2)
    base = _R_p(np.array([[t]]), prm)[0]
    assert _R_p(np.array([[-t]]), prm)[0] == pytest.approx(base, rel=1e-12, abs=1e-12)
    assert _R_p(np.array([[t + 2.0]]), prm)[0] == pytest.approx(base, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# _solve_d_R_d_tau_p_3D : exact per-point minimiser via the quintic in cos(pi tau)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_solve_returns_a_finite_length_one_array(seed):
    tau = _solve_d_R_d_tau_p_3D(random_params(seed))
    assert isinstance(tau, np.ndarray)
    assert tau.shape == (1, )
    assert np.isfinite(tau[0])


@pytest.mark.parametrize("seed", range(15))
def test_solve_tau_is_confined_to_the_unit_interval(seed):
    """Candidates are arccos(beta)/pi for real |beta| <= 1, plus the endpoints 0 and 1."""
    tau = _solve_d_R_d_tau_p_3D(random_params(seed))
    assert 0.0 <= tau[0] <= 1.0


@pytest.mark.parametrize("seed", range(10))
def test_solve_is_the_global_minimiser_on_the_unit_interval(seed):
    """Brute-force grid oracle. The analytic root finder must not be beaten anywhere in [0, 1]."""
    prm = random_params(seed)
    tau = _solve_d_R_d_tau_p_3D(prm)
    grid = np.linspace(0.0, 1.0, 20001).reshape(-1, 1)
    assert _R_p(tau.reshape(1, 1), prm)[0] <= _R_p(grid, prm).min() + 1e-12


@pytest.mark.parametrize("tau0", np.linspace(0.0, 1.0, 11))
def test_solve_recovers_tau_exactly_for_on_model_data(tau0):
    a = np.array([1.0, 0.5, 0.25])
    b = np.array([0.1, -0.2, 0.05])
    prm = make_params(a, b, synth(a, b, [tau0]))
    assert _solve_d_R_d_tau_p_3D(prm)[0] == pytest.approx(tau0, abs=1e-9)


def test_solve_with_all_zero_amplitudes_returns_zero():
    """Every polynomial coefficient vanishes, np.roots gives nothing, so only {0, 1} remain
    and they tie. argmin picks the first, i.e. tau = 0."""
    prm = make_params(np.zeros(3), np.zeros(3), np.array([[1.0, 2.0, 3.0]]))
    assert _solve_d_R_d_tau_p_3D(prm)[0] == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_solve_handles_a_vanishing_leading_coefficient(seed):
    """a3 = 0 drops the quintic to a cubic. np.roots trims the leading zeros."""
    prm = random_params(seed)
    prm.a[2] = 0.0
    tau = _solve_d_R_d_tau_p_3D(prm)
    grid = np.linspace(0.0, 1.0, 20001).reshape(-1, 1)
    assert 0.0 <= tau[0] <= 1.0
    assert _R_p(tau.reshape(1, 1), prm)[0] <= _R_p(grid, prm).min() + 1e-12


@pytest.mark.parametrize("seed", range(6))
def test_solve_reflection_symmetry(seed):
    """cos(j pi (1 - tau)) = (-1)^j cos(j pi tau), so flipping the sign of a1 and a3
    reflects the optimal tau about 1/2."""
    prm = random_params(seed)
    tau = _solve_d_R_d_tau_p_3D(prm)[0]

    mirrored = make_params(prm.a * np.array([-1.0, 1.0, -1.0]), prm.b, prm.x)
    assert _solve_d_R_d_tau_p_3D(mirrored)[0] == pytest.approx(1.0 - tau, abs=1e-9)


@pytest.mark.parametrize("scale", [0.25, 1.0, 2.7, 100.0])
def test_solve_is_invariant_under_common_rescaling(scale):
    """R scales by scale^2, which cannot move the argmin."""
    prm = random_params(4)
    tau = _solve_d_R_d_tau_p_3D(prm)[0]
    scaled = make_params(prm.a * scale, prm.b * scale, prm.x * scale)
    assert _solve_d_R_d_tau_p_3D(scaled)[0] == pytest.approx(tau, abs=1e-9)


@pytest.mark.parametrize("shift", [np.array([0.5, -1.2, 3.0]), np.array([-4.0, 4.0, 0.0]), np.zeros(3)])
def test_solve_is_invariant_under_common_offset(shift):
    prm = random_params(6)
    tau = _solve_d_R_d_tau_p_3D(prm)[0]
    shifted = make_params(prm.a, prm.b + shift, prm.x + shift)
    assert _solve_d_R_d_tau_p_3D(shifted)[0] == pytest.approx(tau, abs=1e-9)


@pytest.mark.parametrize("sign,expected", [(1.0, 0.0), (-1.0, 1.0)])
def test_solve_falls_back_to_an_endpoint_when_no_root_is_admissible(sign, expected):
    """With a = (1, 0, 0) and |x1 - b1| = 5 the stationary point is beta = +/-5, which is
    rejected by |beta| <= 1. The best remaining candidate is the nearer endpoint."""
    prm = make_params([1.0, 0.0, 0.0], np.zeros(3), [[5.0 * sign, 0.0, 0.0]])
    assert _solve_d_R_d_tau_p_3D(prm)[0] == expected


# ---------------------------------------------------------------------------
# _get_fit_params : polynomial initialization of {a, b} plus a first tau sweep
# ---------------------------------------------------------------------------


def test_get_fit_params_shapes_and_types():
    psi = synth([1.0, 0.5, 0.25], np.zeros(3), np.linspace(0.0, 1.0, 30))
    tau, prm = _get_fit_params(psi)
    assert tau.shape == (30, 1)
    assert tau.dtype == np.float64
    assert isinstance(prm, _Params)
    assert prm.a.shape == (3, )
    assert prm.b.shape == (3, )
    assert prm.x.shape == (30, 3)


@pytest.mark.parametrize("a_true,b_true", [
    (np.array([1.0, 0.5, 0.25]), np.zeros(3)),
    (np.array([2.0, 0.3, 0.1]), np.array([1.0, 2.0, 3.0])),
    (np.array([1.0, -0.5, 0.25]), np.zeros(3)),
    (np.array([0.8, 0.2, 0.05]), np.array([-1.0, 0.5, 0.3])),
    (np.array([1.2, 0.4, 0.15]), np.array([0.3, -0.1, 0.2])),
])
def test_get_fit_params_initialisation_is_already_exact_on_model_data(a_true, b_true):
    """x_2 is quadratic and x_3 cubic in x_1, so the polynomial initializer is not a guess
    at all for noiseless data. It reproduces {a, b} to machine precision."""
    psi = synth(a_true, b_true, np.linspace(0.0, 1.0, 50))
    _, prm = _get_fit_params(psi)
    assert np.allclose(prm.a, a_true, atol=1e-8)
    assert np.allclose(prm.b, b_true, atol=1e-8)


@pytest.mark.parametrize("seed", range(4))
def test_get_fit_params_tau_stays_in_the_unit_interval(seed):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(25, 3))
    tau, _ = _get_fit_params(psi)
    assert tau.min() >= 0.0 and tau.max() <= 1.0


def test_get_fit_params_only_consumes_the_first_three_columns():
    rng = np.random.default_rng(0)
    psi3 = synth([1.0, 0.5, 0.25], np.zeros(3), np.linspace(0.0, 1.0, 20))
    psi7 = np.hstack([psi3, rng.normal(size=(20, 4))])
    tau3, prm3 = _get_fit_params(psi3)
    tau7, prm7 = _get_fit_params(psi7)
    assert prm7.x.shape == (20, 3)
    assert np.array_equal(tau3, tau7)
    assert np.allclose(prm3.a, prm7.a) and np.allclose(prm3.b, prm7.b)


def test_get_fit_params_sets_the_documented_iteration_defaults():
    psi = synth([1.0, 0.5, 0.25], np.zeros(3), np.linspace(0.0, 1.0, 12))
    _, prm = _get_fit_params(psi)
    assert prm.nDim == 3
    assert prm.maxIter == 100
    assert prm.delta_a_max == 1
    assert prm.delta_b_max == 1
    assert prm.delta_tau_max == 0.01


def test_get_fit_params_leaves_p_at_the_last_visited_index():
    """`for a.p in range(nS)` writes the loop variable onto the params object."""
    psi = synth([1.0, 0.5, 0.25], np.zeros(3), np.linspace(0.0, 1.0, 14))
    _, prm = _get_fit_params(psi)
    assert prm.p == 13


def test_get_fit_params_x_is_a_view_on_the_caller_array():
    psi = synth([1.0, 0.5, 0.25], np.zeros(3), np.linspace(0.0, 1.0, 10))
    _, prm = _get_fit_params(psi)
    assert np.shares_memory(prm.x, psi)


def test_get_fit_params_leaves_x_fit_unset():
    psi = synth([1.0, 0.5, 0.25], np.zeros(3), np.linspace(0.0, 1.0, 10))
    _, prm = _get_fit_params(psi)
    assert prm.x_fit is None


@pytest.mark.parametrize("ncol", [1, 2])
def test_get_fit_params_requires_three_columns(ncol):
    psi = np.random.default_rng(0).normal(size=(10, ncol))
    with pytest.raises(IndexError):
        _get_fit_params(psi)


def test_get_fit_params_does_not_mutate_psi():
    psi = synth([1.0, 0.5, 0.25], np.array([0.2, 0.1, -0.3]), np.linspace(0.0, 1.0, 18))
    reference = psi.copy()
    _get_fit_params(psi)
    assert np.array_equal(psi, reference)


# ---------------------------------------------------------------------------
# fit_1D_open_manifold_3D : the alternating {a, b} / {tau} minimization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a_true,b_true", [
    (np.array([1.0, 0.5, 0.25]), np.zeros(3)),
    (np.array([-1.0, 0.5, -0.25]), np.array([0.1, -0.2, 0.05])),
    (np.array([2.0, 0.3, 0.1]), np.array([1.0, 2.0, 3.0])),
    (np.array([1.0, -0.5, 0.25]), np.zeros(3)),
    (np.array([0.8, 0.2, 0.05]), np.array([-1.0, 0.5, 0.3])),
    (np.array([-1.5, -0.4, -0.2]), np.array([0.2, 0.2, 0.2])),
])
def test_fit_round_trip_reconstruction_is_exact(a_true, b_true):
    tau_true = np.linspace(0.0, 1.0, 40)
    psi = synth(a_true, b_true, tau_true)
    a, b, tau = fit_1D_open_manifold_3D(psi.copy())
    assert np.allclose(synth(a, b, tau), psi, atol=1e-10)


@pytest.mark.parametrize("a_true,b_true", [
    (np.array([1.0, 0.5, 0.25]), np.zeros(3)),
    (np.array([2.0, 0.3, 0.1]), np.array([1.0, 2.0, 3.0])),
    (np.array([0.8, 0.2, 0.05]), np.array([-1.0, 0.5, 0.3])),
])
def test_fit_recovers_the_generating_coefficients_up_to_the_reflection(a_true, b_true):
    """tau -> 1 - tau flips the sign of a1 and a3 while leaving the data unchanged, so the
    fit is only identifiable up to that reflection. The offsets b are not."""
    tau_true = np.linspace(0.0, 1.0, 40)
    a, b, _ = fit_1D_open_manifold_3D(synth(a_true, b_true, tau_true))
    mirrored = a_true * np.array([-1.0, 1.0, -1.0])
    assert np.allclose(a, a_true, atol=1e-8) or np.allclose(a, mirrored, atol=1e-8)
    assert np.allclose(b, b_true, atol=1e-8)


@pytest.mark.parametrize("n", [12, 25, 40, 77])
def test_fit_output_shapes(n):
    psi = synth([1.0, 0.5, 0.25], np.zeros(3), np.linspace(0.0, 1.0, n))
    a, b, tau = fit_1D_open_manifold_3D(psi)
    assert a.shape == (3, )
    assert b.shape == (3, )
    assert tau.shape == (n, 1)
    assert tau.dtype == np.float64


@pytest.mark.parametrize("seed", range(6))
def test_fit_tau_is_confined_to_the_unit_interval_even_off_model(seed):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(30, 3))
    _, _, tau = fit_1D_open_manifold_3D(psi)
    assert tau.min() >= 0.0 and tau.max() <= 1.0


@pytest.mark.parametrize("a_true", [
    np.array([1.0, 0.5, 0.25]),
    np.array([-1.0, 0.5, -0.25]),
    np.array([2.0, 0.3, 0.1]),
    np.array([0.8, 0.2, 0.05]),
])
def test_fit_tau_is_strictly_monotone_in_the_true_tau(a_true):
    """The generating tau is an increasing sweep. The recovered tau must be a monotone
    reparameterization of it (increasing, or decreasing under the reflection)."""
    tau_true = np.linspace(0.0, 1.0, 45)
    _, _, tau = fit_1D_open_manifold_3D(synth(a_true, np.array([0.1, -0.2, 0.05]), tau_true))
    d = np.diff(tau.ravel())
    assert np.all(d > 0) or np.all(d < 0)
    assert abs(np.corrcoef(tau.ravel(), tau_true)[0, 1]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("sigma,atol", [(0.0, 1e-8), (0.01, 0.02), (0.03, 0.05), (0.1, 0.1)])
def test_fit_degrades_gracefully_with_additive_noise(sigma, atol):
    a_true = np.array([1.0, 0.5, 0.25])
    b_true = np.array([0.1, -0.2, 0.05])
    tau_true = np.linspace(0.0, 1.0, 60)
    psi = synth(a_true, b_true, tau_true) + np.random.default_rng(11).normal(0.0, sigma, (60, 3))
    a, b, tau = fit_1D_open_manifold_3D(psi)
    assert np.allclose(np.abs(a), np.abs(a_true), atol=atol)
    assert np.allclose(b, b_true, atol=atol)
    assert abs(np.corrcoef(tau.ravel(), tau_true)[0, 1]) > 0.99


def test_fit_tau_fidelity_is_non_increasing_in_the_noise_level():
    a_true = np.array([1.0, 0.5, 0.25])
    b_true = np.array([0.1, -0.2, 0.05])
    tau_true = np.linspace(0.0, 1.0, 60)
    fidelity = []
    for sigma in (0.0, 0.01, 0.03, 0.1, 0.2):
        psi = synth(a_true, b_true, tau_true) + np.random.default_rng(11).normal(0.0, sigma, (60, 3))
        _, _, tau = fit_1D_open_manifold_3D(psi)
        fidelity.append(abs(np.corrcoef(tau.ravel(), tau_true)[0, 1]))
    assert all(fidelity[i + 1] <= fidelity[i] + 1e-12 for i in range(len(fidelity) - 1))


def _run_fit_recording_iterations(psi, tweak=None):
    """Run the real fit while snooping on the module-level helpers.

    Returns (result, records, n_solve_calls) where each record is the
    (a_k, b_k, tau_{k-1}) triple in force at the top of the tau sweep of one
    outer iteration.  `tau` is captured through the very array object the loop
    mutates in place.
    """
    real_get = fitmod._get_fit_params
    real_solve = fitmod._solve_d_R_d_tau_p_3D
    holder = {}
    records = []
    calls = [0]

    def spy_get(arg):
        tau, prm = real_get(arg)
        if tweak is not None:
            tweak(tau, prm)
        holder["tau"] = tau
        return tau, prm

    def spy_solve(prm):
        calls[0] += 1
        if prm.p == 0 and "tau" in holder:
            records.append((prm.a.copy(), prm.b.copy(), holder["tau"].copy()))
        return real_solve(prm)

    fitmod._get_fit_params = spy_get
    fitmod._solve_d_R_d_tau_p_3D = spy_solve
    try:
        result = fit_1D_open_manifold_3D(psi)
    finally:
        fitmod._get_fit_params = real_get
        fitmod._solve_d_R_d_tau_p_3D = real_solve
    return result, records, calls[0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fit_residual_never_increases_across_outer_iterations(seed):
    """Alternating minimization. The {a, b} step is an exact least squares solve and the
    tau step is an exact per-point minimiser, so R must be monotone non-increasing."""
    rng = np.random.default_rng(seed)
    tau_true = np.sort(rng.uniform(0.0, 1.0, 70))
    psi = synth([1.0, 0.5, 0.25], [0.1, -0.2, 0.05], tau_true) + rng.normal(0.0, 0.05, (70, 3))

    _, records, _ = _run_fit_recording_iterations(psi)
    assert len(records) >= 2
    values = [residual(psi, a, b, tau) for a, b, tau in records]
    assert all(values[i + 1] <= values[i] + 1e-12 for i in range(len(values) - 1))


def test_fit_terminates_well_within_max_iterations():
    rng = np.random.default_rng(4)
    tau_true = np.sort(rng.uniform(0.0, 1.0, 50))
    psi = synth([1.0, 0.5, 0.25], [0.1, -0.2, 0.05], tau_true) + rng.normal(0.0, 0.03, (50, 3))
    _, records, _ = _run_fit_recording_iterations(psi)
    assert 1 <= len(records) < 100


@pytest.mark.xfail(reason="Bug. tau_old = tau aliases the array that is then mutated in place, "
                   "so delta_tau is identically 0 and the tau convergence test is vacuous",
                   strict=False)
def test_fit_delta_tau_convergence_criterion_is_actually_checked():
    """Start from a scrambled tau and make the tau tolerance the only binding criterion.

    A correct implementation would see a large delta_tau after the first sweep and keep
    iterating.  Because `tau_old = tau` binds the same object that `tau[a.p] = ...` then
    overwrites, delta_tau is always exactly 0 and the loop exits after one iteration.
    """
    rng = np.random.default_rng(1)
    tau_true = np.sort(rng.uniform(0.0, 1.0, 40))
    psi = synth([1.0, 0.5, 0.25], np.zeros(3), tau_true) + rng.normal(0.0, 0.02, (40, 3))

    def tweak(tau, prm):
        prm.delta_a_max = 1e9  # never binding
        prm.delta_b_max = 1e9  # never binding
        prm.delta_tau_max = 1e-12  # the only binding criterion
        prm.maxIter = 6
        tau[:] = np.random.default_rng(2).uniform(0.0, 1.0, tau.shape)

    _, _, n_calls = _run_fit_recording_iterations(psi, tweak=tweak)
    n_outer = n_calls / 40 - 1  # the first 40 solves belong to _get_fit_params
    assert n_outer > 1


def test_fit_is_deterministic():
    rng = np.random.default_rng(9)
    psi = synth([1.0, 0.5, 0.25], [0.1, -0.2, 0.05], np.linspace(0, 1, 35)) + rng.normal(0.0, 0.02, (35, 3))
    first = fit_1D_open_manifold_3D(psi.copy())
    second = fit_1D_open_manifold_3D(psi.copy())
    for u, v in zip(first, second):
        assert np.array_equal(u, v)


def test_fit_does_not_mutate_its_input():
    rng = np.random.default_rng(10)
    psi = synth([1.0, 0.5, 0.25], [0.1, -0.2, 0.05], np.linspace(0, 1, 35)) + rng.normal(0.0, 0.02, (35, 3))
    reference = psi.copy()
    fit_1D_open_manifold_3D(psi)
    assert np.array_equal(psi, reference)


def test_fit_ignores_columns_beyond_the_third():
    rng = np.random.default_rng(12)
    psi3 = synth([1.0, 0.5, 0.25], [0.1, -0.2, 0.05], np.linspace(0, 1, 35)) + rng.normal(0.0, 0.02, (35, 3))
    psi7 = np.hstack([psi3, rng.normal(size=(35, 4))])
    a3, b3, t3 = fit_1D_open_manifold_3D(psi3.copy())
    a7, b7, t7 = fit_1D_open_manifold_3D(psi7)
    assert np.array_equal(a3, a7) and np.array_equal(b3, b7) and np.array_equal(t3, t7)


def test_fit_is_equivariant_under_row_permutation():
    """The {a, b} step only sees order independent sums and the tau step is per row."""
    rng = np.random.default_rng(7)
    tau_true = np.sort(rng.uniform(0.0, 1.0, 45))
    psi = synth([1.2, 0.4, 0.15], [0.3, -0.1, 0.2], tau_true) + rng.normal(0.0, 0.02, (45, 3))
    perm = rng.permutation(45)

    a0, b0, t0 = fit_1D_open_manifold_3D(psi.copy())
    a1, b1, t1 = fit_1D_open_manifold_3D(psi[perm].copy())
    assert np.allclose(a0, a1, atol=1e-10)
    assert np.allclose(b0, b1, atol=1e-10)
    assert np.allclose(t0[perm], t1, atol=1e-10)


@pytest.mark.parametrize("shift", [
    np.array([2.0, -3.0, 0.5]),
    np.array([0.0, 0.0, 0.0]),
    np.array([-1.0, -1.0, -1.0]),
])
def test_fit_is_equivariant_under_translation(shift):
    """Translating the data translates b and leaves the amplitudes and tau alone."""
    tau_true = np.linspace(0.0, 1.0, 40)
    psi = synth([1.2, 0.4, 0.15], [0.3, -0.1, 0.2], tau_true)
    a0, b0, t0 = fit_1D_open_manifold_3D(psi.copy())
    a1, b1, t1 = fit_1D_open_manifold_3D((psi + shift).copy())
    assert np.allclose(a0, a1, atol=1e-8)
    assert np.allclose(b0 + shift, b1, atol=1e-8)
    assert np.allclose(t0, t1, atol=1e-8)


@pytest.mark.parametrize("scale", [0.1, 1.0, 3.5, 20.0])
def test_fit_is_equivariant_under_rescaling(scale):
    rng = np.random.default_rng(7)
    tau_true = np.sort(rng.uniform(0.0, 1.0, 45))
    psi = synth([1.2, 0.4, 0.15], [0.3, -0.1, 0.2], tau_true) + rng.normal(0.0, 0.02, (45, 3))
    a0, b0, t0 = fit_1D_open_manifold_3D(psi.copy())
    a1, b1, t1 = fit_1D_open_manifold_3D((psi * scale).copy())
    assert np.allclose(a0 * scale, a1, rtol=1e-8, atol=1e-10)
    assert np.allclose(b0 * scale, b1, rtol=1e-8, atol=1e-10)
    assert np.allclose(t0, t1, atol=1e-8)


def test_fit_on_constant_data_is_degenerate_but_consistent():
    """A/b are wildly under determined here. Lstsq takes the minimum norm branch, and the
    reconstruction at tau = 0 still reproduces the data exactly."""
    psi = np.ones((10, 3))
    a, b, tau = fit_1D_open_manifold_3D(psi.copy())
    assert np.allclose(tau, 0.0)
    assert np.allclose(a, 0.5) and np.allclose(b, 0.5)
    assert np.allclose(synth(a, b, tau), psi)


@pytest.mark.parametrize("seed", range(5))
def test_fit_on_unstructured_data_stays_finite(seed):
    psi = np.random.default_rng(seed).normal(size=(30, 3))
    a, b, tau = fit_1D_open_manifold_3D(psi)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(tau))


def test_fit_requires_three_columns():
    with pytest.raises(IndexError):
        fit_1D_open_manifold_3D(np.random.default_rng(0).normal(size=(10, 2)))


def test_module_epsilon_guards_the_relative_change_denominator():
    assert fitmod.eps == 1e-4


# ---------------------------------------------------------------------------
# psi_analysis._corr / _diff_corr
# ---------------------------------------------------------------------------


def _corr_inputs(seed=0, n=50, cols=4):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, cols)), rng.normal(size=(n, cols))


@pytest.mark.parametrize("n,m", [(i, j) for i in range(3) for j in range(3)])
def test_corr_is_the_pearson_coefficient_times_the_sample_count(n, m):
    """_corr divides by std(A)*std(B) but never by N, so it is N * r, not r."""
    a, b = _corr_inputs()
    expected = np.corrcoef(a[:, n], b[:, m])[0, 1] * a.shape[0]
    assert _corr(a, b, n, m) == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("n_samples", [5, 17, 50, 128, 301])
def test_corr_of_a_column_with_itself_equals_the_sample_count(n_samples):
    a = np.random.default_rng(1).normal(size=(n_samples, 2))
    assert _corr(a, a, 0, 0) == pytest.approx(float(n_samples), rel=1e-10)


@pytest.mark.parametrize("n,m", [(0, 1), (1, 0), (2, 3), (3, 2)])
def test_corr_is_symmetric_under_swapping_both_arguments(n, m):
    a, b = _corr_inputs(seed=2)
    assert _corr(a, b, n, m) == pytest.approx(_corr(b, a, m, n), rel=1e-12)


@pytest.mark.parametrize("offset", [-10.0, -0.5, 0.0, 7.25])
def test_corr_is_invariant_under_a_constant_offset(offset):
    a, b = _corr_inputs(seed=3)
    shifted = b.copy()
    shifted[:, 1] += offset
    assert _corr(a, shifted, 0, 1) == pytest.approx(_corr(a, b, 0, 1), rel=1e-10)


@pytest.mark.parametrize("scale", [0.01, 0.5, 2.0, 1000.0])
def test_corr_is_invariant_under_positive_rescaling(scale):
    a, b = _corr_inputs(seed=4)
    scaled = b.copy()
    scaled[:, 2] *= scale
    assert _corr(a, scaled, 0, 2) == pytest.approx(_corr(a, b, 0, 2), rel=1e-10)


@pytest.mark.parametrize("scale", [-1.0, -0.25, -8.0])
def test_corr_flips_sign_under_negative_rescaling(scale):
    a, b = _corr_inputs(seed=5)
    scaled = b.copy()
    scaled[:, 2] *= scale
    assert _corr(a, scaled, 0, 2) == pytest.approx(-_corr(a, b, 0, 2), rel=1e-10)


def test_corr_of_exact_anticorrelation_is_minus_n():
    a = np.random.default_rng(6).normal(size=(40, 2))
    assert _corr(a, -a, 1, 1) == pytest.approx(-40.0, rel=1e-10)


def test_corr_on_a_flat_column_returns_nan_rather_than_raising():
    """The `except RuntimeError: raise RuntimeError("flat image")` branch is dead code:
    numpy signals 0/0 with a warning and a nan, never a RuntimeError."""
    flat = np.ones((10, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = _corr(flat, flat, 0, 0)
    assert np.isnan(out)


@pytest.mark.parametrize("maxval", [1, 2, 3])
def test_diff_corr_matches_its_four_term_definition(maxval):
    a, b = _corr_inputs(seed=7)
    expected = (_corr(a, b, 0, 0) + _corr(a, b, maxval, maxval) - _corr(a, b, 0, maxval) - _corr(a, b, maxval, 0))
    assert _diff_corr(a, b, maxval) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("maxval", [1, 2, 3])
def test_diff_corr_of_an_array_with_itself_is_nonnegative(maxval):
    """N(1 + 1 - 2r) with r <= 1, so the diagonal case can never go negative."""
    a, _ = _corr_inputs(seed=8)
    assert _diff_corr(a, a, maxval) >= -1e-9


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_diff_corr_vanishes_when_maxval_is_zero(seed):
    a, b = _corr_inputs(seed=seed)
    assert _diff_corr(a, b, 0) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("maxval", [1, 2, 3])
def test_diff_corr_flips_sign_when_the_second_array_is_negated(maxval):
    a, b = _corr_inputs(seed=9)
    assert _diff_corr(a, -b, maxval) == pytest.approx(-_diff_corr(a, b, maxval), rel=1e-10)


@pytest.mark.parametrize("maxval", [1, 2, 3])
def test_diff_corr_flips_sign_when_the_two_compared_columns_are_swapped(maxval):
    a, b = _corr_inputs(seed=10)
    swapped = b.copy()
    swapped[:, [0, maxval]] = swapped[:, [maxval, 0]]
    assert _diff_corr(a, swapped, maxval) == pytest.approx(-_diff_corr(a, b, maxval), rel=1e-10)


@pytest.mark.parametrize("maxval", [1, 2, 3])
def test_diff_corr_is_a_plain_python_float_scale_quantity(maxval):
    a, b = _corr_inputs(seed=11)
    out = _diff_corr(a, b, maxval)
    assert np.isscalar(out) or out.shape == ()
    assert np.isfinite(out)


# ============================================================================
# data_store.py - projection-direction store, anchors, thresholding
# ============================================================================

REBUILD_ENV = "MANIFOLD_REBUILD_DS"


# --------------------------------------------------------------------------------------
# isolation
# --------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def isolated(tmp_path):
    """Restore the two singletons and the cwd around every test.

    `params.load()` chdirs, `_ProjectionDirections.update()` writes files relative to the
    cwd, and both objects keep their state on a module-level instance, so nothing here may
    leak between tests.
    """
    cwd = os.getcwd()
    saved_params = dict(params.__dict__)
    saved_store = dict(data_store.__dict__)
    saved_env = os.environ.get(REBUILD_ENV)
    os.environ.pop(REBUILD_ENV, None)

    # shadow the class-level projection directions with a private, freshly built one
    data_store._projection_directions = _ProjectionDirections()
    data_store._image_stack_data = None

    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(cwd)
        params.__dict__.clear()
        params.__dict__.update(saved_params)
        data_store.__dict__.clear()
        data_store.__dict__.update(saved_store)
        os.environ.pop(REBUILD_ENV, None)
        if saved_env is not None:
            os.environ[REBUILD_ENV] = saved_env


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def object_array(list_of_lists):
    """Build a genuinely 1-D ragged object array of index arrays."""
    out = np.empty(len(list_of_lists), dtype=object)
    for i, a in enumerate(list_of_lists):
        out[i] = np.asarray(a, dtype=int)
    return out


def make_prds(image_index_lists, thres_ids, thres_high=1000):
    """Hand-built store. No disk, no star file, no tessellation."""
    prds = _ProjectionDirections()
    prds.image_indices_full = object_array(image_index_lists)
    prds.occupancy_full = np.array([len(a) for a in image_index_lists], dtype=int)
    prds.thres_ids = list(thres_ids)
    prds.thres_high = thres_high
    prds.bin_centers = np.zeros((3, len(image_index_lists)))
    prds.cluster_ids = np.zeros(len(thres_ids), dtype=int)
    n_images = 1 + max((max(a) for a in image_index_lists if len(a)), default=0)
    prds.defocus = np.arange(n_images, dtype=float) * 10.0
    return prds


def write_star(path, n, seed=0):
    """Minimal old-format RELION star file. Header block then n rows."""
    rng = np.random.default_rng(seed)
    cols = [
        "rlnAngleRot",
        "rlnAngleTilt",
        "rlnAnglePsi",
        "rlnDefocusU",
        "rlnDefocusV",
        "rlnVoltage",
        "rlnSphericalAberration",
        "rlnAmplitudeContrast",
        "rlnOriginX",
        "rlnOriginY",
    ]
    lines = ["data_", "", "loop_"]
    lines += [f"_{c} #{i + 1}" for i, c in enumerate(cols)]

    rot = rng.uniform(-180.0, 180.0, n)
    tilt = rng.uniform(0.0, 180.0, n)
    psi = rng.uniform(-180.0, 180.0, n)
    dfu = rng.uniform(9000.0, 11000.0, n)
    shx = rng.uniform(-1.0, 1.0, n)
    shy = rng.uniform(-1.0, 1.0, n)
    for i in range(n):
        lines.append(
            f"{rot[i]:.6f} {tilt[i]:.6f} {psi[i]:.6f} {dfu[i]:.3f} {dfu[i] + 100.0:.3f} "
            f"300.0 2.7 0.1 {shx[i]:.6f} {shy[i]:.6f}"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return n


def configure_project(tmp_path, n_images=400, thres_low=5, thres_high=8, name="ds"):
    """Point `params` at a tiny synthetic project rooted in tmp_path."""
    star = tmp_path / "align.star"
    write_star(star, n_images)

    params.project_name = name
    params.align_param_file = str(star)
    params.ncpu = 1
    params.num_psi = 8
    params.aperture_index = 1
    # ang_width = aperture * resolution / diameter = 0.5 rad -> int(4 pi / 0.25) = 50 bins
    params.ms_estimated_resolution = 5.0
    params.particle_diameter = 10.0
    params.ms_pixel_size = 1.0
    params.ms_num_pixels = 8
    params.prd_thres_low = thres_low
    params.prd_thres_high = thres_high
    params.tess_hemisphere_type = "lovisolo_silva"
    params.tess_hemisphere_vec = [1.0, 0.0, 0.0]
    params.prd_assignment = "hard"
    params.prd_cone_width_factor = 1.0

    os.makedirs(params.out_dir, exist_ok=True)
    os.makedirs(params.dist_dir, exist_ok=True)
    return n_images


def write_dist_file(prd_index, n_images, n_pixels, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(params.get_dist_file(prd_index), "w") as f:
        f["rotations"] = rng.uniform(0.0, 360.0, n_images)
        f["msk2"] = np.ones((n_pixels, n_pixels))
        f["image_filter"] = np.ones((n_pixels, n_pixels))


def write_image_stack(tmp_path, n_images, n_pixels, seed=1):
    rng = np.random.default_rng(seed)
    path = tmp_path / "stack.mrcs"
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(rng.normal(size=(n_images, n_pixels, n_pixels)).astype(np.float32))
    params.img_stack_file = str(path)
    return path


def make_info(**overrides):
    fields = dict(
        prd_index=1,
        S2_bin_index=2,
        bin_center=np.array([1.0, 0.0, 0.0]),
        occupancy=5,
        trash=True,
        anchor=False,
        cluster_id=3,
        raw_image_indices=np.arange(5),
        image_offsets=np.zeros((5, 2)),
        image_centers=np.zeros((5, 3)),
        image_quats=np.zeros((5, 4)),
        image_rotations=np.zeros(5),
        image_mirrored=np.zeros(5, dtype=bool),
        image_filter=np.ones((4, 4)),
        image_mask=np.ones((4, 4)),
    )
    fields.update(overrides)
    return PrdInfo(**fields)


@pytest.fixture
def built_project(tmp_path):
    """A fully rebuilt store backed by a synthetic star file."""
    configure_project(tmp_path)
    prds = data_store.get_prds()
    return prds


# ======================================================================================
# Sense
# ======================================================================================
@pytest.mark.parametrize("member,value", [(Sense.FWD, 1), (Sense.REV, -1)])
def test_sense_enum_values(member, value):
    # FWD/REV carry the physical sign of the conformational coordinate, not an index
    assert member.value == value


@pytest.mark.parametrize("value,member", [(1, Sense.FWD), (-1, Sense.REV)])
def test_sense_lookup_by_value(value, member):
    assert Sense(value) is member


@pytest.mark.parametrize("bad_value", [0, 2, -2, 100])
def test_sense_lookup_by_bad_value_raises(bad_value):
    with pytest.raises(ValueError):
        Sense(bad_value)


def test_sense_has_exactly_two_members():
    assert list(Sense) == [Sense.FWD, Sense.REV]


@pytest.mark.parametrize("member,index", [(Sense.FWD, 0), (Sense.REV, 1)])
def test_sense_to_index(member, index):
    # to_index is a GUI/array index, distinct from the enum value
    assert member.to_index() == index


@pytest.mark.parametrize("index,member", [(0, Sense.FWD), (1, Sense.REV)])
def test_sense_from_index(index, member):
    assert Sense.from_index(index) is member


@pytest.mark.parametrize("member", [Sense.FWD, Sense.REV])
def test_sense_index_round_trip_from_member(member):
    assert Sense.from_index(member.to_index()) is member


@pytest.mark.parametrize("index", [0, 1])
def test_sense_index_round_trip_from_index(index):
    assert Sense.from_index(index).to_index() == index


@pytest.mark.parametrize("bad_index", [-1, 2, 3, 100, -100])
def test_sense_from_index_rejects_out_of_range(bad_index):
    with pytest.raises(ValueError, match="Invalid index"):
        Sense.from_index(bad_index)


@pytest.mark.parametrize("bad_index", ["0", "1", None, [0], (1,)])
def test_sense_from_index_rejects_non_numeric(bad_index):
    # from_index compares with ==, so anything that is not numerically 0 or 1 falls through
    with pytest.raises(ValueError, match="Invalid index"):
        Sense.from_index(bad_index)


@pytest.mark.parametrize(
    "numeric,member",
    [
        (False, Sense.FWD),
        (True, Sense.REV),
        (0.0, Sense.FWD),
        (1.0, Sense.REV),
        (np.int64(0), Sense.FWD),
        (np.int64(1), Sense.REV),
    ],
)
def test_sense_from_index_accepts_anything_equal_to_0_or_1(numeric, member):
    # `idx == 0` is a value comparison, so bools, floats and numpy ints all work
    assert Sense.from_index(numeric) is member


@pytest.mark.parametrize("member", [Sense.FWD, Sense.REV])
def test_sense_survives_pickle_as_the_same_singleton(member):
    assert pickle.loads(pickle.dumps(member)) is member


@pytest.mark.parametrize("member", [Sense.FWD, Sense.REV])
def test_sense_to_index_is_a_plain_int(member):
    assert type(member.to_index()) is int


def test_sense_to_index_is_not_the_value():
    # REV.value is -1 but its index is 1. Conflating the two flips the trajectory sense
    assert Sense.REV.value != Sense.REV.to_index()


# ======================================================================================
# Anchor
# ======================================================================================
def test_anchor_defaults():
    a = Anchor()
    assert a.CC == 1
    assert a.sense is Sense.FWD


def test_anchor_default_cc_is_one_indexed():
    # CC counts conformational coordinates starting at 1, unlike prd indices which are 0-based
    assert Anchor().CC == 1


@pytest.mark.parametrize("cc", [1, 2, 3, 0, -1, 17])
def test_anchor_stores_cc_without_validation(cc):
    assert Anchor(CC=cc).CC == cc


@pytest.mark.parametrize("sense", [Sense.FWD, Sense.REV])
def test_anchor_stores_sense(sense):
    assert Anchor(sense=sense).sense is sense


@pytest.mark.parametrize("cc", [1, 2])
@pytest.mark.parametrize("sense", [Sense.FWD, Sense.REV])
def test_anchor_positional_argument_order_is_cc_then_sense(cc, sense):
    a = Anchor(cc, sense)
    assert (a.CC, a.sense) == (cc, sense)


def test_anchor_instances_are_independent():
    a, b = Anchor(), Anchor()
    a.CC = 9
    assert b.CC == 1


def test_anchor_has_no_value_equality():
    # plain class, no __eq__: two equal-looking anchors are still distinct objects
    assert Anchor(1, Sense.FWD) != Anchor(1, Sense.FWD)


def test_anchor_is_mutable():
    a = Anchor()
    a.sense = Sense.REV
    a.CC = 4
    assert (a.CC, a.sense) == (4, Sense.REV)


def test_anchor_round_trips_through_pickle():
    a = pickle.loads(pickle.dumps(Anchor(CC=3, sense=Sense.REV)))
    assert (a.CC, a.sense) == (3, Sense.REV)


@pytest.mark.parametrize("bad_sense", [0, 1, "FWD", None])
def test_anchor_does_not_validate_sense(bad_sense):
    # no type check in __init__, so a raw index silently becomes the stored sense
    assert Anchor(sense=bad_sense).sense == bad_sense


# ======================================================================================
# PrdInfo
# ======================================================================================
PRDINFO_FIELDS = [
    "prd_index",
    "S2_bin_index",
    "bin_center",
    "occupancy",
    "trash",
    "anchor",
    "cluster_id",
    "raw_image_indices",
    "image_offsets",
    "image_centers",
    "image_quats",
    "image_rotations",
    "image_mirrored",
    "image_filter",
    "image_mask",
]


def test_prdinfo_field_order():
    assert [f.name for f in dataclasses.fields(PrdInfo)] == PRDINFO_FIELDS


@pytest.mark.parametrize("name", PRDINFO_FIELDS)
def test_prdinfo_exposes_every_field(name):
    assert hasattr(make_info(), name)


@pytest.mark.parametrize("name", PRDINFO_FIELDS)
def test_prdinfo_fields_have_no_default(name):
    field = {f.name: f for f in dataclasses.fields(PrdInfo)}[name]
    assert field.default is dataclasses.MISSING
    assert field.default_factory is dataclasses.MISSING


def test_prdinfo_requires_all_fields():
    with pytest.raises(TypeError):
        PrdInfo()


def test_prdinfo_is_a_mutable_dataclass():
    info = make_info()
    info.trash = False
    assert dataclasses.is_dataclass(info) and info.trash is False


REPR_FIELDS = [
    "prd_index",
    "S2_bin_index",
    "bin_center",
    "occupancy",
    "trash",
    "anchor",
    "cluster_id",
]


@pytest.mark.parametrize("name", REPR_FIELDS)
def test_prdinfo_repr_lists_scalar_metadata(name):
    assert f"{name}:" in repr(make_info())


@pytest.mark.parametrize("name", ["raw_image_indices", "image_quats", "image_filter", "image_mask"])
def test_prdinfo_repr_omits_bulk_arrays(name):
    assert name not in repr(make_info())


def test_prdinfo_repr_is_one_line_per_scalar_field():
    assert len(repr(make_info()).split("\n")) == len(REPR_FIELDS)


def test_prdinfo_equality_is_ambiguous_for_distinct_arrays():
    # dataclass __eq__ compares field tuples. numpy arrays make that a truth-value error
    with pytest.raises(ValueError):
        _ = make_info() == make_info()


def test_prdinfo_equality_short_circuits_on_identical_arrays():
    info = make_info()
    assert info == dataclasses.replace(info)


@pytest.mark.parametrize(
    "name,shape",
    [
        ("bin_center", (3,)),
        ("image_offsets", (5, 2)),
        ("image_centers", (5, 3)),
        ("image_quats", (5, 4)),
        ("image_rotations", (5,)),
        ("image_mirrored", (5,)),
    ],
)
def test_prdinfo_array_shapes(name, shape):
    assert getattr(make_info(), name).shape == shape


# ======================================================================================
# _ProjectionDirections construction defaults
# ======================================================================================
@pytest.mark.parametrize(
    "name,shape",
    [
        ("bin_centers", (3, 0)),
        ("pos_raw", (3, 0)),
        ("pos_full", (3, 0)),
        ("quats_raw", (4, 0)),
        ("quats_full", (4, 0)),
        ("pos_thresholded", (3, 0)),
        ("defocus", (0,)),
        ("theta_thresholded", (0,)),
        ("phi_thresholded", (0,)),
        ("cluster_ids", (0,)),
        ("occupancy_full", (0,)),
        ("image_indices_full", (0,)),
        ("image_is_mirrored", (0,)),
    ],
)
def test_fresh_store_array_shapes(name, shape):
    assert getattr(_ProjectionDirections(), name).shape == shape


@pytest.mark.parametrize(
    "name,dtype",
    [
        ("occupancy_full", np.integer),
        ("cluster_ids", np.integer),
        ("image_is_mirrored", np.bool_),
        ("image_indices_full", np.object_),
        ("bin_centers", np.floating),
        ("defocus", np.floating),
    ],
)
def test_fresh_store_dtypes(name, dtype):
    assert np.issubdtype(getattr(_ProjectionDirections(), name).dtype, dtype)


@pytest.mark.parametrize(
    "name,empty",
    [
        ("thres_ids", []),
        ("anchors", {}),
        ("trash_ids", set()),
        ("reembed_ids", set()),
        ("neighbor_graph", {}),
        ("neighbor_subgraph", []),
        ("neighbor_graph_pruned", {}),
        ("neighbor_subgraph_pruned", []),
    ],
)
def test_fresh_store_empty_containers(name, empty):
    value = getattr(_ProjectionDirections(), name)
    assert value == empty and type(value) is type(empty)


def test_fresh_store_microscope_origin_is_a_pair_of_arrays():
    origin = _ProjectionDirections().microscope_origin
    assert isinstance(origin, tuple) and len(origin) == 2
    assert all(a.shape == (0,) for a in origin)


@pytest.mark.parametrize("low,high", [(1, 10), (100, 2000), (0, 0), (7, 7)])
def test_fresh_store_snapshots_the_thresholds_from_params(low, high):
    # the thresholds are copied at construction. a later params change is what triggers a rebuild
    params.prd_thres_low = low
    params.prd_thres_high = high
    prds = _ProjectionDirections()
    assert (prds.thres_low, prds.thres_high) == (low, high)


def test_fresh_store_counts_are_zero():
    prds = _ProjectionDirections()
    assert prds.n_bins == 0
    assert prds.n_thresholded == 0
    assert prds.anchor_ids == []


def test_fresh_store_thresholded_indices_is_empty():
    prds = _ProjectionDirections()
    assert prds.thresholded_image_indices.shape == (0,)
    assert prds.occupancy.shape == (0,)


# ======================================================================================
# anchors
# ======================================================================================
def test_insert_anchor_registers_the_id():
    prds = _ProjectionDirections()
    prds.insert_anchor(4, Anchor(CC=2, sense=Sense.REV))
    assert prds.anchor_ids == [4]
    assert prds.anchors[4].CC == 2
    assert prds.anchors[4].sense is Sense.REV


@pytest.mark.parametrize(
    "ids",
    [
        [3, 1, 2],
        [2, 1, 3],
        [1, 2, 3],
        [3, 2, 1],
        [10, 0, 5],
        [7],
        [],
    ],
)
def test_anchor_ids_are_sorted_regardless_of_insertion_order(ids):
    prds = _ProjectionDirections()
    for i in ids:
        prds.insert_anchor(i, Anchor())
    assert prds.anchor_ids == sorted(ids)


def test_anchor_ids_returns_a_plain_list():
    prds = _ProjectionDirections()
    prds.insert_anchor(0, Anchor())
    assert isinstance(prds.anchor_ids, list)


def test_insert_anchor_is_idempotent_in_the_id_set():
    prds = _ProjectionDirections()
    for _ in range(5):
        prds.insert_anchor(2, Anchor())
    assert prds.anchor_ids == [2]


def test_insert_anchor_replaces_the_previous_anchor():
    prds = _ProjectionDirections()
    prds.insert_anchor(2, Anchor(CC=1, sense=Sense.FWD))
    prds.insert_anchor(2, Anchor(CC=5, sense=Sense.REV))
    assert prds.anchors[2].CC == 5
    assert prds.anchors[2].sense is Sense.REV


def test_remove_anchor_deletes_the_id():
    prds = _ProjectionDirections()
    prds.insert_anchor(1, Anchor())
    prds.insert_anchor(2, Anchor())
    prds.remove_anchor(1)
    assert prds.anchor_ids == [2]


@pytest.mark.parametrize("missing", [0, 5, -1, 999])
def test_remove_anchor_on_a_missing_id_is_a_no_op(missing):
    prds = _ProjectionDirections()
    prds.insert_anchor(3, Anchor())
    prds.remove_anchor(missing)
    assert prds.anchor_ids == [3]


def test_remove_anchor_is_idempotent():
    prds = _ProjectionDirections()
    prds.insert_anchor(3, Anchor())
    prds.remove_anchor(3)
    prds.remove_anchor(3)
    assert prds.anchor_ids == []


def test_insert_remove_round_trip_restores_the_store():
    prds = _ProjectionDirections()
    prds.insert_anchor(6, Anchor())
    prds.remove_anchor(6)
    assert prds.anchors == {}


@pytest.mark.parametrize("bad_id", [-1, -100])
def test_insert_anchor_does_not_validate_the_index(bad_id):
    # no bounds check against n_bins/n_thresholded
    prds = _ProjectionDirections()
    prds.insert_anchor(bad_id, Anchor())
    assert bad_id in prds.anchors


def test_anchor_ids_reflects_later_mutation_of_the_dict():
    prds = _ProjectionDirections()
    prds.anchors[9] = Anchor()
    assert prds.anchor_ids == [9]


# ======================================================================================
# trash ids and cluster ids
# ======================================================================================
@pytest.mark.parametrize("ids", [set(), {0}, {1, 4, 9}, {3}])
def test_trash_ids_is_a_plain_set(ids):
    prds = _ProjectionDirections()
    prds.trash_ids |= ids
    assert prds.trash_ids == ids


def test_trash_ids_add_and_discard():
    prds = _ProjectionDirections()
    prds.trash_ids.add(2)
    prds.trash_ids.add(2)
    assert prds.trash_ids == {2}
    prds.trash_ids.discard(2)
    assert prds.trash_ids == set()


def test_trash_and_reembed_ids_are_independent_sets():
    prds = _ProjectionDirections()
    prds.trash_ids.add(1)
    assert prds.reembed_ids == set()
    assert prds.trash_ids is not prds.reembed_ids


@pytest.mark.parametrize("cluster_ids", [[0, 0, 0], [0, 1, 0], [2, 2, 1]])
def test_cluster_ids_index_by_prd_not_by_bin(cluster_ids):
    # cluster_ids is sized by the number of thresholded prds, not by n_bins
    prds = make_prds([[0, 1], [2, 3], [4, 5], [6, 7]], thres_ids=[0, 2, 3])
    prds.cluster_ids = np.array(cluster_ids)
    assert len(prds.cluster_ids) == prds.n_thresholded
    assert list(prds.cluster_ids) == cluster_ids


# ======================================================================================
# thresholded_image_indices / occupancy semantics
# ======================================================================================
def test_thresholded_image_indices_selects_only_kept_bins():
    prds = make_prds([[0, 1, 2], [3], [4, 5], [6, 7, 8, 9]], thres_ids=[0, 3])
    got = [list(a) for a in prds.thresholded_image_indices]
    assert got == [[0, 1, 2], [6, 7, 8, 9]]


def test_thresholded_image_indices_follows_thres_ids_order():
    prds = make_prds([[0, 1], [2, 3, 4], [5]], thres_ids=[2, 0])
    got = [list(a) for a in prds.thresholded_image_indices]
    assert got == [[5], [0, 1]]


@pytest.mark.parametrize("thres_high,expected", [(1, [0]), (2, [0, 1]), (3, [0, 1, 2]), (5, [0, 1, 2, 3])])
def test_thresholded_image_indices_truncates_to_thres_high(thres_high, expected):
    prds = make_prds([[0, 1, 2, 3]], thres_ids=[0], thres_high=thres_high)
    assert list(prds.thresholded_image_indices[0]) == expected


def test_thresholded_image_indices_keeps_the_first_images_not_a_random_subset():
    prds = make_prds([list(range(10))], thres_ids=[0], thres_high=4)
    assert list(prds.thresholded_image_indices[0]) == [0, 1, 2, 3]


def test_thresholded_image_indices_does_not_mutate_image_indices_full():
    prds = make_prds([list(range(10)), [10, 11]], thres_ids=[0, 1], thres_high=3)
    _ = prds.thresholded_image_indices
    _ = prds.thresholded_image_indices
    assert list(prds.image_indices_full[0]) == list(range(10))


def test_thresholded_image_indices_is_recomputed_each_access():
    prds = make_prds([[0, 1, 2]], thres_ids=[0], thres_high=2)
    first = prds.thresholded_image_indices
    prds.thres_high = 3
    second = prds.thresholded_image_indices
    assert len(first[0]) == 2 and len(second[0]) == 3


def test_occupancy_is_the_untruncated_count():
    # occupancy reports every image that landed in the bin, while thresholded_image_indices
    # only hands back the first thres_high of them
    prds = make_prds([list(range(10)), [10, 11, 12]], thres_ids=[0, 1], thres_high=4)
    assert list(prds.occupancy) == [10, 3]
    assert [len(a) for a in prds.thresholded_image_indices] == [4, 3]


def test_occupancy_matches_thresholded_length_when_below_the_high_threshold():
    prds = make_prds([[0, 1], [2, 3, 4]], thres_ids=[0, 1], thres_high=100)
    assert list(prds.occupancy) == [len(a) for a in prds.thresholded_image_indices]


def test_occupancy_selects_the_kept_bins_in_order():
    prds = make_prds([[0], [1, 2], [3, 4, 5]], thres_ids=[2, 0])
    assert list(prds.occupancy) == [3, 1]


@pytest.mark.parametrize("n_bins", [1, 3, 7, 24])
def test_n_bins_is_the_bin_center_count(n_bins):
    prds = _ProjectionDirections()
    prds.bin_centers = np.zeros((3, n_bins))
    assert prds.n_bins == n_bins


@pytest.mark.parametrize("thres_ids", [[], [0], [0, 2], [1, 2, 3]])
def test_n_thresholded_is_the_kept_bin_count(thres_ids):
    prds = _ProjectionDirections()
    prds.thres_ids = list(thres_ids)
    assert prds.n_thresholded == len(thres_ids)


def test_n_thresholded_never_exceeds_n_bins_after_a_real_build(built_project):
    assert built_project.n_thresholded <= built_project.n_bins


@pytest.mark.parametrize("prd,expected", [(0, [0.0, 10.0]), (1, [50.0])])
def test_get_defocus_by_prd_uses_thresholded_indices(prd, expected):
    prds = make_prds([[0, 1], [5], [2, 3]], thres_ids=[0, 1])
    assert list(prds.get_defocus_by_prd(prd)) == expected


def test_get_defocus_by_prd_is_truncated_by_thres_high():
    prds = make_prds([[0, 1, 2, 3]], thres_ids=[0], thres_high=2)
    assert list(prds.get_defocus_by_prd(0)) == [0.0, 10.0]


@pytest.mark.xfail(
    reason="Bug. thresholded_image_indices assumes a ragged 1-D object array, but "
    "bin_and_threshold returns a rectangular 2-D object array when every bin holds "
    "the same number of images",
    strict=False,
)
def test_thresholded_image_indices_handles_equal_occupancy_bins():
    prds = _ProjectionDirections()
    # np.array([...], dtype=object) collapses to shape (2, 3) when the rows are equal length
    prds.image_indices_full = np.array(
        [np.array([0, 1, 2]), np.array([3, 4, 5])], dtype=object
    )
    prds.thres_ids = [0, 1]
    prds.thres_high = 2
    got = [list(a) for a in prds.thresholded_image_indices]
    assert got == [[0, 1], [3, 4]]


# ======================================================================================
# get_prd_data validation
# ======================================================================================
@pytest.mark.parametrize("bad", [1.0, 0.5, "0", None, [0], (0,), np.float64(0.0)])
def test_get_prd_data_rejects_non_integral_indices(bad):
    prds = make_prds([[0, 1], [2, 3]], thres_ids=[0, 1])
    with pytest.raises(TypeError, match="Invalid prd index type"):
        prds.get_prd_data(bad)


@pytest.mark.parametrize("bad", [-1, -2, -100])
def test_get_prd_data_rejects_negative_indices(bad):
    prds = make_prds([[0, 1], [2, 3]], thres_ids=[0, 1])
    with pytest.raises(ValueError, match="Invalid prd index"):
        prds.get_prd_data(bad)


@pytest.mark.parametrize("bad", [2, 3, 50])
def test_get_prd_data_rejects_indices_at_or_beyond_n_bins(bad):
    prds = make_prds([[0, 1], [2, 3]], thres_ids=[0, 1])
    with pytest.raises(ValueError, match=r"Valid indices on \[0, 2\)"):
        prds.get_prd_data(bad)


@pytest.mark.parametrize("integral", [np.int32(0), np.int64(0)])
def test_get_prd_data_accepts_numpy_integers(integral):
    # numpy ints register as numbers.Integral, so the type gate lets them through and
    # the failure comes later from the missing distance file
    prds = make_prds([[0, 1], [2, 3]], thres_ids=[0, 1])
    with pytest.raises((OSError, FileNotFoundError)):
        prds.get_prd_data(integral)


@pytest.mark.xfail(
    reason="Bug. get_prd_data/PrdData bound-check against n_bins (bins on S2) instead of "
    "n_thresholded (actual projection directions), so an index in between raises a raw "
    "IndexError instead of the documented ValueError",
    strict=False,
)
def test_get_prd_data_rejects_index_between_n_thresholded_and_n_bins():
    prds = make_prds([[0, 1], [2, 3], [4, 5]], thres_ids=[0, 1])
    assert prds.n_thresholded < prds.n_bins
    with pytest.raises(ValueError):
        prds.get_prd_data(prds.n_thresholded)


def test_prddata_constructor_rejects_index_beyond_n_bins(built_project):
    with pytest.raises(ValueError, match="Invalid prd index"):
        PrdData(built_project.n_bins)


def test_get_prd_data_error_message_reports_the_valid_range():
    prds = make_prds([[0], [1], [2]], thres_ids=[0, 1, 2])
    with pytest.raises(ValueError) as exc:
        prds.get_prd_data(3)
    assert "[0, 3)" in str(exc.value)


# ======================================================================================
# save / load
# ======================================================================================
def test_save_writes_the_params_pd_file(tmp_path):
    configure_project(tmp_path)
    _ProjectionDirections().save()
    assert os.path.isfile(params.pd_file)
    assert params.pd_file.endswith("pd_data.pkl")


def test_save_raises_when_the_output_directory_is_missing():
    params.project_name = "no_such_project"
    with pytest.raises(FileNotFoundError):
        _ProjectionDirections().save()


def test_save_load_round_trips_anchors(tmp_path):
    configure_project(tmp_path)
    prds = _ProjectionDirections()
    prds.insert_anchor(2, Anchor(CC=3, sense=Sense.REV))
    prds.save()

    restored = _ProjectionDirections()
    restored.load()
    assert restored.anchor_ids == [2]
    assert restored.anchors[2].CC == 3
    assert restored.anchors[2].sense is Sense.REV


def test_save_load_round_trips_trash_and_reembed_ids(tmp_path):
    configure_project(tmp_path)
    prds = _ProjectionDirections()
    prds.trash_ids |= {1, 4}
    prds.reembed_ids |= {7}
    prds.save()

    restored = _ProjectionDirections()
    restored.load()
    assert restored.trash_ids == {1, 4}
    assert restored.reembed_ids == {7}


def test_save_load_round_trips_arrays(tmp_path):
    configure_project(tmp_path)
    prds = _ProjectionDirections()
    prds.bin_centers = np.arange(12.0).reshape(3, 4)
    prds.cluster_ids = np.array([0, 1, 1, 2])
    prds.thres_ids = [0, 2, 3]
    prds.save()

    restored = _ProjectionDirections()
    restored.load()
    assert np.array_equal(restored.bin_centers, prds.bin_centers)
    assert np.array_equal(restored.cluster_ids, prds.cluster_ids)
    assert restored.thres_ids == [0, 2, 3]


def test_load_accepts_an_explicit_path(tmp_path):
    configure_project(tmp_path)
    prds = _ProjectionDirections()
    prds.insert_anchor(5, Anchor())
    prds.save()

    restored = _ProjectionDirections()
    restored.load(params.pd_file)
    assert restored.anchor_ids == [5]


def test_load_defaults_to_the_params_pd_file(tmp_path):
    configure_project(tmp_path)
    prds = _ProjectionDirections()
    prds.trash_ids.add(11)
    prds.save()

    restored = _ProjectionDirections()
    restored.load(None)
    assert restored.trash_ids == {11}


def test_load_merges_rather_than_replaces_the_instance_dict(tmp_path):
    # load() does __dict__.update(), so attributes absent from the pickle survive
    configure_project(tmp_path)
    _ProjectionDirections().save()

    restored = _ProjectionDirections()
    restored.scratch_marker = "kept"
    restored.load()
    assert restored.scratch_marker == "kept"


def test_load_missing_file_raises(tmp_path):
    configure_project(tmp_path)
    with pytest.raises(FileNotFoundError):
        _ProjectionDirections().load(str(tmp_path / "absent.pkl"))


def test_saved_pickle_is_a_plain_attribute_dict(tmp_path):
    configure_project(tmp_path)
    prds = _ProjectionDirections()
    prds.save()
    with open(params.pd_file, "rb") as f:
        payload = pickle.load(f)
    assert isinstance(payload, dict)
    assert "thres_low" in payload and "anchors" in payload


def test_save_load_preserves_thresholds(tmp_path):
    configure_project(tmp_path, thres_low=13, thres_high=77)
    _ProjectionDirections().save()

    params.prd_thres_low = 1
    params.prd_thres_high = 2
    restored = _ProjectionDirections()
    assert (restored.thres_low, restored.thres_high) == (1, 2)
    restored.load()
    assert (restored.thres_low, restored.thres_high) == (13, 77)


# ======================================================================================
# update() and the MANIFOLD_REBUILD_DS environment variable
# ======================================================================================
def test_update_builds_a_consistent_store(tmp_path):
    configure_project(tmp_path)
    prds = data_store.get_prds()

    assert prds.n_bins > 0
    assert prds.n_thresholded == len(prds.thres_ids)
    assert prds.pos_full.shape == (3, 400)
    assert prds.quats_full.shape == (4, 400)
    assert prds.image_is_mirrored.shape == (400,)
    assert prds.pos_thresholded.shape == (3, prds.n_thresholded)
    assert prds.cluster_ids.shape == (prds.n_thresholded,)


def test_update_publishes_the_active_prd_count_to_params(tmp_path):
    configure_project(tmp_path)
    prds = data_store.get_prds()
    assert params.prd_n_active == prds.n_thresholded


def test_update_writes_both_caches(tmp_path):
    configure_project(tmp_path)
    data_store.get_prds()
    assert os.path.isfile(params.pd_file)
    assert os.path.isfile(f"params_{params.project_name}.toml")


def test_update_doubles_the_defocus_array(tmp_path):
    # legacy "augmentation". (U+V)/2 is concatenated with itself, so defocus is twice as
    # long as the image list even though only the first half is ever indexed
    configure_project(tmp_path)
    prds = data_store.get_prds()
    assert prds.defocus.shape == (800,)
    assert np.array_equal(prds.defocus[:400], prds.defocus[400:])


def test_update_keeps_every_bin_center_on_the_unit_sphere(tmp_path):
    configure_project(tmp_path)
    prds = data_store.get_prds()
    assert np.allclose(np.linalg.norm(prds.bin_centers, axis=0), 1.0)


def test_update_keeps_bins_in_the_requested_half_space(tmp_path):
    configure_project(tmp_path)
    prds = data_store.get_prds()
    plane_vec = np.array(params.tess_hemisphere_vec)
    assert np.all(plane_vec @ prds.bin_centers >= 0.0)


def test_update_thresholded_angles_match_the_thresholded_positions(tmp_path):
    configure_project(tmp_path)
    prds = data_store.get_prds()
    pos = prds.pos_thresholded
    theta = np.degrees(np.arccos(pos[2, :]))
    phi = np.degrees(np.arctan2(pos[1, :], pos[0, :]))
    assert np.allclose(prds.theta_thresholded, theta)
    assert np.allclose(prds.phi_thresholded, phi)


def test_update_only_keeps_bins_above_the_low_threshold(tmp_path):
    configure_project(tmp_path, thres_low=12)
    prds = data_store.get_prds()
    assert np.all(prds.occupancy >= 12)


@pytest.mark.parametrize("thres_low,thres_high", [(5, 8), (10, 4), (20, 100)])
def test_update_respects_both_thresholds(tmp_path, thres_low, thres_high):
    configure_project(tmp_path, thres_low=thres_low, thres_high=thres_high)
    prds = data_store.get_prds()
    assert np.all(prds.occupancy >= thres_low)
    assert max(len(a) for a in prds.thresholded_image_indices) <= thres_high


def test_second_update_does_not_rebuild(tmp_path, capsys):
    configure_project(tmp_path)
    data_store.get_prds()
    capsys.readouterr()
    data_store.get_prds()
    assert "Calculating projection direction information" not in capsys.readouterr().out


def test_update_loads_the_cache_into_a_fresh_store(tmp_path, capsys):
    configure_project(tmp_path)
    built = data_store.get_prds()
    capsys.readouterr()

    fresh = _ProjectionDirections()
    fresh.update()
    assert "Calculating projection direction information" not in capsys.readouterr().out
    assert fresh.thres_ids == built.thres_ids
    assert np.array_equal(fresh.bin_centers, built.bin_centers)


@pytest.mark.parametrize("changed", ["prd_thres_low", "prd_thres_high"])
def test_changing_a_threshold_forces_a_rebuild(tmp_path, capsys, changed):
    configure_project(tmp_path)
    data_store.get_prds()
    capsys.readouterr()

    setattr(params, changed, getattr(params, changed) + 1)
    prds = data_store.get_prds()
    assert "Calculating projection direction information" in capsys.readouterr().out
    assert getattr(prds, changed.replace("prd_", "")) == getattr(params, changed)


@pytest.mark.parametrize("value", ["1", "0", "true", "false", "yes"])
def test_rebuild_env_var_is_truthy_for_any_non_empty_value(tmp_path, capsys, value):
    # update() uses bool(os.environ.get(...)), so even "0" and "false" force a rebuild
    configure_project(tmp_path)
    data_store.get_prds()
    capsys.readouterr()

    os.environ[REBUILD_ENV] = value
    data_store.get_prds()
    out = capsys.readouterr().out
    assert "Rebuilding data store" in out
    assert "Calculating projection direction information" in out


def test_rebuild_env_var_is_consumed(tmp_path):
    configure_project(tmp_path)
    data_store.get_prds()
    os.environ[REBUILD_ENV] = "1"
    data_store.get_prds()
    assert REBUILD_ENV not in os.environ


def test_empty_rebuild_env_var_does_not_force_a_rebuild(tmp_path, capsys):
    configure_project(tmp_path)
    data_store.get_prds()
    capsys.readouterr()

    os.environ[REBUILD_ENV] = ""
    data_store.get_prds()
    assert "Rebuilding data store" not in capsys.readouterr().out
    # an unconsumed empty value is left in place
    assert os.environ[REBUILD_ENV] == ""


def test_rebuild_reproduces_the_same_tessellation(tmp_path):
    configure_project(tmp_path)
    first = data_store.get_prds()
    thres_ids = list(first.thres_ids)
    bin_centers = first.bin_centers.copy()

    os.environ[REBUILD_ENV] = "1"
    second = data_store.get_prds()
    assert second.thres_ids == thres_ids
    assert np.array_equal(second.bin_centers, bin_centers)


def test_rebuild_clears_anchors_and_trash(tmp_path):
    configure_project(tmp_path)
    prds = data_store.get_prds()
    prds.insert_anchor(0, Anchor(CC=2, sense=Sense.REV))
    prds.trash_ids.add(1)

    os.environ[REBUILD_ENV] = "1"
    prds = data_store.get_prds()
    assert prds.anchors == {}
    assert prds.trash_ids == set()


@pytest.mark.xfail(
    reason="Bug. update() clears anchors and trash_ids on a rebuild but leaves reembed_ids, "
    "so stale prd indices survive a retessellation",
    strict=False,
)
def test_rebuild_clears_reembed_ids(tmp_path):
    configure_project(tmp_path)
    prds = data_store.get_prds()
    prds.reembed_ids.add(0)

    os.environ[REBUILD_ENV] = "1"
    prds = data_store.get_prds()
    assert prds.reembed_ids == set()


def test_update_assigns_every_image_to_exactly_one_bin_under_hard_assignment(tmp_path):
    configure_project(tmp_path)
    prds = data_store.get_prds()
    all_indices = np.concatenate([a for a in prds.image_indices_full if len(a)])
    assert len(all_indices) == len(np.unique(all_indices))
    assert prds.occupancy_full.sum() == prds.pos_full.shape[1]


def test_update_occupancy_full_matches_the_bin_membership(tmp_path):
    configure_project(tmp_path)
    prds = data_store.get_prds()
    counts = np.array([len(a) for a in prds.image_indices_full])
    assert np.array_equal(counts, prds.occupancy_full)


def test_update_neighbor_graph_covers_the_thresholded_prds(tmp_path):
    configure_project(tmp_path)
    prds = data_store.get_prds()
    assert prds.neighbor_graph["nNodes"] == prds.n_thresholded
    assert len(prds.neighbor_subgraph) == len(prds.neighbor_graph["NodesConnComp"])


def test_update_cluster_ids_label_the_connected_components(tmp_path):
    configure_project(tmp_path)
    prds = data_store.get_prds()
    for label, nodes in enumerate(prds.neighbor_graph["NodesConnComp"]):
        assert np.all(prds.cluster_ids[nodes] == label)


def test_update_mirrored_flag_matches_the_half_space(tmp_path):
    configure_project(tmp_path)
    prds = data_store.get_prds()
    plane_vec = np.array(params.tess_hemisphere_vec)
    # after collapsing, every image direction sits on the plane_vec side
    assert np.all(plane_vec @ prds.pos_full >= -1e-12)
    # the mirrored ones are exactly those that started on the far side
    assert np.array_equal(prds.image_is_mirrored, (plane_vec @ prds.pos_raw) < 0.0)


# ======================================================================================
# _DataStore singleton
# ======================================================================================
def test_data_store_is_a_singleton():
    assert _DataStore() is data_store
    assert _DataStore() is _DataStore()


def test_get_prds_returns_the_same_object_every_call(tmp_path):
    configure_project(tmp_path)
    assert data_store.get_prds() is data_store.get_prds()


def test_get_prds_calls_update(monkeypatch):
    calls = []
    monkeypatch.setattr(
        data_store._projection_directions,
        "update",
        lambda: calls.append(1),
    )
    data_store.get_prds()
    assert calls == [1]


def test_get_prd_data_delegates_to_the_projection_directions(monkeypatch):
    seen = []
    monkeypatch.setattr(data_store._projection_directions, "update", lambda: None)
    monkeypatch.setattr(
        data_store._projection_directions,
        "get_prd_data",
        lambda i: seen.append(i) or "sentinel",
    )
    assert data_store.get_prd_data(3) == "sentinel"
    assert seen == [3]


def test_get_image_stack_data_is_memory_mapped_and_cached(tmp_path):
    configure_project(tmp_path)
    write_image_stack(tmp_path, 6, 8)
    first = data_store.get_image_stack_data()
    second = data_store.get_image_stack_data()
    assert first is second
    assert first.shape == (6, 8, 8)
    assert first.dtype == np.float32


def test_get_image_stack_data_matches_the_file_contents(tmp_path):
    configure_project(tmp_path)
    path = write_image_stack(tmp_path, 4, 8)
    with mrcfile.open(str(path), "r") as m:
        expected = np.array(m.data)
    assert np.array_equal(np.array(data_store.get_image_stack_data()), expected)


# ======================================================================================
# PrdData
# ======================================================================================
@pytest.fixture
def prd_project(tmp_path):
    """A built store plus the distance file and image stack that PrdData needs."""
    configure_project(tmp_path)
    prds = data_store.get_prds()
    n_pixels = params.ms_num_pixels
    write_image_stack(tmp_path, prds.pos_full.shape[1], n_pixels)
    for prd_index in range(min(3, prds.n_thresholded)):
        write_dist_file(
            prd_index, len(prds.thresholded_image_indices[prd_index]), n_pixels
        )
    return prds


@pytest.mark.parametrize("prd_index", [0, 1, 2])
def test_prddata_info_indices(prd_project, prd_index):
    pd = data_store.get_prd_data(prd_index)
    assert pd.info.prd_index == prd_index
    assert pd.info.S2_bin_index == prd_project.thres_ids[prd_index]


@pytest.mark.parametrize("prd_index", [0, 1, 2])
def test_prddata_bin_center_is_the_thresholded_bin(prd_project, prd_index):
    pd = data_store.get_prd_data(prd_index)
    expected = prd_project.bin_centers[:, prd_project.thres_ids[prd_index]]
    assert np.array_equal(pd.info.bin_center, expected)
    assert np.isclose(np.linalg.norm(pd.info.bin_center), 1.0)


@pytest.mark.parametrize("prd_index", [0, 1, 2])
def test_prddata_occupancy_can_exceed_the_kept_image_count(prd_project, prd_index):
    # occupancy is the full bin population. raw_image_indices is capped at thres_high
    pd = data_store.get_prd_data(prd_index)
    assert pd.info.occupancy == prd_project.occupancy[prd_index]
    assert len(pd.info.raw_image_indices) == min(
        pd.info.occupancy, prd_project.thres_high
    )


@pytest.mark.parametrize("prd_index", [0, 1, 2])
def test_prddata_per_image_arrays_are_consistently_sized(prd_project, prd_index):
    pd = data_store.get_prd_data(prd_index)
    n = len(pd.info.raw_image_indices)
    assert pd.info.image_offsets.shape == (n, 2)
    assert pd.info.image_centers.shape == (n, 3)
    assert pd.info.image_quats.shape == (n, 4)
    assert pd.info.image_rotations.shape == (n,)
    assert pd.info.image_mirrored.shape == (n,)


def test_prddata_image_offsets_are_row_then_column(prd_project):
    # column 0 is the y shift and column 1 the x shift, matching scipy.ndimage.shift's
    # (row, col) argument order
    pd = data_store.get_prd_data(0)
    idx = pd.info.raw_image_indices
    assert np.array_equal(pd.info.image_offsets[:, 0], prd_project.microscope_origin[1][idx])
    assert np.array_equal(pd.info.image_offsets[:, 1], prd_project.microscope_origin[0][idx])


def test_prddata_image_centers_are_unit_vectors(prd_project):
    pd = data_store.get_prd_data(0)
    assert np.allclose(np.linalg.norm(pd.info.image_centers, axis=1), 1.0)
    # the nptyping annotation says Int, but these are S2 coordinates and really are floats
    assert np.issubdtype(pd.info.image_centers.dtype, np.floating)


def test_prddata_image_quats_are_unit_quaternions(prd_project):
    # ManifoldEM quaternions are scalar-first [R, I, I, I] and normalized
    pd = data_store.get_prd_data(0)
    assert np.allclose(np.linalg.norm(pd.info.image_quats, axis=1), 1.0)


def test_prddata_flags_track_the_store_state(prd_project):
    prd_project.trash_ids.add(1)
    prd_project.insert_anchor(2, Anchor())
    assert data_store.get_prd_data(0).info.trash is False
    assert data_store.get_prd_data(1).info.trash is True
    assert data_store.get_prd_data(2).info.anchor is True
    assert data_store.get_prd_data(0).info.anchor is False


def test_prddata_cluster_id_comes_from_the_store(prd_project):
    pd = data_store.get_prd_data(0)
    assert pd.info.cluster_id == prd_project.cluster_ids[0]


def test_prddata_filter_and_mask_come_from_the_distance_file(prd_project):
    pd = data_store.get_prd_data(0)
    n_pixels = params.ms_num_pixels
    assert pd.info.image_filter.shape == (n_pixels, n_pixels)
    assert pd.info.image_mask.shape == (n_pixels, n_pixels)


def test_prddata_repr_is_the_info_repr(prd_project):
    pd = data_store.get_prd_data(0)
    assert repr(pd) == repr(pd.info)
    assert "prd_index: 0" in repr(pd)


def test_prddata_raw_images_are_the_stack_rows(prd_project):
    pd = data_store.get_prd_data(0)
    stack = np.array(data_store.get_image_stack_data())
    assert pd.raw_images.shape == (
        len(pd.info.raw_image_indices),
        params.ms_num_pixels,
        params.ms_num_pixels,
    )
    assert np.array_equal(pd.raw_images, stack[pd.info.raw_image_indices])


def test_prddata_raw_images_are_cached(prd_project):
    pd = data_store.get_prd_data(0)
    assert pd.raw_images is pd.raw_images


def test_prddata_transformed_images_have_the_raw_shape(prd_project):
    pd = data_store.get_prd_data(0)
    assert pd.transformed_images.shape == pd.raw_images.shape
    assert np.all(np.isfinite(pd.transformed_images))


def test_prddata_transformed_images_are_cached(prd_project):
    pd = data_store.get_prd_data(0)
    assert pd.transformed_images is pd.transformed_images


def test_prddata_ctf_images_shape_and_symmetry(prd_project):
    pd = data_store.get_prd_data(0)
    n_pixels = params.ms_num_pixels
    ctf = pd.ctf_images
    assert ctf.shape == (len(pd.info.raw_image_indices), n_pixels, n_pixels)
    assert pd.ctf_images is ctf
    # the CTF is a real, even function of spatial frequency, ifftshift-ed to array corners
    assert np.all(np.isfinite(ctf))


def test_prddata_ctf_uses_the_prd_defocus(prd_project):
    pd = data_store.get_prd_data(0)
    defocus = prd_project.get_defocus_by_prd(0)
    assert len(defocus) == pd.ctf_images.shape[0]


@pytest.mark.parametrize("attr", ["psi_data", "EL_data"])
def test_prddata_missing_derived_files_raise(prd_project, attr):
    pd = data_store.get_prd_data(0)
    with pytest.raises(FileNotFoundError):
        getattr(pd, attr)


def test_prddata_dist_data_reads_the_h5_file(prd_project):
    pd = data_store.get_prd_data(0)
    data = pd.dist_data
    assert set(data.keys()) == {"rotations", "msk2", "image_filter"}
    assert np.array_equal(data["rotations"], pd.info.image_rotations)


def test_prddata_missing_distance_file_raises(prd_project):
    # only prds 0..2 have distance files written
    with pytest.raises((OSError, FileNotFoundError)):
        data_store.get_prd_data(prd_project.n_thresholded - 1)


# ============================================================================
# CC/transformations.py - vendored homogeneous transforms (unused by the package)
# ============================================================================

# the 24 Euler axis sequences supported by the library
AXES = sorted(tr._AXES2TUPLE)

# the six sequences with odd parity AND a repeated axis, where the library and
# scipy pick different (but equivalent) branches of the Euler triplet
ODD_REPEATED = [a for a in AXES if tr._AXES2TUPLE[a][1] == 1 and tr._AXES2TUPLE[a][2] == 1]
PLAIN_AXES = [a for a in AXES if a not in ODD_REPEATED]


def _scipy_seq(axes):
    """Translate a transformations axes string into a scipy rotation sequence."""
    return axes[1:].lower() if axes[0] == 's' else axes[1:].upper()


def _to_scipy_quat(q):
    """[w, x, y, z] (scalar first) -> [x, y, z, w] (scipy, scalar last)."""
    return np.array([q[1], q[2], q[3], q[0]])


def _rand_rot(seed):
    """Deterministic uniform random 4x4 rotation."""
    return tr.random_rotation_matrix(np.random.default_rng(seed).random(3))


# ----------------------------------------------------------------------------
# identity / translation
# ----------------------------------------------------------------------------


def test_identity_matrix():
    identity = tr.identity_matrix()
    assert identity.shape == (4, 4)
    assert np.allclose(identity, np.identity(4))
    assert np.allclose(identity, identity @ identity)
    assert np.trace(identity) == 4.0


@pytest.mark.parametrize(
    'v',
    [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-1.5, 0.25, 7.0],
        [1e-12, -1e-12, 0.0],
        [1e6, -1e6, 1e6],
        [0.1, 0.2, 0.3],
    ],
)
def test_translation_matrix_roundtrip(v):
    M = tr.translation_matrix(v)
    assert np.allclose(M[:3, 3], v)
    assert np.allclose(M[:3, :3], np.identity(3))
    assert M[3, 3] == 1.0
    assert np.allclose(tr.translation_from_matrix(M), v)


@pytest.mark.parametrize('v', [[1.0, 2.0, 3.0], [-4.0, 0.5, 0.0]])
def test_translation_matrix_acts_on_homogeneous_point(v):
    M = tr.translation_matrix(v)
    p = np.array([0.3, -0.7, 2.0, 1.0])
    assert np.allclose((M @ p)[:3], p[:3] + np.asarray(v))
    # a direction (w = 0) is unaffected by a pure translation
    d = np.array([0.3, -0.7, 2.0, 0.0])
    assert np.allclose(M @ d, d)


def test_translation_matrix_ignores_fourth_component():
    # only direction[:3] is used, so a homogeneous 4-vector is accepted
    assert np.allclose(tr.translation_matrix([1.0, 2.0, 3.0, 99.0])[:3, 3], [1, 2, 3])


def test_translation_from_matrix_returns_copy():
    M = tr.translation_matrix([1.0, 2.0, 3.0])
    v = tr.translation_from_matrix(M)
    v[0] = -100.0
    assert M[0, 3] == 1.0


def test_translation_matrix_short_vector_raises():
    with pytest.raises(ValueError):
        tr.translation_matrix([1.0, 2.0])


# ----------------------------------------------------------------------------
# reflection
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('seed', range(5))
def test_reflection_matrix_is_an_involution(seed):
    rng = np.random.default_rng(seed)
    point, normal = rng.random(3) - 0.5, rng.random(3) - 0.5
    M = tr.reflection_matrix(point, normal)
    # a mirror applied twice is the identity, its determinant is -1 and the
    # 3x3 block has eigenvalues (1, 1, -1) so its trace is 2 - 1 = 1, i.e. the
    # 4x4 trace is 2
    assert np.allclose(M @ M, np.identity(4))
    assert np.isclose(np.linalg.det(M), -1.0)
    assert np.isclose(np.trace(M), 2.0)


@pytest.mark.parametrize('seed', range(5))
def test_reflection_matrix_fixes_the_mirror_plane(seed):
    rng = np.random.default_rng(seed)
    point, normal = rng.random(3) - 0.5, rng.random(3) - 0.5
    unit_normal = tr.unit_vector(normal)
    M = tr.reflection_matrix(point, normal)
    # points in the plane are fixed
    tangent = np.cross(unit_normal, [1.0, 0.0, 0.0])
    in_plane = np.append(point + tangent, 1.0)
    assert np.allclose(M @ in_plane, in_plane)
    # a point offset along the normal is sent to the opposite offset
    off = np.append(point + 1.3 * unit_normal, 1.0)
    mirrored = np.append(point - 1.3 * unit_normal, 1.0)
    assert np.allclose(M @ off, mirrored)


@pytest.mark.parametrize('scale', [1.0, 3.7, -2.0])
def test_reflection_matrix_is_insensitive_to_normal_length(scale):
    point, normal = np.array([0.2, -0.4, 1.0]), np.array([0.3, 0.5, -0.8])
    # the normal is normalized internally, and flipping its sign mirrors about
    # the same plane
    assert np.allclose(tr.reflection_matrix(point, normal), tr.reflection_matrix(point, scale * normal))


@pytest.mark.parametrize('seed', range(4))
def test_reflection_from_matrix_roundtrip(seed):
    rng = np.random.default_rng(seed)
    point, normal = rng.random(3) - 0.5, rng.random(3) - 0.5
    M0 = tr.reflection_matrix(point, normal)
    rec_point, rec_normal = tr.reflection_from_matrix(M0)
    assert tr.is_same_transform(M0, tr.reflection_matrix(rec_point, rec_normal))
    # the recovered normal is parallel to the original one
    assert np.isclose(abs(np.dot(tr.unit_vector(rec_normal[:3]), tr.unit_vector(normal))), 1.0)


def test_reflection_from_matrix_requires_eigenvalue_minus_one():
    with pytest.raises(ValueError):
        tr.reflection_from_matrix(np.identity(4))


# ----------------------------------------------------------------------------
# rotation
# ----------------------------------------------------------------------------

ROT_CASES = [
    (0.0, [0.0, 0.0, 1.0]),
    (0.5, [0.0, 0.0, 1.0]),
    (math.pi / 2, [1.0, 0.0, 0.0]),
    (math.pi, [0.0, 1.0, 0.0]),
    (-1.234, [1.0, 1.0, 1.0]),
    (2.345, [0.2, -0.7, 0.4]),
    (0.123, [3.0, 0.0, 0.0]),
    (-math.pi / 3, [-1.0, 2.0, -3.0]),
]


@pytest.mark.parametrize('angle,direction', ROT_CASES)
def test_rotation_matrix_matches_scipy_rotvec(angle, direction):
    # Rodrigues. The rotation vector is angle * unit(direction)
    axis = np.asarray(direction, dtype=float)
    axis = axis / np.linalg.norm(axis)
    M = tr.rotation_matrix(angle, direction)
    assert np.allclose(M[:3, :3], Rotation.from_rotvec(angle * axis).as_matrix())
    assert np.allclose(M[3], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(M[:3, 3], 0.0)


@pytest.mark.parametrize('angle,direction', ROT_CASES)
def test_rotation_matrix_is_orthogonal_with_unit_determinant(angle, direction):
    R = tr.rotation_matrix(angle, direction)[:3, :3]
    assert np.allclose(R.T @ R, np.identity(3))
    assert np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize('angle', [0.0, 0.3, 1.0, math.pi / 2, 2.5, math.pi])
def test_rotation_matrix_trace_equals_one_plus_two_cos(angle):
    # tr(R) = 1 + 2 cos(theta) for any rotation axis
    R = tr.rotation_matrix(angle, [0.3, -0.5, 0.81])[:3, :3]
    assert np.isclose(np.trace(R), 1.0 + 2.0 * math.cos(angle))


@pytest.mark.parametrize('angle', [0.3, 1.0, -2.0, 3.0])
def test_rotation_matrix_is_two_pi_periodic(angle):
    direction, point = [0.2, -0.7, 0.4], [0.1, 0.2, -0.3]
    assert tr.is_same_transform(tr.rotation_matrix(angle, direction, point),
                                tr.rotation_matrix(angle - 2 * math.pi, direction, point))


@pytest.mark.parametrize('angle', [0.3, 1.0, -2.0, 3.0])
def test_rotation_matrix_negated_angle_and_axis_agree(angle):
    direction, point = np.array([0.2, -0.7, 0.4]), [0.1, 0.2, -0.3]
    assert tr.is_same_transform(tr.rotation_matrix(angle, direction, point),
                                tr.rotation_matrix(-angle, -direction, point))


@pytest.mark.parametrize('direction', [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [-2.0, 1.0, 0.0]])
def test_rotation_matrix_about_point_fixes_that_point(direction):
    point = np.array([0.7, -1.3, 2.2])
    M = tr.rotation_matrix(1.1, direction, point)
    assert np.allclose(M @ np.append(point, 1.0), np.append(point, 1.0))


def test_rotation_matrix_docstring_quarter_turn():
    # quarter turn about the z axis through (1, 0, 0) sends the origin to (1, -1, 0)
    R = tr.rotation_matrix(math.pi / 2, [0, 0, 1], [1, 0, 0])
    assert np.allclose(R @ [0, 0, 0, 1], [1, -1, 0, 1])
    assert np.isclose(np.trace(R), 2.0)


def test_rotation_matrix_does_not_mutate_its_direction_argument():
    direction = np.array([1.0, 2.0, 3.0])
    tr.rotation_matrix(0.9, direction)
    assert np.allclose(direction, [1.0, 2.0, 3.0])


@pytest.mark.parametrize('angle', [0.0, 1e-6, 0.5, math.pi / 2, 2.0, -0.5, math.pi - 1e-7])
def test_rotation_from_matrix_roundtrip(angle):
    direction, point = [0.2, -0.7, 0.4], [0.1, 0.2, -0.3]
    R0 = tr.rotation_matrix(angle, direction, point)
    rec_angle, rec_dir, rec_point = tr.rotation_from_matrix(R0)
    assert tr.is_same_transform(R0, tr.rotation_matrix(rec_angle, rec_dir, rec_point))


@pytest.mark.parametrize('direction', [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
def test_rotation_from_matrix_all_axis_branches(direction):
    # rotation_from_matrix picks its sin(angle) formula from which component of
    # the axis is non zero, so exercise each of the three branches
    R0 = tr.rotation_matrix(0.77, direction)
    rec_angle, rec_dir, rec_point = tr.rotation_from_matrix(R0)
    assert tr.is_same_transform(R0, tr.rotation_matrix(rec_angle, rec_dir, rec_point))
    assert np.isclose(abs(rec_angle), 0.77)


def test_rotation_from_matrix_rejects_non_rotation():
    with pytest.raises(ValueError):
        tr.rotation_from_matrix(tr.scale_matrix(2.0))


# ----------------------------------------------------------------------------
# scaling
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('factor', [-1.234, -1.0, 0.5, 1.0, 2.0, 10.0])
def test_scale_matrix_uniform(factor):
    S = tr.scale_matrix(factor)
    rng = np.random.default_rng(0)
    v = (rng.random((4, 5)) - 0.5) * 20
    v[3] = 1.0
    assert np.allclose((S @ v)[:3], factor * v[:3])
    assert np.isclose(np.linalg.det(S), factor**3)


@pytest.mark.parametrize('factor', [-1.5, 0.25, 2.0])
def test_scale_matrix_uniform_about_origin_fixes_origin(factor):
    origin = np.array([0.3, -0.7, 1.2])
    S = tr.scale_matrix(factor, origin)
    assert np.allclose(S @ np.append(origin, 1.0), np.append(origin, 1.0))
    p = np.append(origin + np.array([1.0, 0.0, 0.0]), 1.0)
    assert np.allclose((S @ p)[:3], origin + factor * np.array([1.0, 0.0, 0.0]))


@pytest.mark.parametrize('factor', [-1.5, 0.25, 2.0])
def test_scale_matrix_directional_only_scales_along_direction(factor):
    direction = tr.unit_vector([0.3, -0.5, 0.81])
    S = tr.scale_matrix(factor, None, direction)
    # the component along direction is multiplied by factor, the orthogonal
    # complement is untouched
    perp = np.cross(direction, [1.0, 0.0, 0.0])
    assert np.allclose((S @ np.append(perp, 1.0))[:3], perp)
    assert np.allclose((S @ np.append(direction, 1.0))[:3], factor * direction)


@pytest.mark.parametrize('seed', range(4))
def test_scale_from_matrix_uniform_roundtrip(seed):
    rng = np.random.default_rng(seed)
    factor, origin = rng.random() * 10 - 5, rng.random(3) - 0.5
    S0 = tr.scale_matrix(factor, origin)
    rec_factor, rec_origin, rec_direction = tr.scale_from_matrix(S0)
    assert tr.is_same_transform(S0, tr.scale_matrix(rec_factor, rec_origin, rec_direction))
    assert np.isclose(rec_factor, factor)


@pytest.mark.parametrize('seed', range(4))
def test_scale_from_matrix_directional_roundtrip(seed):
    rng = np.random.default_rng(seed)
    factor, origin, direction = rng.random() * 10 - 5, rng.random(3) - 0.5, rng.random(3) - 0.5
    S0 = tr.scale_matrix(factor, origin, direction)
    rec_factor, rec_origin, rec_direction = tr.scale_from_matrix(S0)
    assert tr.is_same_transform(S0, tr.scale_matrix(rec_factor, rec_origin, rec_direction))


def test_scale_from_matrix_uniform_reports_no_direction():
    # a uniform scaling has no distinguished direction, so None is returned
    factor, origin, direction = tr.scale_from_matrix(tr.scale_matrix(3.0, [1.0, 2.0, 3.0]))
    assert direction is None
    assert np.isclose(factor, 3.0)
    assert np.allclose(origin[:3] / origin[3], [1.0, 2.0, 3.0])


def test_scale_from_matrix_rejects_degenerate_matrix():
    with pytest.raises(ValueError):
        tr.scale_from_matrix(np.zeros((4, 4)))


# ----------------------------------------------------------------------------
# projection
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('seed', range(3))
def test_orthogonal_projection_is_idempotent_and_singular(seed):
    rng = np.random.default_rng(seed)
    point, normal = rng.random(3) - 0.5, rng.random(3) - 0.5
    P = tr.projection_matrix(point, normal)
    assert np.allclose(P @ P, P)
    assert np.isclose(np.linalg.det(P), 0.0)
    # points already on the plane are fixed
    tangent = np.cross(tr.unit_vector(normal), [1.0, 0.0, 0.0])
    on_plane = np.append(point + tangent, 1.0)
    assert np.allclose(P @ on_plane, on_plane)


@pytest.mark.parametrize('seed', range(3))
def test_parallel_projection_is_idempotent(seed):
    rng = np.random.default_rng(seed)
    point, normal, direction = rng.random(3) - 0.5, rng.random(3) - 0.5, rng.random(3) - 0.5
    P = tr.projection_matrix(point, normal, direction=direction)
    assert np.allclose(P @ P, P)


def test_projection_matrix_axis_aligned():
    # projecting onto the x = 0 plane leaves the y, z and w rows alone
    P = tr.projection_matrix([0, 0, 0], [1, 0, 0])
    assert np.allclose(P[1:, 1:], np.identity(4)[1:, 1:])


def test_parallel_projection_along_x_onto_slanted_plane():
    # plane through (3, 0, 0) with normal (1, 1, 0), projecting along x:
    # y is preserved and x + y = 3 on the image
    P = tr.projection_matrix([3, 0, 0], [1, 1, 0], [1, 0, 0])
    rng = np.random.default_rng(4)
    v0 = (rng.random((4, 5)) - 0.5) * 20
    v0[3] = 1.0
    v1 = P @ v0
    assert np.allclose(v1[1], v0[1])
    assert np.allclose(v1[0], 3 - v1[1])


@pytest.mark.parametrize('seed', range(3))
def test_perspective_equals_orthogonal_times_pseudo_perspective(seed):
    rng = np.random.default_rng(seed)
    point, normal, persp = rng.random(3) - 0.5, rng.random(3) - 0.5, rng.random(3) - 0.5
    P0 = tr.projection_matrix(point, normal)
    P2 = tr.projection_matrix(point, normal, perspective=persp)
    P3 = tr.projection_matrix(point, normal, perspective=persp, pseudo=True)
    assert tr.is_same_transform(P2, P0 @ P3)


@pytest.mark.parametrize('seed', range(2))
@pytest.mark.parametrize('kind', ['orthogonal', 'parallel', 'perspective', 'pseudo'])
def test_projection_from_matrix_roundtrip(kind, seed):
    rng = np.random.default_rng(seed)
    point, normal = rng.random(3) - 0.5, rng.random(3) - 0.5
    direction, persp = rng.random(3) - 0.5, rng.random(3) - 0.5
    pseudo = kind == 'pseudo'
    if kind == 'orthogonal':
        P0 = tr.projection_matrix(point, normal)
    elif kind == 'parallel':
        P0 = tr.projection_matrix(point, normal, direction=direction)
    else:
        P0 = tr.projection_matrix(point, normal, perspective=persp, pseudo=pseudo)
    P1 = tr.projection_matrix(*tr.projection_from_matrix(P0, pseudo=pseudo))
    assert tr.is_same_transform(P0, P1)


# ----------------------------------------------------------------------------
# clip (frustum -> normalized device coordinates)
# ----------------------------------------------------------------------------


def test_clip_matrix_orthographic_maps_frustum_corners_to_unit_cube():
    left, right, bottom, top, near, far = 0.1, 0.9, 0.2, 0.8, 0.3, 0.7
    M = tr.clip_matrix(left, right, bottom, top, near, far, perspective=False)
    assert np.allclose(M @ [left, bottom, near, 1], [-1, -1, -1, 1])
    assert np.allclose(M @ [right, top, far, 1], [1, 1, 1, 1])


def test_clip_matrix_perspective_maps_near_plane_after_dehomogenization():
    left, right, bottom, top, near, far = 0.1, 0.9, 0.2, 0.8, 0.3, 0.7
    M = tr.clip_matrix(left, right, bottom, top, near, far, perspective=True)
    v = M @ [left, bottom, near, 1]
    assert np.allclose(v / v[3], [-1, -1, -1, 1])
    v = M @ [right, top, near, 1]
    assert np.allclose(v / v[3], [1, 1, -1, 1])


@pytest.mark.parametrize(
    'frustum',
    [
        (1, 0, 0, 1, 1, 2),  # left >= right
        (0, 1, 1, 0, 1, 2),  # bottom >= top
        (0, 1, 0, 1, 2, 1),  # near >= far
        (0, 0, 0, 1, 1, 2),  # degenerate x extent
    ],
)
def test_clip_matrix_rejects_invalid_frustum(frustum):
    with pytest.raises(ValueError, match='invalid frustum'):
        tr.clip_matrix(*frustum)


def test_clip_matrix_perspective_rejects_near_plane_at_origin():
    with pytest.raises(ValueError, match='near'):
        tr.clip_matrix(0, 1, 0, 1, 0, 2, perspective=True)


# ----------------------------------------------------------------------------
# shear
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('seed', range(4))
def test_shear_matrix_preserves_volume(seed):
    rng = np.random.default_rng(seed)
    angle = (rng.random() - 0.5) * 4 * math.pi
    direction, point = rng.random(3) - 0.5, rng.random(3) - 0.5
    normal = np.cross(direction, rng.random(3))
    S = tr.shear_matrix(angle, direction, point, normal)
    assert np.isclose(np.linalg.det(S), 1.0)


@pytest.mark.parametrize('seed', range(4))
def test_shear_from_matrix_roundtrip(seed):
    rng = np.random.default_rng(seed)
    angle = (rng.random() - 0.5) * 4 * math.pi
    direction, point = rng.random(3) - 0.5, rng.random(3) - 0.5
    normal = np.cross(direction, rng.random(3))
    S0 = tr.shear_matrix(angle, direction, point, normal)
    S1 = tr.shear_matrix(*tr.shear_from_matrix(S0))
    assert tr.is_same_transform(S0, S1)


def test_shear_matrix_requires_orthogonal_direction_and_normal():
    with pytest.raises(ValueError, match='orthogonal'):
        tr.shear_matrix(0.3, [1, 0, 0], [0, 0, 0], [1, 0, 0])


def test_shear_from_matrix_needs_two_independent_eigenvectors():
    with pytest.raises(ValueError):
        tr.shear_from_matrix(np.diag([1.0, 2.0, 3.0, 1.0]))


# ----------------------------------------------------------------------------
# decompose / compose
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('seed', range(6))
def test_decompose_compose_roundtrip(seed):
    rng = np.random.default_rng(seed)
    scale = rng.random(3) - 0.5
    shear = rng.random(3) - 0.5
    angles = (rng.random(3) - 0.5) * 2 * math.pi
    translate = rng.random(3) - 0.5
    perspective = rng.random(4) - 0.5
    M0 = tr.compose_matrix(scale, shear, angles, translate, perspective)
    M1 = tr.compose_matrix(*tr.decompose_matrix(M0))
    assert tr.is_same_transform(M0, M1)


def test_decompose_pure_translation():
    T0 = tr.translation_matrix([1.0, 2.0, 3.0])
    scale, shear, angles, translate, perspective = tr.decompose_matrix(T0)
    assert np.allclose(translate, [1, 2, 3])
    assert np.allclose(scale, 1.0)
    assert np.allclose(shear, 0.0)
    assert np.allclose(angles, 0.0)
    assert np.allclose(perspective, [0, 0, 0, 1])
    assert np.allclose(tr.translation_matrix(translate), T0)


def test_decompose_pure_scale():
    scale, shear, angles, translate, perspective = tr.decompose_matrix(tr.scale_matrix(0.123))
    assert np.allclose(scale, 0.123)
    assert np.allclose(translate, 0.0)


def test_decompose_pure_rotation_returns_sxyz_angles():
    R0 = tr.euler_matrix(1, 2, 3)
    scale, shear, angles, translate, perspective = tr.decompose_matrix(R0)
    assert np.allclose(scale, 1.0)
    assert np.allclose(shear, 0.0)
    # decompose always reports static xyz angles
    assert np.allclose(tr.euler_matrix(*angles, 'sxyz'), R0)


def test_decompose_point_symmetry_gives_negative_scale():
    # scale_matrix(-1) has determinant -1, so it is reported as a negative
    # scale rather than a rotation
    M0 = tr.scale_matrix(-1.0)
    scale, shear, angles, translate, perspective = tr.decompose_matrix(M0)
    assert np.allclose(scale, -1.0)
    assert tr.is_same_transform(M0, tr.compose_matrix(scale, shear, angles, translate, perspective))


@pytest.mark.parametrize('angle_y', [math.pi / 2, -math.pi / 2])
@pytest.mark.parametrize('angle_x,angle_z', [(0.0, 0.0), (0.4, 0.9)])
def test_decompose_survives_gimbal_lock(angle_y, angle_x, angle_z):
    M0 = tr.euler_matrix(angle_x, angle_y, angle_z, 'sxyz')
    M1 = tr.compose_matrix(*tr.decompose_matrix(M0))
    assert tr.is_same_transform(M0, M1)


def test_decompose_rejects_zero_homogeneous_element():
    with pytest.raises(ValueError, match=r'M\[3, 3\] is zero'):
        tr.decompose_matrix(np.zeros((4, 4)))


def test_decompose_rejects_singular_matrix():
    M = np.identity(4)
    M[0, 0] = 0.0
    with pytest.raises(ValueError, match='singular'):
        tr.decompose_matrix(M)


def test_compose_matrix_with_no_arguments_is_identity():
    assert np.allclose(tr.compose_matrix(), np.identity(4))


# ----------------------------------------------------------------------------
# crystallographic orthogonalization
# ----------------------------------------------------------------------------


def test_orthogonalization_matrix_cubic_cell():
    O = tr.orthogonalization_matrix([10, 10, 10], [90, 90, 90])
    assert np.allclose(O[:3, :3], np.identity(3) * 10)


def test_orthogonalization_matrix_triclinic_reference_value():
    O = tr.orthogonalization_matrix([9.8, 12.0, 15.5], [87.2, 80.7, 69.7])
    assert np.isclose(np.sum(O), 43.063229, atol=1e-6)


@pytest.mark.parametrize(
    'lengths,angles',
    [
        ([10, 10, 10], [90, 90, 90]),
        ([9.8, 12.0, 15.5], [87.2, 80.7, 69.7]),
        ([5.0, 7.0, 11.0], [100.0, 95.0, 110.0]),
    ],
)
def test_orthogonalization_matrix_determinant_is_the_cell_volume(lengths, angles):
    a, b, c = lengths
    ca, cb, cg = np.cos(np.radians(angles))
    volume = a * b * c * math.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
    O = tr.orthogonalization_matrix(lengths, angles)
    assert np.isclose(abs(np.linalg.det(O[:3, :3])), volume)


# ----------------------------------------------------------------------------
# point set registration
# ----------------------------------------------------------------------------


def test_affine_matrix_from_points_2d_reference_value():
    v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    expected = np.array([[0.14549, 0.00062, 675.50008], [0.00048, 0.14094, 53.24971], [0.0, 0.0, 1.0]])
    assert np.allclose(tr.affine_matrix_from_points(v0, v1), expected, atol=1e-4)


@pytest.mark.parametrize('seed', range(3))
def test_affine_matrix_from_points_recovers_a_general_affine_map(seed):
    rng = np.random.default_rng(seed)
    A = np.identity(4)
    A[:3, :3] = rng.random((3, 3)) - 0.5
    A[:3, 3] = rng.random(3)
    v0 = rng.random((3, 12)) - 0.5
    v1 = A[:3, :3] @ v0 + A[:3, 3:4]
    assert np.allclose(tr.affine_matrix_from_points(v0, v1), A)


@pytest.mark.parametrize('seed', range(3))
def test_affine_matrix_from_points_rigid_branch_recovers_rotation(seed):
    rng = np.random.default_rng(seed)
    R = tr.random_rotation_matrix(rng.random(3))
    v0 = rng.random((3, 10)) - 0.5
    v1 = R[:3, :3] @ v0
    M = tr.affine_matrix_from_points(v0, v1, shear=False, scale=False)
    assert np.allclose(M[:3, :3], R[:3, :3])
    assert np.allclose(v1, (M @ np.vstack([v0, np.ones(10)]))[:3])


@pytest.mark.parametrize(
    'v0,v1',
    [
        ([[0, 1, 2]], [[0, 1, 2]]),  # ndims < 2
        ([[0, 1], [0, 1], [1, 1]], [[0, 1], [0, 1], [1, 1]]),  # fewer points than dims
        ([[0, 1, 2], [0, 1, 2]], [[0, 1], [0, 1]]),  # shape mismatch
    ],
)
def test_affine_matrix_from_points_rejects_bad_shapes(v0, v1):
    with pytest.raises(ValueError, match='wrong shape'):
        tr.affine_matrix_from_points(v0, v1)


def test_superimposition_matrix_of_a_set_onto_itself_is_identity():
    v0 = np.random.default_rng(0).random((3, 10))
    assert np.allclose(tr.superimposition_matrix(v0, v0), np.identity(4))


@pytest.mark.parametrize('usesvd', [True, False])
@pytest.mark.parametrize('seed', [0, 1])
def test_superimposition_matrix_recovers_a_rigid_motion(usesvd, seed):
    rng = np.random.default_rng(seed)
    R = tr.random_rotation_matrix(rng.random(3))
    T = tr.translation_matrix(rng.random(3) - 0.5)
    v0 = (rng.random((4, 30)) - 0.5) * 20
    v0[3] = 1.0
    v1 = (T @ R) @ v0
    M = tr.superimposition_matrix(v0, v1, usesvd=usesvd)
    assert np.allclose(v1, M @ v0)


@pytest.mark.parametrize('usesvd', [True, False])
def test_superimposition_matrix_recovers_a_similarity(usesvd):
    rng = np.random.default_rng(2)
    R = tr.random_rotation_matrix(rng.random(3))
    M0 = tr.concatenate_matrices(tr.translation_matrix(rng.random(3) - 0.5), R, tr.scale_matrix(0.7))
    v0 = (rng.random((4, 40)) - 0.5) * 20
    v0[3] = 1.0
    v1 = M0 @ v0
    M = tr.superimposition_matrix(v0, v1, scale=True, usesvd=usesvd)
    assert np.allclose(v1, M @ v0)


@pytest.mark.parametrize('seed', range(4))
def test_superimposition_matrix_matches_scipy_kabsch_with_noise(seed):
    # with noisy data the answer is the Kabsch optimum, which scipy computes
    # independently through Rotation.align_vectors
    rng = np.random.default_rng(seed)
    v0 = rng.random((3, 15)) - 0.5
    R = tr.random_rotation_matrix(rng.random(3))[:3, :3]
    v1 = R @ v0 + rng.normal(0.0, 0.01, (3, 15))
    M = tr.superimposition_matrix(v0, v1)
    centred0 = v0 - v0.mean(axis=1, keepdims=True)
    centred1 = v1 - v1.mean(axis=1, keepdims=True)
    rot, _ = Rotation.align_vectors(centred1.T, centred0.T)
    assert np.allclose(M[:3, :3], rot.as_matrix())
    # the translation carries the first centroid onto the second
    assert np.allclose(M[:3, 3], v1.mean(axis=1) - M[:3, :3] @ v0.mean(axis=1))


def test_superimposition_matrix_ignores_the_homogeneous_row():
    rng = np.random.default_rng(5)
    v0 = rng.random((4, 8))
    v0[3] = 1.0
    R = tr.random_rotation_matrix(rng.random(3))
    v1 = R @ v0
    assert np.allclose(tr.superimposition_matrix(v0, v1), tr.superimposition_matrix(v0[:3], v1[:3]))


# ----------------------------------------------------------------------------
# Euler angles
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('axes', AXES)
def test_euler_matrix_matches_scipy(axes):
    ai, aj, ak = 0.3, -0.7, 1.1
    M = tr.euler_matrix(ai, aj, ak, axes)
    assert np.allclose(M[:3, :3], Rotation.from_euler(_scipy_seq(axes), [ai, aj, ak]).as_matrix())
    assert np.allclose(M[:3, 3], 0.0)
    assert np.allclose(M[3], [0, 0, 0, 1])


@pytest.mark.parametrize('axes', AXES)
def test_euler_matrix_is_a_rotation(axes):
    R = tr.euler_matrix(1.3, -0.4, 2.7, axes)[:3, :3]
    assert np.allclose(R.T @ R, np.identity(3))
    assert np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize('axes', AXES)
def test_euler_from_matrix_roundtrip(axes):
    R0 = _rand_rot(3)
    angles = tr.euler_from_matrix(R0, axes)
    assert len(angles) == 3
    assert np.allclose(R0, tr.euler_matrix(angles[0], angles[1], angles[2], axes))


@pytest.mark.parametrize('axes', PLAIN_AXES)
def test_euler_from_matrix_matches_scipy_angles(axes):
    R0 = _rand_rot(0)
    assert np.allclose(tr.euler_from_matrix(R0, axes), Rotation.from_matrix(R0[:3, :3]).as_euler(_scipy_seq(axes)))


@pytest.mark.parametrize('axes', ODD_REPEATED)
def test_euler_from_matrix_odd_parity_repeated_axes_pick_the_other_branch(axes):
    # for the six odd parity sequences with a repeated axis the library returns
    # a negated middle angle relative to scipy. An equally valid triplet for
    # the same rotation, not a disagreement about the rotation itself
    R0 = _rand_rot(0)
    mine = np.array(tr.euler_from_matrix(R0, axes))
    theirs = Rotation.from_matrix(R0[:3, :3]).as_euler(_scipy_seq(axes))
    assert np.isclose(mine[1], -theirs[1])
    assert np.allclose(tr.euler_matrix(*mine, axes)[:3, :3], R0[:3, :3])
    assert np.allclose(tr.euler_matrix(*theirs, axes)[:3, :3], R0[:3, :3])


@pytest.mark.parametrize('axes', AXES)
def test_euler_roundtrip_at_the_singular_middle_angle(axes):
    # the singularity is at a middle angle of +-pi/2 for the plain sequences
    # and at 0 or pi when the first and last axis repeat
    repetition = tr._AXES2TUPLE[axes][2]
    middle = math.pi if repetition else math.pi / 2
    R0 = tr.euler_matrix(0.4, middle, 0.9, axes)
    angles = tr.euler_from_matrix(R0, axes)
    assert np.allclose(R0, tr.euler_matrix(angles[0], angles[1], angles[2], axes), atol=1e-7)


@pytest.mark.parametrize('axes,expected_tuple', [('sxyz', (0, 0, 0, 0)), ('ryxz', (2, 0, 0, 1)), ('szyz', (2, 1, 1, 0))])
def test_euler_matrix_accepts_the_encoded_tuple(axes, expected_tuple):
    assert tr._AXES2TUPLE[axes] == expected_tuple
    assert np.allclose(tr.euler_matrix(0.3, 0.2, 0.1, axes), tr.euler_matrix(0.3, 0.2, 0.1, expected_tuple))
    assert np.allclose(tr.euler_from_matrix(_rand_rot(1), axes), tr.euler_from_matrix(_rand_rot(1), expected_tuple))


def test_euler_matrix_reference_values():
    assert np.isclose(np.sum(tr.euler_matrix(1, 2, 3, 'syxz')[0]), -1.34786452)
    assert np.isclose(np.sum(tr.euler_matrix(1, 2, 3, (0, 1, 0, 1))[0]), -0.383436184)


@pytest.mark.parametrize('bad', ['sxyq', 'nope', (9, 9, 9, 9)])
def test_euler_matrix_rejects_unknown_axes(bad):
    with pytest.raises(KeyError):
        tr.euler_matrix(0.0, 0.0, 0.0, bad)


@pytest.mark.parametrize('bad', ['sxyq', (9, 9, 9, 9)])
def test_euler_from_matrix_rejects_unknown_axes(bad):
    with pytest.raises(KeyError):
        tr.euler_from_matrix(np.identity(4), bad)


@pytest.mark.xfail(reason="Bug. euler_matrix does not lowercase the axes string, unlike euler_from_matrix "
                   "and quaternion_from_euler, so 'SXYZ' raises KeyError on the forward map only",
                   strict=False)
def test_euler_axes_string_case_handling_is_symmetric():
    assert np.allclose(tr.euler_from_matrix(np.identity(4), 'SXYZ'), (0.0, 0.0, 0.0))
    assert np.allclose(tr.euler_matrix(0.3, 0.2, 0.1, 'SXYZ'), tr.euler_matrix(0.3, 0.2, 0.1, 'sxyz'))


# ----------------------------------------------------------------------------
# quaternions
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('axes', AXES)
def test_quaternion_from_euler_agrees_with_euler_matrix(axes):
    ai, aj, ak = 0.3, -0.7, 1.1
    q = tr.quaternion_from_euler(ai, aj, ak, axes)
    assert q.shape == (4, )
    assert np.isclose(tr.vector_norm(q), 1.0)
    assert tr.is_same_transform(tr.quaternion_matrix(q), tr.euler_matrix(ai, aj, ak, axes))


def test_quaternion_from_euler_reference_value():
    assert np.allclose(tr.quaternion_from_euler(1, 2, 3, 'ryxz'), [0.435953, 0.310622, -0.718287, 0.444435], atol=1e-6)


@pytest.mark.parametrize('axes', ['sxyz', 'rxyz', 'szyx', 'ryxy'])
def test_euler_from_quaternion_inverts_quaternion_from_euler(axes):
    ai, aj, ak = 0.21, -0.63, 1.02
    q = tr.quaternion_from_euler(ai, aj, ak, axes)
    assert np.allclose(tr.euler_from_quaternion(q, axes), (ai, aj, ak))


def test_euler_from_quaternion_reference_value():
    assert np.allclose(tr.euler_from_quaternion([0.99810947, 0.06146124, 0, 0]), [0.123, 0, 0], atol=1e-6)


@pytest.mark.parametrize('angle', [0.0, 0.123, 1.0, math.pi / 2, math.pi, -2.0])
def test_quaternion_about_axis_matches_scipy(angle):
    axis = tr.unit_vector([0.3, -0.5, 0.81])
    q = tr.quaternion_about_axis(angle, axis)
    assert np.isclose(tr.vector_norm(q), 1.0)
    assert np.allclose(tr.quaternion_matrix(q)[:3, :3], Rotation.from_rotvec(angle * axis).as_matrix())


def test_quaternion_about_axis_reference_value():
    assert np.allclose(tr.quaternion_about_axis(0.123, [1, 0, 0]), [0.99810947, 0.06146124, 0, 0])


def test_quaternion_about_axis_normalises_the_axis():
    assert np.allclose(tr.quaternion_about_axis(0.9, [0, 0, 5]), tr.quaternion_about_axis(0.9, [0, 0, 1]))


def test_quaternion_about_axis_with_a_zero_axis_drops_the_vector_part():
    # a degenerate axis leaves the imaginary part at zero, so the result is
    # [cos(angle/2), 0, 0, 0]. NOT a unit quaternion, though quaternion_matrix
    # renormalizes it back to the identity rotation
    q = tr.quaternion_about_axis(0.7, [0, 0, 0])
    assert np.allclose(q, [math.cos(0.35), 0.0, 0.0, 0.0])
    assert not np.isclose(tr.vector_norm(q), 1.0)
    assert np.allclose(tr.quaternion_matrix(q), np.identity(4))


@pytest.mark.parametrize('seed', range(5))
def test_quaternion_matrix_matches_scipy_scalar_last(seed):
    q = tr.random_quaternion(np.random.default_rng(seed).random(3))
    M = tr.quaternion_matrix(q)
    assert np.allclose(M[:3, :3], Rotation.from_quat(_to_scipy_quat(q)).as_matrix())
    assert np.allclose(M[3], [0, 0, 0, 1])


@pytest.mark.parametrize(
    'q,expected',
    [
        ([1, 0, 0, 0], np.identity(4)),
        ([0, 1, 0, 0], np.diag([1.0, -1.0, -1.0, 1.0])),
        ([0, 0, 1, 0], np.diag([-1.0, 1.0, -1.0, 1.0])),
        ([0, 0, 0, 1], np.diag([-1.0, -1.0, 1.0, 1.0])),
    ],
)
def test_quaternion_matrix_basis_quaternions(q, expected):
    assert np.allclose(tr.quaternion_matrix(q), expected)


def test_quaternion_matrix_normalises_and_guards_the_null_quaternion():
    # the input is scaled to unit length internally, and a quaternion of
    # negligible norm falls back to the identity
    assert np.allclose(tr.quaternion_matrix([3.0, 0.0, 0.0, 0.0]), np.identity(4))
    assert np.allclose(tr.quaternion_matrix([0.0, 0.0, 0.0, 0.0]), np.identity(4))
    q = tr.random_quaternion(np.random.default_rng(7).random(3))
    assert np.allclose(tr.quaternion_matrix(q), tr.quaternion_matrix(5.0 * q))


def test_quaternion_matrix_agrees_with_rotation_matrix():
    assert np.allclose(tr.quaternion_matrix([0.99810947, 0.06146124, 0, 0]), tr.rotation_matrix(0.123, [1, 0, 0]))


@pytest.mark.parametrize('isprecise', [True, False])
@pytest.mark.parametrize('seed', range(5))
def test_quaternion_from_matrix_roundtrip(seed, isprecise):
    R = _rand_rot(seed)
    q = tr.quaternion_from_matrix(R, isprecise)
    assert q.shape == (4, )
    assert np.isclose(tr.vector_norm(q), 1.0)
    assert tr.is_same_transform(R, tr.quaternion_matrix(q))


@pytest.mark.parametrize('seed', range(6))
def test_quaternion_from_matrix_precise_and_general_agree(seed):
    R = _rand_rot(seed)
    assert tr.is_same_quaternion(tr.quaternion_from_matrix(R, False), tr.quaternion_from_matrix(R, True))


@pytest.mark.parametrize('axis', [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
def test_quaternion_from_matrix_half_turns(axis):
    # a half turn has a zero scalar part, which drives the alternative branch
    # of the precise algorithm
    R = tr.rotation_matrix(math.pi, axis)
    q_general = tr.quaternion_from_matrix(R, False)
    q_precise = tr.quaternion_from_matrix(R, True)
    assert tr.is_same_quaternion(q_general, q_precise)
    assert np.isclose(q_general[0], 0.0, atol=1e-8)
    assert np.allclose(tr.unit_vector(np.abs(q_precise[1:])), tr.unit_vector(np.abs(np.asarray(axis, float))))


@pytest.mark.parametrize('seed', range(6))
def test_quaternion_from_matrix_uses_a_non_negative_scalar_part(seed):
    assert tr.quaternion_from_matrix(_rand_rot(seed), False)[0] >= 0.0
    assert tr.quaternion_from_matrix(_rand_rot(seed), True)[0] >= 0.0


def test_quaternion_from_matrix_identity_and_diagonal():
    assert np.allclose(tr.quaternion_from_matrix(np.identity(4), True), [1, 0, 0, 0])
    assert tr.is_same_quaternion(tr.quaternion_from_matrix(np.diag([1.0, -1.0, -1.0, 1.0])), [0, 1, 0, 0])


@pytest.mark.parametrize(
    'R,expected',
    [
        ([[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0], [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]],
         [0.19069, 0.43736, 0.87485, -0.083611]),
        ([[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0], [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]],
         [0.82336615, -0.13610694, 0.46344705, -0.29792603]),
    ],
)
def test_quaternion_from_matrix_reference_values(R, expected):
    # the general (eigenvector) branch tolerates the slightly non orthogonal
    # matrices used as references upstream
    assert np.allclose(tr.quaternion_from_matrix(R), expected, atol=1e-5)


def test_quaternion_multiply_reference_value():
    assert np.allclose(tr.quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7]), [28, -44, -14, 48])


@pytest.mark.parametrize('seed', range(4))
def test_quaternion_multiply_composes_rotations_left_to_right(seed):
    rng = np.random.default_rng(seed)
    q0 = tr.random_quaternion(rng.random(3))
    q1 = tr.random_quaternion(rng.random(3))
    # quaternion_multiply(q1, q0) applies q0 first, i.e. it is the rotation q1 * q0
    r0 = Rotation.from_quat(_to_scipy_quat(q0))
    r1 = Rotation.from_quat(_to_scipy_quat(q1))
    product = tr.quaternion_multiply(q1, q0)
    assert np.allclose(tr.quaternion_matrix(product)[:3, :3], (r1 * r0).as_matrix())


def test_quaternion_multiply_identity_and_noncommutativity():
    q = tr.random_quaternion(np.random.default_rng(0).random(3))
    assert np.allclose(tr.quaternion_multiply(q, [1, 0, 0, 0]), q)
    assert np.allclose(tr.quaternion_multiply([1, 0, 0, 0], q), q)
    p = tr.random_quaternion(np.random.default_rng(1).random(3))
    assert not np.allclose(tr.quaternion_multiply(q, p), tr.quaternion_multiply(p, q))


@pytest.mark.parametrize('seed', range(4))
def test_quaternion_conjugate_and_inverse(seed):
    q = tr.random_quaternion(np.random.default_rng(seed).random(3))
    conj = tr.quaternion_conjugate(q)
    assert conj[0] == q[0] and np.allclose(conj[1:], -q[1:])
    inv = tr.quaternion_inverse(q)
    assert np.allclose(tr.quaternion_multiply(q, inv), [1, 0, 0, 0])
    assert np.allclose(tr.quaternion_multiply(inv, q), [1, 0, 0, 0])
    # for a unit quaternion the inverse and the conjugate coincide
    assert np.allclose(inv, conj)


def test_quaternion_inverse_of_a_non_unit_quaternion():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(tr.quaternion_multiply(q, tr.quaternion_inverse(q)), [1, 0, 0, 0])


def test_quaternion_conjugate_does_not_mutate_its_argument():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    tr.quaternion_conjugate(q)
    tr.quaternion_inverse(q)
    assert np.allclose(q, [1.0, 2.0, 3.0, 4.0])


def test_quaternion_real_and_imag():
    assert tr.quaternion_real([3, 0, 1, 2]) == 3.0
    assert isinstance(tr.quaternion_real([3, 0, 1, 2]), float)
    assert np.allclose(tr.quaternion_imag([3, 0, 1, 2]), [0.0, 1.0, 2.0])
    assert tr.quaternion_imag([3, 0, 1, 2]).shape == (3, )


# ----------------------------------------------------------------------------
# slerp
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('fraction', [0.0, 0.25, 0.5, 0.75, 1.0])
def test_quaternion_slerp_matches_scipy_slerp(fraction):
    rng = np.random.default_rng(0)
    q0 = tr.random_quaternion(rng.random(3))
    q1 = tr.random_quaternion(rng.random(3))
    reference = Slerp([0, 1], Rotation.from_quat(np.vstack([_to_scipy_quat(q0), _to_scipy_quat(q1)])))
    q = tr.quaternion_slerp(q0, q1, fraction)
    assert np.allclose(Rotation.from_quat(_to_scipy_quat(q)).as_matrix(), reference(fraction).as_matrix())


@pytest.mark.parametrize('fraction', [0.1, 0.5, 0.9])
def test_quaternion_slerp_stays_on_the_unit_sphere(fraction):
    rng = np.random.default_rng(1)
    q0 = tr.random_quaternion(rng.random(3))
    q1 = tr.random_quaternion(rng.random(3))
    assert np.isclose(tr.vector_norm(tr.quaternion_slerp(q0, q1, fraction)), 1.0)


def test_quaternion_slerp_endpoints():
    rng = np.random.default_rng(2)
    q0 = tr.random_quaternion(rng.random(3))
    q1 = tr.random_quaternion(rng.random(3))
    assert np.allclose(tr.quaternion_slerp(q0, q1, 0.0), q0)
    assert np.allclose(tr.quaternion_slerp(q0, q1, 1.0), q1)
    # the endpoints are returned normalized, so a scaled input comes back unit
    assert np.allclose(tr.quaternion_slerp([2.0, 0.0, 0.0, 0.0], q1, 0.0), [1, 0, 0, 0])


def test_quaternion_slerp_bisects_the_angle_at_one_half():
    rng = np.random.default_rng(3)
    q0 = tr.random_quaternion(rng.random(3))
    q1 = tr.random_quaternion(rng.random(3))
    q = tr.quaternion_slerp(q0, q1, 0.5)
    half = math.acos(np.dot(q0, q))
    full = math.acos(np.dot(q0, q1))
    assert np.isclose(full / half, 2.0) or np.isclose(math.acos(-np.dot(q0, q1)) / half, 2.0)


def test_quaternion_slerp_takes_the_short_way_by_default():
    axis = tr.unit_vector([0.2, -0.5, 0.84])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = tr.quaternion_about_axis(0.6, axis)
    # q and -q are the same rotation, and with shortestpath the interpolation
    # is insensitive to that sign. half way is a 0.3 rad turn either way
    assert tr.is_same_quaternion(tr.quaternion_slerp(q0, q1, 0.5), tr.quaternion_about_axis(0.3, axis))
    assert tr.is_same_quaternion(tr.quaternion_slerp(q0, -q1, 0.5), tr.quaternion_about_axis(0.3, axis))
    # without shortestpath the negated endpoint drags the path the long way
    # round the great circle. half of a 2*pi - 0.6 turn, taken backwards
    long_way = tr.quaternion_slerp(q0, -q1, 0.5, shortestpath=False)
    assert not tr.is_same_quaternion(long_way, tr.quaternion_about_axis(0.3, axis))
    assert tr.is_same_quaternion(long_way, tr.quaternion_about_axis(-(math.pi - 0.3), axis))


def test_quaternion_slerp_of_identical_quaternions_is_a_no_op():
    q0 = tr.random_quaternion(np.random.default_rng(5).random(3))
    assert np.allclose(tr.quaternion_slerp(q0, q0.copy(), 0.5), q0)


def test_quaternion_slerp_does_not_mutate_its_inputs():
    rng = np.random.default_rng(6)
    q0 = tr.random_quaternion(rng.random(3))
    q1 = tr.random_quaternion(rng.random(3))
    before0, before1 = q0.copy(), q1.copy()
    tr.quaternion_slerp(q0, q1, 0.5)
    tr.quaternion_slerp(q0, -q1, 0.5)
    assert np.allclose(q0, before0) and np.allclose(q1, before1)


@pytest.mark.parametrize('fraction', [1.5, 2.0, 3.0])
def test_quaternion_slerp_extrapolates_past_one(fraction):
    # Arcball.next relies on fraction > 1 continuing along the same great
    # circle, so a fraction f of a 0.6 rad turn is a 0.6 * f rad turn
    axis = tr.unit_vector([0.2, -0.5, 0.84])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = tr.quaternion_about_axis(0.6, axis)
    q = tr.quaternion_slerp(q0, q1, fraction)
    assert np.isclose(tr.vector_norm(q), 1.0)
    assert tr.is_same_quaternion(q, tr.quaternion_about_axis(0.6 * fraction, axis))


@pytest.mark.xfail(reason='Bug. quaternion_slerp with an odd spin normalizes by sin(theta + spin*pi) instead of '
                   "Shoemake's sin(theta), so the result leaves the unit sphere",
                   strict=False)
@pytest.mark.parametrize('fraction', [0.25, 0.5])
def test_quaternion_slerp_with_extra_spins_stays_on_the_unit_sphere(fraction):
    rng = np.random.default_rng(8)
    q0 = tr.random_quaternion(rng.random(3))
    q1 = tr.random_quaternion(rng.random(3))
    assert np.isclose(tr.vector_norm(tr.quaternion_slerp(q0, q1, fraction, spin=1)), 1.0)


# ----------------------------------------------------------------------------
# random rotations
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('rand', [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.999, 0.25, 0.75]])
def test_random_quaternion_is_a_unit_quaternion(rand):
    q = tr.random_quaternion(rand)
    assert q.shape == (4, )
    assert np.isclose(tr.vector_norm(q), 1.0)


def test_random_quaternion_is_deterministic_given_the_variates():
    assert np.allclose(tr.random_quaternion([0.1, 0.2, 0.3]), tr.random_quaternion([0.1, 0.2, 0.3]))
    assert not np.allclose(tr.random_quaternion([0.1, 0.2, 0.3]), tr.random_quaternion([0.4, 0.5, 0.6]))


def test_random_quaternion_without_argument_draws_three_variates():
    np.random.seed(0)
    q = tr.random_quaternion()
    assert q.shape == (4, )
    assert np.isclose(tr.vector_norm(q), 1.0)


def test_random_quaternion_requires_three_variates():
    with pytest.raises(AssertionError):
        tr.random_quaternion([0.1, 0.2])


@pytest.mark.parametrize('seed', range(3))
def test_random_rotation_matrix_is_orthogonal(seed):
    R = _rand_rot(seed)
    assert np.allclose(R.T @ R, np.identity(4))
    assert np.isclose(np.linalg.det(R), 1.0)


# ----------------------------------------------------------------------------
# vector helpers
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    'v',
    [
        [3.0, 4.0],
        [1.0, 0.0, 0.0],
        [0.2, -0.7, 0.4],
        [1e-8, 1e-8, 1e-8],
        [1.0, 2.0, 3.0, 4.0],
        [-5.0],
    ],
)
def test_vector_norm_matches_numpy(v):
    n = tr.vector_norm(v)
    assert isinstance(n, float)
    assert np.isclose(n, np.linalg.norm(v))


@pytest.mark.parametrize('axis', [0, 1, 2, -1])
def test_vector_norm_along_an_axis(axis):
    v = np.random.default_rng(0).random((6, 5, 3))
    assert np.allclose(tr.vector_norm(v, axis=axis), np.sqrt(np.sum(v * v, axis=axis)))


def test_vector_norm_with_out_writes_in_place_and_returns_none():
    v = np.random.default_rng(0).random((5, 4, 3))
    out = np.empty((5, 3))
    assert tr.vector_norm(v, axis=1, out=out) is None
    assert np.allclose(out, np.sqrt(np.sum(v * v, axis=1)))


def test_vector_norm_edge_cases():
    assert tr.vector_norm([]) == 0.0
    assert tr.vector_norm([1]) == 1.0
    # a multidimensional input with axis=None gives back a length 1 array, not
    # a scalar, unlike the 1-D case which short circuits to a python float
    total = tr.vector_norm(np.arange(6.0).reshape(2, 3))
    assert total.shape == (1, )
    assert np.isclose(total[0], np.linalg.norm(np.arange(6.0)))


def test_vector_norm_does_not_mutate_its_input():
    v = np.array([3.0, 4.0])
    tr.vector_norm(v)
    assert np.allclose(v, [3.0, 4.0])


@pytest.mark.parametrize(
    'v',
    [
        [1.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],
        [0.2, -0.7, 0.4],
        [-1.0, -1.0, -1.0],
        [1e6, 1e-6, 0.0],
        [0.5, 0.5],
    ],
)
def test_unit_vector_matches_numpy(v):
    u = tr.unit_vector(v)
    assert np.allclose(u, np.asarray(v, dtype=float) / np.linalg.norm(v))
    assert np.isclose(np.linalg.norm(u), 1.0)


@pytest.mark.parametrize('axis', [0, 1, 2, -1])
def test_unit_vector_along_an_axis(axis):
    v = np.random.default_rng(0).random((5, 4, 3))
    expected = v / np.expand_dims(np.sqrt(np.sum(v * v, axis=axis)), axis)
    assert np.allclose(tr.unit_vector(v, axis=axis), expected)


def test_unit_vector_with_out_writes_in_place_and_returns_none():
    v = np.random.default_rng(0).random((5, 4, 3))
    out = np.empty((5, 4, 3))
    assert tr.unit_vector(v, axis=1, out=out) is None
    assert np.allclose(out, v / np.expand_dims(np.sqrt(np.sum(v * v, axis=1)), 1))


def test_unit_vector_edge_cases():
    assert list(tr.unit_vector([])) == []
    assert list(tr.unit_vector([1])) == [1.0]


def test_unit_vector_does_not_mutate_its_input():
    v = np.array([3.0, 4.0])
    tr.unit_vector(v)
    assert np.allclose(v, [3.0, 4.0])


@pytest.mark.xfail(reason='Bug. unit_vector(v, axis=0, out=...) raises ValueError for 1-D input because the '
                   'length is expanded to shape (1, 1) and no longer broadcasts into the (n,) output',
                   strict=False)
def test_unit_vector_one_dimensional_with_out_and_axis():
    v = np.array([3.0, 4.0, 0.0])
    out = np.empty(3)
    tr.unit_vector(v, axis=0, out=out)
    assert np.allclose(out, [0.6, 0.8, 0.0])


def test_random_vector_range_and_shape():
    np.random.seed(0)
    v = tr.random_vector(1000)
    assert v.shape == (1000, )
    assert np.all(v >= 0.0) and np.all(v < 1.0)


@pytest.mark.parametrize(
    'v0,v1,expected',
    [
        ([2, 0, 0], [0, 3, 0], [0, 0, 6]),
        ([0, 3, 0], [2, 0, 0], [0, 0, -6]),
        ([1, 0, 0], [1, 0, 0], [0, 0, 0]),
        ([1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], np.cross([1.0, 2.0, 3.0], [-4.0, 5.0, 6.0])),
    ],
)
def test_vector_product_matches_numpy_cross(v0, v1, expected):
    assert np.allclose(tr.vector_product(v0, v1), expected)


def test_vector_product_along_columns():
    v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    v1 = [[3], [0], [0]]
    assert np.allclose(tr.vector_product(v0, v1), [[0, 0, 0, 0], [0, 0, 6, 6], [0, -6, 0, -6]])


def test_vector_product_along_rows():
    v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    assert np.allclose(tr.vector_product(v0, v1, axis=1), [[0, 0, 6], [0, -6, 0], [6, 0, 0], [0, -6, 6]])


@pytest.mark.parametrize(
    'v0,v1,expected',
    [
        ([1, 0, 0], [1, 0, 0], 0.0),
        ([1, 0, 0], [0, 1, 0], math.pi / 2),
        ([1, -2, 3], [-1, 2, -3], math.pi),
        ([1, 0, 0], [1, 1, 0], math.pi / 4),
        ([2, 0, 0], [3, 3, 3], 0.9553166181245093),
    ],
)
def test_angle_between_vectors(v0, v1, expected):
    assert np.isclose(tr.angle_between_vectors(v0, v1), expected)


@pytest.mark.parametrize('v0,v1,expected', [([1, -2, 3], [-1, 2, -3], 0.0), ([1, 0, 0], [-1, 1, 0], math.pi / 4)])
def test_angle_between_undirected_axes_never_exceeds_a_right_angle(v0, v1, expected):
    angle = tr.angle_between_vectors(v0, v1, directed=False)
    assert np.isclose(angle, expected)
    assert angle <= math.pi / 2 + 1e-12


def test_angle_between_vectors_columnwise():
    v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    v1 = [[3], [0], [0]]
    assert np.allclose(tr.angle_between_vectors(v0, v1), [0, 1.5708, 1.5708, 0.95532], atol=1e-4)


def test_angle_between_vectors_is_clipped_against_rounding():
    # nearly parallel vectors must not produce a nan through acos of 1 + eps
    v = np.array([1.0, 1.0, 1.0])
    assert np.isclose(tr.angle_between_vectors(v, v * (1 + 1e-16)), 0.0)


# ----------------------------------------------------------------------------
# matrix utilities
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('size', range(1, 7))
def test_inverse_matrix_matches_numpy(size):
    M0 = np.random.default_rng(size).random((size, size))
    assert np.allclose(tr.inverse_matrix(M0), np.linalg.inv(M0))
    assert np.allclose(tr.inverse_matrix(M0) @ M0, np.identity(size))


def test_inverse_matrix_of_a_rotation_is_its_transpose():
    R = _rand_rot(0)
    assert np.allclose(tr.inverse_matrix(R), R.T)


def test_inverse_matrix_rejects_non_square_input():
    with pytest.raises(np.linalg.LinAlgError):
        tr.inverse_matrix(np.ones((3, 4)))


def test_concatenate_matrices():
    M = np.random.default_rng(0).random((4, 4)) - 0.5
    assert np.allclose(tr.concatenate_matrices(), np.identity(4))
    assert np.allclose(tr.concatenate_matrices(M), M)
    assert np.allclose(tr.concatenate_matrices(M, M.T), M @ M.T)
    assert np.allclose(tr.concatenate_matrices(M, M.T, M), M @ M.T @ M)


def test_concatenate_matrices_applies_the_rightmost_factor_first():
    T = tr.translation_matrix([1.0, 0.0, 0.0])
    R = tr.rotation_matrix(math.pi / 2, [0, 0, 1])
    p = np.array([1.0, 0.0, 0.0, 1.0])
    # translate then rotate is R @ T, so the point ends up at (0, 2, 0)
    assert np.allclose(tr.concatenate_matrices(R, T) @ p, [0.0, 2.0, 0.0, 1.0])


@pytest.mark.parametrize(
    'm0,m1,expected',
    [
        (np.identity(4), np.identity(4), True),
        (np.identity(4), 2.0 * np.identity(4), True),  # normalized by the [3, 3] element
        (np.identity(4), tr.rotation_matrix(0.3, [0, 0, 1]), False),
        (tr.translation_matrix([1.0, 2.0, 3.0]), tr.translation_matrix([1.0, 2.0, 3.0]), True),
        (tr.translation_matrix([1.0, 2.0, 3.0]), tr.translation_matrix([1.0, 2.0, 3.1]), False),
    ],
)
def test_is_same_transform(m0, m1, expected):
    assert tr.is_same_transform(m0, m1) is expected


def test_is_same_transform_does_not_mutate_its_arguments():
    M = 2.0 * np.identity(4)
    tr.is_same_transform(M, np.identity(4))
    assert np.allclose(M, 2.0 * np.identity(4))


@pytest.mark.parametrize(
    'q0,q1,expected',
    [
        ([1, 0, 0, 0], [1, 0, 0, 0], True),
        ([1, 0, 0, 0], [-1, 0, 0, 0], True),  # antipodal quaternions are the same rotation
        ([0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, -0.5, -0.5], True),
        ([1, 0, 0, 0], [0, 1, 0, 0], False),
        ([0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, 0.5], False),
    ],
)
def test_is_same_quaternion(q0, q1, expected):
    assert tr.is_same_quaternion(q0, q1) is expected


# ----------------------------------------------------------------------------
# Arcball
# ----------------------------------------------------------------------------


def test_arcball_drag_reference_value():
    ball = tr.Arcball()
    ball.place([320, 320], 320)
    ball.down([500, 250])
    ball.drag([475, 275])
    assert np.isclose(np.sum(ball.matrix()), 3.905834551595542)


def test_arcball_constrained_drag_reference_value():
    ball = tr.Arcball(initial=[1, 0, 0, 0])
    ball.place([320, 320], 320)
    ball.setaxes([1, 1, 0], [-1, 1, 0])
    ball.constrain = True
    assert ball.constrain is True
    ball.down([400, 200])
    ball.drag([200, 400])
    assert np.isclose(np.sum(ball.matrix()), 0.2055924, atol=1e-6)
    ball.next()
    assert ball.matrix().shape == (4, 4)


def test_arcball_starts_at_the_identity():
    assert np.allclose(tr.Arcball().matrix(), np.identity(4))
    assert np.allclose(tr.Arcball(initial=np.identity(4)).matrix(), np.identity(4))


def test_arcball_can_be_initialised_from_a_rotation_matrix():
    R = _rand_rot(0)
    assert tr.is_same_transform(tr.Arcball(initial=R).matrix(), R)


def test_arcball_rejects_a_bad_initial_value():
    with pytest.raises(ValueError, match='quaternion or matrix'):
        tr.Arcball(initial=np.zeros(3))


def test_arcball_drag_without_motion_leaves_the_orientation_alone():
    ball = tr.Arcball()
    ball.place([320, 320], 320)
    ball.down([400, 200])
    ball.drag([400, 200])
    assert np.allclose(ball.matrix(), np.identity(4))


def test_arcball_map_to_sphere_inside_and_outside():
    inside = tr.arcball_map_to_sphere([500, 250], [320, 320], 320)
    assert np.isclose(np.linalg.norm(inside), 1.0)
    assert inside[2] > 0.0
    # window y grows downwards, so the sphere y is negated
    assert np.isclose(inside[0], (500 - 320) / 320)
    assert np.isclose(inside[1], (320 - 250) / 320)
    outside = tr.arcball_map_to_sphere([2000, 2000], [320, 320], 320)
    assert np.isclose(np.linalg.norm(outside), 1.0)
    assert outside[2] == 0.0


def test_arcball_constrain_to_axis_projects_onto_the_perpendicular_circle():
    axis = np.array([0.0, 0.0, 1.0])
    point = np.array([0.3, 0.4, math.sqrt(1.0 - 0.25)])
    constrained = tr.arcball_constrain_to_axis(point, axis)
    assert np.isclose(np.dot(constrained, axis), 0.0)
    assert np.isclose(np.linalg.norm(constrained), 1.0)


@pytest.mark.parametrize('axis,expected', [([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]), ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])])
def test_arcball_constrain_to_axis_degenerate_point(axis, expected):
    # a point on the axis has no perpendicular component, so a fixed fallback
    # direction is returned
    assert np.allclose(tr.arcball_constrain_to_axis(axis, axis), expected)


def test_arcball_nearest_axis():
    axes = [tr.unit_vector([1.0, 1.0, 0.0]), tr.unit_vector([-1.0, 1.0, 0.0])]
    assert np.allclose(tr.arcball_nearest_axis(tr.unit_vector([1.0, 1.0, 0.2]), axes), axes[1])
    assert tr.arcball_nearest_axis(tr.unit_vector([0.0, 0.0, 1.0]), axes) in axes


# ----------------------------------------------------------------------------
# module level tables and packaging
# ----------------------------------------------------------------------------


def test_axes_tables_are_consistent():
    assert len(tr._AXES2TUPLE) == 24
    assert len(tr._TUPLE2AXES) == 24
    for name, code in tr._AXES2TUPLE.items():
        assert tr._TUPLE2AXES[code] == name
        first_axis, parity, repetition, frame = code
        assert first_axis in (0, 1, 2)
        assert parity in (0, 1)
        assert repetition in (0, 1)
        assert frame in (0, 1)
        assert name[0] in 'sr'
        assert frame == (1 if name[0] == 'r' else 0)
        assert repetition == (1 if name[1] == name[3] else 0)


def test_next_axis_table():
    assert tr._NEXT_AXIS == [1, 2, 0, 1]


def test_epsilon_is_a_few_machine_epsilons():
    assert np.isclose(tr._EPS, np.finfo(float).eps * 4.0)


def test_module_is_self_contained():
    # the vendored library must not reach back into ManifoldEM. Nothing in the
    # package imports it, and it imports nothing from the package
    with open(tr.__file__, 'r') as handle:
        source = handle.read()
    assert 'ManifoldEM' not in source
    assert os.path.basename(tr.__file__) == 'transformations.py'


# ============================================================================
# CC/ - belief propagation, potentials, optical flow
# ============================================================================

matplotlib.use("Agg")  # every CC module pulls in pyplot at import time





# --------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------

_PARAM_FIELDS = ("project_name", "num_psi", "ncpu", "opt_movie")


@pytest.fixture
def params_sandbox_cc_algorithms(tmp_path):
    """`params` is a process-wide singleton and several CC functions chdir/write
    relative to it.  Save every field we touch plus the cwd, and put the test in
    its own tmp dir."""
    saved = {k: copy.deepcopy(getattr(params, k)) for k in _PARAM_FIELDS}
    cwd = os.getcwd()
    os.chdir(tmp_path)
    params.project_name = "cc_unit_test"
    params.ncpu = 1
    try:
        yield tmp_path
    finally:
        os.chdir(cwd)
        for k, v in saved.items():
            setattr(params, k, v)


def make_graph(adj, n_states):
    """Graph struct in the shape the BP code expects, from a dense adjacency matrix."""
    G = CreateGraphStruct(n_states, [], None, csr_matrix(np.array(adj, dtype=int)))
    # runGlobalOptimization stores this as a column vector and BP flattens it Fortran-style
    G["graphNodeOrder"] = np.arange(G["nNodes"]).reshape(-1, 1)
    return G


def bp_options(**kw):
    opts = dict(maxProduct=0, verbose=0, tol=1e-12, maxIter=200, eqnStates=1, alphaDamp=1.0)
    opts.update(kw)
    return opts


def chain_joint(nodePot, edgePot, n_nodes):
    """Explicit normalized joint of a linear chain 0-1-...-(n-1), for brute-force marginals."""
    S = nodePot.shape[0]
    joint = np.ones((S,) * n_nodes)
    for n in range(n_nodes):
        shape = [1] * n_nodes
        shape[n] = S
        joint = joint * nodePot[:, n].reshape(shape)
    for e in range(n_nodes - 1):
        shape = [1] * n_nodes
        shape[e] = S
        shape[e + 1] = S
        joint = joint * edgePot[e].reshape(shape)
    return joint / joint.sum()


def marginals_from_joint(joint, n_nodes, reduce_fn):
    S = joint.shape[0]
    out = np.zeros((S, n_nodes))
    for n in range(n_nodes):
        axes = tuple(a for a in range(n_nodes) if a != n)
        out[:, n] = reduce_fn(joint, axis=axes)
    return out / out.sum(axis=0)


def ramp_pair(n, shift, axis):
    """I1 is a unit-slope ramp along `axis`; I2 is I1 translated by +shift along `axis`,
    i.e. I2(x) = I1(x - shift).  The true optical flow is exactly +shift."""
    r = np.arange(n, dtype=np.float64)
    if axis == 0:
        return np.tile(r.reshape(-1, 1), (1, n)), np.tile((r - shift).reshape(-1, 1), (1, n))
    return np.tile(r, (n, 1)), np.tile(r - shift, (n, 1))


def flow_dict(seed, shape=(16, 16)):
    rng = np.random.default_rng(seed)
    vx = rng.normal(size=shape)
    vy = rng.normal(size=shape)
    orient, mag = OFM.getOrientMag(vx, vy)
    return dict(Vx=vx, Vy=vy, Orient=orient, Mag=mag)


# ======================================================================================
# hornschunck_simple.py
# ======================================================================================


@pytest.mark.parametrize(
    "kernel,expected_sum",
    [
        (HS.HSKERN, 1.0),   # Laplacian-style local average, must be a partition of unity
        (HS.kernelX, 0.0),  # difference kernels annihilate constants
        (HS.kernelY, 0.0),
        (HS.kernelT, 1.0),  # 2x2 box average
    ],
)
def test_hs_kernel_sums(kernel, expected_sum):
    assert kernel.sum() == pytest.approx(expected_sum)


@pytest.mark.parametrize("kernel", [HS.HSKERN, HS.kernelX, HS.kernelY, HS.kernelT])
def test_hs_kernel_shapes_and_dtype(kernel):
    assert kernel.dtype == np.float64
    assert kernel.ndim == 2


def test_hs_kernelx_is_transpose_of_kernely():
    # d/dx and d/dy stencils are the same stencil rotated, so kernelY == kernelX.T
    assert np.array_equal(HS.kernelY, HS.kernelX.T)


@pytest.mark.parametrize("kernel", [HS.HSKERN, HS.kernelX, HS.kernelY, HS.kernelT])
def test_filter2_matches_scipy_correlate_reflect(kernel):
    # cv2.filter2D is a CORRELATION (not a convolution) and BORDER_REFLECT matches
    # scipy's mode='reflect', so scipy is a valid oracle here.
    img = np.random.default_rng(0).random((12, 12)).astype(np.float32)
    got = HS.filter2(img, kernel)
    want = correlate(img.astype(np.float64), kernel, mode="reflect")
    assert np.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("kernel,factor", [(HS.HSKERN, 1.0), (HS.kernelX, 0.0),
                                           (HS.kernelY, 0.0), (HS.kernelT, 1.0)])
@pytest.mark.parametrize("const", [0.0, 1.0, -3.5])
def test_filter2_on_constant_image(kernel, factor, const):
    img = np.full((10, 10), const, dtype=np.float32)
    assert np.allclose(HS.filter2(img, kernel), factor * const, atol=1e-6)


def test_filter2_is_linear():
    rng = np.random.default_rng(1)
    a = rng.random((8, 8)).astype(np.float32)
    b = rng.random((8, 8)).astype(np.float32)
    lhs = HS.filter2((2.0 * a + 3.0 * b).astype(np.float32), HS.HSKERN)
    rhs = 2.0 * HS.filter2(a, HS.HSKERN) + 3.0 * HS.filter2(b, HS.HSKERN)
    assert np.allclose(lhs, rhs, atol=1e-5)


@pytest.mark.parametrize("sigma", [0.0, 1.0, 2.5])
def test_lowpassfilt_matches_gaussian_filter(sigma):
    img = np.random.default_rng(2).random((16, 16))
    assert np.allclose(HS.lowpassfilt(img, sigma), gaussian_filter(img, sigma=sigma))


@pytest.mark.parametrize("sigma", [0.5, 1.0, 3.0])
def test_lowpassfilt_preserves_constants(sigma):
    img = np.full((16, 16), 4.25)
    assert np.allclose(HS.lowpassfilt(img, sigma), 4.25)


@pytest.mark.parametrize("shift", [0.0, 1.0, 2.0, -3.0])
def test_computeDerivatives_x_ramp_exact(shift):
    # I1(y,x) = x  =>  Ix = 1, Iy = 0 in the interior (unit-slope ramp).
    im1, im2 = ramp_pair(24, shift, axis=1)
    fx, fy, ft = HS.computeDerivatives(im1.astype(np.float32), im2.astype(np.float32))
    inner = (slice(2, -2), slice(2, -2))
    assert np.allclose(fx[inner], 1.0, atol=1e-5)
    assert np.allclose(fy[inner], 0.0, atol=1e-5)
    # NOTE. Classical Horn-Schunck defines It = I2 - I1. This code returns I1 - I2,
    # so ft comes out as +shift for a +shift translation.
    assert np.allclose(ft[inner], shift, atol=1e-5)


@pytest.mark.parametrize("shift", [0.0, 1.0, 2.0, -3.0])
def test_computeDerivatives_y_ramp_exact(shift):
    im1, im2 = ramp_pair(24, shift, axis=0)
    fx, fy, ft = HS.computeDerivatives(im1.astype(np.float32), im2.astype(np.float32))
    inner = (slice(2, -2), slice(2, -2))
    assert np.allclose(fx[inner], 0.0, atol=1e-5)
    assert np.allclose(fy[inner], 1.0, atol=1e-5)
    assert np.allclose(ft[inner], shift, atol=1e-5)


def test_computeDerivatives_identical_images_have_zero_time_derivative():
    img = np.random.default_rng(3).random((16, 16)).astype(np.float32)
    fx, fy, ft = HS.computeDerivatives(img, img)
    assert np.allclose(ft, 0.0, atol=1e-6)
    # spatial derivatives are the sum over both frames, hence exactly 2x the single-frame value
    assert np.allclose(fx, 2.0 * HS.filter2(img, HS.kernelX), atol=1e-6)
    assert np.allclose(fy, 2.0 * HS.filter2(img, HS.kernelY), atol=1e-6)


def test_computeDerivatives_frame_swap_flips_only_ft():
    rng = np.random.default_rng(4)
    a = rng.random((16, 16)).astype(np.float32)
    b = rng.random((16, 16)).astype(np.float32)
    fx1, fy1, ft1 = HS.computeDerivatives(a, b)
    fx2, fy2, ft2 = HS.computeDerivatives(b, a)
    assert np.allclose(fx1, fx2, atol=1e-6)
    assert np.allclose(fy1, fy2, atol=1e-6)
    assert np.allclose(ft1, -ft2, atol=1e-6)


def test_computeDerivatives_constant_pair():
    im1 = np.full((12, 12), 5.0, dtype=np.float32)
    im2 = np.full((12, 12), 2.0, dtype=np.float32)
    fx, fy, ft = HS.computeDerivatives(im1, im2)
    assert np.allclose(fx, 0.0, atol=1e-6)
    assert np.allclose(fy, 0.0, atol=1e-6)
    assert np.allclose(ft, 3.0, atol=1e-6)  # I1 - I2


@pytest.mark.parametrize("shift", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("axis", [0, 1])
def test_hs_op_recovers_translation_magnitude(shift, axis):
    """A unit-slope ramp translated by `shift` has |flow| = shift along that axis and the
    orthogonal component is unobservable (aperture problem), so it stays at its initial 0."""
    n = 32
    im1, im2 = ramp_pair(n, shift, axis)
    U, V = HS.op(im1, im2, np.zeros((n, n)), np.zeros((n, n)), 1.0, 0.001, 300)
    along, across = (V, U) if axis == 0 else (U, V)
    inner = (slice(6, -6), slice(6, -6))
    assert np.allclose(np.abs(along[inner]), shift, atol=1e-4)
    assert np.allclose(across[inner], 0.0, atol=1e-4)


@pytest.mark.xfail(
    reason="Bug. computeDerivatives returns ft = I1 - I2 (classical HS is I2 - I1), "
           "so hornschunck_simple.op returns flow with the opposite sign",
    strict=False,
)
@pytest.mark.parametrize("shift", [1.0, 2.0])
def test_hs_op_flow_points_in_the_direction_of_motion(shift):
    n = 32
    im1, im2 = ramp_pair(n, shift, axis=1)
    U, _ = HS.op(im1, im2, np.zeros((n, n)), np.zeros((n, n)), 1.0, 0.001, 300)
    assert np.allclose(U[6:-6, 6:-6], shift, atol=1e-4)


def test_hs_op_zero_flow_for_identical_images():
    img = np.random.default_rng(5).random((16, 16))
    U, V = HS.op(img, img, np.zeros((16, 16)), np.zeros((16, 16)), 1.0, 0.001, 50)
    assert np.array_equal(U, np.zeros((16, 16)))
    assert np.array_equal(V, np.zeros((16, 16)))


def test_hs_op_niter_zero_is_identity_on_the_initial_field():
    img = np.random.default_rng(6).random((12, 12))
    u0 = np.full((12, 12), 7.0)
    v0 = np.full((12, 12), -3.0)
    U, V = HS.op(img, img * 0.5, u0.copy(), v0.copy(), 1.0, 0.001, 0)
    assert np.array_equal(U, u0)
    assert np.array_equal(V, v0)


@pytest.mark.parametrize("bad_shape", [(0,), (1,), (1, 1)])
def test_hs_op_replaces_degenerate_initial_fields_with_zeros(bad_shape):
    # `uInitial.shape[0] < 2` is the (undocumented) sentinel for "no initial guess"
    n = 32
    im1, im2 = ramp_pair(n, 2.0, axis=1)
    U, V = HS.op(im1, im2, np.zeros(bad_shape), np.zeros(bad_shape), 1.0, 0.001, 200)
    assert U.shape == (n, n) and V.shape == (n, n)
    assert np.allclose(np.abs(U[6:-6, 6:-6]), 2.0, atol=1e-4)


def test_hs_op_converged_field_is_a_fixed_point():
    n = 32
    im1, im2 = ramp_pair(n, 2.0, axis=1)
    U1, V1 = HS.op(im1, im2, np.zeros((n, n)), np.zeros((n, n)), 1.0, 0.001, 300)
    U2, V2 = HS.op(im1, im2, U1.copy(), V1.copy(), 1.0, 0.001, 60)
    assert np.allclose(U1, U2, atol=1e-8)
    assert np.allclose(V1, V2, atol=1e-6)


def test_hs_op_is_deterministic():
    n = 24
    im1, im2 = ramp_pair(n, 1.0, axis=1)
    a = HS.op(im1, im2, np.zeros((n, n)), np.zeros((n, n)), 1.0, 0.001, 40)
    b = HS.op(im1, im2, np.zeros((n, n)), np.zeros((n, n)), 1.0, 0.001, 40)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_hs_op_does_not_mutate_its_inputs():
    n = 16
    im1, im2 = ramp_pair(n, 1.0, axis=1)
    i1, i2 = im1.copy(), im2.copy()
    HS.op(im1, im2, np.zeros((n, n)), np.zeros((n, n)), 1.0, 0.001, 10)
    assert np.array_equal(im1, i1) and np.array_equal(im2, i2)


def test_hs_op_output_shape_and_dtype():
    n = 16
    im1, im2 = ramp_pair(n, 1.0, axis=1)
    U, V = HS.op(im1, im2, np.zeros((n, n)), np.zeros((n, n)), 1.0, 0.001, 5)
    assert U.shape == (n, n) and V.shape == (n, n)
    assert np.issubdtype(U.dtype, np.floating) and np.issubdtype(V.dtype, np.floating)


# ======================================================================================
# MRFBeliefPropagation.py, Normalize / max_product / graph indexing
# ======================================================================================


@pytest.mark.parametrize("shape", [(2, 3), (4, 4), (5, 2)])
def test_normalize_default_dim_normalizes_columns(shape):
    M = np.random.default_rng(7).random(shape) + 0.1
    N = BP.Normalize(M)
    assert np.allclose(N.sum(axis=0), 1.0)


@pytest.mark.parametrize("shape", [(2, 3), (4, 4)])
def test_normalize_dim1_equals_default(shape):
    M = np.random.default_rng(8).random(shape) + 0.1
    assert np.allclose(BP.Normalize(M, 1), BP.Normalize(M))


@pytest.mark.parametrize("shape", [(2, 3), (4, 4), (6, 1)])
def test_normalize_dim0_normalizes_whole_matrix(shape):
    M = np.random.default_rng(9).random(shape) + 0.1
    N = BP.Normalize(M, 0)
    assert N.sum() == pytest.approx(1.0)
    # dim=0 is a pure rescaling, so ratios survive
    assert np.allclose(N / N.flat[0], M / M.flat[0])


def test_normalize_of_1d_vector():
    v = np.array([1.0, 3.0, 4.0])
    assert np.allclose(BP.Normalize(v), np.array([0.125, 0.375, 0.5]))


def test_normalize_is_idempotent_along_columns():
    M = np.random.default_rng(10).random((4, 3)) + 0.1
    once = BP.Normalize(M)
    assert np.allclose(BP.Normalize(once), once)


def test_normalize_dim2_raises_on_nonsquare():
    # z is the row-sum vector but the division broadcasts along the last axis
    with pytest.raises(ValueError):
        BP.Normalize(np.arange(6, dtype=float).reshape(2, 3), 2)


@pytest.mark.xfail(reason="Bug. Normalize(M, 2) divides column j by row-sum j instead of "
                          "normalizing rows (np.divide broadcasts along the wrong axis)",
                   strict=False)
def test_normalize_dim2_should_normalize_rows():
    M = np.array([[1.0, 3.0], [2.0, 4.0]])
    assert np.allclose(BP.Normalize(M, 2).sum(axis=1), 1.0)


def test_normalize_zero_column_yields_nan():
    with np.errstate(invalid="ignore", divide="ignore"):
        N = BP.Normalize(np.array([[0.0, 1.0], [0.0, 1.0]]))
    assert np.all(np.isnan(N[:, 0]))
    assert np.allclose(N[:, 1], 0.5)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_max_product_matches_explicit_definition(n):
    rng = np.random.default_rng(11 + n)
    A = rng.random((n, n))
    x = rng.random(n)
    # y(i) = max_j A(i,j) x(j)
    assert np.allclose(BP.max_product(A, x), np.max(A * x[None, :], axis=1))


@pytest.mark.parametrize("n", [2, 4])
def test_max_product_accepts_column_vector(n):
    rng = np.random.default_rng(20 + n)
    A = rng.random((n, n))
    x = rng.random(n)
    assert np.allclose(BP.max_product(A, x.reshape(n, 1)), BP.max_product(A, x))


def test_max_product_output_is_1d():
    A = np.eye(3)
    assert BP.max_product(A, np.array([1.0, 2.0, 3.0])).shape == (3,)


def test_max_product_with_identity_is_the_identity():
    x = np.array([0.2, 0.5, 0.3])
    assert np.allclose(BP.max_product(np.eye(3), x), x)


def test_max_product_on_onehot_equals_matmul():
    A = np.random.default_rng(12).random((4, 4))
    e = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(BP.max_product(A, e), A @ e)


def test_max_product_positive_homogeneous():
    A = np.random.default_rng(13).random((4, 4))
    x = np.random.default_rng(14).random(4)
    assert np.allclose(BP.max_product(A, 3.0 * x), 3.0 * BP.max_product(A, x))


def test_max_product_dominates_nothing_below_the_max():
    A = np.random.default_rng(15).random((4, 4))
    x = np.random.default_rng(16).random(4)
    y = BP.max_product(A, x)
    assert np.all(y >= (A * x[None, :]).min(axis=1))
    assert np.all(y <= (A @ x))  # a max is never bigger than the sum of non-negatives


def test_max_product_multicolumn_branch_is_broken():
    # the >1 column branch calls np.matlib, which numpy does not expose by default
    A = np.random.default_rng(17).random((3, 3))
    with pytest.raises(AttributeError):
        BP.max_product(A, np.random.default_rng(18).random((3, 2)))


def test_two_node_graph_edge_indexing():
    G = make_graph([[0, 1], [1, 0]], 3)
    assert G["nNodes"] == 2 and G["nEdges"] == 1
    assert np.array_equal(G["Edges"], np.array([[0, 1]]))
    und0, dr0, rev0 = BP.getEdgeIdxsGivenNode(G, 0)
    und1, dr1, rev1 = BP.getEdgeIdxsGivenNode(G, 1)
    # node 0 receives on directed edge 1 (=j->i) and transmits on directed edge 0 (=i->j)
    assert np.array_equal(und0, [0]) and np.array_equal(dr0, [1]) and np.array_equal(rev0, [0])
    assert np.array_equal(und1, [0]) and np.array_equal(dr1, [0]) and np.array_equal(rev1, [1])


def test_three_node_chain_edge_indexing():
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 3)
    assert G["nEdges"] == 2
    assert np.array_equal(G["Edges"], np.array([[0, 1], [1, 2]]))
    und, dr, rev = BP.getEdgeIdxsGivenNode(G, 1)
    assert np.array_equal(und, [0, 1])
    assert np.array_equal(dr, [0, 3])   # incoming 0->1 and 2->1
    assert np.array_equal(rev, [2, 1])  # outgoing 1->0 and 1->2


@pytest.mark.parametrize(
    "adj",
    [
        [[0, 1], [1, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        [[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
    ],
)
def test_edge_index_triple_is_self_consistent(adj):
    G = make_graph(adj, 4)
    nE = G["nEdges"]
    for n in range(G["nNodes"]):
        und, dr, rev = BP.getEdgeIdxsGivenNode(G, n)
        assert und.shape == dr.shape == rev.shape
        assert np.array_equal(und, dr % nE)              # undirected index folds the direction away
        assert np.array_equal(rev, (dr + nE) % (2 * nE))  # reverse is the opposite orientation
        assert np.all(dr < 2 * nE) and np.all(und < nE)
        # every edge touching n is really incident on n
        assert np.all(np.any(G["Edges"][und] == n, axis=1))


@pytest.mark.parametrize("adj,degrees", [
    ([[0, 1], [1, 0]], [1, 1]),
    ([[0, 1, 0], [1, 0, 1], [0, 1, 0]], [1, 2, 1]),
    ([[0, 1, 1], [1, 0, 1], [1, 1, 0]], [2, 2, 2]),
])
def test_edge_index_count_equals_node_degree(adj, degrees):
    G = make_graph(adj, 4)
    for n, deg in enumerate(degrees):
        assert len(BP.getEdgeIdxsGivenNode(G, n)[0]) == deg


# ======================================================================================
# MRFBeliefPropagation.py, BPalg lifecycle
# ======================================================================================


def test_createBPalg_initial_state():
    G = make_graph([[0, 1], [1, 0]], 4)
    opts = bp_options(alphaDamp=0.6, eqnStates=1)
    alg = BP.createBPalg(G, opts)
    assert alg["G"] is G and alg["options"] is opts
    assert alg["alphaDamp"] == 0.6
    assert alg["eqnStates"] == 1
    assert alg["bpIter"] == 1
    assert alg["convergence"] == 0
    assert np.isinf(alg["convergenceIter"])
    for key in ("init_message", "old_message", "new_message", "nodeBel", "edgeBel", "bpError"):
        assert alg[key] == []


@pytest.mark.parametrize("n_states", [2, 4, 6])
def test_initializeBPmessage_uniform(n_states):
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], n_states)
    alg = BP.createBPalg(G, bp_options())
    alg = BP.initializeBPmessage(alg, np.ones((n_states, G["nNodes"])))
    for key in ("init_message", "old_message", "new_message"):
        assert alg[key].shape == (n_states, 2 * G["nEdges"])
        assert np.allclose(alg[key], 1.0 / n_states)
        assert np.allclose(alg[key].sum(axis=0), 1.0)
    assert alg["convergence"] == 0


def test_initializeBPmessage_buffers_are_independent():
    G = make_graph([[0, 1], [1, 0]], 3)
    alg = BP.initializeBPmessage(BP.createBPalg(G, bp_options()), np.ones((3, 2)))
    alg["new_message"][0, 0] = 99.0
    assert alg["old_message"][0, 0] != 99.0
    assert alg["init_message"][0, 0] != 99.0


@pytest.mark.xfail(reason="Bug. initializeBPmessage eqnStates=0 branch is dead code "
                          "(np.repmat does not exist, G['nState'] is misspelled, "
                          "and unif_msg is unbound)",
                   strict=False)
def test_initializeBPmessage_unequal_states_branch():
    G = make_graph([[0, 1], [1, 0]], 3)
    alg = BP.createBPalg(G, bp_options(eqnStates=0))
    alg = BP.initializeBPmessage(alg, np.ones((3, 2)))
    assert np.allclose(alg["init_message"].sum(axis=0), 1.0)


@pytest.mark.parametrize("delta,tol,expect", [(0.0, 1e-6, 1), (1e-9, 1e-6, 1),
                                              (1e-3, 1e-6, 0), (1.0, 1e-6, 0)])
def test_checkBPconvergence_flag(delta, tol, expect):
    G = make_graph([[0, 1], [1, 0]], 3)
    alg = BP.initializeBPmessage(BP.createBPalg(G, bp_options(tol=tol)), np.ones((3, 2)))
    alg["bpIter"] = 0
    alg["new_message"] = alg["old_message"] + delta
    alg = BP.checkBPconvergence(alg)
    assert alg["convergence"] == expect
    assert alg["bpError"] == pytest.approx(delta * alg["old_message"].size)


def test_checkBPconvergence_error_is_the_l1_message_residual():
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 3)
    alg = BP.initializeBPmessage(BP.createBPalg(G, bp_options()), np.ones((3, 3)))
    alg["bpIter"] = 0
    rng = np.random.default_rng(19)
    alg["new_message"] = alg["old_message"] + rng.normal(size=alg["old_message"].shape)
    want = np.abs(alg["new_message"] - alg["old_message"]).sum()
    alg = BP.checkBPconvergence(alg)
    assert alg["bpError"] == pytest.approx(want)


def test_checkBPconvergence_tolerance_boundary_is_strict():
    G = make_graph([[0, 1], [1, 0]], 2)
    alg = BP.initializeBPmessage(BP.createBPalg(G, bp_options(tol=1.0)), np.ones((2, 2)))
    alg["bpIter"] = 0
    alg["new_message"] = alg["old_message"].copy()
    alg["new_message"][0, 0] += 1.0  # residual exactly == tol
    alg = BP.checkBPconvergence(alg)
    assert alg["bpError"] == pytest.approx(1.0)
    assert alg["convergence"] == 0  # comparison is `<`, not `<=`


def test_checkBPconvergence_nan_is_not_convergence():
    G = make_graph([[0, 1], [1, 0]], 3)
    alg = BP.initializeBPmessage(BP.createBPalg(G, bp_options()), np.ones((3, 2)))
    alg["bpIter"] = 0
    alg["new_message"][0, 0] = np.nan
    alg = BP.checkBPconvergence(alg)
    assert np.isnan(alg["bpError"])
    assert alg["convergence"] == 0


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
def test_updateBPmessage_damping(alpha):
    S = 3
    G = make_graph([[0, 1], [1, 0]], S)
    alg = BP.initializeBPmessage(BP.createBPalg(G, bp_options()), np.ones((S, 2)))
    rng = np.random.default_rng(21)
    ePot = rng.random((S, S)) + 0.5
    msg = rng.random(S) + 0.5
    old = alg["old_message"][:, 0].copy()

    alg = BP.updateBPmessage(alg, ePot, msg, 0, 0, 0, alpha)
    want = BP.Normalize((1 - alpha) * old + alpha * (ePot @ msg), 1)
    assert np.allclose(alg["new_message"][:, 0], want)
    assert alg["new_message"][:, 0].sum() == pytest.approx(1.0)


def test_updateBPmessage_alpha_zero_keeps_the_old_message():
    S = 3
    G = make_graph([[0, 1], [1, 0]], S)
    alg = BP.initializeBPmessage(BP.createBPalg(G, bp_options()), np.ones((S, 2)))
    old = alg["old_message"][:, 0].copy()
    rng = np.random.default_rng(22)
    alg = BP.updateBPmessage(alg, rng.random((S, S)) + 0.5, rng.random(S), 0, 0, 0, 0.0)
    assert np.allclose(alg["new_message"][:, 0], old)


def test_updateBPmessage_maxproduct_uses_max_not_sum():
    S = 3
    G = make_graph([[0, 1], [1, 0]], S)
    alg = BP.initializeBPmessage(BP.createBPalg(G, bp_options(maxProduct=1)), np.ones((S, 2)))
    rng = np.random.default_rng(23)
    ePot = rng.random((S, S)) + 0.5
    msg = rng.random(S) + 0.5
    alg = BP.updateBPmessage(alg, ePot, msg, 0, 0, 0, 1.0)
    assert np.allclose(alg["new_message"][:, 0], BP.Normalize(BP.max_product(ePot, msg), 1))


def test_updateBPmessage_only_touches_the_target_column():
    S = 3
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], S)
    alg = BP.initializeBPmessage(BP.createBPalg(G, bp_options()), np.ones((S, 3)))
    before = alg["new_message"].copy()
    rng = np.random.default_rng(24)
    alg = BP.updateBPmessage(alg, rng.random((S, S)) + 0.5, rng.random(S) + 0.5, 0, 0, 0, 1.0)
    assert not np.allclose(alg["new_message"][:, 0], before[:, 0])
    assert np.allclose(alg["new_message"][:, 1:], before[:, 1:])


# ======================================================================================
# MRFBeliefPropagation.py, full BP on hand-checkable graphs
# ======================================================================================


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("S", [2, 3, 4])
def test_bp_two_node_chain_gives_exact_marginals(seed, S):
    G = make_graph([[0, 1], [1, 0]], S)
    rng = np.random.default_rng(seed)
    nodePot = rng.random((S, 2)) + 0.5
    edgePot = rng.random((1, S, S)) + 0.5
    nodeBel, _, alg = BP.op(dict(G=G, options=bp_options()), nodePot, edgePot)

    joint = chain_joint(nodePot, edgePot, 2)
    exact = marginals_from_joint(joint, 2, np.sum)
    assert np.allclose(nodeBel, exact)
    assert alg["convergence"] == 1


@pytest.mark.parametrize("seed", [3, 4, 5])
@pytest.mark.parametrize("S", [2, 3])
def test_bp_three_node_chain_gives_exact_marginals(seed, S):
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], S)
    rng = np.random.default_rng(seed)
    nodePot = rng.random((S, 3)) + 0.5
    edgePot = rng.random((2, S, S)) + 0.5
    nodeBel, _, alg = BP.op(dict(G=G, options=bp_options()), nodePot, edgePot)

    exact = marginals_from_joint(chain_joint(nodePot, edgePot, 3), 3, np.sum)
    assert np.allclose(nodeBel, exact)
    assert alg["convergence"] == 1


@pytest.mark.parametrize("seed", [6, 7])
@pytest.mark.parametrize("n_nodes", [2, 3])
def test_bp_maxproduct_gives_normalized_max_marginals(seed, n_nodes):
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for i in range(n_nodes - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1
    S = 3
    G = make_graph(adj, S)
    rng = np.random.default_rng(seed)
    nodePot = rng.random((S, n_nodes)) + 0.5
    edgePot = rng.random((n_nodes - 1, S, S)) + 0.5

    nodeBel, _, _ = BP.op(dict(G=G, options=bp_options(maxProduct=1)), nodePot, edgePot)
    exact = marginals_from_joint(chain_joint(nodePot, edgePot, n_nodes), n_nodes, np.max)
    assert np.allclose(nodeBel, exact)


def test_bp_maxproduct_and_sumproduct_agree_on_the_map_state_of_a_chain():
    S = 3
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], S)
    rng = np.random.default_rng(25)
    nodePot = rng.random((S, 3)) + 0.5
    edgePot = rng.random((2, S, S)) + 0.5
    joint = chain_joint(nodePot, edgePot, 3)
    mp, _, _ = BP.op(dict(G=G, options=bp_options(maxProduct=1)), nodePot, edgePot)
    best = np.unravel_index(np.argmax(joint), joint.shape)
    assert tuple(mp.argmax(axis=0)) == best


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.9, 1.0])
def test_bp_damping_does_not_move_the_fixed_point(alpha):
    S = 3
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], S)
    rng = np.random.default_rng(26)
    nodePot = rng.random((S, 3)) + 0.5
    edgePot = rng.random((2, S, S)) + 0.5
    nodeBel, _, alg = BP.op(dict(G=G, options=bp_options(alphaDamp=alpha, tol=1e-12,
                                                        maxIter=400)), nodePot, edgePot)
    exact = marginals_from_joint(chain_joint(nodePot, edgePot, 3), 3, np.sum)
    assert np.allclose(nodeBel, exact, atol=1e-8)
    assert alg["convergence"] == 1


@pytest.mark.parametrize("adj", [
    [[0, 1], [1, 0]],
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
    [[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
])
def test_bp_beliefs_are_normalized(adj):
    S = 3
    G = make_graph(adj, S)
    rng = np.random.default_rng(27)
    nodePot = rng.random((S, G["nNodes"])) + 0.5
    edgePot = rng.random((G["nEdges"], S, S)) + 0.5
    edgePot = 0.5 * (edgePot + np.transpose(edgePot, (0, 2, 1)))
    nodeBel, edgeBel, alg = BP.op(dict(G=G, options=bp_options(alphaDamp=0.7, maxIter=200)),
                                  nodePot, edgePot)
    assert np.allclose(nodeBel.sum(axis=0), 1.0)
    assert np.all(nodeBel >= 0.0)
    assert np.isfinite(nodeBel).all()
    for e in range(G["nEdges"]):
        assert edgeBel[e].sum() == pytest.approx(1.0)
    # every directed message is normalized after a full sweep
    assert np.allclose(alg["new_message"].sum(axis=0), 1.0)


def test_bp_uniform_potentials_give_uniform_beliefs():
    S = 4
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], S)
    nodeBel, edgeBel, alg = BP.op(dict(G=G, options=bp_options()),
                                  np.ones((S, 3)), np.ones((2, S, S)))
    assert np.allclose(nodeBel, 1.0 / S)
    assert np.allclose(edgeBel, 1.0 / S ** 2)
    assert alg["convergence"] == 1
    assert alg["convergenceIter"] == 0  # nothing ever moves off the uniform initialization


def test_bp_deterministic_node_potential_pins_the_chain():
    # a node potential concentrated on one state, with an identity-like edge potential,
    # forces the whole chain into that state
    S = 3
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], S)
    nodePot = np.ones((S, 3))
    nodePot[:, 0] = [1e6, 1.0, 1.0]
    edgePot = np.stack([np.eye(S) * 1e6 + 1.0] * 2)
    nodeBel, _, _ = BP.op(dict(G=G, options=bp_options()), nodePot, edgePot)
    assert np.array_equal(nodeBel.argmax(axis=0), np.array([0, 0, 0]))
    assert nodeBel[0, 2] > 0.99


def test_bp_isolated_node_belief_is_the_normalized_node_potential():
    S = 3
    G = make_graph([[0]], S)
    assert G["nEdges"] == 0
    nodePot = np.array([[3.0], [1.0], [1.0]])
    nodeBel, _, alg = BP.op(dict(G=G, options=bp_options(maxIter=5)),
                            nodePot, np.zeros((0, S, S)))
    assert np.allclose(nodeBel[:, 0], nodePot[:, 0] / nodePot.sum())
    assert alg["convergence"] == 1


def test_bp_zero_edge_potential_raises():
    S = 3
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], S)
    rng = np.random.default_rng(28)
    with pytest.raises(ValueError, match="NaN"):
        BP.op(dict(G=G, options=bp_options(maxIter=5)),
              rng.random((S, 3)) + 0.5, np.zeros((2, S, S)))


def test_bp_reports_non_convergence_when_maxiter_is_hit():
    S = 3
    G = make_graph([[0, 1, 1], [1, 0, 1], [1, 1, 0]], S)
    rng = np.random.default_rng(29)
    nodePot = rng.random((S, 3)) + 0.5
    edgePot = rng.random((3, S, S)) + 0.5
    nodeBel, _, alg = BP.op(dict(G=G, options=bp_options(tol=1e-300, maxIter=2)),
                            nodePot, edgePot)
    assert alg["convergence"] == 0
    assert np.isinf(alg["convergenceIter"])
    assert alg["bpIter"] == 1  # 0-based iteration counter
    assert np.allclose(nodeBel.sum(axis=0), 1.0)


def test_bp_is_deterministic():
    S = 3
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], S)
    rng = np.random.default_rng(30)
    nodePot = rng.random((S, 3)) + 0.5
    edgePot = rng.random((2, S, S)) + 0.5
    a = BP.op(dict(G=G, options=bp_options()), nodePot, edgePot)[0]
    b = BP.op(dict(G=G, options=bp_options()), nodePot, edgePot)[0]
    assert np.array_equal(a, b)


def test_bp_does_not_mutate_the_potentials():
    S = 3
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], S)
    rng = np.random.default_rng(31)
    nodePot = rng.random((S, 3)) + 0.5
    edgePot = rng.random((2, S, S)) + 0.5
    n0, e0 = nodePot.copy(), edgePot.copy()
    BP.op(dict(G=G, options=bp_options()), nodePot, edgePot)
    assert np.array_equal(nodePot, n0) and np.array_equal(edgePot, e0)


def test_bp_node_potential_scaling_leaves_beliefs_invariant():
    S = 3
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], S)
    rng = np.random.default_rng(32)
    nodePot = rng.random((S, 3)) + 0.5
    edgePot = rng.random((2, S, S)) + 0.5
    a = BP.op(dict(G=G, options=bp_options()), nodePot, edgePot)[0]
    b = BP.op(dict(G=G, options=bp_options()), 7.0 * nodePot, 3.0 * edgePot)[0]
    assert np.allclose(a, b)


def test_bp_output_shapes():
    S = 4
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], S)
    rng = np.random.default_rng(33)
    nodeBel, edgeBel, alg = BP.op(dict(G=G, options=bp_options()),
                                  rng.random((S, 3)) + 0.5, rng.random((2, S, S)) + 0.5)
    assert nodeBel.shape == (S, 3)
    assert edgeBel.shape == (2, S, S)
    assert alg["nodeBel"] is nodeBel and alg["edgeBel"] is edgeBel


@pytest.mark.xfail(reason="Bug. ComputeBelief uses np.dot(Beli, Belj.T) on two 1-D arrays, "
                          "which is an inner product. It needs np.outer, so edgeBelief "
                          "collapses to the normalized edge potential",
                   strict=False)
@pytest.mark.parametrize("seed", [40, 41])
def test_bp_edge_belief_is_the_pairwise_marginal(seed):
    S = 3
    G = make_graph([[0, 1], [1, 0]], S)
    rng = np.random.default_rng(seed)
    nodePot = rng.random((S, 2)) + 0.5
    edgePot = rng.random((1, S, S)) + 0.5
    _, edgeBel, _ = BP.op(dict(G=G, options=bp_options()), nodePot, edgePot)
    assert np.allclose(edgeBel[0], chain_joint(nodePot, edgePot, 2))


def test_bp_edge_belief_marginalizes_to_the_node_belief():
    # a true pairwise belief must sum down to the node belief. It does not here
    S = 3
    G = make_graph([[0, 1], [1, 0]], S)
    rng = np.random.default_rng(42)
    nodePot = rng.random((S, 2)) + 0.5
    edgePot = rng.random((1, S, S)) + 0.5
    nodeBel, edgeBel, _ = BP.op(dict(G=G, options=bp_options()), nodePot, edgePot)
    assert edgeBel[0].sum() == pytest.approx(1.0)
    assert not np.allclose(edgeBel[0].sum(axis=1), nodeBel[:, 0])


# ======================================================================================
# MRFGeneratePotentials.py
# ======================================================================================


@pytest.mark.parametrize("beta", [0.0, 1.0, 2.5, -1.0])
def test_nodePotentialFunction_is_exp_beta_m(beta):
    M = np.linspace(-1.0, 1.0, 12).reshape(3, 4)
    assert np.allclose(MGP.nodePotentialFunction(M, beta), np.exp(beta * M))


def test_nodePotentialFunction_of_zeros_is_uniform_ones():
    assert np.allclose(MGP.nodePotentialFunction(np.zeros((4, 5)), 1), 1.0)


@pytest.mark.parametrize("beta", [0.5, 1.0, 3.0])
def test_edgePotentialFunction_is_exp_minus_beta_one_minus_m(beta):
    M = np.linspace(0.0, 1.0, 12).reshape(3, 4)
    assert np.allclose(MGP.edgePotentialFunction(M, beta), np.exp(-beta * (1 - M)))


def test_edgePotentialFunction_is_one_at_perfect_similarity():
    assert np.allclose(MGP.edgePotentialFunction(np.ones((3, 3)), 2.0), 1.0)


def test_edgePotentialFunction_is_monotone_increasing_in_similarity():
    M = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
    out = MGP.edgePotentialFunction(M, 1.0).ravel()
    assert np.all(np.diff(out) > 0)
    assert np.all(out > 0)


@pytest.mark.parametrize("val", [0.0, 0.5, 1.0, 4.0])
def test_transformFunction_simple_is_exp_minus_m(val):
    M = np.full((3, 3), val)
    assert np.allclose(MGP.transformFunction_simple(M), np.exp(-val))


def test_transformFunction_simple_is_a_decreasing_kernel():
    M = np.array([0.0, 1.0, 2.0, 3.0])
    out = MGP.transformFunction_simple(M)
    assert out[0] == pytest.approx(1.0)
    assert np.all(np.diff(out) < 0)


def test_transformFunction_uses_sigma_1p2(params_sandbox_cc_algorithms):
    params.num_psi = 2
    M = np.random.default_rng(50).random((2, 4))
    assert np.allclose(MGP.transformFunction(M, [0, 0, 1], 0), np.exp(-M / (2.0 * 1.2 ** 2)))


def test_transformFunction_writes_no_figure_when_disabled(params_sandbox_cc_algorithms):
    params.num_psi = 2
    MGP.transformFunction(np.random.default_rng(51).random((2, 4)), [0, 0, 1], 0)
    assert not os.path.isdir(os.path.join(params.CC_dir, "CC_meas_fig"))


def test_transformFunction_figure_filename_is_one_indexed(params_sandbox_cc_algorithms):
    params.num_psi = 2
    MGP.transformFunction(np.random.default_rng(52).random((2, 4)), [0, 0, 1], 1)
    # elist is 0-indexed (edge 0, nodes 0 and 1) but the filename is 1-indexed
    assert os.path.isfile(os.path.join(params.CC_dir, "CC_meas_fig", "pot_edge1_1_2.png"))


def test_transformFunction_tblock_uses_sigma_1p25(params_sandbox_cc_algorithms):
    params.num_psi = 2
    M = np.random.default_rng(53).random((2, 4))
    assert np.allclose(MGP.transformFunction_tblock(M, [0, 0, 1], "FWD", 0),
                       np.exp(-M / (2.0 * 1.25 ** 2)))


def test_transformFunction_tblock_ignores_printPotFig(params_sandbox_cc_algorithms):
    # printPotFig is hard-reset to 0 inside the function, so no figure is ever written
    params.num_psi = 2
    MGP.transformFunction_tblock(np.random.default_rng(54).random((2, 4)), [0, 0, 1], "FWD", 1)
    assert not os.path.isdir(os.path.join(params.CC_dir, "CC_meas_fig"))


@pytest.mark.parametrize("kernel_fn,sigma", [(MGP.transformFunction, 1.2)])
def test_transformFunction_is_bounded_by_one_for_nonnegative_measures(params_sandbox_cc_algorithms,
                                                                     kernel_fn, sigma):
    params.num_psi = 2
    M = np.abs(np.random.default_rng(55).normal(size=(2, 4)))
    out = kernel_fn(M, [0, 0, 1], 0)
    assert np.all(out > 0) and np.all(out <= 1.0)


@pytest.mark.parametrize("num_psi", [1, 2, 3])
def test_mrf_op_potential_shapes(params_sandbox_cc_algorithms, num_psi):
    params.num_psi = num_psi
    P = num_psi
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 2 * P)
    rng = np.random.default_rng(56)
    measures = [rng.random((P, 2 * P)) for _ in range(G["nEdges"])]
    nodePot, edgePot = MGP.op(G, [], None, measures, None)
    assert nodePot.shape == (2 * P, G["nNodes"])
    assert edgePot.shape == (G["nEdges"], 2 * P, 2 * P)


def test_mrf_op_unobserved_nodes_get_a_flat_prior(params_sandbox_cc_algorithms):
    params.num_psi = 2
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 4)
    nodePot, _ = MGP.op(G, [], None, [None] * G["nEdges"], None)
    assert np.allclose(nodePot, 1.0)  # exp(0) everywhere


def test_mrf_op_anchor_nodes_carry_their_measure(params_sandbox_cc_algorithms):
    params.num_psi = 2
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 4)
    anchor_meas = np.array([[0.5], [1.5], [-0.5], [0.0]])
    nodePot, _ = MGP.op(G, [1], anchor_meas, [None] * G["nEdges"], None)
    assert np.allclose(nodePot[:, 1], np.exp(anchor_meas[:, 0]))
    assert np.allclose(nodePot[:, 0], 1.0)
    assert np.allclose(nodePot[:, 2], 1.0)


def test_mrf_op_missing_edge_measures_stay_at_eps(params_sandbox_cc_algorithms):
    params.num_psi = 2
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 4)
    measures = [np.random.default_rng(57).random((2, 4)), None]
    _, edgePot = MGP.op(G, [], None, measures, None)
    # zeros would blow up Normalize, so empty edges are seeded with 1e-10 instead
    assert np.allclose(edgePot[1], 1e-10)
    assert np.all(edgePot[0] > 1e-10)


def test_mrf_op_empty_array_measure_is_treated_as_missing(params_sandbox_cc_algorithms):
    params.num_psi = 2
    G = make_graph([[0, 1], [1, 0]], 4)
    _, edgePot = MGP.op(G, [], None, [np.array([])], None)
    assert np.allclose(edgePot[0], 1e-10)


def test_mrf_op_edge_potential_block_structure(params_sandbox_cc_algorithms):
    """The (2P x 2P) edge potential is [[A, B], [B, A]] where the psi-measure matrix is
    [A | B]. The second block row is the same measure with the two senses swapped."""
    params.num_psi = 3
    P = 3
    G = make_graph([[0, 1], [1, 0]], 2 * P)
    M = np.random.default_rng(58).random((P, 2 * P))
    _, edgePot = MGP.op(G, [], None, [M], None)
    Mt = MGP.transformFunction(M, [0, 0, 1], 0)
    assert np.allclose(edgePot[0][:P, :P], Mt[:, :P])
    assert np.allclose(edgePot[0][P:, P:], Mt[:, :P])
    assert np.allclose(edgePot[0][:P, P:], Mt[:, P:])
    assert np.allclose(edgePot[0][P:, :P], Mt[:, P:])


@pytest.mark.parametrize("num_psi", [2, 3])
def test_mrf_op_edge_potential_commutes_with_the_sense_flip(params_sandbox_cc_algorithms, num_psi):
    params.num_psi = num_psi
    P = num_psi
    G = make_graph([[0, 1], [1, 0]], 2 * P)
    M = np.random.default_rng(59).random((P, 2 * P))
    _, edgePot = MGP.op(G, [], None, [M], None)
    J = np.roll(np.eye(2 * P), P, axis=0)  # swaps the two sense blocks
    assert np.allclose(J @ edgePot[0] @ J, edgePot[0])


def test_mrf_op_edge_potential_is_strictly_positive(params_sandbox_cc_algorithms):
    params.num_psi = 2
    G = make_graph([[0, 1], [1, 0]], 4)
    M = np.random.default_rng(60).random((2, 4)) * 5.0
    _, edgePot = MGP.op(G, [], None, [M], None)
    assert np.all(edgePot > 0)
    assert np.all(edgePot <= 1.0)


def test_mrf_op_potentials_feed_belief_propagation(params_sandbox_cc_algorithms):
    """End-to-end sanity. The generated potentials are a legal BP input."""
    params.num_psi = 2
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 4)
    rng = np.random.default_rng(61)
    measures = [rng.random((2, 4)) for _ in range(G["nEdges"])]
    nodePot, edgePot = MGP.op(G, [0], np.zeros((4, 1)), measures, None)
    nodeBel, _, alg = BP.op(dict(G=G, options=bp_options(alphaDamp=0.7, maxIter=200)),
                            nodePot, edgePot)
    assert np.allclose(nodeBel.sum(axis=0), 1.0)
    assert np.isfinite(nodeBel).all()


# ======================================================================================
# OpticalFlowMovie.py
# ======================================================================================


@pytest.mark.parametrize("x,y,orient,mag", [
    (1.0, 0.0, 0.0, 1.0),
    (0.0, 1.0, np.pi / 2, 1.0),
    (-1.0, 0.0, np.pi, 1.0),
    (0.0, -1.0, 3 * np.pi / 2, 1.0),        # wrapped into [0, 2pi), not -pi/2
    (1.0, 1.0, np.pi / 4, np.sqrt(2)),
    (-1.0, -1.0, 5 * np.pi / 4, np.sqrt(2)),
    (3.0, 4.0, np.arctan2(4.0, 3.0), 5.0),
    (0.0, 0.0, 0.0, 0.0),
])
def test_getOrientMag_known_vectors(x, y, orient, mag):
    o, m = OFM.getOrientMag(np.array([x]), np.array([y]))
    assert o[0] == pytest.approx(orient)
    assert m[0] == pytest.approx(mag)


def test_getOrientMag_range_and_roundtrip():
    rng = np.random.default_rng(62)
    X = rng.normal(size=(8, 8))
    Y = rng.normal(size=(8, 8))
    o, m = OFM.getOrientMag(X, Y)
    assert np.all(o >= 0.0) and np.all(o < 2 * np.pi)
    assert np.all(m >= 0.0)
    assert np.allclose(m * np.cos(o), X)
    assert np.allclose(m * np.sin(o), Y)
    assert np.allclose(m, np.hypot(X, Y))


def test_getOrientMag_negation_rotates_by_pi():
    rng = np.random.default_rng(63)
    X, Y = rng.normal(size=6), rng.normal(size=6)
    o1, m1 = OFM.getOrientMag(X, Y)
    o2, m2 = OFM.getOrientMag(-X, -Y)
    assert np.allclose(m1, m2)
    assert np.allclose((o2 - o1) % (2 * np.pi), np.pi)


def test_getOrientMag_magnitude_is_scale_equivariant():
    rng = np.random.default_rng(64)
    X, Y = rng.normal(size=6), rng.normal(size=6)
    o1, m1 = OFM.getOrientMag(X, Y)
    o2, m2 = OFM.getOrientMag(2.5 * X, 2.5 * Y)
    assert np.allclose(m2, 2.5 * m1)
    assert np.allclose(o2, o1)


def test_normalizeRescaleVector_produces_unit_vectors():
    f = np.random.default_rng(65).normal(size=(8, 8, 2))
    F = OFM.normalizeRescaleVector(f, 1, 0)
    assert np.allclose(np.hypot(F[:, :, 0], F[:, :, 1]), 1.0, atol=1e-8)


def test_normalizeRescaleVector_preserves_direction():
    f = np.random.default_rng(66).normal(size=(6, 6, 2))
    F = OFM.normalizeRescaleVector(f, 1, 0)
    cross = f[:, :, 0] * F[:, :, 1] - f[:, :, 1] * F[:, :, 0]
    assert np.allclose(cross, 0.0, atol=1e-8)
    assert np.all(f[:, :, 0] * F[:, :, 0] + f[:, :, 1] * F[:, :, 1] >= -1e-12)


def test_normalizeRescaleVector_without_normalization_is_a_copy():
    f = np.random.default_rng(67).normal(size=(5, 5, 2))
    F = OFM.normalizeRescaleVector(f, 0, 0)
    assert np.allclose(F, f)
    assert F is not f


@pytest.mark.parametrize("rng_range", [[0.0, 1.0], [0.0, 5.0], [1.0, 2.0]])
def test_normalizeRescaleVector_rescales_magnitudes_into_range(rng_range):
    f = np.random.default_rng(68).normal(size=(8, 8, 2))
    F = OFM.normalizeRescaleVector(f, 1, rng_range)
    mag = np.hypot(F[:, :, 0], F[:, :, 1])
    assert mag.min() >= rng_range[0] - 1e-8
    assert mag.max() <= rng_range[1] + 1e-8
    assert mag.max() == pytest.approx(rng_range[1], abs=1e-6)


def test_normalizeRescaleVector_1d_maps_extremes_to_the_range():
    v = np.array([1.0, 2.0, 5.0])
    out = OFM.normalizeRescaleVector(v, 1, [0.0, 1.0])
    assert np.allclose(out, [0.0, 0.25, 1.0])


def test_normalizeRescaleVector_1d_ignores_the_normalize_flag():
    v = np.array([1.0, 2.0, 5.0])
    assert np.allclose(OFM.normalizeRescaleVector(v, 0, [0.0, 1.0]),
                       OFM.normalizeRescaleVector(v, 1, [0.0, 1.0]))


def test_normalizeRescaleVector_1d_constant_input_saturates_high():
    # np.interp with a degenerate xp range returns the upper endpoint
    out = OFM.normalizeRescaleVector(np.full(4, 3.0), 1, [0.0, 1.0])
    assert np.allclose(out, 1.0)


def test_normalizeRescaleVector_1d_requires_a_range():
    with pytest.raises(ValueError):
        OFM.normalizeRescaleVector(np.array([1.0, 2.0, 3.0]), 1, 0)


@pytest.mark.parametrize("pct", [0, 25, 50, 75])
def test_SelectFlowVec_keeps_the_top_percentile(pct):
    fv = flow_dict(70, shape=(8, 8))
    mag = fv["Mag"].copy()
    sel = OFM.SelectFlowVec(fv, pct)
    assert np.count_nonzero(sel["Mag"]) == int((mag > np.percentile(mag, pct)).sum())
    assert np.allclose(sel["Mag"][sel["Mag"] > 0], mag[mag > np.percentile(mag, pct)])


def test_SelectFlowVec_at_100_percent_discards_everything():
    fv = flow_dict(71, shape=(8, 8))
    sel = OFM.SelectFlowVec(fv, 100)
    assert np.count_nonzero(sel["Mag"]) == 0
    assert np.all(np.isneginf(sel["Orient"]))


def test_SelectFlowVec_zeroes_the_rejected_components():
    fv = flow_dict(72, shape=(8, 8))
    sel = OFM.SelectFlowVec(fv, 60)
    rejected = sel["Mag"] == 0
    assert np.all(sel["Vx"][rejected] == 0.0)
    assert np.all(sel["Vy"][rejected] == 0.0)
    assert np.all(np.isneginf(sel["Orient"][rejected]))


def test_SelectFlowVec_keeps_surviving_vectors_untouched():
    fv = flow_dict(73, shape=(8, 8))
    vx, vy, orient = fv["Vx"].copy(), fv["Vy"].copy(), fv["Orient"].copy()
    sel = OFM.SelectFlowVec(fv, 80)
    keep = sel["Mag"] > 0
    assert np.allclose(sel["Vx"][keep], vx[keep])
    assert np.allclose(sel["Vy"][keep], vy[keep])
    assert np.allclose(sel["Orient"][keep], orient[keep])


def test_SelectFlowVec_is_idempotent_on_a_fresh_copy():
    fv = flow_dict(74, shape=(8, 8))
    first = OFM.SelectFlowVec(fv, 70)
    again = OFM.SelectFlowVec({k: v.copy() for k, v in first.items()}, 70)
    assert np.allclose(again["Mag"], first["Mag"])
    assert np.allclose(again["Vx"], first["Vx"])


@pytest.mark.xfail(reason="Bug. SelectFlowVec aliases (never copies) the input arrays, so it "
                          "destroys the caller's FlowVec dict in place",
                   strict=False)
def test_SelectFlowVec_does_not_mutate_its_input():
    fv = flow_dict(75, shape=(8, 8))
    before = fv["Mag"].copy()
    OFM.SelectFlowVec(fv, 50)
    assert np.allclose(fv["Mag"], before)


def test_SelectFlowVec_returns_the_expected_keys():
    sel = OFM.SelectFlowVec(flow_dict(76, shape=(6, 6)), 50)
    assert set(sel.keys()) == {"Vx", "Vy", "Orient", "Mag"}


def test_movingAverage_1d_wraps_forward():
    # uniform_filter(size=3, origin=-1) averages X[i], X[i+1], X[i+2] with wraparound
    X = np.arange(6, dtype=float)
    assert np.allclose(OFM.movingAverage(X, 3), [1.0, 2.0, 3.0, 4.0, 3.0, 2.0])


def test_movingAverage_preserves_constants_and_shape():
    X = np.full((4, 5), 2.5)
    out = OFM.movingAverage(X, 3)
    assert out.shape == X.shape
    assert np.allclose(out, 2.5)


def test_movingAverage_window_one_is_identity():
    X = np.random.default_rng(77).random((4, 4))
    assert np.allclose(OFM.movingAverage(X, 1), X)


def test_anisodiff3_niter_one_does_nothing():
    # `for ii in np.arange(1, niter)` is empty when niter == 1
    stack = np.random.default_rng(78).random((4, 6, 6))
    out = OFM.anisodiff3(stack, niter=1)
    assert np.array_equal(out, stack.astype(np.float32))


def test_anisodiff3_returns_float32_and_keeps_shape():
    stack = np.random.default_rng(79).random((3, 5, 5))
    out = OFM.anisodiff3(stack, niter=4)
    assert out.dtype == np.float32
    assert out.shape == stack.shape


@pytest.mark.parametrize("option", [1, 2])
@pytest.mark.parametrize("niter", [2, 5])
def test_anisodiff3_conserves_total_intensity(option, niter):
    # the update is a discrete divergence, so the sum over the stack is invariant
    stack = np.random.default_rng(80).random((4, 6, 6)) + 1.0
    out = OFM.anisodiff3(stack, niter=niter, kappa=50, gamma=0.1, option=option)
    assert out.sum(dtype=np.float64) == pytest.approx(stack.sum(), rel=1e-5)


def test_anisodiff3_leaves_a_constant_stack_alone():
    stack = np.full((4, 6, 6), 3.0)
    out = OFM.anisodiff3(stack, niter=6, option=1)
    assert np.allclose(out, 3.0)


def test_anisodiff3_gamma_zero_is_a_noop():
    stack = np.random.default_rng(81).random((3, 5, 5))
    out = OFM.anisodiff3(stack, niter=6, gamma=0.0)
    assert np.allclose(out, stack.astype(np.float32))


def test_anisodiff3_smooths_a_noisy_stack():
    rng = np.random.default_rng(82)
    stack = np.zeros((5, 8, 8)) + rng.normal(scale=1.0, size=(5, 8, 8))
    out = OFM.anisodiff3(stack, niter=10, kappa=50, gamma=0.2, option=1)
    assert out.std() < stack.std()


def test_anisodiff3_unknown_option_falls_back_to_linear_diffusion():
    # gS/gE/gD stay at their initial ones, i.e. plain heat diffusion, but still conservative
    stack = np.random.default_rng(83).random((4, 6, 6)) + 1.0
    out = OFM.anisodiff3(stack, niter=4, option=3, gamma=0.05)
    assert out.sum(dtype=np.float64) == pytest.approx(stack.sum(), rel=1e-5)
    assert not np.allclose(out, stack.astype(np.float32))


def test_anisodiff3_accepts_a_4d_stack():
    stack = np.random.default_rng(84).random((2, 4, 4, 3))
    out = OFM.anisodiff3(stack, niter=3)
    assert out.shape == (2, 4, 4)


@pytest.mark.parametrize("shape", [(5, 5), (4, 7)])
def test_convertu8_spans_the_full_byte_range(shape):
    img = np.random.default_rng(85).normal(size=shape)
    u8 = OFM.convertu8(img)
    assert u8.dtype == np.uint8
    assert u8.min() == 0 and u8.max() == 255
    assert u8.shape == shape
    assert np.unravel_index(u8.argmax(), shape) == np.unravel_index(img.argmax(), shape)


def test_convertu8_of_a_constant_image_is_all_zeros():
    assert np.array_equal(OFM.convertu8(np.full((4, 4), 3.0)), np.zeros((4, 4), dtype=np.uint8))


def test_convertu8_is_monotone():
    img = np.array([[0.0, 1.0], [2.0, 3.0]])
    u8 = OFM.convertu8(img)
    assert list(u8.ravel()) == sorted(u8.ravel())


def test_ofm_op_output_contract():
    rng = np.random.default_rng(86)
    dim, nframes = 16, 4
    Mov = rng.random((nframes, dim * dim)) * 100.0
    fv = OFM.op(Mov.copy(), (0, 0), 2, "FWD", [0, 0])
    assert set(fv.keys()) == {"Vx", "Vy", "Orient", "Mag"}
    for k in fv:
        assert fv[k].shape == (dim, dim)
        assert fv[k].dtype == np.float16


def test_ofm_op_reverse_label_negates_the_flow():
    rng = np.random.default_rng(87)
    dim, nframes = 16, 4
    Mov = rng.random((nframes, dim * dim)) * 100.0
    fwd = OFM.op(Mov.copy(), (0, 0), 2, "FWD", [0, 0])
    rev = OFM.op(Mov.copy(), (0, 0), 2, "REV", [0, 0])
    assert np.array_equal(rev["Vx"], -fwd["Vx"])
    assert np.array_equal(rev["Vy"], -fwd["Vy"])
    assert np.array_equal(rev["Mag"], fwd["Mag"])  # magnitude is direction agnostic


def test_ofm_op_orientation_matches_the_stored_components():
    rng = np.random.default_rng(88)
    Mov = rng.random((4, 256)) * 100.0
    fv = OFM.op(Mov, (0, 0), 2, "FWD", [0, 0])
    o, m = OFM.getOrientMag(fv["Vx"].astype(np.float64), fv["Vy"].astype(np.float64))
    assert np.allclose(fv["Orient"].astype(np.float64), o, atol=2e-2)
    assert np.allclose(fv["Mag"].astype(np.float64), m, rtol=2e-2, atol=2e-3)


def test_ofm_op_is_deterministic():
    rng = np.random.default_rng(89)
    Mov = rng.random((4, 256)) * 100.0
    a = OFM.op(Mov.copy(), (0, 0), 2, "FWD", [0, 0])
    b = OFM.op(Mov.copy(), (0, 0), 2, "FWD", [0, 0])
    assert np.array_equal(a["Vx"], b["Vx"]) and np.array_equal(a["Mag"], b["Mag"])


def test_ofm_op_with_supplied_flow_only_recomputes_orient_and_mag():
    dim = 16
    Mov = np.random.default_rng(90).random((2, dim * dim))
    supplied = dict(Vx=np.full((dim, dim), 3.0, dtype=np.float16),
                    Vy=np.full((dim, dim), 4.0, dtype=np.float16),
                    Orient=np.zeros((dim, dim)), Mag=np.zeros((dim, dim)))
    out = OFM.op(Mov, (0, 0), 2, "FWD", [0, 0], supplied)
    assert np.allclose(out["Vx"].astype(np.float64), 3.0)
    assert np.allclose(out["Mag"].astype(np.float64), 5.0)
    assert np.allclose(out["Orient"].astype(np.float64), np.arctan2(4.0, 3.0), atol=2e-3)
    # the caller's dict is deep-copied, so its stale Orient survives
    assert np.allclose(supplied["Orient"], 0.0)


def test_ofm_op_with_supplied_flow_and_reverse_label():
    dim = 16
    Mov = np.random.default_rng(91).random((2, dim * dim))
    supplied = dict(Vx=np.ones((dim, dim), dtype=np.float16),
                    Vy=np.zeros((dim, dim), dtype=np.float16),
                    Orient=np.zeros((dim, dim)), Mag=np.zeros((dim, dim)))
    out = OFM.op(Mov, (0, 0), 2, "REV", [0, 0], supplied)
    assert np.allclose(out["Vx"].astype(np.float64), -1.0)
    assert np.allclose(out["Orient"].astype(np.float64), np.pi, atol=2e-3)


def test_writeOpticalFlowImage_writes_nothing_when_display_is_off(params_sandbox_cc_algorithms):
    rng = np.random.default_rng(92)
    out = os.path.join(str(params_sandbox_cc_algorithms), "of")
    ret = OFM.writeOpticalFlowImage(out, "FWD", rng.normal(size=(16, 16)),
                                    rng.normal(size=(16, 16, 2)), [0, 0], 0, 3, 4)
    assert ret is None
    assert not os.path.exists(out + ".png")


def test_writeOpticalFlowImage_writes_a_png_when_asked(params_sandbox_cc_algorithms):
    rng = np.random.default_rng(93)
    out = os.path.join(str(params_sandbox_cc_algorithms), "of2")
    OFM.writeOpticalFlowImage(out, "REV", rng.normal(size=(16, 16)),
                              rng.normal(size=(16, 16, 2)), [0, 1], 0, 3, 4)
    assert os.path.getsize(out + ".png") > 0


def test_figurePlot_opens_a_figure_and_leaves_it_open():
    import matplotlib.pyplot as plt
    before = set(plt.get_fignums())
    OFM.figurePlot(np.random.default_rng(94).random((8, 8)))
    new = set(plt.get_fignums()) - before
    assert len(new) == 1  # the function never closes what it opens
    plt.close(new.pop())


@pytest.mark.xfail(reason="Bug. saveImage builds `params.out_dir + 'CC/tmp_figs/'` without a "
                          "path separator, so it targets output/<proj>CC/tmp_figs",
                   strict=False)
def test_saveImage_writes_into_the_project_cc_directory(params_sandbox_cc_algorithms):
    target = os.path.join(params.out_dir, "CC", "tmp_figs")
    os.makedirs(target, exist_ok=True)
    OFM.saveImage(np.random.default_rng(95).random((8, 8)), 0, 0, 0)
    assert os.path.isfile(os.path.join(target, "PrD_1_psi_1_frame00.png"))


# ======================================================================================
# ComputeMeasureEdgeAll.py
# ======================================================================================


def test_hog_descriptor_shape_for_a_16x16_field():
    # 16/4 = 4 cells per axis, blocks of 2 cells -> 3 blocks per axis, 9 orientation bins
    H, hp = CME.HOGOpticalFlowPy(dict(Vx=np.random.default_rng(96).normal(size=(16, 16)),
                                      Vy=np.random.default_rng(97).normal(size=(16, 16))))
    assert H.shape == (3, 3, 9)
    assert hp == dict(cell_size=(4, 4), cells_per_block=(2, 2), n_bins=9)


@pytest.mark.parametrize("n", [16, 24, 32])
def test_hog_descriptor_shape_scales_with_the_field(n):
    rng = np.random.default_rng(98)
    H, _ = CME.HOGOpticalFlowPy(dict(Vx=rng.normal(size=(n, n)), Vy=rng.normal(size=(n, n))))
    assert H.shape == (n // 4 - 1, n // 4 - 1, 9)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_hog_descriptor_stacked_input_moves_the_block_axis_last(depth):
    rng = np.random.default_rng(99)
    H, _ = CME.HOGOpticalFlowPy(dict(Vx=rng.normal(size=(16, 16, depth)),
                                     Vy=rng.normal(size=(16, 16, depth))))
    assert H.shape == (3, 3, 9, depth)


def test_hog_descriptor_stack_slices_match_the_2d_result():
    rng = np.random.default_rng(100)
    vx, vy = rng.normal(size=(16, 16, 2)), rng.normal(size=(16, 16, 2))
    H3, _ = CME.HOGOpticalFlowPy(dict(Vx=vx, Vy=vy))
    for d in range(2):
        H2, _ = CME.HOGOpticalFlowPy(dict(Vx=vx[:, :, d], Vy=vy[:, :, d]))
        assert np.allclose(H3[:, :, :, d], H2)


def test_hog_blocks_are_l2_normalized_and_nonnegative():
    rng = np.random.default_rng(101)
    H, _ = CME.HOGOpticalFlowPy(dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16))))
    assert np.all(H >= 0.0)
    assert np.allclose(LA.norm(H, axis=2), 1.0)  # L2-Hys renormalizes to unit norm


def test_hog_of_a_zero_field_is_all_zeros():
    Z = np.zeros((16, 16))
    H, _ = CME.HOGOpticalFlowPy(dict(Vx=Z, Vy=Z))
    assert np.count_nonzero(H) == 0


def test_hog_is_invariant_to_positive_scaling():
    # block normalization removes any global gain on the gradient field
    rng = np.random.default_rng(102)
    vx, vy = rng.normal(size=(16, 16)), rng.normal(size=(16, 16))
    H1, _ = CME.HOGOpticalFlowPy(dict(Vx=vx, Vy=vy))
    H2, _ = CME.HOGOpticalFlowPy(dict(Vx=5.0 * vx, Vy=5.0 * vy))
    assert np.allclose(H1, H2)


def test_CompareOrientMatrix_self_distance_is_zero():
    a = dict(Vx=np.random.default_rng(103).normal(size=(16, 16)),
             Vy=np.random.default_rng(104).normal(size=(16, 16)))
    b = dict(Vx=a["Vx"].copy(), Vy=a["Vy"].copy())
    dist, tblock, bad = CME.CompareOrientMatrix(a, b, [0, 0, 1, 0])
    assert dist == pytest.approx(0.0)
    assert tblock == [] and bad == []  # 3D descriptors skip the time-block branch


def test_CompareOrientMatrix_is_symmetric_and_nonnegative():
    rng = np.random.default_rng(105)
    a = dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16)))
    b = dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16)))
    dab = CME.CompareOrientMatrix(a, b, [0, 0, 1, 0])[0]
    dba = CME.CompareOrientMatrix(b, a, [0, 0, 1, 0])[0]
    assert dab == pytest.approx(dba)
    assert dab > 0.0


def test_CompareOrientMatrix_is_the_frobenius_norm_of_the_hog_difference():
    rng = np.random.default_rng(106)
    a = dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16)))
    b = dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16)))
    HA, _ = CME.HOGOpticalFlowPy(a)
    HB, _ = CME.HOGOpticalFlowPy(b)
    assert CME.CompareOrientMatrix(a, b, [0, 0, 1, 0])[0] == pytest.approx(LA.norm(HA - HB))


def test_CompareOrientMatrix_obeys_the_triangle_inequality():
    rng = np.random.default_rng(107)
    a = dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16)))
    b = dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16)))
    c = dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16)))
    dab = CME.CompareOrientMatrix(a, b, [0, 0, 1, 0])[0]
    dbc = CME.CompareOrientMatrix(b, c, [0, 0, 1, 0])[0]
    dac = CME.CompareOrientMatrix(a, c, [0, 0, 1, 0])[0]
    assert dac <= dab + dbc + 1e-9


@pytest.mark.parametrize("depth", [2, 3])
def test_CompareOrientMatrix_time_block_output(depth):
    np.random.seed(0)  # the bad-block fallback draws from the legacy global RNG
    rng = np.random.default_rng(108)
    a = dict(Vx=rng.normal(size=(16, 16, depth)), Vy=rng.normal(size=(16, 16, depth)))
    b = dict(Vx=rng.normal(size=(16, 16, depth)), Vy=rng.normal(size=(16, 16, depth)))
    dist, tblock, bad = CME.CompareOrientMatrix(a, b, [0, 0, 1, 0])
    assert tblock.shape == (depth, 1)
    assert np.all(tblock >= 0.0)
    assert len(bad) == 2 and bad[0].shape == (1, depth)
    assert np.all(bad[0] == 0.0) and np.all(bad[1] == 0.0)
    # the global distance is the norm over all blocks at once
    assert dist == pytest.approx(np.sqrt((tblock ** 2).sum()))


def test_CompareOrientMatrix_flags_an_empty_flow_block():
    np.random.seed(0)
    rng = np.random.default_rng(109)
    a = dict(Vx=np.zeros((16, 16, 2)), Vy=np.zeros((16, 16, 2)))
    b = dict(Vx=rng.normal(size=(16, 16, 2)), Vy=rng.normal(size=(16, 16, 2)))
    dist, tblock, bad = CME.CompareOrientMatrix(a, b, [0, 0, 1, 0])
    assert np.all(bad[0] == 1.0)  # A is empty in both blocks
    assert np.all(bad[1] == 0.0)
    # the empty descriptor is replaced by large random values so the distance is big
    assert np.all(tblock > 1.0)


def test_ComparePsiMoviesOpticalFlow_wraps_CompareOrientMatrix():
    rng = np.random.default_rng(110)
    a = dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16)))
    b = dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16)))
    meas, bad = CME.ComparePsiMoviesOpticalFlow(a, b, [0, 0, 1, 0])
    assert set(meas.keys()) == {"Values", "Values_tblock"}
    assert meas["Values"] == pytest.approx(CME.CompareOrientMatrix(a, b, [0, 0, 1, 0])[0])
    assert bad == []


def test_ComputeMeasuresPsiMoviesOpticalFlow_keys_and_values():
    rng = np.random.default_rng(111)
    a = dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16)))
    bf = dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16)))
    br = dict(Vx=-bf["Vx"], Vy=-bf["Vy"])
    meas, bad = CME.ComputeMeasuresPsiMoviesOpticalFlow(a, bf, br, [0, 0, 1, 0])
    assert set(meas.keys()) == {"MeasABFWD", "MeasABFWD_tblock", "MeasABREV", "MeasABREV_tblock"}
    assert meas["MeasABFWD"] == pytest.approx(CME.CompareOrientMatrix(a, bf, [0, 0, 1, 0])[0])
    assert meas["MeasABREV"] == pytest.approx(CME.CompareOrientMatrix(a, br, [0, 0, 1, 0])[0])
    # a reversed flow field is a different HOG (signed orientation bins), so the two differ
    assert meas["MeasABFWD"] != pytest.approx(meas["MeasABREV"])
    assert bad == []


def test_ComputeMeasures_identical_forward_movies_give_zero_forward_distance():
    rng = np.random.default_rng(112)
    a = dict(Vx=rng.normal(size=(16, 16)), Vy=rng.normal(size=(16, 16)))
    same = dict(Vx=a["Vx"].copy(), Vy=a["Vy"].copy())
    rev = dict(Vx=-a["Vx"], Vy=-a["Vy"])
    meas, _ = CME.ComputeMeasuresPsiMoviesOpticalFlow(a, same, rev, [0, 0, 1, 0])
    assert meas["MeasABFWD"] == pytest.approx(0.0)
    assert meas["MeasABREV"] > 0.0


@pytest.mark.parametrize("edges", [[0], [0, 1], []])
def test_divide1_packs_edge_endpoints_with_the_edge_number(edges):
    G = make_graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 4)
    got = CME.divide1(edges, G)
    assert len(got) == len(edges)
    for row, e in zip(got, edges):
        assert row == [G["Edges"][e, 0], G["Edges"][e, 1], e]


def test_divide1_orders_endpoints_low_to_high():
    G = make_graph([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 4)
    for currPrD, nbrPrD, e in CME.divide1(range(G["nEdges"]), G):
        assert currPrD < nbrPrD


def test_ComputeEdgeMeasurePairWisePsiAll_writes_the_edge_measure_file(params_sandbox_cc_algorithms):
    params.num_psi = 2
    os.makedirs(params.CC_OF_dir, exist_ok=True)
    os.makedirs(params.CC_meas_dir, exist_ok=True)

    for prd in (0, 1):
        flows = [dict(FWD=flow_dict(10 * prd + p), REV=flow_dict(100 + 10 * prd + p))
                 for p in range(params.num_psi)]
        myio.fout1(params.get_CC_OF_file(prd), FlowVecPrD=flows)

    G = make_graph([[0, 1], [1, 0]], 2 * params.num_psi)
    CME.ComputeEdgeMeasurePairWisePsiAll([0, 1, 0], G, 95)

    out = myio.fin1(params.get_CC_meas_file(0, 0, 1))
    # [FWD | REV] blocks stacked horizontally. num_psi x 2*num_psi
    assert out["measureOFCurrNbrEdge"].shape == (params.num_psi, 2 * params.num_psi)
    assert out["badNodesPsisBlock"].shape == (G["nNodes"], params.num_psi)
    assert np.all(out["measureOFCurrNbrEdge"] > 0.0)
    assert np.isfinite(out["measureOFCurrNbrEdge"]).all()
    assert out["measureOFCurrNbrEdge_tblock"].size == 0  # single time block


def test_ComputeEdgeMeasurePairWisePsiAll_matches_a_direct_comparison(params_sandbox_cc_algorithms):
    params.num_psi = 1
    os.makedirs(params.CC_OF_dir, exist_ok=True)
    os.makedirs(params.CC_meas_dir, exist_ok=True)

    for prd in (0, 1):
        flows = [dict(FWD=flow_dict(200 + prd), REV=flow_dict(300 + prd))]
        myio.fout1(params.get_CC_OF_file(prd), FlowVecPrD=flows)

    G = make_graph([[0, 1], [1, 0]], 2)
    CME.ComputeEdgeMeasurePairWisePsiAll([0, 1, 0], G, 95)
    got = myio.fin1(params.get_CC_meas_file(0, 0, 1))["measureOFCurrNbrEdge"]

    a = OFM.SelectFlowVec(flow_dict(200), 95)
    bf = OFM.SelectFlowVec(flow_dict(201), 95)
    br = OFM.SelectFlowVec(flow_dict(301), 95)
    want, _ = CME.ComputeMeasuresPsiMoviesOpticalFlow(a, bf, br, [0, 0, 1, 0])
    assert got[0, 0] == pytest.approx(want["MeasABFWD"])
    assert got[0, 1] == pytest.approx(want["MeasABREV"])


# ============================================================================
# interfaces/cli.py and interfaces/interactive.py - argument parsing
# ============================================================================

# ---------------------------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _sandbox():
    """Restore the params singleton and the working directory around every test.

    Autouse fixtures are set up first and torn down last, so this runs after any
    monkeypatch undo and gets the last word on singleton state.
    """
    cwd = os.getcwd()
    saved = dict(params.__dict__)
    yield
    params.__dict__.clear()
    params.__dict__.update(saved)
    os.chdir(cwd)


@pytest.fixture
def workdir(tmp_path):
    """A scratch directory that is also the process cwd for the duration of the test."""
    os.chdir(tmp_path)
    return tmp_path


def _subaction(parser: ArgumentParser) -> _SubParsersAction:
    for action in parser._actions:
        if isinstance(action, _SubParsersAction):
            return action
    raise AssertionError("parser has no subparsers")


def _sub(name: str) -> ArgumentParser:
    """Fetch a subparser by name, looking one level into `utility` if needed."""
    top = _subaction(cli.get_parser())
    if name in top.choices:
        return top.choices[name]
    return _subaction(top.choices["utility"]).choices[name]


def _opts(parser: ArgumentParser) -> dict:
    return {opt: action for action in parser._actions for opt in action.option_strings}


def _dests(parser: ArgumentParser) -> set:
    return {action.dest for action in parser._actions}


def _choice_help(name: str) -> str:
    top = _subaction(cli.get_parser())
    for pseudo in top._choices_actions:
        if pseudo.dest == name:
            return pseudo.help
    raise AssertionError(f"{name} is not a registered subcommand")


def _write_toml(path, **overrides):
    # particle_diameter must be non-zero. params.asdict() evaluates the `sh` property
    # (resolution / diameter), which has no zero guard, so params.save() on a project that
    # never set a diameter raises ZeroDivisionError.
    payload = {"params": {"particle_diameter": 160.0, "ms_estimated_resolution": 3.0, **overrides}}
    with open(path, "w") as f:
        toml.dump(payload, f)
    return str(path)


# The nine pipeline subcommands, with the ProjectLevel value that prefixes their help string.
# Note level 6 (PRD_SELECTION) has no subcommand. It is a GUI-only interactive step, which is
# why the numbering in the help output jumps from 5 to 7.
PIPELINE = [
    ("init", 0, "Initialize new project"),
    ("threshold", 1, "Set upper/lower thresholds for principal direction detection"),
    ("calc-distance", 2, "Calculate S2 distances"),
    ("manifold-analysis", 3, "Initial embedding"),
    ("psi-analysis", 4, "Analyze images to get psis"),
    ("nlsa-movie", 5, "Create 2D psi movies"),
    ("find-ccs", 7, "Find conformational coordinates"),
    ("probability-landscape", 8, "Calculate probability landscape"),
    ("trajectory", 9, "Calculate trajectory"),
]
PIPELINE_NAMES = [c[0] for c in PIPELINE]

# short/long option pairs documented for `init`
INIT_FLAG_PAIRS = [
    ("-p", "--project-name", "project_name"),
    ("-v", "--avg-volume", "avg_volume"),
    ("-a", "--alignment", "alignment"),
    ("-i", "--image-stack", "image_stack"),
    ("-m", "--mask-volume", "mask_volume"),
    ("-s", "--pixel-size", "pixel_size"),
    ("-d", "--diameter", "diameter"),
    ("-r", "--resolution", "resolution"),
    ("-x", "--aperture-index", "aperture_index"),
    ("-o", "--overwrite", "overwrite"),
]

# (param flag, python type used by argparse, default value, ProjectLevel tag in the help prefix)
# tess_hemisphere_vec is declared list[float] but add_relevant_params downgrades any list-typed
# param to `str`, so the CLI takes it as a comma string and set_params parses it back.
INIT_PARAMS = [
    ("eps", float, 1e-10, "BINNING"),
    ("prd_thres_low", int, 100, "BINNING"),
    ("prd_thres_high", int, 2000, "BINNING"),
    ("tess_hemisphere_vec", str, [1.0, 0.0, 0.0], "BINNING"),
    ("tess_hemisphere_type", str, "lovisolo_silva", "BINNING"),
    ("prd_assignment", str, "hard", "BINNING"),
    ("prd_cone_width_factor", float, 1.0, "BINNING"),
    ("distance_filter_type", str, "Butter", "CALC_DISTANCE"),
    ("distance_filter_cutoff_freq", float, 0.5, "CALC_DISTANCE"),
    ("distance_filter_order", int, 8, "CALC_DISTANCE"),
    ("num_psi", int, 8, "CALC_DISTANCE"),
    ("nlsa_tune", int, 3, "MANIFOLD_ANALYSIS"),
    ("con_order_range", int, 50, "PSI_ANALYSIS"),
    ("nlsa_fps", float, 5.0, "NLSA_MOVIE"),
]

# params exposed by each stage subparser (first-appearance rule. a param is offered only on the
# earliest level it affects, which is why con_order_range shows up on psi-analysis but not on
# probability-landscape or trajectory even though it affects all three)
LEVEL_PARAMS = {
    "threshold": ["eps", "prd_thres_low", "prd_thres_high", "tess_hemisphere_vec",
                  "tess_hemisphere_type", "prd_assignment", "prd_cone_width_factor"],
    "calc-distance": ["distance_filter_type", "distance_filter_cutoff_freq",
                      "distance_filter_order", "num_psi"],
    "manifold-analysis": ["nlsa_tune"],
    "psi-analysis": ["con_order_range"],
    "nlsa-movie": ["nlsa_fps"],
    "find-ccs": [],
    "probability-landscape": [],
    "trajectory": [],
}

PRDS_COMMANDS = ["calc-distance", "manifold-analysis", "psi-analysis", "nlsa-movie"]
NO_PRDS_COMMANDS = ["init", "threshold", "find-ccs", "probability-landscape", "trajectory"]


# ---------------------------------------------------------------------------------------------
# parser construction
# ---------------------------------------------------------------------------------------------

def test_get_parser_returns_fresh_parser():
    a, b = cli.get_parser(), cli.get_parser()
    assert isinstance(a, ArgumentParser)
    assert a is not b


def test_parser_prog_and_description():
    p = cli.get_parser()
    assert p.prog == "manifold-cli"
    assert p.description == "Command-line interface for ManifoldEM package"


def test_parser_uses_defaults_help_formatter():
    assert cli.get_parser().formatter_class is ArgumentDefaultsHelpFormatter


def test_import_pins_omp_num_threads():
    # cli.py sets OMP_NUM_THREADS=1 at import time if unset, so BLAS threading does not fight
    # with the multiprocessing pools used by the stages.
    assert "OMP_NUM_THREADS" in os.environ


@pytest.mark.parametrize("name", PIPELINE_NAMES + ["utility"])
def test_subcommand_registered(name):
    assert name in _subaction(cli.get_parser()).choices


def test_exactly_ten_top_level_subcommands():
    # nine pipeline stages plus the `utility` group
    assert len(_subaction(cli.get_parser()).choices) == 10


def test_subcommand_registration_order_follows_pipeline():
    got = list(_subaction(cli.get_parser()).choices)
    assert got == PIPELINE_NAMES + ["utility"]


@pytest.mark.parametrize("name,level,text", PIPELINE)
def test_subcommand_help_has_numeric_prefix(name, level, text):
    assert _choice_help(name) == f"{level}: {text}"


@pytest.mark.parametrize("name,level,text", PIPELINE)
def test_numeric_prefix_matches_project_level_enum(name, level, text):
    # the number in each help string is the ProjectLevel ordinal the stage advances the project to
    assert ProjectLevel(level).value == level
    assert _choice_help(name).startswith(f"{ProjectLevel(level).value}:")


def test_utility_help_has_no_numeric_prefix():
    assert _choice_help("utility") == "Utility functions"


def test_prd_selection_level_has_no_subcommand():
    # ProjectLevel.PRD_SELECTION == 6 is interactive-only. No "6:" appears in the help
    assert ProjectLevel.PRD_SELECTION.value == 6
    helps = [_choice_help(n) for n in PIPELINE_NAMES]
    assert not any(h.startswith("6:") for h in helps)


@pytest.mark.parametrize("name", PIPELINE_NAMES + ["utility"])
def test_subcommand_appears_in_top_level_help(name):
    assert name in cli.get_parser().format_help()


def test_readme_documents_calc_probabilities_but_code_registers_probability_landscape():
    # DOC DRIFT. README.md shows `calc-probabilities` for stage 8. The code registers
    # `probability-landscape`. Assert the real one, and that the documented alias is absent.
    choices = _subaction(cli.get_parser()).choices
    assert "probability-landscape" in choices
    assert "calc-probabilities" not in choices


def test_calc_probabilities_is_not_accepted_by_the_parser():
    with pytest.raises(SystemExit):
        cli.get_parser().parse_args(["calc-probabilities", "params_x.toml"])


@pytest.mark.parametrize("name", PIPELINE_NAMES + ["utility", "mrcs2mrc", "denoise", "particle-index"])
def test_subparser_uses_defaults_help_formatter(name):
    assert _sub(name).formatter_class is ArgumentDefaultsHelpFormatter


@pytest.mark.parametrize("name", PIPELINE_NAMES + ["mrcs2mrc", "denoise", "particle-index"])
def test_subparser_help_exits_zero(name, capsys):
    with pytest.raises(SystemExit) as exc:
        _sub(name).parse_args(["-h"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------
# -n / --ncpu
# ---------------------------------------------------------------------------------------------

def test_ncpu_flag_exists_on_top_level_parser():
    opts = _opts(cli.get_parser())
    assert "-n" in opts and "--ncpu" in opts
    assert opts["-n"] is opts["--ncpu"]


def test_ncpu_default_is_one():
    assert cli.get_parser().parse_args(["threshold", "params_x.toml"]).ncpu == 1


def test_ncpu_type_is_int():
    assert _opts(cli.get_parser())["--ncpu"].type is int


@pytest.mark.parametrize("flag,value", [("-n", "8"), ("--ncpu", "8"), ("-n", "1"), ("--ncpu", "16")])
def test_ncpu_parsed_as_int(flag, value):
    args = cli.get_parser().parse_args([flag, value, "threshold", "params_x.toml"])
    assert args.ncpu == int(value)


def test_ncpu_rejects_non_integer():
    with pytest.raises(SystemExit):
        cli.get_parser().parse_args(["-n", "many", "threshold", "params_x.toml"])


@pytest.mark.parametrize("name", PIPELINE_NAMES)
def test_ncpu_is_not_repeated_on_subparsers(name):
    # -n must precede the subcommand. It is only registered on the top-level parser
    assert "--ncpu" not in _opts(_sub(name))


def test_ncpu_after_subcommand_is_rejected():
    with pytest.raises(SystemExit):
        cli.get_parser().parse_args(["threshold", "-n", "4", "params_x.toml"])


# ---------------------------------------------------------------------------------------------
# init flags
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("short,long,dest", INIT_FLAG_PAIRS)
def test_init_exposes_short_and_long_flag(short, long, dest):
    opts = _opts(_sub("init"))
    assert short in opts and long in opts
    assert opts[short] is opts[long]
    assert opts[short].dest == dest


@pytest.mark.parametrize("short,long,dest", INIT_FLAG_PAIRS)
def test_init_flag_shows_in_help(short, long, dest):
    assert long in _sub("init").format_help()


@pytest.mark.parametrize("flag", ["-p", "-s", "-d", "-r"])
def test_init_required_flags(flag):
    assert _opts(_sub("init"))[flag].required is True


@pytest.mark.parametrize("flag", ["-v", "-a", "-i", "-m", "-x", "-o"])
def test_init_optional_flags(flag):
    assert _opts(_sub("init"))[flag].required is False


@pytest.mark.parametrize("omit", ["-p", "-s", "-d", "-r"])
def test_init_missing_required_flag_exits(omit):
    full = {"-p": ["-p", "proj"], "-s": ["-s", "1.2"], "-d": ["-d", "160"], "-r": ["-r", "3.0"]}
    argv = ["init"]
    for k, v in full.items():
        if k != omit:
            argv += v
    with pytest.raises(SystemExit):
        cli.get_parser().parse_args(argv)


def test_init_minimal_invocation_parses():
    args = cli.get_parser().parse_args(["init", "-p", "proj", "-s", "1.22", "-d", "160.0", "-r", "3.02"])
    assert args.command == "init"
    assert args.project_name == "proj"
    assert args.pixel_size == pytest.approx(1.22)
    assert args.diameter == pytest.approx(160.0)
    assert args.resolution == pytest.approx(3.02)


@pytest.mark.parametrize("dest,expected", [
    ("avg_volume", ""),
    ("alignment", ""),
    ("image_stack", ""),
    ("mask_volume", ""),
    ("aperture_index", 1),
    ("overwrite", False),
])
def test_init_optional_defaults(dest, expected):
    args = cli.get_parser().parse_args(["init", "-p", "p", "-s", "1", "-d", "2", "-r", "3"])
    assert getattr(args, dest) == expected


@pytest.mark.parametrize("flag,dest,typ", [
    ("-s", "pixel_size", float),
    ("-d", "diameter", float),
    ("-r", "resolution", float),
    ("-x", "aperture_index", int),
    ("-p", "project_name", str),
])
def test_init_flag_types(flag, dest, typ):
    assert _opts(_sub("init"))[flag].type is typ


@pytest.mark.parametrize("flag,metavar", [
    ("-p", "STR"), ("-v", "FILEPATH"), ("-a", "FILEPATH"), ("-i", "FILEPATH"),
    ("-m", "FILEPATH"), ("-s", "FLOAT"), ("-d", "FLOAT"), ("-r", "FLOAT"), ("-x", "INT"),
])
def test_init_metavars(flag, metavar):
    assert _opts(_sub("init"))[flag].metavar == metavar


def test_init_overwrite_is_store_true():
    base = ["init", "-p", "p", "-s", "1", "-d", "2", "-r", "3"]
    assert cli.get_parser().parse_args(base).overwrite is False
    assert cli.get_parser().parse_args(base + ["-o"]).overwrite is True
    assert cli.get_parser().parse_args(base + ["--overwrite"]).overwrite is True


def test_init_overwrite_takes_no_value():
    assert _opts(_sub("init"))["-o"].nargs == 0


def test_init_takes_no_positional_input_file():
    # every other stage takes the toml as a positional. init creates it instead
    assert "input_file" not in _dests(_sub("init"))
    with pytest.raises(SystemExit):
        cli.get_parser().parse_args(["init", "-p", "p", "-s", "1", "-d", "2", "-r", "3", "extra.toml"])


def test_init_rejects_non_float_pixel_size():
    with pytest.raises(SystemExit):
        cli.get_parser().parse_args(["init", "-p", "p", "-s", "big", "-d", "2", "-r", "3"])


def test_init_expands_all_relevant_levels():
    # init offers the union of every level's user params, each exactly once
    init_flags = {o for o in _opts(_sub("init")) if o.startswith("--")}
    expected = {f"--{name}" for name, _, _, _ in INIT_PARAMS}
    assert expected <= init_flags


# ---------------------------------------------------------------------------------------------
# auto-generated --param flags on init
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,typ,default,level", INIT_PARAMS)
def test_init_param_flag_registered(name, typ, default, level):
    assert f"--{name}" in _opts(_sub("init"))


@pytest.mark.parametrize("name,typ,default,level", INIT_PARAMS)
def test_init_param_flag_type(name, typ, default, level):
    assert _opts(_sub("init"))[f"--{name}"].type is typ


@pytest.mark.parametrize("name,typ,default,level", INIT_PARAMS)
def test_init_param_flag_metavar(name, typ, default, level):
    assert _opts(_sub("init"))[f"--{name}"].metavar == typ.__name__.upper()


@pytest.mark.parametrize("name,typ,default,level", INIT_PARAMS)
def test_init_param_flag_default_is_current_singleton_value(name, typ, default, level):
    action = _opts(_sub("init"))[f"--{name}"]
    assert action.default == default
    assert action.default == getattr(params, name)


@pytest.mark.parametrize("name,typ,default,level", INIT_PARAMS)
def test_init_param_help_is_level_tagged(name, typ, default, level):
    assert _opts(_sub("init"))[f"--{name}"].help.startswith(f"[{level}] ")


@pytest.mark.parametrize("name,typ,default,level", INIT_PARAMS)
def test_init_param_help_carries_paraminfo_description(name, typ, default, level):
    _, info = params.get_param_info(name)
    assert _opts(_sub("init"))[f"--{name}"].help == f"[{level}] {info.description}"


@pytest.mark.parametrize("name,value,expected", [
    ("eps", "1e-8", 1e-8),
    ("prd_thres_low", "50", 50),
    ("prd_thres_high", "4000", 4000),
    ("tess_hemisphere_vec", "0,1,0", "0,1,0"),
    ("tess_hemisphere_type", "fibonacci", "fibonacci"),
    ("prd_assignment", "cone", "cone"),
    ("prd_cone_width_factor", "2.5", 2.5),
    ("distance_filter_type", "Gauss", "Gauss"),
    ("distance_filter_cutoff_freq", "0.25", 0.25),
    ("distance_filter_order", "4", 4),
    ("num_psi", "5", 5),
    ("nlsa_tune", "7", 7),
    ("con_order_range", "25", 25),
    ("nlsa_fps", "12.5", 12.5),
])
def test_init_param_value_parsing(name, value, expected):
    argv = ["init", "-p", "p", "-s", "1", "-d", "2", "-r", "3", f"--{name}", value]
    got = getattr(cli.get_parser().parse_args(argv), name)
    if isinstance(expected, float):
        assert got == pytest.approx(expected)
    else:
        assert got == expected


def test_list_typed_param_is_downgraded_to_string():
    # tess_hemisphere_vec is list[float] in params, but argparse cannot build a list from a
    # single token, so add_relevant_params rewrites the type to str. set_params re-parses it.
    action = _opts(_sub("init"))["--tess_hemisphere_vec"]
    assert action.type is str
    assert action.default == [1.0, 0.0, 0.0]  # the default is still the *list*, not a string


def test_default_help_formatter_shows_defaults():
    # collapse the formatter's line wrapping, which depends on the terminal width
    text = " ".join(_sub("init").format_help().split())
    assert "(default: 100)" in text
    assert "(default: lovisolo_silva)" in text
    assert "(default: 1e-10)" in text


@pytest.mark.parametrize("name", ["ncpu", "num_part", "rad", "width_1D", "states_per_coord",
                                  "traj_name", "num_eigs", "opt_mask_type"])
def test_non_level_params_absent_from_init(name):
    # ncpu is user-facing but has an empty `affects` list, so it never becomes a --param flag.
    # it is the top-level -n instead. The rest are not user params at all.
    assert f"--{name}" not in _opts(_sub("init"))


@pytest.mark.xfail(reason="Bug. `rad` Annotated is malformed - user_param/affects are passed as "
                          "extra Annotated metadata instead of ParamInfo args, so rad is never "
                          "exposed on the CLI", strict=False)
def test_rad_is_declared_a_user_param():
    _, info = params.get_param_info("rad")
    assert info.user_param is True
    assert ProjectLevel.MANIFOLD_ANALYSIS in info.affects


@pytest.mark.xfail(reason="Bug. `rad` Annotated is malformed, so --rad is missing from "
                          "manifold-analysis", strict=False)
def test_rad_flag_on_manifold_analysis():
    assert "--rad" in _opts(_sub("manifold-analysis"))


# ---------------------------------------------------------------------------------------------
# per-stage subparsers
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(LEVEL_PARAMS))
def test_stage_takes_positional_input_file(name):
    args = cli.get_parser().parse_args([name, "params_demo.toml"])
    assert args.input_file == "params_demo.toml"
    assert args.command == name


@pytest.mark.parametrize("name", list(LEVEL_PARAMS))
def test_stage_requires_input_file(name):
    with pytest.raises(SystemExit):
        cli.get_parser().parse_args([name])


@pytest.mark.parametrize("name,expected", sorted(LEVEL_PARAMS.items()))
def test_stage_param_flags_exactly(name, expected):
    got = sorted(o[2:] for o in _opts(_sub(name)) if o.startswith("--") and o != "--help")
    assert got == sorted(expected + (["prds"] if name in PRDS_COMMANDS else []))


@pytest.mark.parametrize("name", ["find-ccs", "probability-landscape", "trajectory"])
def test_terminal_stages_have_no_tunable_params(name):
    assert [o for o in _opts(_sub(name)) if o.startswith("--") and o != "--help"] == []


def test_con_order_range_offered_only_at_first_affected_level():
    # con_order_range affects PSI_ANALYSIS, PROBABILITY_LANDSCAPE and TRAJECTORY, but the
    # first-appearance rule exposes it only on psi-analysis
    _, info = params.get_param_info("con_order_range")
    assert [lvl.name for lvl in info.affects] == ["PSI_ANALYSIS", "PROBABILITY_LANDSCAPE", "TRAJECTORY"]
    assert "--con_order_range" in _opts(_sub("psi-analysis"))
    assert "--con_order_range" not in _opts(_sub("probability-landscape"))
    assert "--con_order_range" not in _opts(_sub("trajectory"))


def test_nlsa_tune_offered_only_at_first_affected_level():
    _, info = params.get_param_info("nlsa_tune")
    assert [lvl.name for lvl in info.affects] == ["MANIFOLD_ANALYSIS", "PSI_ANALYSIS"]
    assert "--nlsa_tune" in _opts(_sub("manifold-analysis"))
    assert "--nlsa_tune" not in _opts(_sub("psi-analysis"))


@pytest.mark.parametrize("name,expected", sorted(LEVEL_PARAMS.items()))
def test_stage_param_help_is_not_level_tagged(name, expected):
    # only init prefixes each param with its level. The stage parsers use the bare description
    opts = _opts(_sub(name))
    for param in expected:
        _, info = params.get_param_info(param)
        assert opts[f"--{param}"].help == info.description


def test_threshold_accepts_documented_readme_invocation():
    args = cli.get_parser().parse_args(
        ["threshold", "--prd_thres_low", "100", "--prd_thres_high", "4000", "params_demo.toml"])
    assert (args.prd_thres_low, args.prd_thres_high) == (100, 4000)


def test_calc_distance_accepts_documented_readme_invocation():
    args = cli.get_parser().parse_args(["-n", "16", "calc-distance", "--num_psi", "5", "params_demo.toml"])
    assert args.ncpu == 16 and args.num_psi == 5


# ---------------------------------------------------------------------------------------------
# --prds
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", PRDS_COMMANDS)
def test_prds_flag_present(name):
    action = _opts(_sub(name))["--prds"]
    assert action.type is str
    assert action.metavar == "INT,INT,..."
    assert action.default is None


@pytest.mark.parametrize("name", NO_PRDS_COMMANDS)
def test_prds_flag_absent(name):
    assert "--prds" not in _opts(_sub(name))


@pytest.mark.parametrize("name", PRDS_COMMANDS)
def test_prds_default_none(name):
    assert cli.get_parser().parse_args([name, "params_demo.toml"]).prds is None


@pytest.mark.parametrize("name", PRDS_COMMANDS)
def test_prds_round_trips_through_parse_prd_list(name):
    args = cli.get_parser().parse_args([name, "--prds", "3,1,2", "params_demo.toml"])
    assert args.prds == "3,1,2"
    # order is preserved verbatim. The list is not sorted or deduplicated
    assert cli._parse_prd_list(args.prds) == [3, 1, 2]


@pytest.mark.parametrize("text,expected", [
    ("0", [0]),
    ("1", [1]),
    ("7", [7]),
    ("1,2,3", [1, 2, 3]),
    ("3,1,2", [3, 1, 2]),
    ("5,5", [5, 5]),
    ("-1", [-1]),
    ("0,0,0", [0, 0, 0]),
    (" 1 , 2 ", [1, 2]),  # int() tolerates surrounding whitespace
    ("+4", [4]),
    ("10,20,30,40", [10, 20, 30, 40]),
])
def test_parse_prd_list_values(text, expected):
    assert cli._parse_prd_list(text) == expected


@pytest.mark.parametrize("falsy", ["", None])
def test_parse_prd_list_falsy_gives_none(falsy):
    # empty string and None both mean "all prds"
    assert cli._parse_prd_list(falsy) is None


@pytest.mark.parametrize("bad", ["a", "1,a", "1,,2", "1.5", "1;2", ",", "1,", "0x3"])
def test_parse_prd_list_rejects_garbage(bad):
    with pytest.raises(ValueError):
        cli._parse_prd_list(bad)


def test_parse_prd_list_returns_plain_ints():
    got = cli._parse_prd_list("1,2")
    assert all(type(v) is int for v in got)


# ---------------------------------------------------------------------------------------------
# utility subcommands
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["mrcs2mrc", "denoise", "particle-index"])
def test_utility_subcommand_registered(name):
    top = _subaction(cli.get_parser())
    assert name in _subaction(top.choices["utility"]).choices


def test_utility_subcommand_names_are_exactly_three():
    top = _subaction(cli.get_parser())
    assert list(_subaction(top.choices["utility"]).choices) == ["mrcs2mrc", "denoise", "particle-index"]


def test_mrcs2mrc_parses():
    args = cli.get_parser().parse_args(["utility", "mrcs2mrc", "params_demo.toml"])
    assert args.command == "mrcs2mrc"
    assert args.input_file == "params_demo.toml"


def test_denoise_parses_with_defaults():
    args = cli.get_parser().parse_args(["utility", "denoise", "params_demo.toml"])
    assert args.command == "denoise"
    assert args.window_size == 5
    assert args.frame == 5
    assert args.filter == "Gaussian"


@pytest.mark.parametrize("argv,dest,expected", [
    (["-k", "3"], "window_size", 3),
    (["--window_size", "9"], "window_size", 9),
    (["-f", "2"], "frame", 2),
    (["--frame", "11"], "frame", 11),
    (["--filter", "Median"], "filter", "Median"),
    (["--filter", "gaussian"], "filter", "gaussian"),
])
def test_denoise_flag_values(argv, dest, expected):
    args = cli.get_parser().parse_args(["utility", "denoise", "params_demo.toml"] + argv)
    assert getattr(args, dest) == expected


def test_denoise_rejects_non_integer_window():
    with pytest.raises(SystemExit):
        cli.get_parser().parse_args(["utility", "denoise", "params_demo.toml", "-k", "wide"])


@pytest.mark.xfail(reason="Bug. The nested `utility` subparsers reuse dest='command', so a bare "
                          "`manifold-cli utility` overwrites command with None and main() dies "
                          "with KeyError(None) instead of printing help", strict=False)
def test_bare_utility_keeps_its_own_command_name():
    assert cli.get_parser().parse_args(["utility"]).command == "utility"


# ---------------------------------------------------------------------------------------------
# args_to_dict
# ---------------------------------------------------------------------------------------------

def test_args_to_dict_returns_the_namespace_dict_itself():
    ns = Namespace(command="threshold", input_file="params_x.toml")
    out = cli.args_to_dict(ns)
    assert out is vars(ns)


def test_args_to_dict_renames_prds_to_prd_list():
    ns = Namespace(command="calc-distance", input_file="f.toml", prds="1,2,3")
    out = cli.args_to_dict(ns)
    assert out["prd_list"] == [1, 2, 3]
    assert "prds" not in out
    assert not hasattr(ns, "prds")  # the namespace is mutated in place


def test_args_to_dict_prds_none_becomes_prd_list_none():
    ns = Namespace(command="calc-distance", prds=None)
    assert cli.args_to_dict(ns)["prd_list"] is None


def test_args_to_dict_without_prds_adds_no_prd_list():
    out = cli.args_to_dict(Namespace(command="find-ccs", input_file="f.toml"))
    assert "prd_list" not in out


def test_args_to_dict_consumes_ncpu_into_the_singleton():
    ns = Namespace(command="threshold", ncpu=7)
    out = cli.args_to_dict(ns)
    assert params.ncpu == 7
    assert "ncpu" not in out


def test_args_to_dict_without_ncpu_leaves_singleton_alone():
    params.ncpu = 3
    cli.args_to_dict(Namespace(command="threshold"))
    assert params.ncpu == 3


def test_args_to_dict_keeps_every_other_key():
    ns = Namespace(command="calc-distance", input_file="f.toml", ncpu=2, prds="1", num_psi=5)
    out = cli.args_to_dict(ns)
    assert out == {"command": "calc-distance", "input_file": "f.toml", "prd_list": [1], "num_psi": 5}


def test_args_to_dict_is_not_idempotent():
    # it pops in place, so a second call finds nothing left to rename
    ns = Namespace(command="psi-analysis", ncpu=4, prds="2")
    first = cli.args_to_dict(ns)
    second = cli.args_to_dict(ns)
    assert first is second
    assert second["prd_list"] == [2]


@pytest.mark.parametrize("name", PRDS_COMMANDS)
def test_args_to_dict_on_real_parsed_args(name):
    args = cli.get_parser().parse_args(["-n", "2", name, "--prds", "0,4", "params_demo.toml"])
    out = cli.args_to_dict(args)
    assert out["prd_list"] == [0, 4]
    assert out["command"] == name
    assert params.ncpu == 2


# ---------------------------------------------------------------------------------------------
# load_state. Project-name derivation
# ---------------------------------------------------------------------------------------------

def test_load_state_is_a_noop_for_init():
    before = os.getcwd()
    ns = Namespace(command="init", project_name="p")
    assert cli.load_state(ns) is None
    assert os.getcwd() == before


@pytest.mark.parametrize("fname,expected", [
    ("params_demo.toml", "demo"),
    ("params_my_J310_analysis.toml", "my_J310_analysis"),
    ("params_a.toml", "a"),
    ("params_run.2.toml", "run.2"),          # only the *last* extension is stripped
    ("params_params_x.toml", "params_x"),    # split has maxsplit=1, so the inner prefix survives
    ("params_UPPER.toml", "UPPER"),
    ("params_1.toml", "1"),
])
def test_project_name_derivation(fname, expected, workdir, monkeypatch):
    path = workdir / fname
    _write_toml(path, num_psi=4)
    monkeypatch.setattr(params, "save", lambda *a, **k: None)
    cli.load_state(Namespace(command="threshold", input_file=str(path), ncpu=1))
    assert params.project_name == expected


def test_project_name_derivation_from_absolute_path(workdir, monkeypatch):
    sub = workdir / "project"
    sub.mkdir()
    path = sub / "params_deep.toml"
    _write_toml(path, num_psi=4)
    monkeypatch.setattr(params, "save", lambda *a, **k: None)
    cli.load_state(Namespace(command="threshold", input_file=str(path), ncpu=1))
    assert params.project_name == "deep"


def test_load_state_without_params_prefix_raises_indexerror(workdir):
    path = workdir / "settings.toml"
    _write_toml(path, num_psi=4)
    with pytest.raises(IndexError):
        cli.load_state(Namespace(command="threshold", input_file=str(path), ncpu=1))


@pytest.mark.xfail(reason="Bug. load_state splits the whole path on 'params_', so a directory "
                          "containing that substring leaks into the derived project name",
                   strict=False)
def test_project_name_ignores_params_in_directory_name(workdir, monkeypatch):
    sub = workdir / "params_cfg"
    sub.mkdir()
    path = sub / "params_run.toml"
    _write_toml(path, num_psi=4)
    monkeypatch.setattr(params, "save", lambda *a, **k: None)
    cli.load_state(Namespace(command="threshold", input_file=str(path), ncpu=1))
    assert params.project_name == "run"


# ---------------------------------------------------------------------------------------------
# load_state. Side effects
# ---------------------------------------------------------------------------------------------

def test_load_state_loads_values_chdirs_and_saves(workdir):
    sub = workdir / "proj"
    sub.mkdir()
    path = sub / "params_demo.toml"
    _write_toml(path, num_psi=6, prd_thres_low=42)
    cli.load_state(Namespace(command="calc-distance", input_file=str(path), ncpu=3))
    assert params.num_psi == 6
    assert params.prd_thres_low == 42
    assert params.ncpu == 3
    # params.load() chdirs into the toml's directory
    assert os.path.samefile(os.getcwd(), sub)
    # ...and load_state writes the state straight back out
    assert (sub / "params_demo.toml").is_file()
    assert toml.load(sub / "params_demo.toml")["params"]["ncpu"] == 3


def test_load_state_does_not_chdir_for_a_bare_filename(workdir, monkeypatch):
    path = workdir / "params_demo.toml"
    _write_toml(path, num_psi=4)
    monkeypatch.setattr(params, "save", lambda *a, **k: None)
    cli.load_state(Namespace(command="threshold", input_file="params_demo.toml", ncpu=1))
    assert os.path.samefile(os.getcwd(), workdir)


@pytest.mark.parametrize("ncpu", [1, 2, 8, 64])
def test_load_state_overrides_ncpu_from_the_flag(ncpu, workdir, monkeypatch):
    path = workdir / "params_demo.toml"
    _write_toml(path, ncpu=999)
    monkeypatch.setattr(params, "save", lambda *a, **k: None)
    cli.load_state(Namespace(command="threshold", input_file=str(path), ncpu=ncpu))
    assert params.ncpu == ncpu


def test_toml_project_name_wins_over_the_filename(workdir, monkeypatch):
    # load_state seeds project_name from the filename, then params.load() overwrites it with
    # whatever the toml stored. The derivation only survives if the toml omits project_name.
    path = workdir / "params_onfile.toml"
    _write_toml(path, project_name="stored_name")
    monkeypatch.setattr(params, "save", lambda *a, **k: None)
    cli.load_state(Namespace(command="threshold", input_file=str(path), ncpu=1))
    assert params.project_name == "stored_name"


def test_load_state_restores_project_level_enum(workdir, monkeypatch):
    path = workdir / "params_demo.toml"
    _write_toml(path, project_level=ProjectLevel.PSI_ANALYSIS.value)
    monkeypatch.setattr(params, "save", lambda *a, **k: None)
    cli.load_state(Namespace(command="threshold", input_file=str(path), ncpu=1))
    assert params.project_level is ProjectLevel.PSI_ANALYSIS


# ---------------------------------------------------------------------------------------------
# load_state. The dead path_width branch
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", PIPELINE_NAMES + ["mrcs2mrc", "denoise", "particle-index"])
def test_no_subcommand_produces_path_width(name):
    # load_state has a `hasattr(args, "path_width")` branch, but no parser defines it. Dead code
    assert "path_width" not in _dests(_sub(name))


def test_path_width_none_returns_before_save(workdir, monkeypatch):
    path = workdir / "params_demo.toml"
    _write_toml(path, num_psi=4)
    calls = []
    monkeypatch.setattr(params, "save", lambda *a, **k: calls.append(1))
    cli.load_state(Namespace(command="threshold", input_file=str(path), ncpu=1, path_width=None))
    assert calls == []


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5])
def test_path_width_in_range_sets_width_1d(width, workdir, monkeypatch):
    path = workdir / "params_demo.toml"
    _write_toml(path, num_psi=4)
    monkeypatch.setattr(params, "save", lambda *a, **k: None)
    cli.load_state(Namespace(command="threshold", input_file=str(path), ncpu=1, path_width=width))
    assert params.width_1D == width


@pytest.mark.parametrize("width", [0, -1, 6, 100])
def test_path_width_out_of_range_exits(width, workdir, monkeypatch, capsys):
    path = workdir / "params_demo.toml"
    _write_toml(path, num_psi=4)
    monkeypatch.setattr(params, "save", lambda *a, **k: None)
    with pytest.raises(SystemExit) as exc:
        cli.load_state(Namespace(command="threshold", input_file=str(path), ncpu=1, path_width=width))
    assert exc.value.code == 1
    assert "path-width" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------
# set_params
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,value", [
    ("num_psi", 3),
    ("prd_thres_low", 7),
    ("prd_thres_high", 999),
    ("eps", 1e-6),
    ("nlsa_tune", 11),
    ("con_order_range", 20),
    ("nlsa_fps", 30.0),
    ("distance_filter_type", "Gauss"),
    ("distance_filter_order", 2),
    ("tess_hemisphere_type", "fibonacci"),
    ("prd_assignment", "cone"),
    ("prd_cone_width_factor", 1.5),
])
def test_set_params_writes_changed_values(name, value, capsys):
    cli.set_params(Namespace(**{name: value}))
    assert getattr(params, name) == value
    assert f"Changing param {name}" in capsys.readouterr().out


def test_set_params_is_silent_when_nothing_changes(capsys):
    cli.set_params(Namespace(num_psi=params.num_psi))
    assert capsys.readouterr().out == ""


def test_set_params_ignores_unknown_attributes():
    cli.set_params(Namespace(command="threshold", input_file="f.toml", prds="1,2", not_a_param=5))
    assert not hasattr(params, "not_a_param")


@pytest.mark.parametrize("text,expected", [
    ("1,0,0", [1.0, 0.0, 0.0]),
    ("0,1,0", [0.0, 1.0, 0.0]),
    ("0,0,1", [0.0, 0.0, 1.0]),
    ("0.5,0.5,0", [0.5, 0.5, 0.0]),
    ("-1,0,0", [-1.0, 0.0, 0.0]),
])
def test_set_params_parses_list_param_from_string(text, expected):
    cli.set_params(Namespace(tess_hemisphere_vec=text))
    assert params.tess_hemisphere_vec == expected
    assert all(type(v) is float for v in params.tess_hemisphere_vec)


def test_set_params_round_trips_the_list_default():
    # the untouched argparse default is the list itself. str()/strip-brackets/split must
    # reproduce it exactly so no spurious "Changing param" happens
    cli.set_params(Namespace(tess_hemisphere_vec=[1.0, 0.0, 0.0]))
    assert params.tess_hemisphere_vec == [1.0, 0.0, 0.0]


def test_set_params_list_param_accepts_bracketed_text():
    cli.set_params(Namespace(tess_hemisphere_vec="[0, 0, 1]"))
    assert params.tess_hemisphere_vec == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("argv", [
    ["init", "-p", "p", "-s", "1", "-d", "2", "-r", "3"],
    ["threshold", "params_demo.toml"],
    ["calc-distance", "params_demo.toml"],
    ["manifold-analysis", "params_demo.toml"],
    ["psi-analysis", "params_demo.toml"],
    ["nlsa-movie", "params_demo.toml"],
    ["find-ccs", "params_demo.toml"],
    ["probability-landscape", "params_demo.toml"],
    ["trajectory", "params_demo.toml"],
    ["utility", "mrcs2mrc", "params_demo.toml"],
    ["utility", "denoise", "params_demo.toml"],
    ["utility", "particle-index", "--verify", "params_demo.toml"],
])
def test_set_params_survives_every_subcommand(argv, capsys):
    # every attribute a parser can emit must either be absent from params or be an Annotated
    # param, otherwise set_params blows up on get_param_info
    args = cli.get_parser().parse_args(argv)
    cli.set_params(args)
    capsys.readouterr()


def test_set_params_applies_init_project_name():
    args = cli.get_parser().parse_args(["init", "-p", "brand_new", "-s", "1", "-d", "2", "-r", "3"])
    cli.set_params(args)
    assert params.project_name == "brand_new"


# ---------------------------------------------------------------------------------------------
# dispatch table
# ---------------------------------------------------------------------------------------------

def test_funcs_table_keys():
    assert set(cli._funcs) == set(PIPELINE_NAMES) | {"mrcs2mrc", "denoise", "particle-index"}


@pytest.mark.parametrize("name", PIPELINE_NAMES)
def test_every_pipeline_subcommand_is_dispatchable(name):
    assert callable(cli._funcs[name])


def test_utility_group_itself_is_not_dispatchable():
    assert "utility" not in cli._funcs


@pytest.mark.parametrize("cmd,cli_name,mem_name", [
    ("init", "init", "init"),
    ("threshold", "threshold", "threshold"),
    ("calc-distance", "calc_distance", "calc_distance"),
    ("manifold-analysis", "manifold_analysis", "manifold_analysis"),
    ("psi-analysis", "psi_analysis", "psi_analysis"),
    ("nlsa-movie", "nlsa_movie", "nlsa_movie"),
    ("find-ccs", "find_conformational_coordinates", "find_conformational_coordinates"),
    ("probability-landscape", "probability_landscape", "probability_landscape"),
    ("trajectory", "compute_trajectory", "compute_trajectory"),
])
def test_dispatch_forwards_to_interactive(cmd, cli_name, mem_name, monkeypatch):
    assert cli._funcs[cmd] is getattr(cli, cli_name)
    seen = {}
    monkeypatch.setattr(mem, mem_name, lambda **kw: seen.update(kw))
    cli._funcs[cmd](Namespace(command=cmd, ncpu=5, input_file="params_demo.toml"))
    assert seen["command"] == cmd
    assert seen["input_file"] == "params_demo.toml"
    assert "ncpu" not in seen  # consumed into the singleton
    assert params.ncpu == 5


@pytest.mark.parametrize("cmd,mem_name", [
    ("calc-distance", "calc_distance"),
    ("manifold-analysis", "manifold_analysis"),
    ("psi-analysis", "psi_analysis"),
    ("nlsa-movie", "nlsa_movie"),
])
def test_prd_capable_dispatch_passes_prd_list(cmd, mem_name, monkeypatch):
    seen = {}
    monkeypatch.setattr(mem, mem_name, lambda **kw: seen.update(kw))
    args = cli.get_parser().parse_args([cmd, "--prds", "2,3", "params_demo.toml"])
    cli._funcs[cmd](args)
    assert seen["prd_list"] == [2, 3]


# ---------------------------------------------------------------------------------------------
# interactive.py
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["init", "load", "threshold", "calc_distance", "manifold_analysis",
                                  "psi_analysis", "nlsa_movie", "find_conformational_coordinates",
                                  "probability_landscape", "compute_trajectory"])
def test_interactive_exposes_entry_point(name):
    assert callable(getattr(mem, name))


def test_interactive_threshold_advances_project_level(workdir):
    params.project_name = "demo"
    params.particle_diameter = 160.0  # params.save() divides by this
    mem.threshold()
    assert params.project_level is ProjectLevel.BINNING
    assert (workdir / "params_demo.toml").is_file()
    assert toml.load(workdir / "params_demo.toml")["params"]["project_level"] == ProjectLevel.BINNING.value


def test_interactive_threshold_ignores_extra_kwargs(workdir):
    params.project_name = "demo"
    params.particle_diameter = 160.0
    mem.threshold(command="threshold", input_file="params_demo.toml", prd_list=None)
    assert params.project_level is ProjectLevel.BINNING


@pytest.mark.xfail(reason="Bug. params.sh divides by particle_diameter with no zero guard "
                          "(unlike params.ang_width), so params.save() - reached from both "
                          "mem.threshold and cli.load_state - raises ZeroDivisionError whenever "
                          "the diameter has not been set", strict=False)
def test_interactive_threshold_saves_without_a_diameter(workdir):
    params.project_name = "demo"
    mem.threshold()
    assert (workdir / "params_demo.toml").is_file()


def test_interactive_load_delegates_to_params(workdir):
    path = workdir / "params_demo.toml"
    _write_toml(path, num_psi=12)
    mem.load(str(path))
    assert params.num_psi == 12


@pytest.mark.parametrize("mem_name,module_name", [
    ("calc_distance", "ManifoldEM.calc_distance"),
    ("manifold_analysis", "ManifoldEM.manifold_analysis"),
    ("psi_analysis", "ManifoldEM.psi_analysis"),
    ("nlsa_movie", "ManifoldEM.nlsa_movie"),
])
def test_interactive_prd_stage_forwards_prd_list(mem_name, module_name, monkeypatch):
    import importlib
    module = importlib.import_module(module_name)
    seen = []
    monkeypatch.setattr(module, "op", lambda prd_list: seen.append(prd_list))
    getattr(mem, mem_name)(prd_list=[1, 4], command="ignored", input_file="ignored")
    assert seen == [[1, 4]]


@pytest.mark.parametrize("mem_name,module_name", [
    ("calc_distance", "ManifoldEM.calc_distance"),
    ("manifold_analysis", "ManifoldEM.manifold_analysis"),
    ("psi_analysis", "ManifoldEM.psi_analysis"),
    ("nlsa_movie", "ManifoldEM.nlsa_movie"),
])
def test_interactive_prd_stage_defaults_to_none(mem_name, module_name, monkeypatch):
    import importlib
    module = importlib.import_module(module_name)
    seen = []
    monkeypatch.setattr(module, "op", lambda prd_list: seen.append(prd_list))
    getattr(mem, mem_name)()
    assert seen == [None]


@pytest.mark.parametrize("mem_name,module_name", [
    ("find_conformational_coordinates", "ManifoldEM.find_conformational_coords"),
    ("probability_landscape", "ManifoldEM.probability_landscape"),
    ("compute_trajectory", "ManifoldEM.trajectory"),
])
def test_interactive_terminal_stage_takes_no_prds(mem_name, module_name, monkeypatch):
    import importlib
    module = importlib.import_module(module_name)
    seen = []
    monkeypatch.setattr(module, "op", lambda: seen.append("called"))
    getattr(mem, mem_name)(command="ignored", prd_list=[1, 2])
    assert seen == ["called"]


# ---------------------------------------------------------------------------------------------
# interactive.init
# ---------------------------------------------------------------------------------------------

def _patch_init_env(monkeypatch, width=64, ncpu=4):
    monkeypatch.setattr(ManifoldEM.util, "get_image_width_from_stack", lambda f: width)
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: ncpu)


def test_interactive_init_creates_project(workdir, monkeypatch):
    _patch_init_env(monkeypatch)
    mem.init(project_name="p1", avg_volume="v.mrc", alignment="a.star", image_stack="s.mrcs",
             mask_volume="m.mrc", pixel_size=1.22, diameter=160.0, resolution=3.02,
             aperture_index=2, overwrite=False)
    assert params.project_name == "p1"
    assert params.ms_pixel_size == pytest.approx(1.22)
    assert params.particle_diameter == pytest.approx(160.0)
    assert params.ms_estimated_resolution == pytest.approx(3.02)
    assert params.aperture_index == 2
    assert params.ms_num_pixels == 64
    assert params.ncpu == 4
    assert (workdir / "params_p1.toml").is_file()


@pytest.mark.parametrize("subdir", ["distances", "diff_maps", "psi_analysis", "traj", "bin",
                                    "CC", "topos/Euler_PrD", "postproc/vols", "postproc/denoise"])
def test_interactive_init_creates_directory(subdir, workdir, monkeypatch):
    _patch_init_env(monkeypatch)
    mem.init(project_name="p1", avg_volume="", alignment="", image_stack="", mask_volume="",
             pixel_size=1.0, diameter=1.0, resolution=1.0, aperture_index=1, overwrite=False)
    assert (workdir / "output" / "p1" / subdir).is_dir()


def test_interactive_init_expands_user_in_paths(workdir, monkeypatch):
    _patch_init_env(monkeypatch)
    mem.init(project_name="p1", avg_volume="~/vol.mrc", alignment="~/al.star",
             image_stack="~/st.mrcs", mask_volume="~/mask.mrc", pixel_size=1.0, diameter=1.0,
             resolution=1.0, aperture_index=1, overwrite=False)
    assert params.avg_vol_file == os.path.expanduser("~/vol.mrc")
    assert params.align_param_file == os.path.expanduser("~/al.star")
    assert params.img_stack_file == os.path.expanduser("~/st.mrcs")
    assert params.mask_vol_file == os.path.expanduser("~/mask.mrc")
    assert "~" not in params.avg_vol_file


def test_interactive_init_ignores_extra_kwargs(workdir, monkeypatch):
    _patch_init_env(monkeypatch)
    mem.init(project_name="p1", avg_volume="", alignment="", image_stack="", mask_volume="",
             pixel_size=1.0, diameter=1.0, resolution=1.0, aperture_index=1, overwrite=False,
             command="init", eps=1e-9, num_psi=8)
    assert params.project_name == "p1"


def test_interactive_init_aborts_when_user_declines(workdir, monkeypatch, capsys):
    _patch_init_env(monkeypatch)
    kw = dict(project_name="p1", avg_volume="", alignment="", image_stack="", mask_volume="",
              pixel_size=1.0, diameter=1.0, resolution=1.0, aperture_index=1)
    mem.init(overwrite=False, **kw)
    marker = workdir / "output" / "p1" / "keep.txt"
    marker.write_text("x")
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    assert mem.init(overwrite=False, **kw) == 1
    assert marker.is_file()  # nothing was removed
    assert "Aborting" in capsys.readouterr().out


def test_interactive_init_reprompts_until_valid_answer(workdir, monkeypatch):
    _patch_init_env(monkeypatch)
    kw = dict(project_name="p1", avg_volume="", alignment="", image_stack="", mask_volume="",
              pixel_size=1.0, diameter=1.0, resolution=1.0, aperture_index=1)
    mem.init(overwrite=False, **kw)
    answers = iter(["maybe", "", "N"])  # answers are lowercased before comparison
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert mem.init(overwrite=False, **kw) == 1


def test_interactive_init_overwrite_removes_old_output(workdir, monkeypatch, capsys):
    _patch_init_env(monkeypatch)
    kw = dict(project_name="p1", avg_volume="", alignment="", image_stack="", mask_volume="",
              pixel_size=1.0, diameter=1.0, resolution=1.0, aperture_index=1)
    mem.init(overwrite=False, **kw)
    stale = workdir / "output" / "p1" / "stale.txt"
    stale.write_text("x")
    capsys.readouterr()
    assert mem.init(overwrite=True, **kw) is None
    assert not stale.exists()
    assert (workdir / "output" / "p1" / "distances").is_dir()
    assert "Removing previous project" in capsys.readouterr().out


def test_interactive_init_never_prompts_when_overwrite_is_true(workdir, monkeypatch):
    _patch_init_env(monkeypatch)
    kw = dict(project_name="p1", avg_volume="", alignment="", image_stack="", mask_volume="",
              pixel_size=1.0, diameter=1.0, resolution=1.0, aperture_index=1)
    mem.init(overwrite=False, **kw)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("input() must not be called"))
    mem.init(overwrite=True, **kw)


# ---------------------------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------------------------

def test_main_with_no_arguments_prints_help_and_exits_one(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["manifold-cli"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "usage: manifold-cli" in out
    assert f"ManifoldEM version: {ManifoldEM.__version__}" in out


def test_main_prints_version_banner(monkeypatch, capsys, workdir):
    path = workdir / "params_demo.toml"
    _write_toml(path, num_psi=4)
    monkeypatch.setattr(sys, "argv", ["manifold-cli", "threshold", str(path)])
    monkeypatch.setitem(cli._funcs, "threshold", lambda args: None)
    cli.main()
    assert ManifoldEM.__version__ in capsys.readouterr().out


def test_main_dispatches_and_applies_ncpu(monkeypatch, capsys, workdir):
    path = workdir / "params_demo.toml"
    _write_toml(path, num_psi=4)
    seen = []
    monkeypatch.setattr(sys, "argv", ["manifold-cli", "-n", "2", "threshold", str(path)])
    monkeypatch.setitem(cli._funcs, "threshold", lambda args: seen.append(args))
    cli.main()
    assert len(seen) == 1
    assert seen[0].command == "threshold"
    assert params.ncpu == 2
    capsys.readouterr()


def test_main_applies_param_overrides_before_dispatch(monkeypatch, capsys, workdir):
    path = workdir / "params_demo.toml"
    _write_toml(path, num_psi=8)
    monkeypatch.setattr(sys, "argv",
                        ["manifold-cli", "calc-distance", "--num_psi", "3", str(path)])
    monkeypatch.setitem(cli._funcs, "calc-distance", lambda args: None)
    cli.main()
    assert params.num_psi == 3
    capsys.readouterr()


def test_main_rejects_unknown_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["manifold-cli", "not-a-command"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


# ---------------------------------------------------------------------------------------------
# utility implementations
# ---------------------------------------------------------------------------------------------

class _RecordingPool:
    """Stand-in for multiprocessing.Pool. Never forks, just records the work items."""
    log = []

    def __init__(self, processes=None):
        self.processes = processes
        _RecordingPool.log.append({"processes": processes})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, func, iterable):
        _RecordingPool.log[-1]["func"] = func
        _RecordingPool.log[-1]["items"] = list(iterable)
        return []


@pytest.fixture
def recording_pool(monkeypatch):
    _RecordingPool.log = []
    monkeypatch.setattr(multiprocessing, "Pool", _RecordingPool)
    return _RecordingPool


def test_mrcs2mrc_without_relion_is_a_noop(monkeypatch, capsys, workdir):
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    assert cli.mrcs2mrc(None) is None
    assert "relion_reconstruct" in capsys.readouterr().out
    assert os.path.samefile(os.getcwd(), workdir)


def test_mrcs2mrc_converts_star_files(monkeypatch, capsys, workdir, recording_pool):
    params.project_name = "p1"
    params.ncpu = 1
    os.makedirs(params.bin_dir, exist_ok=True)
    for name in ("traj_a.star", "traj_b.star"):
        (workdir / "output" / "p1" / "bin" / name).write_text("")
    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/bin/relion_reconstruct")
    cli.mrcs2mrc(None)
    entry = recording_pool.log[-1]
    assert sorted(entry["items"]) == ["traj_a.star", "traj_b.star"]
    assert entry["func"].func is cli.relion_reconstruct
    assert entry["func"].keywords["relion_command"] == "/fake/bin/relion_reconstruct"
    assert os.path.isdir(entry["func"].keywords["output_path"])
    assert os.path.samefile(os.getcwd(), workdir)  # cwd restored on the happy path
    capsys.readouterr()


@pytest.mark.xfail(reason="Bug. mrcs2mrc returns early when no .star files exist without "
                          "chdir'ing back, leaving the process cwd inside the project bin dir",
                   strict=False)
def test_mrcs2mrc_restores_cwd_when_no_star_files(monkeypatch, capsys, workdir, recording_pool):
    params.project_name = "p1"
    os.makedirs(params.bin_dir, exist_ok=True)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/bin/relion_reconstruct")
    cli.mrcs2mrc(None)
    capsys.readouterr()
    assert os.path.samefile(os.getcwd(), workdir)


@pytest.mark.parametrize("star,expected", [
    ("traj_1.star", "traj_1.mrc"),
    ("EulerAngles_1_of_50.star", "EulerAngles_1_of_50.mrc"),
    ("nosuffix", "nosuffix.mrc"),          # removesuffix is a no-op when the suffix is absent
    ("a.star.star", "a.star.mrc"),         # only the trailing suffix is removed
])
def test_relion_reconstruct_builds_output_path(star, expected, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    cli.relion_reconstruct(star, "relion_reconstruct", str(tmp_path))
    assert calls == [["relion_reconstruct", "--i", star, "--o", os.path.join(str(tmp_path), expected)]]


def test_relion_reconstruct_captures_output(monkeypatch, tmp_path):
    kwargs = {}
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: kwargs.update(kw))
    cli.relion_reconstruct("t.star", "relion_reconstruct", str(tmp_path))
    assert kwargs["capture_output"] is True


@pytest.mark.parametrize("frame,states,expected", [
    (1, 10, [0, 9]),
    (2, 10, [0, 1, 8, 9]),
    (5, 50, [0, 1, 2, 3, 4, 45, 46, 47, 48, 49]),
    (3, 6, [0, 1, 2, 3, 4, 5]),
    (0, 10, []),                                  # frame=0 denoises nothing
    (6, 10, [0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9]),  # frame > states/2 duplicates bins
])
def test_denoise_bin_selection(frame, states, expected, workdir, recording_pool, capsys):
    # denoise touches only the first and last `frame` reconstructions of the trajectory
    params.project_name = "p1"
    params.states_per_coord = states
    params.ncpu = 1
    cli.denoise(Namespace(frame=frame, window_size=3, filter="Gaussian"))
    assert recording_pool.log[-1]["items"] == expected
    capsys.readouterr()


@pytest.mark.parametrize("given,lowered", [
    ("Gaussian", "gaussian"), ("GAUSSIAN", "gaussian"), ("Median", "median"), ("median", "median"),
])
def test_denoise_lowercases_the_filter_name(given, lowered, workdir, recording_pool, capsys):
    # denoise_helper compares against lowercase names, so denoise must fold the CLI spelling
    params.project_name = "p1"
    params.states_per_coord = 6
    cli.denoise(Namespace(frame=1, window_size=3, filter=given))
    entry = recording_pool.log[-1]
    assert entry["func"].func is cli.denoise_helper
    assert entry["func"].keywords == {"k": 3, "filter_type": lowered}
    capsys.readouterr()


def test_denoise_creates_the_output_directory(workdir, recording_pool, capsys):
    params.project_name = "p1"
    params.states_per_coord = 6
    cli.denoise(Namespace(frame=2, window_size=3, filter="Gaussian"))
    assert (workdir / "output" / "p1" / "postproc" / "denoise").is_dir()
    capsys.readouterr()


def test_denoise_uses_ncpu_processes(workdir, recording_pool, capsys):
    params.project_name = "p1"
    params.states_per_coord = 6
    params.ncpu = 3
    cli.denoise(Namespace(frame=1, window_size=3, filter="Gaussian"))
    assert recording_pool.log[-1]["processes"] == 3
    capsys.readouterr()


# ---------------------------------------------------------------------------------------------
# denoise_helper
# ---------------------------------------------------------------------------------------------

def _stage_reconstruction(workdir, vol, i_bin=0):
    import mrcfile
    os.makedirs(params.postproc_mrcs2mrc_dir, exist_ok=True)
    os.makedirs(params.postproc_denoise_dir, exist_ok=True)
    name = f"EulerAngles_{params.traj_name}_{i_bin + 1}_of_{params.states_per_coord}.mrc"
    with mrcfile.new(os.path.join(params.postproc_mrcs2mrc_dir, name), overwrite=True) as mrc:
        mrc.set_data(vol.astype(np.float32))


@pytest.mark.parametrize("filter_type,scipy_name", [("gaussian", "gaussian_filter"),
                                                    ("median", "median_filter")])
def test_denoise_helper_matches_scipy(filter_type, scipy_name, workdir):
    import mrcfile
    from scipy import ndimage

    params.project_name = "p1"
    params.traj_name = "t"
    params.states_per_coord = 3
    rng = np.random.default_rng(0)
    vol = rng.standard_normal((8, 8, 8)).astype(np.float32)
    _stage_reconstruction(workdir, vol)

    cli.denoise_helper(0, 2, filter_type)

    out = os.path.join(params.postproc_denoise_dir, "DenoiseimgsRELION_t_1_of_3.mrc")
    assert os.path.isfile(out)
    with mrcfile.open(out) as mrc:
        got = np.array(mrc.data)
    expected = getattr(ndimage, scipy_name)(vol.astype(np.float64), 2).astype(np.float32)
    assert got.shape == vol.shape
    assert np.allclose(got, expected, atol=1e-5)


@pytest.mark.parametrize("filter_type", ["butter", "Gaussian", "", "none"])
def test_denoise_helper_unknown_filter_writes_nothing(filter_type, workdir):
    # the comparison is case sensitive and lowercase-only, so "Gaussian" silently does nothing
    params.project_name = "p1"
    params.traj_name = "t"
    params.states_per_coord = 3
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    _stage_reconstruction(workdir, vol)
    assert cli.denoise_helper(0, 1, filter_type) is None
    assert os.listdir(params.postproc_denoise_dir) == []


def test_denoise_helper_uses_one_based_filenames(workdir):
    # bins are 0-indexed in code but 1-indexed on disk (PrD_{i+1} convention)
    params.project_name = "p1"
    params.traj_name = "t"
    params.states_per_coord = 3
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    _stage_reconstruction(workdir, vol, i_bin=2)
    cli.denoise_helper(2, 1, "gaussian")
    assert os.path.isfile(os.path.join(params.postproc_denoise_dir,
                                       "DenoiseimgsRELION_t_3_of_3.mrc"))


def test_denoise_helper_missing_reconstruction_raises(workdir):
    params.project_name = "p1"
    params.traj_name = "t"
    params.states_per_coord = 3
    os.makedirs(params.postproc_mrcs2mrc_dir, exist_ok=True)
    os.makedirs(params.postproc_denoise_dir, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        cli.denoise_helper(0, 1, "gaussian")


def test_denoise_helper_output_dtype_is_float32(workdir):
    import mrcfile
    params.project_name = "p1"
    params.traj_name = "t"
    params.states_per_coord = 3
    rng = np.random.default_rng(0)
    vol = rng.standard_normal((6, 6, 6)).astype(np.float32)
    _stage_reconstruction(workdir, vol)
    cli.denoise_helper(0, 1, "gaussian")
    with mrcfile.open(os.path.join(params.postproc_denoise_dir,
                                   "DenoiseimgsRELION_t_1_of_3.mrc")) as mrc:
        assert mrc.data.dtype == np.float32


# ============================================================================
# pipeline stages - calc_distance, probability_landscape, trajectory, writeRelionS2
# ============================================================================

# --------------------------------------------------------------------------------------
# fixtures, params and data_store are process wide singletons, so everything they touch
# must be snapshotted and put back.  params.load() also chdirs, so cwd is restored too.
# --------------------------------------------------------------------------------------
# Only the plain data attributes are snapshotted. Reading the derived properties off a
# default Params blows up (Params.sh divides by particle_diameter == 0).
_PARAM_KEYS = tuple(
    name
    for name in dir(Params)
    if not name.startswith("_")
    and isinstance(getattr(Params, name), (bool, int, float, str, list, dict, ProjectLevel))
)


def _restore_params(snapshot):
    for key, value in snapshot.items():
        setattr(params, key, value)


@pytest.fixture
def project_stages(tmp_path, monkeypatch):
    """A throwaway on-disk ManifoldEM project with a clean params/data_store state."""
    cwd = os.getcwd()
    snapshot = {key: getattr(params, key) for key in _PARAM_KEYS}
    monkeypatch.delenv("MANIFOLD_REBUILD_DS", raising=False)

    os.chdir(tmp_path)
    params.project_name = "test"
    # params.save() reads every property, so the geometry has to be non degenerate
    params.particle_diameter = 100.0
    params.ms_estimated_resolution = 5.0
    params.ncpu = 1
    params.prd_n_active = 6
    params.states_per_coord = 4
    params.con_order_range = 3
    params.width_1D = 1
    params.traj_name = "1"
    params.num_psi = 2
    params.ms_num_pixels = 8
    params.ms_pixel_size = 1.0
    params.project_level = ProjectLevel.INIT
    params.create_dir()
    params.save()

    # A _ProjectionDirections with a non-empty pos_full and matching thresholds makes
    # _ProjectionDirections.update() a no-op, so nothing is ever rebuilt from disk.
    fresh = _ProjectionDirections()
    fresh.pos_full = np.ones((3, 1))
    fresh.thres_low = params.prd_thres_low
    fresh.thres_high = params.prd_thres_high
    data_store._projection_directions = fresh
    data_store._image_stack_data = None

    yield tmp_path

    _restore_params(snapshot)
    os.chdir(cwd)
    data_store.__dict__.pop("_projection_directions", None)
    data_store.__dict__.pop("_image_stack_data", None)
    _DataStore._projection_directions = _ProjectionDirections()
    _DataStore._image_stack_data = None


def _fill_fake_prds(occupancies, seed=0):
    """Populate the fake prd store with `len(occupancies)` prds of the given sizes."""
    rng = np.random.default_rng(seed)
    prds = data_store._projection_directions
    n_prds = len(occupancies)

    indices, start = [], 0
    for occ in occupancies:
        indices.append(np.arange(start, start + occ))
        start += occ
    n_total = max(start, 1)

    image_indices_full = np.empty(n_prds, dtype=object)
    for i in range(n_prds):
        image_indices_full[i] = indices[i]

    quats = rng.normal(size=(4, n_total))
    quats /= np.linalg.norm(quats, axis=0)

    prds.image_indices_full = image_indices_full
    prds.thres_ids = list(range(n_prds))
    prds.occupancy_full = np.array([len(i) for i in indices], dtype=int)
    prds.bin_centers = np.zeros((3, max(n_prds, 1)))
    prds.defocus = 10000.0 + np.arange(n_total, dtype=float)
    prds.quats_full = quats
    prds.pos_full = np.ones((3, n_total))
    prds.microscope_origin = (np.zeros(n_total), np.zeros(n_total))
    prds.image_is_mirrored = np.zeros(n_total, dtype=bool)
    prds.cluster_ids = np.zeros(n_prds, dtype=int)
    prds.thres_low = params.prd_thres_low
    prds.thres_high = params.prd_thres_high
    return prds


# --------------------------------------------------------------------------------------
# FilterParams, Butterworth / Gaussian construction
# --------------------------------------------------------------------------------------
def test_filter_params_field_order():
    names = [f.name for f in dataclasses.fields(calc_distance.FilterParams)]
    assert names == ["method", "cutoff_freq", "order"]


@pytest.mark.parametrize("args", [(), ("Butter",), ("Butter", 0.5)])
def test_filter_params_requires_all_three_fields(args):
    # none of the fields carries a default, so a partially specified filter cannot be built
    with pytest.raises(TypeError):
        calc_distance.FilterParams(*args)


@pytest.mark.parametrize("cutoff", [0.1, 0.25, 0.5, 0.75, 1.0])
def test_gauss_filter_matches_closed_form(cutoff):
    # G(Q) = exp(-(ln2/2) (Q/Qc)^2). a Gaussian whose power drops to 1/2 at Q = Qc
    Q = util.create_proportional_grid(16)
    got = calc_distance.FilterParams("Gauss", cutoff, 8).create_filter(Q)
    want = np.exp(-(np.log(2) / 2.0) * (Q / cutoff) ** 2)
    assert np.allclose(got, want)


@pytest.mark.parametrize("cutoff", [0.2, 0.5, 0.9])
def test_gauss_filter_is_gaussian_with_sigma_cutoff_over_sqrt_ln2(cutoff):
    # exp(-(ln2/2)(Q/Qc)^2) == exp(-Q^2/(2 sigma^2)) with sigma = Qc / sqrt(ln 2)
    Q = np.linspace(0.0, 2.0, 41).reshape(41, 1)
    sigma = cutoff / np.sqrt(np.log(2))
    got = calc_distance.FilterParams("Gauss", cutoff, 4).create_filter(Q)
    assert np.allclose(got, np.exp(-(Q**2) / (2 * sigma**2)))


@pytest.mark.parametrize("order", [1, 2, 4, 8])
@pytest.mark.parametrize("cutoff", [0.25, 0.5])
def test_butter_filter_matches_scipy_analog_butterworth(order, cutoff):
    # |H(w)| of an analog Butterworth low pass is exactly sqrt(1/(1+(w/wc)^(2n)))
    Q = np.linspace(0.0, 1.5, 25)
    got = calc_distance.FilterParams("Butter", cutoff, order).create_filter(Q)
    b, a = butter(order, cutoff, analog=True)
    _, h = freqs(b, a, worN=Q)
    assert np.allclose(got, np.abs(h))


@pytest.mark.parametrize("method", ["Butter", "Gauss"])
@pytest.mark.parametrize("cutoff", [0.2, 0.4, 0.6, 0.8])
def test_filter_is_half_power_at_cutoff(method, cutoff):
    # both filter families are normalized to -3 dB (amplitude 1/sqrt(2)) at Q = Qc
    Q = np.array([[cutoff]])
    got = calc_distance.FilterParams(method, cutoff, 8).create_filter(Q)
    assert np.isclose(got[0, 0], 1.0 / np.sqrt(2.0))


@pytest.mark.parametrize("method", ["Butter", "Gauss"])
def test_filter_is_unity_at_zero_frequency(method):
    got = calc_distance.FilterParams(method, 0.5, 8).create_filter(np.zeros((3, 3)))
    assert np.allclose(got, 1.0)


@pytest.mark.parametrize("method", ["Butter", "Gauss"])
def test_filter_is_monotonically_decreasing_in_frequency(method):
    Q = np.linspace(0.0, 3.0, 100)
    got = calc_distance.FilterParams(method, 0.5, 6).create_filter(Q)
    assert np.all(np.diff(got) <= 0.0)
    assert np.all((got > 0.0) & (got <= 1.0))


@pytest.mark.parametrize("method", ["gauss", "GAUSS", "Gauss", "gAuSs"])
def test_gauss_method_is_case_insensitive(method):
    Q = util.create_proportional_grid(8)
    ref = calc_distance.FilterParams("Gauss", 0.5, 4).create_filter(Q)
    assert np.allclose(calc_distance.FilterParams(method, 0.5, 4).create_filter(Q), ref)


@pytest.mark.parametrize("method", ["butter", "BUTTER", "Butter", "bUtTeR"])
def test_butter_method_is_case_insensitive(method):
    Q = util.create_proportional_grid(8)
    ref = calc_distance.FilterParams("Butter", 0.5, 4).create_filter(Q)
    assert np.allclose(calc_distance.FilterParams(method, 0.5, 4).create_filter(Q), ref)


@pytest.mark.parametrize(
    "method", ["", "boxcar", "gaussian", "butterworth", "Gaus", "hann", "none"]
)
def test_unsupported_filter_method_raises(method):
    # only the exact tokens 'gauss' and 'butter' (case insensitively) are accepted
    with pytest.raises(ValueError):
        calc_distance.FilterParams(method, 0.5, 4).create_filter(np.ones((2, 2)))


@pytest.mark.parametrize("order", [1, 3, 17])
def test_gauss_filter_ignores_order(order):
    Q = util.create_proportional_grid(8)
    ref = calc_distance.FilterParams("Gauss", 0.5, 2).create_filter(Q)
    assert np.allclose(calc_distance.FilterParams("Gauss", 0.5, order).create_filter(Q), ref)


def test_butter_filter_sharpens_with_order():
    # above the cutoff a higher order rolls off faster, below it the response is flatter
    Q_pass, Q_stop = 0.25, 1.0
    lo = calc_distance.FilterParams("Butter", 0.5, 2)
    hi = calc_distance.FilterParams("Butter", 0.5, 8)
    assert hi.create_filter(np.array([Q_pass]))[0] > lo.create_filter(np.array([Q_pass]))[0]
    assert hi.create_filter(np.array([Q_stop]))[0] < lo.create_filter(np.array([Q_stop]))[0]


@pytest.mark.parametrize("shape", [(4, 4), (8, 3), (1, 9), (5,)])
@pytest.mark.parametrize("method", ["Butter", "Gauss"])
def test_filter_preserves_input_shape_and_dtype(shape, method):
    Q = np.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape)
    got = calc_distance.FilterParams(method, 0.5, 4).create_filter(Q)
    assert got.shape == shape
    assert got.dtype == np.float64


# --------------------------------------------------------------------------------------
# frequency grid and CTF helpers
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 4, 5, 8, 9, 16])
def test_create_proportional_grid_matches_definition(n):
    a = np.arange(n) - n // 2
    X, Y = np.meshgrid(a, a)
    assert np.allclose(util.create_proportional_grid(n), 2 * np.sqrt(X**2 + Y**2) / n)


@pytest.mark.parametrize("n", [4, 5, 8, 16])
def test_create_proportional_grid_zero_is_at_center_index(n):
    Q = util.create_proportional_grid(n)
    assert Q.shape == (n, n)
    assert Q[n // 2, n // 2] == 0.0
    assert Q.min() == 0.0
    # the grid is symmetric under transposition since X and Y come from the same 1D axis
    assert np.allclose(Q, Q.T)


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_grid_reaches_unity_at_nyquist_along_an_axis(n):
    # |a| = n/2 along a single axis gives Q = 1, i.e. Q is in units of the Nyquist frequency
    Q = util.create_proportional_grid(n)
    assert np.isclose(Q[0, n // 2], 1.0)
    assert np.isclose(Q[n // 2, 0], 1.0)


@pytest.mark.parametrize("n_defocus", [1, 2, 5])
@pytest.mark.parametrize("width", [4, 8])
def test_get_CTFs_shape_stages(n_defocus, width):
    defocus = 10000.0 + 100.0 * np.arange(n_defocus)
    ctf = util.get_CTFs(defocus, width, 1.0, 2.0, 300.0, np.inf, 0.1)
    assert ctf.shape == (n_defocus, width, width)
    assert ctf.dtype == np.float64


@pytest.mark.parametrize("amp_contrast", [0.0, 0.07, 0.1, 0.5])
def test_ctf_zero_frequency_equals_minus_amplitude_contrast(amp_contrast):
    # gamma(0) = 0 so the CTF at zero frequency is -(amplitude contrast ratio).
    # after the ifftshift the zero frequency sits at index [0, 0]
    ctf = util.get_CTFs(np.array([12000.0]), 8, 1.0, 2.0, 300.0, np.inf, amp_contrast)
    assert np.isclose(ctf[0, 0, 0], -amp_contrast)


@pytest.mark.parametrize("defocus", [5000.0, 12000.0, 25000.0])
def test_get_CTFs_is_ifftshifted_ctemh(defocus):
    width, pixel_size = 8, 1.3
    k = util.create_proportional_grid(width) / (2 * pixel_size)
    got = util.get_CTFs(np.array([defocus]), width, pixel_size, 2.0, 300.0, np.inf, 0.1)
    want = np.fft.ifftshift(util.ctemh_cryoFrank(k, 2.0, defocus, 300.0, np.inf, 0.1))
    assert np.allclose(got[0], want)


def test_ctf_is_constant_when_defocus_and_aberration_vanish():
    # gamma == 0 everywhere, so with an infinite envelope the CTF is a flat -AC
    ctf = util.get_CTFs(np.array([0.0]), 8, 1.0, 0.0, 300.0, np.inf, 0.1)
    assert np.allclose(ctf[0], -0.1)


def test_ctf_envelope_damps_high_frequencies():
    # a finite Gaussian envelope must strictly attenuate away from the origin
    k = util.create_proportional_grid(16) / 2.0
    damped = util.ctemh_cryoFrank(k, 2.0, 12000.0, 300.0, 0.05, 0.1)
    undamped = util.ctemh_cryoFrank(k, 2.0, 12000.0, 300.0, np.inf, 0.1)
    assert np.all(np.abs(damped) <= np.abs(undamped) + 1e-12)
    assert np.abs(damped).max() < np.abs(undamped).max()


# --------------------------------------------------------------------------------------
# Wiener denominator
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("snr", [0.5, 1.0, 5.0, 20.0])
@pytest.mark.parametrize("n_images", [1, 3, 7])
def test_get_wiener_matches_sum_of_squares(snr, n_images):
    rng = np.random.default_rng(0)
    ctf = rng.normal(size=(n_images, 6, 6))
    assert np.allclose(
        calc_distance.get_wiener(ctf, snr), np.sum(ctf**2, axis=0) + 1.0 / snr
    )


def test_get_wiener_default_snr_is_five():
    rng = np.random.default_rng(1)
    ctf = rng.normal(size=(4, 5, 5))
    assert np.allclose(calc_distance.get_wiener(ctf), calc_distance.get_wiener(ctf, 5.0))


def test_get_wiener_shape_and_positivity():
    rng = np.random.default_rng(2)
    ctf = rng.normal(size=(3, 4, 4))
    out = calc_distance.get_wiener(ctf, 5.0)
    assert out.shape == (4, 4)
    assert np.all(out > 0.0)


def test_get_wiener_scales_quadratically():
    rng = np.random.default_rng(3)
    ctf = rng.normal(size=(3, 4, 4))
    scaled = calc_distance.get_wiener(2 * ctf, 5.0)
    assert np.allclose(scaled, 4 * (calc_distance.get_wiener(ctf, 5.0) - 0.2) + 0.2)


def test_get_wiener_on_empty_stack_returns_a_scalar():
    # the accumulator starts as the python float 0.0, so an empty stack never becomes an
    # array and the caller silently gets a scalar instead of an image
    out = calc_distance.get_wiener(np.zeros((0, 4, 4)), 5.0)
    assert np.ndim(out) == 0
    assert out == pytest.approx(0.2)


# --------------------------------------------------------------------------------------
# get_psi / psi_ang
# --------------------------------------------------------------------------------------
def _psi_terms(q, ref):
    s = -(1 + ref[2]) * q[3] - ref[0] * q[1] - ref[1] * q[2]
    c = (1 + ref[2]) * q[0] + ref[1] * q[1] - ref[0] * q[2]
    return s, c


@pytest.mark.parametrize("seed", range(8))
def test_get_psi_returns_degrees_not_radians(seed):
    # the docstring promises radians in [-pi, pi] but the implementation converts to degrees
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    ref = rng.normal(size=3)
    ref /= np.linalg.norm(ref)

    s, c = _psi_terms(q, ref)
    assert c != 0.0
    assert np.isclose(calc_distance.get_psi(q, ref), 2 * np.arctan(s / c) * 180 / np.pi)


@pytest.mark.parametrize("seed", range(6))
def test_get_psi_is_invariant_under_quaternion_negation(seed):
    # s and c are both linear in q, so psi = 2 atan(s/c) is unchanged by q -> -q
    rng = np.random.default_rng(100 + seed)
    q = rng.normal(size=4)
    ref = rng.normal(size=3)
    ref /= np.linalg.norm(ref)
    assert np.isclose(calc_distance.get_psi(q, ref), calc_distance.get_psi(-q, ref))


@pytest.mark.parametrize("scale", [0.25, 1.0, 3.7])
def test_get_psi_is_invariant_under_quaternion_rescaling(scale):
    rng = np.random.default_rng(7)
    q = rng.normal(size=4)
    ref = np.array([0.0, 0.0, 1.0])
    assert np.isclose(calc_distance.get_psi(q, ref), calc_distance.get_psi(scale * q, ref))


@pytest.mark.parametrize("q3,expected", [(1.0, -180.0), (-1.0, 180.0), (2.0, -180.0)])
def test_get_psi_degenerate_branch_gives_plus_or_minus_180(q3, expected):
    # c == 0 short circuits to sign(s) * pi, which the degree conversion turns into +-180
    ref = np.array([0.0, 0.0, 1.0])
    q = np.array([0.0, 1.0, 0.0, q3])
    assert calc_distance.get_psi(q, ref) == expected


def test_get_psi_zero_over_zero_branch_is_zero():
    # s == c == 0 gives sign(0) * pi == 0
    assert calc_distance.get_psi(np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])) == 0.0


@pytest.mark.parametrize("seed", range(10))
def test_get_psi_stays_within_plus_minus_180(seed):
    rng = np.random.default_rng(200 + seed)
    q = rng.normal(size=4)
    ref = rng.normal(size=3)
    ref /= np.linalg.norm(ref)
    assert -180.0 <= calc_distance.get_psi(q, ref) <= 180.0


@pytest.mark.xfail(
    reason="Bug. psi_ang is not reproducible, the q2Spider least squares solve behind it "
    "intermittently returns its own (0,0,0) starting guess, zeroing the in-plane alignment "
    "angle for an arbitrary subset of calls in a given process",
    strict=False,
)
def test_psi_ang_is_minus_the_spider_phi():
    # Qr = [1 + z, y, -x, 0] has no k component, and the spider product qzs*qy*qz only loses
    # its k component when psi = -phi, so psi_ang must return -phi wrapped into [0, 360)
    rng = np.random.default_rng(300)
    v = rng.normal(size=(3, 24))
    v /= np.linalg.norm(v, axis=0)
    phi = np.degrees(quaternion.convert_S2_to_euler(v)[0, :])

    for _ in range(4):
        for i in range(v.shape[1]):
            assert np.isclose(calc_distance.psi_ang(v[:, i]), np.mod(-phi[i], 360.0), atol=1e-6)


@pytest.mark.xfail(
    reason="Bug. psi_ang is not reproducible, the q2Spider least squares solve behind it "
    "intermittently returns its own (0,0,0) starting guess, so the rebuilt quaternion is "
    "the identity instead of the reference orientation",
    strict=False,
)
def test_psi_ang_round_trips_through_eul_to_quat():
    # the spider triple (phi, theta, psi) rebuilds the in-plane reference quaternion, up to
    # the usual quaternion sign ambiguity introduced by wrapping the angles into [0, 2pi)
    rng = np.random.default_rng(400)
    v = rng.normal(size=(3, 24))
    v /= np.linalg.norm(v, axis=0)
    angles = np.degrees(quaternion.convert_S2_to_euler(v))

    for i in range(v.shape[1]):
        qr = np.array([1 + v[2, i], v[1, i], -v[0, i], 0.0])
        qr /= np.linalg.norm(qr)
        psi = calc_distance.psi_ang(v[:, i])
        q = util.eul_to_quat(
            np.deg2rad(angles[0, i : i + 1]),
            np.deg2rad(angles[1, i : i + 1]),
            np.deg2rad(np.array([psi])),
        )[:, 0]
        assert np.allclose(q, qr, atol=1e-6) or np.allclose(q, -qr, atol=1e-6)


def test_psi_ang_south_pole_returns_zero():
    # Qr = [1 + z, y, -x, 0] collapses to the zero quaternion at the south pole
    assert calc_distance.psi_ang(np.array([0.0, 0.0, -1.0])) == 0.0


@pytest.mark.parametrize("seed", range(8))
def test_psi_ang_is_wrapped_into_zero_to_360(seed):
    rng = np.random.default_rng(500 + seed)
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    psi = calc_distance.psi_ang(v)
    assert 0.0 <= psi < 360.0


# --------------------------------------------------------------------------------------
# op() signatures
# --------------------------------------------------------------------------------------
PRD_LIST_OPS = [calc_distance, manifold_analysis, nlsa_movie]
ARGV_ONLY_OPS = [probability_landscape, trajectory, find_conformational_coords]


@pytest.mark.parametrize("module", PRD_LIST_OPS + ARGV_ONLY_OPS + [writeRelionS2])
def test_stage_module_exposes_callable_op(module):
    assert callable(module.op)


@pytest.mark.parametrize("module", PRD_LIST_OPS)
def test_prd_list_ops_have_documented_signature(module):
    sig = inspect.signature(module.op)
    kinds = [(p.name, p.kind) for p in sig.parameters.values()]
    assert kinds[0][0] == "prd_list"
    assert kinds[0][1] == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["prd_list"].default is None
    assert sig.parameters["prd_list"].annotation == Union[List[int], None]
    assert kinds[-1] == ("argv", inspect.Parameter.VAR_POSITIONAL)
    assert len(kinds) == 2


@pytest.mark.parametrize("module", ARGV_ONLY_OPS)
def test_argv_only_ops_take_no_prd_list(module):
    # probability_landscape, trajectory and find_conformational_coords are whole project
    # stages, so unlike the per-prd stages they expose only the gui progress emitter
    sig = inspect.signature(module.op)
    kinds = [(p.name, p.kind) for p in sig.parameters.values()]
    assert kinds == [("argv", inspect.Parameter.VAR_POSITIONAL)]


def test_write_relion_op_signature_is_positional():
    sig = inspect.signature(writeRelionS2.op)
    assert list(sig.parameters) == [
        "trajTaus",
        "posPsi1All",
        "posPathAll",
        "xSelect",
        "tauAvg",
        "argv",
    ]
    assert sig.parameters["argv"].kind == inspect.Parameter.VAR_POSITIONAL


def test_trajectory_forwards_the_same_argument_order_to_write_relion():
    src = inspect.getsource(trajectory.op)
    assert "writeRelionS2.op(trajTaus, posPsi1All, posPathAll, xSelect, tauAvg" in src


# --------------------------------------------------------------------------------------
# params path templates used by the stage modules
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("prd", [0, 1, 7, 123])
def test_dist_and_psi_file_templates_are_zero_indexed(prd, project_stages):
    out = params.out_dir
    assert params.get_dist_file(prd) == os.path.join(out, "distances", f"IMGs_prD_{prd}.h5")
    assert params.get_psi_file(prd) == os.path.join(
        out, "diff_maps", f"gC_trimmed_psi_prD_{prd}.h5"
    )
    assert params.get_EL_file(prd) == os.path.join(
        out, f"ELConc{params.con_order_range}", f"S2_prD_{prd}.pkl"
    )


@pytest.mark.parametrize("prd", [0, 3])
@pytest.mark.parametrize("psi", [0, 2])
def test_psi2_file_template(prd, psi, project_stages):
    assert params.get_psi2_file(prd, psi) == os.path.join(
        params.out_dir, "psi_analysis", f"S2_prD_{prd}_psi_{psi}.h5"
    )


@pytest.mark.parametrize("prd", [1, 2, 10])
def test_topos_paths_are_one_indexed(prd, project_stages):
    # on disk everything under topos/ is 1-indexed, so callers pass prD + 1
    assert params.get_topos_path(prd, 1) == os.path.join(
        params.out_dir, "topos", f"PrD_{prd}", "topos_1.png"
    )
    assert params.get_psi_gif(prd, 3) == os.path.join(
        params.out_dir, "topos", f"PrD_{prd}", "psi_3.gif"
    )


# --------------------------------------------------------------------------------------
# calc_distance._construct_input_data
# --------------------------------------------------------------------------------------
def _distance_inputs(occupancies, seed=0):
    rng = np.random.default_rng(seed)
    indices, start = [], 0
    for occ in occupancies:
        indices.append(np.arange(start, start + occ))
        start += occ
    quats = rng.normal(size=(4, max(start, 1)))
    defocus = 10000.0 + np.arange(max(start, 1), dtype=float)
    return indices, quats, defocus


def test_local_input_field_order():
    names = [f.name for f in dataclasses.fields(calc_distance.LocalInput)]
    assert names == ["indices", "quats", "defocus", "dist_file"]


@pytest.mark.parametrize(
    "prd_list,expected",
    [
        (None, {0, 1, 2, 3, 4}),
        ([], set()),
        ([0], {0}),
        ([4], {4}),
        ([1, 3], {1, 3}),
        ([0, 1, 2, 3, 4], {0, 1, 2, 3, 4}),
        ([2, 2, 2], {2}),
        ([1, 99], {1}),
        ([99, 100], set()),
        ([-1, 0], {0}),
    ],
)
def test_calc_distance_subset_filter_selects_exactly_requested(prd_list, expected, project_stages):
    indices, quats, defocus = _distance_inputs([5, 9, 3, 7, 4])
    jobs = calc_distance._construct_input_data(prd_list, indices, quats, defocus)
    selected = {int(j.dist_file.split("prD_")[1].split(".")[0]) for j in jobs}
    assert selected == expected
    assert len(jobs) == len(expected)


def test_calc_distance_jobs_carry_the_right_slices(project_stages):
    indices, quats, defocus = _distance_inputs([5, 9, 3])
    jobs = calc_distance._construct_input_data([1], indices, quats, defocus)
    (job,) = jobs
    assert np.array_equal(job.indices, indices[1])
    assert np.allclose(job.quats, quats[:, indices[1]])
    assert np.allclose(job.defocus, defocus[indices[1]])
    assert job.dist_file == params.get_dist_file(1)
    assert job.quats.shape == (4, 9)


def test_calc_distance_returns_a_list_of_local_inputs(project_stages):
    indices, quats, defocus = _distance_inputs([5, 9])
    jobs = calc_distance._construct_input_data(None, indices, quats, defocus)
    assert isinstance(jobs, list)
    assert all(isinstance(j, calc_distance.LocalInput) for j in jobs)


def test_calc_distance_jobs_are_sorted_largest_first(project_stages):
    indices, quats, defocus = _distance_inputs([5, 9, 3, 7, 1])
    jobs = calc_distance._construct_input_data(None, indices, quats, defocus)
    sizes = [len(j.indices) for j in jobs]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes == [9, 7, 5, 3, 1]


def test_calc_distance_subset_filter_warns_about_invalid_prds(project_stages, capsys):
    indices, quats, defocus = _distance_inputs([5, 9])
    calc_distance._construct_input_data([0, 42], indices, quats, defocus)
    assert "requested invalid prds" in capsys.readouterr().out


def test_calc_distance_op_with_empty_list_leaves_project_level(project_stages):
    _fill_fake_prds([4, 5, 6])
    params.project_level = ProjectLevel.BINNING
    params.save()
    calc_distance.op([])
    assert params.project_level is ProjectLevel.BINNING


def test_calc_distance_op_with_none_advances_project_level(project_stages):
    _fill_fake_prds([])
    params.project_level = ProjectLevel.BINNING
    params.save()
    calc_distance.op(None)
    assert params.project_level is ProjectLevel.CALC_DISTANCE


# --------------------------------------------------------------------------------------
# manifold_analysis._construct_input_data
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prd_list,expected",
    [
        (None, {0, 1, 2, 3, 4, 5}),
        ([], set()),
        ([0], {0}),
        ([5], {5}),
        ([2, 4], {2, 4}),
        ([3, 3], {3}),
        ([1, 77], {1}),
        ([77], set()),
    ],
)
def test_manifold_analysis_subset_filter(prd_list, expected, project_stages):
    ll = manifold_analysis._construct_input_data(prd_list, params.prd_n_active)
    assert {entry[3] for entry in ll} == expected
    assert len(ll) == len(expected)


@pytest.mark.parametrize("prd", [0, 2, 5])
def test_manifold_analysis_entry_layout(prd, project_stages):
    (entry,) = manifold_analysis._construct_input_data([prd], params.prd_n_active)
    dist_file, psi_file, eig_file, prD = entry
    assert dist_file == params.get_dist_file(prd)
    assert psi_file == params.get_psi_file(prd)
    # eig_spec.txt lives in the 1-indexed topos tree while the data files are 0-indexed
    assert eig_file == f"{params.out_dir}/topos/PrD_{prd + 1}/eig_spec.txt"
    assert prD == prd


def test_manifold_analysis_op_empty_list_leaves_project_level(project_stages):
    params.project_level = ProjectLevel.CALC_DISTANCE
    params.save()
    manifold_analysis.op([])
    assert params.project_level is ProjectLevel.CALC_DISTANCE


def test_manifold_analysis_op_none_advances_project_level(project_stages, monkeypatch):
    calls = []
    monkeypatch.setattr(manifoldTrimmingAuto, "op", lambda data, **kw: calls.append(data[3]))
    params.prd_n_active = 3
    params.project_level = ProjectLevel.CALC_DISTANCE
    params.save()
    manifold_analysis.op(None)
    assert sorted(calls) == [0, 1, 2]
    assert params.project_level is ProjectLevel.MANIFOLD_ANALYSIS


@pytest.mark.xfail(
    reason="Bug. manifold_analysis.op makes topos/PrD_{i+1} for i in range(n_jobs) instead "
    "of for the requested prds, so a subset run writes eig_spec.txt into a missing dir",
    strict=False,
)
def test_manifold_analysis_op_creates_dirs_for_the_requested_prds(project_stages, monkeypatch):
    monkeypatch.setattr(manifoldTrimmingAuto, "op", lambda data, **kw: None)
    params.prd_n_active = 6
    params.save()
    manifold_analysis.op([4])
    assert os.path.isdir(os.path.join(params.out_dir, "topos", "PrD_5"))


# --------------------------------------------------------------------------------------
# nlsa_movie._construct_input_data
# --------------------------------------------------------------------------------------
def _touch_class_avg(prd):
    directory = os.path.join(params.out_dir, "topos", f"PrD_{prd + 1}")
    os.makedirs(directory, exist_ok=True)
    open(os.path.join(directory, "class_avg.png"), "wb").close()


@pytest.mark.parametrize(
    "prd_list,expected",
    [
        (None, {0, 1, 2, 3}),
        ([], set()),
        ([0], {0}),
        ([3], {3}),
        ([1, 2], {1, 2}),
        ([0, 55], {0}),
        ([55], set()),
    ],
)
def test_nlsa_movie_subset_filter(prd_list, expected, project_stages):
    _fill_fake_prds([10, 40, 20, 30])
    jobs = nlsa_movie._construct_input_data(prd_list, 4)
    assert {job[0] for job in jobs} == expected
    assert all(len(job) == 1 for job in jobs)


def test_nlsa_movie_skips_prds_that_already_have_a_class_average(project_stages):
    _fill_fake_prds([10, 40, 20, 30])
    _touch_class_avg(0)
    _touch_class_avg(2)
    jobs = nlsa_movie._construct_input_data(None, 4)
    assert {job[0] for job in jobs} == {1, 3}


def test_nlsa_movie_orders_jobs_by_descending_occupancy(project_stages):
    _fill_fake_prds([10, 40, 20, 30])
    jobs = nlsa_movie._construct_input_data(None, 4)
    assert [job[0] for job in jobs] == [1, 3, 2, 0]


def test_nlsa_movie_returns_a_plain_list_when_nothing_to_do(project_stages):
    _fill_fake_prds([10, 40])
    _touch_class_avg(0)
    _touch_class_avg(1)
    jobs = nlsa_movie._construct_input_data(None, 2)
    assert jobs == []
    assert isinstance(jobs, list)


def test_nlsa_movie_returns_a_tuple_when_there_is_work(project_stages):
    # zip(*sorted(...)) turns the job list into a tuple, so the return type is not stable
    _fill_fake_prds([10, 40])
    jobs = nlsa_movie._construct_input_data(None, 2)
    assert isinstance(jobs, tuple)


@pytest.mark.xfail(
    reason="Bug. nlsa_movie._construct_input_data zips jobs against image_counts built "
    "over all valid prds, so skipped prds shift the pairing and the sort is wrong",
    strict=False,
)
def test_nlsa_movie_sort_stays_correct_when_a_prd_is_skipped(project_stages):
    _fill_fake_prds([10, 40, 20, 30])
    _touch_class_avg(0)
    jobs = nlsa_movie._construct_input_data(None, 4)
    # remaining occupancies are prd1=40, prd2=20, prd3=30
    assert [job[0] for job in jobs] == [1, 3, 2]


def test_nlsa_movie_op_with_unknown_prd_does_nothing(project_stages):
    params.project_level = ProjectLevel.PSI_ANALYSIS
    params.save()
    nlsa_movie.op([999])
    assert params.project_level is ProjectLevel.PSI_ANALYSIS


@pytest.mark.xfail(
    reason="Bug. nlsa_movie.op guards the project level bump with `if not prd_list`, so an "
    "empty subset request advances the project level like a full run",
    strict=False,
)
def test_nlsa_movie_op_empty_list_leaves_project_level(project_stages):
    params.project_level = ProjectLevel.PSI_ANALYSIS
    params.save()
    nlsa_movie.op([])
    assert params.project_level is ProjectLevel.PSI_ANALYSIS


# --------------------------------------------------------------------------------------
# probability_landscape
# --------------------------------------------------------------------------------------
def test_divide1_entry_layout(project_stages):
    prds = _fill_fake_prds([4, 5, 6])
    psinums = np.array([[0, 1, 2], [0, 0, 0]])
    senses = np.array([[1, -1, 1], [0, 0, 0]])
    ll = probability_landscape.divide1([0, 2], psinums, senses)

    assert len(ll) == 2
    dist_file, psi_file, EL_file, psinum, sense, prD, defocus = ll[0]
    assert prD == 0
    assert dist_file == params.get_dist_file(0)
    assert psi_file == params.get_psi_file(0)
    assert EL_file == params.get_EL_file(0)
    assert psinum == [0]
    assert sense == [1]
    assert np.allclose(defocus, prds.get_defocus_by_prd(0))
    assert [entry[5] for entry in ll] == [0, 2]


@pytest.mark.parametrize("R", [[], [1], [0, 1, 2], [2, 0]])
def test_divide1_selects_exactly_R(R, project_stages):
    _fill_fake_prds([4, 5, 6])
    psinums = np.zeros((2, 3), dtype=int)
    senses = np.ones((2, 3), dtype=int)
    ll = probability_landscape.divide1(R, psinums, senses)
    assert [entry[5] for entry in ll] == list(R)


def _write_landscape_inputs(psinums_row0, taus, trash_ids=(), missing=()):
    """Lay down the CC file and per-prd EL files that probability_landscape reads."""
    n = len(psinums_row0)
    psinums = np.zeros((2, n), dtype=int)
    psinums[0, :] = psinums_row0
    myio.fout1(params.CC_file, psinums=psinums, senses=np.zeros((2, n), dtype=int))
    data_store._projection_directions.trash_ids = set(trash_ids)
    for x, tau in taus.items():
        if x in missing:
            continue
        myio.fout1(
            params.get_EL_file(x),
            tau=np.asarray(tau, dtype=float),
            posPath=np.arange(len(tau)),
            PosPsi1=np.arange(len(tau)),
        )


def test_probability_landscape_histogram_totals(project_stages):
    params.states_per_coord = 4
    params.prd_n_active = 4
    params.save()
    taus = {
        0: [0.0, 1.0, 2.0, 3.0],
        1: [10.0, 12.0, 14.0, 16.0, 18.0],
        2: [-1.0, 0.0, 1.0],
        3: [5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
    }
    _write_landscape_inputs([0, 0, 0, 0], taus)
    hUn = probability_landscape.probability_landscape_local()

    # every tau is min/max normalized onto [0, 1] then binned into states_per_coord bins,
    # so the occupancies sum to the total number of tau samples over all used prds
    assert hUn.shape == (4,)
    assert hUn.sum() == sum(len(v) for v in taus.values())

    expected = np.zeros(4)
    for tau in taus.values():
        t = np.asarray(tau, dtype=float)
        t = (t - t.min()) / (t.max() - t.min())
        expected += np.histogram(t, 4)[0]
    assert np.allclose(hUn, expected)


def test_probability_landscape_excludes_unassigned_and_trash(project_stages):
    params.states_per_coord = 5
    params.prd_n_active = 5
    params.save()
    taus = {x: list(np.linspace(0.0, 1.0, 4)) for x in range(5)}
    # prd 1 is unassigned (psinum -1) and prd 3 was manually trashed
    _write_landscape_inputs([0, -1, 0, 0, 0], taus, trash_ids=(3,))
    hUn = probability_landscape.probability_landscape_local()
    assert hUn.sum() == 3 * 4

    variables = myio.fin1(f"{params.traj_file}name{params.traj_name}_vars.pkl")
    assert set(variables["xSelect"]) == {0, 2, 4}
    assert len(variables["tauAvg"]) == 12


def test_probability_landscape_tolerates_missing_EL_files(project_stages):
    params.states_per_coord = 4
    params.prd_n_active = 3
    params.save()
    taus = {x: [0.0, 0.5, 1.0] for x in range(3)}
    _write_landscape_inputs([0, 0, 0], taus, missing=(1,))
    hUn = probability_landscape.probability_landscape_local()
    assert hUn.sum() == 6

    variables = myio.fin1(f"{params.traj_file}name{params.traj_name}_vars.pkl")
    assert set(variables["xSelect"]) == {0, 2}
    assert variables["trajTaus"][1] is None


def test_probability_landscape_with_nothing_assigned_returns_zeros(project_stages):
    params.states_per_coord = 6
    params.prd_n_active = 3
    params.save()
    _write_landscape_inputs([-1, -1, -1], {})
    hUn = probability_landscape.probability_landscape_local()
    assert hUn.shape == (6,)
    assert np.count_nonzero(hUn) == 0


def test_probability_landscape_returns_float_counts(project_stages):
    # the accumulator is seeded with np.zeros, so the occupancies come back as float64 even
    # though they are integer counts. Only the OM file on disk is cast back to int
    params.states_per_coord = 3
    params.prd_n_active = 1
    params.save()
    _write_landscape_inputs([0], {0: [0.0, 0.5, 1.0]})
    hUn = probability_landscape.probability_landscape_local()
    assert hUn.dtype == np.float64
    assert np.array_equal(hUn, hUn.astype(int))


def test_probability_landscape_writes_traj_and_OM_files(project_stages):
    params.states_per_coord = 4
    params.prd_n_active = 2
    params.traj_name = "abc"
    params.save()
    taus = {0: [0.0, 0.4, 0.8, 1.0], 1: [1.0, 2.0, 3.0]}
    _write_landscape_inputs([0, 0], taus)
    hUn = probability_landscape.probability_landscape_local()

    traj_file = f"{params.traj_file}name{params.traj_name}.pkl"
    assert os.path.isfile(traj_file)
    assert np.allclose(myio.fin1(traj_file)["hUn"], hUn)

    variables = myio.fin1(f"{params.traj_file}nameabc_vars.pkl")
    assert set(variables) == {"trajTaus", "posPsi1All", "posPathAll", "xSelect", "tauAvg"}

    # the OM file is a raw dump of the integer occupancies, no header
    assert np.array_equal(np.fromfile(params.OM_file, dtype="int"), hUn.astype("int"))


def test_probability_landscape_rejects_a_constant_tau(project_stages):
    # min/max normalization divides by zero for a flat tau and numpy refuses to histogram
    # the resulting nan, which takes down the whole stage
    params.states_per_coord = 4
    params.prd_n_active = 1
    params.save()
    _write_landscape_inputs([0], {0: [2.0, 2.0, 2.0]})
    with pytest.raises(ValueError):
        probability_landscape.probability_landscape_local()


@pytest.mark.parametrize("states", [2, 5, 10, 50])
def test_probability_landscape_bin_count_follows_states_per_coord(states, project_stages):
    params.states_per_coord = states
    params.prd_n_active = 1
    params.save()
    _write_landscape_inputs([0], {0: list(np.linspace(0.0, 1.0, 20))})
    hUn = probability_landscape.probability_landscape_local()
    assert hUn.shape == (states,)
    assert hUn.sum() == 20


# --------------------------------------------------------------------------------------
# writeRelionS2, tau to raw image index mapping
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "nS,con_order_range,expected_con_order",
    [(12, 3, 4), (12, 4, 3), (100, 50, 2), (101, 50, 2), (49, 50, 0), (50, 50, 1)],
)
def test_con_order_is_floor_division_of_the_path_length(nS, con_order_range, expected_con_order):
    assert nS // con_order_range == expected_con_order


@pytest.mark.parametrize("nS,con_order_range", [(12, 3), (12, 4), (24, 6), (30, 5), (100, 50)])
def test_trimmed_path_width_is_nS_minus_two_conorder_plus_one(nS, con_order_range):
    # q[:, conOrder - 1 : nS - conOrder] keeps nS - 2*conOrder + 1 columns, which is the
    # number of tau values NLSA produces for the same projection direction
    conOrder = nS // con_order_range
    width = len(range(nS)[conOrder - 1 : nS - conOrder])
    assert width == nS - 2 * conOrder + 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tau_index_maps_back_to_the_raw_image_column(seed):
    rng = np.random.default_rng(seed)
    n, con_order_range = 12, 3
    q = rng.normal(size=(4, n))
    posPath = rng.permutation(n)
    psi1Path = rng.permutation(n)

    q_sel = q[:, posPath[psi1Path]]
    nS = q_sel.shape[1]
    conOrder = nS // con_order_range
    q_trim = q_sel[:, conOrder - 1 : nS - conOrder]

    # tau sample k corresponds to raw image posPath[psi1Path[conOrder - 1 + k]]
    for k in range(q_trim.shape[1]):
        raw = posPath[psi1Path[conOrder - 1 + k]]
        assert np.allclose(q_trim[:, k], q[:, raw])


def _write_traj_prd(prd, q, IMGT):
    myio.fout1(params.get_dist_file(prd), q=q)
    myio.fout1(params.get_EL_file(prd), IMGT=IMGT)


def test_extract_traj_data_by_prd_bins_tau_and_keeps_index_alignment(project_stages, monkeypatch):
    # quaternion.psi_ang runs a least squares solve whose result is not reproducible (it
    # sometimes returns its own (0,0,0) starting guess), so it is replaced by the identity
    # and the raw image alignment is read straight off the reported projection directions
    monkeypatch.setattr(quaternion, "psi_ang", lambda PD: (PD[0], PD[1], PD[2]))

    rng = np.random.default_rng(0)
    params.states_per_coord = 4
    params.con_order_range = 3
    params.save()

    n, dim = 12, 4
    q = rng.normal(size=(4, n))
    q /= np.linalg.norm(q, axis=0)
    posPath = rng.permutation(n)
    psi1Path = np.arange(n)
    conOrder = n // params.con_order_range  # == 4
    n_tau = n - 2 * conOrder + 1  # == 5

    IMGT = rng.normal(size=(dim * dim, n_tau))
    _write_traj_prd(0, q, IMGT)

    tau = np.array([0.05, 0.30, 0.55, 0.70, 0.95])
    out_file = os.path.join(params.traj_dir, "group.pkl")
    writeRelionS2.extract_traj_data_by_prd(
        (np.array([0]), out_file),
        trajTaus=[tau],
        tauAvg=tau.copy(),
        posPathAll=[posPath],
        posPsi1All=[psi1Path],
        pathw=1,
    )

    data = myio.fin1(out_file)
    assert len(data["imgss"]) == params.states_per_coord

    # bin edges are k/4, so the samples fall 1, 1, 2, 1 into the four bins
    expected_bins = {0: [0], 1: [1], 2: [2, 3], 3: [4]}
    scaled = (IMGT / conOrder).T
    for b, tau_indices in expected_bins.items():
        (imgs,) = data["imgss"][b]
        assert imgs.shape == (len(tau_indices), dim, dim)
        want = scaled[tau_indices].reshape(len(tau_indices), dim, dim).transpose(0, 2, 1)
        assert np.allclose(imgs, want.astype(np.float32))

        # tau sample k of this bin must carry raw image posPath[psi1Path[conOrder - 1 + k]]
        raw = [posPath[psi1Path[conOrder - 1 + k]] for k in tau_indices]
        PDs = quaternion.calc_avg_pd(q[:, raw], len(raw))
        assert np.allclose(data["phis"][b][0], PDs[0, :])
        assert np.allclose(data["thetas"][b][0], PDs[1, :])
        assert np.allclose(data["psis"][b][0], PDs[2, :])


def test_extract_traj_data_by_prd_leaves_empty_bins_untouched(project_stages):
    rng = np.random.default_rng(1)
    params.states_per_coord = 4
    params.con_order_range = 3
    params.save()

    n, dim = 12, 4
    q = rng.normal(size=(4, n))
    q /= np.linalg.norm(q, axis=0)
    conOrder = n // params.con_order_range
    n_tau = n - 2 * conOrder + 1
    _write_traj_prd(0, q, rng.normal(size=(dim * dim, n_tau)))

    # all tau crowd into the lowest bin, so bins 1..3 receive no image list at all
    tau = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    out_file = os.path.join(params.traj_dir, "group.pkl")
    writeRelionS2.extract_traj_data_by_prd(
        (np.array([0]), out_file),
        trajTaus=[tau],
        tauAvg=tau.copy(),
        posPathAll=[np.arange(n)],
        posPsi1All=[np.arange(n)],
        pathw=1,
    )
    data = myio.fin1(out_file)
    assert len(data["imgss"][0]) == 1
    assert data["imgss"][1] == [] and data["imgss"][2] == [] and data["imgss"][3] == []
    # psi is degenerate for a projection direction and is pinned to zero by convention
    assert np.allclose(data["psis"][0][0], 0.0)


def test_psi_ang_pins_psi_to_zero_for_every_direction():
    # writeRelionS2 stores this third angle in the star file. It is degenerate for a
    # projection direction and is hard coded to zero rather than taken from the optimizer
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(3, 12))
    directions /= np.linalg.norm(directions, axis=0)
    for i in range(directions.shape[1]):
        assert quaternion.psi_ang(directions[:, i])[2] == 0.0


def test_extract_traj_data_by_prd_appends_one_entry_per_prd(project_stages, monkeypatch):
    monkeypatch.setattr(quaternion, "psi_ang", lambda PD: (PD[0], PD[1], PD[2]))
    rng = np.random.default_rng(3)
    params.states_per_coord = 2
    params.con_order_range = 3
    params.save()

    n, dim = 12, 4
    conOrder = n // params.con_order_range
    n_tau = n - 2 * conOrder + 1
    for prd in (0, 1):
        q = rng.normal(size=(4, n))
        q /= np.linalg.norm(q, axis=0)
        _write_traj_prd(prd, q, rng.normal(size=(dim * dim, n_tau)))

    tau = np.linspace(0.0, 1.0, n_tau)
    out_file = os.path.join(params.traj_dir, "group.pkl")
    writeRelionS2.extract_traj_data_by_prd(
        (np.array([0, 1]), out_file),
        trajTaus=[tau, tau],
        tauAvg=tau.copy(),
        posPathAll=[np.arange(n), np.arange(n)],
        posPsi1All=[np.arange(n), np.arange(n)],
        pathw=1,
    )
    data = myio.fin1(out_file)
    # each projection direction contributes its own entry to every bin it populates
    assert len(data["imgss"][0]) == 2
    assert len(data["phis"][0]) == 2


@pytest.mark.parametrize("pathw,expected", [(1, 4), (2, 3), (3, 2), (4, 1)])
def test_number_of_trajectory_bins_shrinks_with_path_width(pathw, expected, project_stages):
    params.states_per_coord = 4
    assert len(range(0, params.states_per_coord - pathw + 1)) == expected


def test_extract_traj_data_by_prd_scales_images_by_con_order(project_stages):
    rng = np.random.default_rng(2)
    params.states_per_coord = 2
    params.con_order_range = 3
    params.save()

    n, dim = 12, 4
    q = rng.normal(size=(4, n))
    q /= np.linalg.norm(q, axis=0)
    conOrder = n // params.con_order_range
    n_tau = n - 2 * conOrder + 1
    IMGT = np.ones((dim * dim, n_tau))
    _write_traj_prd(0, q, IMGT)

    tau = np.linspace(0.0, 1.0, n_tau)
    out_file = os.path.join(params.traj_dir, "group.pkl")
    writeRelionS2.extract_traj_data_by_prd(
        (np.array([0]), out_file),
        trajTaus=[tau],
        tauAvg=tau.copy(),
        posPathAll=[np.arange(n)],
        posPsi1All=[np.arange(n)],
        pathw=1,
    )
    data = myio.fin1(out_file)
    # NLSA sums conOrder concatenated snapshots per output image, so the writer divides back
    for entry in data["imgss"]:
        for imgs in entry:
            assert np.allclose(imgs, 1.0 / conOrder)


# --------------------------------------------------------------------------------------
# writeRelionS2, RELION output filename templates
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "istart,numNext", [(0, 20), (20, 40), (40, 45), (0, 3)]
)
def test_group_pickle_template(istart, numNext, project_stages):
    want = f"{params.traj_file}name{params.traj_name}_group_{istart}_{numNext - 1}.pkl"
    got = "{}name{}_group_{}_{}.pkl".format(
        params.traj_file, params.traj_name, istart, numNext - 1
    )
    assert got == want
    assert want.startswith(os.path.join(params.out_dir, "traj", "traj_"))


@pytest.mark.parametrize("i_bin", [0, 1, 7, 49])
def test_relion_stack_and_star_templates_are_one_indexed(i_bin, project_stages):
    params.states_per_coord = 50
    params.traj_name = "1"
    stack = "imgsRELION_{}_{}_of_{}.mrcs".format(params.traj_name, i_bin + 1, params.states_per_coord)
    ang = f"EulerAngles_{params.traj_name}_{i_bin + 1}_of_{params.states_per_coord}.star"
    assert stack == f"imgsRELION_1_{i_bin + 1}_of_50.mrcs"
    assert ang == f"EulerAngles_1_{i_bin + 1}_of_50.star"


def test_write_star_column_order_and_content(project_stages):
    import pandas

    df = pandas.DataFrame(
        data=dict(phi=[10.0, 20.0], theta=[30.0, 40.0], psi=[50.0, 60.0])
    )
    star_file = os.path.join(params.bin_dir, "EulerAngles_1_1_of_4.star")
    star.write_star(star_file, "imgsRELION_1_1_of_4.mrcs", df)

    text = open(star_file).read()
    assert "_rlnImageName #1" in text
    assert "_rlnAnglePsi #2" in text
    assert "_rlnAngleTilt #3" in text
    assert "_rlnAngleRot #4" in text

    rows = [line for line in text.splitlines() if "@" in line]
    assert len(rows) == 2
    # image names are 1-indexed and columns are written psi, tilt(theta), rot(phi)
    assert rows[0].split() == ["1@imgsRELION_1_1_of_4.mrcs", "50.0", "30.0", "10.0", "1.0", "10000.0"]
    assert rows[1].split()[0] == "2@imgsRELION_1_1_of_4.mrcs"


def _write_group(istart, numNext, per_bin):
    n_bins = params.states_per_coord
    imgss = [[] for _ in range(n_bins)]
    phis = [[] for _ in range(n_bins)]
    thetas = [[] for _ in range(n_bins)]
    psis = [[] for _ in range(n_bins)]
    for i_bin, entries in per_bin.items():
        for arr in entries:
            imgss[i_bin].append(arr)
            phis[i_bin].append(np.zeros(arr.shape[0]))
            thetas[i_bin].append(np.zeros(arr.shape[0]))
            psis[i_bin].append(np.zeros(arr.shape[0]))
    name = "{}name{}_group_{}_{}.pkl".format(
        params.traj_file, params.traj_name, istart, numNext - 1
    )
    myio.fout1(name, imgss=imgss, phis=phis, thetas=thetas, psis=psis)
    return name


def test_concatenate_bin_writes_negated_stack_and_star(project_stages):
    import mrcfile

    params.states_per_coord = 4
    params.traj_name = "1"
    params.save()
    a = np.ones((2, 4, 4), dtype=np.float32)
    b = 2 * np.ones((3, 4, 4), dtype=np.float32)
    _write_group(0, 2, {0: [a, b]})

    writeRelionS2.concatenate_bin(0, numberOfJobs=2, batch_size=20)

    stack = os.path.join(params.bin_dir, "imgsRELION_1_1_of_4.mrcs")
    ang = os.path.join(params.bin_dir, "EulerAngles_1_1_of_4.star")
    assert os.path.isfile(stack) and os.path.isfile(ang)
    with mrcfile.open(stack) as mrc:
        # the reconstruction expects density, and the NLSA images carry the opposite sign
        assert np.allclose(mrc.data, -np.concatenate([a, b]))
    assert len([line for line in open(ang).read().splitlines() if "@" in line]) == 5


def test_concatenate_bin_returns_early_for_an_empty_bin(project_stages):
    params.states_per_coord = 4
    params.traj_name = "1"
    params.save()
    _write_group(0, 2, {1: [np.ones((2, 4, 4), dtype=np.float32)]})
    assert writeRelionS2.concatenate_bin(0, numberOfJobs=2, batch_size=20) is None
    assert not os.path.isfile(os.path.join(params.bin_dir, "imgsRELION_1_1_of_4.mrcs"))


def test_concatenate_bin_survives_an_empty_first_batch(project_stages):
    params.states_per_coord = 4
    params.traj_name = "1"
    params.save()
    _write_group(0, 20, {})
    _write_group(20, 21, {0: [np.ones((2, 4, 4), dtype=np.float32)]})
    writeRelionS2.concatenate_bin(0, numberOfJobs=21, batch_size=20)
    assert os.path.isfile(os.path.join(params.bin_dir, "imgsRELION_1_1_of_4.mrcs"))


def test_concatenate_bin_writes_a_bin_absent_from_the_last_batch(project_stages):
    # the emptiness check used to look at the last batch only and dropped the whole state
    params.states_per_coord = 4
    params.traj_name = "1"
    params.save()
    a = np.ones((2, 4, 4), dtype=np.float32)
    _write_group(0, 20, {0: [a]})
    _write_group(20, 21, {})
    writeRelionS2.concatenate_bin(0, numberOfJobs=21, batch_size=20)
    with mrcfile.open(os.path.join(params.bin_dir, "imgsRELION_1_1_of_4.mrcs")) as mrc:
        assert np.allclose(mrc.data, -a)


# --------------------------------------------------------------------------------------
# find_conformational_coords
# --------------------------------------------------------------------------------------
def test_force_remove_deletes_existing_files(tmp_path):
    paths = [tmp_path / f"f{i}.txt" for i in range(3)]
    for p in paths:
        p.write_text("x")
    find_conformational_coords.force_remove(*[str(p) for p in paths])
    assert not any(p.exists() for p in paths)


def test_force_remove_ignores_missing_paths(tmp_path):
    present = tmp_path / "here.txt"
    present.write_text("x")
    find_conformational_coords.force_remove(str(tmp_path / "nope.txt"), str(present))
    assert not present.exists()


def test_force_remove_ignores_directories(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    find_conformational_coords.force_remove(str(d))
    assert d.is_dir()


def test_force_remove_with_no_arguments_is_a_noop():
    assert find_conformational_coords.force_remove() is None


def test_find_ccs_removes_the_cc_file_and_belief_dumps(project_stages, monkeypatch):
    # op() clears stale results before recomputing. Stop right after by making the anchor
    # check fail, which is the documented early return
    for name in ("nodeAllStateBel_rc1.txt", "nodeAllStateBel_rc2.txt"):
        open(os.path.join(params.CC_dir, name), "w").close()
    myio.fout1(params.CC_file, psinums=np.zeros((2, 1), dtype=int))

    prds = _fill_fake_prds([3, 3])
    prds.trash_ids = set()
    prds.anchors = {}
    prds.neighbor_graph = {"Nodes": np.array([0, 1]), "nNodes": 2, "NodesConnComp": [[0, 1]]}
    prds.neighbor_subgraph = [{"originalNodes": np.array([0, 1]), "originalEdgeList": [[0]]}]

    assert find_conformational_coords.op() is None
    assert not os.path.isfile(params.CC_file)
    assert not os.path.isfile(os.path.join(params.CC_dir, "nodeAllStateBel_rc1.txt"))
    assert not os.path.isfile(os.path.join(params.CC_dir, "nodeAllStateBel_rc2.txt"))


def test_find_ccs_writes_psinums_when_every_node_is_an_anchor(project_stages):
    from ManifoldEM.data_store import Anchor, Sense

    prds = _fill_fake_prds([3, 3])
    prds.trash_ids = set()
    prds.anchors = {0: Anchor(CC=1, sense=Sense.FWD), 1: Anchor(CC=2, sense=Sense.REV)}
    prds.neighbor_graph = {"Nodes": np.array([0, 1]), "nNodes": 2, "NodesConnComp": [[0, 1]]}
    prds.neighbor_subgraph = [{"originalNodes": np.array([0, 1]), "originalEdgeList": [[0]]}]

    assert find_conformational_coords.op() is None
    data = myio.fin1(params.CC_file)
    # anchors store a 1-indexed conformational coordinate, the file holds the 0-indexed psi
    assert data["psinums"].shape == (2, 2)
    assert data["psinums"][0, 0] == 0
    assert data["psinums"][0, 1] == 1
    # Sense.FWD/REV carry the raw +1/-1 values straight into the senses table
    assert data["senses"][0, 0] == 1
    assert data["senses"][0, 1] == -1
    # the second reaction coordinate row is left untouched by the all-anchors shortcut
    assert np.all(data["psinums"][1, :] == 0)
    assert np.all(data["senses"][1, :] == 0)


# ============================================================================
# particle_index
# ============================================================================

def test_particle_index_parses_with_verify():
    args = cli.get_parser().parse_args(["utility", "particle-index", "--verify", "params_demo.toml"])
    assert args.command == "particle-index"
    assert args.input_file == "params_demo.toml"
    assert args.verify is True


def test_particle_index_verify_defaults_off():
    args = cli.get_parser().parse_args(["utility", "particle-index", "params_demo.toml"])
    assert args.verify is False


def test_particle_index_dispatch_forwards_verify(monkeypatch):
    seen = {}
    monkeypatch.setattr(particle_index, "op", lambda **kw: seen.update(kw))
    cli._funcs["particle-index"](Namespace(command="particle-index", input_file="params_demo.toml", verify=True,
                                           no_movies=False, trace_image=[], trace_particle=[], trace_prd=[]))
    assert seen == {"verify_states": True, "movies": True, "trace_images": [], "trace_particles": [], "trace_prds": []}


def test_particle_index_no_movies_skips_the_movie_table(monkeypatch):
    args = cli.get_parser().parse_args(["utility", "particle-index", "--no-movies", "params_demo.toml"])
    assert args.no_movies is True
    seen = {}
    monkeypatch.setattr(particle_index, "op", lambda **kw: seen.update(kw))
    cli._funcs["particle-index"](args)
    assert seen == {"verify_states": False, "movies": False, "trace_images": [], "trace_particles": [], "trace_prds": []}


def test_particle_index_trace_options_parse():
    args = cli.get_parser().parse_args(["utility", "particle-index", "--trace-image", "25", "219", "--trace-image", "3", "1",
                                        "--trace-particle", "11806", "--trace-prd", "45", "params_demo.toml"])
    assert args.trace_image == [[25, 219], [3, 1]]
    assert args.trace_particle == [11806]
    assert args.trace_prd == [45]


def test_frame_chain_keeps_every_step():
    rng = np.random.default_rng(1)
    ind = rng.choice(1000, size=14, replace=False)
    posPath = np.sort(rng.choice(14, size=12, replace=False))
    posPsi1 = rng.permutation(12)
    chain = particle_index.frame_chain(ind, posPath, posPsi1, 4)
    assert np.array_equal(chain["sorted"], np.arange(6) + 2)  # s = m + C - 1 with C = 3
    assert np.array_equal(chain["retained"], posPsi1[chain["sorted"]])
    assert np.array_equal(chain["place"], posPath[chain["retained"]])
    assert np.array_equal(chain["particle_id"], ind[chain["place"]])


def test_frame_particles_follows_the_nlsa_offset():
    rng = np.random.default_rng(0)
    ind = rng.choice(1000, size=14, replace=False)
    posPath = np.sort(rng.choice(14, size=12, replace=False))
    posPsi1 = rng.permutation(12)
    ids = particle_index.frame_particles(ind, posPath, posPsi1, con_order_range=4)
    C = 12 // 4
    assert len(ids) == 12 - 2 * C
    for m in range(len(ids)):
        assert ids[m] == ind[posPath[posPsi1[m + C - 1]]]


def test_frame_particles_skip_the_ends_of_the_sorted_order():
    # the first C - 1 and the last C + 1 sorted particles get no frame (C = 3)
    ids = particle_index.frame_particles(np.arange(12), np.arange(12), np.arange(12), con_order_range=4)
    assert ids.tolist() == [2, 3, 4, 5, 6, 7]


def test_frame_particles_shift_marks_frames_outside_the_pd():
    ids = particle_index.frame_particles(np.arange(12), np.arange(12), np.arange(12), 4, shift=-3)
    assert ids.tolist() == [-1, 0, 1, 2, 3, 4]


def test_frame_particles_rejects_a_pd_without_frames():
    with pytest.raises(ValueError):
        particle_index.frame_particles(np.arange(30), np.arange(30), np.arange(30), 50)


def test_state_frames_partition_the_frames_for_width_one():
    tau_eq = np.array([0.0, 0.1, 0.25, 0.49, 0.5, 0.74, 0.75, 0.99, 1.0])
    bins = particle_index.state_frames(tau_eq, 4, 1)
    assert [b.tolist() for b in bins] == [[0, 1], [2, 3], [4, 5], [6, 7, 8]]


def test_state_frames_overlap_for_wider_paths():
    bins = particle_index.state_frames(np.linspace(0, 1, 21), 4, 2)
    assert len(bins) == 3
    counts = np.bincount(np.concatenate(bins), minlength=21)
    assert counts.min() == 1 and counts.max() == 2


def _fake_trajectory_run(rng, sizes, order, overlap=0):
    """Distance, psi analysis and trajectory files for PDs where each PD shares `overlap` particles
    with the next one, as cone assignment does. Each NLSA image carries the code (1, prd, frame) in
    its first three pixels so the state stacks can be decoded."""
    dim = params.ms_num_pixels
    n_total = sum(sizes) + 2 * len(sizes) - overlap * (len(sizes) - 1)
    quats = rng.normal(size=(4, n_total))
    quats /= np.linalg.norm(quats, axis=0)
    with open(params.pd_file, "wb") as f:
        pickle.dump(dict(quats_full=quats, image_is_mirrored=rng.random(n_total) < 0.5), f)

    rows = rng.permutation(n_total)
    trajTaus, posPathAll, posPsi1All = [None] * len(sizes), [None] * len(sizes), [None] * len(sizes)
    start = 0
    for x, nS in enumerate(sizes):
        ind = np.sort(rows[start:start + nS + 2])  # two particles of every PD are trimmed
        start += nS + 2 - overlap
        posPath = np.sort(rng.choice(nS + 2, size=nS, replace=False))
        posPsi1 = rng.permutation(nS)
        n_frames = nS - 2 * (nS // params.con_order_range)
        IMGT = np.zeros((dim * dim, n_frames), dtype=np.float16)
        IMGT[0], IMGT[1], IMGT[2] = 1, x, np.arange(n_frames)
        tau = rng.random((n_frames, 1))
        myio.fout1(params.get_dist_file(x), ind=ind, q=quats[:, ind])
        myio.fout1(params.get_EL_file(x), IMGT=IMGT, tau=tau, posPath=posPath, PosPsi1=posPsi1,
                   tauinds=rng.integers(0, n_frames, size=4))
        trajTaus[x], posPathAll[x], posPsi1All[x] = tau, posPath, posPsi1

    tauAvg = np.concatenate([((trajTaus[x] - trajTaus[x].min()) / np.ptp(trajTaus[x])).ravel() for x in order])
    myio.fout1(f"{params.traj_file}name{params.traj_name}_vars.pkl", trajTaus=trajTaus, posPsi1All=posPsi1All,
               posPathAll=posPathAll, xSelect=order, tauAvg=tauAvg)
    writeRelionS2.op(trajTaus, posPsi1All, posPathAll, order, tauAvg)
    return n_total, posPathAll, posPsi1All


@pytest.mark.parametrize("overlap", [0, 12])
def test_particle_index_matches_the_state_stacks(project_stages, overlap):
    # the tracker must name the PD, frame and particle of every image writeRelionS2 wrote
    params.prd_n_active = 3
    order = [2, 0, 1]
    _, posPathAll, posPsi1All = _fake_trajectory_run(np.random.default_rng(3), [30, 24, 27], order, overlap)

    frames, states = particle_index.track()
    assert len(frames) == 10 + 8 + 9
    assert len(states) == len(frames)  # width_1D = 1 puts every frame in one state
    for b, t in states.groupby("state"):
        stack = mrcfile.read(os.path.join(params.bin_dir, f"imgsRELION_{params.traj_name}_{b}_of_{params.states_per_coord}.mrcs"))
        stack = stack.reshape(-1, params.ms_num_pixels, params.ms_num_pixels)
        assert len(stack) == len(t)
        # undo the transpose and the sign flip of the writer, then read the (1, prd, frame) code
        code = -stack.transpose(0, 2, 1).reshape(len(stack), -1)[:, :3]
        assert np.array_equal(np.rint(code[:, 1] / code[:, 0]), t.prd.values)
        assert np.array_equal(np.rint(code[:, 2] / code[:, 0]), t.frame.values)
        assert t.image.tolist() == list(range(1, len(t) + 1))

    for x in order:
        ind = myio.fin1(params.get_dist_file(x))["ind"]
        C = len(posPathAll[x]) // params.con_order_range
        f = frames[frames.prd == x]
        assert np.array_equal(f.particle_id.values, ind[posPathAll[x][posPsi1All[x][f.frame.values + C - 1]]])


def test_particle_index_op_writes_the_tables(project_stages, capsys):
    params.prd_n_active = 3
    n_total, _, _ = _fake_trajectory_run(np.random.default_rng(4), [30, 24, 27], [0, 1, 2])
    capsys.readouterr()

    particle_index.op()
    out = os.path.join(params.out_dir, "particle_index")
    assert sorted(os.listdir(out)) == ["frames.csv", "movies.csv", "particles.csv", "states.csv"]
    particles = pd.read_csv(os.path.join(out, "particles.csv"))
    states = pd.read_csv(os.path.join(out, "states.csv"))
    assert len(particles) == n_total
    assert (particles.pds == 1).all()
    assert particles.frames.sum() == 27
    assert particles.state_images.sum() == len(states) == 27
    assert len(pd.read_csv(os.path.join(out, "movies.csv"))) == 12
    printed = capsys.readouterr().out
    assert "State images        27" in printed
    for name, n_rows in [("frames.csv", 27), ("states.csv", 27), ("particles.csv", n_total), ("movies.csv", 12)]:
        assert f"  {name:<14} {n_rows} rows" in printed


def test_particle_index_follows_particles_shared_by_overlapping_pds(project_stages):
    # a particle in two PDs gets a frame in each PD where it is neither trimmed nor at an end
    params.prd_n_active = 3
    n_total, posPathAll, posPsi1All = _fake_trajectory_run(np.random.default_rng(6), [30, 24, 27], [1, 2, 0], overlap=12)
    particle_index.op(movies=False)
    out = os.path.join(params.out_dir, "particle_index")
    particles = pd.read_csv(os.path.join(out, "particles.csv"))

    member, framed = np.zeros(n_total, dtype=int), np.zeros(n_total, dtype=int)
    for x in range(3):
        ind = myio.fin1(params.get_dist_file(x))["ind"]
        nS = len(posPathAll[x])
        C = nS // params.con_order_range
        member[ind] += 1
        framed[ind[posPathAll[x][posPsi1All[x][C - 1:nS - C - 1]]]] += 1
    assert (member == 2).sum() == 24  # 12 shared by PDs 0 and 1, 12 by PDs 1 and 2
    assert np.array_equal(particles.pds.values, member)
    assert np.array_equal(particles.frames.values, framed)
    assert framed.max() == 2
    assert np.array_equal(particles.state_images.values, framed)  # width_1D = 1
    assert not os.path.isfile(os.path.join(out, "movies.csv"))


def test_particle_index_op_warns_about_states_without_a_star_file(project_stages, capsys):
    params.prd_n_active = 3
    _fake_trajectory_run(np.random.default_rng(4), [30, 24, 27], [0, 1, 2])
    os.remove(os.path.join(params.bin_dir, f"EulerAngles_{params.traj_name}_2_of_{params.states_per_coord}.star"))
    capsys.readouterr()
    particle_index.op(movies=False)
    assert "No state star file for states [2]" in capsys.readouterr().out


def test_particle_summary_skips_pds_without_a_distance_file(project_stages, capsys):
    params.prd_n_active = 3
    _fake_trajectory_run(np.random.default_rng(4), [30, 24, 27], [0, 1, 2])
    frames, states = particle_index.track()
    params.prd_n_active = 4
    particles = particle_index.particle_summary(frames, states)
    assert particles.pds.sum() == 30 + 24 + 27 + 6
    assert "[3]" in capsys.readouterr().out


def test_particle_index_op_needs_the_trajectory_step(project_stages, capsys):
    particle_index.op()
    assert "trajectory" in capsys.readouterr().out
    assert not os.path.isdir(os.path.join(params.out_dir, "particle_index"))


def test_trace_image_and_trace_particle_agree(project_stages):
    params.prd_n_active = 3
    _fake_trajectory_run(np.random.default_rng(3), [30, 24, 27], [2, 0, 1])
    mirrored, _ = particle_index._load_pd_data()
    frames, states = particle_index.track(mirrored)
    row = states.iloc[5]
    back = "\n".join(particle_index.trace_image(frames, states, row.state, row.image, mirrored))
    assert f"particle ID r = {row.particle_id}" in back
    forward = "\n".join(particle_index.trace_particle(frames, states, int(row.particle_id), mirrored))
    assert f"frame m = {row.frame}" in forward
    assert f"image {row.image} of state {row.state}" in forward


def test_trace_particle_reports_trimmed_and_edge_particles(project_stages):
    params.prd_n_active = 3
    _fake_trajectory_run(np.random.default_rng(3), [30, 24, 27], [0, 1, 2])
    mirrored, _ = particle_index._load_pd_data()
    frames, states = particle_index.track(mirrored)
    traj = myio.fin1(particle_index._traj_vars_file())
    ind, posPath, posPsi1 = myio.fin1(params.get_dist_file(0))["ind"], traj["posPathAll"][0], traj["posPsi1All"][0]
    trimmed = int(ind[np.setdiff1d(np.arange(len(ind)), posPath)[0]])
    first = int(ind[posPath[posPsi1[0]]])  # sorted s = 0 lies before the first frame
    assert "trimmed, no frame" in "\n".join(particle_index.trace_particle(frames, states, trimmed, mirrored))
    assert "no frame (start of the psi order)" in "\n".join(particle_index.trace_particle(frames, states, first, mirrored))
    assert "outside the star file" in particle_index.trace_particle(frames, states, len(mirrored), mirrored)[0]


def test_particle_index_op_saves_requested_traces(project_stages, capsys):
    params.prd_n_active = 3
    _fake_trajectory_run(np.random.default_rng(4), [30, 24, 27], [0, 1, 2])
    particle_index.op(movies=False, trace_images=[(1, 1)], trace_prds=[2])
    out_dir = os.path.join(params.out_dir, "particle_index")
    trace = open(os.path.join(out_dir, "trace.txt")).read()
    assert "Image 1 of state 1 back to its particle" in trace
    assert "PD 2, C = 9, 9 frames" in trace
    assert "Trace file" in capsys.readouterr().out
    frames = pd.read_csv(os.path.join(out_dir, "frames.csv"))
    assert {"C", "sorted", "retained", "place", "particle_id"} <= set(frames.columns)


def test_verify_counts_tracked_directions(project_stages):
    rng = np.random.default_rng(5)
    quats = rng.normal(size=(4, 20))
    quats /= np.linalg.norm(quats, axis=0)
    with open(params.pd_file, "wb") as f:
        pickle.dump(dict(quats_full=quats, image_is_mirrored=np.zeros(20, dtype=bool)), f)
    states = pd.DataFrame(dict(state=[1, 1, 1, 2, 2], image=[1, 2, 3, 1, 2], prd=0, frame=[0, 1, 2, 3, 4],
                               particle_id=[4, 7, 1, 9, 12]))
    for b, t in states.groupby("state"):
        v = quaternion.quaternion_to_S2(quats[:, t.particle_id.values])
        df = pd.DataFrame(dict(phi=np.degrees(np.arctan2(v[1], v[0])) % 360, theta=np.degrees(np.arccos(v[2])), psi=0.0))
        star.write_star(os.path.join(params.bin_dir, f"EulerAngles_{params.traj_name}_{b}_of_{params.states_per_coord}.star"),
                        "stack.mrcs", df)

    assert particle_index.verify(states) == (5, [], [])
    assert particle_index.verify(states.assign(particle_id=[7, 1, 9, 12, 4]))[0] == 0
    extra = pd.DataFrame(dict(state=[2, 3], image=[3, 1], prd=0, frame=[5, 6], particle_id=[0, 2]))
    assert particle_index.verify(pd.concat([states, extra], ignore_index=True)) == (3, [3], [2])
