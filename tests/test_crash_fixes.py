"""Tests for the crash fixes in writeRelionS2, util and CC/OpticalFlowMovie."""

import os

import mrcfile
import numpy as np
import pytest

from ManifoldEM import myio, util, writeRelionS2
from ManifoldEM.CC import OpticalFlowMovie
from ManifoldEM.params import params


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway project folder, with the params singleton restored afterwards."""
    snapshot = dict(vars(params))
    monkeypatch.chdir(tmp_path)
    params.project_name = "test"
    params.particle_diameter = 100.0
    params.ms_estimated_resolution = 5.0
    params.states_per_coord = 4
    params.traj_name = "1"
    params.create_dir()
    params.save()
    yield tmp_path
    params.__dict__.update(snapshot)


def _write_group(istart, n_next, per_bin):
    """One batch file of extract_traj_data_by_prd, with the given image arrays per state."""
    lists = {key: [[] for _ in range(params.states_per_coord)] for key in ("imgss", "phis", "thetas", "psis")}
    for b, arrays in per_bin.items():
        for a in arrays:
            lists["imgss"][b].append(a)
            for key in ("phis", "thetas", "psis"):
                lists[key][b].append(np.zeros(len(a)))
    myio.fout1(f"{params.traj_file}name{params.traj_name}_group_{istart}_{n_next - 1}.pkl", **lists)


def _stack(state):
    return os.path.join(params.bin_dir, f"imgsRELION_{params.traj_name}_{state}_of_{params.states_per_coord}.mrcs")


def test_concatenate_bin_survives_an_empty_first_batch(project):
    # the accumulator used to start only at the first batch, so an empty first batch crashed
    a = np.ones((2, 4, 4), dtype=np.float32)
    _write_group(0, 20, {})
    _write_group(20, 21, {0: [a]})
    writeRelionS2.concatenate_bin(0, numberOfJobs=21, batch_size=20)
    with mrcfile.open(_stack(1)) as mrc:
        assert np.allclose(mrc.data, -a)


def test_concatenate_bin_writes_a_state_absent_from_the_last_batch(project):
    # the emptiness check used to look at the last batch only and dropped the whole state
    a = np.ones((2, 4, 4), dtype=np.float32)
    _write_group(0, 20, {0: [a]})
    _write_group(20, 21, {})
    writeRelionS2.concatenate_bin(0, numberOfJobs=21, batch_size=20)
    with mrcfile.open(_stack(1)) as mrc:
        assert np.allclose(mrc.data, -a)


def test_concatenate_bin_skips_a_state_with_no_images(project):
    _write_group(0, 2, {1: [np.ones((2, 4, 4), dtype=np.float32)]})
    assert writeRelionS2.concatenate_bin(0, numberOfJobs=2, batch_size=20) is None
    assert not os.path.isfile(_stack(1))


def test_util_defines_the_logger_it_uses():
    assert util._logger.name == "ManifoldEM.util"


def test_optical_flow_movie_imports_warnings():
    assert OpticalFlowMovie.warnings.warn is not None
