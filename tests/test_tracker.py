"""Tests for ManifoldEM.tracker on a small fake run written with the real writeRelionS2."""

import os
import pickle
import sys

import h5py
import mrcfile
import numpy as np
import pandas as pd
import pytest

from ManifoldEM import myio, tracker, writeRelionS2
from ManifoldEM.interfaces import cli
from ManifoldEM.params import params

SIZES = [30, 24, 27]  # retained particles per PD, C is 10, 8 and 9 with con_order_range 3


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway project folder, with the params singleton restored afterwards."""
    snapshot = dict(vars(params))
    monkeypatch.chdir(tmp_path)
    for key, value in dict(project_name="test", particle_diameter=100.0, ms_estimated_resolution=5.0, ncpu=1,
                           prd_n_active=3, states_per_coord=4, con_order_range=3, width_1D=1, traj_name="1",
                           num_psi=2, ms_num_pixels=8, ms_pixel_size=1.0).items():
        setattr(params, key, value)
    params.create_dir()
    params.save()
    yield tmp_path
    params.__dict__.update(snapshot)


def fake_run(rng, order=(2, 0, 1), overlap=0, extra=0):
    """Every file a finished run leaves, for PDs where each PD shares `overlap` particles with the next.
    Each NLSA image carries the code (1, prd, frame) in its first three pixels. With `extra` > 0 the
    first extra particle sits in PD 0 beyond prd_thres_high and the others in no PD."""
    dim, order = params.ms_num_pixels, list(order)
    n_total = sum(SIZES) + 2 * len(SIZES) - overlap * (len(SIZES) - 1)
    quats = rng.normal(size=(4, n_total + extra))
    quats /= np.linalg.norm(quats, axis=0)
    members = np.empty(len(SIZES), dtype=object)
    rows, start = rng.permutation(n_total), 0
    trajTaus, posPathAll, posPsi1All = [None] * 3, [None] * 3, [None] * 3
    for x, n_s in enumerate(SIZES):
        ind = np.sort(rows[start:start + n_s + 2])  # two particles of every PD are trimmed
        start += n_s + 2 - overlap
        members[x] = np.concatenate([ind, [n_total]]) if extra and x == 0 else ind
        pos_path = np.sort(rng.choice(n_s + 2, size=n_s, replace=False))
        psi = rng.normal(size=(n_s, params.num_psi))
        pos_psi1 = np.argsort(psi[:, 0])
        n_frames = n_s - 2 * (n_s // params.con_order_range)
        tau = rng.random((n_frames, 1))
        IMGT = np.zeros((dim * dim, n_frames), dtype=np.float16)
        IMGT[0], IMGT[1], IMGT[2] = 1, x, np.arange(n_frames)
        myio.fout1(params.get_dist_file(x), ind=ind, q=quats[:, ind])
        myio.fout1(params.get_psi_file(x), ind=ind, posPath=pos_path, psi=psi)
        for k in range(params.num_psi):
            myio.fout1(params.get_psi2_file(x, k), tau=rng.random((n_frames, 1)))
        myio.fout1(params.get_EL_file(x), IMGT=IMGT, tau=tau, posPath=pos_path, PosPsi1=pos_psi1,
                   tauinds=rng.integers(0, n_frames, size=4))
        trajTaus[x], posPathAll[x], posPsi1All[x] = tau, pos_path, pos_psi1

    with open(params.pd_file, "wb") as f:
        pickle.dump(dict(quats_full=quats, image_is_mirrored=rng.random(n_total + extra) < 0.5,
                         image_indices_full=members, thres_ids=[0, 1, 2], thres_low=10, thres_high=max(SIZES) + 2,
                         occupancy_full=np.array([len(m) for m in members]), trash_ids=set()), f)
    psinums = np.full((2, 3), -1)
    psinums[0, order] = 0
    myio.fout1(params.CC_file, psinums=psinums, senses=np.ones_like(psinums))
    tauAvg = np.concatenate([((trajTaus[x] - trajTaus[x].min()) / np.ptp(trajTaus[x])).ravel() for x in order])
    myio.fout1(f"{params.traj_file}name{params.traj_name}_vars.pkl", trajTaus=trajTaus, posPsi1All=posPsi1All,
               posPathAll=posPathAll, xSelect=order, tauAvg=tauAvg)
    writeRelionS2.op(trajTaus, posPsi1All, posPathAll, order, tauAvg)
    return n_total + extra, posPathAll, posPsi1All


def record_all():
    return [tracker.record(step, quiet=True) for step in tracker.STEPS]


def failed_checks(step):
    return {c[0]: c[2] for c in tracker.group_info(step.replace("-", "_"))["checks"]}


def test_every_step_is_recorded_and_passes_its_checks(project):
    fake_run(np.random.default_rng(3))
    assert all(record_all())
    assert [s[1] for s in tracker.status()] == ["recorded"] * 7
    for step in tracker.STEPS:
        # States is left out, its angles come from q2Spider, which stalls at random
        assert all(n == 0 for name, n in failed_checks(step).items() if name != "States"), step


def test_mapper_names_every_fate(project):
    n_all, _, _ = fake_run(np.random.default_rng(3), order=(0, 1, 2), extra=3)
    record_all()
    mapper = tracker.read_group("mapper")
    assert list(mapper.columns) == tracker.MAPPER
    assert mapper.particle_id.is_unique and len(mapper) == n_all
    # each PD loses C - 1 sorted particles at the start and C + 1 at the end
    assert mapper.fate.value_counts().to_dict() == {
        "In a state": 27, "Trimmed": 6, "No frame (start of psi order)": 9 + 7 + 8,
        "No frame (end of psi order)": 11 + 9 + 10, "Over prd_thres_high": 1, "In no PD": 2}


def test_mapper_chain_matches_the_state_stacks(project):
    _, posPathAll, posPsi1All = fake_run(np.random.default_rng(3))
    record_all()
    mapper = tracker.read_group("mapper")
    ind = {x: myio.fin1(params.get_dist_file(x))["ind"] for x in range(3)}
    in_state = mapper[mapper.state > 0]
    for b, t in in_state.groupby("state"):
        stack = mrcfile.read(os.path.join(params.bin_dir, f"imgsRELION_1_{b}_of_4.mrcs")).reshape(-1, 8, 8)
        t = t.sort_values("image")
        code = -stack[t.image.values - 1].transpose(0, 2, 1).reshape(len(t), -1)[:, :3]
        assert np.array_equal(np.rint(code[:, 1] / code[:, 0]), t.prd.values)
        assert np.array_equal(np.rint(code[:, 2] / code[:, 0]), t.frame.values)
    for r in in_state.itertuples(index=False):
        s = r.frame + r.C - 1
        assert r.sorted == s and posPsi1All[r.prd][s] == r.retained
        assert posPathAll[r.prd][r.retained] == r.place and ind[r.prd][r.place] == r.particle_id


def test_left_out_pd_is_marked_not_selected(project):
    fake_run(np.random.default_rng(3), order=(0, 1))
    record_all()
    mapper = tracker.read_group("mapper")
    assert mapper[mapper.prd == 2].fate.value_counts().to_dict() == {"PD not selected": 27, "Trimmed": 2}


def test_shared_particles_get_one_row_per_pd(project):
    n_total, _, _ = fake_run(np.random.default_rng(6), overlap=12)
    record_all()
    mapper = tracker.read_group("mapper")
    rows = mapper.groupby("particle_id").size()
    assert (rows == 2).sum() == 24  # 12 shared by PDs 0 and 1, 12 by PDs 1 and 2
    assert failed_checks("calc-distance")["Binning"] == 24  # a hard run must not share particles


def test_rerun_marks_later_steps_stale(project):
    fake_run(np.random.default_rng(3))
    record_all()
    tracker.record("calc-distance", quiet=True)
    states = dict((s[0], s[1]) for s in tracker.status())
    assert states["calc-distance"] == "recorded"
    assert all(states[s] == "stale" for s in tracker.STEPS[1:])


def test_fill_records_blank_steps(project, capsys):
    fake_run(np.random.default_rng(3))
    tracker.record("calc-distance", quiet=True)
    tracker.op(fill=True)
    assert [s[1] for s in tracker.status()] == ["recorded"] * 7
    assert "Filling trajectory" in capsys.readouterr().out


def test_verify_catches_a_changed_file(project, capsys):
    fake_run(np.random.default_rng(3))
    record_all()
    path = params.get_psi_file(0)
    with h5py.File(path, "r") as f:
        data = {k: f[k][()] for k in f}
    data["posPath"] = data["posPath"][1:]
    myio.fout1(path, **data)
    capsys.readouterr()
    tracker.op(verify=True)
    out = capsys.readouterr().out
    assert "differs from the files now" in out


def test_record_never_stops_the_pipeline(project, capsys):
    assert tracker.record("trajectory") is False
    assert "could not record trajectory" in capsys.readouterr().out


def test_trace_and_csv(project, capsys):
    fake_run(np.random.default_rng(3))
    record_all()
    mapper = tracker.read_group("mapper")
    pid = int(mapper[mapper.state == 1].particle_id.iloc[0])
    capsys.readouterr()
    tracker.op(trace_images=[(1, 1)], trace_particles=[pid], trace_prds=[0], csv=True)
    out = capsys.readouterr().out
    assert "Image 1 of state 1 back to its particle" in out and f"Particle ID {pid}" in out and "PD 0," in out
    table = pd.read_csv(os.path.join(params.out_dir, "particle_index.csv"))
    assert list(table.columns) == tracker.MAPPER and len(table) == len(mapper)


@pytest.mark.parametrize("step", tracker.STEPS)
def test_tracker_is_on_by_default_for_every_step(step):
    assert cli.get_parser().parse_args([step, "params_x.toml"]).tracker is True
    assert cli.get_parser().parse_args([step, "--no-tracker", "params_x.toml"]).tracker is False


def test_tracker_utility_parses():
    args = cli.get_parser().parse_args(["utility", "tracker", "--fill", "--verify", "--trace-image", "25", "219",
                                        "--trace-particle", "7", "--trace-prd", "3", "--csv", "params_x.toml"])
    assert (args.fill, args.verify, args.trace_image, args.trace_particle, args.trace_prd, args.csv) == \
        (True, True, [[25, 219]], [7], [3], True)


@pytest.mark.parametrize("argv,calls", [(["calc-distance", "params_x.toml"], ["calc-distance"]),
                                        (["calc-distance", "--no-tracker", "params_x.toml"], [])])
def test_main_records_after_the_step(monkeypatch, argv, calls):
    seen = []
    monkeypatch.setattr(sys, "argv", ["manifold-cli"] + argv)
    monkeypatch.setattr(cli, "load_state", lambda args: None)
    monkeypatch.setattr(cli, "set_params", lambda args: None)
    monkeypatch.setitem(cli._funcs, "calc-distance", lambda args: None)
    monkeypatch.setattr(tracker, "record", lambda step: seen.append(step))
    cli.main()
    assert seen == calls
