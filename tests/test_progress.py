"""Tests for ManifoldEM.progress on the fake run of test_tracker."""

import json
import os
import sys

import numpy as np

from ManifoldEM import progress, tracker
from ManifoldEM.interfaces import cli
from ManifoldEM.params import params

sys.path.insert(0, os.path.dirname(__file__))
from test_tracker import SIZES, fake_run, project  # noqa: E402, F401  (project is a fixture)


def by_step(snap):
    return {s["step"]: s for s in snap["steps"]}


def test_finished_run_counts_every_unit_and_particle(project):
    fake_run(np.random.default_rng(3))
    snap = progress.snapshot()
    s = by_step(snap)
    assert snap["n_pd"] == 3 and snap["n_states"] == 4
    assert s["calc-distance"]["done"] == [0, 1, 2] and s["calc-distance"]["expected"] == 3
    assert s["manifold-analysis"]["particles"] == f"{sum(SIZES):,} retained"
    framed = sum(n - 2 * (n // params.con_order_range) for n in SIZES)
    assert s["probability-landscape"]["particles"] == f"{framed:,} with an NLSA frame"
    assert s["psi-analysis"]["done"] == [0, 1, 2]          # every topo file present
    assert s["nlsa-movie"]["done"] == []                   # the fake run writes no movies
    assert s["find-ccs"]["complete"] is True
    stars = len([n for n in os.listdir(params.bin_dir) if n.endswith(".star")])
    assert len(s["trajectory"]["done"]) == stars
    assert s["trajectory"]["particles"] == f"{framed:,} in a state"


def test_partial_run_counts_only_the_files_present(project):
    fake_run(np.random.default_rng(3))
    os.remove(params.get_EL_file(1))
    star = [n for n in os.listdir(params.bin_dir) if n.endswith(".star")][0]
    os.remove(os.path.join(params.bin_dir, star))
    s = by_step(progress.snapshot())
    assert s["probability-landscape"]["done"] == [0, 2]
    framed = sum(n - 2 * (n // params.con_order_range) for p, n in enumerate(SIZES) if p != 1)
    assert s["probability-landscape"]["particles"] == f"{framed:,} with an NLSA frame"
    assert not progress.finished(progress.snapshot())


def test_empty_project_reads_as_not_started(project):
    s = by_step(progress.snapshot())
    assert s["threshold"]["done"] == [] and s["calc-distance"]["done"] == []
    assert s["calc-distance"]["tracker"]["state"] == "blank"


def test_tracker_state_and_failed_checks_are_shown(project):
    fake_run(np.random.default_rng(3))
    tracker.record("calc-distance", quiet=True)
    s = by_step(progress.snapshot())
    assert s["calc-distance"]["tracker"] == dict(state="recorded", failed=0)
    assert s["manifold-analysis"]["tracker"]["state"] == "blank"
    text = progress.render(progress.snapshot())
    assert "recorded  0 failed" in text


def test_render_has_a_table_and_a_grid(project):
    fake_run(np.random.default_rng(3))
    text = progress.render(progress.snapshot())
    assert "calc-distance          3 of 3 PDs" in text
    assert "find-ccs               Done" in text
    lines = {l.split()[0]: l for l in text.splitlines() if l.strip()}
    assert lines["calc-distance"].rstrip().endswith("###")
    assert "###" not in progress.render(progress.snapshot(), grid=False)


def test_many_units_draw_a_percent_bar():
    row = progress._grid_row(list(range(250)), 1000)
    assert row.endswith(" 25%") and row.count("#") == 12


def test_json_file_and_history(project, capsys):
    fake_run(np.random.default_rng(3))
    progress.op(json_out=True)
    progress.op(json_out=True)
    with open(os.path.join(params.out_dir, "progress.json")) as f:
        latest = json.load(f)
    assert latest["project"] == "test" and len(latest["steps"]) == len(progress.STEPS)
    with open(os.path.join(params.out_dir, "progress.jsonl")) as f:
        assert len(f.readlines()) == 2
    assert "Progress file" in capsys.readouterr().out


def test_watch_stops_when_the_run_is_finished(project, capsys):
    fake_run(np.random.default_rng(3))
    progress.op(watch=1)   # the run is finished, so one pass and return
    assert "trajectory" in capsys.readouterr().out


def test_progress_utility_parses():
    args = cli.get_parser().parse_args(["utility", "progress", "params_x.toml", "--watch", "30", "--json", "--no-grid"])
    assert (args.command, args.watch, args.json, args.grid) == ("progress", 30, True, False)
    assert cli.get_parser().parse_args(["utility", "progress", "params_x.toml"]).watch == 0
