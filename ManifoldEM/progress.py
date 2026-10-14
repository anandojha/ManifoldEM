"""Progress of a run from the files each step writes, live while the run is going.

Every pipeline step writes one file per PD (or per state) as it finishes that unit, so the files
present in output/<project_name> say how far a running job got without touching the pipeline.
The particle counts come from the same small reads the tracker uses.

    threshold              pd_data.pkl                       particles in a PD
    calc-distance          distances/IMGs_prD_N.h5           per PD
    manifold-analysis      diff_maps/gC_trimmed_psi_prD_N.h5 per PD, retained particles
    psi-analysis           psi_analysis/S2_prD_N_psi_K.h5    per PD, all num_psi topos
    nlsa-movie             topos/PrD_N/psi_K.gif             per PD, all num_psi movies
    find-ccs               CC/CC_OF/OF_prD_N.pkl             per PD, then CC/CC_file.pkl
    probability-landscape  ELConc*/S2_prD_N.pkl              per PD, particles with a frame
    trajectory             bin/EulerAngles_*_B_of_*.star     per state, particles in a state
"""

import json
import os
import re
import time

import h5py
import numpy as np

from ManifoldEM import tracker
from ManifoldEM.params import params

STEPS = ["threshold", "calc-distance", "manifold-analysis", "psi-analysis", "nlsa-movie", "find-ccs",
         "probability-landscape", "trajectory"]
BAR_LIMIT = 100  # steps with more units than this draw a percent bar instead of one box per unit


def _names(folder):
    return os.listdir(folder) if os.path.isdir(folder) else []


def _units(folder, pattern):
    """Unit numbers of the files in folder whose name matches pattern (group 1 is the number)."""
    rx = re.compile(pattern)
    return sorted({int(m.group(1)) for n in _names(folder) for m in [rx.match(n)] if m})


def _complete(folder, pattern, per_unit):
    """Units with all per_unit files present, for steps that write several files per PD."""
    rx = re.compile(pattern)
    count = {}
    for n in _names(folder):
        m = rx.match(n)
        if m:
            count[int(m.group(1))] = count.get(int(m.group(1)), 0) + 1
    return sorted(u for u, c in count.items() if c >= per_unit)


def _pd_data():
    if not os.path.isfile(params.pd_file):
        return None
    return tracker._pd_data()


def _n_pd(pd_data):
    return len(pd_data["thres_ids"]) if pd_data is not None else int(params.prd_n_active)


def _movies_done(n_pd):
    """PDs whose topos folder holds every psi movie."""
    done = []
    for p in range(n_pd):
        folder = os.path.join(params.out_dir, "topos", f"PrD_{p + 1}")
        if len([n for n in _names(folder) if re.match(r"psi_\d+\.gif$", n)]) >= params.num_psi:
            done.append(p)
    return done


def _retained(done_pds):
    """Retained particles of each PD with a diff_maps file (one small read per file)."""
    out = {}
    for p in done_pds:
        with h5py.File(params.get_psi_file(p), "r") as f:
            out[p] = int(f["posPath"].shape[0])
    return out


def _star_rows(path):
    with open(path) as f:
        return sum(1 for line in f if "@" in line)


def _tracker_info(step):
    info = tracker.group_info(step.replace("-", "_"))
    if info is None:
        return dict(state="blank", failed=None)
    stale = any(s[0] == step and s[1] == "stale" for s in tracker.status())
    return dict(state="stale" if stale else "recorded", failed=int(sum(c[2] for c in info["checks"])))


def snapshot():
    """Every step with the units done, the units expected and the particles that reached it."""
    pd_data = _pd_data()
    n_pd, n_states = _n_pd(pd_data), int(params.states_per_coord) - int(params.width_1D) + 1
    thres_high = pd_data.get("thres_high", params.prd_thres_high) if pd_data is not None else params.prd_thres_high
    sizes = [min(len(pd_data["image_indices_full"][t]), thres_high) for t in pd_data["thres_ids"]] if pd_data is not None else []

    dist = _units(params.dist_dir, r"IMGs_prD_(\d+)\.h5$")
    diff = _units(params.psi_dir, r"gC_trimmed_psi_prD_(\d+)\.h5$")
    psi2 = _complete(params.psi2_dir, r"S2_prD_(\d+)_psi_\d+\.h5$", params.num_psi)
    movies = _movies_done(n_pd)
    flow = _units(params.CC_OF_dir, r"OF_prD_(\d+)\.pkl$")
    el = _units(params.EL_dir, r"S2_prD_(\d+)\.pkl$")
    states = _units(params.bin_dir, rf"EulerAngles_{re.escape(str(params.traj_name))}_(\d+)_of_{params.states_per_coord}\.star$")

    retained = _retained(diff)
    in_pd = sum(sizes[p] for p in dist if p < len(sizes))
    framed = sum(retained[p] - 2 * (retained[p] // params.con_order_range) for p in el if p in retained)
    in_state = sum(_star_rows(os.path.join(params.bin_dir, f"EulerAngles_{params.traj_name}_{b}_of_{params.states_per_coord}.star"))
                   for b in states)
    cc_done = os.path.isfile(params.CC_file)

    steps = [
        dict(step="threshold", unit="file", done=[0] if pd_data is not None else [], expected=1,
             particles=(f"{sum(sizes):,} in a PD" if pd_data is not None else ""), tracker=None),
        dict(step="calc-distance", unit="PDs", done=dist, expected=n_pd, particles=f"{in_pd:,} in a PD",
             tracker=_tracker_info("calc-distance")),
        dict(step="manifold-analysis", unit="PDs", done=diff, expected=n_pd, particles=f"{sum(retained.values()):,} retained",
             tracker=_tracker_info("manifold-analysis")),
        dict(step="psi-analysis", unit="PDs", done=psi2, expected=n_pd, particles="", tracker=_tracker_info("psi-analysis")),
        dict(step="nlsa-movie", unit="PDs", done=movies, expected=n_pd, particles="", tracker=None),
        dict(step="find-ccs", unit="PDs", done=flow, expected=n_pd, particles="CC_file.pkl written" if cc_done else "",
             tracker=_tracker_info("find-ccs"), complete=cc_done),
        dict(step="probability-landscape", unit="PDs", done=el, expected=n_pd, particles=f"{framed:,} with an NLSA frame",
             tracker=_tracker_info("probability-landscape")),
        dict(step="trajectory", unit="states", done=[b - 1 for b in states], expected=n_states, particles=f"{in_state:,} in a state",
             tracker=_tracker_info("trajectory")),  # done counts from 0 like the PDs, so unit b is state b + 1
    ]
    return dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), project=params.project_name, out_dir=os.path.realpath(params.out_dir),
                n_pd=n_pd, n_states=n_states, steps=steps)


def finished(snap):
    last = snap["steps"][-1]
    return last["expected"] > 0 and len(last["done"]) >= last["expected"]


def _grid_row(done, expected):
    """One box per unit, in blocks of ten, or a percent bar when there are many units."""
    if expected > BAR_LIMIT:
        filled = round(50 * len(done) / expected)
        return "#" * filled + "." * (50 - filled) + f"  {100 * len(done) / expected:3.0f}%"
    boxes = ["#" if u in set(done) else "." for u in range(expected)]
    return " ".join("".join(boxes[i:i + 10]) for i in range(0, expected, 10))


def render(snap, grid=True):
    lines = [f"Run {snap['project']}  {snap['n_pd']} PDs  {snap['n_states']} states  {snap['time']}",
             f"  {'Step':<22} {'Done':<16} {'Particles':<28} Tracker"]
    for s in snap["steps"]:
        done = "Done" if s.get("complete") else f"{len(s['done'])} of {s['expected']} {s['unit']}"
        t = s["tracker"]
        track = "" if t is None else t["state"] if t["failed"] is None else f"{t['state']}  {t['failed']} failed"
        lines.append(f"  {s['step']:<22} {done:<16} {s['particles']:<28} {track}".rstrip())
    if grid:
        lines.append("")
        for s in snap["steps"]:
            if s["unit"] in ("PDs", "states"):
                lines.append(f"  {s['step']:<22} {_grid_row(s['done'], s['expected'])}")
    return "\n".join(lines)


def write_json(snap):
    """progress.json holds the latest snapshot and progress.jsonl every snapshot in order."""
    path = os.path.join(params.out_dir, "progress.json")
    with open(path, "w") as f:
        json.dump(snap, f)
    with open(os.path.join(params.out_dir, "progress.jsonl"), "a") as f:
        f.write(json.dumps(snap) + "\n")
    return path


def op(watch=0, json_out=False, grid=True):
    """utility progress. Print the progress once, or every `watch` seconds until the trajectory step is done."""
    while True:
        snap = snapshot()
        if watch:
            print("\033[2J\033[H", end="")
        print(render(snap, grid))
        if json_out:
            print(f"\n  Progress file       {write_json(snap)}")
        if not watch or finished(snap):
            return
        time.sleep(watch)
