"""Particle ID (0 based row of the input star file) of every NLSA frame, state image and 2D movie frame.

NLSA frame m of a PD is sorted particle s = m + C - 1 with C = nS // con_order_range (psi_analysis._NLSA).
Its particle ID is ind[posPath[PosPsi1[s]]], read from files a finished run already saved. Needs the
trajectory step. state, image and movie_frame count from 1 as in the state file names, all other indices
from 0.
"""

import os
import pickle

import h5py
import numpy as np
import pandas as pd

from ManifoldEM import myio, star, util
from ManifoldEM.params import params
from ManifoldEM.quaternion import quaternion_to_S2


def frame_chain(ind, posPath, posPsi1, con_order_range, shift=0):
    """Every step from NLSA frame to particle ID for one PD, -1 where `shift` leaves the PD (controls only)."""
    nS = len(posPath)
    C = nS // con_order_range
    if C < 1:
        raise ValueError(f"PD with {nS} particles has no NLSA frames for con_order_range {con_order_range}")
    s = np.arange(nS - 2 * C) + C - 1 + shift
    inside = (s >= 0) & (s < nS)
    chain = {name: np.full(len(s), -1, dtype=int) for name in ("sorted", "retained", "place", "particle_id")}
    chain["sorted"][inside] = s[inside]
    chain["retained"][inside] = np.asarray(posPsi1)[s[inside]]
    chain["place"][inside] = np.asarray(posPath)[chain["retained"][inside]]
    chain["particle_id"][inside] = np.asarray(ind)[chain["place"][inside]]
    return chain


def frame_particles(ind, posPath, posPsi1, con_order_range, shift=0):
    """Particle ID of every NLSA frame of one PD, -1 where `shift` leaves the PD (controls only)."""
    return frame_chain(ind, posPath, posPsi1, con_order_range, shift)["particle_id"]


def state_frames(tau_eq, n_states, width):
    """Frames of each state, selected as in writeRelionS2.extract_traj_data_by_prd."""
    last = n_states - width
    frames = []
    for b in range(last + 1):
        lower = tau_eq >= b / n_states
        if b == last:
            upper = tau_eq <= (b + width) / n_states
        else:
            upper = tau_eq < (b + width) / n_states
        frames.append(np.nonzero(lower & upper)[0])
    return frames


def _traj_vars_file():
    return f"{params.traj_file}name{params.traj_name}_vars.pkl"


def _state_star_file(state):
    return os.path.join(params.bin_dir, f"EulerAngles_{params.traj_name}_{state}_of_{params.states_per_coord}.star")


def _state_stack_file(state):
    return os.path.join(params.bin_dir, f"imgsRELION_{params.traj_name}_{state}_of_{params.states_per_coord}.mrcs")


def _rel(path):
    return os.path.relpath(path, params.out_dir)


def _read_ind(prd):
    with h5py.File(params.get_dist_file(prd), "r") as f:
        return f["ind"][()]


def _load_pd_data():
    """Mirror flag and folded quaternion of every particle."""
    with open(params.pd_file, "rb") as f:
        pd_data = pickle.load(f)
    return pd_data["image_is_mirrored"], pd_data["quats_full"]


def track(mirrored=None, shift=0):
    """Frame table (one row per NLSA frame) and state table (one row per state image)."""
    if mirrored is None:
        mirrored, _ = _load_pd_data()
    traj = myio.fin1(_traj_vars_file())

    frames = []
    chunks = [[] for _ in range(params.states_per_coord - params.width_1D + 1)]
    for x in traj["xSelect"]:
        chain = frame_chain(_read_ind(x), traj["posPathAll"][x], traj["posPsi1All"][x],
                            params.con_order_range, shift)
        ids = chain["particle_id"]
        tau = np.ravel(traj["trajTaus"][x])
        tau_eq = np.ravel(util.hist_match(traj["trajTaus"][x], traj["tauAvg"]))
        C = len(traj["posPathAll"][x]) // params.con_order_range
        frames.append(pd.DataFrame(dict(prd=x, frame=np.arange(len(ids)), C=C, **chain,
                                        mirrored=mirrored[ids.clip(0)] & (ids >= 0), tau=tau, tau_eq=tau_eq)))
        for b, fr in enumerate(state_frames(tau_eq, params.states_per_coord, params.width_1D)):
            if len(fr):
                chunks[b].append(pd.DataFrame(dict(prd=x, frame=fr, particle_id=ids[fr])))

    # state stacks are written PD by PD in xSelect order (writeRelionS2.concatenate_bin)
    states = []
    for b, chunk in enumerate(chunks):
        if chunk:
            t = pd.concat(chunk, ignore_index=True)
            t.insert(0, "image", np.arange(1, len(t) + 1))
            t.insert(0, "state", b + 1)
            states.append(t)

    return pd.concat(frames, ignore_index=True), pd.concat(states, ignore_index=True)


def movie_frames(frames):
    """One row per 2D movie position (tauinds of psi analysis) of every PD in the frame table."""
    rows = []
    for x, f in frames.groupby("prd", sort=False):
        tauinds = np.ravel(myio.fin1(params.get_EL_file(x))["tauinds"]).astype(int)
        rows.append(pd.DataFrame(dict(prd=x, movie_frame=np.arange(1, len(tauinds) + 1), frame=tauinds,
                                      particle_id=f.particle_id.values[tauinds])))
    return pd.concat(rows, ignore_index=True)


def particle_summary(frames, states, mirrored=None):
    """One row per particle with the active PDs, NLSA frames and state images it reaches."""
    if mirrored is None:
        mirrored, _ = _load_pd_data()
    n = len(mirrored)
    pds = np.zeros(n, dtype=int)
    absent = []
    for p in range(params.prd_n_active):
        if os.path.isfile(params.get_dist_file(p)):
            pds += np.bincount(_read_ind(p), minlength=n)
        else:
            absent.append(p)
    if absent:
        print(f"{len(absent)} active PDs have no distance file and are left out of pds {absent}")
    ids = frames.particle_id.values
    return pd.DataFrame(dict(particle_id=np.arange(n), mirrored=mirrored, pds=pds,
                             frames=np.bincount(ids[ids >= 0], minlength=n),
                             state_images=np.bincount(states.particle_id.values, minlength=n)))


def missing_states(states):
    """States of the table without a state star file on disk."""
    return [int(b) for b in states.state.unique() if not os.path.isfile(_state_star_file(b))]


def verify(states, quats=None, tol=0.01):
    """Count state images whose star file direction is within `tol` degrees of the tracked particle."""
    if quats is None:
        _, quats = _load_pd_data()
    hit, missing, mismatched = 0, [], []
    for b, t in states.groupby("state"):
        star_file = _state_star_file(b)
        if not os.path.isfile(star_file):
            missing.append(int(b))
            continue
        df = star.parse_star(star_file, 0)
        if len(df) != len(t):
            mismatched.append(int(b))
            continue
        rot, tilt = np.deg2rad(df["rlnAngleRot"].values), np.deg2rad(df["rlnAngleTilt"].values)
        written = np.vstack((np.sin(tilt) * np.cos(rot), np.sin(tilt) * np.sin(rot), np.cos(tilt)))
        ids = t.particle_id.values
        tracked = quaternion_to_S2(quats[:, ids.clip(0)])
        angle = np.degrees(np.arccos(np.clip(np.sum(written * tracked, axis=0), -1, 1)))
        hit += int(np.sum((angle < tol) & (ids >= 0)))
    return hit, missing, mismatched


def _step_table(rows):
    """Aligned lines for (step, read from, index, result) rows."""
    head = ("Step", "Read from", "Index", "Result")
    width = [max(len(str(r[k])) for r in rows + [head]) for k in range(3)]
    return [f"  {r[0]:<{width[0]}}  {r[1]:<{width[1]}}  {r[2]:<{width[2]}}  {r[3]}" for r in [head] + rows]


def trace_image(frames, states, state, image, mirrored):
    """Steps from one state image back to its particle ID."""
    hit = states[(states.state == state) & (states.image == image)]
    if hit.empty:
        return [f"Image {image} of state {state} does not exist"]
    x, m = int(hit.prd.iloc[0]), int(hit.frame.iloc[0])
    f = frames[(frames.prd == x) & (frames.frame == m)].iloc[0]
    C, s, j, i, r = (int(f[k]) for k in ("C", "sorted", "retained", "place", "particle_id"))
    traj = _rel(_traj_vars_file())
    rows = [("1", _rel(_state_stack_file(state)), f"image {image}", f"PD {x}, frame m = {m} (stack written PD by PD)"),
            ("2", f"NLSA offset, C = {C}", "s = m + C - 1", f"sorted s = {s}"),
            ("3", f"{traj} (posPsi1All[{x}])", f"PosPsi1[{s}]", f"retained j = {j}"),
            ("4", f"{traj} (posPathAll[{x}])", f"posPath[{j}]", f"place i = {i}"),
            ("5", f"{_rel(params.get_dist_file(x))} (ind)", f"ind[{i}]", f"particle ID r = {r}" + (" (mirrored)" if mirrored[r] else ""))]
    return [f"Image {image} of state {state} back to its particle"] + _step_table(rows) + \
           [f"  Particle ID {r} is row {r} of {os.path.basename(params.align_param_file)} (0 based)"]


def trace_particle(frames, states, pid, mirrored):
    """Steps from one particle ID through every active PD that holds it."""
    if not 0 <= pid < len(mirrored):
        return [f"Particle ID {pid} is outside the star file (0 to {len(mirrored) - 1})"]
    traj = myio.fin1(_traj_vars_file())
    selected = set(int(x) for x in traj["xSelect"])
    out = [f"Particle ID {pid} (row {pid} of {os.path.basename(params.align_param_file)}, 0 based)"
           + (", mirrored" if mirrored[pid] else "")]
    found = False
    for p in range(params.prd_n_active):
        if not os.path.isfile(params.get_dist_file(p)):
            continue
        ind = _read_ind(p)
        places = np.nonzero(ind == pid)[0]
        if not len(places):
            continue
        found = True
        i = int(places[0])
        rows = [("1", f"{_rel(params.get_dist_file(p))} (ind)", f"ind[{i}] = {pid}", f"place i = {i}")]
        if p not in selected:
            out += [f"PD {p}"] + _step_table(rows + [("2", "Trajectory selection", "xSelect", "PD not selected, no frames")])
            continue
        posPath, posPsi1 = np.asarray(traj["posPathAll"][p]), np.asarray(traj["posPsi1All"][p])
        name = _rel(_traj_vars_file())
        j = np.nonzero(posPath == i)[0]
        if not len(j):
            out += [f"PD {p}"] + _step_table(rows + [("2", f"{name} (posPathAll[{p}])", f"posPath has no {i}", "trimmed, no frame")])
            continue
        j = int(j[0])
        s = int(np.nonzero(posPsi1 == j)[0][0])
        C = len(posPath) // params.con_order_range
        m = s - C + 1
        rows += [("2", f"{name} (posPathAll[{p}])", f"posPath[{j}] = {i}", f"retained j = {j}"),
                 ("3", f"{name} (posPsi1All[{p}])", f"PosPsi1[{s}] = {j}", f"sorted s = {s}")]
        if not 0 <= m < len(posPath) - 2 * C:
            end = "start" if m < 0 else "end"
            out += [f"PD {p}"] + _step_table(rows + [("4", f"NLSA offset, C = {C}", "m = s - C + 1", f"no frame ({end} of the psi order)")])
            continue
        rows.append(("4", f"NLSA offset, C = {C}", "m = s - C + 1", f"frame m = {m}"))
        img = states[(states.prd == p) & (states.frame == m)]
        for k, (b, n) in enumerate(zip(img.state, img.image)):
            rows.append((str(5 + k), _rel(_state_stack_file(int(b))), "stack write order", f"image {int(n)} of state {int(b)}"))
        out += [f"PD {p}"] + _step_table(rows)
    if not found:
        out.append("  In no active PD (sparse direction)")
    return out


def trace_prd(frames, states, prd):
    """Every step for every NLSA frame of one PD."""
    f = frames[frames.prd == prd]
    if f.empty:
        return [f"PD {prd} has no NLSA frames in the trajectory selection"]
    where = {int(m): (int(b), int(n)) for b, n, m in
             states[states.prd == prd][["state", "image", "frame"]].itertuples(index=False)}
    lines = [f"PD {prd}, C = {int(f.C.iloc[0])}, {len(f)} frames",
             f"  {'Frame':>6} {'Sorted':>7} {'Retained':>9} {'Place':>6} {'Particle ID':>12} {'State':>6} {'Image':>6}"]
    for row in f.itertuples(index=False):
        b, n = where.get(int(row.frame), ("none", ""))
        lines.append(f"  {row.frame:>6} {row.sorted:>7} {row.retained:>9} {row.place:>6} {row.particle_id:>12} {b:>6} {n:>6}")
    return lines


def op(verify_states=False, movies=True, trace_images=(), trace_particles=(), trace_prds=()):
    if not os.path.isfile(_traj_vars_file()):
        print("No trajectory variables found. Have you run the 'trajectory' step?")
        return

    mirrored, quats = _load_pd_data()
    frames, states = track(mirrored)
    particles = particle_summary(frames, states, mirrored)

    out_dir = os.path.join(params.out_dir, "particle_index")
    os.makedirs(out_dir, exist_ok=True)
    frames.to_csv(os.path.join(out_dir, "frames.csv"), index=False)
    states.to_csv(os.path.join(out_dir, "states.csv"), index=False)
    particles.to_csv(os.path.join(out_dir, "particles.csv"), index=False)
    saved = [("frames.csv", len(frames)), ("states.csv", len(states)), ("particles.csv", len(particles))]

    print(f"Particles           {len(particles)}")
    print(f"In an active PD     {int((particles.pds > 0).sum())}")
    print(f"In several PDs      {int((particles.pds > 1).sum())}")
    print(f"With an NLSA frame  {int((particles.frames > 0).sum())}")
    print(f"In a state          {int((particles.state_images > 0).sum())}")
    print(f"State images        {len(states)} in {states.state.nunique()} states")
    if movies:
        movie_table = movie_frames(frames)
        movie_table.to_csv(os.path.join(out_dir, "movies.csv"), index=False)
        saved.append(("movies.csv", len(movie_table)))
        print(f"2D movie positions  {len(movie_table)} ({movie_table.particle_id.nunique()} distinct particles)")
    print(f"Output folder       {os.path.realpath(out_dir)}")
    for name, n_rows in saved:
        print(f"  {name:<14} {n_rows} rows")

    missing = missing_states(states)
    if missing:
        print(f"No state star file for states {missing}. Their rows in states.csv are images the trajectory step should have written.")

    if verify_states:
        hit, _, mismatched = verify(states, quats)
        control, _, _ = verify(track(mirrored, shift=1)[1], quats)
        print("\nVerification against the state star files (directions within 0.01 degrees)")
        print(f"Tracked particles   {hit} of {len(states)}")
        print(f"Offset shifted by 1 {control} of {len(states)} (control)")
        if mismatched:
            print(f"States with a different row count in their star file {mismatched}")

    lines = []
    for state, image in trace_images:
        lines += [""] + trace_image(frames, states, state, image, mirrored)
    for pid in trace_particles:
        lines += [""] + trace_particle(frames, states, pid, mirrored)
    for prd in trace_prds:
        lines += [""] + trace_prd(frames, states, prd)
    if lines:
        trace_file = os.path.join(out_dir, "trace.txt")
        with open(trace_file, "w") as f:
            f.write("\n".join(lines[1:]) + "\n")
        print("\n".join(lines))
        print(f"\nTrace file          {os.path.realpath(trace_file)} ({len(lines) - 1} lines)")
