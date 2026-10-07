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


FATES = ["In a state", "Frame without a state", "No frame (start of psi order)", "No frame (end of psi order)",
         "Trimmed", "PD not selected", "PD not analyzed", "Over prd_thres_high", "In no PD"]
COLUMNS = ["particle_id", "mirrored", "prd", "C", "place", "retained", "sorted", "frame", "state", "image",
           "movie_frames", "tau", "tau_eq", "fate"]


def _read_pos_path(prd):
    path = params.get_psi_file(prd)
    if not os.path.isfile(path):
        return None
    with h5py.File(path, "r") as f:
        return f["posPath"][()]


def particle_map(frames, states, movies, pd_data):
    """Where every particle image goes, one row per particle and PD (one row per state when a frame sits in several)."""
    traj = myio.fin1(_traj_vars_file())
    selected = set(int(x) for x in traj["xSelect"])
    mirrored = pd_data["image_is_mirrored"]
    members = [np.asarray(pd_data["image_indices_full"][t]) for t in pd_data["thres_ids"]]
    frame_info = frames.set_index(["prd", "frame"])[["tau", "tau_eq"]]
    in_state = states.groupby(["prd", "frame"])[["state", "image"]].apply(lambda t: list(zip(t.state, t.image)))
    in_movie = {} if movies is None else \
        movies.groupby(["prd", "frame"]).movie_frame.apply(lambda v: " ".join(map(str, v))).to_dict()

    rows, absent = [], []
    for p in range(params.prd_n_active):
        if not os.path.isfile(params.get_dist_file(p)):
            absent.append(p)
            continue
        ind = _read_ind(p)
        rows += [dict(particle_id=int(r), prd=p, fate="Over prd_thres_high") for r in np.setdiff1d(members[p], ind)]
        if p in selected:
            pos_path, pos_psi1 = np.asarray(traj["posPathAll"][p]), np.asarray(traj["posPsi1All"][p])
        else:
            pos_path, pos_psi1 = _read_pos_path(p), None
        retained = {} if pos_path is None else {int(i): j for j, i in enumerate(pos_path)}
        sorted_of = {} if pos_psi1 is None else {int(j): k for k, j in enumerate(pos_psi1)}
        C = None if pos_path is None else len(pos_path) // params.con_order_range
        for i, r in enumerate(ind):
            row = dict(particle_id=int(r), prd=p, place=i)
            if pos_path is None:
                rows.append({**row, "fate": "PD not analyzed"})
                continue
            if i not in retained:
                rows.append({**row, "fate": "Trimmed"})
                continue
            row.update(C=C, retained=retained[i])
            if p not in selected:
                rows.append({**row, "fate": "PD not selected"})
                continue
            s = sorted_of[retained[i]]
            m = s - C + 1
            row["sorted"] = s
            if m < 0 or m >= len(pos_path) - 2 * C:
                rows.append({**row, "fate": f"No frame ({'start' if m < 0 else 'end'} of psi order)"})
                continue
            row.update(frame=m, movie_frames=in_movie.get((p, m), ""), **frame_info.loc[(p, m)].to_dict())
            hits = in_state.get((p, m), [])
            rows += [{**row, "state": b, "image": n, "fate": "In a state"} for b, n in hits] or \
                    [{**row, "fate": "Frame without a state"}]
    if absent:
        print(f"{len(absent)} active PDs have no distance file and are left out {absent}")

    in_pd = set(int(r) for m in members for r in m)
    rows += [dict(particle_id=r, fate="In no PD") for r in range(len(mirrored)) if r not in in_pd]
    table = pd.DataFrame(rows).reindex(columns=COLUMNS)
    table["mirrored"] = mirrored[table.particle_id.values]
    for col in ["prd", "C", "place", "retained", "sorted", "frame", "state", "image"]:
        table[col] = table[col].astype("Int64")
    table["movie_frames"] = table["movie_frames"].fillna("")
    return table.sort_values(["particle_id", "prd", "state"], na_position="first").reset_index(drop=True)


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

    with open(params.pd_file, "rb") as f:
        pd_data = pickle.load(f)
    mirrored, quats = pd_data["image_is_mirrored"], pd_data["quats_full"]
    frames, states = track(mirrored)
    movie_table = movie_frames(frames) if movies else None
    table = particle_map(frames, states, movie_table, pd_data)

    out_dir = os.path.join(params.out_dir, "particle_index")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "particle_index.csv")
    table.to_csv(out_file, index=False)

    print(f"Particles           {len(mirrored)}")
    print(f"In a state          {table[table.fate == 'In a state'].particle_id.nunique()} particles, "
          f"{len(states)} state images in {states.state.nunique()} states")
    if movies:
        print(f"2D movie positions  {len(movie_table)} ({movie_table.particle_id.nunique()} distinct particles)")
    print("Rows by fate")
    counts = table.fate.value_counts()
    for fate in FATES:
        if fate in counts:
            print(f"  {fate:<30} {counts[fate]}")
    print(f"Output file         {os.path.realpath(out_file)} ({len(table)} rows)")

    missing = missing_states(states)
    if missing:
        print(f"No state star file for states {missing}. Their rows in particle_index.csv are images the trajectory step should have written.")

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
