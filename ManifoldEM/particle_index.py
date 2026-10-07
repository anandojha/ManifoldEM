"""Particle IDs of NLSA frames, state images and 2D movie frames.

The particle ID is the 0 based row of the input star file. NLSA frame m of a projection
direction (PD) is sorted particle m + C - 1 with C = nS // con_order_range (psi_analysis._NLSA),
so its particle ID is

    ind[posPath[PosPsi1[m + C - 1]]]

with ind from the distance file and posPath, PosPsi1 from psi analysis. Every array is saved by
a finished run, so no stage has to be rerun. Requires the trajectory step.
"""

import os
import pickle

import h5py
import numpy as np
import pandas as pd

from ManifoldEM import myio, star, util
from ManifoldEM.params import params
from ManifoldEM.quaternion import quaternion_to_S2


def frame_particles(ind, posPath, posPsi1, con_order_range, shift=0):
    """Particle ID of every NLSA frame of one PD, -1 where `shift` leaves the PD (controls only)."""
    nS = len(posPath)
    C = nS // con_order_range
    if C < 1:
        raise ValueError(f"PD with {nS} particles has no NLSA frames for con_order_range {con_order_range}")
    sorted_ids = np.asarray(ind)[np.asarray(posPath)[np.asarray(posPsi1)]]
    s = np.arange(nS - 2 * C) + C - 1 + shift
    ids = np.full(len(s), -1, dtype=int)
    inside = (s >= 0) & (s < nS)
    ids[inside] = sorted_ids[s[inside]]
    return ids


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


def _read_ind(prd):
    with h5py.File(params.get_dist_file(prd), "r") as f:
        return f["ind"][()]


def track(shift=0):
    """Frame table (one row per NLSA frame) and state table (one row per state image)."""
    traj = myio.fin1(_traj_vars_file())
    with open(params.pd_file, "rb") as f:
        mirrored = pickle.load(f)["image_is_mirrored"]

    frames = []
    chunks = [[] for _ in range(params.states_per_coord - params.width_1D + 1)]
    for x in traj["xSelect"]:
        ids = frame_particles(_read_ind(x), traj["posPathAll"][x], traj["posPsi1All"][x],
                              params.con_order_range, shift)
        tau = np.ravel(traj["trajTaus"][x])
        tau_eq = np.ravel(util.hist_match(traj["trajTaus"][x], traj["tauAvg"]))
        m = np.arange(len(ids))
        C = len(traj["posPathAll"][x]) // params.con_order_range
        frames.append(pd.DataFrame(dict(prd=x, frame=m, sorted=m + C - 1, particle_id=ids,
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


def particle_summary(frames, states):
    """One row per particle: active PDs holding it, NLSA frames and state images it reaches."""
    with open(params.pd_file, "rb") as f:
        mirrored = pickle.load(f)["image_is_mirrored"]
    n = len(mirrored)
    pds = np.zeros(n, dtype=int)
    for p in range(params.prd_n_active):
        pds += np.bincount(_read_ind(p), minlength=n)
    ids = frames.particle_id.values
    return pd.DataFrame(dict(particle_id=np.arange(n), mirrored=mirrored, pds=pds,
                             frames=np.bincount(ids[ids >= 0], minlength=n),
                             state_images=np.bincount(states.particle_id.values, minlength=n)))


def verify(states, tol=0.01):
    """Count state images whose star file direction is within `tol` degrees of the tracked particle."""
    with open(params.pd_file, "rb") as f:
        quats = pickle.load(f)["quats_full"]
    hit, missing, mismatched = 0, [], []
    for b, t in states.groupby("state"):
        star_file = os.path.join(params.bin_dir, f"EulerAngles_{params.traj_name}_{b}_of_{params.states_per_coord}.star")
        if not os.path.isfile(star_file):
            missing.append(b)
            continue
        df = star.parse_star(star_file, 0)
        if len(df) != len(t):
            mismatched.append(b)
            continue
        rot, tilt = np.deg2rad(df["rlnAngleRot"].values), np.deg2rad(df["rlnAngleTilt"].values)
        written = np.vstack((np.sin(tilt) * np.cos(rot), np.sin(tilt) * np.sin(rot), np.cos(tilt)))
        ids = t.particle_id.values
        tracked = quaternion_to_S2(quats[:, ids.clip(0)])
        angle = np.degrees(np.arccos(np.clip(np.sum(written * tracked, axis=0), -1, 1)))
        hit += int(np.sum((angle < tol) & (ids >= 0)))
    return hit, missing, mismatched


def op(verify_states=False):
    if not os.path.isfile(_traj_vars_file()):
        print("No trajectory variables found. Have you run the 'trajectory' step?")
        return

    frames, states = track()
    movies = movie_frames(frames)
    particles = particle_summary(frames, states)

    out_dir = os.path.join(params.out_dir, "particle_index")
    os.makedirs(out_dir, exist_ok=True)
    frames.to_csv(os.path.join(out_dir, "frames.csv"), index=False)
    states.to_csv(os.path.join(out_dir, "states.csv"), index=False)
    movies.to_csv(os.path.join(out_dir, "movies.csv"), index=False)
    particles.to_csv(os.path.join(out_dir, "particles.csv"), index=False)

    print(f"Particles           {len(particles)}")
    print(f"In an active PD     {int((particles.pds > 0).sum())}")
    print(f"With an NLSA frame  {int((particles.frames > 0).sum())}")
    print(f"In a state          {int((particles.state_images > 0).sum())}")
    print(f"State images        {len(states)} in {states.state.nunique()} states")
    print(f"2D movie positions  {len(movies)} ({movies.particle_id.nunique()} distinct particles)")
    print(f"Output in: {os.path.realpath(out_dir)}")

    if verify_states:
        hit, missing, mismatched = verify(states)
        control, _, _ = verify(track(shift=1)[1])
        print("\nVerification against the state star files (directions within 0.01 degrees)")
        print(f"Tracked particles   {hit} of {len(states)}")
        print(f"Offset shifted by 1 {control} of {len(states)} (control)")
        if missing:
            print(f"States without a star file: {missing}")
        if mismatched:
            print(f"States whose star file row count differs: {mismatched}")
