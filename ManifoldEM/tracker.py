"""Particle tracker. After each pipeline step, record where every particle image went in tracker.h5.

The particle ID is the 0 based row of the input star file. Every step renumbers the particles of a
PD, and each group of output/<project_name>/tracker.h5 stores the particle ID and PD next to the
numbers of that step

    calc_distance          place i        ind[i] is the particle ID
    manifold_analysis      retained j     posPath[j] is place i
    psi_analysis           sorted s       PosPsi1[s] is retained j, for every topo
    find_ccs               chosen topo    one per PD
    probability_landscape  frame m        m = s - C + 1 for the chosen topo, C = nS // con_order_range
    trajectory             state, image   stack write order gives PD and frame m

The mapper group joins them into one row per particle and PD. state, image and movie_frame count
from 1, all other numbers from 0, and -1 marks a number a particle never reached.
"""

import json
import os
import pickle
import time

import h5py
import mrcfile
import numpy as np
import pandas as pd

import ManifoldEM
from ManifoldEM import myio, star, util
from ManifoldEM.params import params
from ManifoldEM.quaternion import quaternion_to_S2

STEPS = ["calc-distance", "manifold-analysis", "psi-analysis", "find-ccs", "probability-landscape", "trajectory"]
MAPPER = ["particle_id", "mirrored", "prd", "C", "place", "retained", "topo", "sorted", "frame", "state", "image",
          "movie_frames", "tau", "tau_eq", "fate"]


def tracker_file():
    return os.path.join(params.out_dir, "tracker.h5")


def _group(step):
    return step.replace("-", "_")


# files written by the pipeline

def _pd_data():
    with open(params.pd_file, "rb") as f:
        return pickle.load(f)


def _members(pd_data):
    return [np.asarray(pd_data["image_indices_full"][t]) for t in pd_data["thres_ids"]]


def _dist_ind(prd):
    with h5py.File(params.get_dist_file(prd), "r") as f:
        return f["ind"][()]


def _diff_map(prd, psi=False):
    path = params.get_psi_file(prd)
    if not os.path.isfile(path):
        return None
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in (("ind", "posPath", "psi") if psi else ("ind", "posPath"))}


def _traj():
    path = f"{params.traj_file}name{params.traj_name}_vars.pkl"
    return myio.fin1(path) if os.path.isfile(path) else None


def _cc():
    return myio.fin1(params.CC_file) if os.path.isfile(params.CC_file) else None


def _state_star_file(state):
    return os.path.join(params.bin_dir, f"EulerAngles_{params.traj_name}_{state}_of_{params.states_per_coord}.star")


def _state_stack_file(state):
    return os.path.join(params.bin_dir, f"imgsRELION_{params.traj_name}_{state}_of_{params.states_per_coord}.mrcs")


def _check(step, checked, failed, against, detail=""):
    return [step, checked, int(failed), against, detail]


def _frame_numbers(pos_psi1, n_s, con_order_range, shift=0):
    """Sorted position, frame and fate of every retained particle j (frame -1 when none)."""
    C = n_s // con_order_range
    if C < 1:
        raise ValueError(f"PD with {n_s} particles has no NLSA frames for con_order_range {con_order_range}")
    sorted_pos = np.empty(n_s, dtype=int)
    sorted_pos[np.asarray(pos_psi1)] = np.arange(n_s)
    frame = sorted_pos - C + 1 - shift
    fate = np.where(frame < 0, "No frame (start of psi order)",
                    np.where(frame >= n_s - 2 * C, "No frame (end of psi order)", "Frame")).astype(object)
    frame[fate != "Frame"] = -1
    return C, sorted_pos, frame, fate


def state_frames(tau_eq, n_states, width):
    """Frames of each state, selected as in writeRelionS2.extract_traj_data_by_prd."""
    last = n_states - width
    frames = []
    for b in range(last + 1):
        upper = tau_eq <= (b + width) / n_states if b == last else tau_eq < (b + width) / n_states
        frames.append(np.nonzero((tau_eq >= b / n_states) & upper)[0])
    return frames


# one builder per step

def build_calc_distance():
    pd_data = _pd_data()
    members = _members(pd_data)
    mirrored = np.asarray(pd_data["image_is_mirrored"])
    thres_low, thres_high = pd_data.get("thres_low", params.prd_thres_low), pd_data.get("thres_high", params.prd_thres_high)
    parts, have, bad_trunc = [], [], []
    for p, m in enumerate(members):
        if not os.path.isfile(params.get_dist_file(p)):
            continue
        ind = _dist_ind(p)
        have.append(p)
        if not np.array_equal(ind, m[:thres_high]):
            bad_trunc.append(p)
        parts.append(pd.DataFrame(dict(particle_id=ind, prd=p, place=np.arange(len(ind)), fate="Kept")))
        over = np.setdiff1d(m, ind)
        parts.append(pd.DataFrame(dict(particle_id=over, prd=p, place=-1, fate="Over prd_thres_high")))
    in_pd = np.zeros(len(mirrored), dtype=bool)
    for m in members:
        in_pd[m] = True
    nowhere = np.nonzero(~in_pd)[0]
    parts.append(pd.DataFrame(dict(particle_id=nowhere, prd=-1, place=-1, fate="In no PD")))
    table = pd.concat(parts, ignore_index=True)
    table.insert(1, "mirrored", mirrored[table.particle_id.values])

    bad_bin = [p for p, m in enumerate(members) if len(np.unique(m)) != len(m)]
    if "occupancy_full" in pd_data:
        bad_bin += [p for p, t in enumerate(pd_data["thres_ids"]) if pd_data["occupancy_full"][t] < thres_low]
    twice = 0
    if getattr(params, "prd_assignment", "hard") == "hard" and members:
        twice = int((np.bincount(np.concatenate(members), minlength=len(mirrored)) > 1).sum())
    checks = [_check("Binning", f"{len(members)} PDs", len(set(bad_bin)) + twice, "pd_data.pkl",
                     f"particles in two PDs {twice}" if twice else ""),
              _check("Truncation", f"{len(have)} PDs", len(bad_trunc), "distances/", f"PDs {bad_trunc}" if bad_trunc else "")]
    settings = dict(prd_thres_low=int(thres_low), prd_thres_high=int(thres_high), tess_hemisphere_type=params.tess_hemisphere_type,
                    prd_assignment=getattr(params, "prd_assignment", "hard"))
    return table, checks, settings


def build_manifold_analysis():
    parts, checked, bad = [], 0, []
    for p in range(params.prd_n_active):
        dm = _diff_map(p)
        if dm is None:
            continue
        checked += 1
        ind, pos_path = dm["ind"], dm["posPath"]
        same = os.path.isfile(params.get_dist_file(p)) and np.array_equal(ind, _dist_ind(p))
        if not (same and np.all(np.diff(pos_path) > 0) and pos_path.min() >= 0 and pos_path.max() < len(ind)):
            bad.append(p)
        retained = np.full(len(ind), -1)
        retained[pos_path] = np.arange(len(pos_path))
        parts.append(pd.DataFrame(dict(particle_id=ind, prd=p, place=np.arange(len(ind)), retained=retained,
                                       fate=np.where(retained >= 0, "Kept", "Trimmed").astype(object))))
    if not parts:
        raise FileNotFoundError("no diff_maps files")
    table = pd.concat(parts, ignore_index=True)
    checks = [_check("Trimming", f"{checked} PDs", len(bad), "diff_maps/ against distances/", f"PDs {bad}" if bad else "")]
    return table, checks, dict(rad=float(params.rad))


def build_psi_analysis():
    parts, checked, bad = [], 0, []
    for p in range(params.prd_n_active):
        dm = _diff_map(p, psi=True)
        if dm is None:
            continue
        ids = dm["ind"][dm["posPath"]]
        n_s = len(ids)
        for k in range(min(params.num_psi, dm["psi"].shape[1])):
            psi2 = params.get_psi2_file(p, k)
            if not os.path.isfile(psi2):
                continue
            C, sorted_pos, frame, fate = _frame_numbers(np.argsort(dm["psi"][:, k]), n_s, params.con_order_range)
            with h5py.File(psi2, "r") as f:
                n_tau = f["tau"][()].size
            checked += 1
            if n_tau != n_s - 2 * C:
                bad.append((p, k))
            parts.append(pd.DataFrame(dict(particle_id=ids, prd=p, topo=k, sorted=sorted_pos, frame=frame, fate=fate)))
    if not parts:
        raise FileNotFoundError("no psi_analysis files")
    table = pd.concat(parts, ignore_index=True)
    checks = [_check("NLSA frames (every topo)", f"{checked} topos", len(bad), "psi_analysis/ tau", f"PD and topo {bad}" if bad else "")]
    return table, checks, dict(num_psi=int(params.num_psi), con_order_range=int(params.con_order_range))


def build_find_ccs():
    cc, pd_data = _cc(), _pd_data()
    if cc is None:
        raise FileNotFoundError("no CC/CC_file.pkl")
    psinums, senses = np.asarray(cc["psinums"][0]), np.asarray(cc["senses"][0])
    trash = set(int(p) for p in pd_data.get("trash_ids", set()))
    status = np.array(["Trash" if p in trash else "Unassigned" if psinums[p] == -1 else "Chosen"
                       for p in range(len(psinums))], dtype=object)
    table = pd.DataFrame(dict(prd=np.arange(len(psinums)), topo=psinums, sense=senses, status=status))
    bad = [p for p in range(len(psinums)) if psinums[p] >= params.num_psi or psinums[p] < -1
           or (p in trash and psinums[p] != -1)]
    checks = [_check("Topo choice", f"{len(psinums)} PDs", len(bad), "CC_file.pkl and trash", f"PDs {bad}" if bad else "")]
    return table, checks, {}


def build_probability_landscape():
    traj, cc, pd_data = _traj(), _cc(), _pd_data()
    if traj is None:
        raise FileNotFoundError("no traj/traj_name*_vars.pkl")
    selected = [int(x) for x in traj["xSelect"]]
    psinums = None if cc is None else np.asarray(cc["psinums"][0])
    parts, carried, sort_bad, frame_bad = [], [], [], []
    for x in selected:
        pos_path, pos_psi1 = np.asarray(traj["posPathAll"][x]), np.asarray(traj["posPsi1All"][x])
        ids = _dist_ind(x)[pos_path]
        n_s = len(pos_path)
        C, sorted_pos, frame, fate = _frame_numbers(pos_psi1, n_s, params.con_order_range)
        tau_all = np.ravel(traj["trajTaus"][x])
        if tau_all.size != n_s - 2 * C:
            frame_bad.append(x)
        tau = np.where(frame >= 0, tau_all[frame.clip(0, tau_all.size - 1)], np.nan)
        dm = _diff_map(x, psi=psinums is not None)
        if dm is None or not np.array_equal(dm["posPath"], pos_path):
            carried.append(x)
        if psinums is not None and dm is not None:
            along = dm["psi"][pos_psi1, int(psinums[x])]
            if not (np.array_equal(np.sort(pos_psi1), np.arange(n_s)) and np.all(np.diff(along) >= 0)):
                sort_bad.append(x)
        parts.append(pd.DataFrame(dict(particle_id=ids, prd=x, C=C, retained=np.arange(n_s),
                                       topo=-1 if psinums is None else int(psinums[x]), sorted=sorted_pos,
                                       frame=frame, tau=tau, fate=fate)))
    table = pd.concat(parts, ignore_index=True)

    checks = [_check("Carried forward", f"{len(selected)} PDs", len(carried), "traj/ against diff_maps/", f"PDs {carried}" if carried else ""),
              _check("Sorting", f"{len(selected)} PDs", len(sort_bad), "diff_maps/ psi and CC_file.pkl", f"PDs {sort_bad}" if sort_bad else ""),
              _check("NLSA frames", f"{len(selected)} PDs", len(frame_bad), "traj/ tau against posPath", f"PDs {frame_bad}" if frame_bad else "")]
    if psinums is not None:
        trash = set(int(p) for p in pd_data.get("trash_ids", set()))
        expected = {p for p in range(len(psinums)) if psinums[p] != -1 and p not in trash and os.path.isfile(params.get_EL_file(p))}
        diff = sorted(expected ^ set(selected))
        checks.append(_check("PD selection", f"{len(psinums)} PDs", len(diff), "CC_file.pkl and trash", f"PDs {diff}" if diff else ""))
    return table, checks, dict(con_order_range=int(params.con_order_range), xSelect=selected)


def _states(traj, frame_ids):
    """One row per state image, in the write order of writeRelionS2.concatenate_bin."""
    chunks = [[] for _ in range(params.states_per_coord - params.width_1D + 1)]
    for x in traj["xSelect"]:
        tau_eq = np.ravel(util.hist_match(traj["trajTaus"][x], traj["tauAvg"]))
        for b, fr in enumerate(state_frames(tau_eq, params.states_per_coord, params.width_1D)):
            if len(fr):
                chunks[b].append(pd.DataFrame(dict(prd=int(x), frame=fr, particle_id=frame_ids[int(x)][fr], tau_eq=tau_eq[fr])))
    states = []
    for b, chunk in enumerate(chunks):
        if chunk:
            t = pd.concat(chunk, ignore_index=True)
            t.insert(0, "image", np.arange(1, len(t) + 1))
            t.insert(0, "state", b + 1)
            states.append(t)
    return pd.concat(states, ignore_index=True)


def _frame_ids(traj, shift=0):
    ids = {}
    for x in traj["xSelect"]:
        pos_path, pos_psi1 = np.asarray(traj["posPathAll"][x]), np.asarray(traj["posPsi1All"][x])
        n_s, C = len(pos_path), len(pos_path) // params.con_order_range
        s = np.arange(n_s - 2 * C) + C - 1 + shift
        ok = (s >= 0) & (s < n_s)
        frame_ids = np.full(len(s), -1)
        frame_ids[ok] = _dist_ind(int(x))[pos_path[pos_psi1[s[ok]]]]
        ids[int(x)] = frame_ids
    return ids


def _angle_hits(states, quats, tol=0.01):
    hit, missing, mismatched = 0, [], []
    for b, t in states.groupby("state"):
        if not os.path.isfile(_state_star_file(b)):
            missing.append(int(b))
            continue
        df = star.parse_star(_state_star_file(b), 0)
        if len(df) != len(t):
            mismatched.append(int(b))
            continue
        rot, tilt = np.deg2rad(df["rlnAngleRot"].values), np.deg2rad(df["rlnAngleTilt"].values)
        written = np.vstack((np.sin(tilt) * np.cos(rot), np.sin(tilt) * np.sin(rot), np.cos(tilt)))
        ids = t.particle_id.values
        angle = np.degrees(np.arccos(np.clip(np.sum(written * quaternion_to_S2(quats[:, ids.clip(0)]), axis=0), -1, 1)))
        hit += int(np.sum((angle < tol) & (ids >= 0)))
    return hit, missing, mismatched


def build_trajectory():
    traj, pd_data = _traj(), _pd_data()
    if traj is None:
        raise FileNotFoundError("no traj/traj_name*_vars.pkl")
    states = _states(traj, _frame_ids(traj))
    stacks = {b: mrcfile.mmap(_state_stack_file(b), mode="r").data for b in sorted(states.state.unique())
              if os.path.isfile(_state_stack_file(b))}
    movies, checked, failed, dim = {}, 0, 0, params.ms_num_pixels
    for x, t in states.groupby("prd"):
        el = myio.fin1(params.get_EL_file(x))  # one PD at a time keeps memory low
        for pos, m in enumerate(np.ravel(el["tauinds"]).astype(int), start=1):
            movies.setdefault((x, int(m)), []).append(str(pos))
        t = t[t.state.isin(list(stacks))]
        if len(t):
            imgt = (el["IMGT"] / (len(traj["posPathAll"][x]) // params.con_order_range)).T
            for b, k, m in zip(t.state.values, t.image.values - 1, t.frame.values):
                checked += 1
                failed += not np.array_equal(np.asarray(stacks[b][k]), imgt[m].astype(np.float32).reshape(dim, dim).T * -1)
        del el
    states["movie_frames"] = [" ".join(movies.get((p, m), [])) for p, m in zip(states.prd, states.frame)]
    table = states[["particle_id", "prd", "frame", "state", "image", "movie_frames", "tau_eq"]].assign(fate="In a state")

    quats = pd_data["quats_full"]
    hit, missing, mismatched = _angle_hits(states, quats)
    control, _, _ = _angle_hits(_states(traj, _frame_ids(traj, shift=1)), quats)
    detail = f"shifted control {control} of {len(states)}"
    detail += f", no star file for states {missing}" if missing else ""
    detail += f", row count differs in states {mismatched}" if mismatched else ""

    checks = [_check("States", f"{len(states)} images", len(states) - hit, "bin/*.star angles", detail),
              _check("Images", f"{checked} images", failed, "bin/*.mrcs against ELConc IMGT",
                     f"{len(stacks)} of {states.state.nunique()} stacks on disk")]
    return table, checks, dict(states_per_coord=int(params.states_per_coord), width_1D=int(params.width_1D), traj_name=str(params.traj_name))


BUILDERS = {"calc-distance": build_calc_distance, "manifold-analysis": build_manifold_analysis,
            "psi-analysis": build_psi_analysis, "find-ccs": build_find_ccs,
            "probability-landscape": build_probability_landscape, "trajectory": build_trajectory}


# tracker.h5

def write_group(name, table, checks=(), settings=None):
    with h5py.File(tracker_file(), "a") as f:
        if name in f:
            del f[name]
        g = f.create_group(name)
        for col in table.columns:
            values = table[col].values
            if table[col].dtype == object:
                g.create_dataset(col, data=np.asarray(values, dtype=object).astype(str).astype(object),
                                 dtype=h5py.string_dtype(), compression="gzip")
            else:
                g.create_dataset(col, data=values, compression="gzip")
        g.attrs["columns"] = json.dumps(list(table.columns))
        g.attrs["written"] = time.time()
        g.attrs["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        g.attrs["version"] = ManifoldEM.__version__
        g.attrs["settings"] = json.dumps(settings or {})
        g.attrs["checks"] = json.dumps([list(c) for c in checks])


def read_group(name):
    with h5py.File(tracker_file(), "r") as f:
        g = f[name]
        data = {}
        for col in json.loads(g.attrs["columns"]):
            ds = g[col]
            data[col] = ds.asstr()[()] if h5py.check_string_dtype(ds.dtype) else ds[()]
        return pd.DataFrame(data)


def group_info(name):
    """Time written, settings and checks of one group, None when the group is missing."""
    if not os.path.isfile(tracker_file()):
        return None
    with h5py.File(tracker_file(), "r") as f:
        if name not in f:
            return None
        a = f[name].attrs
        return dict(written=float(a["written"]), written_at=str(a["written_at"]), version=str(a["version"]),
                    settings=json.loads(a["settings"]), checks=json.loads(a["checks"]), rows=len(f[name][json.loads(a["columns"])[0]]))


def status():
    """Recorded, blank or stale for every step and the mapper."""
    out, latest = [], 0.0
    for step in STEPS:
        info = group_info(_group(step))
        if info is None:
            out.append((step, "blank", "", 0))
            continue
        state = "stale" if info["written"] < latest else "recorded"
        latest = max(latest, info["written"])
        out.append((step, state, info["written_at"], info["rows"]))
    info = group_info("mapper")
    out.append(("mapper", "blank" if info is None else "stale" if info["written"] < latest else "recorded",
                "" if info is None else info["written_at"], 0 if info is None else info["rows"]))
    return out


def build_mapper():
    """One row per particle and PD with every number it reached, joined on particle ID and PD."""
    have = {s: group_info(_group(s)) is not None for s in STEPS}
    if not have["calc-distance"]:
        raise FileNotFoundError("calc_distance is not recorded")
    table = read_group("calc_distance")
    fate = table.fate.where(table.fate != "Kept", "In a PD")
    if have["manifold-analysis"]:
        ma = read_group("manifold_analysis")[["particle_id", "prd", "retained", "fate"]]
        table = table.merge(ma.rename(columns={"fate": "fate_ma"}), on=["particle_id", "prd"], how="left")
        fate = fate.where(table.fate_ma != "Trimmed", "Trimmed").where(table.fate_ma != "Kept", "Retained")
    if have["probability-landscape"]:
        pl = read_group("probability_landscape").drop(columns=["retained"])
        table = table.merge(pl.rename(columns={"fate": "fate_pl"}), on=["particle_id", "prd"], how="left")
        selected = set(group_info("probability_landscape")["settings"].get("xSelect", []))
        fate = fate.where(~((fate == "Retained") & ~table.prd.isin(selected)), "PD not selected")
        fate = fate.where(table.fate_pl.isna(), table.fate_pl)
        table["frame"] = table.frame.astype("Int64")
    if have["trajectory"]:
        tr = read_group("trajectory").drop(columns=["fate"])
        keys = ["particle_id", "prd"]
        if "frame" in table:
            tr["frame"] = tr.frame.astype("Int64")
            keys.append("frame")
        else:
            tr = tr.drop(columns=["frame"])
        table = table.merge(tr, on=keys, how="left")
        in_state = table.state.notna()
        fate = fate.where(~(fate == "Frame"), "Frame without a state").where(~in_state, "In a state")
    table["fate"] = fate
    for col in MAPPER:
        if col not in table:
            table[col] = np.nan
    table = table[MAPPER]
    for col in ["prd", "C", "place", "retained", "topo", "sorted", "frame", "state", "image"]:
        table[col] = pd.to_numeric(table[col]).fillna(-1).astype(int)
    table["movie_frames"] = table.movie_frames.fillna("").astype(str)
    return table.sort_values(["particle_id", "prd", "state"]).reset_index(drop=True)


def _print_checks(title, checks):
    if not checks:
        return
    print(title)
    head = ["Step", "Checked", "Failed", "Against", ""]
    w = [max(len(str(r[k])) for r in [head] + checks) for k in range(4)]
    for r in [head] + checks:
        print(f"  {r[0]:<{w[0]}}  {str(r[1]):<{w[1]}}  {str(r[2]):<{w[2]}}  {r[3]:<{w[3]}}  {r[4]}".rstrip())


def record(step, quiet=False):
    """Write the group of one step and rebuild the mapper. Never stops the pipeline."""
    try:
        table, checks, settings = BUILDERS[step]()
        write_group(_group(step), table, checks, settings)
    except Exception as e:  # the tracker must never break a pipeline step
        print(f"Tracker could not record {step} ({type(e).__name__} {e})")
        return False
    if not quiet:
        _print_checks(f"Tracker recorded {step} in {os.path.realpath(tracker_file())}", checks)
    try:
        write_group("mapper", build_mapper())
    except Exception as e:
        print(f"Tracker mapper not rebuilt ({type(e).__name__} {e})")
    return True


# reading tracker.h5 back

def trace_image(mapper, state, image):
    row = mapper[(mapper.state == state) & (mapper.image == image)]
    if row.empty:
        return [f"Image {image} of state {state} is not in the mapper"]
    r = row.iloc[0]
    return [f"Image {image} of state {state} back to its particle",
            f"  trajectory             image {image} of state {state}  PD {r.prd}, frame m = {r.frame}",
            f"  probability_landscape  frame m = {r.frame}  sorted s = {r.sorted} (topo {r.topo}, C = {r.C})",
            f"  probability_landscape  sorted s = {r.sorted}  retained j = {r.retained} (PosPsi1)",
            f"  manifold_analysis      retained j = {r.retained}  place i = {r.place} (posPath)",
            f"  calc_distance          place i = {r.place}  particle ID {r.particle_id} (ind)" + (", mirrored" if r.mirrored else "")]


def trace_particle(mapper, pid):
    rows = mapper[mapper.particle_id == pid]
    if rows.empty:
        return [f"Particle ID {pid} is not in the mapper"]
    out = [f"Particle ID {pid}" + (", mirrored" if rows.mirrored.iloc[0] else "")]
    for r in rows.itertuples(index=False):
        numbers = [f"{name} {getattr(r, name)}" for name in ("prd", "place", "retained", "sorted", "frame", "state", "image")
                   if getattr(r, name) >= 0]
        out.append("  " + ", ".join(numbers + [r.fate]))
    return out


def trace_prd(mapper, prd):
    rows = mapper[mapper.prd == prd].sort_values(["frame", "place"])
    if rows.empty:
        return [f"PD {prd} is not in the mapper"]
    out = [f"PD {prd}, {len(rows)} rows", f"  {'Place':>6} {'Retained':>9} {'Sorted':>7} {'Frame':>6} {'Particle ID':>12} {'State':>6} {'Image':>6}  Fate"]
    for r in rows.itertuples(index=False):
        out.append(f"  {r.place:>6} {r.retained:>9} {r.sorted:>7} {r.frame:>6} {r.particle_id:>12} {r.state:>6} {r.image:>6}  {r.fate}")
    return out


def op(fill=False, verify=False, trace_images=(), trace_particles=(), trace_prds=(), csv=False):
    """utility tracker. Report the steps, fill blank or stale ones, cross verify, trace and export."""
    if fill:
        for step, state, _, _ in status()[:-1]:
            if state != "recorded":
                print(f"Filling {step}")
                record(step)
    rows = status()
    print(f"Tracker file        {os.path.realpath(tracker_file())}")
    for step, state, when, n_rows in rows:
        print(f"  {step:<22} {state:<9} {when:<20} {n_rows if n_rows else ''}")

    if verify:
        for step, state, _, _ in rows[:-1]:
            if state == "blank":
                continue
            try:
                table, checks, _ = BUILDERS[step]()
                stored = read_group(_group(step))
                same = len(table) == len(stored) and all(np.array_equal(np.asarray(table[c]).astype(str), np.asarray(stored[c]).astype(str))
                                                         for c in table.columns)
                checks.append(_check("Recorded group", f"{len(stored)} rows", 0 if same else 1, "files now",
                                     "matches" if same else "differs from the files now"))
            except Exception as e:
                checks = [_check("Recorded group", "", 1, "files now", f"{type(e).__name__} {e}")]
            _print_checks(f"\nVerify {step}", checks)

    if trace_images or trace_particles or trace_prds or csv:
        if group_info("mapper") is None:
            print("No mapper in tracker.h5. Record or fill the steps first.")
            return
        mapper = read_group("mapper")
        lines = []
        for state, image in trace_images:
            lines += [""] + trace_image(mapper, state, image)
        for pid in trace_particles:
            lines += [""] + trace_particle(mapper, pid)
        for prd in trace_prds:
            lines += [""] + trace_prd(mapper, prd)
        if lines:
            print("\n".join(lines))
        if csv:
            path = os.path.join(params.out_dir, "particle_index.csv")
            mapper.replace(-1, "").to_csv(path, index=False)
            print(f"\nCSV file            {os.path.realpath(path)} ({len(mapper)} rows)")
