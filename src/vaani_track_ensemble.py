"""
vaani_track_ensemble.py -- Experiment A: late track-level ensemble of cached V5 runs (CPU only, numpy only).

Plugs into the V5 kernel: it takes V5's own `decode_v5`, `official_combined_score`, `match_events`,
`events_to_frames` as arguments (nothing is re-implemented that the organiser scorer defines; the clip-level
statistics below are the SAME rules, and `selftest()` proves they re-aggregate to `official_combined_score`).

Inputs are the pickles V5 already writes:
    <V5_DIR>/<tag>_val_tracks.pkl      {"fps","N","S","hash"}      N/S: key -> {"cat","onset","offset","duration"}
    <V5_DIR>/<tag>_test_tracks32.pkl   {"fps","tracks","hash"}     float32 (regenerate with V5's _test_tracks32 if only
                                                                  the float16 <tag>_test_tracks.pkl exists)
Nothing here trains, searches a decoder, or writes a submission unless `write_zip=True` and the accept rule passes.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

EPS = 1e-6


# ----------------------------------------------------------------------------------------------
# blending
# ----------------------------------------------------------------------------------------------
def _logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def blend_one(ta: dict, tb: dict, w_b: float, mode: str = "prob") -> dict:
    """(1-w_b)*A + w_b*B on the common 10 ms-ish grid. Same fps, same duration, same length are REQUIRED
    (both runs use the same 8 s windows / 801-position grid), otherwise this raises instead of resampling."""
    if abs(ta["duration"] - tb["duration"]) > 1e-6:
        raise ValueError("duration mismatch between members")
    out = {"duration": ta["duration"]}
    for k in ("cat", "onset", "offset"):
        a, b = ta[k], tb[k]
        if a.shape != b.shape:
            raise ValueError(f"track '{k}' shape mismatch {a.shape} vs {b.shape}")
        if mode == "prob":
            m = (1.0 - w_b) * a.astype(np.float64) + w_b * b.astype(np.float64)
        elif mode == "logit":
            z = (1.0 - w_b) * _logit(a) + w_b * _logit(b)
            m = 1.0 / (1.0 + np.exp(-z))
        else:
            raise ValueError(mode)
        out[k] = m.astype(np.float32)
    return out


def blend_set(sa: Dict, sb: Dict, w_b: float, mode: str = "prob", keys: Optional[Sequence] = None) -> Dict:
    ks = list(keys) if keys is not None else sorted(set(sa) & set(sb))
    if keys is None and (set(sa) != set(sb)):
        raise ValueError("members cover different clips; pass keys explicitly if intended")
    return {k: blend_one(sa[k], sb[k], w_b, mode) for k in ks}


# ----------------------------------------------------------------------------------------------
# clip-level statistics that re-aggregate to the official score (needed for the paired bootstrap)
# ----------------------------------------------------------------------------------------------
def clip_stats(ref, pred, match_events: Callable, events_to_frames: Callable):
    tp, fp, fn = match_events(ref, pred)
    allev = list(ref) + list(pred)
    if not allev:
        dice = 1.0
    else:
        mt = max(off for _, off in allev) + 0.5
        r = events_to_frames(ref, mt); p = events_to_frames(pred, mt)
        t = int(r.sum() + p.sum())
        dice = 1.0 if t == 0 else 2.0 * int((r & p).sum()) / t
    return tp, fp, fn, dice


def aggregate(stats: np.ndarray) -> dict:
    """stats (n,4) = tp,fp,fn,dice per clip -> official-style micro F1 + clip-macro Dice."""
    TP, FP, FN = stats[:, 0].sum(), stats[:, 1].sum(), stats[:, 2].sum()
    p = TP / (TP + FP) if (TP + FP) else 0.0
    r = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    d = float(stats[:, 3].mean()) if len(stats) else 0.0
    return {"event_f1": f1, "precision": p, "recall": r, "segment_dice": d, "combined": f1 + d}


def decode_all(tracks: Dict, keys: Sequence, dec: dict, fps: float, decode_fn: Callable) -> Dict:
    return {k: decode_fn(tracks[k], dec, fps) for k in keys}


def stats_for(pred: Dict, refs: Dict, keys: Sequence, match_events, events_to_frames) -> np.ndarray:
    return np.array([clip_stats(refs[k], pred[k], match_events, events_to_frames) for k in keys], dtype=np.float64)


# ----------------------------------------------------------------------------------------------
# paired cluster bootstrap (resample districts / groups, not clips)
# ----------------------------------------------------------------------------------------------
def paired_cluster_bootstrap(stats_x: np.ndarray, stats_y: np.ndarray, groups: Sequence, n_boot: int = 2000,
                             seed: int = 0) -> dict:
    """Delta = combined(Y) - combined(X) on the SAME clips; groups resampled with replacement so that
    clips from one district (which share speakers/recording conditions) are not treated as independent."""
    g = np.asarray(list(groups)); ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    rng = np.random.default_rng(seed)
    base = aggregate(stats_y)["combined"] - aggregate(stats_x)["combined"]
    ds = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        ds[b] = aggregate(stats_y[ii])["combined"] - aggregate(stats_x[ii])["combined"]
    return {"delta": base, "ci90": (float(np.percentile(ds, 5)), float(np.percentile(ds, 95))),
            "p_gt0": float((ds > 0).mean()), "n_groups": int(len(ug))}


# ----------------------------------------------------------------------------------------------
# diagnostics requested in the brief
# ----------------------------------------------------------------------------------------------
def _greedy_pairs(ref, pred, tol_frac=0.20, min_tol=0.05):
    cand = []
    for ri, (ro, rf) in enumerate(ref):
        tol = max(tol_frac * (rf - ro), min_tol)
        for pi, (po, pf) in enumerate(pred):
            if abs(po - ro) <= tol and abs(pf - rf) <= tol:
                cand.append((abs(po - ro) + abs(pf - rf), ri, pi))
    mr, mp, pairs = set(), set(), []
    for _, ri, pi in sorted(cand):
        if ri not in mr and pi not in mp:
            mr.add(ri); mp.add(pi); pairs.append((ri, pi))
    return pairs


def diagnostics(pred: Dict, refs_cat: Dict, keys: Sequence, cat_names: Sequence[str], short_sec: float = 0.3) -> dict:
    """refs_cat: key -> [(start, end, cat_idx)]. Per-class + short-event recall, events/clip, empty clips,
    boundary errors of matched pairs (signed mean + |median|)."""
    hit = np.zeros(len(cat_names)); tot = np.zeros(len(cat_names)); sh_hit = sh_tot = 0
    on_err, off_err = [], []; n_ev = 0; empty = 0
    for k in keys:
        rc = refs_cat[k]; pr = list(pred[k]); n_ev += len(pr); empty += (len(pr) == 0)
        pairs = _greedy_pairs([(s, e) for s, e, _ in rc], pr)
        matched = {r for r, _ in pairs}
        for ri, (s, e, c) in enumerate(rc):
            tot[c] += 1; hit[c] += ri in matched
            if e - s < short_sec:
                sh_tot += 1; sh_hit += ri in matched
        for ri, pi in pairs:
            on_err.append(pr[pi][0] - rc[ri][0]); off_err.append(pr[pi][1] - rc[ri][1])
    oe, fe = np.array(on_err or [0.0]), np.array(off_err or [0.0])
    return {"events_per_clip": n_ev / max(1, len(keys)), "empty_clips": int(empty),
            "short_recall": sh_hit / max(1, sh_tot), "n_short": int(sh_tot),
            "class_recall": {cat_names[i]: (float(hit[i] / tot[i]) if tot[i] else None) for i in range(len(cat_names))},
            "onset_err_median_ms": float(np.median(np.abs(oe)) * 1e3), "offset_err_median_ms": float(np.median(np.abs(fe)) * 1e3),
            "onset_bias_ms": float(oe.mean() * 1e3), "offset_bias_ms": float(fe.mean() * 1e3)}


def calibration(tracks: Dict, refs_cat: Dict, keys: Sequence, fps: float, n_bins: int = 10) -> dict:
    """ECE / Brier of the union frame probability 1-prod(1-p_c) against the 'any reference event' frame label,
    on frames inside the clip (frame i is at time i/fps; label = covered by any reference event)."""
    P, Y = [], []
    for k in keys:
        cat = tracks[k]["cat"].astype(np.float64)
        pu = 1.0 - np.prod(1.0 - cat, axis=1)
        t = (np.arange(len(pu)) + 0.5) / fps
        y = np.zeros(len(pu), bool)
        for s, e, _ in refs_cat[k]:
            y |= (t >= s) & (t < e)
        P.append(pu); Y.append(y)
    P = np.concatenate(P); Y = np.concatenate(Y).astype(np.float64)
    bins = np.minimum((P * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(P[m].mean() - Y[m].mean())
    return {"ece": float(ece), "brier": float(((P - Y) ** 2).mean()), "pos_rate": float(Y.mean())}


# ----------------------------------------------------------------------------------------------
# the experiment
# ----------------------------------------------------------------------------------------------
def _load(p):
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing")
    with open(p, "rb") as f:
        return pickle.load(f)


PREDECLARED_BLENDS = (            # fixed BEFORE looking at any result: (name, w_on_B, mode)
    ("A only", 0.0, "prob"), ("B only", 1.0, "prob"),
    ("equal prob", 0.5, "prob"), ("0.65 B prob", 0.65, "prob"), ("equal logit", 0.5, "logit"))


def run_ensemble_experiment(ns: dict, tag_a: str, tag_b: str, ens_tag: str = "ens_v5a_v5c_atst_v1",
                            decoders: Optional[Dict[str, dict]] = None, primary: str = "A", control: str = "best",
                            min_gain: float = 0.01, max_s_drop: float = 0.02, p_min: float = 0.90,
                            write_zip: bool = False, n_boot: int = 2000, log: Callable = print) -> dict:
    """ns = the V5 kernel namespace (globals()). Uses: V5_DIR, SUB_DIR, VALN_REFS, VALN_HALF, VALS_REFS, VALS_HALF,
    VALN (for district), decode_v5, match_events, events_to_frames, parse_events, label_table, SYNTH_VAL,
    CAT_IDX, CATS, write_submission, EXPECTED_TEST_CLIPS, TEST_FILES.
    Decoders default to each member's own saved decoder.json; `primary` names the decoder used for the verdict."""
    V5_DIR = ns["V5_DIR"]
    va, vb = _load(f"{V5_DIR}/{tag_a}_val_tracks.pkl"), _load(f"{V5_DIR}/{tag_b}_val_tracks.pkl")
    assert abs(va["fps"] - vb["fps"]) < 1e-9, "members use different time grids"
    fps = va["fps"]
    dec = decoders or {}
    for t, nm in ((tag_a, "A"), (tag_b, "B")):
        if nm not in dec:
            dec[nm] = json.load(open(f"{V5_DIR}/{t}_decoder.json"))["decoder"]
    sha = {k: hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12] for k, d in dec.items()}
    log(f"ENS members A={tag_a} (hash {va['hash']}) B={tag_b} (hash {vb['hash']}); decoders {sha}; primary decoder={primary}")

    refsN, refsS = ns["VALN_REFS"], ns["VALS_REFS"]
    keysN = [k for k in refsN if k in va["N"] and k in vb["N"]]
    keysS = [k for k in refsS if k in va["S"] and k in vb["S"]]
    if len(keysN) != len(refsN) or len(keysS) != len(refsS):
        raise RuntimeError(f"coverage mismatch: Val-N {len(keysN)}/{len(refsN)}, Val-S {len(keysS)}/{len(refsS)}")
    halves = {h: ([k for k in ns["VALN_HALF"][h]], [k for k in ns["VALS_HALF"][h]]) for h in ("A", "B")}
    districts = [ns["VALN"][k]["district"] for k in keysN]
    refs_catN = {k: ns["parse_events"](ns["label_table"]["NoiseSubCategoryTimeStamp"][int(k)],
                                       ns["label_table"]["NoiseCategory"][int(k)]) for k in keysN}
    refs_catS = {sid: [(s, e, ns["CAT_IDX"][c]) for s, e, c in ns["SYNTH_VAL"].items[sid]["events"]] for sid in keysS}
    me, ef = ns["match_events"], ns["events_to_frames"]
    W_N, W_S = ns.get("W_NAT", 0.62), ns.get("W_SYN", 0.38)

    rows, store = {}, {}
    for name, w, mode in PREDECLARED_BLENDS:
        tN = blend_set(va["N"], vb["N"], w, mode, keysN); tS = blend_set(va["S"], vb["S"], w, mode, keysS)
        for dn, d in dec.items():
            pN = decode_all(tN, keysN, d, fps, ns["decode_v5"]); pS = decode_all(tS, keysS, d, fps, ns["decode_v5"])
            sN = stats_for(pN, refsN, keysN, me, ef); sS = stats_for(pS, refsS, keysS, me, ef)
            r = {"N": aggregate(sN), "S": aggregate(sS)}
            for h, (hn, hs) in halves.items():
                iN = [keysN.index(k) for k in hn]; iS = [keysS.index(k) for k in hs]
                r[h] = W_N * aggregate(sN[iN])["combined"] + W_S * aggregate(sS[iS])["combined"]
            r["J"] = W_N * r["N"]["combined"] + W_S * r["S"]["combined"]
            r["diag"] = diagnostics(pN, refs_catN, keysN, ns["CATS"])
            r["calib"] = calibration(tN, refs_catN, keysN, fps)
            rows[(name, dn)] = r; store[(name, dn)] = (sN, sS, pN, pS)
            log(f"ENS [{name:<11}|dec {dn}] J {r['J']:.4f} | Val-N {r['N']['combined']:.4f} (F1 {r['N']['event_f1']:.4f} "
                f"P {r['N']['precision']:.3f} R {r['N']['recall']:.3f} Dice {r['N']['segment_dice']:.4f}) halves A {r['A']:.4f} B {r['B']:.4f} "
                f"| Val-S {r['S']['combined']:.4f} | ev/clip {r['diag']['events_per_clip']:.2f} empty {r['diag']['empty_clips']} "
                f"short-recall {r['diag']['short_recall']:.3f} | ECE {r['calib']['ece']:.4f}")

    # verdict: each non-member blend vs the best single member, on the PRIMARY decoder
    members = {"A only": rows[("A only", primary)]["J"], "B only": rows[("B only", primary)]["J"]}
    ctrl = max(members, key=members.get) if control == "best" else control   # control="A only" -> candidate-vs-incumbent
    verdicts = {}
    for name, _, _ in PREDECLARED_BLENDS:
        if name == ctrl or (control == "best" and name in members):
            continue
        r, c = rows[(name, primary)], rows[(ctrl, primary)]
        dA, dB = r["A"] - c["A"], r["B"] - c["B"]; dJ = r["J"] - c["J"]; dS = r["S"]["combined"] - c["S"]["combined"]
        sN_r, sN_c = store[(name, primary)][0], store[(ctrl, primary)][0]
        bs = paired_cluster_bootstrap(sN_c, sN_r, districts, n_boot=n_boot)
        ok = dA >= min_gain and dB >= min_gain and dJ >= min_gain and dS >= -max_s_drop and bs["p_gt0"] >= p_min
        verdicts[name] = {"dJ": dJ, "dA": dA, "dB": dB, "dValS": dS, "bootstrap_valN": bs, "accept": bool(ok)}
        log(f"ENS verdict '{name}' vs best member '{ctrl}' (decoder {primary}): dJ {dJ:+.4f} dA {dA:+.4f} dB {dB:+.4f} "
            f"dValS {dS:+.4f} | Val-N paired cluster bootstrap: delta {bs['delta']:+.4f} CI90 [{bs['ci90'][0]:+.4f},{bs['ci90'][1]:+.4f}] "
            f"P(>0) {bs['p_gt0']:.2f} -> " + ("ACCEPT (still only a candidate: spend at most one LB probe)" if ok else "REJECT"))
    out = {"rows": rows, "verdicts": verdicts, "control": ctrl, "fps": fps, "decoder_sha": sha}

    if write_zip:
        acc = [n for n, v in verdicts.items() if v["accept"]]
        if not acc:
            log("ENS: no blend passed the predeclared rule -> no zip written"); return out
        best = max(acc, key=lambda n: rows[(n, primary)]["J"])
        spec = next(p for p in PREDECLARED_BLENDS if p[0] == best)
        ta = _load(f"{V5_DIR}/{tag_a}_test_tracks32.pkl"); tb = _load(f"{V5_DIR}/{tag_b}_test_tracks32.pkl")
        assert ta["hash"] == va["hash"] and tb["hash"] == vb["hash"], "test/val tracks come from different checkpoints"
        for t in (ta, tb):
            assert next(iter(t["tracks"].values()))["cat"].dtype == np.float32, "test tracks must be float32"
        assert set(ta["tracks"]) == set(tb["tracks"])
        if not ns.get("SMOKE", False):
            assert len(ta["tracks"]) == ns["EXPECTED_TEST_CLIPS"], len(ta["tracks"])
        preds, durs = {}, {}
        for cid in sorted(ta["tracks"]):
            tr = blend_one(ta["tracks"][cid], tb["tracks"][cid], spec[1], spec[2])
            preds[cid] = ns["decode_v5"](tr, dec[primary], fps); durs[cid] = tr["duration"]
        zp = ns["write_submission"](preds, ns["SUB_DIR"], f"submission_{ens_tag}_{best.replace(' ', '_')}{'_SMOKE' if ns.get('SMOKE', False) else ''}.zip", durs,
                                    expected=None if ns.get("SMOKE", False) else ns["EXPECTED_TEST_CLIPS"])
        log(f"ENS zip {zp} (blend '{best}', decoder {primary}/{sha[primary]})")
        out["zip"] = zp
    return out


# ----------------------------------------------------------------------------------------------
# self-test (pure numpy; pass the official functions from the V5 notebook)
# ----------------------------------------------------------------------------------------------
def selftest(official_combined_score, match_events, events_to_frames, decode_v5=None, seed=0):
    rng = np.random.default_rng(seed)
    refs, preds = {}, {}
    for i in range(60):
        n = rng.integers(0, 4); cur = 0.0; r = []
        for _ in range(n):
            s = cur + rng.uniform(0.1, 1.0); e = s + rng.uniform(0.05, 1.5); r.append((round(s, 3), round(e, 3))); cur = e
        p = [(round(max(0, s + rng.normal(0, 0.06)), 3), round(e + rng.normal(0, 0.06), 3)) for s, e in r if rng.random() < 0.8]
        p = [(a, b) for a, b in p if b > a]
        if rng.random() < 0.3:
            p.append((round(cur + 0.3, 3), round(cur + 0.9, 3)))
        refs[f"c{i}"] = r; preds[f"c{i}"] = sorted(p)
    off = official_combined_score(refs, preds)
    keys = sorted(refs)
    st = stats_for(preds, refs, keys, match_events, events_to_frames)
    agg = aggregate(st)
    for k in ("event_f1", "precision", "recall", "segment_dice", "combined"):
        assert abs(agg[k] - off[k]) < 1e-12, (k, agg[k], off[k])
    # blending identities
    T = 120
    ta = {"cat": rng.uniform(0.02, 0.98, (T, 7)).astype(np.float32), "onset": rng.uniform(0, 1, T).astype(np.float32),
          "offset": rng.uniform(0, 1, T).astype(np.float32), "duration": 1.2}
    tb = {k: (rng.uniform(0.02, 0.98, v.shape).astype(np.float32) if k != "duration" else v) for k, v in ta.items()}
    for mode in ("prob", "logit"):
        for k in ("cat", "onset", "offset"):
            assert np.allclose(blend_one(ta, tb, 0.0, mode)[k], ta[k], atol=1e-5)
            assert np.allclose(blend_one(ta, tb, 1.0, mode)[k], tb[k], atol=1e-5)
    m = blend_one(ta, tb, 0.5, "prob")["cat"]
    assert np.allclose(m, 0.5 * (ta["cat"] + tb["cat"]), atol=1e-6)
    z = blend_one(ta, tb, 0.5, "logit")["cat"]
    assert np.all(z >= np.minimum(ta["cat"], tb["cat"]) - 1e-5) and np.all(z <= np.maximum(ta["cat"], tb["cat"]) + 1e-5)
    try:
        blend_one(ta, dict(tb, cat=tb["cat"][:-1]), 0.5); raise AssertionError("shape mismatch not caught")
    except ValueError:
        pass
    # bootstrap sanity: identical systems -> delta 0
    g = [i % 5 for i in range(len(keys))]
    bs = paired_cluster_bootstrap(st, st, g, n_boot=200)
    assert abs(bs["delta"]) < 1e-12 and bs["p_gt0"] == 0.0
    if decode_v5 is not None:   # decoding the blended-with-itself tracks equals decoding the member
        d = dict(thr_short=0.5, thr_long=0.5, hyst=0.1, sw_short=3, sw_long=9, mindur_short=0.06, mindur_long=0.15,
                 gap_short=0.05, gap_long=0.15, snap=0.10, snap_min=0.2, nms_iou=0.95, max_events=99)
        assert decode_v5(blend_one(ta, ta, 0.37, "prob"), d, 100.125) == decode_v5(ta, d, 100.125)
    return True
