"""
atst_v5_integration.py -- frozen ATST-Frame features inside / next to the V5 pipeline.

HOW TO USE: run it INSIDE the V5 kernel, after the V5 notebook's setup cells (1-26, all RUN_* flags off, so that
VALN, VALS, SYNTH_*, TRAIN_ROWS, tracks_v5, V5Dataset, v5_losses, select_decoder ... exist):

        %run -i atst_v5_integration.py

`-i` executes it in the notebook namespace, so it uses (and does not copy) V5's definitions. Importing it does NO work:
every stage is a function you call explicitly (see the bottom of the file and RUN_GUIDE.md).

Nothing here overwrites a V5 artifact: every file carries a unique tag (ATST_CACHE_TAG / ATST_RUN_TAG) and checkpoints
store the sha256 of the ATST weights plus the V5 state hash.

NOT EXECUTED in the authoring sandbox (no GPU, torch could not be installed there). It is exercised by
`atst_selftest_v5()` below, which you must run (SMOKE) before any cache build or training.
"""
import os, sys, json, math, time, pickle, hashlib, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

_HERE = os.path.dirname(os.path.abspath(globals().get("__file__", "."))) if "__file__" in globals() else os.getcwd()
for _p in (_HERE, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import atst_common as AC
import vaani_track_ensemble as VE

# ---- fail early, with a clear message, if the V5 namespace is not loaded --------------------------------------------
_NEED = ("VaaniSEDV5", "V5Dataset", "collate_v5", "v5_losses", "tracks_v5", "selection_J", "select_decoder", "decode_v5",
         "make_tier_sampler", "update_teacher", "state_hash", "write_submission", "TRAIN_ROWS", "VALN", "VALS", "SYNTH_TRAIN",
         "SYNTH_VAL", "CFG_V5", "BASE_DIR", "V5_DIR", "SUB_DIR", "TEST_FILES", "load_test_wav", "load_test_beats",
         "beats_rows_for_window", "canonical_tags", "mel_v5", "weak_augment", "TemporalRefinement", "log")
_missing = [n for n in _NEED if n not in globals()]
if _missing:
    raise RuntimeError(f"V5 definitions missing from the kernel: {_missing}. Run the V5 setup cells first (flags off).")

# =====================================================================================================================
# configuration (edit here; tags are the ONLY thing that names artifacts)
# =====================================================================================================================
ATST_CKPT = f"{BASE_DIR}/atst_as2M.ckpt"                 # weak-label AudioSet-2M ATST-Frame (see atst_common.PROVENANCE)
ATST_MEL_VARIANT = "pretrain"                            # "pretrain" (audiossl) | "sed" (ATST-SED: extra clamp -50..80 dB)
ATST_CACHE_TAG = "atst_cache_v1"
ATST_RUN_TAG = "v5c_atst_fusion_e1"                      # experiment B3 (BEATs + mel + short-mel + gated ATST residual)
ATST_ENS_TAG = "ens_v5a_v5c_atst_v1"
ATST_CACHE_DIR = f"{BASE_DIR}/{ATST_CACHE_TAG}" + ("_smoke" if globals().get("SMOKE", False) else "")
ATST_EXPECTED_SHA256 = None                              # paste the sha256 printed by the first load to pin the file
ATST_ROW_MASK_PROB = 0.0                                 # ATST token masking aug; 0 = off (keep experiments isolated)
ATST_SMOKE_MAX_NEW = 24                                  # entries per kind built in SMOKE mode

ATST_DIM = AC.EMBED_DIM
CANON_TOK = int(round(W_SEC / AC.TOKEN_SEC))             # 200 tokens per 8 s window
assert CANON_TOK == 200 and int(round(W_SEC * SR)) % AC.TOKEN_SAMPLES == 0
assert PATCH_SAMPLES % AC.TOKEN_SAMPLES == 0 and PATCH_SAMPLES // AC.TOKEN_SAMPLES == 4   # V5 patch grid = 4 tokens
_META = {"sha256": None, "mel_variant": ATST_MEL_VARIANT, "n_last": 1, "chunk": AC.MAX_TOKENS, "hop": 125,
         "embed_dim": ATST_DIM}


def atst_open_cache(sha256=None, create=True):
    meta = dict(_META, sha256=sha256)
    return AC.ATSTCache(ATST_CACHE_DIR, ATST_CACHE_TAG, meta, create=create)


# =====================================================================================================================
# 1. cache building (resumable; features computed once per clip, sliced per window at train/inference time)
# =====================================================================================================================
def atst_cache_plan():
    """Returns {kind: (keys, wav_getter)} for everything V5 may touch. nat = global dataset index."""
    nat = [i for i in range(len(full_dataset)) if i not in BAD_INDICES]
    plan = {"nat": (nat, lambda i: _load_waveform(i, full_dataset, SR, WAV_CACHE_DIRS))}
    if SYNTH_TRAIN is not None:
        plan["synth_train"] = (sorted(SYNTH_TRAIN.items), lambda s: SYNTH_TRAIN.wav(s))
    if SYNTH_VAL is not None:
        plan["synth_val"] = (sorted(SYNTH_VAL.items), lambda s: SYNTH_VAL.wav(s))
    plan["test"] = ([os.path.splitext(os.path.basename(p))[0] for p in TEST_FILES],
                    None)
    return plan


def atst_estimate_storage():
    secs = 0.0
    if "DURATION_COL" in globals() and DURATION_COL is not None:
        secs += float(sum(d for i, d in enumerate(DURATION_COL) if i not in BAD_INDICES))
    if SYNTH_TRAIN is not None:
        secs += sum(float(r.get("duration", 8.0)) for r in SYNTH_TRAIN.items.values())   # 8 s assumed if the manifest has no duration
    if SYNTH_VAL is not None:
        secs += sum(float(r.get("duration", 8.0)) for r in SYNTH_VAL.items.values())
    gb = AC.estimate_storage_gb(secs)
    log(f"ATST cache estimate: {secs / 3600:.1f} h of train/val/synth audio -> {gb:.1f} GB fp16 at 40 ms "
        f"({gb / 2:.1f} GB if pooled to 80 ms); test (11 h) ~1.5 GB. Free disk is not checked automatically.")
    return gb


def atst_build_caches(subset="all", device=None, kinds=("nat", "synth_train", "synth_val", "test"), verify=True):
    """subset='probe': Gold train+val natural clips + Val-S + the first 3000 synthetic train mixtures only.
    subset='all': everything (needed for the fusion training run and for test inference).
    SMOKE mode builds ATST_SMOKE_MAX_NEW entries per kind. Safe to interrupt and re-run."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = AC.load_atst_encoder(ATST_CKPT, device=device, mel_variant=ATST_MEL_VARIANT,
                               expected_sha256=ATST_EXPECTED_SHA256)
    cache = atst_open_cache(enc.meta["sha256"])
    ex = AC.ATSTFeatureExtractor(enc, device)
    plan = atst_cache_plan()
    smoke = bool(globals().get("SMOKE", False))
    if subset == "probe":
        gold = set(tiers["gold"]) | set(VAL_GOLD)
        plan["nat"] = ([i for i in plan["nat"][0] if i in gold], plan["nat"][1])
        if "synth_train" in plan:
            plan["synth_train"] = (plan["synth_train"][0][:3000], plan["synth_train"][1])
        plan.pop("test", None)
    out = {}
    for kind in kinds:
        if kind not in plan:
            continue
        keys, getter = plan[kind]
        ck = {"synth_train": "synth", "synth_val": "synth"}.get(kind, kind)
        if kind == "test":
            getter_ = lambda cid, _m={os.path.splitext(os.path.basename(p))[0]: p for p in TEST_FILES}: load_test_wav(_m[cid])
        else:
            getter_ = getter
        out[kind] = AC.build_cache(keys, getter_, cache, ck, ex, log=log, max_new=ATST_SMOKE_MAX_NEW if smoke else None)
        if verify and not smoke:
            def n_samples_of(k, kind=kind):
                if kind == "nat":
                    return int(round(float(DURATION_COL[k]) * SR)) if DURATION_COL is not None else getter_(k).numel()
                return getter_(k).numel()
            AC.verify_cache(cache, ck, keys, n_samples_of, log=log, full=False)
    del enc, ex
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# =====================================================================================================================
# 2. dataset: V5Dataset + the ATST token rows of the SAME crop (no V5 logic duplicated)
# =====================================================================================================================
class V5ATSTDataset(V5Dataset):
    """Runs V5Dataset.__getitem__ unchanged (identical crops, labels, weak targets, RNG stream) and records the crop
    start through a wrapper around the global `beats_rows_for_window`; then slices the cached ATST tokens for exactly
    that crop. Whole-clip (<= 8 s) items recover the shift from `vspan`. atst rows are zero outside the real region."""

    def __init__(self, *a, atst_cache=None, atst_mask_prob=0.0, **kw):
        super().__init__(*a, **kw)
        self.atst_cache = atst_cache
        self.atst_mask_prob = atst_mask_prob
        self._rng = random.Random(1234)

    def _full(self, kind, key):
        if kind == "nat":
            return self.atst_cache.load("nat", int(key))
        return self.atst_cache.load("synth", key)

    def __getitem__(self, i):
        kind, key, tier = self.rows[i]
        g = V5Dataset.__getitem__.__globals__
        orig = g["beats_rows_for_window"]
        cap = {}

        def rec(beats_full, start_sample):
            cap["start"] = int(start_sample)
            return orig(beats_full, start_sample)

        g["beats_rows_for_window"] = rec
        try:
            item = super().__getitem__(i)
        finally:
            g["beats_rows_for_window"] = orig
        full = self._full(kind, key)
        if "start" in cap:
            rows, n_real = AC.atst_rows_for_window(full, cap["start"], CANON_TOK)
        else:
            k = int(round(item["vspan"][0] / BEATS_PATCH_HOP_SEC))
            assert abs(item["vspan"][0] - k * BEATS_PATCH_HOP_SEC) < 1e-6, item["vspan"]
            rows, n_real = AC.atst_rows_for_shifted_clip(full, 4 * k, CANON_TOK)
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        if self.training and self.atst_mask_prob > 0 and self._rng.random() < self.atst_mask_prob:
            w = self._rng.randint(4, 12); s = self._rng.randint(0, max(0, CANON_TOK - w)); rows[s:s + w] = 0
        assert np.isfinite(rows).all(), f"non-finite ATST rows for {kind}:{key}"
        item["atst"] = torch.from_numpy(rows)
        return item


def collate_v5_atst(batch):
    out = collate_v5(batch)
    out["atst"] = torch.stack([b["atst"] for b in batch])
    return out


# =====================================================================================================================
# 3. models
# =====================================================================================================================
class VaaniSEDV5ATST(VaaniSEDV5):
    """V5 with an ADDED frozen-ATST path (architecture change, stated explicitly):

      mode 'fuse'    x = fuse_norm( gate*BEATs + (1-gate)*mel  +  ga * P(LN(ATST)) )
                     P = Linear(768->192), ZERO-initialised -> at step 0 the network is exactly the incumbent
                     (verified by atst_selftest_v5). ga = sigmoid(Linear(3*192->192)), bias -2 (~0.12), so the
                     residual enters slowly. GRU input width is unchanged (192), so no GRU surgery is needed.
      mode 'replace' ATST replaces BEATs (bf = P(LN(ATST)), P randomly initialised): NOT a near-incumbent start.

    ATST rows (200 per 8 s window, 40 ms tokens) are linearly interpolated onto the 801-position V5 grid, exactly
    like BEATs rows are. This adds no temporal information: the real resolution stays 40 ms.
    ATST rows beyond the real region are masked with V5's gate_span."""
    uses_atst = True

    def __init__(self, n_fine, mode="fuse", atst_dim=ATST_DIM, **kw):
        super().__init__(n_fine, **kw)
        assert mode in ("fuse", "replace")
        self.mode = mode
        pd = self.proj_dim
        self.atst_ln = nn.LayerNorm(atst_dim)
        self.atst_proj = nn.Linear(atst_dim, pd)
        if mode == "fuse":
            nn.init.zeros_(self.atst_proj.weight); nn.init.zeros_(self.atst_proj.bias)
            self.atst_gate = nn.Linear(pd * 3, pd)
            nn.init.constant_(self.atst_gate.bias, -2.0)

    def new_keys(self):
        return super().new_keys() | {n for n in self.state_dict() if n.startswith("atst_")}

    def forward(self, waveform, beats, gate_span, mel=None, atst=None):
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        if atst is None:
            raise RuntimeError("VaaniSEDV5ATST needs atst rows (no silent fallback to BEATs-only)")
        a_raw = self.atst_proj(self.atst_ln(atst))
        bf = self.beats_proj(beats) if self.mode == "fuse" else a_raw
        if mel is None:
            mel = self.mel_frontend(waveform)
        fdy = self.fdy_proj(self.dfd_lka(mel.unsqueeze(1)))
        T = fdy.shape[1]
        if self.short_mel is not None:
            fdy = fdy + self.short_mel(waveform, T).to(fdy.dtype)

        def up(x):
            return F.interpolate(x.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)

        bf = self.feat_drop(up(bf))
        fdy = self.feat_drop(fdy)
        gate = self.fuse_gate(torch.cat([bf, fdy], dim=-1))
        centres = (torch.arange(T, device=waveform.device, dtype=torch.float32) + 0.5) * (W_SEC / T)
        gspan = gate_span.float()
        bvalid = ((centres[None] >= gspan[:, :1]) & (centres[None] < gspan[:, 1:2])).to(gate.dtype).unsqueeze(-1)
        gate = gate * bvalid
        mix = gate * bf + (1.0 - gate) * fdy
        if self.mode == "fuse":
            a = self.feat_drop(up(a_raw)) * bvalid
            ga = torch.sigmoid(self.atst_gate(torch.cat([bf, fdy, a], dim=-1)))
            mix = mix + ga * a
        x = self.fuse_norm(mix)
        x, _ = self.gru(x)
        x = self.temporal_refine(self.gru_norm(x))
        if self.per_cat_boundary:
            on, off = self.cat_onset_head(x), self.cat_offset_head(x)
        else:
            on, off = self.event_onset_head(x)[..., 0], self.event_offset_head(x)[..., 0]
        return {"cat_logits": self.cat_frame_head(x), "onset_logits": on, "offset_logits": off,
                "attn_logits": self.cat_attn(x.transpose(1, 2)).transpose(1, 2), "T": T}


class ATSTOnlySED(nn.Module):
    """Experiment C: standalone ATST model (no BEATs, no mel). LN -> Linear(768->192) -> [interp to 801] -> BiGRU ->
    TemporalRefinement -> 7 category tracks + ONE class-agnostic onset and offset head + MIL attention. It trains with
    V5's own masked losses (v5_losses), so the targets/tier masking are the V5 ones and the outputs are V5-compatible.
    Temporal resolution is that of the 40 ms tokens; interpolation to 801 positions is for decoder compatibility only."""
    uses_atst = True
    per_cat_boundary = False
    short_mel = None
    mel_frontend = None

    def __init__(self, atst_dim=ATST_DIM, proj_dim=192, gru_hidden=128, gru_layers=2, feat_dropout=0.2):
        super().__init__()
        self.atst_ln = nn.LayerNorm(atst_dim)
        self.proj = nn.Linear(atst_dim, proj_dim)
        self.feat_drop = nn.Dropout(feat_dropout)
        self.fuse_norm = nn.LayerNorm(proj_dim)
        self.gru = nn.GRU(proj_dim, gru_hidden, num_layers=gru_layers, batch_first=True, bidirectional=True, dropout=0.10)
        go = 2 * gru_hidden
        self.gru_norm = nn.LayerNorm(go)
        self.temporal_refine = TemporalRefinement(go)
        self.cat_frame_head = nn.Linear(go, NC)
        self.event_onset_head = nn.Linear(go, 1)
        self.event_offset_head = nn.Linear(go, 1)
        self.cat_attn = nn.Conv1d(go, NC, kernel_size=1)
        self.T_out = int(round(W_SEC * SR / 160)) + 1          # 801, the V5 grid

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, waveform, beats, gate_span, mel=None, atst=None):
        if atst is None:
            raise RuntimeError("ATSTOnlySED needs atst rows")
        T = self.T_out
        h = self.proj(self.atst_ln(atst))
        h = F.interpolate(h.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)
        centres = (torch.arange(T, device=h.device, dtype=torch.float32) + 0.5) * (W_SEC / T)
        gs = gate_span.float()
        valid = ((centres[None] >= gs[:, :1]) & (centres[None] < gs[:, 1:2])).to(h.dtype).unsqueeze(-1)
        x = self.fuse_norm(self.feat_drop(h) * valid)
        x, _ = self.gru(x)
        x = self.temporal_refine(self.gru_norm(x))
        return {"cat_logits": self.cat_frame_head(x), "onset_logits": self.event_onset_head(x)[..., 0],
                "offset_logits": self.event_offset_head(x)[..., 0],
                "attn_logits": self.cat_attn(x.transpose(1, 2)).transpose(1, 2), "T": T}


def new_atst_model(kind, cfg=None):
    cfg = cfg or CFG_ATST
    if kind in ("fuse", "replace"):
        return VaaniSEDV5ATST(n_fine=len(canonical_tags), mode=kind, mel_frontend=mel_v5, feat_dropout=cfg["feat_dropout"],
                              per_cat_boundary=cfg.get("per_cat_boundary", False), short_mel=cfg.get("short_mel", False)).to(device)
    if kind == "atst_only":
        return ATSTOnlySED(feat_dropout=cfg["feat_dropout"]).to(device)
    raise ValueError(kind)


CFG_ATST = dict(CFG_V5)          # SAME recipe as the V5-C run (updates, lr, shares, losses, seed); only the model gains a path


# =====================================================================================================================
# 4. inference: windowed tracks (copy of tracks_v5 with the extra ATST input; parity-tested against tracks_v5)
# =====================================================================================================================
@torch.no_grad()
def tracks_atst(net, wav, beats_full, atst_full, device, T_win, fps, max_batch=8):
    net.eval()
    n = wav.numel(); dur = n / SR
    starts = grid_window_starts(n)
    total_frames = int(round(dur * fps)) + T_win + 1
    nb_ = NC if getattr(net, "per_cat_boundary", False) else 1
    acc = np.zeros((total_frames, NC + 2 * nb_), np.float64); cnt = np.zeros(total_frames)
    use_atst = getattr(net, "uses_atst", False)
    if use_atst and atst_full is None:
        raise RuntimeError("model needs ATST features but none were given")
    for b0 in range(0, len(starts), max_batch):
        bs = starts[b0:b0 + max_batch]
        chunks, rows, arows, gsp = [], [], [], []
        for s0 in bs:
            ch = wav[s0:s0 + WIN_SAMPLES]; real = ch.numel()
            if real < WIN_SAMPLES:
                ch = F.pad(ch, (0, WIN_SAMPLES - real))
            r, n_real = beats_rows_for_window(beats_full, s0)
            chunks.append(ch); rows.append(torch.from_numpy(np.ascontiguousarray(r, np.float32)))
            if use_atst:
                ar, _ = AC.atst_rows_for_window(atst_full, s0, CANON_TOK)
                arows.append(torch.from_numpy(np.ascontiguousarray(ar, np.float32)))
            gsp.append((0.0, min(real / SR, n_real * BEATS_PATCH_HOP_SEC)))
        kw = {"atst": torch.stack(arows).to(device)} if use_atst else {}
        out = net(torch.stack(chunks).to(device), torch.stack(rows).to(device),
                  torch.tensor(gsp, dtype=torch.float32, device=device), **kw)
        on_, of_ = out["onset_logits"].float(), out["offset_logits"].float()
        if on_.ndim == 2:
            on_, of_ = on_[..., None], of_[..., None]
        lg = torch.cat([out["cat_logits"].float(), on_, of_], -1).cpu().numpy()
        for j, s0 in enumerate(bs):
            off = int(round(s0 / SR * fps))
            assert abs(off / fps - s0 / SR) <= 0.5 / fps + 1e-9
            end = min(off + T_win, total_frames); L = end - off
            acc[off:end] += lg[j, :L]; cnt[off:end] += 1
    vl = min(int(round(dur * fps)) + 1, total_frames)
    m = acc[:vl] / np.maximum(cnt[:vl, None], 1e-6)
    pr = 1.0 / (1.0 + np.exp(-m))
    on_p, of_p = pr[:, NC:NC + nb_], pr[:, NC + nb_:NC + 2 * nb_]
    if nb_ == 1:
        on_p, of_p = on_p[:, 0], of_p[:, 0]
    return {"cat": pr[:, :NC].astype(np.float32), "onset": on_p.astype(np.float32),
            "offset": of_p.astype(np.float32), "duration": float(dur)}


def atst_time_grid(net):
    net.eval()
    kw = {"atst": torch.zeros(1, CANON_TOK, ATST_DIM, device=device)} if getattr(net, "uses_atst", False) else {}
    with torch.no_grad():
        o = net(torch.zeros(1, WIN_SAMPLES, device=device), torch.zeros(1, CANON_T, 768, device=device),
                torch.tensor([[0.0, W_SEC]], device=device), **kw)
    assert o["T"] == 801, f"unexpected grid {o['T']}"
    return o["T"], o["T"] / W_SEC


def atst_val_store(cache):
    """RAM store of ATST tokens for Val-N (keys = str(global idx)) and Val-S (synthetic ids), fp16."""
    st = {"N": {k: cache.load("nat", int(k)) for k in VALN}, "S": {k: cache.load("synth", k) for k in VALS}}
    return st


def eval_tracks_atst(net, valset, store, T_win, fps):
    return {k: tracks_atst(net, v["wav"], v["beats"].astype(np.float32), store[k].astype(np.float32), device, T_win, fps)
            for k, v in valset.items()}


# =====================================================================================================================
# 5. training (same loop as V5's train_v5 -- AMP-skip counting, EMA, selection J, checkpoint hash -- plus ATST rows)
# =====================================================================================================================
def atst_from_ckpt(path, kind):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = new_atst_model(kind, ck["cfg"]); net.load_state_dict(ck["model"]); net.eval()
    assert state_hash(net.state_dict()) == ck["hash"], f"{path} does not match its stored hash"
    return net, ck


def train_atst(kind, run_tag, rows, total_updates, cache, init_state=None, do_eval=True, atst_sha=None):
    set_seed(SEED)
    model = new_atst_model(kind)
    if init_state is not None:
        model.init_from_v3(init_state)                 # same warm start V5-A/V5-C used; ATST path zero/new-initialised
    ema = new_atst_model(kind); ema.load_state_dict(model.state_dict()); freeze_model(ema)
    T_win, fps = atst_time_grid(model)
    ds = V5ATSTDataset(rows, full_dataset, label_table, SYNTH_TRAIN, training=True, atst_cache=cache,
                       atst_mask_prob=ATST_ROW_MASK_PROB, shift_prob=CFG_ATST["shift_prob"],
                       beats_mask_prob=CFG_ATST["beats_mask_prob"])
    n_draw = (total_updates + 50) * CFG_ATST["batch_size"] * CFG_ATST["grad_accum_steps"]
    loader = DataLoader(ds, batch_size=CFG_ATST["batch_size"], sampler=make_tier_sampler(rows, CFG_ATST["shares"], n_draw),
                        num_workers=0, collate_fn=collate_v5_atst, drop_last=True, pin_memory=(device.type == "cuda"))
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=CFG_ATST["lr"], weight_decay=CFG_ATST["weight_decay"])
    floor = CFG_ATST["min_lr"] / CFG_ATST["lr"]

    def lr_at(u):
        wu = CFG_ATST["warmup_updates"]
        if u < wu:
            return 0.1 + 0.9 * u / wu
        pr = min(1.0, (u - wu) / max(1, total_updates - wu))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * pr))

    for g in opt.param_groups:
        g["lr"] = CFG_ATST["lr"] * lr_at(0)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    accum = CFG_ATST["grad_accum_steps"]
    paths = {"student": f"{V5_DIR}/{run_tag}_student.pt", "ema": f"{V5_DIR}/{run_tag}_ema.pt"}
    for p in paths.values():
        if os.path.exists(p) and not globals().get("SMOKE", False):
            raise RuntimeError(f"{p} exists: refusing to overwrite. Use a new run tag.")
    best = {"student": -1.0, "ema": -1.0}; best_info = {}; hist = []
    store = atst_val_store(cache) if do_eval else None
    if do_eval:
        b0 = selection_J(eval_tracks_atst(model, VALN, store["N"], T_win, fps), eval_tracks_atst(model, VALS, store["S"], T_win, fps), fps)
        log(f"{run_tag} start: J={b0['J']:.4f} N={b0['N']:.4f} S={b0['S']} thr={b0['thr']}")
        hist.append({"update": 0, "ema": b0})
    upd = micro = skipped = 0; run = defaultdict(float); evals_since = 0; t0 = time.time()
    model.train(); opt.zero_grad(set_to_none=True)
    for batch in loader:
        wav = batch["waveform"].to(device, non_blocking=True)
        mel = None
        if getattr(model, "mel_frontend", None) is not None:
            with autocast(device_type="cuda", enabled=False):
                mel = weak_augment(model.mel_frontend(wav.float()))
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            out = model(wav, batch["beats"].to(device, non_blocking=True), batch["gspan"].to(device), mel=mel,
                        atst=batch["atst"].to(device, non_blocking=True))
        loss, parts = v5_losses(out, batch, CFG_ATST, device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss {({k: float(v) for k, v in parts.items()})} keys={batch['key']}")
        scaler.scale(loss / accum).backward()
        for k, v in parts.items():
            run[k] += float(v.detach())
        run["n"] += 1; micro += 1
        if micro % accum:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        prev = scaler.get_scale()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if scaler.get_scale() < prev:
            skipped += 1; continue
        upd += 1
        for g in opt.param_groups:
            g["lr"] = CFG_ATST["lr"] * lr_at(upd)
        update_teacher(model, ema, decay=CFG_ATST["ema_decay"])
        if upd % 100 == 0:
            with open(PROGRESS_LIVE_PATH, "w") as f:
                f.write(f"run={run_tag}\nupdate={upd}/{total_updates}\nskipped={skipped}\nlr={opt.param_groups[0]['lr']:.2e}\n"
                        f"loss={float(loss):.4f}\nelapsed_min={(time.time() - t0) / 60:.1f}\n")
        last = upd >= total_updates
        if (upd % CFG_ATST["eval_every"] == 0) or last:
            tr = " ".join(f"{k}={v / run['n']:.4f}" for k, v in run.items() if k != "n"); run = defaultdict(float)
            line = f"{run_tag} upd={upd} skipped={skipped} lr={opt.param_groups[0]['lr']:.2e} {tr}"
            if do_eval:
                improved = False
                for name, net in (("student", model), ("ema", ema)):
                    sc = selection_J(eval_tracks_atst(net, VALN, store["N"], T_win, fps),
                                     eval_tracks_atst(net, VALS, store["S"], T_win, fps), fps)
                    line += f" | {name}: J={sc['J']:.4f} N={sc['N']:.4f} S={sc['S'] if sc['S'] is None else round(sc['S'], 4)}"
                    if sc["J"] > best[name]:
                        best[name] = sc["J"]; improved = True; best_info[name] = {"update": upd, **sc}
                        sd = net.state_dict()
                        torch.save({"model": sd, "update": upd, "score": sc, "hash": state_hash(sd), "canonical_tags": canonical_tags,
                                    "cfg": CFG_ATST, "atst": {"kind": kind, "cache_tag": ATST_CACHE_TAG, "atst_sha256": atst_sha,
                                                              "mel_variant": ATST_MEL_VARIANT}}, paths[name])
                        line += " *saved*"
                    hist.append({"update": upd, name: sc})
                evals_since = 0 if improved else evals_since + 1
            model.train(); log(line)
            if do_eval and evals_since >= CFG_ATST["early_stop_evals"]:
                log(f"{run_tag}: early stop"); break
        if upd >= total_updates:
            break
    if upd < total_updates and evals_since < CFG_ATST["early_stop_evals"]:
        raise RuntimeError(f"loader exhausted early at {upd}/{total_updates}")
    with open(f"{V5_DIR}/{run_tag}_history.json", "w") as f:
        json.dump({"best": best_info, "hist": hist, "updates": upd, "skipped": skipped}, f, default=float)
    log(f"{run_tag} done: {upd} updates ({skipped} AMP-skipped) in {(time.time() - t0) / 60:.1f} min; best {best_info}")
    return paths, best_info


# =====================================================================================================================
# 6. evaluation artifacts, test tracks (float32), submission -- same file schema as V5 so the ensemble code can consume them
# =====================================================================================================================
def atst_pick_checkpoint(run_tag):
    c = []
    for name in ("ema", "student"):
        p = f"{V5_DIR}/{run_tag}_{name}.pt"
        if os.path.exists(p):
            c.append((torch.load(p, map_location="cpu", weights_only=False)["score"]["J"], name, p))
    if not c:
        raise RuntimeError(f"no checkpoint for {run_tag}")
    c.sort(reverse=True); return c[0][2]


def atst_eval_and_decoder(kind, run_tag, cache, select=False):
    """Writes <tag>_val_tracks.pkl (+ <tag>_decoder.json). select=False stores the FIXED V5-A decoder (primary
    comparison); select=True runs V5's own select_decoder on this model (secondary: a decoder effect, report separately)."""
    ck_path = atst_pick_checkpoint(run_tag)
    net, ck = atst_from_ckpt(ck_path, kind)
    T_win, fps = atst_time_grid(net)
    st = atst_val_store(cache)
    TN = eval_tracks_atst(net, VALN, st["N"], T_win, fps); TS = eval_tracks_atst(net, VALS, st["S"], T_win, fps)
    if select:
        dec, search_summary = select_decoder(TN, TS, fps, run_tag)       # V5 returns (chosen, chosen_s)
    else:
        dj = json.load(open(f"{V5_DIR}/v5a_decoder.json")); dec, search_summary = dj["decoder"], None
    with open(f"{V5_DIR}/{run_tag}_val_tracks.pkl", "wb") as f:
        pickle.dump({"fps": fps, "N": TN, "S": TS, "hash": ck["hash"]}, f)
    with open(f"{V5_DIR}/{run_tag}_decoder.json", "w") as f:
        json.dump({"decoder": dec, "search_summary": search_summary, "checkpoint": ck_path, "hash": ck["hash"],
                   "decoder_source": "searched" if select else "fixed v5a decoder"}, f, indent=1)
    return ck_path


def atst_test_tracks32(kind, run_tag, cache, ck_path=None):
    """float32 test tracks (no fp16 round trip) -> <tag>_test_tracks32.pkl. Requires the test ATST + BEATs caches."""
    p32 = f"{V5_DIR}/{run_tag}_test_tracks32.pkl"
    if os.path.exists(p32):
        return p32
    ck_path = ck_path or atst_pick_checkpoint(run_tag)
    net, ck = atst_from_ckpt(ck_path, kind)
    T_win, fps = atst_time_grid(net)
    out = {}
    for pth in TEST_FILES:
        cid = os.path.splitext(os.path.basename(pth))[0]
        w = load_test_wav(pth)
        out[cid] = tracks_atst(net, w, load_test_beats(cid, w.numel()), cache.load("test", cid).astype(np.float32), device, T_win, fps)
    assert len(out) == len(TEST_FILES)
    with open(p32, "wb") as f:
        pickle.dump({"fps": fps, "tracks": out, "hash": ck["hash"]}, f)
    return p32


def ensure_test_tracks32_v5(tag):
    """float32 test tracks of an EXISTING V5 run (v5a / v5c_smel) for the ensemble; regenerated from its checkpoint."""
    p32 = f"{V5_DIR}/{tag}_test_tracks32.pkl"
    if os.path.exists(p32):
        return p32
    dj = json.load(open(f"{V5_DIR}/{tag}_decoder.json"))
    net, ck = v5_from_ckpt(dj["checkpoint"]); assert ck["hash"] == dj["hash"]
    T_win, fps = model_time_grid(net, device)
    out = {}
    for pth in TEST_FILES:
        cid = os.path.splitext(os.path.basename(pth))[0]; w = load_test_wav(pth)
        out[cid] = tracks_v5(net, w, load_test_beats(cid, w.numel()), device, T_win, fps)
    with open(p32, "wb") as f:
        pickle.dump({"fps": fps, "tracks": out, "hash": ck["hash"]}, f)
    return p32


def atst_write_submission(run_tag, zip_name=None):
    """Decode the float32 test tracks with the decoder stored for this tag; hard schema/coverage checks in write_submission."""
    t = pickle.load(open(f"{V5_DIR}/{run_tag}_test_tracks32.pkl", "rb"))
    dj = json.load(open(f"{V5_DIR}/{run_tag}_decoder.json")); assert dj["hash"] == t["hash"]
    preds, durs = {}, {}
    for cid, tr in t["tracks"].items():
        preds[cid] = decode_v5(tr, dj["decoder"], t["fps"]); durs[cid] = tr["duration"]
    smoke = bool(globals().get("SMOKE", False))
    zp = write_submission(preds, SUB_DIR, zip_name or f"submission_{run_tag}{'_SMOKE' if smoke else ''}.zip", durs,
                          expected=None if smoke else EXPECTED_TEST_CLIPS)
    ids = {os.path.splitext(os.path.basename(p))[0] for p in TEST_FILES}
    assert set(preds) == ids, "submission clip ids differ from the test files"
    return zp


def atst_verify_zip(zip_path, expected_ids=None):
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist(); assert names == ["predictions.jsonl"], names
        lines = [json.loads(l) for l in z.read("predictions.jsonl").decode().splitlines() if l.strip()]
    ids = [l["clip_id"] for l in lines]
    assert len(ids) == len(set(ids)), "duplicate clip ids"
    if expected_ids is not None:
        assert set(ids) == set(expected_ids) and len(ids) == len(expected_ids), "coverage differs from test set"
    for l in lines:
        assert set(l) == {"clip_id", "events"}
        for e in l["events"]:
            assert set(e) == {"onset", "offset"} and 0 <= e["onset"] < e["offset"]
    print(f"zip OK: {zip_path}: {len(ids)} clips, {sum(len(l['events']) for l in lines)} events")
    return len(ids)


# =====================================================================================================================
# 7. Experiment A (track ensemble) and candidate comparison: thin wrappers around vaani_track_ensemble
# =====================================================================================================================
def atst_ensemble_experiment(tag_a="v5a", tag_b="v5c_smel", write_zip=False, control="best", fixed_decoder_tag="v5a", v5_dir=None):
    """Primary decoder = the V5-A decoder for BOTH members (the 2x2 protocol: model effect only).
    v5_dir: read the REAL v5_work pickles even when the kernel runs in SMOKE (whose V5_DIR is v5_work_smoke)."""
    ns = dict(globals())
    if v5_dir:
        ns["V5_DIR"] = v5_dir
    dec = json.load(open(f"{ns['V5_DIR']}/{fixed_decoder_tag}_decoder.json"))["decoder"]
    if write_zip:
        for t in (tag_a, tag_b):
            if t in ("v5a", "v5c_smel") and not os.path.exists(f"{ns['V5_DIR']}/{t}_test_tracks32.pkl"):
                globals()["V5_DIR"], _old = ns["V5_DIR"], globals()["V5_DIR"]
                try:
                    ensure_test_tracks32_v5(t)
                finally:
                    globals()["V5_DIR"] = _old
    return VE.run_ensemble_experiment(ns, tag_a, tag_b, ens_tag=ATST_ENS_TAG, decoders={"A": dec, "B": dec},
                                      primary="A", control=control, write_zip=write_zip, log=log)


# =====================================================================================================================
# 8. frozen-feature probe (go/no-go BEFORE the 2.5 h fusion run): identical small head on BEATs vs ATST vs both
# =====================================================================================================================
def atst_probe(cache, n_updates=1500, batch=16, n_boot=100, seed=7):
    """Trains the same Linear->BiGRU head on (a) BEATs rows, (b) ATST rows, (c) both, on Gold+synthetic strong rows only,
    and compares frame-level macro average precision on all Val-N clips (first 8 s). Threshold-free, so far less noisy than
    event F1. It measures feature quality for this task, NOT the gain inside V5: a pass is necessary, not sufficient."""
    from sklearn.metrics import average_precision_score
    set_seed(seed)
    rows = [r for r in TRAIN_ROWS if r[2] in (T_GOLD, T_SYNTH) and
            cache.has("nat" if r[0] == "nat" else "synth", r[1])]
    assert len(rows) > 500, f"only {len(rows)} probe rows have ATST features; run atst_build_caches('probe') first"
    log(f"probe rows: {len(rows)} (gold {sum(r[2] == T_GOLD for r in rows)}, synth {sum(r[2] == T_SYNTH for r in rows)})")

    class Head(nn.Module):
        def __init__(self, din):
            super().__init__(); self.p = nn.Linear(din, 128); self.g = nn.GRU(128, 64, batch_first=True, bidirectional=True); self.o = nn.Linear(128, NC)
        def forward(self, x, T=801):
            h = F.interpolate(self.p(x).transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)
            return self.o(self.g(F.gelu(h))[0])

    def feats(batch_, which):
        b, a = batch_["beats"], batch_["atst"]
        if which == "beats":
            return F.interpolate(b.transpose(1, 2), size=801, mode="linear", align_corners=False).transpose(1, 2)
        a_ = F.interpolate(a.transpose(1, 2), size=801, mode="linear", align_corners=False).transpose(1, 2)
        if which == "atst":
            return a_
        return torch.cat([F.interpolate(b.transpose(1, 2), size=801, mode="linear", align_corners=False).transpose(1, 2), a_], -1)

    din = {"beats": 768, "atst": 768, "both": 1536}
    # fixed validation inputs: first window of each Val-N clip
    VS = atst_val_store(cache)["N"]
    keys = list(VALN)
    vb = np.stack([beats_rows_for_window(VALN[k]["beats"].astype(np.float32), 0)[0] for k in keys])
    va = np.stack([AC.atst_rows_for_window(VS[k].astype(np.float32), 0, CANON_TOK)[0] for k in keys])
    yv, vm = [], []
    for k in keys:
        n = VALN[k]["wav"].numel(); ve = min(W_SEC, n / SR)
        fr, _, _, valid, _ = build_targets_v5([[(s, e, c) for s, e, c in parse_events(label_table["NoiseSubCategoryTimeStamp"][int(k)], label_table["NoiseCategory"][int(k)])]],
                                              torch.tensor([[0.0, ve]]), 801, "cpu")
        yv.append(fr[0].numpy()); vm.append(valid[0].numpy())
    yv, vm = np.stack(yv), np.stack(vm) > 0.5
    res = {}
    for which in ("beats", "atst", "both"):
        set_seed(seed)
        head = Head(din[which]).to(device); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
        ds = V5ATSTDataset(rows, full_dataset, label_table, SYNTH_TRAIN, training=True, atst_cache=cache, beats_mask_prob=0.0)
        ld = DataLoader(ds, batch_size=batch, sampler=make_tier_sampler(rows, {T_GOLD: 0.6, T_SYNTH: 0.4}, n_updates * batch),
                        num_workers=0, collate_fn=collate_v5_atst, drop_last=True)
        head.train()
        for u, bt in enumerate(ld):
            if u >= n_updates:
                break
            fr, _, _, valid, _ = build_targets_v5(bt["events"], bt["vspan"], 801, device)
            lo = head(feats({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in bt.items()}, which))
            m = valid.unsqueeze(-1)
            loss = (F.binary_cross_entropy_with_logits(lo, fr, reduction="none") * m).sum() / (m.sum() * NC)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            P = []
            for i in range(0, len(keys), 64):
                bt = {"beats": torch.from_numpy(vb[i:i + 64]).to(device), "atst": torch.from_numpy(va[i:i + 64]).to(device)}
                P.append(torch.sigmoid(head(feats(bt, which))).cpu().numpy())
        res[which] = np.concatenate(P)
    def macro_ap(P, idx):
        aps = []
        for c in range(NC):
            y = yv[idx][..., c][vm[idx]]
            if y.sum() > 0:
                aps.append(average_precision_score(y, P[idx][..., c][vm[idx]]))
        return float(np.mean(aps))
    allidx = np.arange(len(keys)); rng = np.random.default_rng(seed)
    dist = np.array([VALN[k]["district"] for k in keys]); ud = np.unique(dist)
    out = {w: macro_ap(res[w], allidx) for w in res}
    for a, b in (("beats", "atst"), ("beats", "both")):
        d = []
        for _ in range(n_boot):
            idx = np.concatenate([np.where(dist == u)[0] for u in rng.choice(ud, len(ud), replace=True)])
            d.append(macro_ap(res[b], idx) - macro_ap(res[a], idx))
        out[f"{b}-{a}"] = (out[b] - out[a], (float(np.percentile(d, 5)), float(np.percentile(d, 95))))
    log(f"PROBE frame macro-AP on Val-N: BEATs {out['beats']:.4f} | ATST {out['atst']:.4f} | both {out['both']:.4f} | "
        f"both-BEATs {out['both-beats'][0]:+.4f} CI90 {out['both-beats'][1]} | ATST-BEATs {out['atst-beats'][0]:+.4f} CI90 {out['atst-beats'][1]}")
    log("PROBE go/no-go: proceed to the fusion run only if 'both - BEATs' > 0 with the CI90 lower bound > 0 "
        "(ATST adds information beyond BEATs); otherwise stop and keep the ensemble route.")
    return out


# =====================================================================================================================
# 9. in-kernel self-tests (run in SMOKE before anything expensive)
# =====================================================================================================================
class _FakeCache:
    """Deterministic fake ATST store: value[t, 0] = t (token index), so slicing offsets can be asserted exactly."""
    def __init__(self, lengths):
        self.lengths = lengths
    def has(self, kind, key):
        return True
    def load(self, kind, key):
        a = np.zeros((self.lengths(kind, key), ATST_DIM), np.float16)
        a[:, 0] = np.arange(a.shape[0]) % 2048
        return a


def atst_selftest_v5():
    """Raises on the first failure. Needs only the V5 kernel (uses a fake cache; no ATST weights, no GPU)."""
    dev = device      # NEVER .to(cpu): mel_v5 is a shared module and would be moved off the GPU for later training
    # 0. the dataset hook only works if both classes share one globals dict
    assert V5ATSTDataset.__getitem__.__globals__ is V5Dataset.__getitem__.__globals__, "run with `%run -i` in the V5 kernel"

    # 1. dataset parity + alignment (same seed -> same crop, labels, BEATs rows; ATST rows = same window)
    def toks(kind, key):
        n = _load_waveform(int(key), full_dataset, SR, WAV_CACHE_DIRS).numel() if kind == "nat" else SYNTH_TRAIN.wav(key).numel()
        return AC.atst_T_expected(max(n, SR))
    fc = _FakeCache(toks)
    rows = [r for r in TRAIN_ROWS][:: max(1, len(TRAIN_ROWS) // 24)][:24]
    base = V5Dataset(rows, full_dataset, label_table, SYNTH_TRAIN, training=True)
    ours = V5ATSTDataset(rows, full_dataset, label_table, SYNTH_TRAIN, training=True, atst_cache=fc)
    n_long = n_short = 0
    for i in range(len(rows)):
        random.seed(100 + i); np.random.seed(100 + i); torch.manual_seed(100 + i); a = base[i]
        random.seed(100 + i); np.random.seed(100 + i); torch.manual_seed(100 + i); b = ours[i]
        for k in ("waveform", "beats", "z", "zmask"):
            assert torch.equal(a[k], b[k]), f"row {i}: '{k}' differs from V5Dataset"
        assert a["events"] == b["events"] and a["gspan"] == b["gspan"] and a["vspan"] == b["vspan"]
        r = b["atst"].numpy()
        assert r.shape == (CANON_TOK, ATST_DIM) and np.isfinite(r).all()
        # alignment: the first real token index must equal (crop start)/640 samples; for shifted clips tokens start at 4k
        first = int(np.argmax(r[:, 0] > 0)) if (r[:, 0] > 0).any() else None
        vs = b["vspan"][0]
        if vs > 0:   # whole clip placed after k zero patches
            n_short += 1; k4 = int(round(vs / BEATS_PATCH_HOP_SEC)) * 4
            assert (r[:k4] == 0).all() and r[k4 + 1, 0] == 1, f"row {i}: shifted clip misaligned"
        else:
            n_long += 1
    log(f"selftest 1/6 dataset parity+alignment OK on {len(rows)} rows ({n_long} unshifted/long, {n_short} shifted)")

    # 2. zero-init: fusion model == incumbent at step 0; the new path trains
    set_seed(0)
    base_net = VaaniSEDV5(n_fine=len(canonical_tags), mel_frontend=mel_v5, short_mel=True).to(dev).eval()
    fus = VaaniSEDV5ATST(n_fine=len(canonical_tags), mode="fuse", mel_frontend=mel_v5, short_mel=True).to(dev)
    miss, unexp = fus.load_state_dict(base_net.state_dict(), strict=False)
    assert set(miss) == {n for n in fus.state_dict() if n.startswith("atst_")} and not unexp, (miss, unexp)
    fus.eval()
    wav = (torch.randn(2, WIN_SAMPLES) * 0.05).to(dev); bt = torch.randn(2, CANON_T, 768).to(dev)
    gs = torch.tensor([[0.0, 8.0], [0.0, 5.0]]).to(dev); at = torch.randn(2, CANON_TOK, ATST_DIM).to(dev)
    with torch.no_grad():
        o0 = base_net(wav, bt, gs); o1 = fus(wav, bt, gs, atst=at)
    for k in ("cat_logits", "onset_logits", "offset_logits", "attn_logits"):
        d = float((o0[k] - o1[k]).abs().max()); assert d < 1e-5, f"{k}: fusion != incumbent at init (max diff {d})"
    log("selftest 2/6 zero-init equivalence OK (max |diff| < 1e-5)")
    fus.train()
    out = fus(wav, bt, gs, atst=at, mel=None)
    loss = out["cat_logits"].pow(2).mean() + out["onset_logits"].pow(2).mean()
    loss.backward()
    g = fus.atst_proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0, "no gradient reaches atst_proj"
    assert all(torch.isfinite(p.grad).all() for p in fus.parameters() if p.grad is not None)
    log("selftest 3/6 gradient flow OK (atst_proj receives finite non-zero gradient at init)")
    with torch.no_grad():
        fus.eval(); fus.atst_proj.weight.normal_(0, 0.05)
        o2 = fus(wav, bt, gs, atst=at)
    assert float((o2["cat_logits"] - o0["cat_logits"]).abs().max()) > 1e-4, "ATST rows have no effect once the projection is non-zero"

    # 4. tracks_atst == tracks_v5 for a plain V5 net (the copied windowing must not change anything)
    w20 = torch.randn(int(20.3 * SR)) * 0.05
    bf20 = torch.randn(beats_T_expected(w20.numel()), 768)
    T_win, fps = model_time_grid(base_net, dev)
    r_v5 = tracks_v5(base_net, w20, bf20.numpy(), dev, T_win, fps)
    r_me = tracks_atst(base_net, w20, bf20.numpy(), None, dev, T_win, fps)
    for k in ("cat", "onset", "offset"):
        assert np.allclose(r_v5[k], r_me[k], atol=1e-5), f"tracks_atst differs from tracks_v5 on '{k}'"
    af20 = np.random.randn(AC.atst_T_expected(w20.numel()), ATST_DIM).astype(np.float32)
    r_f = tracks_atst(fus, w20, bf20.numpy(), af20, dev, T_win, fps)
    assert r_f["cat"].shape == r_v5["cat"].shape and np.isfinite(r_f["cat"]).all()
    log(f"selftest 4/6 tracks_atst parity OK (20.3 s clip, {r_v5['cat'].shape[0]} frames, 3 windows)")

    # 5. standalone model: shapes + V5 losses run
    c = ATSTOnlySED().to(dev); oc = c(wav, bt, gs, atst=at)
    assert oc["cat_logits"].shape == (2, 801, NC) and oc["onset_logits"].shape == (2, 801) and oc["T"] == 801
    batch = {"events": [[(1.0, 2.0, 0)], []], "vspan": torch.tensor([[0.0, 8.0], [0.0, 5.0]]), "tier": torch.tensor([T_GOLD, T_SYNTH]),
             "z": torch.zeros(2, NC), "zmask": torch.ones(2, NC)}   # v5_losses moves these to dev itself
    l, parts = v5_losses(oc, batch, CFG_ATST, dev); assert torch.isfinite(l)
    log("selftest 5/6 ATSTOnlySED shapes + v5_losses OK")

    # 6. fixed-time-grid assertions
    assert atst_time_grid(c)[0] == 801 and atst_time_grid(fus)[0] == 801
    log("selftest 6/6 time grid 801 OK. ALL ATST/V5 SELF-TESTS PASSED")
    return True


# =====================================================================================================================
# 10. stage drivers (call these from notebook cells; nothing runs on import)
# =====================================================================================================================
def atst_stage_train_fusion(kind="fuse", run_tag=ATST_RUN_TAG, updates=None):
    """B3 (kind='fuse') / B2 (kind='replace') / C (kind='atst_only'). Warm start = best_stage2 (as V5-A and V5-C)."""
    cache = atst_open_cache(create=False)
    needed = [("nat" if r[0] == "nat" else "synth", r[1]) for r in TRAIN_ROWS] + [("nat", int(k)) for k in VALN] + [("synth", k) for k in VALS]
    miss = [x for x in needed if not cache.has(*x)]
    if miss:
        raise RuntimeError(f"{len(miss)} ATST cache entries missing (first {miss[:5]}): run atst_build_caches('all')")
    sha = json.load(open(os.path.join(ATST_CACHE_DIR, "meta.json")))["sha256"]
    v3 = torch.load(f"{BASE_DIR}/best_stage2.pt", map_location="cpu", weights_only=False)
    assert v3["canonical_tags"] == canonical_tags
    init = v3["model"] if kind in ("fuse", "replace") else None
    paths, best = train_atst(kind, run_tag, TRAIN_ROWS, updates or (30 if globals().get("SMOKE") else CFG_ATST["total_updates"]),
                             cache, init_state=init, atst_sha=sha)
    return paths, best
