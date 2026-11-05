"""
atst_common.py -- ATST-Frame utilities for the VAANI Track-1 SED project.

Self-contained: needs only torch + torchaudio + numpy. It deliberately does NOT import
audiossl / pytorch_lightning / fairseq / einops, so there is no fairseq stub and nothing
that breaks on Windows + Python 3.13. The encoder below is a minimal re-implementation of
audiossl's `FrameAST` (MIT licence, (c) Audio-WestlakeU) restricted to what is needed for
inference: Linear patch embedding, "cut" positional embedding, pre-LN transformer blocks with
a key-padding mask, and the final `norm_frame`. `test_atst_common.py` checks it numerically
against audiossl's own `FrameAST` when a clone of audiossl is available.

PROVENANCE (verified from the primary repositories, see PROVENANCE below):
  * `atst_as2M.ckpt` is the ATST-Frame checkpoint that ATST-SED (ICASSP'24) uses: SSL
    pre-trained, then fine-tuned on AudioSet-2M WEAK (clip-level) labels. The Google-Drive
    file id given for it in ATST-SED's README is the same id PretrainedSED lists for
    "ATST-F_weak". It is NOT `ATST-F_strong_1`, which received an extra AudioSet-Strong
    (frame-level, knowledge-distilled) training stage. Do not attribute strong-checkpoint
    results to it.

Token / time conventions (all verified in code, not assumed):
  * mel: 16 kHz, n_fft=win=1024, hop=160, 64 bands, f 60-7800 Hz, `center=True` -> mel frame m
    is centred at m*10 ms; n_frames = 1 + n_samples // 160.
  * token = 64 mel bins x 4 frames (patch_h=64, patch_w=4) -> one token per 40 ms (25 tokens/s),
    n_tokens = n_frames // 4; token p is centred at 0.04*p + 0.015 s.
  * positional table has 250 usable positions (10 s). Longer audio must be chunked.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
SR = 16000
HOP = 160                                   # 10 ms mel hop
PATCH_H, PATCH_W = 64, 4                    # token = 64 mel bins x 4 frames
TOKEN_SAMPLES = PATCH_W * HOP               # 640 samples = 40 ms
TOKEN_SEC = TOKEN_SAMPLES / SR              # 0.04
TOKEN_CENTRE_SEC = (PATCH_W - 1) / 2 * (HOP / SR)   # 0.015 -> token p centre = 0.04 p + 0.015
MAX_TOKENS = 250                            # pos_embed has 1 + 250 rows (10 s)
EMBED_DIM = 768
MEL_MIN, MEL_MAX = -79.6482, 50.6842        # audiossl MinMax constants (dB)

PROVENANCE = {
    "atst_as2M.ckpt": (
        "ATST-Frame, SSL pre-trained + fine-tuned on AudioSet-2M WEAK labels "
        "(ATST-SED README: 'this checkpoint is fine-tuned by the AudioSet-2M'). Same Google-Drive "
        "id (1_xb0_n3UNbUG_pH1vLHTviLfsaSfCzxz) that PretrainedSED lists for 'ATST-F_weak'."),
    "ATST-F_strong_1.pt": (
        "ATST-F weak checkpoint + extra AudioSet-Strong training with ensemble knowledge "
        "distillation (PretrainedSED, ICASSP'25). GitHub release v0.0.1 of fschmid56/PretrainedSED. "
        "A different checkpoint with frame-level pre-training."),
    "licences": (
        "audiossl code MIT, checkpoints linked from audiossl CC BY 4.0 (audiossl LICENSE); "
        "ATST-SED code MIT; PretrainedSED code MIT (check the release page for the licence of "
        "the *_strong_1 weights before relying on them)."),
}


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def sha256_file(path: str, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atst_T_expected(n_samples: int) -> int:
    """Tokens produced for an n-sample clip (mel center=True -> 1 + n//160 frames, 4 per token)."""
    return (1 + int(n_samples) // HOP) // PATCH_W


def estimate_storage_gb(total_seconds: float, dim: int = EMBED_DIM, bytes_per: int = 2, pool: int = 1) -> float:
    """ATST cache size estimate (GB) for `total_seconds` of audio, fp16 by default."""
    tokens = total_seconds / TOKEN_SEC / pool
    return tokens * dim * bytes_per / 1e9


# --------------------------------------------------------------------------------------
# preprocessing: log-mel exactly as audiossl does it (optionally the ATST-SED clamp)
# --------------------------------------------------------------------------------------
class ATSTMel(nn.Module):
    """waveform (B, N) in [-1, 1] (RAW level, no per-clip z-scoring) -> (B, 1, 64, F).

    variant "pretrain": audiossl pre-training / embedding.py pipeline (dB with top_db=80, MinMax).
    variant "sed"     : ATST-SED / PretrainedSED downstream pipeline = the above + clamp(-50, 80) dB.
    The MinMax constants are absolute dB values, so the input level matters: do NOT standardise
    the waveform per clip (that shifts every dB value by an arbitrary clip-dependent gain).
    """

    def __init__(self, variant: str = "pretrain"):
        super().__init__()
        if variant not in ("pretrain", "sed"):
            raise ValueError(variant)
        self.variant = variant
        self.mel = torchaudio.transforms.MelSpectrogram(
            SR, f_min=60, f_max=7800, hop_length=HOP, win_length=1024, n_fft=1024, n_mels=64)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim != 2:
            raise ValueError(f"expected (B, N), got {tuple(wav.shape)}")
        with torch.autocast(device_type=wav.device.type, enabled=False):
            # (B,1,N) -> (B,1,64,F): torchaudio's top_db floor is then computed PER EXAMPLE.
            # (A 3-D (B,64,F) input would take one floor over the whole batch.)
            x = self.mel(wav.float().unsqueeze(1))
            x = self.to_db(x)
            if self.variant == "sed":
                x = x.clamp(min=-50.0, max=80.0)
            x = (x - MEL_MIN) / (MEL_MAX - MEL_MIN) * 2.0 - 1.0
        return x


# --------------------------------------------------------------------------------------
# minimal FrameAST (inference only)
# --------------------------------------------------------------------------------------
class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)      # qkv_bias=False in FrameAST_base
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, add_mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # softmax(q k^T / sqrt(d) + mask) v ; mask is additive (-1e4 on padded keys), as in audiossl
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=add_mask)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, add_mask):
        x = x + self.attn(self.norm1(x), add_mask)
        return x + self.mlp(self.norm2(x))


class _PatchEmbedLinear(nn.Module):
    """Key names mirror audiossl PatchEmbed_v2: `patch_embed.patch_embed.{weight,bias}`."""

    def __init__(self, dim: int):
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_H * PATCH_W, dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        B, _, H, Fr = mel.shape
        assert H == PATCH_H, f"mel must have {PATCH_H} bands, got {H}"
        W = Fr - Fr % PATCH_W
        # einops 'b c (h p1) (w p2) -> b (w h) (p1 p2 c)' with h=1, c=1 -> (b, w, p1*4 + p2)
        p = mel[:, 0, :, :W].reshape(B, PATCH_H, W // PATCH_W, PATCH_W).permute(0, 2, 1, 3)
        return self.patch_embed(p.reshape(B, W // PATCH_W, PATCH_H * PATCH_W))


class FrameASTMin(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, depth: int = 12, heads: int = 12, spec_w: int = 1001):
        super().__init__()
        self.embed_dim, self.depth = embed_dim, depth
        self.patch_embed = _PatchEmbedLinear(embed_dim)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))          # unused at inference, kept for strict loading
        self.pos_embed = nn.Parameter(torch.zeros(1, spec_w // PATCH_W + 1, embed_dim))
        self.blocks = nn.ModuleList([_Block(embed_dim, heads) for _ in range(depth)])
        self.norm_frame = nn.LayerNorm(embed_dim, eps=1e-6)

    @property
    def max_tokens(self) -> int:
        return self.pos_embed.shape[1] - 1

    def forward(self, mel: torch.Tensor, token_len: Optional[torch.Tensor] = None, n_last: int = 1) -> torch.Tensor:
        """mel (B,1,64,F); token_len (B,) valid tokens per item (padding is masked as keys).
        Returns the concatenation of norm_frame(x) after each of the last `n_last` blocks: (B,T,n_last*D)."""
        x = self.patch_embed(mel)
        B, T, _ = x.shape
        if T > self.max_tokens:
            raise ValueError(f"{T} tokens exceed the positional table ({self.max_tokens}); chunk the audio")
        x = x + self.pos_embed[:, 1:T + 1]
        add_mask = None
        if token_len is not None:
            pad = torch.arange(T, device=x.device)[None, :] >= token_len.to(x.device)[:, None]
            if bool(pad.any()):
                add_mask = (-10000.0 * pad[:, None, None, :].to(x.dtype))
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, add_mask)
            if self.depth - i <= n_last:
                outs.append(self.norm_frame(x))
        return torch.cat(outs, dim=-1)


# --------------------------------------------------------------------------------------
# checkpoint survey + loader (ONE authoritative path)
# --------------------------------------------------------------------------------------
# (layout name, prefix) -- ordered most specific first. Mirrors the three layouts that ATST-SED's own
# `ATST.load_atst` accepts, plus the PretrainedSED wrapper layout and a bare FrameAST state dict.
LAYOUTS = (
    ("c2f_frame_encoder", "encoder.encoder.frame_encoder."),
    ("lightning_teacher", "model.teacher.encoder."),
    ("downstream_finetuned", "encoder.encoder."),
    ("pretrainedsed_wrapper", "model.atst."),
    ("bare", ""),
)


def _safe(v, depth=0):
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    if depth > 2:
        return repr(v)[:80]
    if isinstance(v, dict):
        return {str(k): _safe(x, depth + 1) for k, x in list(v.items())[:30]}
    if isinstance(v, (list, tuple)):
        return [_safe(x, depth + 1) for x in list(v)[:30]]
    return repr(v)[:120]


def survey_checkpoint(path: str, verbose: bool = True) -> dict:
    """Open a checkpoint and report what is inside, WITHOUT building a model.
    weights_only=False is required for Lightning checkpoints (they pickle hyper-parameter objects);
    only use it on files you trust. The sha256 is recorded so the exact file is traceable."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    info = {"path": path, "size_mb": round(os.path.getsize(path) / 1e6, 1), "sha256": sha256_file(path),
            "top_level_keys": list(ck.keys())[:20] if isinstance(ck, dict) else type(ck).__name__}
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else None
    if not isinstance(sd, dict) or not sd or not all(isinstance(k, str) for k in sd):
        raise RuntimeError(f"no usable state dict in {path}; top-level: {info['top_level_keys']}")
    sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    info["n_tensors"] = len(sd)
    hist: Dict[str, int] = {}
    for k in sd:
        p = ".".join(k.split(".")[:3])
        hist[p] = hist.get(p, 0) + 1
    info["prefix_histogram"] = dict(sorted(hist.items(), key=lambda kv: -kv[1])[:20])
    counts = {name: sum(k.startswith(pref) for k in sd) for name, pref in LAYOUTS if pref}
    info["layout_counts"] = counts
    info["hyper_parameters"] = _safe(ck.get("hyper_parameters", {})) if isinstance(ck, dict) else {}
    pe = [k for k in sd if k.endswith("pos_embed")]
    info["pos_embed"] = {k: tuple(sd[k].shape) for k in pe}
    if verbose:
        print("[ATST] checkpoint survey")
        for k, v in info.items():
            print(f"   {k}: {v}")
    return info


def _strip(sd: Dict[str, torch.Tensor], layout: str, prefix: str) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        k2 = k[len(prefix):]
        if k2.startswith("teacher_module.") or "cls_token" in k2:
            continue                                   # skipped by ATST-SED's loader as well
        if layout in ("lightning_teacher", "downstream_finetuned", "c2f_frame_encoder") and k2.startswith("norm."):
            k2 = "norm_frame." + k2[len("norm."):]      # ATST-SED: encoder.norm -> norm_frame
        out[k2] = v
    return out


def remap_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], str]:
    """Pick the checkpoint layout by key counts and return the encoder-only state dict."""
    sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    counts = {name: sum(k.startswith(pref) for k in sd) for name, pref in LAYOUTS if pref}
    min_keys = 100                                      # a 12-block encoder has ~150 tensors
    cand = [(n, p) for n, p in LAYOUTS if p and counts[n] >= min_keys]
    if "c2f_frame_encoder" in [c[0] for c in cand]:
        cand = [c for c in cand if c[0] != "downstream_finetuned"]   # superset prefix: prefer the specific one
    if len(cand) > 1:
        raise RuntimeError(f"ambiguous checkpoint layout, key counts {counts}. Refusing to guess "
                           f"(student/teacher style checkpoints must be resolved explicitly).")
    if cand:
        name, prefix = cand[0]
    elif any(k.startswith("blocks.0.") for k in sd):
        name, prefix = "bare", ""
    else:
        raise RuntimeError(f"unrecognised checkpoint layout; key counts {counts}; sample keys: {list(sd)[:8]}")
    return _strip(sd, name, prefix), name


class ATSTEncoder(nn.Module):
    """mel front-end + FrameASTMin + provenance metadata."""

    def __init__(self, net: FrameASTMin, mel: ATSTMel, meta: dict):
        super().__init__()
        self.net, self.mel, self.meta = net, mel, meta


def load_atst_encoder(ckpt_path: str, device="cpu", mel_variant: str = "pretrain", freeze: bool = True,
                      expected_sha256: Optional[str] = None, verbose: bool = True) -> ATSTEncoder:
    """The single ATST loading path. Fails loudly: strict key matching, no silent fallback of any kind
    (no AST substitute, no randomly initialised layers). Returns an eval-mode, frozen encoder."""
    info = survey_checkpoint(ckpt_path, verbose=verbose)
    if expected_sha256 and info["sha256"] != expected_sha256:
        raise RuntimeError(f"sha256 mismatch: {info['sha256']} != {expected_sha256}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    enc_sd, layout = remap_state_dict(sd)
    if any(".prompt_embed" in k or k == "prompt_embed" for k in enc_sd):
        raise NotImplementedError("checkpoint uses prompt tokens (nprompt>0); not supported here")
    if any(k.startswith("patch_embed.proj.") for k in enc_sd):
        raise NotImplementedError("checkpoint uses the CNN patch embedding; only the Linear one is implemented")
    try:
        dim = int(enc_sd["pos_embed"].shape[-1]); spec_w = (int(enc_sd["pos_embed"].shape[1]) - 1) * PATCH_W
        depth = 1 + max(int(k.split(".")[1]) for k in enc_sd if k.startswith("blocks."))
    except KeyError as e:
        raise RuntimeError(f"encoder state dict lacks {e}; layout={layout}; keys: {list(enc_sd)[:8]}")
    heads = {768: 12, 384: 6}.get(dim)
    if heads is None:
        raise NotImplementedError(f"embed dim {dim}")
    net = FrameASTMin(embed_dim=dim, depth=depth, heads=heads, spec_w=spec_w)
    try:
        net.load_state_dict(enc_sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"strict load failed (layout={layout}): {str(e)[:1200]}") from e
    mel = ATSTMel(mel_variant)
    n_params = sum(p.numel() for p in net.parameters())
    meta = {"ckpt": os.path.abspath(ckpt_path), "sha256": info["sha256"], "layout": layout, "embed_dim": dim,
            "depth": depth, "n_params": n_params, "mel_variant": mel_variant,
            "token_sec": TOKEN_SEC, "max_tokens": net.max_tokens,
            "provenance": "ATST-Frame, weak-label (AudioSet-2M) fine-tuned unless the sha256 says otherwise"}
    enc = ATSTEncoder(net, mel, meta).to(device).eval()
    if freeze:
        for p in enc.parameters():
            p.requires_grad_(False)
    if verbose:
        print(f"[ATST] loaded OK: layout={layout} dim={dim} depth={depth} params={n_params / 1e6:.1f}M "
              f"mel={mel_variant} sha256={info['sha256'][:16]}...")
    return enc


# --------------------------------------------------------------------------------------
# clip -> token features (chunked for audio longer than 10 s)
# --------------------------------------------------------------------------------------
def plan_chunks(n_tok: int, chunk: int = MAX_TOKENS, hop: int = 125) -> List[Tuple[int, int, int, int]]:
    """[(start, end, keep_start, keep_end)] in token units. Keep ranges tile [0, n_tok) exactly; the
    boundary between neighbouring chunks is the midpoint of their overlap, so each kept token has
    at least ~(chunk-hop)/2 tokens of context on each side (except at the clip edges)."""
    if n_tok <= 0:
        return []
    if n_tok <= chunk:
        return [(0, n_tok, 0, n_tok)]
    if hop >= chunk:
        raise ValueError("hop must be < chunk")
    starts = list(range(0, n_tok - chunk + 1, hop))
    if starts[-1] + chunk < n_tok:
        starts.append(n_tok - chunk)
    plan = []
    for i, s in enumerate(starts):
        e = s + chunk
        k0 = 0 if i == 0 else (s + starts[i - 1] + chunk) // 2
        k1 = n_tok if i == len(starts) - 1 else (e + starts[i + 1]) // 2
        plan.append((s, e, k0, k1))
    return plan


class ATSTFeatureExtractor:
    """clip waveform -> (T, D) float32 token features on the 40 ms grid.
    The mel is computed ONCE for the whole clip (so chunk seams never see reflection padding) and then
    sliced into <=250-token chunks. Clips shorter than 1 s are zero-padded to 1 s for extraction only,
    exactly like V5's test-BEATs extraction; the returned T is atst_T_expected(max(n, SR))."""

    def __init__(self, enc: ATSTEncoder, device, n_last: int = 1, chunk: int = MAX_TOKENS, hop: int = 125,
                 batch_chunks: int = 8):
        self.enc, self.device, self.n_last = enc, torch.device(device), n_last
        self.chunk, self.hop, self.batch_chunks = chunk, hop, batch_chunks
        self.dim = enc.net.embed_dim

    @staticmethod
    def padded_len(n_samples: int) -> int:
        return max(int(n_samples), SR)

    @torch.no_grad()
    def __call__(self, wav) -> np.ndarray:
        w = torch.as_tensor(wav, dtype=torch.float32).flatten()
        if w.numel() < SR:
            w = F.pad(w, (0, SR - w.numel()))
        w = w.to(self.device)
        mel = self.enc.mel(w[None])                                  # (1,1,64,F)
        n_tok = mel.shape[-1] // PATCH_W
        assert n_tok == atst_T_expected(w.numel()), (n_tok, atst_T_expected(w.numel()))
        plan = plan_chunks(n_tok, self.chunk, self.hop)
        out = torch.zeros(n_tok, self.dim * self.n_last, device=self.device)
        for b0 in range(0, len(plan), self.batch_chunks):
            part = plan[b0:b0 + self.batch_chunks]
            width = max(e - s for s, e, _, _ in part) * PATCH_W
            fill = float(mel.min())
            batch = torch.stack([F.pad(mel[0, :, :, s * PATCH_W:e * PATCH_W], (0, width - (e - s) * PATCH_W), value=fill)
                                 for s, e, _, _ in part])            # (b,1,64,width)
            lens = torch.tensor([e - s for s, e, _, _ in part], device=self.device)
            feats = self.enc.net(batch, lens, n_last=self.n_last)    # (b,T,n_last*D)
            for j, (s, e, k0, k1) in enumerate(part):
                out[k0:k1] = feats[j, k0 - s:k1 - s]
        if self.n_last > 1:
            out = out.reshape(n_tok, self.n_last, self.dim).mean(1)
        return out.float().cpu().numpy()


# --------------------------------------------------------------------------------------
# alignment of cached token rows to V5 windows
# --------------------------------------------------------------------------------------
def atst_rows_for_window(atst_full: np.ndarray, start_sample: int, canon_t: int = 200) -> Tuple[np.ndarray, int]:
    """Token rows for an 8 s window starting at `start_sample` (must sit on the 0.16 s V5 patch grid, which
    is 4 tokens). Returns (rows[canon_t, D], n_real_rows). Zero-pads past the end of the clip."""
    if start_sample % TOKEN_SAMPLES:
        raise AssertionError(f"window start {start_sample} is not on the 40 ms token grid")
    k0 = start_sample // TOKEN_SAMPLES
    rows = atst_full[k0:k0 + canon_t]
    n_real = rows.shape[0]
    if n_real < canon_t:
        rows = np.concatenate([rows, np.zeros((canon_t - n_real, atst_full.shape[1]), atst_full.dtype)], 0)
    return rows, n_real


def atst_rows_for_shifted_clip(atst_full: np.ndarray, shift_tokens: int, canon_t: int = 200) -> Tuple[np.ndarray, int]:
    """Whole clip placed `shift_tokens` tokens into an 8 s window (V5 shift augmentation: k patches = 4k tokens)."""
    D = atst_full.shape[1]
    body = atst_full[:max(0, canon_t - shift_tokens)]
    n_real = body.shape[0]
    rows = np.concatenate([np.zeros((shift_tokens, D), atst_full.dtype), body,
                           np.zeros((canon_t - shift_tokens - n_real, D), atst_full.dtype)], 0)
    return rows, n_real


def interp_landing_time(p: int, n_rows: int = 200, w_sec: float = 8.0) -> float:
    """Time (s, within the window) at which F.interpolate(align_corners=False) puts input row p on the
    V5 grid: (p + 0.5) * w_sec / n_rows. ATST: 0.04p + 0.02 vs true token centre 0.04p + 0.015 (5 ms)."""
    return (p + 0.5) * w_sec / n_rows


# --------------------------------------------------------------------------------------
# resumable, verified feature cache
# --------------------------------------------------------------------------------------
class ATSTCache:
    """root/<kind>/<key>.npy (fp16, (T,D)); root/meta.json pins checkpoint sha256 + preprocessing.
    kinds: 'nat' (global dataset index), 'synth' (synthetic id), 'test' (clip id)."""

    def __init__(self, root: str, tag: str, meta: dict, create: bool = True):
        self.root, self.tag = root, tag
        self.meta = dict(meta, tag=tag)
        if create:
            os.makedirs(root, exist_ok=True)
        mp = os.path.join(root, "meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                old = json.load(f)
            for k in ("tag", "sha256", "mel_variant", "n_last", "chunk", "hop", "embed_dim"):
                if old.get(k) != self.meta.get(k):
                    raise RuntimeError(f"cache {root} was built with {k}={old.get(k)!r}, now {self.meta.get(k)!r}. "
                                       f"Use a new tag/folder instead of mixing features.")
        elif create:
            with open(mp, "w") as f:
                json.dump(dict(self.meta, created=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=1)

    @staticmethod
    def _name(key) -> str:
        return f"{int(key):06d}" if isinstance(key, (int, np.integer)) else str(key)

    def path(self, kind: str, key) -> str:
        return os.path.join(self.root, kind, self._name(key) + ".npy")

    def has(self, kind: str, key) -> bool:
        return os.path.exists(self.path(kind, key))

    def load(self, kind: str, key) -> np.ndarray:
        p = self.path(kind, key)
        if not os.path.exists(p):
            raise RuntimeError(f"missing ATST cache entry {p} (no live fallback by design)")
        return np.load(p)

    def save(self, kind: str, key, arr16: np.ndarray):
        p = self.path(kind, key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp.npy"
        np.save(tmp, arr16)
        os.replace(tmp, p)


def build_cache(items: Iterable, get_wav: Callable, cache: ATSTCache, kind: str, extractor: ATSTFeatureExtractor,
                key_of: Callable = lambda x: x, log: Callable = print, max_new: Optional[int] = None,
                fp16_check_every: int = 200, log_every: int = 500) -> dict:
    """Extract + store missing entries. Hard-fails on non-finite output, wrong shape or fp16 overflow.
    Resumable (existing files are skipped; use verify_cache to audit them)."""
    t0 = time.time(); n_new = n_skip = 0; worst_cos = 1.0; worst_abs = 0.0
    for i, it in enumerate(items):
        key = key_of(it)
        if cache.has(kind, key):
            n_skip += 1
            continue
        if max_new is not None and n_new >= max_new:
            break
        w = torch.as_tensor(get_wav(it), dtype=torch.float32).flatten()
        feat = extractor(w)                                         # fp32 (T,D)
        T_exp = atst_T_expected(extractor.padded_len(w.numel()))
        if feat.shape != (T_exp, extractor.dim):
            raise RuntimeError(f"{kind}:{key}: shape {feat.shape}, expected {(T_exp, extractor.dim)}")
        if not np.isfinite(feat).all():
            raise RuntimeError(f"{kind}:{key}: non-finite ATST features")
        if np.abs(feat).max() > 6e4:
            raise RuntimeError(f"{kind}:{key}: |feature| {np.abs(feat).max():.0f} would overflow fp16")
        f16 = feat.astype(np.float16)
        if fp16_check_every and n_new % fp16_check_every == 0:
            a, b = feat.reshape(-1), f16.astype(np.float32).reshape(-1)
            worst_cos = min(worst_cos, float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
            worst_abs = max(worst_abs, float(np.abs(a - b).max()))
        cache.save(kind, key, f16); n_new += 1
        if n_new % log_every == 0:
            log(f"  [atst-cache:{kind}] {n_new} new, {n_skip} skipped, {time.time() - t0:.0f}s")
    stats = {"kind": kind, "new": n_new, "skipped": n_skip, "seconds": round(time.time() - t0, 1),
             "fp16_min_cosine": worst_cos, "fp16_max_abs_err": worst_abs}
    if worst_cos < 0.9999:
        raise RuntimeError(f"fp16 storage too lossy: cosine {worst_cos:.6f}")
    log(f"  [atst-cache:{kind}] done {stats}")
    return stats


def verify_cache(cache: ATSTCache, kind: str, keys: Iterable, n_samples_of: Callable, log: Callable = print,
                 full: bool = False) -> dict:
    """Exhaustive presence check (+ shape/finite when full=True). Raises on any problem."""
    missing, bad = [], []
    keys = list(keys)
    for k in keys:
        if not cache.has(kind, k):
            missing.append(k); continue
        if full:
            a = cache.load(kind, k)
            exp = atst_T_expected(max(int(n_samples_of(k)), SR))
            if a.ndim != 2 or a.shape[0] != exp or not np.isfinite(a.astype(np.float32)).all():
                bad.append((k, a.shape, exp))
    if missing or bad:
        raise RuntimeError(f"[{kind}] ATST cache problems: {len(missing)} missing (first {missing[:5]}), "
                           f"{len(bad)} bad (first {bad[:3]})")
    log(f"  [atst-cache:{kind}] verified {len(keys)} entries" + (" (shape+finite)" if full else " (presence)"))
    return {"kind": kind, "n": len(keys)}
