"""
test_atst_common.py -- run locally:   python test_atst_common.py [--audiossl C:/Users/231b258/audiossl] [--ckpt C:/Users/231b258/atst_as2M.ckpt]

Synthetic tests (no real weights needed) + optional equivalence against audiossl's own FrameAST + optional smoke test of the
real checkpoint. NOTE: written in a sandbox where torch could not be installed, so this file has not been executed there;
the numpy-only parts (chunk plan, row slicing, alignment, storage) were verified separately. Run it before building caches.
"""
import argparse, os, sys, tempfile, types
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atst_common as AC

torch.manual_seed(0)


def rand_encoder(seed=0):
    torch.manual_seed(seed)
    net = AC.FrameASTMin()
    for p in net.parameters():
        torch.nn.init.normal_(p, 0, 0.02) if p.ndim > 1 else torch.nn.init.zeros_(p)
    net.eval()
    return net


def test_mel_per_example_floor():
    m = AC.ATSTMel("pretrain")
    a, b = torch.randn(1, 32000) * 0.3, torch.randn(1, 32000) * 0.003
    batch = m(torch.cat([a, b]))
    assert torch.allclose(batch[0], m(a)[0], atol=1e-5) and torch.allclose(batch[1], m(b)[0], atol=1e-5), \
        "top_db floor leaked across batch items"
    assert batch.shape == (2, 1, 64, 1 + 32000 // 160)
    print("ok mel: per-example floor, shape", tuple(batch.shape))


def test_zscore_shifts_levels():
    m = AC.ATSTMel("pretrain"); x = torch.randn(1, 64000) * 0.01            # quiet clip, std 0.01
    z = (x - x.mean()) / (x.std() + 1e-5)                                   # what the old notebook's prep_wav did
    d = float(m(z).mean() - m(x).mean())
    print(f"info: per-clip z-scoring moves the MinMax-normalised mel mean by {d:+.3f} (range of the scale is 2.0)")
    assert abs(d) > 0.1


def test_padding_mask_equivalence():
    net = rand_encoder(); mel = AC.ATSTMel()(torch.randn(2, 16000 * 6) * 0.1)      # (2,1,64,601) -> 150 tokens
    short = mel[:1, :, :, :4 * 100]                                               # first item cut to 100 tokens
    alone = net(short, torch.tensor([100]))
    padded = torch.cat([torch.nn.functional.pad(short, (0, 4 * 50)), mel[1:]], 0)
    both = net(padded, torch.tensor([100, 150]))
    assert torch.allclose(alone[0], both[0, :100], atol=1e-4), float((alone[0] - both[0, :100]).abs().max())
    print("ok padded batch == unbatched (key-padding mask exact)")


def test_extractor_shapes_and_chunking():
    enc = AC.ATSTEncoder(rand_encoder(), AC.ATSTMel(), {"embed_dim": 768})
    ex = AC.ATSTFeatureExtractor(enc, "cpu")
    for sec in (0.4, 1.0, 7.9, 8.0, 10.0, 10.3, 25.7):
        w = torch.randn(int(sec * AC.SR)) * 0.05
        f = ex(w)
        exp = AC.atst_T_expected(max(w.numel(), AC.SR))
        assert f.shape == (exp, 768) and np.isfinite(f).all(), (sec, f.shape, exp)
    w = torch.randn(int(9.0 * AC.SR)) * 0.05                                      # single chunk == direct forward
    direct = enc.net(enc.mel(w[None]), torch.tensor([225]))[0].numpy()
    assert np.allclose(ex(w), direct, atol=1e-4)
    print("ok extractor: shapes for 0.4-25.7 s, single-chunk == direct")


def _write_ckpt(path, layout, sd):
    pre = {"lightning_teacher": "model.teacher.encoder.", "downstream_finetuned": "encoder.encoder.",
           "pretrainedsed_wrapper": "model.atst.", "bare": ""}[layout]
    out = {}
    for k, v in sd.items():
        k2 = k
        if layout in ("lightning_teacher", "downstream_finetuned") and k.startswith("norm_frame."):
            k2 = "norm." + k[len("norm_frame."):]       # how ATST-SED's loader expects the final norm in these layouts
        out[pre + k2] = v
    if layout == "lightning_teacher":
        out.update({"model.student.encoder." + k: v for k, v in sd.items()})   # student must be ignored
    torch.save({"state_dict": out, "hyper_parameters": {"pos_type": "cut"}}, path)


def test_loader_layouts_and_failures():
    net = rand_encoder(1); sd = net.state_dict(); x = AC.ATSTMel()(torch.randn(1, 32000) * 0.1)
    ref = net(x, torch.tensor([50]))
    with tempfile.TemporaryDirectory() as d:
        for layout in ("lightning_teacher", "downstream_finetuned", "pretrainedsed_wrapper", "bare"):
            p = os.path.join(d, layout + ".ckpt"); _write_ckpt(p, layout, sd)
            enc = AC.load_atst_encoder(p, verbose=False)
            assert enc.meta["layout"] == layout, (layout, enc.meta["layout"])
            assert torch.allclose(enc.net(x, torch.tensor([50])), ref, atol=1e-6)
            assert not any(q.requires_grad for q in enc.parameters())
        bad = {k: v for k, v in sd.items() if "blocks.5." not in k}                # missing layer -> must raise, not fall back
        p = os.path.join(d, "bad.ckpt"); _write_ckpt(p, "bare", bad)
        try:
            AC.load_atst_encoder(p, verbose=False); raise AssertionError("incomplete checkpoint was accepted")
        except RuntimeError:
            pass
        p = os.path.join(d, "junk.ckpt"); torch.save({"state_dict": {"foo.bar": torch.zeros(2)}}, p)
        try:
            AC.load_atst_encoder(p, verbose=False); raise AssertionError("junk checkpoint accepted")
        except RuntimeError:
            pass
        try:
            AC.load_atst_encoder(os.path.join(d, "bare.ckpt"), expected_sha256="0" * 64, verbose=False); raise AssertionError
        except RuntimeError:
            pass
    print("ok loader: 4 layouts load strictly and reproduce outputs; incomplete/junk/sha-mismatch raise")


def test_cache_roundtrip_and_resume():
    enc = AC.ATSTEncoder(rand_encoder(), AC.ATSTMel(), {"embed_dim": 768})
    ex = AC.ATSTFeatureExtractor(enc, "cpu")
    meta = {"sha256": "x", "mel_variant": "pretrain", "n_last": 1, "chunk": 250, "hop": 125, "embed_dim": 768}
    with tempfile.TemporaryDirectory() as d:
        c = AC.ATSTCache(d, "t1", meta)
        wavs = {i: torch.randn(int((1 + i) * 1.3 * AC.SR)) * 0.05 for i in range(4)}
        s1 = AC.build_cache(list(wavs), lambda i: wavs[i], c, "nat", ex, log=lambda *_: None, max_new=2)
        s2 = AC.build_cache(list(wavs), lambda i: wavs[i], c, "nat", ex, log=lambda *_: None)
        assert s1["new"] == 2 and s2["new"] == 2 and s2["skipped"] == 2
        AC.verify_cache(c, "nat", list(wavs), lambda k: wavs[k].numel(), log=lambda *_: None, full=True)
        try:
            AC.ATSTCache(d, "other_tag", meta); raise AssertionError("mixed tags accepted")
        except RuntimeError:
            pass
        a = c.load("nat", 3); assert a.dtype == np.float16
    print("ok cache: resume, shape/finite verification, tag pinning")


def test_audiossl_equivalence(audiossl_dir):
    if not audiossl_dir:
        print("skip audiossl equivalence (pass --audiossl)"); return
    sys.path.insert(0, audiossl_dir)
    if "fairseq" not in sys.modules:                     # random_mask imports fairseq at import time; never called here
        for n in ("fairseq", "fairseq.data", "fairseq.data.data_utils"):
            sys.modules[n] = types.ModuleType(n)
        sys.modules["fairseq.data.data_utils"].compute_mask_indices = lambda *a, **k: None
    from audiossl.methods.atstframe.audio_transformer import FrameAST_base
    ref = FrameAST_base(patch_h=64, patch_w=4, drop_path_rate=0.0).eval()
    mine = AC.FrameASTMin().eval()
    mine.load_state_dict(ref.state_dict(), strict=True)
    mel = AC.ATSTMel()(torch.randn(2, 16000 * 8) * 0.1)
    for n_last in (1, 4, 12):
        a = ref.get_intermediate_layers(mel, torch.tensor([200, 150]), n=n_last, scene=False)
        b = mine(mel, torch.tensor([200, 150]), n_last=n_last)
        assert torch.allclose(a[0], b[0], atol=1e-4) and torch.allclose(a[1, :150], b[1, :150], atol=1e-4), n_last
    print("ok equivalence with audiossl FrameAST (n_last 1/4/12, padded item). 12-layer concat width =", a.shape[-1])


def test_real_checkpoint(ckpt):
    if not ckpt:
        print("skip real checkpoint (pass --ckpt)"); return
    enc = AC.load_atst_encoder(ckpt, device="cpu", verbose=True)
    ex = AC.ATSTFeatureExtractor(enc, "cpu")
    f = ex(torch.randn(8 * AC.SR) * 0.05)
    assert f.shape == (200, 768) and np.isfinite(f).all()
    print(f"real checkpoint OK: 8 s -> {f.shape}, feature std {f.std():.3f}, sha256 {enc.meta['sha256']}")
    print("PASTE this sha256 into ATST_EXPECTED_SHA256 to pin the file.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--audiossl", default=None); ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    test_mel_per_example_floor(); test_zscore_shifts_levels(); test_padding_mask_equivalence()
    test_extractor_shapes_and_chunking(); test_loader_layouts_and_failures(); test_cache_roundtrip_and_resume()
    test_audiossl_equivalence(a.audiossl); test_real_checkpoint(a.ckpt)
    print("ALL ATST COMMON TESTS PASSED")
