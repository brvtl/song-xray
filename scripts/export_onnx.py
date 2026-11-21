"""Export the four Dettato models to ONNX for the browser (ONNX Runtime Web, WASM). CPU only.

    python scripts/export_onnx.py            # weights/ -> web/models/*.onnx + web/models/meta.json + web/models/track_head.bin

What runs where
---------------
The neural networks go to ONNX. Everything without learned weights (STFTs, filterbanks, PCEN, peak picking, clustering) runs in
JavaScript (web/js/), because `torch.stft` / `torch.fft` do not export, and because the drum features need DFTs of length 1411 and
6144. The graphs therefore start where the reference's front ends end:

    transcriber.onnx      HCQT (B, 9, 800, 288)  -> probs (B, 6, 400, 88) sigmoid, hidden (B, 400, 88, 368), stem (B, 400, 88, 368)
    repairer.onnx         one 8 s window of notes + evidence -> the tensors Repairer.candidates() turns into edits
    grouper.onnx          grouper front (X, P) + 12 note scalars + 3 transcriber taps -> instrument probs, track embedding, note state
    attn.onnx             plain softmax attention (q, k, v) -> out; the JS side calls it on query chunks, so the drum model's
                          whole-song attention never materialises a T x T matrix (long songs fit in wasm32 memory)
    drums/kit_*.onnx      the drum-kit DrumNet, split around its song attention
    drums/ex{i}_*.onnx    the three orchestral experts, split the same way

Every graph is checked against the PyTorch module at two different dynamic sizes (max abs diff printed; the script fails when a
diff is above tolerance), so a shape that tracing baked into a graph shows up here rather than in the browser.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DETTATO_WEIGHTS", os.path.join(ROOT, "weights"))

from dettato.drums import DrumTranscriber  # noqa: E402
from dettato.evidence import EvidenceRepairer  # noqa: E402
from dettato.evidence import model as EM  # noqa: E402
from dettato.grouper import Grouper  # noqa: E402
from dettato.grouper.instruments import CREDIT, INSTRUMENTS, PROGRAM  # noqa: E402
from dettato.transcriber import Transcriber  # noqa: E402
from dettato.transcriber.frontend import N_KEYS  # noqa: E402

OPSET = 17
torch.manual_seed(0)
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- helpers
def export(module, args, path, in_names, out_names, dyn):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            args,
            path,
            input_names=in_names,
            output_names=out_names,
            dynamic_axes=dyn,
            opset_version=OPSET,
            dynamo=False,
            do_constant_folding=True,
        )
    return path


def check(path, module, cases, in_names, tol, label):
    """Run the ONNX file with onnxruntime (CPU) on every case and compare with the torch module."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = min(16, os.cpu_count() or 4)
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    worst = 0.0
    for args in cases:
        with torch.no_grad():
            ref = module(*args)
        ref = ref if isinstance(ref, (tuple, list)) else (ref,)
        feeds = {n: a.numpy() for n, a in zip(in_names, args)}
        out = sess.run(None, feeds)
        for r, o in zip(ref, out):
            r = r.detach().numpy()
            if r.shape != o.shape:
                raise SystemExit(f"{label}: shape mismatch {r.shape} vs {o.shape} (a shape was baked into the graph)")
            if r.dtype.kind in "iub":
                d = float((r != o).mean())
            else:
                d = float(np.abs(r.astype(np.float64) - o).max() / (np.abs(r).max() + 1e-6))
            worst = max(worst, d)
    status = "ok" if worst <= tol else "FAIL"
    print(f"  {label:<22} {os.path.getsize(path) / 1e6:6.1f} MB   max rel diff {worst:.2e}  ({len(cases)} sizes)  {status}")
    if worst > tol:
        raise SystemExit(f"{label}: ONNX output differs from torch by {worst:.3g} (> {tol})")


# ----------------------------------------------------------------------------- transcriber
class TranscriberNet(nn.Module):
    """LaneNet.forward minus the HCQT front end (the front end runs in JS)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, hcqt):
        m = self.m
        h = m.stem(hcqt)
        stem = h
        if m.n_lanes < h.shape[3]:
            h = h[:, :, :, : m.n_lanes]
        skip, T = h, h.shape[2]
        h = m.down(h)
        for layer in m.layers:
            h = layer(h)
        h = m.up_norm(h.repeat_interleave(2, dim=2)[:, :, :T] + skip)
        for layer in m.refine:
            h = layer(h)
        h = h[:, :, :, :N_KEYS]
        probs = torch.sigmoid(m.head(h))
        return probs, h.permute(0, 2, 3, 1).contiguous(), stem[:, :, :, :N_KEYS].permute(0, 2, 3, 1).contiguous()


def export_transcriber(out_dir):
    t = Transcriber.from_pretrained("cpu")
    net = TranscriberNet(t.model).eval()
    audio = torch.randn(2, 128000) * 0.05
    with torch.no_grad():
        x = t.model.front(audio)
    path = export(
        net,
        (x[:1],),
        f"{out_dir}/transcriber.onnx",
        ["hcqt"],
        ["probs", "hidden", "stem"],
        {"hcqt": {0: "B"}, "probs": {0: "B"}, "hidden": {0: "B"}, "stem": {0: "B"}},
    )
    check(path, net, [(x[:1],), (x,)], ["hcqt"], 1e-4, "transcriber")
    return dict(thresholds=t.th, hop=t.hop, channels=t.channels)


# ----------------------------------------------------------------------------- repairer
def context_features_full(b):
    """dettato.evidence.model.context_features with one block spanning all N notes (the original's 256-note blocks only bound
    memory, and its Python loop over blocks would be frozen at the export size)."""
    on, dur, mask = b["on"], b["dur"], b["mask"]
    off = on + dur
    key = b["inst"] * 128 + b["pitch"]
    big = 10.0
    N = on.shape[1]
    mq = mask
    same_key = (key[:, :, None] == key[:, None, :]) & mask[:, None, :] & mq[:, :, None]
    same_inst = (b["inst"][:, :, None] == b["inst"][:, None, :]) & mask[:, None, :] & mq[:, :, None]
    dt = on[:, None, :] - on[:, :, None]
    ar = torch.arange(N, device=on.device)
    eye = (ar[None, :] == ar[:, None])[None]
    sk = same_key & ~eye
    prev_dt = torch.where(sk & (dt <= 0), -dt, torch.full_like(dt, big)).min(-1)
    prev_gap_on = prev_dt.values
    prev_off = torch.gather(off, 1, prev_dt.indices)
    prev_gap_off = torch.where(prev_gap_on < big, on - prev_off, torch.full_like(on, big))
    nxt = torch.where(sk & (dt > 0), dt, torch.full_like(dt, big)).min(-1)
    next_gap_on = nxt.values
    next_overlap = torch.where(next_gap_on < big, off - (on + next_gap_on), torch.full_like(on, -big))
    adt = dt.abs()
    chord = ((adt < 0.035) & same_inst).float().sum(-1)
    dens = ((adt < 0.5) & mask[:, None, :]).float().sum(-1)
    dens_inst = ((adt < 0.5) & same_inst).float().sum(-1)
    same_p_near = ((adt < 0.25) & sk).float().sum(-1)

    def sl(x):
        return torch.sign(x) * torch.log1p(x.abs() * 20)

    f = torch.stack(
        [
            sl(prev_gap_on),
            sl(prev_gap_off),
            sl(next_gap_on),
            sl(next_overlap),
            torch.log1p(chord),
            torch.log1p(dens),
            torch.log1p(dens_inst),
            torch.log1p(same_p_near),
        ],
        -1,
    )
    return f * mask[..., None]


class RepairerNet(nn.Module):
    """Repairer.forward + the tensor part of Repairer.candidates (everything before the numpy packing)."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        c = torch.tensor(EM.DUR_CENT)
        tol = torch.clamp(0.2 * c[:, None], min=0.05)
        self.register_buffer("W", ((c[None, :] - c[:, None]).abs() <= tol).float())
        self.register_buffer("cent", c)

    def forward(self, inst, pitch, vel, on, dur, mask, ev):
        b = dict(inst=inst, pitch=pitch, vel=vel, on=on, dur=dur, mask=mask, ev=ev)
        o = self.m(b)
        pk = torch.sigmoid(o["keep"].float())
        don_b, _ = EM._box_argmax(torch.softmax(o["don"].float(), -1), int(round(0.05 / EM.DON_STEP)))
        rb, _ = EM._box_argmax(torch.softmax(o["dur"].float(), -1), 3)
        dur_k = dur.clamp(min=1e-4) * torch.exp(rb.float() * EM.DR_STEP - EM.DR_MAX)
        on_k = on + (don_b.float() * EM.DON_STEP - EM.DON_MAX)
        ins = o["ins"].float()
        P, T = EM.N_INS_P, EM.N_INS_T
        pe = torch.sigmoid(ins[..., 0])
        ip = ins[..., 1 : 1 + P].argmax(-1) - EM.INS_P_MAX
        it, _ = EM._box_argmax(torch.softmax(ins[..., 1 + P : 1 + P + T], -1), int(round(0.05 / EM.INS_T_STEP)))
        idd = (torch.softmax(ins[..., 1 + P + T :], -1) @ self.W).argmax(-1)
        on_i = on[..., None] + (it.float() * EM.INS_T_STEP + EM.INS_T_MIN)
        dur_i = self.cent[idd]
        pitch_i = (pitch[..., None] + ip).clamp(0, 127)
        return pk, on_k, dur_k, pe, on_i, dur_i, pitch_i


def _repairer_case(n, rng):
    on = np.sort(rng.uniform(0, 8, n)).astype(np.float32)
    return (
        torch.zeros(1, n, dtype=torch.long),
        torch.from_numpy(rng.integers(30, 100, (1, n))),
        torch.from_numpy(rng.integers(1, 127, (1, n))),
        torch.from_numpy(on[None]),
        torch.from_numpy(rng.uniform(0.05, 2, (1, n)).astype(np.float32)),
        torch.ones(1, n, dtype=torch.bool),
        torch.from_numpy(rng.normal(0, 1, (1, n, EM.N_EV if hasattr(EM, "N_EV") else 769)).astype(np.float32)),
    )


def export_repairer(out_dir):
    rng = np.random.default_rng(0)
    orig = EM.context_features
    for n in (7, 300, 600):  # the one-block version equals the blocked original (blocks of 256)
        b = dict(zip(["inst", "pitch", "vel", "on", "dur", "mask", "ev"], _repairer_case(n, rng)))
        b["mask"][0, -3:] = False
        assert torch.allclose(orig(b), context_features_full(b), atol=1e-6), "context_features_full differs"
    EM.context_features = context_features_full  # the NoteEmbed looks the function up in the module at call time
    rep = EvidenceRepairer.from_pretrained("cpu")
    net = RepairerNet(rep.model).eval()
    names = ["inst", "pitch", "vel", "on", "dur", "mask", "ev"]
    outs = ["pk", "on_k", "dur_k", "pe", "on_i", "dur_i", "pitch_i"]
    dyn = {k: {1: "N"} for k in names + outs}
    path = export(net, _repairer_case(40, rng), f"{out_dir}/repairer.onnx", names, outs, dyn)
    # argmax outputs can flip on exact ties in random data; the float outputs are what the tolerance guards
    check(path, net, [_repairer_case(13, rng), _repairer_case(300, rng)], names, 2e-3, "repairer")
    return dict(keep_th=rep.keep_th, insert_th=rep.insert_th, L_WIN=EM.L_WIN)


# ----------------------------------------------------------------------------- grouper
class GrouperNet(nn.Module):
    """NoteNet.forward with the front end's output (X, P) as inputs (the front end runs in JS). Returns the softmaxed instrument
    probabilities, the unit track embedding and the note state h."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, X, P, notes, mask, feat):
        m = self.m
        x = m.pitch_emb(notes[..., 2].round().long().clamp(0, 127)) + m.feat(_note_scalars(notes))
        xa = m.hd(P, notes, mask) + m.prof(X, notes) + m.lanes(X, notes, mask).to(x.dtype)
        x = x + xa
        f = feat[:, :, list(m.s15_taps)].float()
        mk = mask[:, :, None, None].float()
        n = mk.sum(1, keepdim=True).clamp(min=1.0)
        mu = (f * mk).sum(1, keepdim=True) / n
        sd = (((f - mu) ** 2) * mk).sum(1, keepdim=True).div(n).sqrt().clamp(min=1e-3)
        f = m.s15_net(m.s15_ln((f - mu) / sd).flatten(2))
        x = m.ln_in(x + f)
        pos = notes[..., 0] * m.pos_scale
        for blk in m.blocks:
            x = blk(x, pos, mask)
        o = m.heads(m.ln_out(x))
        return torch.softmax(o["inst"].float(), -1), o["emb"], o["h"]


def _note_scalars(notes):
    from dettato.grouper.model import note_scalars

    return note_scalars(notes)


def _grouper_case(g, n, rng, audio_s=25.5):
    audio = torch.from_numpy((rng.normal(0, 0.05, (1, int(audio_s * 16000)))).astype(np.float32))
    X, P = g.model.front(audio)
    on = np.sort(rng.uniform(0, 24, n))
    notes = np.zeros((1, n, 12), np.float32)
    notes[0, :, 0] = on
    notes[0, :, 1] = on + rng.uniform(0.05, 1.5, n)
    notes[0, :, 2] = rng.integers(30, 96, n)
    notes[0, :, 3] = rng.integers(0, 2, n)
    notes[0, :, 4:] = rng.uniform(0, 1, (n, 8))
    mask = np.ones((1, n), bool)
    mask[0, -2:] = False
    feat = rng.normal(0, 1, (1, n, 3, 368)).astype(np.float32)
    return (X, P, torch.from_numpy(notes), torch.from_numpy(mask), torch.from_numpy(feat))


def export_grouper(out_dir):
    g = Grouper.from_pretrained("cpu")
    net = GrouperNet(g.model).eval()
    rng = np.random.default_rng(1)
    names = ["X", "P", "notes", "mask", "feat"]
    outs = ["probs", "emb", "h"]
    dyn = {
        "X": {2: "T"},
        "P": {2: "T"},
        "notes": {1: "N"},
        "mask": {1: "N"},
        "feat": {1: "N"},
        "probs": {1: "N"},
        "emb": {1: "N"},
        "h": {1: "N"},
    }
    path = export(net, _grouper_case(g, 64, rng), f"{out_dir}/grouper.onnx", names, outs, dyn)
    check(
        path, net, [_grouper_case(g, 9, rng), _grouper_case(g, 300, rng), _grouper_case(g, 50, rng, audio_s=12.3)], names, 2e-3, "grouper"
    )
    heads = [
        (lin.weight.detach().double().numpy(), lin.bias.detach().double().numpy()) for lin in g.model.heads.th if isinstance(lin, nn.Linear)
    ]
    flat, shapes = [], []
    for Wm, b in heads:
        flat += [Wm.ravel(), b.ravel()]
        shapes.append([list(Wm.shape), list(b.shape)])
    np.concatenate(flat).astype("<f8").tofile(f"{out_dir}/track_head.bin")
    fr = g.model.front
    return dict(
        window_s=g.window_s,
        window_hop=g.window_hop,
        max_notes=g.max_notes,
        decode=g.params,
        track_head_shapes=shapes,
        front=dict(lo=int(fr.lo), NE=int(fr.NE), ch=int(fr.ch)),
    )


# ----------------------------------------------------------------------------- drums
class Attn(nn.Module):
    """softmax(q k^T / sqrt(d)) v, the same as F.scaled_dot_product_attention without a mask."""

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)


class Embed(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.embed(x)


def _split_temporal(net):
    blocks = list(net.temporal)
    from dettato.drums.model import SongAttention

    idx = [i for i, b in enumerate(blocks) if isinstance(b, SongAttention)]
    assert idx == [len(blocks) - 1], "the export assumes one song attention, at the end of the temporal stack"
    return blocks[:-1], blocks[-1]


class TrunkA(nn.Module):
    """DrumNet.trunk up to its song attention -> (h, q, k, v) with q/k/v as (1, heads, T, d)."""

    def __init__(self, net):
        super().__init__()
        self.pre, self.att = _split_temporal(net)
        self.pre = nn.ModuleList(self.pre)

    def forward(self, c):
        h = c
        for b in self.pre:
            h = b(h)
        a = self.att
        B, T, _ = h.shape
        q, k, v = a.qkv(a.n(h)).view(B, T, 3, a.heads, a.da // a.heads).permute(2, 0, 3, 1, 4)
        return h, q, k, v


class TrunkB(nn.Module):
    """The rest of DrumNet.trunk: attention output -> (trunk state, onset logits, velocity logits)."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        _, self.att = _split_temporal(net)

    def forward(self, h, a):
        B, T, _ = h.shape
        h = h + self.att.o(a.transpose(1, 2).reshape(B, T, self.att.da))
        return h, self.net.on(h), self.net.vel(h)


class CQ(nn.Module):
    def __init__(self, ex):
        super().__init__()
        self.cq = ex.cq

    def forward(self, xc):
        return self.cq(xc)


class AdaptA(nn.Module):
    """OrchestralExpert.forward from (h0, z0, cq) up to its attention -> (r, q, k, v)."""

    def __init__(self, ex):
        super().__init__()
        self.ex = ex

    def forward(self, h0, z0, cq):
        ex = self.ex
        a = F.silu(ex.inp(torch.cat([h0.float(), cq, torch.sigmoid(z0.float())], -1)))
        r, _ = ex.rnn(a)
        B, Tn, D = r.shape
        q, k, v = ex.qkv(r).view(B, Tn, 3, ex.heads, D // ex.heads).permute(2, 0, 3, 1, 4)
        return r, q, k, v


class AdaptB(nn.Module):
    """The rest of OrchestralExpert.forward -> (corrected drum logits, timpani logits, router logit)."""

    def __init__(self, ex):
        super().__init__()
        self.ex = ex

    def forward(self, r, att, z0):
        ex = self.ex
        B, Tn, D = r.shape
        r = ex.ln(r + ex.ao(att.transpose(1, 2).reshape(B, Tn, D)))
        router = ex.gate(r.float().mean(1))[:, 0]
        return z0.float() + ex.delta(r).float(), ex.timp(r).float(), router


def _drum_parts(net, x160, prefix, out_dir, rng_T):
    """Export embed / trunkA / trunkB of one DrumNet; returns callables for chaining the checks."""
    T1, T2 = rng_T
    emb = Embed(net).eval()
    p = export(emb, (x160[:, :T1],), f"{out_dir}/drums/{prefix}_embed.onnx", ["x"], ["c"], {"x": {1: "T"}, "c": {1: "T2"}})
    check(p, emb, [(x160[:, :T1],), (x160[:, :T2],)], ["x"], 1e-4, f"{prefix}_embed")
    with torch.no_grad():
        c1, c2 = net.embed(x160[:, :T1]), net.embed(x160[:, :T2])
    ta = TrunkA(net).eval()
    p = export(
        ta,
        (c1,),
        f"{out_dir}/drums/{prefix}_trunkA.onnx",
        ["c"],
        ["h", "q", "k", "v"],
        {"c": {1: "T"}, "h": {1: "T"}, "q": {2: "T"}, "k": {2: "T"}, "v": {2: "T"}},
    )
    check(p, ta, [(c1,), (c2,)], ["c"], 1e-4, f"{prefix}_trunkA")
    tb = TrunkB(net).eval()
    with torch.no_grad():
        h1, q1, k1, v1 = ta(c1)
        h2, q2, k2, v2 = ta(c2)
        a1, a2 = F.scaled_dot_product_attention(q1, k1, v1), F.scaled_dot_product_attention(q2, k2, v2)
    p = export(
        tb,
        (h1, a1),
        f"{out_dir}/drums/{prefix}_trunkB.onnx",
        ["h", "a"],
        ["h2", "z", "v"],
        {"h": {1: "T"}, "a": {2: "T"}, "h2": {1: "T"}, "z": {1: "T"}, "v": {1: "T"}},
    )
    check(p, tb, [(h1, a1), (h2, a2)], ["h", "a"], 1e-4, f"{prefix}_trunkB")
    # the split chain must equal the module's own trunk
    with torch.no_grad():
        ref = net.trunk(c2)
        got = tb(h2, a2)
    for r, g_ in zip(ref, got):
        assert torch.allclose(r, g_, atol=1e-5), f"{prefix}: split trunk differs from DrumNet.trunk"
    return (h1, tb(h1, a1)), (h2, tb(h2, a2))


def export_drums(out_dir):
    d = DrumTranscriber.from_pretrained("cpu")
    rng = np.random.default_rng(2)
    x = torch.from_numpy(rng.normal(-6, 2, (1, 1200, 280)).astype(np.float32))
    T1, T2 = 400, 1200
    attn = Attn().eval()
    q = torch.randn(1, 4, 300, 64)
    k = torch.randn(1, 4, 700, 64)
    p = export(
        attn,
        (q, k, k),
        f"{out_dir}/attn.onnx",
        ["q", "k", "v"],
        ["o"],
        {"q": {1: "H", 2: "Lq", 3: "D"}, "k": {1: "H", 2: "L", 3: "D"}, "v": {1: "H", 2: "L", 3: "D"}, "o": {1: "H", 2: "Lq", 3: "D"}},
    )
    check(
        p,
        attn,
        [(q, k, k), (torch.randn(1, 2, 50, 16), torch.randn(1, 2, 90, 16), torch.randn(1, 2, 90, 16))],
        ["q", "k", "v"],
        1e-4,
        "attn",
    )
    with torch.no_grad():
        _drum_parts(d.kit, x[..., :160], "kit", out_dir, (T1, T2))
        for i, ex in enumerate(d.experts):
            (h1, (h01, z01, _)), (h2, (h02, z02, _)) = _drum_parts(ex.base, x[..., :160], f"ex{i}", out_dir, (T1, T2))
            cq = CQ(ex).eval()
            pth = export(cq, (x[:, :T1, 160:],), f"{out_dir}/drums/ex{i}_cq.onnx", ["xc"], ["cq"], {"xc": {1: "T"}, "cq": {1: "T2"}})
            check(pth, cq, [(x[:, :T1, 160:],), (x[:, :T2, 160:],)], ["xc"], 1e-4, f"ex{i}_cq")
            cq1, cq2 = cq(x[:, :T1, 160:]), cq(x[:, :T2, 160:])
            aa = AdaptA(ex).eval()
            pth = export(
                aa,
                (h01, z01, cq1),
                f"{out_dir}/drums/ex{i}_adaptA.onnx",
                ["h0", "z0", "cq"],
                ["r", "q", "k", "v"],
                {"h0": {1: "T"}, "z0": {1: "T"}, "cq": {1: "T"}, "r": {1: "T"}, "q": {2: "T"}, "k": {2: "T"}, "v": {2: "T"}},
            )
            check(pth, aa, [(h01, z01, cq1), (h02, z02, cq2)], ["h0", "z0", "cq"], 1e-4, f"ex{i}_adaptA")
            r1, qq1, kk1, vv1 = aa(h01, z01, cq1)
            r2, qq2, kk2, vv2 = aa(h02, z02, cq2)
            at1, at2 = F.scaled_dot_product_attention(qq1, kk1, vv1), F.scaled_dot_product_attention(qq2, kk2, vv2)
            ab = AdaptB(ex).eval()
            pth = export(
                ab,
                (r1, at1, z01),
                f"{out_dir}/drums/ex{i}_adaptB.onnx",
                ["r", "att", "z0"],
                ["drum", "timpani", "router"],
                {"r": {1: "T"}, "att": {2: "T"}, "z0": {1: "T"}, "drum": {1: "T"}, "timpani": {1: "T"}},
            )
            check(pth, ab, [(r1, at1, z01), (r2, at2, z02)], ["r", "att", "z0"], 1e-4, f"ex{i}_adaptB")
            # the whole split chain equals OrchestralExpert.forward on the same input
            ref = ex(x[:, :T2])
            got = ab(r2, at2, z02)
            for key, gv in zip(("drum", "timpani", "router"), got):
                assert torch.allclose(ref[key], gv, atol=1e-4), f"ex{i}: split chain differs from OrchestralExpert ({key})"
    th = d.config["thresholds"]
    return dict(
        keys=d.config["keys"],
        timpani_keys=d.config["timpani_keys"],
        thresholds=th,
        n_experts=len(d.experts),
        decode=d.config.get("decode", {}),
        min_source_rate=d.config["min_source_rate"],
    )


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "models"))
    ap.add_argument("--only", nargs="*", choices=["transcriber", "repairer", "grouper", "drums"])
    a = ap.parse_args()
    torch.set_num_threads(min(16, os.cpu_count() or 4))
    stages = a.only or ["transcriber", "repairer", "grouper", "drums"]
    meta_path = os.path.join(a.out, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    print(f"exporting {stages} -> {a.out}  (torch {torch.__version__}, CPU only)")
    fns = dict(transcriber=export_transcriber, repairer=export_repairer, grouper=export_grouper, drums=export_drums)
    for s in stages:
        meta[s] = fns[s](a.out)
    meta["instruments"] = dict(names=INSTRUMENTS, programs=[int(p) for p in PROGRAM], credit=np.asarray(CREDIT).tolist())
    meta["opset"] = OPSET
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    print("wrote", meta_path)


if __name__ == "__main__":
    sys.exit(main())
