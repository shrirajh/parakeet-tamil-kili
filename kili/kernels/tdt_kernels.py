"""Triton kernels behind the gathered TDT loss (kili.kernels.tdt_loss): joint-row reductions and the TDT recursion.

1. Row kernels. For each lattice cell (row) of the joint output z[N, Vt + D] (Vt = vocab + blank, D = durations):
     forward : lse_other = logsumexp(z[j] + b[j] for j < Vt, j != y, j != blank)   (online, fp32)
               zl = z[y] + b[y], zb = z[blank] + b[blank], zd = z[Vt:Vt+D] + b[Vt:Vt+D]
     backward: dz[j] = g_other * exp(z[j] + b[j] - lse_other) for "other" tokens; dz[y] = g_l, dz[blank] = g_b,
               dz[Vt + k] = g_d[k]
   One read of the logits forward, one write backward, instead of ~10 memory-bound PyTorch passes. The joint's output
   bias b is optional: with it, the projection is a plain GEMM (no addmm) and the bias gradient comes from
   per-32-row-group column sums written by the backward kernel (no separate full reduction over dz).

2. TDT recursion on log-probabilities (replaces NeMo's numba kernel). Inputs per cell (b, t, u), t < T_b, u <= U_b:
     LL log p(label y_{u+1} | t, u) (u < U), LB log p(blank | t, u), DL / DB log p(duration i | t, u) for label /
     blank arcs (TDT: both = log_softmax of the duration logits).
   Arcs exactly as NeMo's compute_tdt_{alphas,betas}_kernel (sigma subtracted per arc):
     label (t,u) -> (t+d, u+1) for any duration d (incl. 0), t+d < T
     blank (t,u) -> (t+d, u)   for d > 0, t+d < T; at u = U the blank with t+d == T terminates.
   NeMo's omega branch (plain RNN-T with probability omega) is the same lattice with label arcs only at d=0, blank arcs
   only at d=1, no duration terms and sigma = 0.
   One program per utterance walks t forward (alpha) / backward (beta); the d=0 label chain inside a row,
   x_u = logaddexp(c_u, x_{u-1} + w_u), is a linear recurrence in the (logaddexp, +) semiring solved with
   tl.associative_scan (O(log U) depth instead of an O(T+U) anti-diagonal sweep). Gradients are one parallel pass over
   cells. No host synchronisation. NeMo differentiates w.r.t. raw logits (its kernel applies log_softmax); this
   differentiates w.r.t. log-probs and autograd applies the log_softmax Jacobian: the same gradient.
"""
from __future__ import annotations

import os
import random

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # explicit (Triton warns the default will change)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

# ---------------------------------------------------------------------------------------------- 1. row kernels

# launch configs (RTX PRO 6000 Blackwell: backward 8 warps is 1.73x over 4 and bit-identical: the backward is
# elementwise per column; the forward's BLOCK fixes the logsumexp summation order = fp32 round-off only)
FWD_BLOCK, FWD_WARPS = 1024, 4
BWD_BLOCK, BWD_WARPS = 1024, 8
BIAS_ROWS = 32  # rows per program in the bias backward (one partial column sum per group)


@triton.jit
def _rows_fwd(Z, Y, LSE_O, ZL, ZB, ZD, BIAS, stride_z, Vt, blank, D: tl.constexpr, DP: tl.constexpr,
              BLOCK: tl.constexpr, HAS_BIAS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    zrow = Z + row * stride_z
    y = tl.load(Y + row)
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK], tl.float32)
    for start in range(0, Vt, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        valid = (cols < Vt) & (cols != y) & (cols != blank)
        x = tl.load(zrow + cols, mask=valid, other=float("-inf")).to(tl.float32)
        if HAS_BIAS:
            x += tl.load(BIAS + cols, mask=valid, other=0.0).to(tl.float32)
        m_new = tl.maximum(m, x)
        scale = tl.where(m_new == float("-inf"), 0.0, tl.exp(m - m_new))  # guard (-inf) - (-inf) lanes
        s = s * scale + tl.where(valid, tl.exp(x - m_new), 0.0)
        m = m_new
    mx = tl.max(m, 0)
    tot = tl.sum(s * tl.where(m == float("-inf"), 0.0, tl.exp(m - mx)), 0)
    tl.store(LSE_O + row, mx + tl.log(tot))
    zl = tl.load(zrow + y).to(tl.float32)
    zb = tl.load(zrow + blank).to(tl.float32)
    dcols = tl.arange(0, DP)
    dm = dcols < D
    zd = tl.load(zrow + Vt + dcols, mask=dm, other=0.0).to(tl.float32)
    if HAS_BIAS:
        zl += tl.load(BIAS + y).to(tl.float32)
        zb += tl.load(BIAS + blank).to(tl.float32)
        zd += tl.load(BIAS + Vt + dcols, mask=dm, other=0.0).to(tl.float32)
    tl.store(ZL + row, zl)
    tl.store(ZB + row, zb)
    tl.store(ZD + row * D + dcols, zd, mask=dm)


@triton.jit
def _rows_bwd(Z, Y, LSE_O, GO, GL, GB, GD, DZ, stride_z, stride_dz, Vt, blank,
              D: tl.constexpr, DP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    zrow = Z + row * stride_z
    dzrow = DZ + row * stride_dz
    y = tl.load(Y + row)
    lse_o = tl.load(LSE_O + row)
    go = tl.load(GO + row)
    gl = tl.load(GL + row)
    gb = tl.load(GB + row)
    for start in range(0, Vt, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        inb = cols < Vt
        x = tl.load(zrow + cols, mask=inb, other=0.0).to(tl.float32)
        g = go * tl.exp(x - lse_o)
        g = tl.where(cols == y, gl, g)
        g = tl.where(cols == blank, gb, g)
        tl.store(dzrow + cols, g.to(DZ.dtype.element_ty), mask=inb)
    dcols = tl.arange(0, DP)
    dm = dcols < D
    tl.store(dzrow + Vt + dcols, tl.load(GD + row * D + dcols, mask=dm, other=0.0).to(DZ.dtype.element_ty),
             mask=dm)


@triton.jit
def _rows_bwd_bias(Z, Y, LSE_O, GO, GL, GB, GD, DZ, BIAS, DBP, N, stride_z, stride_dz, Vt, blank,
                   D: tl.constexpr, DP: tl.constexpr, BLOCK: tl.constexpr, R: tl.constexpr):
    """Backward with the output bias folded in (z excludes the bias): R rows x one BLOCK of columns per program;
    writes dz and this row group's column sums of dz (the bias gradient, summed over groups in torch)."""
    pg = tl.program_id(0)
    pc = tl.program_id(1)
    cols = pc * BLOCK + tl.arange(0, BLOCK)
    inb = cols < Vt
    b = tl.load(BIAS + cols, mask=inb, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK], tl.float32)
    for i in range(R):
        row = (pg * R + i).to(tl.int64)
        if row < N:
            y = tl.load(Y + row)
            lse_o = tl.load(LSE_O + row)
            go = tl.load(GO + row)
            gl = tl.load(GL + row)
            gb = tl.load(GB + row)
            x = tl.load(Z + row * stride_z + cols, mask=inb, other=0.0).to(tl.float32) + b
            g = go * tl.exp(x - lse_o)
            g = tl.where(cols == y, gl, g)
            g = tl.where(cols == blank, gb, g)
            g = tl.where(inb, g, 0.0)
            gq = g.to(DZ.dtype.element_ty)
            tl.store(DZ + row * stride_dz + cols, gq, mask=inb)
            acc += gq.to(tl.float32)  # bias grad = column sum of the dz actually propagated (as addmm's backward)
            if pc == 0:
                dcols = tl.arange(0, DP)
                dm = dcols < D
                tl.store(DZ + row * stride_dz + Vt + dcols,
                         tl.load(GD + row * D + dcols, mask=dm, other=0.0).to(DZ.dtype.element_ty), mask=dm)
    tl.store(DBP + pg * Vt + cols, acc, mask=inb)


class _GatheredRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z2, y, Vt: int, blank: int, D: int, bias=None):
        N = z2.shape[0]
        dev = z2.device
        lse_o = torch.empty(N, device=dev, dtype=torch.float32)
        zl = torch.empty_like(lse_o)
        zb = torch.empty_like(lse_o)
        zd = torch.empty(N, D, device=dev, dtype=torch.float32)
        has_b = bias is not None
        if N:
            _rows_fwd[(N,)](z2, y, lse_o, zl, zb, zd, bias if has_b else z2, z2.stride(0), Vt, blank, D=D,
                            DP=triton.next_power_of_2(D), BLOCK=FWD_BLOCK, HAS_BIAS=has_b, num_warps=FWD_WARPS)
        if has_b:
            ctx.save_for_backward(z2, y, lse_o, bias)
        else:
            ctx.save_for_backward(z2, y, lse_o)
        ctx.meta = (Vt, blank, D)  # z2 may be wider than Vt + D (alignment padding): those columns are ignored
        return lse_o, zl, zb, zd

    @staticmethod
    def backward(ctx, g_o, g_l, g_b, g_d):
        st = ctx.saved_tensors  # unpack once (a checkpoint recompute forbids a second unpack)
        z2, y, lse_o = st[:3]
        bias = st[3] if len(st) > 3 else None
        Vt, blank, D = ctx.meta
        N = z2.shape[0]
        dz = torch.empty_like(z2, memory_format=torch.contiguous_format)
        if z2.shape[1] > Vt + D:
            dz[:, Vt + D:].zero_()  # padding columns get no gradient (the kernels write only the first Vt + D)

        def prep(g, *shape):
            return g.contiguous().float() if g is not None else torch.zeros(*shape, device=z2.device)

        g_o, g_l, g_b, g_d = prep(g_o, N), prep(g_l, N), prep(g_b, N), prep(g_d, N, D)
        if bias is not None:
            G = triton.cdiv(N, BIAS_ROWS)
            dbp = torch.zeros(max(G, 1), Vt, device=z2.device, dtype=torch.float32)
            if N:
                _rows_bwd_bias[(G, triton.cdiv(Vt, BWD_BLOCK))](
                    z2, y, lse_o, g_o, g_l, g_b, g_d, dz, bias, dbp, N, z2.stride(0), dz.stride(0), Vt, blank,
                    D=D, DP=triton.next_power_of_2(D), BLOCK=BWD_BLOCK, R=BIAS_ROWS, num_warps=BWD_WARPS)
            db = torch.zeros(bias.shape, device=z2.device, dtype=torch.float32)
            db[:Vt] = dbp.sum(0)
            db[Vt:Vt + D] = dz[:, Vt:Vt + D].float().sum(0)
            return dz, None, None, None, None, db.to(bias.dtype)
        if N:
            _rows_bwd[(N,)](z2, y, lse_o, g_o, g_l, g_b, g_d, dz, z2.stride(0), dz.stride(0), Vt, blank,
                            D=D, DP=triton.next_power_of_2(D), BLOCK=BWD_BLOCK, num_warps=BWD_WARPS)
        return dz, None, None, None, None, None


def gathered_rows(z, y, blank: int, n_dur: int, width: int | None = None, bias=None):
    """z: [..., W] logits (any float dtype, last dim contiguous) whose first `width` (default W) columns are
    [Vt vocab+blank | D durations]; columns beyond `width` are alignment padding (ignored, no gradient).
    y: [...] target ids. bias: the output projection's bias (len >= Vt + D) when z was computed without it.
    Returns (lse_other, zl, zb, zd) in fp32 with shapes [...], [...], [...], [..., D]."""
    lead = z.shape[:-1]
    Vt = (width or z.shape[-1]) - n_dur
    z2 = z.reshape(-1, z.shape[-1])
    if z2.stride(-1) != 1:
        z2 = z2.contiguous()
    y2 = y.reshape(-1).to(torch.int32).contiguous()
    lse_o, zl, zb, zd = _GatheredRows.apply(z2, y2, Vt, blank, n_dur, bias)
    return lse_o.view(lead), zl.view(lead), zb.view(lead), zd.view(*lead, n_dur)


# ---------------------------------------------------------------------------------------------- 2. TDT recursion

NEG = float("-inf")


@triton.jit
def _lae(a, b):
    """logaddexp, safe for -inf on both sides."""
    m = tl.maximum(a, b)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    return tl.where(m == float("-inf"), float("-inf"), m_safe + tl.log(tl.exp(a - m_safe) + tl.exp(b - m_safe)))


@triton.jit
def _combine(w1, c1, w2, c2):
    # compose affine maps f(x) = lae(c, x + w): first (w1, c1) then (w2, c2)
    return w1 + w2, _lae(c1 + w2, c2)


@triton.jit
def _alpha_kernel(LL, LB, DL, DB, DUR, XLEN, YLEN, A, LLF, sigma, maxT, maxU,
                  D: tl.constexpr, BLOCK_U: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    T = tl.load(XLEN + b)
    U1 = tl.load(YLEN + b) + 1  # number of u positions
    u = tl.arange(0, BLOCK_U)
    inU = u < U1
    base = b * maxT * maxU
    for t in range(0, T):
        c = tl.full([BLOCK_U], float("-inf"), tl.float32)
        for i in tl.static_range(D):
            d = tl.load(DUR + i)
            src = t - d
            ok = (src >= 0) & (d > 0)
            # blank arc (src, u) -> (t, u)
            m = inU & ok
            a = tl.load(A + base + src * maxU + u, mask=m, other=float("-inf"))
            lb = tl.load(LB + base + src * maxU + u, mask=m, other=float("-inf"))
            db = tl.load(DB + (base + src * maxU + u) * D + i, mask=m, other=float("-inf"))
            c = _lae(c, a + lb + db - sigma)
            # label arc (src, u-1) -> (t, u), d > 0
            m2 = inU & ok & (u >= 1)
            a2 = tl.load(A + base + src * maxU + u - 1, mask=m2, other=float("-inf"))
            ll2 = tl.load(LL + base + src * maxU + u - 1, mask=m2, other=float("-inf"))
            dl2 = tl.load(DL + (base + src * maxU + u - 1) * D + i, mask=m2, other=float("-inf"))
            c = _lae(c, a2 + ll2 + dl2 - sigma)
        c = tl.where((u == 0) & (t == 0), 0.0, c)
        # d = 0 label arcs inside row t: x_u = lae(c_u, x_{u-1} + w_u), w_u = LL[t,u-1] + DL[t,u-1,i0] - sigma
        w = tl.full([BLOCK_U], float("-inf"), tl.float32)
        for i in tl.static_range(D):
            d = tl.load(DUR + i)
            m3 = inU & (u >= 1) & (d == 0)
            ll3 = tl.load(LL + base + t * maxU + u - 1, mask=m3, other=float("-inf"))
            dl3 = tl.load(DL + (base + t * maxU + u - 1) * D + i, mask=m3, other=float("-inf"))
            w = tl.where(m3, ll3 + dl3 - sigma, w)
        w = tl.where(u == 0, float("-inf"), w)
        _, x = tl.associative_scan((w, c), 0, _combine)
        tl.store(A + base + t * maxU + u, x, mask=inU)
        tl.debug_barrier()
    # terminal: blank from (T-d, U-1) with d > 0
    ll = float("-inf")
    for i in tl.static_range(D):
        d = tl.load(DUR + i)
        src = T - d
        ok = (d > 0) & (src >= 0)
        off = base + tl.where(ok, src, 0) * maxU + U1 - 1
        a = tl.load(A + off)
        lb = tl.load(LB + off)
        db = tl.load(DB + off * D + i)
        ll = tl.where(ok, _lae(ll, a + lb + db - sigma), ll)
    tl.store(LLF + b, ll)


@triton.jit
def _beta_kernel(LL, LB, DL, DB, DUR, XLEN, YLEN, Bt, sigma, maxT, maxU,
                 D: tl.constexpr, BLOCK_U: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    T = tl.load(XLEN + b)
    U1 = tl.load(YLEN + b) + 1
    u = tl.arange(0, BLOCK_U)
    inU = u < U1
    last = u == U1 - 1
    base = b * maxT * maxU
    for k in range(0, T):
        t = T - 1 - k
        c = tl.full([BLOCK_U], float("-inf"), tl.float32)
        lb = tl.load(LB + base + t * maxU + u, mask=inU, other=float("-inf"))
        ll = tl.load(LL + base + t * maxU + u, mask=inU & ~last, other=float("-inf"))
        for i in tl.static_range(D):
            d = tl.load(DUR + i)
            dst = t + d
            db = tl.load(DB + (base + t * maxU + u) * D + i, mask=inU, other=float("-inf"))
            dl = tl.load(DL + (base + t * maxU + u) * D + i, mask=inU & ~last, other=float("-inf"))
            # blank arc (t,u) -> (dst,u), d > 0; terminal when dst == T at u == U-1
            mb = inU & (d > 0) & (dst < T)
            bb = tl.load(Bt + base + dst * maxU + u, mask=mb, other=float("-inf"))
            bb = tl.where(inU & (d > 0) & (dst == T) & last, 0.0, bb)
            c = _lae(c, bb + lb + db - sigma)
            # label arc (t,u) -> (dst,u+1), d > 0 (d = 0 handled by the scan)
            ml = inU & ~last & (d > 0) & (dst < T)
            bl = tl.load(Bt + base + dst * maxU + u + 1, mask=ml, other=float("-inf"))
            c = _lae(c, bl + ll + dl - sigma)
        w = tl.full([BLOCK_U], float("-inf"), tl.float32)
        for i in tl.static_range(D):
            d = tl.load(DUR + i)
            m3 = inU & ~last & (d == 0)
            dl0 = tl.load(DL + (base + t * maxU + u) * D + i, mask=m3, other=float("-inf"))
            w = tl.where(m3, ll + dl0 - sigma, w)
        _, x = tl.associative_scan((w, c), 0, _combine, reverse=True)
        tl.store(Bt + base + t * maxU + u, x, mask=inU)
        tl.debug_barrier()


@triton.jit
def _grad_kernel(LL, LB, DL, DB, DUR, XLEN, YLEN, A, Bt, LLF, GOUT,
                 GLL, GLB, GDL, GDB, sigma, maxT, maxU, D: tl.constexpr, BLOCK_U: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1)
    T = tl.load(XLEN + b)
    U1 = tl.load(YLEN + b) + 1
    u = tl.arange(0, BLOCK_U)
    base = b * maxT * maxU
    valid = (u < U1) & (t < T)
    last = u == U1 - 1
    off = base + t * maxU + u
    lnorm = tl.load(LLF + b)
    g = tl.load(GOUT + b)  # d(total)/d(cost_b); cost = -ll  ->  d cost / d logp(arc) = -posterior
    a = tl.load(A + off, mask=valid, other=float("-inf"))
    lb = tl.load(LB + off, mask=valid, other=float("-inf"))
    ll = tl.load(LL + off, mask=valid & ~last, other=float("-inf"))
    g_ll = tl.zeros([BLOCK_U], tl.float32)
    g_lb = tl.zeros([BLOCK_U], tl.float32)
    for i in tl.static_range(D):
        d = tl.load(DUR + i)
        dst = t + d
        db = tl.load(DB + off * D + i, mask=valid, other=float("-inf"))
        dl = tl.load(DL + off * D + i, mask=valid & ~last, other=float("-inf"))
        mb = valid & (d > 0) & (dst < T)
        bb = tl.load(Bt + base + dst * maxU + u, mask=mb, other=float("-inf"))
        bb = tl.where(valid & (d > 0) & (dst == T) & last, 0.0, bb)
        pb = tl.exp(a + lb + db - sigma + bb - lnorm)
        pb = tl.where(valid, pb, 0.0)
        ml = valid & ~last & (dst < T)
        bl = tl.load(Bt + base + dst * maxU + u + 1, mask=ml, other=float("-inf"))
        pl = tl.exp(a + ll + dl - sigma + bl - lnorm)
        pl = tl.where(ml, pl, 0.0)
        g_lb += pb
        g_ll += pl
        tl.store(GDB + off * D + i, -g * pb, mask=u < maxU)
        tl.store(GDL + off * D + i, -g * pl, mask=u < maxU)
    tl.store(GLB + off, -g * g_lb, mask=u < maxU)
    tl.store(GLL + off, -g * g_ll, mask=u < maxU)


class _TDTLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, LL, LB, DL, DB, xlen, ylen, durations, sigma: float):
        Bn, maxT, maxU = LB.shape
        D = DL.shape[-1]
        LL, LB, DL, DB = (x.contiguous().float() for x in (LL, LB, DL, DB))
        xlen = xlen.to(torch.int32).contiguous()
        ylen = ylen.to(torch.int32).contiguous()
        dur = durations.to(device=LB.device, dtype=torch.int32).contiguous()
        A = torch.full_like(LB, NEG)
        llf = torch.empty(Bn, device=LB.device, dtype=torch.float32)
        BU = max(16, triton.next_power_of_2(maxU))
        nw = 4 if BU <= 256 else 8
        _alpha_kernel[(Bn,)](LL, LB, DL, DB, dur, xlen, ylen, A, llf, float(sigma), maxT, maxU,
                             D=D, BLOCK_U=BU, num_warps=nw)
        ctx.save_for_backward(LL, LB, DL, DB, dur, xlen, ylen, A, llf)
        ctx.sigma, ctx.BU, ctx.nw = float(sigma), BU, nw
        return -llf

    @staticmethod
    def backward(ctx, gout):
        LL, LB, DL, DB, dur, xlen, ylen, A, llf = ctx.saved_tensors
        Bn, maxT, maxU = LB.shape
        D = DL.shape[-1]
        Bt = torch.full_like(LB, NEG)
        _beta_kernel[(Bn,)](LL, LB, DL, DB, dur, xlen, ylen, Bt, ctx.sigma, maxT, maxU,
                            D=D, BLOCK_U=ctx.BU, num_warps=ctx.nw)
        gLL, gLB = torch.empty_like(LL), torch.empty_like(LB)
        gDL, gDB = torch.empty_like(DL), torch.empty_like(DB)
        _grad_kernel[(Bn, maxT)](LL, LB, DL, DB, dur, xlen, ylen, A, Bt, llf, gout.contiguous().float(),
                                 gLL, gLB, gDL, gDB, ctx.sigma, maxT, maxU, D=D, BLOCK_U=ctx.BU,
                                 num_warps=ctx.nw)
        return gLL, gLB, gDL, gDB, None, None, None, None


_DUR_CACHE: dict = {}


def _dur_tensor(durations: tuple, device):
    key = (durations, str(device))
    if key not in _DUR_CACHE:
        _DUR_CACHE[key] = torch.as_tensor(durations, device=device)  # cached: as_tensor(list, device) syncs
    return _DUR_CACHE[key]


def tdt_loss_from_reduced(red, act_lens, label_lens, durations, sigma: float, omega: float):
    """red: [B, T, U+1, 3 + D] = [lp_label, lp_other, lp_blank, duration logits...] (kili.kernels.tdt_loss).
    Returns per-utterance costs [B] (NeMo semantics: with probability omega, the standard RNN-T loss)."""
    LL, LB = red[..., 0], red[..., 2]
    D = red.shape[-1] - 3
    dur_t = _dur_tensor(tuple(durations), red.device)
    if omega > 0 and random.uniform(0, 1) < omega:
        # NeMo's omega branch: plain RNN-T (label d=0, blank d=1, no duration terms, no sigma)
        zero = torch.zeros((), device=red.device)
        ninf = torch.full((), NEG, device=red.device)
        shape = (*LB.shape, D)
        DL = torch.where(dur_t == 0, zero, ninf).expand(shape)
        DB = torch.where(dur_t == 1, zero, ninf).expand(shape)
        return _TDTLogProbs.apply(LL, LB, DL, DB, act_lens, label_lens, dur_t, 0.0)
    dlp = torch.log_softmax(red[..., 3:], dim=-1)
    return _TDTLogProbs.apply(LL, LB, dlp, dlp, act_lens, label_lens, dur_t, sigma)
