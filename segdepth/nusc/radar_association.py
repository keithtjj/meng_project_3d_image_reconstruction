"""
radar_association.py
--------------------
Depth-based radar-to-segment association by energy minimisation.

    E(A, theta) = sum_i U_i + sum_j C_j + sum_(i,k) phi_ik + R(theta)

U_i     Negative log marginal likelihood that radar point i came from segment j,
        from agreement between the radar range and the depth map. Unknown
        elevation is integrated out by sampling the beam arc in 3D in the radar
        frame and projecting through the true radar->camera transform (mounting
        offset, timing and Doppler range correction respected).
        Null hypotheses: clutter and occluded.
C_j     Optional extent consistency: points on one 'thing' must lie within a
        plausible object length of each other.
phi_ik  Optional ordinal consistency: for nearby pairs of points on different
        segments, the radar's near/far order should match the depth map's order.
        Scale-free: independent of the depth scale (a, b).
R       Priors: ground-plane depth consistency and small per-radar yaw.

Boundary robustness (AssocConfig.profile_depth): per-column segment depth and
grown masks. Depth input may contain NaN (e.g. saturated metric depth), treated
as unknown depth.

RCS and velocity are not used for assignment; they are only reported.

Optimisation: ICM over assignments, then closed-form proposals and an adaptive
pattern search over log(a), b and per-radar yaw. Every accepted step lowers E.
Stops when no point moves, no parameter change is accepted and all step sizes
have shrunk below their minimums.

Contents:
  * build_assoc_maps         Per-column segment depth and grown masks.
  * Params                   Global parameters (a, b, sigma_inv, yaw per radar).
  * RadarSegmentAssociator   Builds the energy, optimises it, and exports
                             summaries, per-point tables and per-segment stats.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from ..config import EXTENT, AssocConfig
from ..depth_scale import closeness_to_metric, fit_affine_inverse
from ..utils import dilate_mask, erode_mask

GROWABLE = set(EXTENT) | {"pole", "traffic sign", "traffic light"}


def _finite_or_none(x, nd=2):
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


# --------------------------------------------------------------------------- #
# Association maps
# --------------------------------------------------------------------------- #
def build_assoc_maps(seg_map, closeness, cand_ids, near_first, grow_ids,
                     dilate_px=6, erode_px=3, min_core=20):
    """
    Returns (seg_map_assoc, closeness_assoc):
      * closeness inside every candidate segment replaced by the per-column
        median of its eroded core (removes boundary bleeding);
      * segments in `grow_ids` grown by `dilate_px` into non-candidate pixels
        (road, sky, unlabelled), taking the segment's column depth.
    `near_first` orders segments so nearer ones claim contested pixels first.
    """
    seg_out = seg_map.copy()
    c_out = closeness.astype(np.float32).copy()
    free = ~np.isin(seg_map, cand_ids)
    W = seg_map.shape[1]
    grow_ids = {int(g) for g in grow_ids}

    for sid in near_first:
        mask = seg_map == sid
        if not mask.any():
            continue
        core = erode_mask(mask, erode_px)
        if core.sum() < min_core:
            core = mask
        ys, xs = np.nonzero(core)
        fin = np.isfinite(closeness[ys, xs])
        ys, xs = ys[fin], xs[fin]
        if len(xs) == 0:
            continue

        order = np.argsort(xs, kind="stable")
        xs_s = xs[order]
        vals = closeness[ys[order], xs_s]
        cols, start = np.unique(xs_s, return_index=True)
        med = np.array([np.median(g) for g in np.split(vals, start[1:])])
        profile = np.interp(np.arange(W), cols, med)

        region = mask.copy()
        if int(sid) in grow_ids and dilate_px > 0:
            mx = np.nonzero(mask)[1]
            grow = dilate_mask(mask, dilate_px) & free
            grow[:, :mx.min()] = False
            grow[:, mx.max() + 1:] = False
            seg_out[grow] = sid
            free &= ~grow
            region |= grow
        rr, cc = np.nonzero(region)
        c_out[rr, cc] = profile[cc]

    return seg_out, c_out


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    a: float                                   # closeness ≈ a / Z + b
    b: float
    sigma_inv: float
    yaw: dict = field(default_factory=dict)    # radar channel -> yaw correction (rad)


# --------------------------------------------------------------------------- #
# Associator
# --------------------------------------------------------------------------- #
class RadarSegmentAssociator:
    def __init__(self, pts, cam, seg_map, closeness, results, params,
                 cfg=None, ground=None):
        self.cfg = cfg or AssocConfig()
        self.pts, self.cam = pts, cam
        self.seg_map, self.closeness = seg_map, closeness
        self.results, self.params = results, params
        self.ground = ground if ground is not None else (np.zeros(0), np.zeros(0))
        self.rng = np.random.default_rng(self.cfg.seed)

        self.ids = np.array([r["segment_id"] for r in results], dtype=np.int64)
        self.labels = [r["label"].lower() for r in results]
        self.M = len(self.ids)
        self.CLUTTER, self.OCCLUDED = self.M, self.M + 1

        excluded = {c.lower() for c in self.cfg.exclude_classes}
        self.candidate = np.array([l not in excluded for l in self.labels], dtype=bool)
        self.cand_idx = np.nonzero(self.candidate)[0]

        if self.cfg.profile_depth and len(self.cand_idx):
            cand_results = [r for r, c in zip(results, self.candidate) if c]
            near_first = [r["segment_id"] for r in
                          sorted(cand_results, key=lambda r: -r["median_closeness"])]
            grow_ids = [r["segment_id"] for r in cand_results
                        if r["label"].lower() in GROWABLE]
            self.seg_map_assoc, self.closeness_assoc = build_assoc_maps(
                seg_map, closeness, self.ids[self.candidate], near_first, grow_ids,
                dilate_px=self.cfg.mask_dilate_px, erode_px=self.cfg.profile_erode_px)
        else:
            self.seg_map_assoc, self.closeness_assoc = seg_map, closeness

        sm = self.seg_map_assoc
        lut = np.full(max(int(sm.max()), int(self.ids.max(initial=0))) + 1, -1)
        lut[self.ids[self.candidate]] = self.cand_idx
        self.col_map = np.where(sm >= 0, lut[np.clip(sm, 0, None)], -1)

        self.center = (self.cfg.n_elev // 2) * self.cfg.n_az + self.cfg.n_az // 2
        self._geo_cache = {}
        self._reset_pairs(len(pts))

    # ---------------- geometry (depends only on yaw) ---------------- #
    def _geometry(self, yaw):
        key = tuple(sorted(yaw.items()))
        if key in self._geo_cache:
            return self._geo_cache[key]
        if len(self._geo_cache) > 64:
            self._geo_cache.clear()

        cfg, p, cam = self.cfg, self.pts, self.cam
        N = len(p)
        r = np.linalg.norm(p.xy, axis=1) + p.v_r * p.dt       # Doppler range correction
        az = np.arctan2(p.xy[:, 1], p.xy[:, 0]) + np.array([yaw.get(c, 0.0) for c in p.channel])

        phi = np.deg2rad(np.linspace(-cfg.beam_half_elev_deg, cfg.beam_half_elev_deg, cfg.n_elev))
        s_az = max(np.deg2rad(cfg.az_sigma_deg), 1e-9)
        daz = np.linspace(-1.5, 1.5, cfg.n_az) * s_az if cfg.n_az > 1 else np.zeros(1)
        PHI, DAZ = (g.ravel() for g in np.meshgrid(phi, daz, indexing="ij"))
        w0 = np.exp(-0.5 * (DAZ / s_az) ** 2)

        A = az[:, None] + DAZ[None]                            # (N, S)
        P = r[:, None, None] * np.stack(
            [np.cos(PHI)[None] * np.cos(A), np.cos(PHI)[None] * np.sin(A),
             np.broadcast_to(np.sin(PHI)[None], A.shape)], axis=-1)

        Pc, h = np.empty_like(P), np.empty(A.shape)
        xy_ego, n_ego = np.empty((N, 2)), np.empty((N, 2))
        P0 = np.stack([r * np.cos(az), r * np.sin(az), np.zeros(N)], axis=1)
        for k in np.unique(p.sweep):
            m = p.sweep == k
            Tc, Te, Tg = p.T_cam[k], p.T_ego[k], p.T_egocam[k]
            Pc[m] = P[m] @ Tc[:3, :3].T + Tc[:3, 3]
            h[m] = P[m] @ Te[2, :3] + Te[2, 3]
            q = P0[m] @ Tg[:3, :3].T + Tg[:3, 3]
            xy_ego[m] = q[:, :2]
            los = q[:, :2] - Tg[:2, 3]
            n_ego[m] = los / np.linalg.norm(los, axis=1, keepdims=True)

        Z = Pc[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cam.K[0, 0] * Pc[..., 0] / Z + cam.K[0, 2]
            v = cam.K[1, 1] * Pc[..., 1] / Z + cam.K[1, 2]

        height_ok = (h >= cfg.h_min) & (h <= cfg.h_max)
        prior = w0[None] * height_ok
        prior = prior / np.maximum(prior.sum(1, keepdims=True), 1e-12)
        valid = (height_ok & (Z > 1.0) & (u >= 0) & (u < cam.width)
                 & (v >= 0) & (v < cam.height))

        geo = dict(u=np.where(valid, u, 0).astype(int), v=np.where(valid, v, 0).astype(int),
                   Z=Z, h=h, phi=PHI, valid=valid, prior=prior, xy_ego=xy_ego, n_ego=n_ego)
        self._geo_cache[key] = geo
        return geo

    # ---------------- unary costs ---------------- #
    def _unary(self, params, geo):
        cfg = self.cfg
        u, v, Z, valid, prior = geo["u"], geo["v"], geo["Z"], geo["valid"], geo["prior"]
        col = np.where(valid, self.col_map[v, u], -1)
        c = self.closeness_assoc[v, u]
        D = closeness_to_metric(c, params.a, params.b, cfg.max_range)

        sZ = np.sqrt(cfg.sigma_range ** 2 + (Z ** 2 * params.sigma_inv) ** 2)
        with np.errstate(invalid="ignore", over="ignore"):
            res = np.where(np.isfinite(D), (D - Z) / sZ, np.inf)
            p_d = ((1 - cfg.outlier_eps) * np.exp(-0.5 * res ** 2) / (np.sqrt(2 * np.pi) * sZ)
                   + cfg.outlier_eps / cfg.max_range)
        margin = np.maximum(cfg.occlusion_abs, cfg.occlusion_rel * Z)
        occluded = valid & np.isfinite(D) & (D < Z - margin)
        visible = valid & ~occluded
        lik = np.where(visible, prior * p_d, 0.0)

        N = len(self.pts)
        L = np.zeros((N, self.M + 2))
        best = np.zeros((N, self.M + 2), dtype=int)
        chyp = np.full((N, self.M + 2), np.nan)       # scale-free segment closeness per point
        cfin = np.isfinite(c)
        c0 = np.where(cfin, c, 0.0)
        for j in self.cand_idx:
            lj = np.where(col == j, lik, 0.0)
            L[:, j] = lj.sum(1)
            best[:, j] = lj.argmax(1)
            on = (col == j) & cfin
            wsum = np.where(on, prior, 0.0).sum(1)
            csum = np.where(on, prior * c0, 0.0).sum(1)
            chyp[:, j] = np.where(wsum > 0, csum / np.maximum(wsum, 1e-12), np.nan)
        L[:, self.CLUTTER] = cfg.rho_clutter / cfg.max_range
        occ_mass = np.where(occluded, prior, 0.0)
        L[:, self.OCCLUDED] = cfg.rho_occluded * occ_mass.sum(1) / cfg.max_range
        best[:, self.OCCLUDED] = occ_mass.argmax(1)

        U = -np.log(L + 1e-300)
        return U, dict(best=best, c=c, D=D, Z=Z, chyp=chyp)

    # ---------------- ordinal (relative-distance) term ---------------- #
    def _reset_pairs(self, N):
        empty = (np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0))
        self.nbrs = [empty] * N
        self.pair_i = self.pair_k = np.zeros(0, int)
        self.pair_w = self.pair_exp = np.zeros(0)

    def _build_pairs(self, geo):
        """Pairs (i nearer than k) of points in nearby columns whose radar depths
        clearly differ. Weights decay with column distance and are normalised so
        no point's total pair weight exceeds 1. pair_exp holds the closeness
        separation the radar predicts under the initial scale (fixed per run)."""
        cfg, N = self.cfg, len(self.pts)
        self._reset_pairs(N)
        if not cfg.use_ordinal or N < 2:
            return
        sc = self.center
        u0 = np.where(geo["valid"][:, sc], geo["u"][:, sc], np.nan).astype(float)
        z0 = geo["Z"][:, sc]
        with np.errstate(invalid="ignore"):
            du = np.abs(u0[:, None] - u0[None, :])
            dz = z0[None, :] - z0[:, None]
            sep = np.maximum(cfg.ordinal_min_sep_abs,
                             cfg.ordinal_min_sep_rel * np.minimum(z0[:, None], z0[None, :]))
            near = (du <= cfg.ordinal_window_px) & (dz > sep)
        i, k = np.nonzero(near)
        if len(i) == 0:
            return
        w = np.exp(-du[i, k] / cfg.ordinal_window_px)
        deg = np.bincount(i, w, N) + np.bincount(k, w, N)
        w = w / np.maximum(1.0, np.maximum(deg[i], deg[k]))
        exp = np.maximum(self.params.a * (1.0 / z0[i] - 1.0 / z0[k]), 1e-6)
        self.pair_i, self.pair_k, self.pair_w, self.pair_exp = i, k, w, exp

        node = np.concatenate([i, k])
        other = np.concatenate([k, i])
        ww = np.concatenate([w, w])
        ee = np.concatenate([exp, exp])
        sgn = np.concatenate([np.ones(len(i)), -np.ones(len(i))])
        order = np.argsort(node, kind="stable")
        node, other, ww, ee, sgn = node[order], other[order], ww[order], ee[order], sgn[order]
        st = np.searchsorted(node, np.arange(N + 1))
        self.nbrs = [(other[st[n]:st[n + 1]], ww[st[n]:st[n + 1]],
                      sgn[st[n]:st[n + 1]], ee[st[n]:st[n + 1]]) for n in range(N)]

    def _phi(self, d, exp=None):
        """Log-likelihood-ratio cost of a closeness difference d (nearer minus
        farther); negative when the image agrees with the radar."""
        cfg = self.cfg
        if cfg.ordinal_adaptive and exp is not None:
            x, tau = d / exp, cfg.ordinal_tau_rel
        else:
            x, tau = d, cfg.ordinal_tau
        sig = 0.5 * (1.0 + np.tanh(0.5 * x / tau))
        p = 0.5 * cfg.ordinal_eps + (1.0 - cfg.ordinal_eps) * sig
        return -cfg.ordinal_weight * np.log(p / 0.5)

    def _pair_diffs(self, A, chyp):
        if len(self.pair_i) == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        i, k = self.pair_i, self.pair_k
        ai, ak = A[i], A[k]
        ok = (ai < self.M) & (ak < self.M) & (ai != ak)
        d = chyp[i[ok], ai[ok]] - chyp[k[ok], ak[ok]]
        fin = np.isfinite(d)
        return d[fin], self.pair_w[ok][fin], self.pair_exp[ok][fin]

    def _ordinal_energy(self, A, chyp):
        if not self.cfg.use_ordinal:
            return 0.0
        d, w, e = self._pair_diffs(A, chyp)
        return float(np.sum(w * self._phi(d, e))) if len(d) else 0.0

    def _pair_cost_point(self, i, j, A, chyp):
        if not self.cfg.use_ordinal or j >= self.M:
            return 0.0
        nb, w, s, e = self.nbrs[i]
        if len(nb) == 0:
            return 0.0
        ci = chyp[i, j]
        if not np.isfinite(ci):
            return 0.0
        ao = A[nb]
        ok = (ao < self.M) & (ao != j)
        if not ok.any():
            return 0.0
        d = s[ok] * (ci - chyp[nb[ok], ao[ok]])
        fin = np.isfinite(d)
        return float(np.sum(w[ok][fin] * self._phi(d[fin], e[ok][fin])))

    def ordinal_stats(self, A, chyp):
        d, w, _ = self._pair_diffs(A, chyp)
        if len(d) == 0:
            return {"pairs": 0, "agreement": None}
        agree = np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))
        return {"pairs": int(len(d)), "agreement": round(float(np.sum(w * agree) / np.sum(w)), 3)}

    # ---------------- coupling and prior ---------------- #
    def _seg_energy(self, j, members, geo):
        if not self.cfg.use_extent:
            return 0.0
        lab = self.labels[j]
        if lab not in EXTENT or len(members) < 2:
            return 0.0
        idx = np.array(sorted(members), dtype=int)
        p = geo["xy_ego"][idx]
        d = np.linalg.norm(p - np.median(p, axis=0), axis=1)
        excess = np.maximum(0.0, d - EXTENT[lab] / 2)
        return float(np.sum(excess ** 2) / (2 * self.cfg.sigma_extent ** 2))

    def _prior(self, params):
        cfg, (c_g, Z_g) = self.cfg, self.ground
        e = 0.0
        if len(c_g):
            w = cfg.ground_weight / len(c_g)
            e += w * np.sum((c_g - params.a / Z_g - params.b) ** 2) / (2 * cfg.sigma_closeness ** 2)
        e += sum(y ** 2 for y in params.yaw.values()) / (2 * np.deg2rad(cfg.sigma_yaw_deg) ** 2)
        return float(e)

    def energy(self, A, U, geo, params, extra):
        unary = float(U[np.arange(len(A)), A].sum())
        coupling = sum(self._seg_energy(j, np.nonzero(A == j)[0], geo) for j in self.cand_idx)
        ordinal = self._ordinal_energy(A, extra["chyp"])
        prior = self._prior(params)
        return (unary + coupling + ordinal + prior,
                dict(unary=unary, coupling=coupling, ordinal=ordinal, prior=prior))

    def _evaluate(self, params, A):
        geo = self._geometry(params.yaw)
        U, extra = self._unary(params, geo)
        E, parts = self.energy(A, U, geo, params, extra)
        return E, parts, geo, U, extra

    # ---------------- assignment step: ICM ---------------- #
    def _icm(self, A, U, geo, chyp):
        """Returns (assignments, sweeps run, total point moves)."""
        A = A.copy()
        members = {j: set(np.nonzero(A == j)[0].tolist()) for j in range(self.M)}
        seg_E = {j: self._seg_energy(j, members[j], geo) for j in range(self.M)}

        total, sweeps = 0, 0
        for sweeps in range(1, self.cfg.icm_max_sweeps + 1):
            changed = 0
            for i in self.rng.permutation(len(A)):
                cur = A[i]
                cands = np.nonzero(U[i] <= U[i].min() + self.cfg.cand_margin)[0]
                e_cur_wo, d_rem = None, 0.0
                if cur < self.M:
                    e_cur_wo = self._seg_energy(cur, members[cur] - {i}, geo)
                    d_rem = e_cur_wo - seg_E[cur]
                pair_cur = self._pair_cost_point(i, cur, A, chyp)
                best_j, best_d, best_e = cur, 0.0, None
                for j in cands:
                    if j == cur:
                        continue
                    d, e_new = U[i, j] - U[i, cur] + d_rem, None
                    if j < self.M:
                        e_new = self._seg_energy(j, members[j] | {i}, geo)
                        d += e_new - seg_E[j]
                    d += self._pair_cost_point(i, j, A, chyp) - pair_cur
                    if d < best_d - 1e-9:
                        best_j, best_d, best_e = j, d, e_new
                if best_j != cur:
                    if cur < self.M:
                        members[cur].discard(i)
                        seg_E[cur] = e_cur_wo
                    if best_j < self.M:
                        members[best_j].add(i)
                        seg_E[best_j] = best_e
                    A[i] = best_j
                    changed += 1
            total += changed
            if not changed:
                break
        return A, sweeps, total

    # ---------------- parameter proposals and search ---------------- #
    def _anchors(self, A, extra):
        sel = np.nonzero(A < self.M)[0]
        return sel, extra["best"][sel, A[sel]]

    def _depth_proposal(self, A, extra, params):
        sel, s = self._anchors(A, extra)
        c_g, Z_g = self.ground
        c = np.concatenate([extra["c"][sel, s], c_g])
        Z = np.concatenate([extra["Z"][sel, s], Z_g])
        w = np.concatenate([np.ones(len(sel)),
                            np.full(len(c_g), self.cfg.ground_weight / max(len(c_g), 1))])
        ok = np.isfinite(c) & np.isfinite(Z)
        c, Z, w = c[ok], Z[ok], w[ok]
        if len(c) < 3:
            return None
        a, b = fit_affine_inverse(c, Z, w)
        return replace(params, a=a, b=b) if a > 0 else None

    def _sigma_proposal(self, A, extra, params):
        sel, s = self._anchors(A, extra)
        D, Z = extra["D"][sel, s], extra["Z"][sel, s]
        ok = np.isfinite(D)
        if ok.sum() < 3:
            return None
        res = 1.0 / D[ok] - 1.0 / Z[ok]
        sig = 1.4826 * np.median(np.abs(res - np.median(res)))
        return replace(params, sigma_inv=float(np.clip(sig, *self.cfg.sigma_inv_bounds)))

    def _try(self, cand, A, E_ref):
        E_c, parts_c, geo_c, U_c, extra_c = self._evaluate(cand, A)
        if E_c < E_ref - 1e-9:
            return E_c, cand, parts_c, geo_c, U_c, extra_c
        return None

    @staticmethod
    def _perturb(params, key, delta):
        if key == "log_a":
            return replace(params, a=params.a * float(np.exp(delta)))
        if key == "b":
            return replace(params, b=params.b + delta)
        yaw = dict(params.yaw)
        ch = key[4:]
        yaw[ch] = yaw.get(ch, 0.0) + delta
        return replace(params, yaw=yaw)

    # ---------------- main loop ---------------- #
    def optimise(self, verbose=True):
        cfg, params = self.cfg, self.params
        if len(self.pts) == 0 or len(self.cand_idx) == 0:
            raise ValueError("Need at least one radar point and one candidate segment.")

        geo = self._geometry(params.yaw)
        self._build_pairs(geo)                       # pairs fixed from initial geometry
        U, extra = self._unary(params, geo)
        A = U.argmin(1)
        self.A_init = A.copy()
        self.ordinal_init = self.ordinal_stats(A, extra["chyp"])
        E, parts = self.energy(A, U, geo, params, extra)
        history = [dict(iter=0, energy=E, **parts)]

        proposals = []
        if cfg.fit_depth_scale:
            proposals.append(("refit_scale", self._depth_proposal))
        if cfg.fit_sigma:
            proposals.append(("refit_sigma", self._sigma_proposal))

        steps, mins = {}, {}
        if cfg.fit_depth_scale:
            steps.update(log_a=cfg.scale_step_init, b=cfg.shift_step_init)
            mins.update(log_a=cfg.scale_step_min, b=cfg.scale_step_min)
        if cfg.fit_yaw:
            for ch in np.unique(self.pts.channel):
                steps[f"yaw:{ch}"] = np.deg2rad(cfg.yaw_step_init_deg)
                mins[f"yaw:{ch}"] = np.deg2rad(cfg.yaw_step_min_deg)

        stop_reason = "max_iter"
        for it in range(1, cfg.max_iter + 1):
            E_prev = E
            A, sweeps, moves = self._icm(A, U, geo, extra["chyp"])
            E, parts = self.energy(A, U, geo, params, extra)
            accepted = []

            for name, propose in proposals:
                cand = propose(A, extra, params)
                if cand is not None:
                    res = self._try(cand, A, E)
                    if res:
                        E, params, parts, geo, U, extra = res
                        accepted.append(name)

            for key in steps:
                if steps[key] < mins[key]:
                    continue
                best = None
                for sign in (1.0, -1.0):
                    res = self._try(self._perturb(params, key, sign * steps[key]), A,
                                    best[0] if best else E)
                    if res:
                        best = res
                if best:
                    E, params, parts, geo, U, extra = best
                    accepted.append(key)
                else:
                    steps[key] *= 0.5

            history.append(dict(iter=it, energy=E, **parts, icm_sweeps=sweeps,
                                point_moves=moves, accepted=accepted, decrease=E_prev - E))
            if verbose:
                print(f"  iter {it:2d}: E={E:9.2f} (-{E_prev - E:.3f})  moves {moves:3d} "
                      f"in {sweeps} sweeps  ordinal {parts['ordinal']:.1f}  "
                      f"accepted {accepted or '-'}")

            exhausted = all(steps[k] < mins[k] for k in steps)
            if it >= cfg.min_iter and moves == 0 and not accepted and exhausted:
                stop_reason = "converged"
                break

        self.stop_reason = stop_reason
        self.A, self.params, self.geo, self.U, self.extra = A, params, geo, U, extra
        self.history = history
        self.post, self.entropy = self._posteriors(A, U, geo, extra["chyp"])
        return self.summary()

    def _posteriors(self, A, U, geo, chyp):
        """Conditional posterior of each point's assignment, others held fixed."""
        members = {j: set(np.nonzero(A == j)[0].tolist()) for j in range(self.M)}
        post, ent = np.zeros(len(A)), np.zeros(len(A))
        for i in range(len(A)):
            cands = set(np.nonzero(U[i] <= U[i].min() + self.cfg.cand_margin)[0]) | {A[i]}
            e = np.full(self.M + 2, np.inf)
            for j in cands:
                e[j] = U[i, j] + self._pair_cost_point(i, j, A, chyp)
                if j < self.M:
                    base = members[j] - {i}
                    e[j] += self._seg_energy(j, base | {i}, geo) - self._seg_energy(j, base, geo)
            p = np.exp(-(e - e.min()))
            p /= p.sum()
            post[i] = p[A[i]]
            nz = p[p > 0]
            ent[i] = -np.sum(nz * np.log(nz))
        return post, ent

    # ---------------- outputs ---------------- #
    def summary(self):
        A, p = self.A, self.params
        E, parts = self.energy(A, self.U, self.geo, p, self.extra)
        terms = ["depth"] + (["extent"] if self.cfg.use_extent else []) \
            + (["ordinal"] if self.cfg.use_ordinal and self.cfg.ordinal_weight > 0 else [])
        return {
            "objective": "+".join(terms),
            "energy": round(E, 3),
            "energy_per_point": round(E / len(A), 3),
            "energy_parts": {k: round(v, 3) for k, v in parts.items()},
            "iterations": len(self.history) - 1,
            "stop_reason": getattr(self, "stop_reason", None),
            "params": {"a": round(p.a, 5), "b": round(p.b, 5),
                       "sigma_inv": round(p.sigma_inv, 5),
                       "yaw_deg": {k: round(float(np.rad2deg(v)), 3) for k, v in p.yaw.items()}},
            "n_points": int(len(A)),
            "n_assigned": int((A < self.M).sum()),
            "n_clutter": int((A == self.CLUTTER).sum()),
            "n_occluded": int((A == self.OCCLUDED).sum()),
            "mean_posterior": round(float(self.post.mean()), 3),
            "mean_entropy": round(float(self.entropy.mean()), 3),
            "ordinal": {"candidate_pairs": int(len(self.pair_i)),
                        "initial": getattr(self, "ordinal_init", None),
                        "final": self.ordinal_stats(A, self.extra["chyp"])},
            "history": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in h.items()}
                        for h in self.history],
        }

    def point_table(self):
        rows = []
        A, geo, extra, p = self.A, self.geo, self.extra, self.pts
        for i, j in enumerate(A):
            row = {
                "channel": str(p.channel[i]),
                "range_m": round(float(np.linalg.norm(p.xy[i])), 2),
                "v_r_mps": round(float(p.v_r[i]), 2),
                "rcs_dbsm": round(float(p.rcs[i]), 1),
                "posterior": round(float(self.post[i]), 3),
            }
            if j < self.M:
                s = extra["best"][i, j]
                row.update(assignment="segment", segment_id=int(self.ids[j]),
                           label=self.labels[j],
                           pixel_uv=[int(geo["u"][i, s]), int(geo["v"][i, s])],
                           camera_depth_m=_finite_or_none(geo["Z"][i, s]),
                           map_depth_m=_finite_or_none(extra["D"][i, s]),
                           segment_closeness=_finite_or_none(extra["chyp"][i, j], 4),
                           elevation_deg=round(float(np.rad2deg(geo["phi"][s])), 2),
                           height_m=round(float(geo["h"][i, s]), 2))
            else:
                row.update(assignment="clutter" if j == self.CLUTTER else "occluded",
                           segment_id=None, label=None)
            rows.append(row)
        return rows

    def annotate_results(self, moving_thresh=0.5):
        A, geo, extra, p = self.A, self.geo, self.extra, self.pts
        for j, r in enumerate(self.results):
            idx = np.nonzero(A == j)[0]
            r["radar_points"] = int(len(idx))
            if len(idx) == 0:
                r.update(radar_median_m=None, radar_speed_mps=None, radar_rcs_dbsm=None,
                         radar_moving=None, radar_confidence=None, radar_velocity_xy=None)
                continue
            s = extra["best"][idx, j]
            speed, vel_xy = abs(float(np.median(p.v_r[idx]))), None
            n = geo["n_ego"][idx]
            if len(idx) >= 3 and self.labels[j] in EXTENT:
                sv = np.linalg.svd(n, compute_uv=False)
                if sv[-1] / sv[0] > 0.2:          # enough spread in viewing angle
                    vel, *_ = np.linalg.lstsq(n, p.v_r[idx], rcond=None)
                    vel_xy = [round(float(x), 2) for x in vel]
                    speed = float(np.linalg.norm(vel))

            r.update(
                radar_median_m=round(float(np.median(geo["Z"][idx, s])), 2),
                radar_speed_mps=round(speed, 2),
                radar_rcs_dbsm=round(float(np.median(p.rcs[idx])), 1),
                radar_moving=bool(speed > moving_thresh),
                radar_confidence=round(float(self.post[idx].mean()), 3),
                radar_velocity_xy=vel_xy,
            )
