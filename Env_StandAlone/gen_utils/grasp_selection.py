import torch
from typing import Optional

def _knn_indices(x: torch.Tensor, k: int) -> torch.LongTensor:
    N = x.shape[0]
    if N == 0:
        return torch.empty((0, k), dtype=torch.long, device=x.device)
    dist = torch.cdist(x, x, p=2.0)
    idx = torch.arange(N, device=x.device)
    dist[idx, idx] = float('inf')
    k_sel = min(k, N - 1) if N > 1 else 0
    if k_sel == 0:
        out = idx.unsqueeze(1).repeat(1, k)
        return out
    vals, idxs = torch.topk(dist, k=k_sel, largest=False, dim=1)
    if k_sel < k:
        pad_count = k - k_sel
        pad_idx = idxs[:, -1].unsqueeze(1).repeat(1, pad_count)
        idxs = torch.cat([idxs, pad_idx], dim=1)
    return idxs

# def _local_pca_normal(neighbors: torch.Tensor):
#     centered = neighbors - neighbors.mean(dim=0, keepdim=True)
#     denom = max(neighbors.shape[0] - 1, 1)
#     cov = centered.t() @ centered / denom
#     eigvals, eigvecs = torch.linalg.eigh(cov)
#     normal = eigvecs[:, 0]
#     e1 = eigvecs[:, -1]
#     e2 = eigvecs[:, -2]
#     return normal, e1, e2, eigvals

import torch
from typing import Tuple


def _local_pca_normal(neighbors: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a local surface normal and two tangent directions from neighbor points.

    Inputs:
    - neighbors: Tensor of shape (M, 3)
    - eps: small regularizer added to covariance diagonal

    Returns:
    - normal: (3,) tensor (unit length) smallest-variance direction
    - e1: (3,) tensor (unit length) largest-variance direction
    - e2: (3,) tensor (unit length) orthogonal tangent
    - eigvals: (3,) tensor of eigenvalues in ascending order
    """
    if neighbors.ndim != 2 or neighbors.shape[1] != 3:
        raise ValueError("neighbors must have shape (M, 3)")

    device = neighbors.device
    dtype = neighbors.dtype

    if not torch.isfinite(neighbors).all():
        normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        e1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        e2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        eigvals = torch.zeros(3, device=device, dtype=dtype)
        return normal, e1, e2, eigvals

    M = neighbors.shape[0]
    if M <= 1:
        normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        e1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        e2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        eigvals = torch.zeros(3, device=device, dtype=dtype)
        return normal, e1, e2, eigvals

    centered = neighbors - neighbors.mean(dim=0, keepdim=True)

    # Use higher precision for covariance if input is float32
    use_high_precision = (dtype == torch.float32)
    centered64 = centered.to(torch.float64) if use_high_precision else centered
    denom = max(M - 1, 1)
    cov = (centered64.t() @ centered64) / float(denom)
    cov = cov + (eps * torch.eye(3, device=device, dtype=cov.dtype))

    try:
        eigvals64, eigvecs64 = torch.linalg.eigh(cov)
        normal64 = eigvecs64[:, 0]
        e1_64 = eigvecs64[:, -1]
        e2_64 = eigvecs64[:, -2]
    except RuntimeError:
        # fallback to SVD
        try:
            U, S, Vh = torch.linalg.svd(centered64, full_matrices=False)
            V = Vh.T
            # If V has >=3 columns use them, else pad
            if V.shape[1] >= 3:
                e1_64 = V[:, 0]
                e2_64 = V[:, 1]
                normal64 = V[:, -1]
            else:
                pad = torch.eye(3, device=device, dtype=V.dtype)
                Vpad = pad.clone()
                Vpad[:, :V.shape[1]] = V
                e1_64 = Vpad[:, 0]
                e2_64 = Vpad[:, 1]
                normal64 = Vpad[:, 2]
            eigvals_from_s = (S ** 2) / float(denom)
            eigvals64 = torch.cat([eigvals_from_s.new_zeros(3 - eigvals_from_s.numel()), eigvals_from_s]).flip(0)
        except Exception:
            normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
            e1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
            e2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
            eigvals = torch.zeros(3, device=device, dtype=dtype)
            return normal, e1, e2, eigvals

    # convert back to original dtype if needed
    if use_high_precision:
        normal = normal64.to(dtype)
        e1 = e1_64.to(dtype)
        e2 = e2_64.to(dtype)
        eigvals = eigvals64.to(dtype)
    else:
        normal = normal64
        e1 = e1_64
        e2 = e2_64
        eigvals = eigvals64

    def safe_normalize(v: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        nrm = torch.norm(v)
        if nrm < 1e-8:
            return fallback.to(v.device, dtype=v.dtype)
        return v / nrm

    fallback_normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=normal.dtype)
    fallback_e1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=normal.dtype)
    fallback_e2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=normal.dtype)

    normal = safe_normalize(normal, fallback_normal)
    e1 = safe_normalize(e1, fallback_e1)

    # Use torch.linalg.cross to avoid the deprecation warning
    e2 = torch.linalg.cross(normal, e1)
    e2 = safe_normalize(e2, fallback_e2)

    # Re-orthogonalize e1 to be perpendicular to normal
    e1 = e1 - normal * (normal @ e1)
    e1 = safe_normalize(e1, fallback_e1)

    # Ensure eigvals length and ascending order
    if eigvals.numel() < 3:
        pad = torch.zeros(3 - eigvals.numel(), device=device, dtype=eigvals.dtype)
        eigvals = torch.cat([eigvals, pad], dim=0)
    eigvals, _ = torch.sort(eigvals)

    return normal, e1, e2, eigvals


def _vec_to_quaternion_from_z(normal: torch.Tensor):
    z = torch.tensor([0.0, 0.0, 1.0], device=normal.device, dtype=normal.dtype)
    n = normal / (normal.norm() + 1e-12)
    dot = torch.clamp((z * n).sum(), -1.0, 1.0)
    if dot > 0.999999:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=normal.device, dtype=normal.dtype)
    if dot < -0.999999:
        axis = torch.tensor([1.0, 0.0, 0.0], device=normal.device, dtype=normal.dtype)
        q = torch.stack([torch.tensor(0.0, device=normal.device, dtype=normal.dtype),
                         axis[0], axis[1], axis[2]])
        return q / (q.norm() + 1e-12)
    axis = torch.linalg.cross(z, n)
    axis_norm = axis / (axis.norm() + 1e-12)
    angle = torch.acos(dot)
    half = angle * 0.5
    w = torch.cos(half)
    s = torch.sin(half)
    q = torch.stack([w, axis_norm[0] * s, axis_norm[1] * s, axis_norm[2] * s])
    return q / (q.norm() + 1e-12)

def select_two_boundary_picks_batch(
    points_batch: torch.Tensor,
    min_distance: float,
    k: int = 30,
    gap_thresh: float = 1.0,
    max_attempts: int = 50,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Inputs:
    - points_batch: Tensor of shape (B, N, 3)
    - min_distance: float

    Output:
    - picks: Tensor of shape (B, 2, 7) with per-pick layout [x, y, z, w, qx, qy, qz]
    """
    if device is None:
        device = points_batch.device

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    B, N, _ = points_batch.shape
    pos_out = torch.zeros((B, 2, 3), dtype=points_batch.dtype, device=device)
    quat_out = torch.zeros((B, 2, 4), dtype=points_batch.dtype, device=device)

    idx_1 = torch.zeros((B, 1))
    idx_2 = torch.zeros((B, 1))
    idx_1 = []
    idx_2 = []

    for b in range(B):
        pts = points_batch[b].to(device)
        knn_idx = _knn_indices(pts, k=k)
        scores = torch.zeros((N,), dtype=pts.dtype, device=device)
        normals = torch.zeros((N, 3), dtype=pts.dtype, device=device)

        for i in range(N):
            nbr_idx = knn_idx[i]
            nbr = pts[nbr_idx]
            normal, e1, e2, eigvals = _local_pca_normal(nbr)
            normals[i] = normal
            rel = (nbr - pts[i])
            x = (rel @ e1)
            y = (rel @ e2)
            ang = torch.atan2(y, x)
            ang_sorted, _ = torch.sort(ang)
            if ang_sorted.numel() > 1:
                diffs = ang_sorted[1:] - ang_sorted[:-1]
                wrap = (ang_sorted[0] + 2.0 * torch.pi) - ang_sorted[-1]
                diffs = torch.cat([diffs, wrap.unsqueeze(0)], dim=0)
                max_gap = diffs.max()
            else:
                max_gap = torch.tensor(2.0 * torch.pi, device=device, dtype=pts.dtype)
            scores[i] = max_gap

        candidates = torch.where(scores >= gap_thresh)[0]
        if candidates.numel() == 0:
            topk = min(max(min(N // 5, N), 8), N)
            vals, ids = torch.topk(scores, topk, largest=True)
            candidates = ids

        cand_scores = scores[candidates]
        cand_scores = cand_scores - cand_scores.min() + 1e-6
        probs = cand_scores / cand_scores.sum()
        if candidates.numel() == 1:
            idx1 = int(candidates[0].item())
        else:
            if gen is None:
                idx_choice = torch.multinomial(probs, 1).item()
            else:
                idx_choice = torch.multinomial(probs, 1, generator=gen).item()
            idx1 = int(candidates[idx_choice].item())
        pos1 = pts[idx1]

        other_candidates = candidates[candidates != idx1]
        if other_candidates.numel() == 0:
            dists = torch.cdist(pos1.unsqueeze(0), pts).squeeze(0)
            dists[idx1] = -1.0
            idx2 = int(torch.argmax(dists).item())
        else:
            cand_scores2 = scores[other_candidates]
            cand_scores2 = cand_scores2 - cand_scores2.min() + 1e-6
            probs2 = cand_scores2 / cand_scores2.sum()
            chosen = None
            for _ in range(max_attempts):
                if gen is None:
                    sel = torch.multinomial(probs2, 1).item()
                else:
                    sel = torch.multinomial(probs2, 1, generator=gen).item()
                cand_idx = int(other_candidates[sel].item())
                if torch.dist(pts[cand_idx], pos1) >= min_distance:
                    chosen = cand_idx
                    break
            if chosen is None:
                dists = torch.cdist(pos1.unsqueeze(0), pts[other_candidates]).squeeze(0)
                sel = torch.argmax(dists).item()
                chosen = int(other_candidates[sel].item())
            idx2 = chosen
        idx_1.append(idx1)
        idx_2.append(idx2)
        pos_out[b, 0] = pts[idx1]
        pos_out[b, 1] = pts[idx2]
        q1 = _vec_to_quaternion_from_z(normals[idx1])
        q2 = _vec_to_quaternion_from_z(normals[idx2])
        quat_out[b, 0] = q1
        quat_out[b, 1] = q2

    picks = torch.cat([pos_out, quat_out], dim=2)
    return picks, idx_1, idx_2
