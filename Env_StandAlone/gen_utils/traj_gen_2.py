import torch
import math
from typing import Optional

def normalize(v: torch.Tensor, eps: float = 1e-8):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)

def quat_from_matrix(mat: torch.Tensor):
    m00 = mat[..., 0, 0]; m01 = mat[..., 0, 1]; m02 = mat[..., 0, 2]
    m10 = mat[..., 1, 0]; m11 = mat[..., 1, 1]; m12 = mat[..., 1, 2]
    m20 = mat[..., 2, 0]; m21 = mat[..., 2, 1]; m22 = mat[..., 2, 2]

    trace = m00 + m11 + m22
    qw = torch.zeros_like(trace); qx = torch.zeros_like(trace)
    qy = torch.zeros_like(trace); qz = torch.zeros_like(trace)

    mask = trace > 0
    if mask.any():
        s = torch.sqrt(trace[mask] + 1.0) * 2.0
        denom = s.clamp_min(1e-12)
        qw[mask] = 0.25 * s
        qx[mask] = (m21[mask] - m12[mask]) / denom
        qy[mask] = (m02[mask] - m20[mask]) / denom
        qz[mask] = (m10[mask] - m01[mask]) / denom

    mask0 = (~mask) & (m00 > m11) & (m00 > m22)
    if mask0.any():
        s0 = torch.sqrt(1.0 + m00[mask0] - m11[mask0] - m22[mask0]) * 2.0
        denom0 = s0.clamp_min(1e-12)
        qw[mask0] = (m21[mask0] - m12[mask0]) / denom0
        qx[mask0] = 0.25 * s0
        qy[mask0] = (m01[mask0] + m10[mask0]) / denom0
        qz[mask0] = (m02[mask0] + m20[mask0]) / denom0

    mask1 = (~mask) & (~mask0) & (m11 > m22)
    if mask1.any():
        s1 = torch.sqrt(1.0 + m11[mask1] - m00[mask1] - m22[mask1]) * 2.0
        denom1 = s1.clamp_min(1e-12)
        qw[mask1] = (m02[mask1] - m20[mask1]) / denom1
        qx[mask1] = (m01[mask1] + m10[mask1]) / denom1
        qy[mask1] = 0.25 * s1
        qz[mask1] = (m12[mask1] + m21[mask1]) / denom1

    mask2 = (~mask) & (~mask0) & (~mask1)
    if mask2.any():
        s2 = torch.sqrt(1.0 + m22[mask2] - m00[mask2] - m11[mask2]) * 2.0
        denom2 = s2.clamp_min(1e-12)
        qw[mask2] = (m10[mask2] - m01[mask2]) / denom2
        qx[mask2] = (m02[mask2] + m20[mask2]) / denom2
        qy[mask2] = (m12[mask2] + m21[mask2]) / denom2
        qz[mask2] = 0.25 * s2

    quat = torch.stack([qx, qy, qz, qw], dim=-1)
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)

def look_at_quaternion(forward: torch.Tensor, up: Optional[torch.Tensor] = None):
    if up is None:
        up = torch.tensor([0.0, 0.0, 1.0], device=forward.device, dtype=forward.dtype).expand_as(forward)
    f = normalize(forward)
    right = torch.cross(up, f, dim=-1)
    right = normalize(right)
    up_corr = torch.cross(f, right, dim=-1)
    mat = torch.stack([f, right, up_corr], dim=-1)
    return quat_from_matrix(mat)

def bimanual_arc_trajectories(
    point_cloud: torch.Tensor,
    pick_positions: torch.Tensor,
    arc_height: float = 0.12,
    min_pair_distance: float = 0.15,
    num_steps: int = 64,
    max_resample_attempts: int = 64,
    lateral_separation: Optional[float] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
):
    assert point_cloud.ndim == 3 and point_cloud.shape[2] == 3
    assert pick_positions.ndim == 3 and pick_positions.shape[1] == 2 and pick_positions.shape[2] == 3

    if device is None:
        device = point_cloud.device
    pc = point_cloud.to(device)
    picks = pick_positions.to(device)

    B, N, _ = pc.shape
    T = num_steps
    dtype = pc.dtype

    if lateral_separation is None:
        lateral_separation = max(0.05, 0.25 * float(min_pair_distance))

    t = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)
    arc_profile = torch.sin(math.pi * t).view(1, 1, T)

    valid_mask = torch.zeros(B, dtype=torch.bool, device=device)
    final_places = torch.zeros(B, 2, 3, device=device, dtype=dtype)

    gen = None
    if seed is not None:
        if hasattr(device, "type") and device.type == "cuda":
            gen = torch.Generator(device=device)
        else:
            gen = torch.Generator()
        gen.manual_seed(seed)

    attempts = 0

    # Precompute input pick distance constraint per batch
    input_pair_dist = (picks[:, 0] - picks[:, 1]).norm(dim=-1)  # (B,)

    # helper: 2D segment intersection test for two segments (p1->q1) and (p2->q2)
    def segments_intersect_2d(p1, q1, p2, q2):
        # p1,q1,p2,q2: (...,2)
        def orient(a, b, c):
            return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

        o1 = orient(p1, q1, p2)
        o2 = orient(p1, q1, q2)
        o3 = orient(p2, q2, p1)
        o4 = orient(p2, q2, q1)

        cond1 = (o1 * o2) < 0
        cond2 = (o3 * o4) < 0
        return cond1 & cond2

    while (~valid_mask).any() and attempts < max_resample_attempts:
        attempts += 1
        if gen is None:
            rand_scores = torch.rand(B, N, device=device, dtype=dtype)
        else:
            rand_scores = torch.rand(B, N, device=device, dtype=dtype, generator=gen)

        # --- Sample first target per batch (based on scores) ---
        first_idx = rand_scores.argmax(dim=1)  # (B,)
        first_pts = torch.gather(pc, 1, first_idx.view(B, 1, 1).expand(-1, 1, 3)).squeeze(1)  # (B,3)

        # If all batches already valid, break
        need_idx = (~valid_mask).nonzero(as_tuple=False).squeeze(-1)
        if need_idx.numel() == 0:
            break

        # Compute distances from first point to all points: (B, N)
        first_pts_exp = first_pts.unsqueeze(1).expand(-1, N, 3)
        dist_first_to_all = (pc - first_pts_exp).norm(dim=-1)  # (B, N)

        # Allowed max distance between sampled targets is the input pair distance
        allowed_dist = input_pair_dist.view(B, 1)  # (B,1)

        # Build mask of valid second candidates: within allowed distance and not equal to first index
        idx_range = torch.arange(N, device=device).view(1, N).expand(B, N)
        not_first_mask = idx_range != first_idx.view(B, 1)
        candidate_mask = (dist_first_to_all <= allowed_dist + 1e-8) & not_first_mask  # (B,N)

        # Determine which batches have any valid candidate
        has_candidate = candidate_mask.any(dim=1)  # (B,)

        # Prepare scores for second selection: mask out invalid candidates by setting score to -inf
        scores_second = rand_scores.clone()
        scores_second[~candidate_mask] = -1e9

        # pick second as top-1 of scores_second
        second_idx = scores_second.argmax(dim=1)  # (B,)

        # For batches where second candidate set is empty, fallback to top-2 distinct indices
        second_valid = has_candidate.clone()
        if (~second_valid).any():
            bad = (~second_valid).nonzero(as_tuple=False).squeeze(-1)
            if bad.numel() > 0:
                top2 = rand_scores[bad].topk(2, dim=1).indices  # (M,2)
                places_fallback = torch.gather(pc[bad], 1, top2.unsqueeze(-1).expand(-1, -1, 3))  # (M,2,3)
                # assign fallback second indices
                second_idx[bad] = top2[:, 1]
                # we'll compose places_candidate below and then re-evaluate constraints

        # Compose candidate pairs
        places_candidate = torch.stack([
            torch.gather(pc, 1, first_idx.view(B, 1, 1).expand(-1, 1, 3)).squeeze(1),
            torch.gather(pc, 1, second_idx.view(B, 1, 1).expand(-1, 1, 3)).squeeze(1),
        ], dim=1)  # (B,2,3)

        # Enforce min_pair_distance constraint (distance from picks to places)
        dif = picks - places_candidate  # (B,2,3)
        pair_dists = dif.norm(dim=-1)  # (B,2)
        min_ok = (pair_dists >= float(min_pair_distance)).all(dim=1)  # (B,)

        # Ensure the distance between the two sampled places <= input_pair_dist
        place_pair_dist = (places_candidate[:, 0] - places_candidate[:, 1]).norm(dim=-1)  # (B,)
        within_input_dist = place_pair_dist <= (input_pair_dist + 1e-8)

        # Check for crossing in 2D (x,y)
        p1 = picks[:, 0, :2]
        q1 = places_candidate[:, 0, :2]
        p2 = picks[:, 1, :2]
        q2 = places_candidate[:, 1, :2]
        no_cross = ~segments_intersect_2d(p1, q1, p2, q2)

        # A batch is successful in this attempt if all constraints hold
        success = min_ok & within_input_dist & no_cross

        if success.any():
            succ_idx = success.nonzero(as_tuple=False).squeeze(-1)
            final_places[succ_idx] = places_candidate[succ_idx]
            valid_mask[succ_idx] = True

        # continue loop to resample for batches that failed this attempt

    # Failsafe for any batches still invalid
    if (~valid_mask).any():
        invalid_idx = torch.nonzero(~valid_mask).squeeze(-1)
        if invalid_idx.numel() > 0:
            # compute distances from each pick to every point: (B,2,N)
            picks_exp = picks.unsqueeze(2)  # (B,2,1,3)
            pc_exp = pc.unsqueeze(1)        # (B,1,N,3)
            dists_all = (picks_exp - pc_exp).norm(dim=-1)  # (B,2,N)

            # restrict to invalid batches
            dists_inv = dists_all[invalid_idx]            # (M,2,N)

            M = dists_inv.shape[0]
            d0 = dists_inv[:, 0, :]  # (M,N)
            d1 = dists_inv[:, 1, :]  # (M,N)

            # For each point, determine which pick it's closest to
            closest_pick_per_point_inv = dists_inv.argmin(dim=1)  # (M,N)

            mask0 = closest_pick_per_point_inv == 0  # (M,N)
            mask1 = closest_pick_per_point_inv == 1  # (M,N)

            large_neg = -1e9

            # pick0: choose the farthest among points assigned to pick0, else farthest overall
            d0_masked = d0.clone()
            d0_masked[~mask0] = large_neg
            any_mask0 = mask0.any(dim=1)
            idx0 = d0_masked.argmax(dim=1)  # (M,)
            if (~any_mask0).any():
                idx0_no = (~any_mask0).nonzero(as_tuple=False).squeeze(-1)
                if idx0_no.numel() > 0:
                    idx0[idx0_no] = d0[idx0_no].argmax(dim=1)

            # pick1: same logic for pick1
            d1_masked = d1.clone()
            d1_masked[~mask1] = large_neg
            any_mask1 = mask1.any(dim=1)
            idx1 = d1_masked.argmax(dim=1)
            if (~any_mask1).any():
                idx1_no = (~any_mask1).nonzero(as_tuple=False).squeeze(-1)
                if idx1_no.numel() > 0:
                    idx1[idx1_no] = d1[idx1_no].argmax(dim=1)

            # resolve conflicts where idx1 == idx0
            conflict = idx1 == idx0
            if conflict.any():
                conf_idx = conflict.nonzero(as_tuple=False).squeeze(-1)
                if conf_idx.numel() > 0:
                    d1_alt = d1_masked.clone()
                    rows = conf_idx
                    cols = idx0[conf_idx]
                    d1_alt[rows, cols] = large_neg
                    idx1_alt = d1_alt.argmax(dim=1)
                    still_conflict = idx1_alt == idx0
                    if still_conflict.any():
                        sc_idx = still_conflict.nonzero(as_tuple=False).squeeze(-1)
                        for ii in sc_idx.tolist():
                            d1_b = d1[ii]
                            d1_b_ex = d1_b.clone()
                            d1_b_ex[idx0[ii]] = large_neg
                            idx1_alt[ii] = d1_b_ex.argmax()
                    idx1 = idx1_alt

            # gather failsafe places for invalid batches
            indices = torch.stack([idx0, idx1], dim=1)            # (M,2)
            indices_exp = indices.unsqueeze(-1).expand(-1, -1, 3) # (M,2,3)
            failsafe_places = torch.gather(pc[invalid_idx], 1, indices_exp)  # (M,2,3)

            final_places[invalid_idx] = failsafe_places
            valid_mask[invalid_idx] = True

    # All batches now have final_places
    picks_exp = picks.unsqueeze(2).expand(-1, -1, T, -1)
    places_exp = final_places.unsqueeze(2).expand(-1, -1, T, -1)
    t_view = t.view(1, 1, T, 1)
    pos_linear = (1.0 - t_view) * picks_exp + t_view * places_exp
    arc_vals = (float(arc_height) * arc_profile).to(dtype=dtype)
    pos = pos_linear.clone()
    pos[..., 2] = pos_linear[..., 2] + arc_vals

    dir_vec = (final_places - picks)
    dir_xy = dir_vec[..., :2]
    perp_xy = torch.stack([-dir_xy[..., 1], dir_xy[..., 0]], dim=-1)
    perp_xy = normalize(perp_xy)
    signs = torch.tensor([1.0, -1.0], device=device, dtype=dtype).view(1, 2, 1)
    lateral_offsets_xy = signs * lateral_separation * perp_xy
    lateral_offsets = lateral_offsets_xy.unsqueeze(2) * arc_profile.unsqueeze(-1)
    pos[..., :2] = pos[..., :2] + lateral_offsets

    forward = dir_vec.clone()
    zero_mask = forward.norm(dim=-1) < 1e-6
    forward[zero_mask] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    quats = look_at_quaternion(forward)
    quats_time = quats.unsqueeze(2).expand(-1, -1, T, -1)

    pose_tensor = torch.cat([pos, quats_time], dim=-1)
    return pose_tensor

def mask_actions_torch(classification: torch.Tensor, actions: torch.Tensor, inplace: bool = False):
    """
    classification: bool tensor of shape (B, 2)
    actions: numeric tensor of shape (B, 2, 7)
    Returns: actions with entries zeroed where classification is False
    """
    mask = classification.bool()            # shape (B, 2)
    mask_expanded = mask.unsqueeze(-1)      # shape (B, 2, 1)

    if inplace:
        actions *= mask_expanded.to(actions.dtype)
        return actions
    else:
        return actions * mask_expanded.to(actions.dtype)
