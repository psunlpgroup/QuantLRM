import os
import math
import numpy as np
import torch
from typing import Iterable, List, Optional, Tuple


# =========================
#  Global stats (positives only): x_min, x_max, exact median (x_mid)
# =========================
def compute_global_stats_pos_only(
    matrices: Iterable[torch.Tensor],
    memmap_path: str = "tmp",
    dtype_for_stats: torch.dtype = torch.float32,
    cleanup_memmap: bool = True,
    print_summary: bool = True,
) -> Tuple[float, float, float, int, int]:
    """
    Compute global min, max, and exact median **over strictly positive values only**.
    Zeros are ignored. If you want to inspect the memmap, set cleanup_memmap=False.

    Returns:
        x_min_pos (float), x_max_pos (float), x_mid_pos (float),
        pos_count (int), zero_count (int)
    """
    matrices_list = list(matrices)  # allow two passes
    pos_min = math.inf
    pos_max = -math.inf
    pos_count = 0
    zero_count = 0

    # Pass 1: gather counts and positive min/max (on per-matrix CPU chunks)
    for mat in matrices_list:
        cpu = mat.detach().to("cpu", dtype=dtype_for_stats, non_blocking=False)
        flat = cpu.view(-1)
        zeros = (flat == 0).sum().item()
        zero_count += int(zeros)

        # strictly positive mask
        pos = flat[flat > 0]
        npos = pos.numel()
        pos_count += int(npos)
        if npos > 0:
            pmin = pos.min().item()
            pmax = pos.max().item()
            if pmin < pos_min:
                pos_min = pmin
            if pmax > pos_max:
                pos_max = pmax

        del cpu, flat, pos

    if pos_count == 0:
        raise ValueError(
            "All values are zero (no strictly positive entries). "
            "Cannot define positive-only stats."
        )

    # Pass 2: write strictly positive values into a memmap to compute exact median
    mm = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(pos_count,))
    write_ptr = 0
    for mat in matrices_list:
        cpu = mat.detach().to("cpu", dtype=dtype_for_stats, non_blocking=False)
        flat = cpu.view(-1)
        pos = flat[flat > 0]  # strictly positive
        n = pos.numel()
        if n > 0:
            mm[write_ptr:write_ptr + n] = pos.numpy()
            write_ptr += n
        del cpu, flat, pos
    mm.flush()

    # Exact median via np.partition on memmap of positives
    N = pos_count
    if N % 2 == 1:
        k = N // 2
        np.partition(mm, k)
        x_mid = float(mm[k])
    else:
        k1, k2 = N // 2 - 1, N // 2
        np.partition(mm, [k1, k2])
        x_mid = float((mm[k1] + mm[k2]) / 2.0)

    # Cleanup memmap
    del mm
    if cleanup_memmap:
        try:
            os.remove(memmap_path)
        except OSError:
            pass

    if print_summary:
        print(f"[stats+] x_min_pos={pos_min:.6g}, x_max_pos={pos_max:.6g}, x_mid_pos={x_mid:.6g}, "
              f"pos_count={pos_count}, zero_count={zero_count}")

    return float(pos_min), float(pos_max), float(x_mid), int(pos_count), int(zero_count)


# =========================
#  Core (float32-safe) branch formulas (ignoring zeros at call site)
# =========================
def _left_quadratic(
    x: torch.Tensor, R: float, x_mid: float, left_den: float, clamp_unit: bool
) -> torch.Tensor:
    """
    y_left(x) = 1 + (R - 1) * ((x_mid - x) / left_den)^2
    Assumes x > 0. Zeros are handled outside this function.
    """
    dev, dt = x.device, x.dtype
    xmid_t = torch.tensor(x_mid, device=dev, dtype=dt)
    lden_t = torch.tensor(left_den, device=dev, dtype=dt)
    one    = torch.tensor(1.0, device=dev, dtype=dt)
    Rm1    = torch.tensor(R - 1.0, device=dev, dtype=dt)

    z = (xmid_t - x) / lden_t
    if clamp_unit:
        z = z.clamp(-1, 1)
    return one + Rm1 * (z * z)


def _right_quadratic(
    x: torch.Tensor, R: float, x_mid: float, right_den: float, clamp_unit: bool
) -> torch.Tensor:
    """
    y_right(x) = 1 + (R - 1) * ((x - x_mid) / right_den)^2
    Assumes x > 0. Zeros are handled outside this function.
    """
    dev, dt = x.device, x.dtype
    xmid_t = torch.tensor(x_mid, device=dev, dtype=dt)
    rden_t = torch.tensor(right_den, device=dev, dtype=dt)
    one    = torch.tensor(1.0, device=dev, dtype=dt)
    Rm1    = torch.tensor(R - 1.0, device=dev, dtype=dt)

    z = (x - xmid_t) / rden_t
    if clamp_unit:
        z = z.clamp(-1, 1)
    return one + Rm1 * (z * z)


def _apply_piecewise_two_quadratics_ignore_zero_out_of_place(
    x: torch.Tensor,
    *,
    R: float,
    x_mid: float,
    left_den: float,
    right_den: float,
    clamp_unit: bool = True,
    upcast_to_float32: bool = True,
) -> torch.Tensor:
    """
    Return y with:
      - if x == 0 → y = 1  (ignored)
      - else if x <= x_mid → left quadratic
      - else               → right quadratic
    """
    orig_dtype = x.dtype
    if upcast_to_float32 and x.dtype in (torch.float16, torch.bfloat16):
        x_work = x.to(dtype=torch.float32)
    else:
        x_work = x

    x_work = x_work.to("cpu")
    dev, dt = x_work.device, x_work.dtype
    xmid_t = torch.tensor(x_mid, device=dev, dtype=dt)
    one    = torch.tensor(1.0, device=dev, dtype=dt)

    y = torch.empty_like(x_work)

    zero_mask = (x_work == 0)
    nonzero   = ~zero_mask
    left_mask = nonzero & (x_work <= xmid_t)
    right_mask = nonzero & ~left_mask  # i.e., x > x_mid and x != 0

    if zero_mask.any().item():
        y[zero_mask] = one  # y=1 for zeros

    if left_mask.any().item():
        y[left_mask] = _left_quadratic(
            x_work[left_mask], R, x_mid, left_den, clamp_unit
        )

    if right_mask.any().item():
        y[right_mask] = _right_quadratic(
            x_work[right_mask], R, x_mid, right_den, clamp_unit
        )

    if y.dtype != orig_dtype:
        y = y.to(dtype=orig_dtype)
    return y


# =========================
#  In-place, autograd-safe application (zeros → y=1)
# =========================
def apply_two_quadratics_ignore_zero_inplace(
    matrices: Iterable[torch.Tensor],
    x_min_pos: float,
    x_max_pos: float,
    x_mid_pos: float,
    *,
    eps: float = 1e-30,
    clamp_unit: bool = True,
    upcast_to_float32: bool = True,
) -> float:
    """
    Apply the two-quadratic mapping in-place under torch.no_grad(),
    ignoring x==0 by setting y=1 for zeros.

      R        = x_max_pos / max(x_min_pos, eps)
      left_den = max(x_mid_pos - x_min_pos, eps)
      right_den= max(x_max_pos - x_mid_pos, eps)
    """
    R = 10
    left_den  = max(x_mid_pos - x_min_pos, eps)
    right_den = max(x_max_pos - x_mid_pos, eps)

    print(f"[transform+] R={R:.6g}, left_den={left_den:.6g}, right_den={right_den:.6g} (zeros → y=1)")

    with torch.no_grad():
        for mat in matrices:
            y = _apply_piecewise_two_quadratics_ignore_zero_out_of_place(
                mat,
                R=R,
                x_mid=x_mid_pos,
                left_den=left_den,
                right_den=right_den,
                clamp_unit=clamp_unit,
                upcast_to_float32=upcast_to_float32,
            )
            mat.copy_(y)
    return R


# =========================
#  End-to-end convenience
# =========================
def piecewise_two_quadratics_ignore_zero_inplace(
    matrices: Iterable[torch.Tensor],
    *,
    memmap_path: str = "tmp",
    dtype_for_stats: torch.dtype = torch.float32,
    cleanup_memmap: bool = True,
    eps: float = 1e-30,
    clamp_unit: bool = True,
    upcast_to_float32: bool = True,
) -> Tuple[float, float, float, float, int, int]:
    """
    1) Compute positive-only global stats (x_min, x_max, exact median), ignoring zeros.
    2) Apply the two-quadratic transform in-place with zeros mapped to y=1.

    Returns:
        (x_min_pos, x_max_pos, x_mid_pos, R_used, pos_count, zero_count)
    """
    matrices_list = list(matrices)
    x_min_pos, x_max_pos, x_mid_pos, pos_count, zero_count = compute_global_stats_pos_only(
        matrices_list,
        memmap_path=memmap_path,
        dtype_for_stats=dtype_for_stats,
        cleanup_memmap=cleanup_memmap,
        print_summary=True,
    )
    R_used = apply_two_quadratics_ignore_zero_inplace(
        matrices_list,
        x_min_pos, x_max_pos, x_mid_pos,
        eps=eps,
        clamp_unit=clamp_unit,
        upcast_to_float32=upcast_to_float32,
    )
    return x_min_pos, x_max_pos, x_mid_pos, R_used, pos_count, zero_count


# =========================
#  (Optional) save helpers
# =========================
def save_matrices_pt(
    matrices: Iterable[torch.Tensor],
    out_dirs: List[str],
    filenames: Optional[List[str]] = None,
    move_to_cpu_on_save: bool = True,
) -> List[str]:
    matrices_list = list(matrices)
    if len(out_dirs) != len(matrices_list):
        raise ValueError(
            f"out_dirs length ({len(out_dirs)}) must equal number of matrices ({len(matrices_list)})."
        )
    if filenames is None:
        filenames = ["weights.pt"] * len(out_dirs)
    elif len(filenames) != len(out_dirs):
        raise ValueError("filenames must be None or have same length as out_dirs.")

    paths: List[str] = []
    for mat, d, fname in zip(matrices_list, out_dirs, filenames):
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, fname)
        tensor_to_save = mat.detach().cpu() if move_to_cpu_on_save else mat.detach()
        torch.save(tensor_to_save, path)
        paths.append(path)
    return paths


# =========================
#  Example usage
# =========================
if __name__ == "__main__":
    # Make some demo tensors with a few zeros mixed in
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    matrix_types = ["o", "down"]
    mats: List[torch.Tensor] = []
    for layer in range(32):
        print(layer)
        for i, matrix_name in enumerate(matrix_types):
            dev = devices[i % len(devices)]
            weight_diff = torch.load("weight_difference/" + matrix_name + "/" + str(layer) + ".pt", map_location="cpu")
            mats.append(torch.nn.Parameter(weight_diff.to(dev), requires_grad=True))

    # Transform in-place, ignoring zeros (zeros → y=1)
    x_min_pos, x_max_pos, x_mid_pos, R_used, pos_cnt, zero_cnt = piecewise_two_quadratics_ignore_zero_inplace(mats, memmap_path="tmp", cleanup_memmap=True, upcast_to_float32=True)

    print(f"[done] x_min_pos={x_min_pos:.6g}, x_max_pos={x_max_pos:.6g}, "
          f"x_mid_pos={x_mid_pos:.6g}, R={R_used:.6g}, "
          f"positives={pos_cnt}, zeros={zero_cnt}")

    # (Optional) save results
    out_dirs, filenames = [], []
    for layer in range(32):
        for i, matrix_name in enumerate(matrix_types):
            out_dirs.append("quadratic/" + matrix_name)
            filenames.append(str(layer) + ".pt")

    paths = save_matrices_pt(mats, out_dirs, filenames=filenames, move_to_cpu_on_save=True)
    print("SAVED!")

