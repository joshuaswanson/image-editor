"""Vectorized SHARP .ply -> antimatter15 .splat conversion.

The vendored viewer ships a per-vertex Python converter; this does the same
transform with numpy so a 1.2M-gaussian scene converts in a second instead of a
minute. The .splat record is 32 bytes: position (3xf32), scale (3xf32), colour
(RGBA u8), rotation quaternion (4xu8, mapped to 0..255).
"""

from pathlib import Path

import numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def ply_to_splat(ply_path: str | Path, out_path: str | Path) -> Path:
    v = PlyData.read(str(ply_path))["vertex"]
    n = v.count

    pos = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)).astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    rgb = 0.5 + SH_C0 * f_dc
    rgba = np.concatenate([rgb, opacity[:, None]], axis=1)
    rgba_u8 = (rgba * 255).clip(0, 255).astype(np.uint8)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    rot_u8 = (rot * 128 + 128).clip(0, 255).astype(np.uint8)

    # Same ordering the viewer expects (largest, most opaque splats first).
    order = np.argsort(-(scale.prod(axis=1)) * opacity)

    buf = np.zeros((n, 32), np.uint8)
    buf[:, 0:12] = pos[order].view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = scale[order].view(np.uint8).reshape(n, 12)
    buf[:, 24:28] = rgba_u8[order]
    buf[:, 28:32] = rot_u8[order]

    out_path = Path(out_path)
    out_path.write_bytes(buf.tobytes())
    return out_path
