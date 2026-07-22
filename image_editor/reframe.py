"""Reframe a photo by moving the camera, then clean the holes with FLUX.1-Fill.

Pipeline:
  1. SHARP turns the photo into a 3D Gaussian splat (run separately via
     `sharp predict`, or pass an existing .ply).
  2. This module renders the splat from a moved camera. Moving the camera
     reveals areas the original photo never saw, which come out as holes.
  3. Those holes are handed to FLUX.1-Fill (the same fill this repo already
     uses) to paint plausible content, so the reframed view looks complete.

The renderer is an occlusion-aware, alpha-weighted splatter (torch, runs on
CPU/MPS). SHARP's gsplat renderer is CUDA-only, so this is the Mac-friendly
substitute: it blends overlapping gaussians instead of overwriting, so flat
regions are smooth rather than speckled, respects a soft depth buffer so the
background does not bleed through the subject, and returns a clean coverage map
that becomes the fill mask.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

SH_C0 = 0.28209479177387814  # 0th-order spherical-harmonic coefficient


def focal_px(width: int, height: int, f_mm: float = 30.0) -> float:
    """SHARP's focal-length convention (30mm-equivalent on a full-frame diagonal)."""
    return f_mm * np.sqrt(width**2 + height**2) / np.sqrt(36**2 + 24**2)


def load_gaussians(ply_path: str | Path) -> dict:
    """Load Gaussian centers, colors, opacities and sizes from a SHARP .ply."""
    v = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0).astype(np.float32)
    opacity = (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))).astype(np.float32)  # sigmoid
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32))
    return {"xyz": xyz, "rgb": rgb, "opacity": opacity, "scale": scale.mean(axis=1)}


def remove_outliers(g: dict, k: int = 16, std_ratio: float = 2.0) -> dict:
    """Drop floater gaussians whose mean distance to k neighbors is an outlier.

    SHARP leaves a few stray gaussians floating off the surface. They project
    onto the background as isolated specks that the coverage mask cannot catch,
    so it is cleaner to remove them from the cloud before rendering.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(g["xyz"])
    dist, _ = tree.query(g["xyz"], k=k + 1, workers=-1)  # +1 for the point itself
    mean_d = dist[:, 1:].mean(axis=1)
    keep = mean_d < mean_d.mean() + std_ratio * mean_d.std()
    return {key: val[keep] for key, val in g.items()}


def look_at(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-to-camera rotation for a camera at `center` looking at `target`.

    Uses the OpenCV convention SHARP follows: x right, y down, z forward.
    """
    world_up = np.array([0.0, -1.0, 0.0])  # +y is down, so "up" is -y
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, world_up)  # ordered so the original camera gives identity
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)  # rows are camera axes = world->camera


def render(
    g: dict,
    K: np.ndarray,
    R_w2c: np.ndarray,
    center: np.ndarray,
    width: int,
    height: int,
    max_radius: int = 3,
    rel_tol: float = 0.05,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Alpha-weighted, occlusion-aware splatting. Returns (image, coverage)."""
    dev = torch.device(device)
    X = torch.from_numpy(g["xyz"]).to(dev)
    C = torch.from_numpy(g["rgb"]).to(dev)
    op = torch.from_numpy(g["opacity"]).to(dev)
    scale = torch.from_numpy(g["scale"]).to(dev)
    R = torch.from_numpy(R_w2c.astype(np.float32)).to(dev)
    cen = torch.from_numpy(center.astype(np.float32)).to(dev)
    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))

    cam = (X - cen) @ R.T
    z = cam[:, 2]
    front = z > 1e-3
    cam, C, op, scale, z = cam[front], C[front], op[front], scale[front], z[front]

    u = fx * cam[:, 0] / z + cx
    v = fy * cam[:, 1] / z + cy
    ui, vi = torch.round(u).long(), torch.round(v).long()
    inb = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    # Per-gaussian screen-space sigma (px), clamped to fit the splat window.
    sigma = (fx * scale / z).clamp(0.7, float(max_radius))

    offsets = [(du, dv) for dv in range(-max_radius, max_radius + 1)
               for du in range(-max_radius, max_radius + 1)]
    P = height * width

    # Pass 1: soft depth buffer (nearest surface per pixel).
    zbuf = torch.full((P,), float("inf"), device=dev)
    for du, dv in offsets:
        uu, vv = ui + du, vi + dv
        m = inb & (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        m = m & ((du * du + dv * dv) <= (2.5 * sigma) ** 2)
        idx = (vv * width + uu)[m]
        zbuf.scatter_reduce_(0, idx, z[m], reduce="amin", include_self=True)

    # Pass 2: accumulate alpha-weighted colour with a soft depth falloff, so a
    # gaussian just behind the front surface still contributes (filling gaps at
    # depth edges) while far-behind gaussians are smoothly suppressed.
    csum = torch.zeros((P, 3), device=dev)
    wsum = torch.zeros((P,), device=dev)
    for du, dv in offsets:
        uu, vv = ui + du, vi + dv
        m = inb & (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        d2 = float(du * du + dv * dv)
        w = op * torch.exp(-0.5 * d2 / (sigma * sigma))
        m = m & (d2 <= (2.5 * sigma) ** 2) & (w > 0.01)
        idx = (vv * width + uu)[m]
        behind = (z[m] - zbuf[idx]).clamp(min=0.0)
        depth_w = torch.exp(-behind / (rel_tol * zbuf[idx] + 1e-6))
        wk = w[m] * depth_w
        keep = wk > 1e-4
        idx, wk, ck = idx[keep], wk[keep], C[m][keep]
        csum.index_add_(0, idx, ck * wk[:, None])
        wsum.index_add_(0, idx, wk)

    img = (csum / wsum.clamp_min(1e-6)[:, None]).reshape(height, width, 3)
    weight = wsum.reshape(height, width)
    return img.cpu().numpy(), weight.cpu().numpy()


def reframe(
    ply_path: str | Path,
    width: int,
    height: int,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    max_radius: int = 3,
    dilate: int = 2,
    rel_tol: float = 0.05,
    cov_thresh: float = 0.15,
    clean: bool = True,
    device: str = "cpu",
) -> tuple[Image.Image, Image.Image]:
    """Render a reframed view and its hole mask.

    dx/dy/dz move the camera (in SHARP's metric world units) while it keeps
    looking at the scene center, giving an orbit-like reframe. Returns the
    reframed image and a fill mask (white where holes need to be painted).
    Pixels whose accumulated coverage weight is below `cov_thresh` (holes and
    the thin, sparsely covered fringe) are filled; `dilate` grows the mask so
    its seam is regenerated too.
    """
    from scipy.ndimage import binary_dilation

    g = load_gaussians(ply_path)
    if clean:
        g = remove_outliers(g)
    f = focal_px(width, height)
    K = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]])

    target = np.array([0.0, 0.0, float(np.median(g["xyz"][:, 2]))])
    center = np.array([dx, dy, dz])
    R_w2c = look_at(center, target)

    img, weight = render(g, K, R_w2c, center, width, height, max_radius, rel_tol, device)
    holes = weight < cov_thresh
    if dilate > 0:
        holes = binary_dilation(holes, iterations=dilate)

    image = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")
    mask = Image.fromarray(np.where(holes, 255, 0).astype(np.uint8), "L").convert("RGB")
    return image, mask


def main():
    parser = argparse.ArgumentParser(description="Reframe a photo via SHARP + FLUX.1-Fill.")
    parser.add_argument("ply", help="Path to a SHARP .ply (from `sharp predict`)")
    parser.add_argument("--width", type=int, required=True, help="Original image width")
    parser.add_argument("--height", type=int, required=True, help="Original image height")
    parser.add_argument("--dx", type=float, default=0.08, help="Camera shift x (metric)")
    parser.add_argument("--dy", type=float, default=0.0, help="Camera shift y (metric)")
    parser.add_argument("--dz", type=float, default=0.0, help="Camera shift z (metric)")
    parser.add_argument("--max-radius", type=int, default=3, help="Max splat radius in pixels")
    parser.add_argument("--dilate", type=int, default=2, help="Grow the hole mask by N pixels")
    parser.add_argument("--rel-tol", type=float, default=0.05, help="Soft depth falloff scale")
    parser.add_argument("--cov-thresh", type=float, default=0.15, help="Fill below this coverage weight")
    parser.add_argument("--no-clean", action="store_true", help="Skip floater-gaussian removal")
    parser.add_argument("--device", default="cpu", help="torch device for the renderer")
    parser.add_argument("--prompt", default="", help="Prompt for the fill step")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0, help="Fill seed (fixed for reproducibility)")
    parser.add_argument("--out", default="reframe/out", help="Output directory")
    parser.add_argument("--no-fill", action="store_true", help="Skip the FLUX.1-Fill step")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    image, mask = reframe(
        args.ply, args.width, args.height, args.dx, args.dy, args.dz,
        args.max_radius, args.dilate, args.rel_tol, args.cov_thresh,
        not args.no_clean, args.device,
    )
    image.save(out / "reframed.png")
    mask.save(out / "reframed_mask.png")
    print(f"Rendered reframed view and mask to {out}/")

    if args.no_fill:
        return

    from image_editor.fill import FillModel

    print("Filling holes with FLUX.1-Fill (loads the model on first run)...")
    fill = FillModel(quantize=8)
    result = fill.generate(
        image_path=out / "reframed.png",
        mask_path=out / "reframed_mask.png",
        width=args.width,
        height=args.height,
        prompt=args.prompt,
        steps=args.steps,
        seed=args.seed,
        on_step=lambda s, t, _p: print(f"  step {s}/{t}", end="\r"),
    )
    result.save(out / "reframed_filled.png")
    print(f"\nSaved final reframed image to {out}/reframed_filled.png")


if __name__ == "__main__":
    main()
