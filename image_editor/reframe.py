"""Reframe a photo by moving the camera, then clean the holes with FLUX.1-Fill.

Pipeline:
  1. SHARP turns the photo into a 3D Gaussian splat (run separately via
     `sharp predict`, or pass an existing .ply).
  2. This module renders the splat from a moved camera. Moving the camera
     reveals areas the original photo never saw, which come out as holes.
  3. Those holes are handed to FLUX.1-Fill (the same fill this repo already
     uses) to paint plausible content, so the reframed view looks complete.

Rendering is a simple painter's-algorithm point splatter in numpy. SHARP's
gsplat renderer is CUDA-only, so this gives us a Mac-friendly renderer that
also yields the hole mask the fill step needs.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from plyfile import PlyData

SH_C0 = 0.28209479177387814  # 0th-order spherical-harmonic coefficient


def focal_px(width: int, height: int, f_mm: float = 30.0) -> float:
    """SHARP's focal-length convention (30mm-equivalent on a full-frame diagonal)."""
    return f_mm * np.sqrt(width**2 + height**2) / np.sqrt(36**2 + 24**2)


def load_gaussians(ply_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Gaussian centers and display colors from a SHARP .ply."""
    v = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)
    return xyz, rgb


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
    xyz: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    R_w2c: np.ndarray,
    center: np.ndarray,
    width: int,
    height: int,
    radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Splat points into an image from a given camera. Returns (image, coverage)."""
    cam = (xyz - center) @ R_w2c.T  # world -> camera
    z = cam[:, 2]
    front = z > 1e-3
    cam, col, z = cam[front], rgb[front], z[front]

    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]

    order = np.argsort(-z)  # far to near, so nearer points overwrite
    ui = np.round(u[order]).astype(np.int64)
    vi = np.round(v[order]).astype(np.int64)
    col = col[order]

    img = np.zeros((height, width, 3), np.float32)
    cov = np.zeros((height, width), bool)
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            uu, vv = ui + du, vi + dv
            m = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            img[vv[m], uu[m]] = col[m]
            cov[vv[m], uu[m]] = True
    return img, cov


def reframe(
    ply_path: str | Path,
    width: int,
    height: int,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    radius: int = 2,
    dilate: int = 3,
    window: int = 7,
    min_density: float = 0.5,
    smooth: int = 3,
) -> tuple[Image.Image, Image.Image]:
    """Render a reframed view and its hole mask.

    dx/dy/dz move the camera (in SHARP's metric world units) while it keeps
    looking at the scene center, giving an orbit-like reframe. Returns the
    reframed image and a fill mask (white where holes need to be painted).

    At depth edges the splat tears into a sparse fringe of surviving points.
    Rather than a ragged per-pixel mask, we mark any pixel whose local coverage
    (fraction of a `window`x`window` neighborhood) is below `min_density` as a
    hole, giving solid fill regions the fill step can reconstruct cleanly.
    `dilate` then grows those regions slightly to hide the seam.
    """
    from scipy.ndimage import binary_dilation, uniform_filter

    xyz, rgb = load_gaussians(ply_path)
    K = np.array([[focal_px(width, height), 0, (width - 1) / 2],
                  [0, focal_px(width, height), (height - 1) / 2],
                  [0, 0, 1]])

    # Scene center: a robust point in front of the original camera to orbit around.
    target = np.array([0.0, 0.0, float(np.median(xyz[:, 2]))])
    center = np.array([dx, dy, dz])
    R_w2c = look_at(center, target)

    img, cov = render(xyz, rgb, K, R_w2c, center, width, height, radius)
    density = uniform_filter(cov.astype(np.float32), size=window)
    holes = density < min_density  # solid regions wherever coverage is sparse
    if dilate > 0:
        holes = binary_dilation(holes, iterations=dilate)
    image = Image.fromarray((img * 255).astype(np.uint8), "RGB")
    if smooth > 1:
        # The overwrite splat leaves salt-and-pepper noise; a median pass removes
        # it while keeping edges, so the fill gets a clean image to extend from.
        image = image.filter(ImageFilter.MedianFilter(smooth))
    # White where holes need painting, black where we keep the rendered pixels.
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
    parser.add_argument("--radius", type=int, default=2, help="Splat radius in pixels")
    parser.add_argument("--dilate", type=int, default=3, help="Grow the hole mask by N pixels")
    parser.add_argument("--window", type=int, default=7, help="Neighborhood for coverage density")
    parser.add_argument("--min-density", type=float, default=0.5, help="Fill below this coverage")
    parser.add_argument("--smooth", type=int, default=3, help="Median filter size to denoise render")
    parser.add_argument("--prompt", default="", help="Prompt for the fill step")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--out", default="reframe/out", help="Output directory")
    parser.add_argument("--no-fill", action="store_true", help="Skip the FLUX.1-Fill step")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    image, mask = reframe(
        args.ply, args.width, args.height, args.dx, args.dy, args.dz,
        args.radius, args.dilate, args.window, args.min_density, args.smooth,
    )
    image.save(out / "reframed.png")
    mask.save(out / "reframed_mask.png")
    print(f"Rendered reframed view and mask to {out}/")

    if args.no_fill:
        return

    # Hand the holes to FLUX.1-Fill for a clean, complete image.
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
        on_step=lambda s, t, _p: print(f"  step {s}/{t}", end="\r"),
    )
    result.save(out / "reframed_filled.png")
    print(f"\nSaved final reframed image to {out}/reframed_filled.png")


if __name__ == "__main__":
    main()
