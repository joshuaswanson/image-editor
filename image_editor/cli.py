import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def parse_aspect_ratio(ratio_str: str) -> tuple[int, int]:
    """Parse '16:9' into (16, 9)."""
    parts = ratio_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid aspect ratio: {ratio_str!r} (expected W:H)")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Aspect ratio components must be positive: {ratio_str!r}")
    return w, h


def compute_target_dimensions(
    orig_w: int, orig_h: int, ratio_w: int, ratio_h: int
) -> tuple[int, int]:
    """Compute target dimensions that contain the original image at the given aspect ratio.

    Always extends (never crops). Result is rounded to a multiple of 16.
    """
    orig_ratio = orig_w / orig_h
    target_ratio = ratio_w / ratio_h

    if target_ratio > orig_ratio:
        target_h = orig_h
        target_w = round(target_h * target_ratio)
    else:
        target_w = orig_w
        target_h = round(target_w / target_ratio)

    target_w = math.ceil(target_w / 16) * 16
    target_h = math.ceil(target_h / 16) * 16

    return target_w, target_h


def create_canvas_and_mask(
    image: Image.Image, target_w: int, target_h: int
) -> tuple[Image.Image, Image.Image]:
    """Place the original image centered on a canvas and create a mask for the extended regions."""
    paste_x = (target_w - image.width) // 2
    paste_y = (target_h - image.height) // 2

    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(image, (paste_x, paste_y))

    # White = regenerate, black = keep
    mask = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    draw = ImageDraw.Draw(mask)
    draw.rectangle(
        [paste_x, paste_y, paste_x + image.width - 1, paste_y + image.height - 1],
        fill=(0, 0, 0),
    )

    return canvas, mask


def main():
    parser = argparse.ArgumentParser(
        description="Outpaint an image to a target aspect ratio using FLUX.1-Fill-dev via mflux."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("ratio", help="Target aspect ratio (e.g. 16:9)")
    parser.add_argument(
        "--prompt",
        default="",
        help="Prompt describing the outpainted content",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--output", "-o", default=None, help="Output path (default: <input>_outpainted.png)"
    )
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps (default: 25)")
    parser.add_argument("--guidance", type=float, default=30.0, help="Guidance scale (default: 30)")

    args = parser.parse_args()

    # Load input image
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)

    orig_w, orig_h = image.size
    print(f"Input: {args.image} ({orig_w}x{orig_h})")

    # Compute target dimensions
    ratio_w, ratio_h = parse_aspect_ratio(args.ratio)
    target_w, target_h = compute_target_dimensions(orig_w, orig_h, ratio_w, ratio_h)
    print(f"Target: {target_w}x{target_h} (ratio {ratio_w}:{ratio_h})")

    if target_w == orig_w and target_h == orig_h:
        print("Image already matches the target aspect ratio. Nothing to do.")
        sys.exit(0)

    # Create canvas with image centered and mask for extended regions
    canvas, mask = create_canvas_and_mask(image, target_w, target_h)

    canvas_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    canvas_path = canvas_file.name
    canvas.save(canvas_path)
    canvas_file.close()

    mask_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    mask_path = mask_file.name
    mask.save(mask_path)
    mask_file.close()

    # Build mflux command
    output_path = args.output
    if output_path is None:
        stem = args.image.rsplit(".", 1)[0]
        output_path = f"{stem}_outpainted.png"

    cmd = [
        "mflux-generate-fill",
        "--image-path", canvas_path,
        "--masked-image-path", mask_path,
        "--prompt", args.prompt,
        "-q", "8",
        "--steps", str(args.steps),
        "--guidance", str(args.guidance),
        "--height", str(target_h),
        "--width", str(target_w),
        "--output", output_path,
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "Error: mflux-generate-fill not found. Make sure mflux is installed.",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: mflux-generate-fill exited with code {e.returncode}", file=sys.stderr)
        sys.exit(1)
    finally:
        Path(canvas_path).unlink(missing_ok=True)
        Path(mask_path).unlink(missing_ok=True)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
