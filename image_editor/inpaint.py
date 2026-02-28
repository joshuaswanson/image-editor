import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def parse_region(region_str: str) -> tuple[int, int, int, int]:
    """Parse 'x1,y1,x2,y2' into a coordinate tuple."""
    parts = region_str.split(",")
    if len(parts) != 4:
        raise ValueError(f"Invalid region: {region_str!r} (expected x1,y1,x2,y2)")
    x1, y1, x2, y2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid region: x1,y1 must be less than x2,y2 (got {x1},{y1},{x2},{y2})")
    return x1, y1, x2, y2


def create_mask(width: int, height: int, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    """Create a black/white mask. White = region to regenerate, black = keep."""
    mask = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=(255, 255, 255))
    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Inpaint a region of an image using FLUX.1-Fill-dev via mflux."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--region", required=True, help="Region to inpaint: x1,y1,x2,y2")
    parser.add_argument("--prompt", default="", help="Prompt describing the inpainted content")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--output", "-o", default=None, help="Output path (default: <input>_inpainted.png)"
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

    width, height = image.size
    print(f"Input: {args.image} ({width}x{height})")

    # Parse and validate region
    try:
        x1, y1, x2, y2 = parse_region(args.region)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if x2 > width or y2 > height:
        print(
            f"Error: region ({x1},{y1},{x2},{y2}) exceeds image bounds ({width}x{height})",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Region: ({x1},{y1}) -> ({x2},{y2}) [{x2-x1}x{y2-y1} pixels]")

    # Create mask and save to temp file
    mask = create_mask(width, height, x1, y1, x2, y2)
    mask_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    mask_path = mask_file.name
    mask.save(mask_path)
    mask_file.close()
    print(f"Mask saved to {mask_path}")

    # Build mflux command
    output_path = args.output
    if output_path is None:
        stem = args.image.rsplit(".", 1)[0]
        output_path = f"{stem}_inpainted.png"

    cmd = [
        "mflux-generate-fill",
        "--image-path", args.image,
        "--masked-image-path", mask_path,
        "--prompt", args.prompt,
        "-q", "8",
        "--steps", str(args.steps),
        "--guidance", str(args.guidance),
        "--height", str(height),
        "--width", str(width),
        "--output", output_path,
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
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
        Path(mask_path).unlink(missing_ok=True)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
