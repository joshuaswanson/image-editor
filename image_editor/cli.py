import argparse
import math
import sys

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image


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
        # Wider than original: extend horizontally, keep height
        target_h = orig_h
        target_w = round(target_h * target_ratio)
    else:
        # Taller than original: extend vertically, keep width
        target_w = orig_w
        target_h = round(target_w / target_ratio)

    # Round up to nearest multiple of 16
    target_w = math.ceil(target_w / 16) * 16
    target_h = math.ceil(target_h / 16) * 16

    return target_w, target_h


def create_green_canvas(
    image: Image.Image, target_w: int, target_h: int
) -> Image.Image:
    """Place the original image centered on a green (0,255,0) canvas."""
    canvas = Image.new("RGB", (target_w, target_h), (0, 255, 0))
    paste_x = (target_w - image.width) // 2
    paste_y = (target_h - image.height) // 2
    canvas.paste(image, (paste_x, paste_y))
    return canvas


def load_pipeline() -> Flux2KleinPipeline:
    """Load the FLUX.2 Klein pipeline with the outpainting LoRA on MPS."""
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    lora_repo = "fal/flux-2-klein-4B-outpaint-lora"
    lora_filename = "flux-outpaint-lora.safetensors"

    print(f"Loading pipeline from {model_id}...")
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    print(f"Loading outpainting LoRA from {lora_repo}...")
    pipe.load_lora_weights(lora_repo, weight_name=lora_filename, adapter_name="outpaint")
    pipe.set_adapters(["outpaint"], adapter_weights=[1.1])

    pipe.to("mps")
    pipe.enable_attention_slicing()
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    return pipe


def run_outpaint(
    pipe: Flux2KleinPipeline,
    canvas: Image.Image,
    prompt: str,
    seed: int | None,
    width: int,
    height: int,
) -> Image.Image:
    """Run the outpainting pipeline."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=canvas,
        height=height,
        width=width,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=generator,
    )
    return result.images[0]


def main():
    parser = argparse.ArgumentParser(
        description="Outpaint an image to a target aspect ratio using FLUX.2 Klein."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("ratio", help="Target aspect ratio (e.g. 16:9)")
    parser.add_argument(
        "--prompt",
        default="Fill the green spaces according to the image",
        help="Prompt describing the outpainted content",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--output", "-o", default=None, help="Output path (default: <input>_outpainted.png)"
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1024,
        help="Max generation dimension (default: 1024). Larger images are scaled down for generation then scaled back up.",
    )

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

    # Scale down if needed for generation
    max_dim = args.max_dim
    scale_factor = 1.0
    if max(target_w, target_h) > max_dim:
        scale_factor = max_dim / max(target_w, target_h)
        gen_w = math.ceil((target_w * scale_factor) / 16) * 16
        gen_h = math.ceil((target_h * scale_factor) / 16) * 16
        scaled_image = image.resize(
            (round(orig_w * scale_factor), round(orig_h * scale_factor)),
            Image.Resampling.LANCZOS,
        )
        print(f"Scaled down for generation: {gen_w}x{gen_h} (factor {scale_factor:.3f})")
    else:
        gen_w, gen_h = target_w, target_h
        scaled_image = image

    # Create green canvas
    canvas = create_green_canvas(scaled_image, gen_w, gen_h)

    # Load pipeline and run
    pipe = load_pipeline()
    print(f"Running outpaint ({gen_w}x{gen_h}, 4 steps)...")
    result = run_outpaint(pipe, canvas, args.prompt, args.seed, gen_w, gen_h)

    # Scale back up if we downscaled
    if scale_factor < 1.0:
        result = result.resize((target_w, target_h), Image.Resampling.LANCZOS)
        print(f"Scaled result back up to {target_w}x{target_h}")

    # Save output
    if args.output:
        output_path = args.output
    else:
        stem = args.image.rsplit(".", 1)[0]
        output_path = f"{stem}_outpainted.png"

    result.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
