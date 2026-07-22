# Image Editor

Local AI image editing on Apple Silicon. Extend an image to a new aspect ratio (outpaint) or regenerate a region of it (inpaint) using [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) via [mflux](https://github.com/filipstrand/mflux). Everything runs on-device, so no API keys and no images leave your machine.

There are two ways to use it: a browser UI with a live progress bar, or the command line.

## Requirements

- Apple Silicon Mac (mflux uses MLX)
- [uv](https://github.com/astral-sh/uv)

The FLUX weights download automatically on first use and are cached by Hugging Face afterward.

## Install

```bash
uv sync
```

## Web UI

```bash
uv run image-editor-web
```

This starts a local server (default `http://127.0.0.1:5005`) and opens it in your browser. Drop in an image, choose **Extend** (outpaint) or **Fill region** (inpaint), and hit Generate. A progress bar tracks every phase: model loading, preparation, and each denoising step.

The model is loaded once and kept in memory, so only the first edit pays the load cost. Every edit after that starts generating immediately.

Options:

```bash
uv run image-editor-web --host 0.0.0.0 --port 8000 --no-browser
```

## Command line

### Outpaint (extend to an aspect ratio)

Extends the image, without cropping, until it matches the target ratio. New dimensions are rounded up to a multiple of 16.

```bash
uv run outpaint image.jpeg 1:1
uv run outpaint image.jpeg 16:9 --prompt "a wide mountain landscape" --steps 30
```

Arguments: `outpaint <image> <ratio> [--prompt TEXT] [--steps N] [--guidance G] [--seed S] [--output PATH]`

Default output is `<input>_outpainted.png`.

### Inpaint (regenerate a region)

Regenerates the rectangular region given by `x1,y1,x2,y2`, leaving the rest untouched.

```bash
uv run inpaint image.jpeg --region 100,100,400,400 --prompt "a bunch of flowers"
```

Arguments: `inpaint <image> --region x1,y1,x2,y2 [--prompt TEXT] [--steps N] [--guidance G] [--seed S] [--output PATH]`

Default output is `<input>_inpainted.png`.

## Notes

- A prompt is optional. When omitted, the fill is guided only by the surrounding image.
- More steps means higher quality and longer runtime. The default is 25. Generation time scales with resolution, so large images take a while per step.
