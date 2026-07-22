# Image Editor

Local AI image editing on Apple Silicon. Extend an image to a new aspect ratio (outpaint) or regenerate a rectangular region of it (inpaint), powered by [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) running through [mflux](https://github.com/filipstrand/mflux). Everything runs on-device on your Mac's GPU. No API keys, no accounts, and no images leave your machine.

You can use it two ways: a browser UI with a live step-by-step preview, or the command line.

## Features

- Outpaint an image to any aspect ratio without cropping the original
- Inpaint a selected region, guided by an optional text prompt
- Live preview that shows the image resolving from noise at each denoising step
- Original stays crisp in the preview; only the area being generated animates
- Model loaded once and kept resident, so only the first edit waits for load
- Runs entirely locally on the Metal GPU via MLX

## Requirements

- Apple Silicon Mac. FLUX.1-Fill is a ~12B model, so 32 GB of unified memory is comfortable at the default 8-bit quantization.
- [uv](https://github.com/astral-sh/uv)

The FLUX weights download automatically on first use and are cached by Hugging Face afterward. The first run is slow because of that download plus model load.

## Install

```bash
uv sync
```

## Web UI

```bash
uv run image-editor-web
```

This starts a local server (default `http://127.0.0.1:5005`) and opens it in your browser.

1. Drop an image onto the left panel, or click to browse.
2. Choose **Extend** to outpaint to an aspect ratio, or **Fill region** to inpaint. In Fill region mode, drag a box on the image to mark the area to regenerate.
3. Optionally add a prompt, set the number of steps, or fix a seed.
4. Press **Generate**.

While it runs, a progress bar tracks every phase: model loading (first edit only), preparation, and each denoising step with a live percentage. With **Live preview** on, the panel updates every step so you watch the result form. When it finishes, use **Download** to save the result, or **Use as input** to feed it back in for a chained edit.

The model is loaded once and stays in memory, so only the first edit pays the load cost. Every edit after that starts generating immediately.

Options:

```bash
uv run image-editor-web --host 0.0.0.0 --port 8000 --no-browser
```

## Command line

The CLI reloads the model on every invocation, so for repeated edits the web UI is faster. The CLI is handy for one-off or scripted runs.

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

## How it works

Both outpaint and inpaint are the same operation underneath: build a canvas and a mask, then let FLUX.1-Fill regenerate the masked area.

- **Outpaint** centers the original on a larger canvas and masks the new border.
- **Inpaint** keeps the original as the canvas and masks the region you selected.

FLUX.1-Fill does not freeze the kept pixels. It denoises the whole canvas from noise while using the original as a conditioning reference, then reconstructs the kept area to match. That is why, in the live preview, the whole image starts fuzzy. To make the preview match intuition, the app composites your original back over the kept region on each preview frame, so only the generated area shows the in-progress model output.

The live preview works by decoding each step's latents through the VAE into a small image and streaming it to the browser over Server-Sent Events. That extra decode per step is why Live preview is slightly slower, and why it is a toggle.

## Notes and tips

- A prompt is optional. When omitted, the fill is guided only by the surrounding image.
- More steps means higher quality and longer runtime. The default is 25. Returns diminish past about 35.
- Generation time scales with resolution, so large canvases take longer per step.
- The final downloaded image contains FLUX's reconstruction of the kept region, which can differ slightly from your original pixels. The preview overlay does not change the saved file.
