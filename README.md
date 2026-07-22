# Image Editor

*This document uses ASD-STE100 Simplified Technical English.*

Image Editor is a tool for local image edits on an Apple Silicon Mac. It uses the FLUX.1-Fill-dev model through mflux. All operations run on the Metal GPU of your Mac. The tool does not send your images to a network.

The tool has two interfaces: a web page and a command line.

## Terms

- Outpaint: an operation that makes the image larger and fills the new area.
- Inpaint: an operation that fills a selected area of the image.
- Step: one cycle of the model that removes some noise.
- Mask: a black and white image. White shows the area to fill. Black shows the area to keep.
- Prompt: text that tells the model what to put in the new area.

## Features

- The tool extends an image to a new aspect ratio. It does not cut the original image.
- The tool fills a selected area of the image. A prompt can control the new content.
- The live preview shows the image after each step.
- The original image stays clear in the preview. Only the new area changes.
- The tool loads the model one time and keeps it in memory.
- All operations run on the Metal GPU with MLX.

## Requirements

- An Apple Silicon Mac. The model is large (about 12 billion parameters). Use a Mac with 32 GB of memory for the default 8-bit quantization.
- uv.

The model weights download on the first run. Hugging Face keeps the weights for later runs. The first run is slow because of this download and the model load.

## Installation

To install the tool, enter this command:

```
uv sync
```

## Web page

To start the web page, enter this command:

```
uv run image-editor-web
```

The tool starts a local server. The default address is `http://127.0.0.1:5005`. The tool opens the page in your web browser.

To make an edit, do these steps:

1. Put an image on the left panel. To do this, move the image onto the panel, or click the panel to select a file.
2. Select the edit type. Click **Extend** for an outpaint. Click **Fill region** for an inpaint.
3. For an inpaint, make a box on the image. To make the box, hold the mouse button and move the pointer. The box shows the area to fill.
4. Type a prompt if you want one. Set the number of steps. Set a seed if you want one.
5. Click **Generate**.

While the model runs, a progress bar shows each phase:

- the model load (on the first edit only)
- the preparation
- each step, with a percentage

If **Live preview** is on, the panel shows the image after each step. As a result, you see the result as it forms.

When the edit is complete, do one of these steps:

- Click **Download** to save the result.
- Click **Use as input** to put the result back as the input for a new edit.

The tool loads the model one time and keeps it in memory. As a result, only the first edit waits for the load. Each edit after the first starts immediately.

To change the host, the port, or the browser behavior, use these options:

```
uv run image-editor-web --host 0.0.0.0 --port 8000 --no-browser
```

## Command line

The command line tools load the model on each command. As a result, the web page is faster for more than one edit. Use the command line for a single edit or for a script.

### Outpaint command

The outpaint command makes the image larger to the given aspect ratio. It does not cut the image. The tool increases each new dimension to a multiple of 16.

```
uv run outpaint image.jpeg 1:1
uv run outpaint image.jpeg 16:9 --prompt "a wide mountain landscape" --steps 30
```

Command format:

```
outpaint <image> <ratio> [--prompt TEXT] [--steps N] [--guidance G] [--seed S] [--output PATH]
```

The default output file is `<input>_outpainted.png`.

### Inpaint command

The inpaint command fills the rectangular area `x1,y1,x2,y2`. It keeps the other parts of the image.

```
uv run inpaint image.jpeg --region 100,100,400,400 --prompt "a bunch of flowers"
```

Command format:

```
inpaint <image> --region x1,y1,x2,y2 [--prompt TEXT] [--steps N] [--guidance G] [--seed S] [--output PATH]
```

The default output file is `<input>_inpainted.png`.

## How the tool works

An outpaint and an inpaint use the same operation. The tool makes a canvas and a mask. Then the model fills the area that the mask shows.

- For an outpaint, the tool puts the original image at the center of a larger canvas. The mask shows the new border.
- For an inpaint, the tool uses the original image as the canvas. The mask shows the area that you selected.

The model does not keep the original pixels fixed. The model removes noise from the full canvas. At the same time, the model uses the original image as an example. Then the model makes the kept area again to match the original. For this reason, the full image is fuzzy at the start of the live preview.

The tool puts your original image on the kept area of each preview frame. As a result, only the new area shows the model output during the edit.

The live preview works in this way:

1. The model completes a step.
2. The tool decodes the step data into a small image.
3. The tool sends the image to the browser with Server-Sent Events.

The decode adds time to each step. For this reason, Live preview is an option that you can turn off.

## Notes

- A prompt is optional. If you do not give a prompt, the model uses only the image around the area.
- More steps give better quality but a longer time. The default is 25 steps. More than 35 steps gives a small change only.
- A larger canvas needs more time for each step.
- The saved file has the model's version of the kept area. This can be a little different from your original pixels. The preview overlay does not change the saved file.
