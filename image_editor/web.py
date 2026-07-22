"""A small local web UI for FLUX.1-Fill-dev outpainting and inpainting.

The FLUX model is loaded once and kept resident, so only the first edit pays the
load cost. Denoising progress is streamed to the browser over Server-Sent Events
so the loading bar reflects real step-by-step progress.
"""

import argparse
import json
import queue
import tempfile
import threading
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from PIL import Image

from image_editor.cli import compute_target_dimensions, create_canvas_and_mask, parse_aspect_ratio
from image_editor.fill import FillModel
from image_editor.inpaint import create_mask, parse_region

STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=None)

FILL = FillModel(quantize=8)

# job_id -> {"queue": Queue, "dir": Path, "output": Path | None}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _round_up_16(value: int) -> int:
    return ((value + 15) // 16) * 16


def _emit(job: dict, event: "dict | None"):
    job["queue"].put(event)


def _run_job(job_id: str, mode: str, prompt: str, steps: int, seed, params: dict):
    job = JOBS[job_id]
    job_dir = job["dir"]
    try:
        image = Image.open(job_dir / "input.png").convert("RGB")

        if mode == "outpaint":
            ratio_w, ratio_h = parse_aspect_ratio(params["ratio"])
            target_w, target_h = compute_target_dimensions(image.width, image.height, ratio_w, ratio_h)
            canvas, mask = create_canvas_and_mask(image, target_w, target_h)
            crop_box = None
        else:  # inpaint
            x1, y1, x2, y2 = parse_region(params["region"])
            # FLUX fill needs dimensions that are multiples of 16; pad if needed
            # and crop the result back to the original size afterwards.
            target_w, target_h = _round_up_16(image.width), _round_up_16(image.height)
            canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            canvas.paste(image, (0, 0))
            mask = create_mask(target_w, target_h, x1, y1, x2, y2)
            crop_box = (0, 0, image.width, image.height)

        canvas_path = job_dir / "canvas.png"
        mask_path = job_dir / "mask.png"
        canvas.save(canvas_path)
        mask.save(mask_path)

        if not FILL.is_loaded:
            _emit(job, {"phase": "loading"})
        FILL.ensure_loaded()
        _emit(job, {"phase": "preparing", "total": steps})

        def on_step(current, total):
            _emit(job, {"phase": "generating", "step": current, "total": total})

        result = FILL.generate(
            image_path=canvas_path,
            mask_path=mask_path,
            width=target_w,
            height=target_h,
            prompt=prompt,
            steps=steps,
            seed=seed,
            on_step=on_step,
        )

        if crop_box is not None:
            result = result.crop(crop_box)

        output_path = job_dir / "output.png"
        result.save(output_path)
        job["output"] = output_path
        _emit(job, {"phase": "done", "url": f"/api/result/{job_id}"})
    except Exception as exc:  # surface the failure to the UI rather than hanging
        _emit(job, {"phase": "error", "message": str(exc)})
    finally:
        _emit(job, None)  # sentinel: closes the SSE stream


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    mode = request.form.get("mode", "outpaint")
    prompt = request.form.get("prompt", "")
    steps = int(request.form.get("steps", 25))
    seed_raw = request.form.get("seed", "").strip()
    seed = int(seed_raw) if seed_raw else None

    job_id = uuid.uuid4().hex
    job_dir = Path(tempfile.mkdtemp(prefix=f"imgedit_{job_id}_"))
    Image.open(request.files["image"].stream).convert("RGB").save(job_dir / "input.png")

    params = {"ratio": request.form.get("ratio", "1:1"), "region": request.form.get("region", "")}

    # Validate arguments before spawning the worker so errors return synchronously.
    try:
        if mode == "outpaint":
            parse_aspect_ratio(params["ratio"])
        else:
            parse_region(params["region"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with JOBS_LOCK:
        JOBS[job_id] = {"queue": queue.Queue(), "dir": job_dir, "output": None}

    threading.Thread(
        target=_run_job,
        args=(job_id, mode, prompt, steps, seed, params),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/api/events/<job_id>")
def events(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404

    def stream():
        while True:
            event = job["queue"].get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return Response(stream(), mimetype="text/event-stream")


@app.route("/api/result/<job_id>")
def result(job_id):
    job = JOBS.get(job_id)
    if job is None or job["output"] is None:
        return jsonify({"error": "No result"}), 404
    return send_file(job["output"], mimetype="image/png")


def main():
    parser = argparse.ArgumentParser(description="Local web UI for FLUX image editing.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"Image editor UI running at {url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    # threaded=True so SSE streams and generation can run alongside new requests.
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
