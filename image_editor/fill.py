"""Shared FLUX.1-Fill-dev helpers used by the CLI and the web server.

The web server loads the model once and keeps it resident, so repeated edits
skip the slow load/quantize step that the CLI pays on every invocation.
"""

import base64
import threading
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image

# Longest side of the streamed step preview, in pixels. Small keeps the
# per-step base64 payload light while still showing the image take shape.
PREVIEW_MAX_SIDE = 640


def normalize_prompt(prompt: str | None) -> str:
    """Return a prompt safe to feed to mflux.

    mflux 0.16.6 tokenizes an all-empty prompt to a zero-length token array,
    which then crashes CLIP's pooled-output argmax. A single space tokenizes to
    the bare BOS/EOS pair, i.e. the genuine unconditional embedding, which is
    exactly what a prompt-less fill should use.
    """
    if prompt is None or not prompt.strip():
        return " "
    return prompt


# Step callback: (current_step, total_steps, preview_data_url | None) -> None
StepCallback = Callable[[int, int, "str | None"], None]


def _latents_to_data_url(model, latents, config) -> str:
    """Decode a step's latents to a small JPEG data URL for live preview.

    This mirrors what mflux does at the end of generation (unpack the packed
    latents, then VAE-decode), but on the in-flight latents so the caller can
    watch the image resolve. The extra decode is the cost of the preview.
    """
    # Imported lazily to keep importing this module (and the CLI) cheap.
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
    from mflux.utils.image_util import ImageUtil

    unpacked = FluxLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
    decoded = model.vae.decode(unpacked)
    image = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(ImageUtil._denormalize(decoded)))
    image.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class _ProgressSubscriber:
    """mflux in-loop callback that forwards denoising progress to on_step.

    When preview_every > 0 it also decodes the current latents to an image and
    passes it along, so the caller can render intermediate results.
    """

    def __init__(self, on_step: StepCallback, model=None, preview_every: int = 0):
        self.on_step = on_step
        self.model = model
        self.preview_every = preview_every

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        total = config.num_inference_steps
        current = min(t - config.init_time_step + 1, total)
        preview = None
        if self.preview_every and (current % self.preview_every == 0 or current == total):
            preview = _latents_to_data_url(self.model, latents, config)
        self.on_step(current, total, preview)


class FillModel:
    """Lazily-loaded, process-wide FLUX.1-Fill-dev model.

    Loading is guarded so only the first request pays the cost, and generation
    is serialized because a single mflux model instance is not safe to run
    concurrently (callbacks and caches are shared mutable state).
    """

    def __init__(self, quantize: int = 8):
        self.quantize = quantize
        self._model = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            # Imported lazily so importing this module (and the CLI) stays cheap.
            from mflux.models.flux.variants.fill.flux_fill import Flux1Fill

            self._model = Flux1Fill(quantize=self.quantize)

    def generate(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        width: int,
        height: int,
        prompt: str = "",
        steps: int = 25,
        guidance: float = 30.0,
        seed: int | None = None,
        on_step: StepCallback | None = None,
        preview_every: int = 0,
    ) -> Image.Image:
        """Run a single fill and return the generated PIL image.

        preview_every controls live previews: 0 disables them, N decodes and
        forwards an intermediate image every Nth step (and on the final step).
        """
        from mflux.callbacks.callback_registry import CallbackRegistry

        self.ensure_loaded()

        with self._run_lock:
            registry = CallbackRegistry()
            if on_step is not None:
                registry.register(_ProgressSubscriber(on_step, model=self._model, preview_every=preview_every))
            self._model.callbacks = registry

            generated = self._model.generate_image(
                seed=seed if seed is not None else 0,
                prompt=normalize_prompt(prompt),
                image_path=str(image_path),
                masked_image_path=str(mask_path),
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            )
            return generated.image
