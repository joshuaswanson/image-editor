"""Shared FLUX.1-Fill-dev helpers used by the CLI and the web server.

The web server loads the model once and keeps it resident, so repeated edits
skip the slow load/quantize step that the CLI pays on every invocation.
"""

import threading
from pathlib import Path
from typing import Callable

from PIL import Image


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


# Step callback: (current_step, total_steps) -> None
StepCallback = Callable[[int, int], None]


class _ProgressSubscriber:
    """mflux in-loop callback that forwards denoising progress to on_step."""

    def __init__(self, on_step: StepCallback):
        self.on_step = on_step

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        total = config.num_inference_steps
        current = min(t - config.init_time_step + 1, total)
        self.on_step(current, total)


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
    ) -> Image.Image:
        """Run a single fill and return the generated PIL image."""
        from mflux.callbacks.callback_registry import CallbackRegistry

        self.ensure_loaded()

        with self._run_lock:
            registry = CallbackRegistry()
            if on_step is not None:
                registry.register(_ProgressSubscriber(on_step))
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
