"""Detection backends: NudeNet and EraX (ultralytics YOLO)."""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelInfo(NamedTuple):
    backend: str
    path: str | None


DEFAULT_MODEL = "320n"

KNOWN_MODELS: dict[str, ModelInfo] = {
    "320n": ModelInfo(backend="nudenet", path=None),
    "640m": ModelInfo(backend="nudenet", path="~/.config/bsafe/models/640m.onnx"),
    "erax-nano": ModelInfo(
        backend="erax", path="~/.config/bsafe/models/erax-anti-nsfw-yolo11n-v1.1.pt"
    ),
    "erax-small": ModelInfo(
        backend="erax", path="~/.config/bsafe/models/erax-anti-nsfw-yolo11s-v1.1.pt"
    ),
    "erax-medium": ModelInfo(
        backend="erax", path="~/.config/bsafe/models/erax-anti-nsfw-yolo11m-v1.1.pt"
    ),
}

_DOWNLOAD_URLS: dict[str, str] = {
    "640m": "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx",
    "erax-nano": "https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1/resolve/main/erax-anti-nsfw-yolo11n-v1.1.pt",
    "erax-small": "https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1/resolve/main/erax-anti-nsfw-yolo11s-v1.1.pt",
    "erax-medium": "https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1/resolve/main/erax-anti-nsfw-yolo11m-v1.1.pt",
}

# EraX class names → canonical NudeNet names used by censor.py
_ERAX_CLASS_MAP: dict[str, list[str]] = {
    "anus": ["ANUS_EXPOSED"],
    "penis": ["MALE_GENITALIA_EXPOSED"],
    "vagina": ["FEMALE_GENITALIA_EXPOSED"],
    "nipple": ["FEMALE_BREAST_EXPOSED"],
}

# NMS IoU threshold for EraX: controls when overlapping detections of the
# same class are merged by ultralytics (unrelated to tracking IoU).
_ERAX_NMS_IOU: float = 0.3

# Flags that require NudeNet-only classes not available in EraX
ERAX_UNSUPPORTED_FLAGS: dict[str, str] = {
    "covered": "--covered",
    "face_male": "--face-male",
    "face_female": "--face-female",
    "feet": "--feet",
    "buttocks": "--buttocks",
}


def resolve_model(name: str | None) -> ModelInfo:
    """Resolve a model name to a ModelInfo.

    Raises ValueError for unknown names or missing model files.
    """
    if name is None:
        name = DEFAULT_MODEL

    if name not in KNOWN_MODELS:
        raise ValueError(f"unknown model '{name}'. Known models: {', '.join(sorted(KNOWN_MODELS))}")

    info = KNOWN_MODELS[name]
    if info.path is None:
        return info

    path = os.path.expanduser(info.path)
    if not os.path.isfile(path):
        url = _DOWNLOAD_URLS.get(name, "<unknown>")
        raise ValueError(
            f"model file not found: {path}\n"
            f"Download it with:\n"
            f"  mkdir -p ~/.config/bsafe/models && curl -Lo {path} {url}"
        )

    return ModelInfo(backend=info.backend, path=path)


def get_model_backend(name: str | None) -> str:
    """Return the backend type for a model name without checking if the file exists."""
    if name is None:
        name = DEFAULT_MODEL
    if name not in KNOWN_MODELS:
        raise ValueError(f"unknown model '{name}'. Known models: {', '.join(sorted(KNOWN_MODELS))}")
    return KNOWN_MODELS[name].backend


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h


class _NudeNetBackend:
    def __init__(self, min_confidence: float, model_path: str | None):
        from nudenet import NudeDetector

        self.min_confidence = min_confidence
        if model_path:
            self._detector = NudeDetector(model_path=model_path)
        else:
            self._detector = NudeDetector()
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix="bsafe-det-", delete=False)
        self._tmp_path = tmp.name
        tmp.close()

    def close(self):
        try:
            os.unlink(self._tmp_path)
        except OSError:
            pass

    def _parse_results(self, results: list[dict]) -> list[Detection]:
        detections = []
        for r in results:
            confidence = r["score"]
            if confidence < self.min_confidence:
                continue
            box = r["box"]
            detections.append(
                Detection(
                    class_name=r["class"],
                    confidence=confidence,
                    box=(box[0], box[1], box[2], box[3]),
                )
            )
        return detections

    def detect(self, jpeg_bytes: bytes) -> list[Detection]:
        with open(self._tmp_path, "wb") as f:
            f.write(jpeg_bytes)
        results = self._detector.detect(self._tmp_path)
        return self._parse_results(results)

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        results = self._detector.detect(frame)
        return self._parse_results(results)


class _EraXBackend:
    def __init__(self, min_confidence: float, model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for EraX models. Install with: uv sync --extra erax"
            ) from None

        self.min_confidence = min_confidence
        self._model = YOLO(model_path)

    def close(self):
        pass

    def _parse_results(self, results) -> list[Detection]:
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                confidence = float(boxes.conf[i])
                if confidence < self.min_confidence:
                    continue
                cls_id = int(boxes.cls[i])
                cls_name = result.names[cls_id]
                mapped = _ERAX_CLASS_MAP.get(cls_name)
                if mapped is None:
                    continue
                # xyxy → (x, y, w, h)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                for canonical_name in mapped:
                    detections.append(
                        Detection(class_name=canonical_name, confidence=confidence, box=box)
                    )
        return detections

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._model.predict(
            rgb, conf=self.min_confidence, iou=_ERAX_NMS_IOU, verbose=False
        )
        return self._parse_results(results)

    def detect(self, jpeg_bytes: bytes) -> list[Detection]:
        import cv2

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return self.detect_frame(frame)


class Detector:
    """Unified detector: delegates to NudeNet or EraX backend."""

    def __init__(self, min_confidence: float | None = None, model: str | None = None):
        info = resolve_model(model)
        if min_confidence is None:
            min_confidence = 0.2 if info.backend == "erax" else 0.0
        if info.backend == "erax":
            self._backend = _EraXBackend(min_confidence, info.path)
        else:
            self._backend = _NudeNetBackend(min_confidence, info.path)
        logger.info(
            "Detector initialized (model=%s, backend=%s)", model or DEFAULT_MODEL, info.backend
        )

    def close(self):
        self._backend.close()

    def detect(self, jpeg_bytes: bytes) -> list[Detection]:
        """Run inference on JPEG bytes. Returns filtered detections."""
        return self._backend.detect(jpeg_bytes)

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a BGR numpy frame directly."""
        return self._backend.detect_frame(frame)
