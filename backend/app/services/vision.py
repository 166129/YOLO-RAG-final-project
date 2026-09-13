"""Road-sign detection, and the mapping from a detected class to a handbook lookup.

The weights are produced by notebooks/yolo_training_colab.ipynb. If they are absent the
service degrades gracefully: text queries keep working and /query/image reports that the
vision model is unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

# Each sign class maps to the phrasing the handbooks actually use, so retrieval lands on
# the rule rather than on a passing mention of the sign.
#
# Note: `speedlimit` says a speed-limit sign is present, not which number it shows -
# reading the digits would need a second OCR stage, which is out of scope.
CLASS_TO_QUERY: dict[str, str] = {
    "stop": "stop sign rules: where to stop and when to proceed",
    "speedlimit": "posted speed limit rules and maximum speed limits",
    "crosswalk": "pedestrian crosswalk rules and yielding right of way to pedestrians",
    "trafficlight": "traffic signal lights: red, yellow and green meanings",
}


class SignDetector:
    """Loaded once in the FastAPI lifespan. CPU inference: ~60 ms for one image."""

    def __init__(self, weights_path: Path | None = None) -> None:
        self.weights_path = Path(weights_path or settings.yolo_weights_path)
        self.model = None
        if not self.weights_path.exists():
            logger.warning(
                "YOLO weights missing at %s - image queries will report vision unavailable. "
                "Train them with notebooks/yolo_training_colab.ipynb.",
                self.weights_path,
            )
            return
        from ultralytics import YOLO  # imported lazily: torch load is slow

        self.model = YOLO(str(self.weights_path))
        self.model.to("cpu")
        logger.info("YOLO loaded from %s, classes=%s", self.weights_path, self.model.names)

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect(self, image_path: str | Path) -> list[dict]:
        if self.model is None:
            return []
        result = self.model.predict(
            str(image_path), conf=settings.detection_confidence, verbose=False
        )[0]
        return [
            {
                "label": self.model.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [round(v) for v in box.xyxy[0].tolist()],
            }
            for box in result.boxes
        ]

    @staticmethod
    def to_question(detections: list[dict]) -> tuple[str, dict] | tuple[None, None]:
        """Turn the highest-confidence detection into a handbook question."""
        if not detections:
            return None, None
        top = max(detections, key=lambda d: d["confidence"])
        label = top["label"]
        lookup = CLASS_TO_QUERY.get(label, f"{label} sign rules")
        question = f"A {label.replace('_', ' ')} sign was detected in a photo. {lookup}"
        return question, top
