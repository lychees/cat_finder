from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .classifier import _valid_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELS = ("Bird", "Cat", "Human")


class CatFinderClassifier:
    def __init__(self, model_dir: Path) -> None:
        self.model_path = model_dir / "catFinderV14_yoloWeights.onnx"
        self._session: ort.InferenceSession | None = None
        self._input_name = ""
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return
            if not _valid_model(self.model_path):
                raise RuntimeError(
                    "CatFinder ONNX 模型未准备好，请先将 catFinderV14_yoloWeights.pt 转换为 ONNX"
                )
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            try:
                session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(f"无法加载 CatFinder 模型：{exc}") from exc
            self._input_name = session.get_inputs()[0].name
            self._session = session

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        height, width = crop.shape[:2]
        ratio = min(640 / width, 640 / height)
        resized_width = max(1, round(width * ratio))
        resized_height = max(1, round(height * ratio))
        resized = cv2.resize(
            crop,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        left = (640 - resized_width) // 2
        top = (640 - resized_height) // 2
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top : top + resized_height, left : left + resized_width] = resized
        return cv2.dnn.blobFromImage(
            canvas,
            1 / 255.0,
            (640, 640),
            swapRB=True,
            crop=False,
        )

    def classify(self, crop: np.ndarray) -> tuple[float, str]:
        if crop is None or crop.size == 0:
            raise ValueError("待分类的对象裁图为空")
        self._ensure_loaded()
        blob = self._preprocess(crop)
        with self._inference_lock:
            output = self._session.run(None, {self._input_name: blob})[0]
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 7:
            raise RuntimeError(f"CatFinder 输出形状为 {output.shape}，预期为 (1, 7, N)")
        if not np.isfinite(output).all():
            raise RuntimeError("CatFinder 输出包含 NaN 或无穷值")
        scores = output[0].T[:, 4:]
        cat_confidence = float(np.max(scores[:, 1]))
        best_class = int(np.unravel_index(np.argmax(scores), scores.shape)[1])
        return cat_confidence, LABELS[best_class]


_DEFAULT_CLASSIFIER = CatFinderClassifier(DATA_DIR / "models")


def classify_crop(crop: np.ndarray) -> tuple[float, str]:
    return _DEFAULT_CLASSIFIER.classify(crop)
