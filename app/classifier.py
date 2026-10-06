from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort
import requests
from tokenizers import Tokenizer

MODEL_ROOT = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
VISION_URL = f"{MODEL_ROOT}/onnx/vision_model_quantized.onnx"
TEXT_URL = f"{MODEL_ROOT}/onnx/text_model_quantized.onnx"
TOKENIZER_URL = f"{MODEL_ROOT}/tokenizer.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
LABELS = (
    "cat",
    "dog",
    "person",
    "vehicle",
    "bicycle or motorcycle",
    "household objects and clutter",
    "plants and outdoor objects",
    "empty street",
)
PROMPTS = (
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a person",
    "a photo of a car or other vehicle",
    "a photo of a bicycle or motorcycle",
    "a photo of household objects and clutter",
    "a photo of plants and outdoor objects",
    "a photo of an empty street or pavement",
)


def _valid_model(path: Path) -> bool:
    try:
        if path.stat().st_size < 1_000_000:
            return False
        with path.open("rb") as model_file:
            header = model_file.read(256)
    except OSError:
        return False
    return not header.startswith(b"version https://git-lfs.github.com/spec/v1")


def _valid_tokenizer(path: Path) -> bool:
    try:
        if path.stat().st_size < 10_000:
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and "model" in data


def _download(
    url: str,
    destination: Path,
    validator: Callable[[Path], bool],
    description: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if not validator(partial):
            raise RuntimeError(f"下载的{description}无效，可能是模型指针或错误页面")
        os.replace(partial, destination)
    except requests.RequestException as exc:
        raise RuntimeError(f"下载{description}失败：{exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"保存{description}失败：{exc}") from exc
    finally:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass


def _session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    try:
        return ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"无法加载 CLIP 模型 {model_path.name}：{exc}") from exc


class ClipClassifier:
    def __init__(self, model_dir: Path) -> None:
        self.vision_path = model_dir / "clip-vision-quantized.onnx"
        self.text_path = model_dir / "clip-text-quantized.onnx"
        self.tokenizer_path = model_dir / "clip-tokenizer.json"
        self._vision: ort.InferenceSession | None = None
        self._text_embeddings: np.ndarray | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._vision is not None:
            return
        with self._load_lock:
            if self._vision is not None:
                return
            if not _valid_model(self.vision_path):
                _download(VISION_URL, self.vision_path, _valid_model, "CLIP 图像模型")
            if not _valid_model(self.text_path):
                _download(TEXT_URL, self.text_path, _valid_model, "CLIP 文本模型")
            if not _valid_tokenizer(self.tokenizer_path):
                _download(
                    TOKENIZER_URL,
                    self.tokenizer_path,
                    _valid_tokenizer,
                    "CLIP 分词器",
                )
            tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            text_session = _session(self.text_path)
            embeddings = []
            for prompt in PROMPTS:
                input_ids = np.asarray([tokenizer.encode(prompt).ids], dtype=np.int64)
                embedding = text_session.run(None, {"input_ids": input_ids})[0][0]
                norm = float(np.linalg.norm(embedding))
                if embedding.shape != (512,) or not np.isfinite(embedding).all() or norm == 0:
                    raise RuntimeError("CLIP 文本模型返回了无效向量")
                embeddings.append(embedding / norm)
            self._text_embeddings = np.asarray(embeddings, dtype=np.float32)
            self._vision = _session(self.vision_path)

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        height, width = crop.shape[:2]
        scale = 224 / min(height, width)
        resized_width = max(224, round(width * scale))
        resized_height = max(224, round(height * scale))
        resized = cv2.resize(
            crop,
            (resized_width, resized_height),
            interpolation=cv2.INTER_CUBIC,
        )
        left = (resized_width - 224) // 2
        top = (resized_height - 224) // 2
        cropped = resized[top : top + 224, left : left + 224]
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(blob, dtype=np.float32)

    def classify(self, crop: np.ndarray) -> tuple[float, str]:
        if crop is None or crop.size == 0:
            raise ValueError("待分类的对象裁图为空")
        self._ensure_loaded()
        blob = self._preprocess(crop)
        with self._inference_lock:
            output = self._vision.run(None, {"pixel_values": blob})[0]
        if output.shape != (1, 512) or not np.isfinite(output).all():
            raise RuntimeError(f"CLIP 图像模型返回了无效输出：{output.shape}")
        embedding = output[0]
        norm = float(np.linalg.norm(embedding))
        if norm == 0:
            raise RuntimeError("CLIP 图像模型返回了零向量")
        embedding = embedding / norm
        logits = (self._text_embeddings @ embedding * 100).astype(np.float64)
        weights = np.exp(logits - np.max(logits))
        probabilities = weights / np.sum(weights)
        return float(probabilities[0]), LABELS[int(np.argmax(probabilities))]


_DEFAULT_CLASSIFIER = ClipClassifier(DATA_DIR / "models")


def classify_crop(crop: np.ndarray) -> tuple[float, str]:
    return _DEFAULT_CLASSIFIER.classify(crop)
